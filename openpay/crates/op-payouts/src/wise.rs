//! Wise Platform API driver.
//!
//! Wise's payout flow is three calls:
//!   1. `POST /v3/profiles/{profile}/quotes` — quote the FX.
//!   2. `POST /v1/transfers` — create the transfer against a recipient
//!      id.
//!   3. `POST /v3/profiles/{profile}/transfers/{id}/payments` — fund.
//!
//! The orchestrator only needs the transfer creation body offline; the
//! quote and funding calls are operator-driven.

use serde::Serialize;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutRequest, PayoutResult, PayoutStatus,
};

/// Wise Platform driver.
#[derive(Clone, Debug, Default)]
pub struct WiseDriver {
    /// Operator's Wise profile id (numeric, opaque).
    pub profile_id: u64,
    /// Quote id obtained from a prior `POST /v3/profiles/.../quotes`.
    pub quote_uuid: String,
}

#[derive(Debug, Serialize)]
struct TransferBody<'a> {
    #[serde(rename = "targetAccount")]
    target_account: u64,
    #[serde(rename = "quoteUuid")]
    quote_uuid: &'a str,
    #[serde(rename = "customerTransactionId")]
    customer_transaction_id: &'a str,
    details: TransferDetails<'a>,
}

#[derive(Debug, Serialize)]
struct TransferDetails<'a> {
    reference: &'a str,
}

impl Payout for WiseDriver {
    fn rail(&self) -> &'static str {
        "wise"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        let recipient_id = match &req.beneficiary.account {
            BeneficiaryAccount::WiseRecipientId(id) => *id,
            _ => return Err(Error::UnsupportedMethod { rail: "wise" }),
        };
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "wise",
                detail: "amount must be positive".to_string(),
            });
        }
        let body = TransferBody {
            target_account: recipient_id,
            quote_uuid: &self.quote_uuid,
            customer_transaction_id: &req.idempotency_key,
            details: TransferDetails {
                reference: req.memo.as_deref().unwrap_or(&req.idempotency_key),
            },
        };
        let json = serde_json::to_vec(&body).map_err(|e| Error::DriverValidation(e.to_string()))?;
        Ok(PayoutResult {
            idempotency_key: req.idempotency_key.clone(),
            payout_id: Uuid::now_v7().to_string(),
            status: PayoutStatus::PreparedOffline,
            raw_status: None,
            reason_code: None,
            reason_text: None,
            rail_txn_id: None,
            settled_amount: Some(req.amount),
            wire_payload: Some(json),
        })
    }

    fn status(&self, _payout_id: &str) -> Result<PayoutResult> {
        Err(Error::DriverValidation(
            "Wise status query requires GET /v1/transfers/{id} via live client".to_string(),
        ))
    }
}
