//! Mastercard Send driver.
//!
//! Mastercard Send is the OCT-based push-to-card service. The
//! "Disbursements" API endpoint
//! (`/send/v1/partners/{partnerId}/disbursements/payments`) accepts a
//! JSON body keyed by `transactionReference`. We build that body
//! offline; the operator submits with mTLS + OAuth 1.0a one-leg
//! signature.

use serde::Serialize;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutRequest, PayoutResult, PayoutStatus,
};
use crate::visa_direct::format_amount;

/// Mastercard Send disbursements driver.
#[derive(Clone, Debug, Default)]
pub struct MastercardSendDriver {
    /// Partner id assigned by Mastercard.
    pub partner_id: String,
    /// Hub disbursement program id (`programId`) configured for this
    /// operator.
    pub program_id: String,
}

#[derive(Debug, Serialize)]
struct DisbursementBody<'a> {
    #[serde(rename = "transactionReference")]
    transaction_reference: &'a str,
    #[serde(rename = "programId")]
    program_id: &'a str,
    #[serde(rename = "currency")]
    currency: &'a str,
    #[serde(rename = "amount")]
    amount: String,
    #[serde(rename = "recipientAccountUri")]
    recipient_account_uri: String,
    #[serde(rename = "recipientName")]
    recipient_name: &'a str,
}

impl Payout for MastercardSendDriver {
    fn rail(&self) -> &'static str {
        "mc_send"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        let pan = match &req.beneficiary.account {
            BeneficiaryAccount::CardPan(pan) | BeneficiaryAccount::CardToken(pan) => pan,
            _ => return Err(Error::UnsupportedMethod { rail: "mc_send" }),
        };
        if pan.len() < 13 || pan.len() > 19 || !pan.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidBeneficiary {
                rail: "mc_send",
                detail: "PAN must be 13–19 digits".to_string(),
            });
        }
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "mc_send",
                detail: "amount must be positive".to_string(),
            });
        }
        let body = DisbursementBody {
            transaction_reference: &req.idempotency_key,
            program_id: &self.program_id,
            currency: req.amount.currency.code(),
            amount: format_amount(req.amount),
            recipient_account_uri: format!("pan:{pan}"),
            recipient_name: &req.beneficiary.name,
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
            "live status query requires Mastercard Send REST client".to_string(),
        ))
    }
}
