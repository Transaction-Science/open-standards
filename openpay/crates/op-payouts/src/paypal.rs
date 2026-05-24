//! PayPal Payouts API driver.
//!
//! Builds the JSON body for `POST /v1/payments/payouts` per the
//! current PayPal Payouts REST spec. A "batch" with a single item
//! covers the one-payout case the orchestrator drives.

use serde::Serialize;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutRequest, PayoutResult, PayoutStatus,
};
use crate::visa_direct::format_amount;

/// PayPal Payouts driver.
#[derive(Clone, Debug, Default)]
pub struct PayPalPayoutsDriver {
    /// Subject line shown to recipients.
    pub email_subject: String,
}

#[derive(Debug, Serialize)]
struct PayoutBatch<'a> {
    sender_batch_header: BatchHeader<'a>,
    items: Vec<PayoutItem<'a>>,
}

#[derive(Debug, Serialize)]
struct BatchHeader<'a> {
    sender_batch_id: &'a str,
    email_subject: &'a str,
    recipient_type: &'static str,
}

#[derive(Debug, Serialize)]
struct PayoutItem<'a> {
    recipient_type: &'static str,
    amount: Amount<'a>,
    receiver: &'a str,
    note: Option<&'a str>,
    sender_item_id: &'a str,
}

#[derive(Debug, Serialize)]
struct Amount<'a> {
    value: String,
    currency: &'a str,
}

impl Payout for PayPalPayoutsDriver {
    fn rail(&self) -> &'static str {
        "paypal_payouts"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        let email = match &req.beneficiary.account {
            BeneficiaryAccount::PayPalEmail(addr) => addr,
            _ => {
                return Err(Error::UnsupportedMethod {
                    rail: "paypal_payouts",
                });
            }
        };
        if !email.contains('@') {
            return Err(Error::InvalidBeneficiary {
                rail: "paypal_payouts",
                detail: "PayPal receiver must look like an email".to_string(),
            });
        }
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "paypal_payouts",
                detail: "amount must be positive".to_string(),
            });
        }
        let body = PayoutBatch {
            sender_batch_header: BatchHeader {
                sender_batch_id: &req.idempotency_key,
                email_subject: &self.email_subject,
                recipient_type: "EMAIL",
            },
            items: vec![PayoutItem {
                recipient_type: "EMAIL",
                amount: Amount {
                    value: format_amount(req.amount),
                    currency: req.amount.currency.code(),
                },
                receiver: email,
                note: req.memo.as_deref(),
                sender_item_id: &req.idempotency_key,
            }],
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
            "PayPal status query requires GET /v1/payments/payouts/{batch_id} via live client"
                .to_string(),
        ))
    }
}
