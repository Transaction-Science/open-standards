//! Visa Direct OCT (Original Credit Transaction) driver.
//!
//! Visa Direct pushes funds onto a debit-card PAN or network token via
//! the Visa Direct REST API (`/visadirect/fundstransfer/v1/pushfundstransactions`).
//! This module builds the offline JSON body. The operator submits with
//! their own mTLS-bound client and `x-pay-token` signature.
//!
//! Spec sources:
//! - Visa Developer "Visa Direct — Push Funds API" (Jan 2026 revision).
//! - PCI guidance on PAN handling — we never log the PAN.

use op_core::Money;
use serde::Serialize;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutRequest, PayoutResult, PayoutStatus,
};

/// Visa Direct OCT driver, offline-pure.
#[derive(Clone, Debug, Default)]
pub struct VisaDirectDriver {
    /// `acquiringBin` assigned by the operator's acquiring partner.
    pub acquiring_bin: String,
    /// `acquirerCountryCode` (ISO 3166-1 numeric, e.g. `"840"` for US).
    pub acquirer_country_code: String,
    /// `businessApplicationId` ("AA" account-to-account is the default
    /// for marketplace payouts).
    pub business_application_id: String,
}

#[derive(Debug, Serialize)]
struct PushFundsBody<'a> {
    #[serde(rename = "acquirerCountryCode")]
    acquirer_country_code: &'a str,
    #[serde(rename = "acquiringBin")]
    acquiring_bin: &'a str,
    amount: String,
    #[serde(rename = "businessApplicationId")]
    business_application_id: &'a str,
    #[serde(rename = "transactionCurrencyCode")]
    transaction_currency_code: &'a str,
    #[serde(rename = "recipientPrimaryAccountNumber")]
    recipient_pan: &'a str,
    #[serde(rename = "senderReference")]
    sender_reference: &'a str,
    #[serde(rename = "systemsTraceAuditNumber")]
    stan: u32,
    #[serde(rename = "transactionIdentifier")]
    transaction_identifier: u64,
}

pub(crate) fn format_amount(amount: Money) -> String {
    let exp = u32::from(amount.currency.exponent());
    if exp == 0 {
        return amount.minor_units.to_string();
    }
    let divisor = 10_i64.pow(exp);
    let whole = amount.minor_units / divisor;
    let frac = amount.minor_units.abs() % divisor;
    format!("{whole}.{frac:0width$}", width = exp as usize)
}

impl Payout for VisaDirectDriver {
    fn rail(&self) -> &'static str {
        "visa_direct"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        let pan = match &req.beneficiary.account {
            BeneficiaryAccount::CardPan(pan) | BeneficiaryAccount::CardToken(pan) => pan,
            _ => {
                return Err(Error::UnsupportedMethod {
                    rail: "visa_direct",
                });
            }
        };
        if pan.len() < 13 || pan.len() > 19 || !pan.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidBeneficiary {
                rail: "visa_direct",
                detail: "PAN must be 13–19 digits".to_string(),
            });
        }
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "visa_direct",
                detail: "amount must be positive".to_string(),
            });
        }
        let body = PushFundsBody {
            acquirer_country_code: &self.acquirer_country_code,
            acquiring_bin: &self.acquiring_bin,
            amount: format_amount(req.amount),
            business_application_id: &self.business_application_id,
            transaction_currency_code: req.amount.currency.code(),
            recipient_pan: pan,
            sender_reference: &req.idempotency_key,
            stan: 0,
            transaction_identifier: 0,
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
            "live status query requires VisaDirect REST client".to_string(),
        ))
    }
}
