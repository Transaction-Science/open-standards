//! GrabPay — Pay-with-Grab (SEA super-app rail).
//!
//! GrabPay's merchant surface uses an OAuth2 client-credentials
//! flow and a `POST /grabpay/partner/v2/charge/init` initialize
//! call. Six currencies in scope: SGD/MYR/PHP/THB/IDR/VND.

use op_core::Currency;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// Transport injected by the operator.
pub trait GrabTransport: Send + Sync {
    /// POST a JSON body to `path` and return the response body.
    fn post_json(&self, path: &str, body: &str) -> Result<String>;
}

/// Adapter.
pub struct GrabAdapter {
    /// GrabPay partner-merchant id.
    pub partner_merchant_id: String,
    /// Country code (`SG` / `MY` / `PH` / `TH` / `ID` / `VN`).
    pub country: String,
    /// Operator-supplied transport (handles OAuth2 token caching).
    pub transport: Box<dyn GrabTransport>,
}

/// Charge-init response.
#[derive(Serialize, Deserialize, Debug)]
pub struct GrabInitResp {
    /// GrabPay-assigned partnerTxID.
    pub partner_tx_id: String,
    /// Deeplink the consumer's Grab app handles.
    pub request_url: String,
}

impl AsiaWallet for GrabAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::GrabPay
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        intent.validate_common()?;
        if !matches!(
            intent.presentment,
            PresentmentMode::Deeplink | PresentmentMode::Browser
        ) {
            return Err(crate::Error::Unsupported(
                "grab: only Deeplink / Browser supported",
            ));
        }
        // GrabPay accepts any of the six SEA currencies. We rely on
        // the operator to have configured the right one.
        let accepted = [
            Currency::try_new(*b"SGD", 2)?,
            Currency::try_new(*b"MYR", 2)?,
            Currency::try_new(*b"PHP", 2)?,
            Currency::try_new(*b"THB", 2)?,
            Currency::try_new(*b"IDR", 2)?,
            Currency::try_new(*b"VND", 0)?,
        ];
        if !accepted.iter().any(|c| *c == intent.amount.currency) {
            return Err(crate::Error::InvalidIntent(format!(
                "grab: {} not accepted",
                intent.amount.currency
            )));
        }

        let body = serde_json::json!({
            "partnerTxID": intent.merchant_order_id,
            "partnerGroupTxID": intent.merchant_order_id,
            "amount": intent.amount.minor_units,
            "currency": format!("{}", intent.amount.currency),
            "merchantID": self.partner_merchant_id,
            "description": intent.description,
            "countryCode": self.country,
        })
        .to_string();
        let resp = self
            .transport
            .post_json("/grabpay/partner/v2/charge/init", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let partner_tx_id = v
            .get("partnerTxID")
            .and_then(|s| s.as_str())
            .ok_or(crate::Error::MissingField("partnerTxID"))?
            .to_string();
        let request_url = v
            .get("request")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        Ok(ChargeResult {
            merchant_order_id: intent.merchant_order_id.clone(),
            provider_transaction_id: partner_tx_id,
            status: ChargeStatus::Pending,
            presentment_payload: request_url,
        })
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        let body = serde_json::json!({ "partnerTxID": merchant_order_id }).to_string();
        let resp = self
            .transport
            .post_json("/grabpay/partner/v2/charge/status", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let status = match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
            "completed" | "success" => ChargeStatus::Succeeded,
            "pending" | "in_progress" => ChargeStatus::Pending,
            "failed" | "declined" => ChargeStatus::Failed,
            "expired" | "cancelled" => ChargeStatus::Expired,
            _ => ChargeStatus::Unknown,
        };
        Ok(ChargeResult {
            merchant_order_id: merchant_order_id.to_string(),
            provider_transaction_id: String::new(),
            status,
            presentment_payload: String::new(),
        })
    }
}
