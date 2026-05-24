//! GoPay (Gojek / GoTo Financial, Indonesia).
//!
//! GoPay's merchant surface is reached through GoTo Financial's
//! Midtrans `Snap` and Core API endpoints. We model the
//! `/v2/charge` GoPay path: server-to-server POST returns a
//! `deeplink_url` the consumer's Gojek app handles. IDR only.

use op_core::Currency;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// Transport injected by the operator.
pub trait GoPayTransport: Send + Sync {
    /// POST a JSON body and return the response body.
    fn post_json(&self, path: &str, body: &str) -> Result<String>;
}

/// Adapter.
pub struct GoPayAdapter {
    /// Midtrans server-key, base64-encoded as `Basic <key>:` for
    /// HTTP basic auth (the transport handles header injection).
    pub server_key_b64: String,
    /// Operator-supplied transport.
    pub transport: Box<dyn GoPayTransport>,
}

/// GoPay charge response.
#[derive(Serialize, Deserialize, Debug)]
pub struct GoPayResp {
    /// Midtrans transaction id.
    pub transaction_id: String,
    /// Deeplink URL the consumer's Gojek app handles.
    pub deeplink_url: String,
}

impl AsiaWallet for GoPayAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::GoPay
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        intent.validate_common()?;
        intent.require_currency(Currency::try_new(*b"IDR", 2)?)?;
        if !matches!(intent.presentment, PresentmentMode::Deeplink) {
            return Err(crate::Error::Unsupported("gopay supports deeplink only"));
        }
        // IDR is 2-exponent in op-core but stored as whole rupiah at
        // the Midtrans surface.
        let gross_amount = intent.amount.minor_units / 100;
        let body = serde_json::json!({
            "payment_type": "gopay",
            "transaction_details": {
                "order_id": intent.merchant_order_id,
                "gross_amount": gross_amount,
            },
            "gopay": {
                "enable_callback": intent.notify_url.is_some(),
                "callback_url": intent.notify_url.as_deref().unwrap_or(""),
            },
        })
        .to_string();
        let resp = self.transport.post_json("/v2/charge", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let status_code = v
            .get("status_code")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        if status_code != "201" && status_code != "200" {
            return Err(crate::Error::ProviderRejected {
                code: status_code.to_string(),
                message: v
                    .get("status_message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("gopay rejected")
                    .to_string(),
            });
        }
        let transaction_id = v
            .get("transaction_id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let deeplink_url = v
            .get("actions")
            .and_then(|a| a.as_array())
            .and_then(|arr| {
                arr.iter().find_map(|act| {
                    if act.get("name").and_then(|n| n.as_str()) == Some("deeplink-redirect") {
                        act.get("url").and_then(|u| u.as_str()).map(String::from)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default();
        Ok(ChargeResult {
            merchant_order_id: intent.merchant_order_id.clone(),
            provider_transaction_id: transaction_id,
            status: ChargeStatus::Pending,
            presentment_payload: deeplink_url,
        })
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        let resp = self
            .transport
            .post_json(&format!("/v2/{merchant_order_id}/status"), "")?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let txn_status = v
            .get("transaction_status")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let status = match txn_status {
            "settlement" | "capture" => ChargeStatus::Succeeded,
            "pending" => ChargeStatus::Pending,
            "deny" | "failure" => ChargeStatus::Failed,
            "expire" | "cancel" => ChargeStatus::Expired,
            _ => ChargeStatus::Unknown,
        };
        let transaction_id = v
            .get("transaction_id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        Ok(ChargeResult {
            merchant_order_id: merchant_order_id.to_string(),
            provider_transaction_id: transaction_id,
            status,
            presentment_payload: String::new(),
        })
    }
}
