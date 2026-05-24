//! Paytm — Standard Checkout + Paytm UPI intent (India).
//!
//! Paytm's merchant surface offers two parallel flows:
//!
//! - **Standard Checkout** — merchant initiates a transaction via
//!   `/theia/api/v1/initiateTransaction`, receives a `txnToken`,
//!   redirects the consumer to Paytm's hosted page where they
//!   choose Paytm wallet, UPI, net-banking, or card-on-file.
//! - **Paytm UPI intent** — merchant emits a `paytmmp://pay?...`
//!   deeplink; the consumer's Paytm app handles it directly
//!   without the hosted page.

use op_core::Currency;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// Transport injected by the operator.
pub trait PaytmTransport: Send + Sync {
    /// POST `body` to `path` and return the response body.
    fn post_json(&self, path: &str, body: &str) -> Result<String>;
}

/// Adapter.
pub struct PaytmAdapter {
    /// Paytm-assigned merchant id (MID).
    pub mid: String,
    /// Paytm-assigned merchant key (signed checksum input).
    pub merchant_key: String,
    /// Industry type id (Paytm taxonomy).
    pub industry_type: String,
    /// Channel id ("WEB" / "WAP" / "APP").
    pub channel_id: String,
    /// Operator-supplied transport.
    pub transport: Box<dyn PaytmTransport>,
}

/// Standard-checkout initiate response.
#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateResp {
    /// Paytm-assigned txnToken.
    pub txn_token: String,
}

impl PaytmAdapter {
    /// Build the standard-checkout initiate body.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidIntent`] on bad input.
    pub fn build_initiate_body(&self, intent: &ChargeIntent) -> Result<String> {
        intent.validate_common()?;
        intent.require_currency(Currency::INR)?;
        let major = intent.amount.minor_units / 100;
        let minor = intent.amount.minor_units % 100;
        let body = serde_json::json!({
            "body": {
                "requestType": "Payment",
                "mid": self.mid,
                "websiteName": "DEFAULT",
                "orderId": intent.merchant_order_id,
                "txnAmount": {
                    "value": format!("{major}.{minor:02}"),
                    "currency": "INR",
                },
                "userInfo": {
                    "custId": intent.consumer_hint.as_deref().unwrap_or("anon"),
                },
                "callbackUrl": intent.notify_url.as_deref().unwrap_or(""),
            },
            "head": {
                "signature": "<operator-injected>",
            }
        });
        Ok(body.to_string())
    }

    /// Parse a standard-checkout initiate response.
    ///
    /// # Errors
    /// Returns [`crate::Error::Parse`] / [`crate::Error::MissingField`].
    pub fn parse_initiate_response(body: &str) -> Result<InitiateResp> {
        let v: serde_json::Value =
            serde_json::from_str(body).map_err(|e| crate::Error::Parse(e.to_string()))?;
        if let Some(head) = v.get("head") {
            if let Some(code) = head.get("responseCode").and_then(|c| c.as_str()) {
                if code != "0000" {
                    let message = head
                        .get("responseMessage")
                        .and_then(|m| m.as_str())
                        .unwrap_or("paytm rejected")
                        .to_string();
                    return Err(crate::Error::ProviderRejected {
                        code: code.to_string(),
                        message,
                    });
                }
            }
        }
        let txn_token = v
            .get("body")
            .and_then(|b| b.get("txnToken"))
            .and_then(|s| s.as_str())
            .ok_or(crate::Error::MissingField("txnToken"))?
            .to_string();
        Ok(InitiateResp { txn_token })
    }

    /// Build a Paytm UPI intent deeplink.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidIntent`] on pre-flight violation.
    pub fn build_upi_intent(&self, intent: &ChargeIntent) -> Result<String> {
        intent.validate_common()?;
        intent.require_currency(Currency::INR)?;
        let major = intent.amount.minor_units / 100;
        let minor = intent.amount.minor_units % 100;
        let uri = format!(
            "paytmmp://pay?pa=paytm.{mid}@paytm&pn=Paytm&tr={order}&am={major}.{minor:02}&cu=INR&mc={ind}",
            mid = self.mid,
            order = intent.merchant_order_id,
            ind = self.industry_type,
        );
        Ok(uri)
    }
}

impl AsiaWallet for PaytmAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::Paytm
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        match intent.presentment {
            PresentmentMode::Deeplink => {
                let uri = self.build_upi_intent(intent)?;
                Ok(ChargeResult {
                    merchant_order_id: intent.merchant_order_id.clone(),
                    provider_transaction_id: String::new(),
                    status: ChargeStatus::Pending,
                    presentment_payload: uri,
                })
            }
            PresentmentMode::Browser | PresentmentMode::InAppJsApi => {
                let body = self.build_initiate_body(intent)?;
                let resp = self.transport.post_json(
                    &format!(
                        "/theia/api/v1/initiateTransaction?mid={}&orderId={}",
                        self.mid, intent.merchant_order_id
                    ),
                    &body,
                )?;
                let parsed = Self::parse_initiate_response(&resp)?;
                Ok(ChargeResult {
                    merchant_order_id: intent.merchant_order_id.clone(),
                    provider_transaction_id: String::new(),
                    status: ChargeStatus::Pending,
                    presentment_payload: parsed.txn_token,
                })
            }
            _ => Err(crate::Error::Unsupported(
                "paytm: only Browser/InAppJsApi/Deeplink supported",
            )),
        }
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        let body = serde_json::json!({
            "body": {
                "mid": self.mid,
                "orderId": merchant_order_id,
            }
        })
        .to_string();
        let resp = self
            .transport
            .post_json("/v3/order/status", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let result = v.get("body").and_then(|b| b.get("resultInfo"));
        let status_code = result
            .and_then(|r| r.get("resultStatus"))
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let status = match status_code {
            "TXN_SUCCESS" => ChargeStatus::Succeeded,
            "PENDING" => ChargeStatus::Pending,
            "TXN_FAILURE" => ChargeStatus::Failed,
            _ => ChargeStatus::Unknown,
        };
        let provider_id = v
            .get("body")
            .and_then(|b| b.get("txnId"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        Ok(ChargeResult {
            merchant_order_id: merchant_order_id.to_string(),
            provider_transaction_id: provider_id,
            status,
            presentment_payload: String::new(),
        })
    }
}
