//! Touch'n Go eWallet (Malaysia).
//!
//! TnG eWallet's merchant API follows Ant Group's "Mini Program"
//! payment-handle shape (TnG eWallet is jointly operated by Ant
//! Group + CIMB). Two flows in scope:
//!
//! - **Mini-program / in-app** — server-to-server call to
//!   `/v1/payments/pay` returns a payment-handle the in-app
//!   mini-program surfaces.
//! - **Merchant-presented QR** — `/v1/payments/precreate` returns
//!   a `qrCodeValue` the merchant renders.

use op_core::Currency;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// Transport injected by the operator.
pub trait TngTransport: Send + Sync {
    /// POST a JSON body and return the response body.
    fn post_json(&self, path: &str, body: &str) -> Result<String>;
}

/// Adapter.
pub struct TngAdapter {
    /// TnG-assigned merchant id.
    pub merchant_id: String,
    /// Store id (TnG-assigned per-physical-location).
    pub store_id: String,
    /// Operator-supplied transport.
    pub transport: Box<dyn TngTransport>,
}

/// `/v1/payments/precreate` response.
#[derive(Serialize, Deserialize, Debug)]
pub struct TngPrecreateResp {
    /// QR string the merchant renders.
    pub qr_code_value: String,
}

impl AsiaWallet for TngAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::TouchNGo
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        intent.validate_common()?;
        intent.require_currency(Currency::try_new(*b"MYR", 2)?)?;
        let path = match intent.presentment {
            PresentmentMode::MerchantPresentedQr => "/v1/payments/precreate",
            PresentmentMode::InAppJsApi => "/v1/payments/pay",
            _ => {
                return Err(crate::Error::Unsupported(
                    "tng: only MerchantPresentedQr / InAppJsApi supported",
                ));
            }
        };
        let body = serde_json::json!({
            "merchantId": self.merchant_id,
            "storeId": self.store_id,
            "merchantOrderId": intent.merchant_order_id,
            "amount": {
                "currency": "MYR",
                "value": intent.amount.minor_units,
            },
            "subject": intent.description,
            "notifyUrl": intent.notify_url.as_deref().unwrap_or(""),
        })
        .to_string();
        let resp = self.transport.post_json(path, &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        if let Some(code) = v.get("resultCode").and_then(|c| c.as_str()) {
            if code != "SUCCESS" {
                return Err(crate::Error::ProviderRejected {
                    code: code.to_string(),
                    message: v
                        .get("resultMessage")
                        .and_then(|m| m.as_str())
                        .unwrap_or("tng rejected")
                        .to_string(),
                });
            }
        }
        let payload = v
            .get("qrCodeValue")
            .or_else(|| v.get("paymentHandle"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        Ok(ChargeResult {
            merchant_order_id: intent.merchant_order_id.clone(),
            provider_transaction_id: String::new(),
            status: ChargeStatus::Pending,
            presentment_payload: payload,
        })
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        let body = serde_json::json!({
            "merchantId": self.merchant_id,
            "merchantOrderId": merchant_order_id,
        })
        .to_string();
        let resp = self.transport.post_json("/v1/payments/inquiry", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let status = match v.get("paymentStatus").and_then(|s| s.as_str()).unwrap_or("") {
            "SUCCESS" => ChargeStatus::Succeeded,
            "PENDING" | "INIT" => ChargeStatus::Pending,
            "FAILED" | "DECLINED" => ChargeStatus::Failed,
            "EXPIRED" | "CANCELLED" => ChargeStatus::Expired,
            _ => ChargeStatus::Unknown,
        };
        let provider_id = v
            .get("acquirerTradeNo")
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
