//! Alipay Cross-Border Open API v3 adapter.
//!
//! Alipay's v3 Open API (the 2023-onward replacement for the legacy
//! Alipay Global "openapi" surface) is a JSON-over-HTTPS protocol
//! signed with RSA-SHA256. Three flows are in scope:
//!
//! - **Direct-debit** — `alipay.fund.trans.uni.transfer` for
//!   merchant-initiated auto-debit against a previously-authorized
//!   consumer account.
//! - **QR-code (in-store)** — `alipay.trade.precreate` returns a
//!   short URL the merchant renders as a QR; the consumer scans
//!   inside their Alipay app.
//! - **Mini-program (in-app)** — `alipay.trade.create` returns a
//!   `trade_no` the mini-program JS bridge surfaces to
//!   `tp.tradePay` for in-Alipay payment.
//!
//! ## What's in this crate
//!
//! - [`AlipayTransport`] trait (operator-supplied HTTP layer).
//! - [`AlipaySigner`] trait (operator-supplied RSA-SHA256 signer —
//!   keeps the RSA stack out of this crate).
//! - Request shapers + response parsers as pure functions over
//!   `&str` JSON.

use op_core::Currency;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// HTTP transport the operator injects.
///
/// This crate never touches `reqwest`/`tokio` directly. The operator
/// passes in a transport that already knows about timeouts, retries,
/// circuit-breakers, and certificate pinning.
pub trait AlipayTransport: Send + Sync {
    /// POST a signed JSON body to `path` (e.g.
    /// `/v3/alipay/trade/precreate`) and return the response body.
    fn post_json(&self, path: &str, headers: &[(String, String)], body: &str) -> Result<String>;
}

/// RSA-SHA256 signer for the v3 `Authorization` header.
///
/// Alipay v3 signs `auth_string + body` with the merchant's RSA
/// private key. We accept the signer as an injected closure so
/// this crate does not pull in an RSA implementation; operators
/// wire `rsa`, `ring`, KMS, or HSM as they prefer.
pub trait AlipaySigner: Send + Sync {
    /// Return the base64-encoded RSA-SHA256 signature over `payload`.
    fn sign(&self, payload: &[u8]) -> Result<String>;
}

/// Adapter for the three Alipay Cross-Border v3 flows.
pub struct AlipayAdapter {
    /// Alipay-assigned application id.
    pub app_id: String,
    /// Live host (e.g. `https://openapi.alipay.com`).
    pub host: String,
    /// Operator-supplied transport.
    pub transport: Box<dyn AlipayTransport>,
    /// Operator-supplied RSA signer.
    pub signer: Box<dyn AlipaySigner>,
}

/// Shape of the `biz_content` for `alipay.trade.precreate` (QR).
#[derive(Serialize, Deserialize, Debug)]
pub struct PrecreateBiz<'a> {
    /// Merchant out-of-band order id.
    pub out_trade_no: &'a str,
    /// Total amount in CNY major units, two decimal places.
    pub total_amount: String,
    /// Subject line displayed inside Alipay.
    pub subject: &'a str,
    /// Optional notify URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notify_url: Option<&'a str>,
}

/// Decoded response from `alipay.trade.precreate`.
#[derive(Serialize, Deserialize, Debug)]
pub struct PrecreateResp {
    /// Echo of the merchant order id.
    pub out_trade_no: String,
    /// The short URL that becomes a QR.
    pub qr_code: String,
}

impl AlipayAdapter {
    /// Build the `biz_content` JSON for the precreate (QR) flow.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidIntent`] on pre-flight invariant
    /// violations.
    pub fn build_precreate_body(intent: &ChargeIntent) -> Result<String> {
        intent.validate_common()?;
        intent.require_currency(Currency::CNY)?;
        let major = intent.amount.minor_units / 100;
        let minor = intent.amount.minor_units % 100;
        let biz = PrecreateBiz {
            out_trade_no: &intent.merchant_order_id,
            total_amount: format!("{major}.{minor:02}"),
            subject: &intent.description,
            notify_url: intent.notify_url.as_deref(),
        };
        serde_json::to_string(&biz).map_err(|e| crate::Error::Parse(e.to_string()))
    }

    /// Parse a precreate response body into the QR URL.
    ///
    /// Alipay wraps the business response under a top-level
    /// `alipay_trade_precreate_response` envelope with a `code`
    /// field — `"10000"` means success.
    ///
    /// # Errors
    /// Returns [`crate::Error::ProviderRejected`] on non-`"10000"`
    /// response codes, [`crate::Error::Parse`] on malformed JSON.
    pub fn parse_precreate_response(body: &str) -> Result<PrecreateResp> {
        let v: serde_json::Value =
            serde_json::from_str(body).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let envelope = v
            .get("alipay_trade_precreate_response")
            .ok_or(crate::Error::MissingField("alipay_trade_precreate_response"))?;
        let code = envelope
            .get("code")
            .and_then(|c| c.as_str())
            .unwrap_or("unknown");
        if code != "10000" {
            let message = envelope
                .get("sub_msg")
                .and_then(|m| m.as_str())
                .unwrap_or("alipay rejected")
                .to_string();
            let sub_code = envelope
                .get("sub_code")
                .and_then(|c| c.as_str())
                .unwrap_or(code)
                .to_string();
            return Err(crate::Error::ProviderRejected {
                code: sub_code,
                message,
            });
        }
        let out_trade_no = envelope
            .get("out_trade_no")
            .and_then(|s| s.as_str())
            .ok_or(crate::Error::MissingField("out_trade_no"))?
            .to_string();
        let qr_code = envelope
            .get("qr_code")
            .and_then(|s| s.as_str())
            .ok_or(crate::Error::MissingField("qr_code"))?
            .to_string();
        Ok(PrecreateResp {
            out_trade_no,
            qr_code,
        })
    }
}

impl AsiaWallet for AlipayAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::Alipay
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        // We support precreate (QR) and mini-program create here;
        // direct-debit lives behind `alipay.fund.trans.uni.transfer`
        // and is exposed by the refund/payouts surface separately.
        let (path, method) = match intent.presentment {
            PresentmentMode::MerchantPresentedQr => {
                ("/v3/alipay/trade/precreate", "alipay.trade.precreate")
            }
            PresentmentMode::InAppJsApi => {
                ("/v3/alipay/trade/create", "alipay.trade.create")
            }
            _ => {
                return Err(crate::Error::Unsupported(
                    "alipay supports MerchantPresentedQr or InAppJsApi",
                ));
            }
        };

        let body = Self::build_precreate_body(intent)?;
        let signature = self.signer.sign(body.as_bytes())?;
        let auth = format!(
            "ALIPAY-SHA256withRSA app_id={app_id},sign={sig},method={method}",
            app_id = self.app_id,
            sig = signature,
            method = method,
        );
        let headers = vec![
            ("Authorization".to_string(), auth),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        let resp = self.transport.post_json(path, &headers, &body)?;
        let parsed = Self::parse_precreate_response(&resp)?;
        Ok(ChargeResult {
            merchant_order_id: parsed.out_trade_no,
            provider_transaction_id: String::new(),
            status: ChargeStatus::Pending,
            presentment_payload: parsed.qr_code,
        })
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        let body = serde_json::json!({ "out_trade_no": merchant_order_id }).to_string();
        let signature = self.signer.sign(body.as_bytes())?;
        let auth = format!(
            "ALIPAY-SHA256withRSA app_id={},sign={},method=alipay.trade.query",
            self.app_id, signature
        );
        let headers = vec![
            ("Authorization".to_string(), auth),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        let resp = self
            .transport
            .post_json("/v3/alipay/trade/query", &headers, &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let envelope = v
            .get("alipay_trade_query_response")
            .ok_or(crate::Error::MissingField("alipay_trade_query_response"))?;
        let status_str = envelope
            .get("trade_status")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let status = match status_str {
            "TRADE_SUCCESS" | "TRADE_FINISHED" => ChargeStatus::Succeeded,
            "WAIT_BUYER_PAY" => ChargeStatus::Pending,
            "TRADE_CLOSED" => ChargeStatus::Expired,
            _ => ChargeStatus::Unknown,
        };
        let provider_id = envelope
            .get("trade_no")
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
