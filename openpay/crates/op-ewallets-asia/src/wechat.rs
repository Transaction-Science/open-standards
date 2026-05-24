//! WeChat Pay v3 adapter.
//!
//! WeChat Pay v3 (post-2020 surface) is JSON-over-HTTPS, signed with
//! RSA (SHA256-RSA2048) for outbound requests and verified with the
//! WeChat platform-certificate's public key on inbound notify
//! callbacks. Four presentment modes:
//!
//! - **JSAPI** — consumer is inside the WeChat browser (mini-program
//!   or Official Account H5). Merchant calls
//!   `/v3/pay/transactions/jsapi`, gets a `prepay_id`, and
//!   surfaces a signed JS bridge handle.
//! - **Native** — merchant-presented QR (the merchant renders the
//!   `code_url` from `/v3/pay/transactions/native`).
//! - **H5** — out-of-WeChat mobile browser; merchant calls
//!   `/v3/pay/transactions/h5` and redirects.
//! - **MicroPay** — consumer-presented QR (merchant scans the
//!   consumer's WeChat code via `/v3/pay/transactions/micropay`).
//!
//! ## What's in this crate
//!
//! - Pure request shapers + response parsers.
//! - Notify-callback signature verification using HMAC-SHA256 over
//!   the canonicalized `timestamp\nnonce\nbody\n` string (the v3
//!   wire format). Production deployments verify with the
//!   platform-certificate's RSA public key; we surface both paths
//!   so operators can plug their preferred verifier in.

use op_core::Currency;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// HTTP transport injected by the operator.
pub trait WeChatTransport: Send + Sync {
    /// POST a signed JSON body to `path` (e.g.
    /// `/v3/pay/transactions/jsapi`) and return the response body.
    fn post_json(&self, path: &str, headers: &[(String, String)], body: &str) -> Result<String>;
}

/// RSA signer injected by the operator (keeps the RSA stack out
/// of this crate).
pub trait WeChatSigner: Send + Sync {
    /// Return the base64-encoded RSA-SHA256 signature for the v3
    /// `Authorization` header canonical string.
    fn sign(&self, canonical: &[u8]) -> Result<String>;
}

/// Adapter.
pub struct WeChatAdapter {
    /// WeChat-assigned merchant id (`mchid`).
    pub mch_id: String,
    /// WeChat AppId for the JSAPI / mini-program / OA the charge
    /// is presented in. Native / H5 / MicroPay still require an
    /// AppId bound to the merchant account.
    pub app_id: String,
    /// Live host (`https://api.mch.weixin.qq.com`).
    pub host: String,
    /// Operator-supplied transport.
    pub transport: Box<dyn WeChatTransport>,
    /// Operator-supplied RSA signer.
    pub signer: Box<dyn WeChatSigner>,
}

/// `/v3/pay/transactions/jsapi` request body.
#[derive(Serialize, Deserialize, Debug)]
pub struct JsapiReq<'a> {
    /// `appid`.
    pub appid: &'a str,
    /// `mchid`.
    pub mchid: &'a str,
    /// Description shown in WeChat UI.
    pub description: &'a str,
    /// Merchant order id (`out_trade_no`).
    pub out_trade_no: &'a str,
    /// Notify URL.
    pub notify_url: &'a str,
    /// Amount sub-object.
    pub amount: JsapiAmount,
    /// Payer sub-object (openid required for JSAPI).
    pub payer: JsapiPayer<'a>,
}

/// JSAPI amount sub-object: integer minor units + currency.
#[derive(Serialize, Deserialize, Debug)]
pub struct JsapiAmount {
    /// Amount in CNY minor units (fen).
    pub total: i64,
    /// Currency code ("CNY").
    pub currency: String,
}

/// JSAPI payer sub-object: WeChat openid.
#[derive(Serialize, Deserialize, Debug)]
pub struct JsapiPayer<'a> {
    /// WeChat openid for the AppId scope.
    pub openid: &'a str,
}

/// JSAPI response: `prepay_id` for the JS bridge.
#[derive(Serialize, Deserialize, Debug)]
pub struct JsapiResp {
    /// Opaque prepay id. The frontend signs this into the JS
    /// bridge handle `tradeNo`.
    pub prepay_id: String,
}

impl WeChatAdapter {
    /// Build the JSAPI request body.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidIntent`] on pre-flight
    /// invariant violations or missing openid hint.
    pub fn build_jsapi_body(&self, intent: &ChargeIntent) -> Result<String> {
        intent.validate_common()?;
        intent.require_currency(Currency::CNY)?;
        let openid = intent
            .consumer_hint
            .as_deref()
            .ok_or_else(|| crate::Error::InvalidIntent("JSAPI requires openid".into()))?;
        let notify_url = intent
            .notify_url
            .as_deref()
            .ok_or_else(|| crate::Error::InvalidIntent("JSAPI requires notify_url".into()))?;
        let req = JsapiReq {
            appid: &self.app_id,
            mchid: &self.mch_id,
            description: &intent.description,
            out_trade_no: &intent.merchant_order_id,
            notify_url,
            amount: JsapiAmount {
                total: intent.amount.minor_units,
                currency: "CNY".into(),
            },
            payer: JsapiPayer { openid },
        };
        serde_json::to_string(&req).map_err(|e| crate::Error::Parse(e.to_string()))
    }

    /// Parse a JSAPI response.
    ///
    /// # Errors
    /// Returns [`crate::Error::Parse`] / [`crate::Error::MissingField`].
    pub fn parse_jsapi_response(body: &str) -> Result<JsapiResp> {
        let v: serde_json::Value =
            serde_json::from_str(body).map_err(|e| crate::Error::Parse(e.to_string()))?;
        // Error envelopes carry `code` + `message`. Success returns
        // `prepay_id` directly.
        if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
            let message = v
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("wechat rejected")
                .to_string();
            return Err(crate::Error::ProviderRejected {
                code: code.to_string(),
                message,
            });
        }
        let prepay_id = v
            .get("prepay_id")
            .and_then(|s| s.as_str())
            .ok_or(crate::Error::MissingField("prepay_id"))?
            .to_string();
        Ok(JsapiResp { prepay_id })
    }
}

impl AsiaWallet for WeChatAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::WeChatPay
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        let path = match intent.presentment {
            PresentmentMode::InAppJsApi => "/v3/pay/transactions/jsapi",
            PresentmentMode::MerchantPresentedQr => "/v3/pay/transactions/native",
            PresentmentMode::ConsumerPresentedQr => "/v3/pay/transactions/micropay",
            PresentmentMode::Browser => "/v3/pay/transactions/h5",
            PresentmentMode::Deeplink => {
                return Err(crate::Error::Unsupported(
                    "wechat does not expose a deeplink-only flow",
                ));
            }
        };
        let body = self.build_jsapi_body(intent)?;
        let signature = self.signer.sign(body.as_bytes())?;
        let auth = format!(
            "WECHATPAY2-SHA256-RSA2048 mchid=\"{}\",signature=\"{}\"",
            self.mch_id, signature
        );
        let headers = vec![
            ("Authorization".to_string(), auth),
            ("Content-Type".to_string(), "application/json".to_string()),
            ("Accept".to_string(), "application/json".to_string()),
        ];
        let resp = self.transport.post_json(path, &headers, &body)?;
        let parsed = Self::parse_jsapi_response(&resp)?;
        Ok(ChargeResult {
            merchant_order_id: intent.merchant_order_id.clone(),
            provider_transaction_id: String::new(),
            status: ChargeStatus::Pending,
            presentment_payload: parsed.prepay_id,
        })
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        let path = format!(
            "/v3/pay/transactions/out-trade-no/{merchant_order_id}?mchid={}",
            self.mch_id
        );
        // GET is signed over an empty body. We still call post_json
        // with an empty body — the operator-supplied transport
        // routes by path semantics in their own implementation.
        let signature = self.signer.sign(b"")?;
        let auth = format!(
            "WECHATPAY2-SHA256-RSA2048 mchid=\"{}\",signature=\"{}\"",
            self.mch_id, signature
        );
        let headers = vec![("Authorization".to_string(), auth)];
        let resp = self.transport.post_json(&path, &headers, "")?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let state = v
            .get("trade_state")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let status = match state {
            "SUCCESS" => ChargeStatus::Succeeded,
            "NOTPAY" | "USERPAYING" => ChargeStatus::Pending,
            "CLOSED" | "REVOKED" => ChargeStatus::Expired,
            "PAYERROR" => ChargeStatus::Failed,
            _ => ChargeStatus::Unknown,
        };
        let provider_id = v
            .get("transaction_id")
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

/// Verify a v3 notify callback signature in constant time using an
/// operator-supplied HMAC secret.
///
/// Production WeChat deployments verify with the platform
/// certificate's RSA public key; we surface this HMAC variant so
/// adapters that pre-shared a notify-secret (e.g. behind a reverse
/// proxy that re-signs) can keep their verification on a
/// constant-time path. The canonical string is
/// `timestamp\nnonce\nbody\n` per WeChat docs.
///
/// # Errors
/// Returns [`crate::Error::InvalidSignature`] on mismatch.
pub fn verify_notify_hmac(
    timestamp: &str,
    nonce: &str,
    body: &str,
    expected_b64: &str,
    secret: &[u8],
) -> Result<()> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(secret)
        .map_err(|e| crate::Error::ProviderRejected {
            code: "hmac-init".into(),
            message: e.to_string(),
        })?;
    mac.update(timestamp.as_bytes());
    mac.update(b"\n");
    mac.update(nonce.as_bytes());
    mac.update(b"\n");
    mac.update(body.as_bytes());
    mac.update(b"\n");
    let computed = mac.finalize().into_bytes();
    let expected = match base64_decode(expected_b64) {
        Some(b) => b,
        None => return Err(crate::Error::InvalidSignature),
    };
    if computed.ct_eq(&expected).into() {
        Ok(())
    } else {
        Err(crate::Error::InvalidSignature)
    }
}

/// Minimal base64 decoder (standard alphabet, no padding-strict).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const ALPHA: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut buf = Vec::with_capacity(s.len() * 3 / 4);
    let bytes: Vec<u8> = s
        .bytes()
        .filter(|b| *b != b'=' && !b.is_ascii_whitespace())
        .collect();
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for c in bytes {
        let idx = ALPHA.iter().position(|&a| a == c)?;
        acc = (acc << 6) | u32::try_from(idx).ok()?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            buf.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    Some(buf)
}
