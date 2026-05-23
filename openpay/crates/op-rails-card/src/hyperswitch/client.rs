//! Hyperswitch HTTP client.
//!
//! Implements [`CardAcquirer`] by talking JSON over HTTPS to a Hyperswitch
//! deployment. Default endpoint is `https://sandbox.hyperswitch.io`; the
//! struct accepts any base URL so self-hosted deployments work too.

use op_core::PaymentMethod;
use uuid::Uuid;

use crate::acquirer::{
    AuthDecision, AuthRequest, CaptureRequest, CardAcquirer, RefundRequest, ThreeDsMode,
    VoidReason, VoidRequest,
};
use crate::error::{Error, Result};

use super::status_map;
use super::wire;

/// Hyperswitch HTTP client.
///
/// Holds the API key and base URL. Cloneable for sharing across threads
/// (ureq agents are internally `Arc`-shared so this is cheap).
#[derive(Clone)]
pub struct HyperswitchClient {
    base_url: String,
    api_key: String,
    agent: ureq::Agent,
}

impl HyperswitchClient {
    /// Default Hyperswitch sandbox URL.
    pub const SANDBOX: &'static str = "https://sandbox.hyperswitch.io";
    /// Default Hyperswitch production URL.
    pub const PRODUCTION: &'static str = "https://api.hyperswitch.io";

    /// Construct a client targeting the given base URL with the given
    /// API key. Use [`Self::SANDBOX`] for testing.
    #[must_use]
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(30))
                .build(),
        }
    }

    /// Build the full URL for an endpoint path (e.g. `"/payments"`).
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    /// Hyperswitch's `payment_id` field is 30 chars. UUID v7 hex is 32
    /// chars; we truncate to 30 to satisfy the API.
    fn generate_payment_id() -> String {
        let mut s = Uuid::now_v7().simple().to_string();
        s.truncate(30);
        s
    }

    /// POST a JSON body, parse a JSON response. Centralized so we don't
    /// repeat error-mapping logic.
    fn post_json<Req, Resp>(&self, path: &str, body: &Req) -> Result<Resp>
    where
        Req: serde::Serialize,
        Resp: serde::de::DeserializeOwned,
    {
        let url = self.url(path);
        let resp = self
            .agent
            .post(&url)
            .set("api-key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(body);

        match resp {
            Ok(r) => r
                .into_json::<Resp>()
                .map_err(|e| Error::Parse(e.to_string())),
            Err(ureq::Error::Status(status, response)) => {
                // PSP responded with a non-2xx. Try to parse the
                // documented error envelope; fall back to raw body.
                let body = response.into_string().unwrap_or_default();
                if let Ok(env) = serde_json::from_str::<wire::ErrorEnvelope>(&body) {
                    Err(Error::PspRejected {
                        status,
                        code: env.error.code,
                        message: env.error.message,
                    })
                } else {
                    Err(Error::PspRejected {
                        status,
                        code: "unknown".into(),
                        message: body,
                    })
                }
            }
            Err(ureq::Error::Transport(t)) => Err(Error::Transport(t.to_string())),
        }
    }

    /// Map a [`wire::PaymentsResponse`] into an [`AuthDecision`].
    fn to_decision(resp: wire::PaymentsResponse) -> Result<AuthDecision> {
        let status = status_map::map(&resp.status)?;

        // Compute authorized_amount: prefer received, else capturable+received.
        let authorized_amount = resp
            .amount_received
            .map(|r| op_core::Money::from_minor(r, currency_from_code(&resp.currency)));

        let redirect_url = resp
            .next_action
            .as_ref()
            .and_then(|na| na.redirect_to_url.clone());

        Ok(AuthDecision {
            psp_payment_id: resp.payment_id,
            status,
            raw_status: resp.status,
            authorized_amount,
            redirect_url,
            error_code: resp.error_code,
            error_message: resp.error_message,
        })
    }
}

/// Translate an alpha-3 currency code to [`op_core::Currency`].
///
/// Falls back to constructing a generic currency with 2 decimal places
/// if the code is unknown. Hyperswitch only sends alpha-3 codes from
/// its declared enum, so this fallback is defensive.
fn currency_from_code(code: &str) -> op_core::Currency {
    match code {
        "USD" => op_core::Currency::USD,
        "EUR" => op_core::Currency::EUR,
        "BRL" => op_core::Currency::BRL,
        "INR" => op_core::Currency::INR,
        "GBP" => op_core::Currency::GBP,
        "JPY" => op_core::Currency::JPY,
        "CNY" => op_core::Currency::CNY,
        _ => {
            if code.len() == 3 {
                let bytes = code.as_bytes();
                let arr = [bytes[0], bytes[1], bytes[2]];
                op_core::Currency::try_new(arr, 2).unwrap_or(op_core::Currency::USD)
            } else {
                op_core::Currency::USD
            }
        }
    }
}

impl CardAcquirer for HyperswitchClient {
    fn name(&self) -> &'static str {
        "hyperswitch"
    }

    fn supports(&self, method: &PaymentMethod) -> bool {
        matches!(
            method,
            PaymentMethod::Vault(_) | PaymentMethod::Wallet(_) | PaymentMethod::Emv(_)
        )
    }

    fn authorize(&self, req: &AuthRequest) -> Result<AuthDecision> {
        if !self.supports(&req.method) {
            return Err(Error::UnsupportedMethod);
        }

        let capture_method = if req.auto_capture {
            "automatic"
        } else {
            "manual"
        };
        let auth_type = match req.three_ds {
            Some(ThreeDsMode::Required) => Some("three_ds"),
            Some(ThreeDsMode::Skip) => Some("no_three_ds"),
            None => None,
        };

        // Payment-method specifics. For Vault refs, pass payment_token;
        // for Wallet and Emv, pass payment_method_data with connector
        // metadata since Tap-to-Pay TLV doesn't map to standard fields.
        let (pm, pmt, pm_data, conn_meta, token_ref) = match &req.method {
            PaymentMethod::Vault(v) => (None, None, None, None, Some(v.as_str().to_owned())),
            PaymentMethod::Wallet(t) => (
                Some("wallet"),
                None,
                Some(serde_json::json!({
                    "wallet": { "encrypted_payload": hex::encode(t.as_bytes()) }
                })),
                None,
                None,
            ),
            PaymentMethod::Emv(t) => (
                Some("card"),
                None,
                None,
                Some(serde_json::json!({
                    "emv_tlv_hex": hex::encode(t.as_bytes()),
                })),
                None,
            ),
            _ => unreachable!("supports() filtered above"),
        };

        let payment_id = Self::generate_payment_id();

        let body = wire::CreatePayment {
            amount: req.amount.minor_units,
            currency: req.amount.currency.code(),
            capture_method: Some(capture_method),
            authentication_type: auth_type,
            payment_id: Some(payment_id),
            customer: None,
            confirm: Some(true), // Auto-confirm with the method we provide.
            payment_method: pm,
            payment_method_type: pmt,
            payment_token: token_ref.as_deref(),
            payment_method_data: pm_data,
            metadata: req.metadata.clone(),
            connector_metadata: conn_meta,
        };

        let resp: wire::PaymentsResponse = self.post_json("/payments", &body)?;
        Self::to_decision(resp)
    }

    fn capture(&self, req: &CaptureRequest) -> Result<AuthDecision> {
        let path = format!("/payments/{}/capture", req.psp_payment_id);
        let body = wire::CapturePayment {
            amount_to_capture: req.amount.minor_units,
        };
        let resp: wire::PaymentsResponse = self.post_json(&path, &body)?;
        Self::to_decision(resp)
    }

    fn void(&self, req: &VoidRequest) -> Result<AuthDecision> {
        let path = format!("/payments/{}/cancel", req.psp_payment_id);
        let reason = match req.reason {
            VoidReason::RequestedByCustomer => "requested_by_customer",
            VoidReason::Duplicate => "duplicate",
            VoidReason::Fraudulent => "fraudulent",
            VoidReason::Other => "other",
        };
        let body = wire::CancelPayment {
            cancellation_reason: reason,
        };
        let resp: wire::PaymentsResponse = self.post_json(&path, &body)?;
        Self::to_decision(resp)
    }

    fn refund(&self, req: &RefundRequest) -> Result<AuthDecision> {
        let body = wire::CreateRefund {
            payment_id: &req.psp_payment_id,
            amount: req.amount.minor_units,
            reason: req.reason.as_deref(),
            merchant_refund_id: req.idempotency_key.clone(),
        };
        let refund: wire::RefundResponse = self.post_json("/refunds", &body)?;
        let status = status_map::map_refund(&refund.status)?;
        Ok(AuthDecision {
            psp_payment_id: refund.payment_id,
            status,
            raw_status: refund.status,
            authorized_amount: Some(op_core::Money::from_minor(
                refund.amount,
                currency_from_code(&refund.currency),
            )),
            redirect_url: None,
            error_code: None,
            error_message: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_constructs_with_sandbox_url() {
        let c = HyperswitchClient::new(HyperswitchClient::SANDBOX, "sk_test");
        assert_eq!(c.base_url, "https://sandbox.hyperswitch.io");
        assert_eq!(c.api_key, "sk_test");
    }

    #[test]
    fn url_strips_trailing_slash() {
        let c = HyperswitchClient::new("https://example.com/", "k");
        assert_eq!(c.url("/payments"), "https://example.com/payments");
    }

    #[test]
    fn supports_vault_wallet_emv_only() {
        use op_core::{A2aKey, Token, VaultRef};
        let c = HyperswitchClient::new(HyperswitchClient::SANDBOX, "k");
        assert!(c.supports(&PaymentMethod::Vault(VaultRef::new("tok_x"))));
        assert!(c.supports(&PaymentMethod::Wallet(Token::new(vec![1, 2]))));
        assert!(c.supports(&PaymentMethod::Emv(Token::new(vec![3, 4]))));
        // A2A and QR are not card-rail.
        assert!(!c.supports(&PaymentMethod::A2a(A2aKey::Pix("k".into()))));
        assert!(!c.supports(&PaymentMethod::Qr("00020101...".into())));
    }

    #[test]
    fn name_is_stable_identifier() {
        let c = HyperswitchClient::new(HyperswitchClient::SANDBOX, "k");
        assert_eq!(c.name(), "hyperswitch");
    }

    #[test]
    fn payment_id_is_30_chars() {
        let id = HyperswitchClient::generate_payment_id();
        assert_eq!(id.len(), 30);
        // Should be hex-ish (UUID v7 simple form is 32 hex chars).
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn currency_from_code_handles_known() {
        assert_eq!(currency_from_code("USD").code(), "USD");
        assert_eq!(currency_from_code("EUR").code(), "EUR");
        assert_eq!(currency_from_code("JPY").exponent(), 0);
        assert_eq!(currency_from_code("USD").exponent(), 2);
    }

    #[test]
    fn currency_from_code_falls_back_to_2dp_for_unknown() {
        // CHF is not in our hardcoded list; we should still construct it
        // with 2 decimal places as the safe default.
        let chf = currency_from_code("CHF");
        assert_eq!(chf.code(), "CHF");
        assert_eq!(chf.exponent(), 2);
    }

    #[test]
    fn currency_from_code_invalid_falls_back_to_usd() {
        let bad = currency_from_code("zzz"); // lowercase, would fail try_new
        // Per our fallback, this should give USD.
        assert_eq!(bad.code(), "USD");
    }
}
