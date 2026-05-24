//! UPI 2.0 (NPCI, India) — collect, intent, recurring mandate, and
//! VPA resolution, all in this single flat file.
//!
//! UPI is a push-and-pull A2A rail operated by NPCI (National
//! Payments Corporation of India). The merchant talks to a PSP
//! (Payment Service Provider) bank's API, which in turn talks to
//! the NPCI switch. The acceptance flows we model:
//!
//! - **Collect** — merchant initiates a `pay` request against a
//!   consumer VPA. The consumer is pushed a notification on their
//!   UPI app and authenticates the payment.
//! - **Intent** — merchant emits a deeplink URI
//!   (`upi://pay?pa=...`) that the consumer's wallet app handles.
//!   The merchant has no consumer VPA — the consumer's wallet
//!   surfaces its own account.
//! - **Mandate** — UPI 2.0's recurring-payment primitive (the
//!   2018-spec addition). Two phases: `mandate_create` and
//!   `mandate_execute`. Each execution carries a UMN (Unique
//!   Mandate Number) and a per-execution nonce.
//! - **VPA resolution** — `validate_address` lookup that returns
//!   the registered customer-name behind a VPA before payment.
//!
//! ## What's in this crate
//!
//! - [`UpiTransport`] trait for the PSP HTTPS surface.
//! - Pure helpers for VPA validation + intent-URI encoding.
//! - The [`UpiAdapter`] that implements [`AsiaWallet`] and the
//!   mandate primitives.

use op_core::Currency;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// PSP transport (the merchant's UPI-PSP-bank API).
pub trait UpiTransport: Send + Sync {
    /// POST a JSON body to a PSP-relative `path`. The transport is
    /// expected to handle PSP-specific mTLS, auth headers, and
    /// retries.
    fn post_json(&self, path: &str, body: &str) -> Result<String>;
}

/// UPI adapter.
pub struct UpiAdapter {
    /// Merchant VPA (e.g. `acme@hdfcbank`).
    pub merchant_vpa: String,
    /// Merchant display name shown in the consumer's UPI app.
    pub merchant_name: String,
    /// Merchant category code (4-digit MCC).
    pub mcc: String,
    /// PSP transport.
    pub transport: Box<dyn UpiTransport>,
}

/// Recurrence pattern for a UPI 2.0 mandate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MandateRecurrence {
    /// One-time mandate (pre-debit auth, single execution).
    OneTime,
    /// Daily.
    Daily,
    /// Weekly.
    Weekly,
    /// Monthly.
    Monthly,
    /// Quarterly.
    Quarterly,
    /// Half-yearly.
    HalfYearly,
    /// Yearly.
    Yearly,
    /// "AS_AND_WHEN_PRESENTED" — variable-frequency, capped by amount.
    AsPresented,
}

impl MandateRecurrence {
    fn npci_code(self) -> &'static str {
        match self {
            Self::OneTime => "ONETIME",
            Self::Daily => "DAILY",
            Self::Weekly => "WEEKLY",
            Self::Monthly => "MONTHLY",
            Self::Quarterly => "QUARTERLY",
            Self::HalfYearly => "HALFYEARLY",
            Self::Yearly => "YEARLY",
            Self::AsPresented => "ASPRESENTED",
        }
    }
}

/// A UPI 2.0 mandate request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MandateRequest {
    /// Merchant correlation id for the mandate creation.
    pub merchant_reference: String,
    /// Consumer VPA.
    pub payer_vpa: String,
    /// Maximum per-execution amount in INR minor units (paise).
    pub amount_limit_minor: i64,
    /// Recurrence.
    pub recurrence: MandateRecurrence,
    /// Validity window start (inclusive).
    pub valid_from: OffsetDateTime,
    /// Validity window end (inclusive).
    pub valid_until: OffsetDateTime,
}

/// Response carrying the NPCI-assigned UMN (Unique Mandate Number).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MandateResponse {
    /// Echo of the merchant reference.
    pub merchant_reference: String,
    /// NPCI-assigned Unique Mandate Number.
    pub umn: String,
    /// Current lifecycle state (`"ACTIVE"`, `"REVOKED"`, `"PAUSED"`).
    pub state: String,
}

/// A single mandate execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MandateExecution {
    /// UMN to execute against.
    pub umn: String,
    /// Per-execution merchant order id.
    pub merchant_order_id: String,
    /// Amount in INR minor units. Must be ≤ the mandate's
    /// `amount_limit_minor`.
    pub amount_minor: i64,
    /// 16-hex-byte execution nonce. Operators generate per-call.
    pub nonce_hex: String,
}

/// VPA-resolution result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VpaResolution {
    /// Echo of the VPA queried.
    pub vpa: String,
    /// Registered customer name as known to the NPCI directory.
    pub customer_name: String,
    /// True if the VPA is currently active.
    pub active: bool,
}

impl UpiAdapter {
    /// Validate a VPA's `<handle>@<psp>` syntactic form.
    ///
    /// NPCI VPAs are ASCII, `<handle>@<psp-suffix>`, handle 3-20
    /// chars `[a-zA-Z0-9._-]`, psp 2-20 chars `[a-zA-Z0-9]`.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidVpa`] on malformed input.
    pub fn validate_vpa_syntax(vpa: &str) -> Result<()> {
        let (handle, psp) = vpa
            .split_once('@')
            .ok_or_else(|| crate::Error::InvalidVpa(vpa.into()))?;
        if handle.len() < 3 || handle.len() > 50 {
            return Err(crate::Error::InvalidVpa(vpa.into()));
        }
        if psp.len() < 2 || psp.len() > 20 {
            return Err(crate::Error::InvalidVpa(vpa.into()));
        }
        for c in handle.chars() {
            if !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) {
                return Err(crate::Error::InvalidVpa(vpa.into()));
            }
        }
        for c in psp.chars() {
            if !c.is_ascii_alphanumeric() {
                return Err(crate::Error::InvalidVpa(vpa.into()));
            }
        }
        Ok(())
    }

    /// Encode a UPI deeplink (`upi://pay?...`).
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidIntent`] on pre-flight
    /// invariant violations.
    pub fn encode_intent_uri(&self, intent: &ChargeIntent) -> Result<String> {
        intent.validate_common()?;
        intent.require_currency(Currency::INR)?;
        Self::validate_vpa_syntax(&self.merchant_vpa)?;
        let major = intent.amount.minor_units / 100;
        let minor = intent.amount.minor_units % 100;
        let mut uri = String::from("upi://pay?");
        uri.push_str(&format!("pa={}", percent_encode(&self.merchant_vpa)));
        uri.push_str(&format!("&pn={}", percent_encode(&self.merchant_name)));
        uri.push_str(&format!("&tr={}", percent_encode(&intent.merchant_order_id)));
        uri.push_str(&format!("&am={major}.{minor:02}"));
        uri.push_str("&cu=INR");
        uri.push_str(&format!("&mc={}", percent_encode(&self.mcc)));
        uri.push_str(&format!("&tn={}", percent_encode(&intent.description)));
        Ok(uri)
    }

    /// Resolve a VPA to its registered customer name via the PSP's
    /// `validateAddress` endpoint.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidVpa`] or
    /// [`crate::Error::Transport`].
    pub fn resolve_vpa(&self, vpa: &str) -> Result<VpaResolution> {
        Self::validate_vpa_syntax(vpa)?;
        let body = serde_json::json!({ "vpa": vpa }).to_string();
        let resp = self.transport.post_json("/upi/v2/validate-address", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let customer_name = v
            .get("customerName")
            .and_then(|s| s.as_str())
            .ok_or(crate::Error::MissingField("customerName"))?
            .to_string();
        let active = v.get("active").and_then(|b| b.as_bool()).unwrap_or(false);
        Ok(VpaResolution {
            vpa: vpa.to_string(),
            customer_name,
            active,
        })
    }

    /// Create a UPI 2.0 mandate.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidVpa`] on bad payer VPA,
    /// [`crate::Error::Transport`] on PSP failure.
    pub fn create_mandate(&self, req: &MandateRequest) -> Result<MandateResponse> {
        Self::validate_vpa_syntax(&req.payer_vpa)?;
        if req.amount_limit_minor <= 0 {
            return Err(crate::Error::InvalidIntent("amount_limit must be positive".into()));
        }
        if req.valid_until <= req.valid_from {
            return Err(crate::Error::InvalidIntent("valid_until must be after valid_from".into()));
        }
        let body = serde_json::json!({
            "merchantReference": req.merchant_reference,
            "payerVpa": req.payer_vpa,
            "payeeVpa": self.merchant_vpa,
            "amountLimitMinor": req.amount_limit_minor,
            "recurrence": req.recurrence.npci_code(),
            "validFrom": req.valid_from.unix_timestamp(),
            "validUntil": req.valid_until.unix_timestamp(),
        })
        .to_string();
        let resp = self.transport.post_json("/upi/v2/mandate/create", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let umn = v
            .get("umn")
            .and_then(|s| s.as_str())
            .ok_or(crate::Error::MissingField("umn"))?
            .to_string();
        let state = v
            .get("state")
            .and_then(|s| s.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        Ok(MandateResponse {
            merchant_reference: req.merchant_reference.clone(),
            umn,
            state,
        })
    }

    /// Execute a previously-created mandate.
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidIntent`] on a malformed
    /// nonce, [`crate::Error::Transport`] on PSP failure.
    pub fn execute_mandate(&self, exec: &MandateExecution) -> Result<ChargeResult> {
        if exec.umn.is_empty() {
            return Err(crate::Error::InvalidIntent("empty umn".into()));
        }
        if exec.nonce_hex.len() != 32 {
            return Err(crate::Error::InvalidIntent(
                "nonce_hex must be 32 hex chars (16 bytes)".into(),
            ));
        }
        hex::decode(&exec.nonce_hex)
            .map_err(|e| crate::Error::InvalidIntent(format!("nonce: {e}")))?;
        if exec.amount_minor <= 0 {
            return Err(crate::Error::InvalidIntent("amount must be positive".into()));
        }
        let body = serde_json::json!({
            "umn": exec.umn,
            "merchantOrderId": exec.merchant_order_id,
            "amountMinor": exec.amount_minor,
            "nonceHex": exec.nonce_hex,
        })
        .to_string();
        let resp = self.transport.post_json("/upi/v2/mandate/execute", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let txn_id = v
            .get("txnId")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let status = match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
            "SUCCESS" => ChargeStatus::Succeeded,
            "PENDING" | "DEEMED" => ChargeStatus::Pending,
            "FAILURE" | "DECLINED" => ChargeStatus::Failed,
            _ => ChargeStatus::Unknown,
        };
        Ok(ChargeResult {
            merchant_order_id: exec.merchant_order_id.clone(),
            provider_transaction_id: txn_id,
            status,
            presentment_payload: String::new(),
        })
    }
}

impl AsiaWallet for UpiAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::Upi
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        match intent.presentment {
            PresentmentMode::Deeplink => {
                let uri = self.encode_intent_uri(intent)?;
                Ok(ChargeResult {
                    merchant_order_id: intent.merchant_order_id.clone(),
                    provider_transaction_id: String::new(),
                    status: ChargeStatus::Pending,
                    presentment_payload: uri,
                })
            }
            _ => {
                // Collect flow: push request against consumer VPA.
                intent.validate_common()?;
                intent.require_currency(Currency::INR)?;
                let payer = intent.consumer_hint.as_deref().ok_or_else(|| {
                    crate::Error::InvalidIntent("UPI collect requires payer VPA".into())
                })?;
                Self::validate_vpa_syntax(payer)?;
                let body = serde_json::json!({
                    "merchantOrderId": intent.merchant_order_id,
                    "payerVpa": payer,
                    "payeeVpa": self.merchant_vpa,
                    "amountMinor": intent.amount.minor_units,
                    "currency": "INR",
                    "note": intent.description,
                })
                .to_string();
                let resp = self.transport.post_json("/upi/v2/collect", &body)?;
                let v: serde_json::Value = serde_json::from_str(&resp)
                    .map_err(|e| crate::Error::Parse(e.to_string()))?;
                let txn_id = v
                    .get("txnId")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();
                let status =
                    match v.get("status").and_then(|s| s.as_str()).unwrap_or("PENDING") {
                        "SUCCESS" => ChargeStatus::Succeeded,
                        "PENDING" | "DEEMED" => ChargeStatus::Pending,
                        "FAILURE" | "DECLINED" => ChargeStatus::Failed,
                        _ => ChargeStatus::Unknown,
                    };
                Ok(ChargeResult {
                    merchant_order_id: intent.merchant_order_id.clone(),
                    provider_transaction_id: txn_id,
                    status,
                    presentment_payload: String::new(),
                })
            }
        }
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        let body =
            serde_json::json!({ "merchantOrderId": merchant_order_id }).to_string();
        let resp = self.transport.post_json("/upi/v2/status", &body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| crate::Error::Parse(e.to_string()))?;
        let txn_id = v
            .get("txnId")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let status = match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
            "SUCCESS" => ChargeStatus::Succeeded,
            "PENDING" | "DEEMED" => ChargeStatus::Pending,
            "FAILURE" | "DECLINED" => ChargeStatus::Failed,
            _ => ChargeStatus::Unknown,
        };
        Ok(ChargeResult {
            merchant_order_id: merchant_order_id.to_string(),
            provider_transaction_id: txn_id,
            status,
            presentment_payload: String::new(),
        })
    }
}

/// Minimal `application/x-www-form-urlencoded` percent encoder for
/// the UPI intent URI. Encodes everything outside the RFC 3986
/// unreserved set.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
