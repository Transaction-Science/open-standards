//! Inbound webhook verification + event types.
//!
//! All three providers sign outbound webhooks with HMAC-SHA256 over
//! the raw request body. The signing key is provisioned by the
//! merchant per-provider:
//!
//! | Provider             | Header name                              | Encoding |
//! |----------------------|------------------------------------------|----------|
//! | Affirm               | `X-Affirm-Signature`                     | base64   |
//! | Klarna               | `Klarna-Signature`                       | base64   |
//! | Afterpay / Clearpay  | `Afterpay-Signature` / `Clearpay-Signature` | hex     |
//!
//! We expose three verification helpers plus a unified [`BnplEvent`]
//! deserialiser. Verification is constant-time via `subtle::ConstantTimeEq`.

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::error::{Error, Result};

type HmacSha256 = Hmac<Sha256>;

/// Which provider this event came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebhookProvider {
    /// Affirm.
    Affirm,
    /// Klarna.
    Klarna,
    /// Afterpay or Clearpay.
    AfterpayClearpay,
}

/// Normalised event kind across providers. Each provider emits its
/// own native event names; we map them onto this enum so the
/// orchestrator can fan out a single handler.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BnplEventKind {
    /// Authorisation was created (consumer accepted offer).
    AuthorizationCreated,
    /// Capture occurred.
    CaptureCreated,
    /// Refund was processed.
    RefundCreated,
    /// Consumer raised a dispute.
    DisputeCreated,
    /// Merchant or processor opened a chargeback.
    ChargebackOpened,
    /// Catch-all for provider-native event names we don't have a
    /// dedicated variant for.
    Other,
}

impl BnplEventKind {
    /// Map a provider-native event name to a normalised kind.
    #[must_use]
    pub fn from_event_name(s: &str) -> Self {
        match s {
            "authorization.created" | "auth" | "authorized" | "AUTHORISED" => {
                Self::AuthorizationCreated
            }
            "capture.created" | "capture" | "captured" | "CAPTURED" => Self::CaptureCreated,
            "refund.created" | "refund" | "refunded" | "REFUNDED" => Self::RefundCreated,
            "dispute.created" | "dispute" | "DISPUTED" => Self::DisputeCreated,
            "chargeback.opened" | "chargeback" | "CHARGEBACK_OPENED" => Self::ChargebackOpened,
            _ => Self::Other,
        }
    }
}

/// Normalised inbound event after signature verification + parsing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BnplEvent {
    /// Which provider sent it.
    pub provider: WebhookProvider,
    /// Normalised event kind.
    pub kind: BnplEventKind,
    /// Provider-issued event id.
    pub event_id: Option<String>,
    /// The provider-side resource id the event refers to (charge_id,
    /// order_id, payment_id...).
    pub resource_ref: Option<String>,
    /// Raw provider-native event name, preserved for diagnostics.
    pub raw_event: String,
}

/// Verify an Affirm webhook signature.
///
/// Affirm signs with HMAC-SHA256(secret, body), base64-encoded, sent
/// in the `X-Affirm-Signature` header.
///
/// # Errors
/// [`Error::InvalidSignature`] if the digest mismatches.
/// [`Error::MalformedSignatureHeader`] if the header isn't valid base64.
pub fn verify_affirm_webhook(secret: &[u8], header_value: &str, body: &[u8]) -> Result<()> {
    let provided = B64
        .decode(header_value.trim())
        .map_err(|e| Error::MalformedSignatureHeader(e.to_string()))?;
    verify_hmac_sha256(secret, body, &provided)
}

/// Verify a Klarna webhook signature.
///
/// Klarna's `Klarna-Signature` header carries a base64-encoded
/// HMAC-SHA256 digest over the raw body.
///
/// # Errors
/// See [`verify_affirm_webhook`].
pub fn verify_klarna_webhook(secret: &[u8], header_value: &str, body: &[u8]) -> Result<()> {
    let provided = B64
        .decode(header_value.trim())
        .map_err(|e| Error::MalformedSignatureHeader(e.to_string()))?;
    verify_hmac_sha256(secret, body, &provided)
}

/// Verify an Afterpay / Clearpay webhook signature.
///
/// Afterpay sends a hex-encoded HMAC-SHA256 digest in
/// `Afterpay-Signature` (or `Clearpay-Signature` for UK/EU merchants).
///
/// # Errors
/// See [`verify_affirm_webhook`].
pub fn verify_afterpay_webhook(secret: &[u8], header_value: &str, body: &[u8]) -> Result<()> {
    let provided = hex::decode(header_value.trim())
        .map_err(|e| Error::MalformedSignatureHeader(e.to_string()))?;
    verify_hmac_sha256(secret, body, &provided)
}

/// Common HMAC-SHA256 verifier with constant-time compare.
fn verify_hmac_sha256(secret: &[u8], body: &[u8], provided: &[u8]) -> Result<()> {
    if secret.is_empty() {
        return Err(Error::MalformedSignatureHeader("empty secret".into()));
    }
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| Error::MalformedSignatureHeader(format!("hmac init: {e}")))?;
    mac.update(body);
    let computed = mac.finalize().into_bytes();
    if computed.ct_eq(provided).into() {
        Ok(())
    } else {
        Err(Error::InvalidSignature)
    }
}

#[derive(Deserialize)]
struct AffirmNative {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    charge_id: Option<String>,
}

/// Parse + verify an Affirm webhook in one step.
///
/// # Errors
/// As per [`verify_affirm_webhook`] and JSON parsing.
pub fn parse_affirm_event(
    secret: &[u8],
    header_value: &str,
    body: &[u8],
) -> Result<BnplEvent> {
    verify_affirm_webhook(secret, header_value, body)?;
    let n: AffirmNative =
        serde_json::from_slice(body).map_err(|e| Error::Parse(e.to_string()))?;
    Ok(BnplEvent {
        provider: WebhookProvider::Affirm,
        kind: BnplEventKind::from_event_name(&n.kind),
        event_id: n.id,
        resource_ref: n.charge_id,
        raw_event: n.kind,
    })
}

#[derive(Deserialize)]
struct KlarnaNative {
    #[serde(default)]
    event_id: Option<String>,
    #[serde(rename = "event_type", default)]
    kind: String,
    #[serde(default)]
    order_id: Option<String>,
}

/// Parse + verify a Klarna webhook.
///
/// # Errors
/// See [`verify_klarna_webhook`].
pub fn parse_klarna_event(
    secret: &[u8],
    header_value: &str,
    body: &[u8],
) -> Result<BnplEvent> {
    verify_klarna_webhook(secret, header_value, body)?;
    let n: KlarnaNative =
        serde_json::from_slice(body).map_err(|e| Error::Parse(e.to_string()))?;
    Ok(BnplEvent {
        provider: WebhookProvider::Klarna,
        kind: BnplEventKind::from_event_name(&n.kind),
        event_id: n.event_id,
        resource_ref: n.order_id,
        raw_event: n.kind,
    })
}

#[derive(Deserialize)]
struct AfterpayNative {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "eventType", default)]
    kind: String,
    #[serde(default, rename = "merchantReference")]
    merchant_reference: Option<String>,
    #[serde(default, rename = "paymentId")]
    payment_id: Option<String>,
}

/// Parse + verify an Afterpay / Clearpay webhook.
///
/// # Errors
/// See [`verify_afterpay_webhook`].
pub fn parse_afterpay_event(
    secret: &[u8],
    header_value: &str,
    body: &[u8],
) -> Result<BnplEvent> {
    verify_afterpay_webhook(secret, header_value, body)?;
    let n: AfterpayNative =
        serde_json::from_slice(body).map_err(|e| Error::Parse(e.to_string()))?;
    Ok(BnplEvent {
        provider: WebhookProvider::AfterpayClearpay,
        kind: BnplEventKind::from_event_name(&n.kind),
        event_id: n.id,
        resource_ref: n.payment_id.or(n.merchant_reference),
        raw_event: n.kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign_b64(secret: &[u8], body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret).unwrap();
        mac.update(body);
        B64.encode(mac.finalize().into_bytes())
    }

    fn sign_hex(secret: &[u8], body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret).unwrap();
        mac.update(body);
        hex::encode(mac.finalize().into_bytes())
    }

    #[test]
    fn affirm_verify_positive() {
        let secret = b"affirm-secret";
        let body = br#"{"id":"e1","type":"capture.created","charge_id":"c1"}"#;
        let sig = sign_b64(secret, body);
        assert!(verify_affirm_webhook(secret, &sig, body).is_ok());
    }

    #[test]
    fn affirm_verify_tampered_body() {
        let secret = b"affirm-secret";
        let body = br#"{"id":"e1","type":"capture.created"}"#;
        let sig = sign_b64(secret, body);
        let tampered = br#"{"id":"e1","type":"refund.created"}"#;
        assert!(matches!(
            verify_affirm_webhook(secret, &sig, tampered),
            Err(Error::InvalidSignature)
        ));
    }

    #[test]
    fn affirm_verify_tampered_signature() {
        let secret = b"affirm-secret";
        let body = br#"{"id":"e1"}"#;
        let sig = sign_b64(secret, body);
        let mut bad = sig;
        // Flip a base64 char to a definitely different (still-valid b64) one.
        if let Some(first) = bad.chars().next() {
            let new = if first == 'a' { 'b' } else { 'a' };
            bad.replace_range(..1, &new.to_string());
        }
        assert!(matches!(
            verify_affirm_webhook(secret, &bad, body),
            Err(Error::InvalidSignature)
        ));
    }

    #[test]
    fn klarna_verify_positive() {
        let secret = b"klarna-secret";
        let body = br#"{"event_id":"e","event_type":"capture.created","order_id":"o"}"#;
        let sig = sign_b64(secret, body);
        assert!(verify_klarna_webhook(secret, &sig, body).is_ok());
    }

    #[test]
    fn afterpay_verify_positive_hex() {
        let secret = b"afterpay-secret";
        let body = br#"{"id":"e","eventType":"REFUNDED","paymentId":"p"}"#;
        let sig = sign_hex(secret, body);
        assert!(verify_afterpay_webhook(secret, &sig, body).is_ok());
    }

    #[test]
    fn afterpay_verify_malformed_header() {
        let secret = b"k";
        let body = b"{}";
        let r = verify_afterpay_webhook(secret, "not-hex!", body);
        assert!(matches!(r, Err(Error::MalformedSignatureHeader(_))));
    }

    #[test]
    fn parse_affirm_event_round_trip() {
        let secret = b"k";
        let body = br#"{"id":"evt_1","type":"capture.created","charge_id":"ch_9"}"#;
        let sig = sign_b64(secret, body);
        let evt = parse_affirm_event(secret, &sig, body).unwrap();
        assert_eq!(evt.provider, WebhookProvider::Affirm);
        assert_eq!(evt.kind, BnplEventKind::CaptureCreated);
        assert_eq!(evt.resource_ref.as_deref(), Some("ch_9"));
        assert_eq!(evt.event_id.as_deref(), Some("evt_1"));
    }

    #[test]
    fn parse_klarna_event_round_trip() {
        let secret = b"k";
        let body = br#"{"event_id":"e","event_type":"refund.created","order_id":"o1"}"#;
        let sig = sign_b64(secret, body);
        let evt = parse_klarna_event(secret, &sig, body).unwrap();
        assert_eq!(evt.kind, BnplEventKind::RefundCreated);
        assert_eq!(evt.resource_ref.as_deref(), Some("o1"));
    }

    #[test]
    fn parse_afterpay_event_round_trip() {
        let secret = b"k";
        let body = br#"{"id":"e","eventType":"CHARGEBACK_OPENED","paymentId":"p1"}"#;
        let sig = sign_hex(secret, body);
        let evt = parse_afterpay_event(secret, &sig, body).unwrap();
        assert_eq!(evt.kind, BnplEventKind::ChargebackOpened);
    }

    #[test]
    fn event_kind_mapping_handles_known() {
        assert_eq!(
            BnplEventKind::from_event_name("authorization.created"),
            BnplEventKind::AuthorizationCreated
        );
        assert_eq!(
            BnplEventKind::from_event_name("CAPTURED"),
            BnplEventKind::CaptureCreated
        );
        assert_eq!(
            BnplEventKind::from_event_name("nonsense"),
            BnplEventKind::Other
        );
    }
}
