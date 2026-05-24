//! Integration test for inbound webhook verification across all three
//! providers. Positive cases verify the round trip end-to-end; negative
//! cases exercise the constant-time signature comparison.

use hmac::{Hmac, Mac};
use op_bnpl::{
    BnplEventKind, WebhookProvider, verify_affirm_webhook, verify_afterpay_webhook,
    verify_klarna_webhook,
};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

fn sign_b64(secret: &[u8], body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).expect("hmac");
    mac.update(body);
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
}

fn sign_hex(secret: &[u8], body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).expect("hmac");
    mac.update(body);
    hex::encode(mac.finalize().into_bytes())
}

#[test]
fn affirm_event_round_trip_with_signature() {
    let secret = b"affirm-shared";
    let body = br#"{"id":"e1","type":"capture.created","charge_id":"CHG_9"}"#;
    let sig = sign_b64(secret, body);

    assert!(verify_affirm_webhook(secret, &sig, body).is_ok());

    let evt = op_bnpl::webhook::parse_affirm_event(secret, &sig, body).unwrap();
    assert_eq!(evt.provider, WebhookProvider::Affirm);
    assert_eq!(evt.kind, BnplEventKind::CaptureCreated);
    assert_eq!(evt.resource_ref.as_deref(), Some("CHG_9"));
}

#[test]
fn klarna_event_round_trip_with_signature() {
    let secret = b"klarna-shared";
    let body =
        br#"{"event_id":"evt","event_type":"refund.created","order_id":"ORD_9"}"#;
    let sig = sign_b64(secret, body);

    assert!(verify_klarna_webhook(secret, &sig, body).is_ok());

    let evt = op_bnpl::webhook::parse_klarna_event(secret, &sig, body).unwrap();
    assert_eq!(evt.kind, BnplEventKind::RefundCreated);
    assert_eq!(evt.resource_ref.as_deref(), Some("ORD_9"));
}

#[test]
fn afterpay_event_round_trip_with_signature() {
    let secret = b"afterpay-shared";
    let body =
        br#"{"id":"evt_a","eventType":"AUTHORISED","paymentId":"PAY_9"}"#;
    let sig = sign_hex(secret, body);

    assert!(verify_afterpay_webhook(secret, &sig, body).is_ok());

    let evt = op_bnpl::webhook::parse_afterpay_event(secret, &sig, body).unwrap();
    assert_eq!(evt.kind, BnplEventKind::AuthorizationCreated);
    assert_eq!(evt.resource_ref.as_deref(), Some("PAY_9"));
}

#[test]
fn tampered_affirm_body_rejected() {
    let secret = b"k";
    let body = br#"{"id":"e","type":"capture.created"}"#;
    let sig = sign_b64(secret, body);
    let tampered = br#"{"id":"e","type":"refund.created"}"#;
    let r = verify_affirm_webhook(secret, &sig, tampered);
    assert!(matches!(r, Err(op_bnpl::Error::InvalidSignature)));
}

#[test]
fn tampered_afterpay_hex_signature_rejected() {
    let secret = b"k";
    let body = br#"{"id":"e","eventType":"CAPTURED"}"#;
    let mut sig = sign_hex(secret, body);
    // Flip one hex digit to a different one
    let first = sig.chars().next().unwrap();
    let new = if first == 'a' { 'b' } else { 'a' };
    sig.replace_range(..1, &new.to_string());
    let r = verify_afterpay_webhook(secret, &sig, body);
    assert!(matches!(r, Err(op_bnpl::Error::InvalidSignature)));
}

#[test]
fn klarna_malformed_header_rejected() {
    let r = verify_klarna_webhook(b"k", "!!! not base64 !!!", b"{}");
    assert!(matches!(
        r,
        Err(op_bnpl::Error::MalformedSignatureHeader(_))
    ));
}
