//! Webhook signing & verification.
//!
//! The scheme is byte-for-byte compatible with Stripe's so any
//! merchant who has already integrated Stripe webhooks can verify
//! ours with their existing code (modulo the header name).
//!
//! ## On-the-wire format
//!
//! ```text
//! POST /webhooks/payments HTTP/1.1
//! Host: merchant.example.com
//! Content-Type: application/json
//! OpenPay-Signature: t=1700000000,v1=4f4c4d4e4f4c4d4e4f4c4d4e4f4c4d4e4f4c4d4e4f4c4d4e4f4c4d4e4f4c4d4e
//!
//! { ...JSON event body... }
//! ```
//!
//! The signature value: `hmac_sha256_hex(endpoint.secret, "{ts}.{body}")`.
//!
//! ## Why the timestamp in the signed payload?
//!
//! Replay protection. Without the timestamp, an attacker who
//! captures a signed payload (e.g. via a leaked log) could replay
//! it indefinitely. With the timestamp, valid signatures expire
//! after the verifier's tolerance window (default 5 minutes).
//!
//! ## Why `subtle::ConstantTimeEq`?
//!
//! Plain `==` on byte slices short-circuits on the first
//! mismatching byte. An attacker who can measure verification
//! latency can extract the correct signature byte-by-byte. The
//! `subtle` crate's `ct_eq` always processes the full input.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::error::{Error, Result};
use crate::hexutil;

/// HTTP header name carrying the signature.
pub const SIGNATURE_HEADER: &str = "OpenPay-Signature";

/// Default tolerance for the timestamp in seconds.
pub const DEFAULT_TIMESTAMP_TOLERANCE_SECS: i64 = 300; // 5 minutes

type HmacSha256 = Hmac<Sha256>;

/// The byte string that gets HMAC'd: `"{timestamp}.{body}"`.
///
/// Exposed as a type so verifiers can construct it once and reuse,
/// and so the signing algorithm is auditable in one place.
#[derive(Debug, Clone)]
pub struct SignedPayload<'a> {
    /// Unix epoch seconds.
    pub timestamp: u64,
    /// Raw request body bytes.
    pub body: &'a [u8],
}

impl<'a> SignedPayload<'a> {
    /// Construct.
    #[must_use]
    pub fn new(timestamp: u64, body: &'a [u8]) -> Self {
        Self { timestamp, body }
    }

    /// Render `"{timestamp}.{body}"` as a fresh byte vector.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let ts_str = self.timestamp.to_string();
        let mut out = Vec::with_capacity(ts_str.len() + 1 + self.body.len());
        out.extend_from_slice(ts_str.as_bytes());
        out.push(b'.');
        out.extend_from_slice(self.body);
        out
    }
}

/// Compute the HMAC-SHA256 signature in hex.
///
/// The returned string is the value of `v1=...` in the signature
/// header.
///
/// # Errors
/// [`Error::InvalidInput`] if `secret` is empty. The HMAC algorithm
/// technically accepts an empty key, but it's almost always a
/// configuration bug (operator forgot to set it).
pub fn compute_signature(secret: &[u8], payload: &SignedPayload<'_>) -> Result<String> {
    if secret.is_empty() {
        return Err(Error::InvalidInput("signing secret is empty".into()));
    }
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| Error::InvalidInput(format!("HMAC key init: {e}")))?;
    mac.update(payload.to_bytes().as_slice());
    let bytes = mac.finalize().into_bytes();
    Ok(hexutil::encode(&bytes))
}

/// Build the full signature header value: `t=<ts>,v1=<hex>`.
#[must_use]
pub fn build_signature_header(timestamp: u64, signature_hex: &str) -> String {
    format!("t={timestamp},v1={signature_hex}")
}

/// Parse a signature header into (timestamp, list of v1 sigs).
///
/// Stripe permits multiple `v1=...` entries during key rotation.
/// We accept the same shape: a comma-separated list of `k=v` pairs
/// where `t` appears once and `v1` may appear multiple times.
///
/// # Errors
/// [`Error::MalformedSignature`] if `t` is missing, has no valid
/// u64 value, or no `v1` entries are present.
pub fn parse_signature_header(header: &str) -> Result<(u64, Vec<String>)> {
    let mut timestamp: Option<u64> = None;
    let mut sigs: Vec<String> = Vec::new();
    for part in header.split(',') {
        let part = part.trim();
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| Error::MalformedSignature(format!("part lacks '=': {part}")))?;
        match key {
            "t" => {
                let ts: u64 = value
                    .parse()
                    .map_err(|e| Error::MalformedSignature(format!("bad timestamp: {e}")))?;
                timestamp = Some(ts);
            }
            "v1" => sigs.push(value.to_owned()),
            _ => { /* ignore unknown schemes; lets us add v2 later */ }
        }
    }
    let timestamp = timestamp.ok_or_else(|| Error::MalformedSignature("missing t=".into()))?;
    if sigs.is_empty() {
        return Err(Error::MalformedSignature("no v1= signatures".into()));
    }
    Ok((timestamp, sigs))
}

/// Verify a signature header against the raw body and shared secret.
///
/// Accepts ANY of the listed `v1=` signatures (for rotation
/// scenarios). Performs constant-time comparison.
///
/// # Errors
/// - [`Error::MalformedSignature`] if the header doesn't parse.
/// - [`Error::TimestampOutOfTolerance`] if `|now - t| > tolerance_secs`.
/// - [`Error::SignatureMismatch`] if no `v1=` signature matches.
pub fn verify_signature(
    secret: &[u8],
    body: &[u8],
    header: &str,
    now_unix_secs: u64,
    tolerance_secs: i64,
) -> Result<()> {
    let (ts, sigs) = parse_signature_header(header)?;
    // Tolerance check.
    let delta = (now_unix_secs as i64) - (ts as i64);
    if delta.abs() > tolerance_secs {
        return Err(Error::TimestampOutOfTolerance { delta_secs: delta });
    }
    // Compute expected.
    let expected = compute_signature(secret, &SignedPayload::new(ts, body))?;
    let expected_bytes = expected.as_bytes();
    // Try each provided v1.
    for sig in &sigs {
        let sig_bytes = sig.as_bytes();
        if sig_bytes.len() == expected_bytes.len() && sig_bytes.ct_eq(expected_bytes).into() {
            return Ok(());
        }
    }
    Err(Error::SignatureMismatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &[u8] = b"whsec_test_super_secret_key_v1";

    #[test]
    fn signed_payload_renders_correctly() {
        let p = SignedPayload::new(1700000000, b"hello");
        assert_eq!(p.to_bytes(), b"1700000000.hello".to_vec());
    }

    #[test]
    fn signed_payload_with_empty_body() {
        let p = SignedPayload::new(1700000000, b"");
        assert_eq!(p.to_bytes(), b"1700000000.".to_vec());
    }

    #[test]
    fn compute_signature_is_deterministic() {
        let p = SignedPayload::new(1700000000, b"body");
        let a = compute_signature(SECRET, &p).unwrap();
        let b = compute_signature(SECRET, &p).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64, "SHA-256 = 32 bytes = 64 hex chars");
    }

    #[test]
    fn different_secrets_produce_different_signatures() {
        let p = SignedPayload::new(1700000000, b"body");
        let a = compute_signature(SECRET, &p).unwrap();
        let b = compute_signature(b"other_secret", &p).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn different_bodies_produce_different_signatures() {
        let a = compute_signature(SECRET, &SignedPayload::new(1700000000, b"body-a")).unwrap();
        let b = compute_signature(SECRET, &SignedPayload::new(1700000000, b"body-b")).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn different_timestamps_produce_different_signatures() {
        let a = compute_signature(SECRET, &SignedPayload::new(1700000000, b"x")).unwrap();
        let b = compute_signature(SECRET, &SignedPayload::new(1700000001, b"x")).unwrap();
        assert_ne!(a, b, "timestamp is part of the signed payload");
    }

    #[test]
    fn empty_secret_rejected() {
        let p = SignedPayload::new(0, b"body");
        let r = compute_signature(&[], &p);
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[test]
    fn build_signature_header_format() {
        let h = build_signature_header(1700000000, "abc123");
        assert_eq!(h, "t=1700000000,v1=abc123");
    }

    #[test]
    fn parse_signature_header_extracts_t_and_v1() {
        let (ts, sigs) = parse_signature_header("t=1700000000,v1=abc123").unwrap();
        assert_eq!(ts, 1700000000);
        assert_eq!(sigs, vec!["abc123".to_string()]);
    }

    #[test]
    fn parse_signature_header_supports_multiple_v1() {
        // Key rotation: old + new secrets both produce v1 entries.
        let (ts, sigs) = parse_signature_header("t=1700000000,v1=old,v1=new").unwrap();
        assert_eq!(ts, 1700000000);
        assert_eq!(sigs, vec!["old".to_string(), "new".to_string()]);
    }

    #[test]
    fn parse_signature_header_ignores_unknown_schemes() {
        // v2 might be a future scheme; we ignore it without failing
        // so old clients work against new senders.
        let (ts, sigs) = parse_signature_header("t=1700000000,v1=abc,v2=xyz").unwrap();
        assert_eq!(ts, 1700000000);
        assert_eq!(sigs, vec!["abc".to_string()]);
    }

    #[test]
    fn parse_signature_header_missing_t_fails() {
        let r = parse_signature_header("v1=abc");
        assert!(matches!(r, Err(Error::MalformedSignature(_))));
    }

    #[test]
    fn parse_signature_header_no_v1_fails() {
        let r = parse_signature_header("t=1700000000");
        assert!(matches!(r, Err(Error::MalformedSignature(_))));
    }

    #[test]
    fn parse_signature_header_bad_timestamp_fails() {
        let r = parse_signature_header("t=notanumber,v1=abc");
        assert!(matches!(r, Err(Error::MalformedSignature(_))));
    }

    #[test]
    fn parse_signature_header_malformed_part_fails() {
        let r = parse_signature_header("not-a-pair");
        assert!(matches!(r, Err(Error::MalformedSignature(_))));
    }

    #[test]
    fn verify_signature_round_trip_succeeds() {
        let body = b"{\"event\":\"payment_authorized\"}";
        let ts = 1700000000u64;
        let sig = compute_signature(SECRET, &SignedPayload::new(ts, body)).unwrap();
        let header = build_signature_header(ts, &sig);
        verify_signature(SECRET, body, &header, ts, 300).unwrap();
    }

    #[test]
    fn verify_signature_wrong_body_fails() {
        let ts = 1700000000u64;
        let sig = compute_signature(SECRET, &SignedPayload::new(ts, b"body-a")).unwrap();
        let header = build_signature_header(ts, &sig);
        let r = verify_signature(SECRET, b"body-b", &header, ts, 300);
        assert!(matches!(r, Err(Error::SignatureMismatch)));
    }

    #[test]
    fn verify_signature_wrong_secret_fails() {
        let body = b"body";
        let ts = 1700000000u64;
        let sig = compute_signature(SECRET, &SignedPayload::new(ts, body)).unwrap();
        let header = build_signature_header(ts, &sig);
        let r = verify_signature(b"other_secret", body, &header, ts, 300);
        assert!(matches!(r, Err(Error::SignatureMismatch)));
    }

    #[test]
    fn verify_signature_outside_tolerance_fails() {
        let body = b"body";
        let ts = 1700000000u64;
        let sig = compute_signature(SECRET, &SignedPayload::new(ts, body)).unwrap();
        let header = build_signature_header(ts, &sig);
        // Pretend "now" is 600 seconds later (10 minutes), tolerance 300.
        let r = verify_signature(SECRET, body, &header, ts + 600, 300);
        match r {
            Err(Error::TimestampOutOfTolerance { delta_secs }) => {
                assert_eq!(delta_secs, 600);
            }
            other => panic!("expected TimestampOutOfTolerance, got {other:?}"),
        }
    }

    #[test]
    fn verify_signature_future_timestamp_outside_tolerance_fails() {
        // Negative delta — server clock is behind.
        let body = b"body";
        let ts = 1700000000u64;
        let sig = compute_signature(SECRET, &SignedPayload::new(ts, body)).unwrap();
        let header = build_signature_header(ts, &sig);
        // "Now" is 600 seconds earlier than the signed timestamp.
        let r = verify_signature(SECRET, body, &header, ts - 600, 300);
        match r {
            Err(Error::TimestampOutOfTolerance { delta_secs }) => {
                assert_eq!(delta_secs, -600);
            }
            other => panic!("expected TimestampOutOfTolerance, got {other:?}"),
        }
    }

    #[test]
    fn verify_signature_rotation_accepts_any_v1() {
        let body = b"body";
        let ts = 1700000000u64;
        let new_sig = compute_signature(SECRET, &SignedPayload::new(ts, body)).unwrap();
        // Header has an old (wrong) sig AND the correct new one.
        let header = format!("t={ts},v1=oldsigthatswrong,v1={new_sig}");
        // Must accept because new_sig matches.
        verify_signature(SECRET, body, &header, ts, 300).unwrap();
    }

    #[test]
    fn verify_signature_constant_time_does_not_short_circuit() {
        // We can't truly measure constant-time without microbenchmarks,
        // but we can at least verify that the rejection path is taken
        // even when the signature length matches. If we used naive ==,
        // most byte-string comparisons would short-circuit; ct_eq does
        // not. Smoke test only.
        let body = b"x";
        let ts = 1700000000u64;
        let real = compute_signature(SECRET, &SignedPayload::new(ts, body)).unwrap();
        // Build a "near-miss" signature with the same length, last byte
        // flipped.
        let mut wrong = real.clone();
        let last = wrong.pop().unwrap();
        let flipped = if last == 'a' { 'b' } else { 'a' };
        wrong.push(flipped);
        let header = build_signature_header(ts, &wrong);
        let r = verify_signature(SECRET, body, &header, ts, 300);
        assert!(matches!(r, Err(Error::SignatureMismatch)));
    }
}
