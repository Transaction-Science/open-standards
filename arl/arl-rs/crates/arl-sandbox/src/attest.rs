//! Attestation — Ed25519 over JCS-canonicalized JSON with SHA-256.
//!
//! The Supervisor signs each session so the measurement is tamper-evident
//! and the artifact is content-addressable. The primitive set is the one
//! ARL-S names and the one the broader receipt ecosystem (Microsoft Agent
//! Governance Toolkit, Mastercard Verifiable Intent) uses, so an
//! attestation is verifiable by anyone with no trust in the issuer:
//!
//! - **JCS (RFC 8785)** canonical JSON — deterministic bytes for the same
//!   logical record, independent of field order. Implemented here for the
//!   integer/string/bool/array/object subset (sessions are float-free by
//!   design); a non-integer float is rejected rather than risk the
//!   ECMAScript number-formatting ambiguity.
//! - **SHA-256 (FIPS 180-4)** of the canonical bytes — the content address.
//! - **Ed25519 (RFC 8032)** signature over the canonical bytes.
//!
//! `verify_attestation` checks both the content hash and the signature.
//! It proves *this key signed this exact session*; the caller must still
//! confirm the key is the expected Supervisor's via
//! [`Attestation::signer_is`].

use std::collections::BTreeMap;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::session::Session;

/// Errors producing or checking an attestation.
#[derive(Debug, Error)]
pub enum AttestError {
    #[error("serialize session: {0}")]
    Serialize(String),
    #[error("session contains a non-integer number, which JCS canonicalization here does not handle; keep the signed record float-free")]
    NonIntegerNumber,
    #[error("malformed hex in attestation field `{0}`")]
    BadHex(&'static str),
    #[error("malformed Ed25519 key or signature")]
    BadKeyOrSig,
}

/// A signed attestation over a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attestation {
    /// SHA-256 of the JCS-canonical session bytes, hex.
    pub content_sha256_hex: String,
    /// Ed25519 signature over the JCS-canonical session bytes, hex (64 B).
    pub signature_hex: String,
    /// Supervisor public key, hex (32 B).
    pub public_key_hex: String,
}

impl Attestation {
    /// True if this attestation was signed by `expected_public_key_hex`.
    /// Use after [`verify_attestation`] to confirm the signer is the
    /// Supervisor you trust, not merely *a* valid signer.
    pub fn signer_is(&self, expected_public_key_hex: &str) -> bool {
        self.public_key_hex.eq_ignore_ascii_case(expected_public_key_hex)
    }
}

/// JCS-canonicalize a JSON value: object keys sorted, no whitespace,
/// integers and strings/bools/null/arrays emitted canonically. Rejects
/// non-integer floats.
pub fn jcs_canonicalize(value: &Value) -> Result<String, AttestError> {
    let mut out = String::new();
    write_canonical(value, &mut out)?;
    Ok(out)
}

fn write_canonical(value: &Value, out: &mut String) -> Result<(), AttestError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                out.push_str(&n.to_string());
            } else {
                return Err(AttestError::NonIntegerNumber);
            }
        }
        Value::String(s) => {
            // serde_json string serialization is RFC 8259 / RFC 8785
            // compliant (minimal escaping).
            let encoded =
                serde_json::to_string(s).map_err(|e| AttestError::Serialize(e.to_string()))?;
            out.push_str(&encoded);
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Sort keys lexicographically (ASCII keys → byte order is
            // the RFC 8785 UTF-16 order).
            let sorted: BTreeMap<&String, &Value> = map.iter().collect();
            out.push('{');
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let key = serde_json::to_string(k.as_str())
                    .map_err(|e| AttestError::Serialize(e.to_string()))?;
                out.push_str(&key);
                out.push(':');
                write_canonical(v, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn from_hex(s: &str, field: &'static str) -> Result<Vec<u8>, AttestError> {
    if s.len() % 2 != 0 {
        return Err(AttestError::BadHex(field));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16).ok_or(AttestError::BadHex(field))?;
        let lo = (bytes[i + 1] as char).to_digit(16).ok_or(AttestError::BadHex(field))?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Ok(out)
}

/// The JCS-canonical bytes of a session (the message that gets hashed and
/// signed). Exposed so a verifier on another stack can reproduce them.
pub fn canonical_bytes(session: &Session) -> Result<Vec<u8>, AttestError> {
    let value = serde_json::to_value(session).map_err(|e| AttestError::Serialize(e.to_string()))?;
    Ok(jcs_canonicalize(&value)?.into_bytes())
}

/// Sign a session with the Supervisor's key.
pub fn attest_session(
    session: &Session,
    signing_key: &SigningKey,
) -> Result<Attestation, AttestError> {
    let msg = canonical_bytes(session)?;
    let digest = Sha256::digest(&msg);
    let signature = signing_key.sign(&msg);
    Ok(Attestation {
        content_sha256_hex: to_hex(&digest),
        signature_hex: to_hex(&signature.to_bytes()),
        public_key_hex: to_hex(signing_key.verifying_key().as_bytes()),
    })
}

/// Verify an attestation against a session: recompute the canonical
/// bytes, check the content hash, and verify the Ed25519 signature
/// against the embedded public key. Returns `Ok(true)` iff both hold.
///
/// This proves the key in the attestation signed this exact session. To
/// confirm the signer is the Supervisor you trust, also call
/// [`Attestation::signer_is`] with the known key.
pub fn verify_attestation(session: &Session, att: &Attestation) -> Result<bool, AttestError> {
    let msg = canonical_bytes(session)?;

    // Content address must match the recomputed canonical bytes.
    let digest = to_hex(&Sha256::digest(&msg));
    if !digest.eq_ignore_ascii_case(&att.content_sha256_hex) {
        return Ok(false);
    }

    // Signature must verify against the embedded key.
    let pk_bytes = from_hex(&att.public_key_hex, "public_key_hex")?;
    let pk_arr: [u8; 32] = pk_bytes.try_into().map_err(|_| AttestError::BadKeyOrSig)?;
    let vk = VerifyingKey::from_bytes(&pk_arr).map_err(|_| AttestError::BadKeyOrSig)?;

    let sig_bytes = from_hex(&att.signature_hex, "signature_hex")?;
    let sig_arr: [u8; 64] = sig_bytes.try_into().map_err(|_| AttestError::BadKeyOrSig)?;
    let sig = Signature::from_bytes(&sig_arr);

    Ok(vk.verify(&msg, &sig).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{IsolationTier, Session, TelemetryPresence};

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    fn session() -> Session {
        Session {
            session_id: "s-1".into(),
            sut_id: "model-x v2".into(),
            harness_id: "harness v1".into(),
            supervisor_id: "supervisor v1".into(),
            tier: IsolationTier::Tier3,
            telemetry: TelemetryPresence {
                logical: true,
                resource: true,
                physical: true,
            },
            replayable: true,
            probing_detected: false,
            tampering_detected: false,
            arl_claim_sha256_hex: "ab".repeat(32),
            measured_unix: 1_900_000_000,
            valid_through_unix: 1_931_536_000,
        }
    }

    #[test]
    fn jcs_sorts_keys_and_is_compact() {
        let v: Value = serde_json::json!({ "b": 2, "a": 1, "c": [3, 1, 2] });
        assert_eq!(jcs_canonicalize(&v).unwrap(), r#"{"a":1,"b":2,"c":[3,1,2]}"#);
    }

    #[test]
    fn jcs_rejects_non_integer_floats() {
        let v: Value = serde_json::json!({ "x": 1.5 });
        assert!(matches!(jcs_canonicalize(&v), Err(AttestError::NonIntegerNumber)));
    }

    #[test]
    fn canonical_is_field_order_independent() {
        // Two sessions equal in content produce identical canonical bytes.
        let a = session();
        let b = session();
        assert_eq!(canonical_bytes(&a).unwrap(), canonical_bytes(&b).unwrap());
    }

    #[test]
    fn attest_then_verify_round_trips() {
        let s = session();
        let att = attest_session(&s, &key()).unwrap();
        assert!(verify_attestation(&s, &att).unwrap());
        // The attestation self-describes the signer.
        assert!(att.signer_is(&to_hex(key().verifying_key().as_bytes())));
    }

    #[test]
    fn tampering_with_the_session_breaks_verification() {
        let s = session();
        let att = attest_session(&s, &key()).unwrap();
        // Flip a field after signing.
        let mut tampered = s.clone();
        tampered.tampering_detected = true;
        assert!(!verify_attestation(&tampered, &att).unwrap());
    }

    #[test]
    fn forged_signature_does_not_verify() {
        let s = session();
        let mut att = attest_session(&s, &key()).unwrap();
        // Corrupt the signature.
        att.signature_hex.replace_range(0..2, "00");
        // Either it fails to verify, or the hex was structurally bad.
        let ok = verify_attestation(&s, &att).unwrap_or(false);
        assert!(!ok);
    }

    #[test]
    fn signer_is_distinguishes_keys() {
        let s = session();
        let att = attest_session(&s, &key()).unwrap();
        let other = to_hex(SigningKey::from_bytes(&[9u8; 32]).verifying_key().as_bytes());
        assert!(!att.signer_is(&other));
    }

    #[test]
    fn hex_round_trips() {
        let bytes = [0u8, 1, 15, 16, 255, 128];
        assert_eq!(from_hex(&to_hex(&bytes), "t").unwrap(), bytes);
        assert!(from_hex("zz", "t").is_err());
        assert!(from_hex("abc", "t").is_err()); // odd length
    }
}
