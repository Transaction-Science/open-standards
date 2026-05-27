//! BLAKE3-hash deterministic-output verifier.
//!
//! Confirms that a deterministic tier (L0 cache hit, L1 lawful
//! primitive) produced byte-exact the answer the spec says it
//! should. The natural companion to JouleClaw's conformance vectors:
//! "input `X` at tier L1 MUST hash to `H`". A single hash check
//! collapses all of that to one comparison.
//!
//! Not useful against L3/L4 output — stochastic generators do not
//! produce byte-exact identical answers. For those tiers reach for
//! [`crate::RegexVerifier`] or
//! [`crate::JsonSchemaVerifier`] instead.

use crate::error::VerifyError;
use crate::verifier::{OutputVerifier, VerifyResult};

/// Default microjoule cost charged to a BLAKE3 hash verifier touch.
/// Tiny — BLAKE3 hashes a few kilobytes in well under a microsecond
/// — but non-zero so the verifier shows up in the receipt.
pub const DEFAULT_HASH_COST_UJ: u64 = 20;

/// A verifier that passes iff `blake3(output)` matches an expected
/// hex digest.
#[derive(Debug)]
pub struct BlakeHashVerifier {
    /// Expected BLAKE3 digest as lowercase hex (64 chars).
    expected_hex: String,
    /// Verifier name as it appears in the receipt.
    name: String,
    /// Declared microjoule cost.
    cost_uj: u64,
}

impl BlakeHashVerifier {
    /// Build a verifier checking against `expected_hex`. The hex
    /// string must be exactly 64 lowercase hex characters; anything
    /// else returns [`VerifyError::InvalidHash`].
    pub fn new(expected_hex: impl Into<String>) -> Result<Self, VerifyError> {
        let hex = expected_hex.into();
        if hex.len() != 64 {
            return Err(VerifyError::InvalidHash(format!(
                "expected 64 hex chars, got {}",
                hex.len()
            )));
        }
        if !hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
            return Err(VerifyError::InvalidHash(
                "expected lowercase hex digits only".to_string(),
            ));
        }
        Ok(Self {
            expected_hex: hex,
            name: "verify:blake3".to_string(),
            cost_uj: DEFAULT_HASH_COST_UJ,
        })
    }

    /// Override the verifier name (used in receipts). Convention:
    /// prefix with `verify:`.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the declared microjoule cost.
    pub fn with_cost_uj(mut self, cost_uj: u64) -> Self {
        self.cost_uj = cost_uj;
        self
    }
}

impl OutputVerifier for BlakeHashVerifier {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self, output: &[u8]) -> VerifyResult {
        let actual = blake3::hash(output).to_hex().to_string();
        if actual == self.expected_hex {
            VerifyResult::Pass
        } else {
            VerifyResult::fail(format!(
                "blake3 mismatch: expected {}, got {}",
                self.expected_hex, actual
            ))
        }
    }

    fn declared_cost_uj(&self) -> u64 {
        self.cost_uj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_of(b: &[u8]) -> String {
        blake3::hash(b).to_hex().to_string()
    }

    #[test]
    fn matching_hash_passes() {
        let payload = b"the quick brown fox";
        let v = BlakeHashVerifier::new(hex_of(payload))
            .expect("construct")
            .named("verify:hash/canonical");
        assert_eq!(v.verify(payload), VerifyResult::Pass);
        assert_eq!(v.name(), "verify:hash/canonical");
    }

    #[test]
    fn mismatched_hash_fails() {
        let payload = b"hello";
        let other = hex_of(b"goodbye");
        let v = BlakeHashVerifier::new(other).expect("construct");
        match v.verify(payload) {
            VerifyResult::Fail { reason } => assert!(reason.contains("blake3 mismatch")),
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn wrong_length_hex_errors() {
        let err = BlakeHashVerifier::new("deadbeef").unwrap_err();
        match err {
            VerifyError::InvalidHash(msg) => assert!(msg.contains("64 hex chars")),
            _ => panic!("expected InvalidHash"),
        }
    }

    #[test]
    fn non_hex_or_uppercase_errors() {
        // 64-char string with non-hex chars
        let bad = "z".repeat(64);
        let err = BlakeHashVerifier::new(bad).unwrap_err();
        match err {
            VerifyError::InvalidHash(msg) => assert!(msg.contains("lowercase hex")),
            _ => panic!("expected InvalidHash"),
        }
        // 64-char uppercase hex is rejected too — receipts are
        // lowercase by convention.
        let upper = "A".repeat(64);
        let err = BlakeHashVerifier::new(upper).unwrap_err();
        match err {
            VerifyError::InvalidHash(_) => {}
            _ => panic!("expected InvalidHash"),
        }
    }

    #[test]
    fn declared_cost_is_reported() {
        let v = BlakeHashVerifier::new(hex_of(b""))
            .expect("construct")
            .with_cost_uj(7);
        assert_eq!(v.declared_cost_uj(), 7);
    }
}
