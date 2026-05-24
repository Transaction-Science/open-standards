//! SHA-256 hashlock condition + fulfillment.
//!
//! ILPv4 uses a "hashlock" to bind a `Prepare` to its `Fulfill`: the
//! preparer commits to a 32-byte `condition`, the receiver later
//! reveals a 32-byte `fulfillment` such that `SHA-256(fulfillment) ==
//! condition`. Every connector on the path validates that equality
//! before crediting the next hop.

use crate::error::{IlpError, Result};
use sha2::{Digest, Sha256};

/// A 32-byte SHA-256 condition, the lock side of the hashlock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Condition(pub [u8; 32]);

/// A 32-byte fulfillment, the preimage side of the hashlock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fulfillment(pub [u8; 32]);

impl Condition {
    /// Wrap a 32-byte array as a condition without modification.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the underlying bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Constant-time equality check.
    pub fn ct_eq(&self, other: &Self) -> bool {
        let mut diff = 0u8;
        for i in 0..32 {
            diff |= self.0[i] ^ other.0[i];
        }
        diff == 0
    }
}

impl Fulfillment {
    /// Wrap a 32-byte array as a fulfillment without modification.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the underlying bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Derive the matching SHA-256 condition for this fulfillment.
    pub fn condition(&self) -> Condition {
        let mut hasher = Sha256::new();
        hasher.update(self.0);
        let out = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&out);
        Condition(bytes)
    }

    /// Check that this fulfillment opens the supplied condition.
    pub fn verify(&self, condition: &Condition) -> Result<()> {
        if self.condition().ct_eq(condition) {
            Ok(())
        } else {
            Err(IlpError::ConditionMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let f = Fulfillment::new([7u8; 32]);
        let c = f.condition();
        assert!(f.verify(&c).is_ok());
    }

    #[test]
    fn mismatch_rejected() {
        let f = Fulfillment::new([1u8; 32]);
        let bogus = Condition::new([2u8; 32]);
        assert!(f.verify(&bogus).is_err());
    }
}
