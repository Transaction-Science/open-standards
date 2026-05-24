//! EIP-2612 — `permit(...)` on ERC-20s.
//!
//! EIP-2612 lets the token owner sign an EIP-712 typed-data message
//! authorising a `spender` to transfer `value` units until
//! `deadline`. The relayer submits the signature via the token's
//! `permit(owner, spender, value, deadline, v, r, s)` function; no
//! prior on-chain `approve` call is needed.
//!
//! The typed-data hash is
//! `keccak256("\x19\x01" || DOMAIN_SEP || keccak256(Permit struct))`
//! where the Permit type-hash is over:
//! ```text
//! Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)
//! ```
//!
//! This module computes the struct hash. Domain separators are
//! token-specific (each EIP-2612 ERC-20 publishes its own
//! `DOMAIN_SEPARATOR()`), so the caller supplies it.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

use crate::error::{Error, Result};

/// EIP-2612 `Permit` parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Eip2612Permit {
    /// Token holder. Lowercase hex `0x...`.
    pub owner: String,
    /// Spender being authorised. Lowercase hex `0x...`.
    pub spender: String,
    /// Amount in token's smallest unit.
    pub value: u128,
    /// Owner's nonce inside the token contract. Token publishes
    /// `nonces(owner)`.
    pub nonce: u128,
    /// Unix-seconds deadline.
    pub deadline: u64,
}

impl Eip2612Permit {
    /// EIP-712 type-hash for `Permit`.
    #[must_use]
    pub fn type_hash() -> [u8; 32] {
        keccak256(
            b"Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)",
        )
    }

    /// Compute the struct hash (keccak of type-hash || params).
    ///
    /// # Errors
    /// Returns [`Error::Constraint`] when `owner` or `spender` aren't
    /// valid 20-byte hex.
    pub fn struct_hash(&self) -> Result<[u8; 32]> {
        let owner_padded = pad_address(&self.owner)?;
        let spender_padded = pad_address(&self.spender)?;
        let mut buf = Vec::with_capacity(192);
        buf.extend_from_slice(&Self::type_hash());
        buf.extend_from_slice(&owner_padded);
        buf.extend_from_slice(&spender_padded);
        buf.extend_from_slice(&u256_be(self.value));
        buf.extend_from_slice(&u256_be(self.nonce));
        buf.extend_from_slice(&u256_be(u128::from(self.deadline)));
        Ok(keccak256(&buf))
    }
}

/// Compute the final EIP-712 typed-data digest the user signs.
///
/// `domain_separator` is the token's published `DOMAIN_SEPARATOR()`.
/// The output is what an ECDSA signature is computed over.
///
/// # Errors
/// Returns [`Error::Constraint`] on bad addresses inside `permit`.
pub fn eip2612_digest(permit: &Eip2612Permit, domain_separator: &[u8; 32]) -> Result<[u8; 32]> {
    let struct_hash = permit.struct_hash()?;
    let mut buf = Vec::with_capacity(2 + 32 + 32);
    buf.extend_from_slice(&[0x19, 0x01]);
    buf.extend_from_slice(domain_separator);
    buf.extend_from_slice(&struct_hash);
    Ok(keccak256(&buf))
}

fn pad_address(s: &str) -> Result<[u8; 32]> {
    let stripped = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).ok_or(
        Error::Constraint {
            field: "address",
            reason: "missing 0x prefix".into(),
        },
    )?;
    if stripped.len() != 40 {
        return Err(Error::Constraint {
            field: "address",
            reason: format!("must be 40 hex chars, got {}", stripped.len()),
        });
    }
    let raw = hex::decode(stripped).map_err(|e| Error::Constraint {
        field: "address",
        reason: format!("hex decode: {e}"),
    })?;
    let mut out = [0u8; 32];
    out[12..].copy_from_slice(&raw);
    Ok(out)
}

fn u256_be(n: u128) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[16..].copy_from_slice(&n.to_be_bytes());
    out
}

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Eip2612Permit {
        Eip2612Permit {
            owner: "0x1111111111111111111111111111111111111111".into(),
            spender: "0x2222222222222222222222222222222222222222".into(),
            value: 1_000_000,
            nonce: 0,
            deadline: 2_000_000_000,
        }
    }

    #[test]
    fn type_hash_is_stable() {
        let a = Eip2612Permit::type_hash();
        let b = Eip2612Permit::type_hash();
        assert_eq!(a, b);
        assert_ne!(a, [0u8; 32]);
    }

    #[test]
    fn struct_hash_changes_with_value() {
        let mut p = sample();
        let h1 = p.struct_hash().unwrap();
        p.value += 1;
        let h2 = p.struct_hash().unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn digest_changes_with_domain() {
        let p = sample();
        let d1 = eip2612_digest(&p, &[0xaa; 32]).unwrap();
        let d2 = eip2612_digest(&p, &[0xbb; 32]).unwrap();
        assert_ne!(d1, d2);
    }

    #[test]
    fn digest_envelope_prefix() {
        let p = sample();
        // Manually compute and confirm length-shape: digest is
        // keccak of 66 bytes (2 magic + 32 domain + 32 struct).
        // No direct way to read intermediate, but we can confirm
        // determinism.
        let d1 = eip2612_digest(&p, &[0xaa; 32]).unwrap();
        let d2 = eip2612_digest(&p, &[0xaa; 32]).unwrap();
        assert_eq!(d1, d2);
    }

    #[test]
    fn bad_owner_errors() {
        let mut p = sample();
        p.owner = "not-an-address".into();
        let err = p.struct_hash().unwrap_err();
        assert!(matches!(err, Error::Constraint { .. }));
    }
}
