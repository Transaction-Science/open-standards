//! Method-specific resolvers.
//!
//! Each submodule implements the [`crate::resolver::Resolver`] trait for
//! one DID method. The [`crate::resolver::UniversalResolver`] wires them
//! together by method name.

pub mod jwk;
pub mod key;
pub mod peer;
pub mod web;

#[cfg(feature = "ion")]
pub mod ion;

/// Multicodec varint prefixes used by `did:key` and Multikey.
pub(crate) mod multicodec {
    /// Ed25519 public key.
    pub const ED25519_PUB: u64 = 0xed;
    /// secp256k1 (k256) public key.
    pub const SECP256K1_PUB: u64 = 0xe7;
    /// P-256 (NIST secp256r1) public key.
    pub const P256_PUB: u64 = 0x1200;
    /// BLS12-381 G2 public key.
    pub const BLS12_381_G2_PUB: u64 = 0xeb;

    /// Encode an unsigned varint (LEB128) — used for multicodec prefixes.
    pub fn varint_encode(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push(((value & 0x7f) as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    /// Decode an unsigned varint. Returns `(value, bytes_consumed)`.
    pub fn varint_decode(bytes: &[u8]) -> Option<(u64, usize)> {
        let mut value: u64 = 0;
        let mut shift: u32 = 0;
        for (i, &b) in bytes.iter().enumerate() {
            // Cap at 9 bytes (u64 max in varint).
            if i >= 9 {
                return None;
            }
            value |= ((b & 0x7f) as u64) << shift;
            if (b & 0x80) == 0 {
                return Some((value, i + 1));
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
        None
    }
}
