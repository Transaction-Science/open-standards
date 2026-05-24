//! EIP-7702 — Set EOA Account Code.
//!
//! EIP-7702 (mainnet from the Pectra upgrade) lets an EOA include
//! an `AuthorizationList` in a type-`0x04` transaction whose effect
//! is to attach contract bytecode to the EOA *for the duration of
//! that transaction*. The result: an EOA can act as a smart account
//! without giving up its address.
//!
//! Each authorization in the list is the tuple:
//! ```text
//! (chain_id, address, nonce, y_parity, r, s)
//! ```
//! over the message `keccak256(MAGIC || rlp([chain_id, address, nonce]))`
//! where `MAGIC = 0x05`. This module models the unsigned and signed
//! authorization shape. **It does not perform signing** — operators
//! sign with their own key material.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// EIP-7702 magic byte. The byte prepended to the RLP-encoded
/// `(chain_id, address, nonce)` tuple before keccak256-hashing for
/// the authorization signature.
pub const EIP7702_MAGIC: u8 = 0x05;

/// One authorization: an EOA delegates its code slot to `address`
/// for the duration of one transaction.
///
/// `chain_id == 0` is a special wildcard authorization — replayable
/// on any chain. Operators should require `chain_id != 0` for
/// production flows unless they're explicitly minting cross-chain
/// session keys.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Eip7702Authorization {
    /// EIP-155 chain id, or 0 for "any chain".
    pub chain_id: u64,
    /// Contract address whose code the EOA is adopting. Lowercase
    /// hex with `0x` prefix.
    pub address: String,
    /// EOA's nonce at signing time. Anti-replay.
    pub nonce: u64,
    /// Recovery parity (0 or 1).
    pub y_parity: u8,
    /// Signature `r` value, 32 bytes.
    pub r: [u8; 32],
    /// Signature `s` value, 32 bytes.
    pub s: [u8; 32],
}

impl Eip7702Authorization {
    /// Construct an *unsigned* authorization (`r`, `s`, `y_parity`
    /// all zero). The caller hashes [`Self::signing_preimage`] and
    /// signs externally, then fills in the signature fields.
    #[must_use]
    pub fn unsigned(chain_id: u64, address: impl Into<String>, nonce: u64) -> Self {
        Self {
            chain_id,
            address: address.into(),
            nonce,
            y_parity: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        }
    }

    /// True iff the authorization has a non-zero signature filled
    /// in. Operators check this before serializing the transaction.
    #[must_use]
    pub fn is_signed(&self) -> bool {
        self.r != [0u8; 32] || self.s != [0u8; 32]
    }

    /// Bytes to hash with keccak256 to produce the signing digest.
    ///
    /// Returns `MAGIC || rlp([chain_id, address, nonce])` where
    /// `address` is the 20 raw bytes (NOT the hex string).
    ///
    /// We hand-roll a small RLP-list encoder rather than pull in
    /// `rlp` as a hard dep — the encoding here is fixed-shape (3
    /// items, each a small integer or a 20-byte string).
    ///
    /// # Errors
    /// Returns [`Error::Constraint`] when `address` isn't valid
    /// 20-byte hex.
    pub fn signing_preimage(&self) -> Result<Vec<u8>> {
        let addr = parse_evm_addr_20(&self.address)?;
        let chain_rlp = rlp_encode_uint(self.chain_id.into());
        let addr_rlp = rlp_encode_bytes(&addr);
        let nonce_rlp = rlp_encode_uint(self.nonce.into());

        let mut payload = Vec::new();
        payload.extend_from_slice(&chain_rlp);
        payload.extend_from_slice(&addr_rlp);
        payload.extend_from_slice(&nonce_rlp);

        let mut out = Vec::with_capacity(1 + payload.len() + 4);
        out.push(EIP7702_MAGIC);
        out.extend_from_slice(&rlp_list_header(payload.len()));
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Validate signature shape (y_parity in {0, 1}, r and s
    /// non-zero, s in the low half of the secp256k1 order).
    ///
    /// # Errors
    /// Returns [`Error::Integrity`] on malformed signature.
    pub fn validate_signature_shape(&self) -> Result<()> {
        if self.y_parity > 1 {
            return Err(Error::Integrity(format!(
                "y_parity must be 0 or 1, got {}",
                self.y_parity
            )));
        }
        if self.r == [0u8; 32] || self.s == [0u8; 32] {
            return Err(Error::Integrity("zero signature scalar".into()));
        }
        // secp256k1 group order N divided by 2:
        // 0x7FFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
        const HALF_N: [u8; 32] = [
            0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xfe, 0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c,
            0xd0, 0x36, 0x41, 0x41,
        ];
        if self.s > HALF_N {
            return Err(Error::Integrity("s must be in low half of N".into()));
        }
        Ok(())
    }
}

/// List of authorizations carried in a type-`0x04` transaction.
/// Aliased so the surface reads `AuthorizationList` (matching the
/// EIP) without re-exporting `Vec`.
pub type AuthorizationList = Vec<Eip7702Authorization>;

fn parse_evm_addr_20(s: &str) -> Result<[u8; 20]> {
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
    let mut out = [0u8; 20];
    out.copy_from_slice(&raw);
    Ok(out)
}

/// Encode an integer as RLP. Leading-zero stripping per the RLP
/// canonical form rules.
fn rlp_encode_uint(n: u128) -> Vec<u8> {
    if n == 0 {
        return vec![0x80];
    }
    let bytes = n.to_be_bytes();
    let mut start = 0;
    while start < bytes.len() && bytes[start] == 0 {
        start += 1;
    }
    rlp_encode_bytes(&bytes[start..])
}

fn rlp_encode_bytes(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match b.len() {
        1 if b[0] < 0x80 => out.push(b[0]),
        0..=55 => {
            // single-byte length prefix
            #[allow(clippy::cast_possible_truncation)]
            out.push(0x80 + b.len() as u8);
            out.extend_from_slice(b);
        }
        _ => {
            let len_bytes = encode_be_length(b.len());
            #[allow(clippy::cast_possible_truncation)]
            out.push(0xb7 + len_bytes.len() as u8);
            out.extend_from_slice(&len_bytes);
            out.extend_from_slice(b);
        }
    }
    out
}

fn rlp_list_header(payload_len: usize) -> Vec<u8> {
    if payload_len <= 55 {
        #[allow(clippy::cast_possible_truncation)]
        let header = 0xc0_u8 + (payload_len as u8);
        vec![header]
    } else {
        let len_bytes = encode_be_length(payload_len);
        #[allow(clippy::cast_possible_truncation)]
        let header = 0xf7_u8 + (len_bytes.len() as u8);
        let mut out = vec![header];
        out.extend_from_slice(&len_bytes);
        out
    }
}

fn encode_be_length(n: usize) -> Vec<u8> {
    let bytes = (n as u64).to_be_bytes();
    let mut start = 0;
    while start < bytes.len() && bytes[start] == 0 {
        start += 1;
    }
    bytes[start..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_is_not_signed() {
        let a = Eip7702Authorization::unsigned(
            1,
            "0x1111111111111111111111111111111111111111",
            7,
        );
        assert!(!a.is_signed());
    }

    #[test]
    fn signing_preimage_begins_with_magic() {
        let a = Eip7702Authorization::unsigned(
            1,
            "0x1111111111111111111111111111111111111111",
            7,
        );
        let pre = a.signing_preimage().unwrap();
        assert_eq!(pre[0], EIP7702_MAGIC);
    }

    #[test]
    fn signing_preimage_rejects_bad_address() {
        let a = Eip7702Authorization::unsigned(1, "0xnope", 7);
        let err = a.signing_preimage().unwrap_err();
        assert!(matches!(err, Error::Constraint { .. }));
    }

    #[test]
    fn signature_shape_rejects_zero() {
        let a = Eip7702Authorization::unsigned(
            1,
            "0x1111111111111111111111111111111111111111",
            7,
        );
        let err = a.validate_signature_shape().unwrap_err();
        assert!(matches!(err, Error::Integrity(_)));
    }

    #[test]
    fn signature_shape_accepts_low_s() {
        let mut a = Eip7702Authorization::unsigned(
            1,
            "0x1111111111111111111111111111111111111111",
            7,
        );
        a.r = [0xaa; 32];
        a.s = [0x01; 32]; // well below N/2
        a.y_parity = 1;
        a.validate_signature_shape().unwrap();
    }

    #[test]
    fn signature_shape_rejects_high_s() {
        let mut a = Eip7702Authorization::unsigned(
            1,
            "0x1111111111111111111111111111111111111111",
            7,
        );
        a.r = [0xaa; 32];
        a.s = [0xff; 32]; // > N/2
        a.y_parity = 0;
        let err = a.validate_signature_shape().unwrap_err();
        assert!(matches!(err, Error::Integrity(_)));
    }

    #[test]
    fn rlp_encode_uint_zero_is_empty_string() {
        // RLP encodes 0 as 0x80 (empty byte string).
        assert_eq!(rlp_encode_uint(0), vec![0x80]);
    }

    #[test]
    fn rlp_encode_uint_strips_leading_zeros() {
        // 0x01 should encode as a single byte 0x01.
        assert_eq!(rlp_encode_uint(1), vec![0x01]);
        // 0x80 needs a length prefix.
        assert_eq!(rlp_encode_uint(0x80), vec![0x81, 0x80]);
    }
}
