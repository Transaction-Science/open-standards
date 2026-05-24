//! Status lists referenced by OID4VCI-issued credentials.
//!
//! Two ecosystems coexist:
//!
//! * **Bitstring Status List 2021** — W3C TR. A status list credential
//!   carries `encodedList`, a GZIP+base64url-encoded bitstring; each
//!   credential's `credentialStatus.statusListIndex` selects a bit.
//!   This crate re-exports [`smart_byte_vc::status_list::check_status`]
//!   for compatibility.
//! * **Token Status List** (IETF `oauth-status-list`) — for JWT / CWT
//!   tokens including SD-JWT VCs. The list is itself a JWT / CWT whose
//!   payload contains `status_list.bits` + `status_list.lst`; the
//!   referenced token's `status.status_list.idx` selects a slot.
//!
//! Both forms compress identically with GZIP; bit ordering is
//! big-endian per the W3C spec and little-endian within a byte per the
//! IETF spec. We expose helpers for both.

use std::io::{Read, Write};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};

use crate::error::OidcError;

pub use smart_byte_vc::status_list::{
    BitstringStatusList, StatusPurpose, check_status as check_bitstring_status,
};

/// A `status_list` claim inside a Token Status List JWT/CWT payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenStatusList {
    /// Bits per slot. Permitted values: 1, 2, 4, 8.
    pub bits: u8,
    /// `lst` — GZIP-compressed, base64url-encoded bitstring.
    pub lst: String,
}

/// A `status.status_list` claim attached to a referenced token.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenStatusReference {
    /// Slot index inside the list.
    pub idx: u64,
    /// URI of the status list token.
    pub uri: String,
}

/// In-memory Token Status List slot map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenStatusListBytes {
    /// Bits per slot.
    pub bits: u8,
    /// Raw decompressed status bytes.
    pub bytes: Vec<u8>,
}

impl TokenStatusListBytes {
    /// Build an empty status list with `slot_count` slots of `bits`-bit
    /// width.
    pub fn new(bits: u8, slot_count: usize) -> Result<Self, OidcError> {
        Self::check_bits(bits)?;
        let total_bits = slot_count * bits as usize;
        let byte_len = total_bits.div_ceil(8);
        Ok(Self {
            bits,
            bytes: vec![0u8; byte_len],
        })
    }

    fn check_bits(bits: u8) -> Result<(), OidcError> {
        if !matches!(bits, 1 | 2 | 4 | 8) {
            return Err(OidcError::StatusList(format!(
                "bits must be 1, 2, 4, or 8 (got {bits})"
            )));
        }
        Ok(())
    }

    /// Maximum slot value (`2^bits - 1`).
    pub fn max_value(&self) -> u8 {
        (1u16 << self.bits as u16) as u8 - 1
    }

    /// Number of addressable slots.
    pub fn slot_count(&self) -> usize {
        self.bytes.len() * 8 / self.bits as usize
    }

    /// Set the status value at `idx`.
    pub fn set(&mut self, idx: usize, value: u8) -> Result<(), OidcError> {
        let max = self.max_value();
        if value > max {
            return Err(OidcError::StatusList(format!(
                "value {value} exceeds max {max} for {}-bit slots",
                self.bits
            )));
        }
        if idx >= self.slot_count() {
            return Err(OidcError::StatusList(format!(
                "idx {idx} out of bounds (slots={})",
                self.slot_count()
            )));
        }
        let total_bit = idx * self.bits as usize;
        let byte = total_bit / 8;
        let lo_bit = total_bit % 8;
        // Per IETF token-status-list: LSB-first ordering within a byte.
        let mask = (max as u16) << lo_bit;
        let new = ((value as u16) << lo_bit) & mask;
        let cur = self.bytes[byte] as u16;
        let cleared = cur & !mask;
        self.bytes[byte] = (cleared | new) as u8;
        Ok(())
    }

    /// Read the status value at `idx`.
    pub fn get(&self, idx: usize) -> Result<u8, OidcError> {
        if idx >= self.slot_count() {
            return Err(OidcError::StatusList(format!(
                "idx {idx} out of bounds (slots={})",
                self.slot_count()
            )));
        }
        let total_bit = idx * self.bits as usize;
        let byte = total_bit / 8;
        let lo_bit = total_bit % 8;
        let max = self.max_value();
        let v = (self.bytes[byte] as u16 >> lo_bit) & max as u16;
        Ok(v as u8)
    }

    /// Compress and base64url-encode the list bytes (the `lst` form).
    pub fn to_encoded(&self) -> Result<String, OidcError> {
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&self.bytes)
            .map_err(|e| OidcError::StatusList(e.to_string()))?;
        let gz = enc
            .finish()
            .map_err(|e| OidcError::StatusList(e.to_string()))?;
        Ok(URL_SAFE_NO_PAD.encode(gz))
    }

    /// Decode an `lst` string back into raw bytes.
    pub fn from_encoded(
        lst: &str,
        bits: u8,
        byte_len_hint: Option<usize>,
    ) -> Result<Self, OidcError> {
        Self::check_bits(bits)?;
        let gz = URL_SAFE_NO_PAD
            .decode(lst)
            .map_err(|e| OidcError::StatusList(e.to_string()))?;
        let mut decoder = GzDecoder::new(gz.as_slice());
        let mut out = Vec::new();
        decoder
            .read_to_end(&mut out)
            .map_err(|e| OidcError::StatusList(e.to_string()))?;
        if let Some(expected) = byte_len_hint {
            if out.len() != expected {
                return Err(OidcError::StatusList(format!(
                    "decoded list is {} bytes, expected {expected}",
                    out.len()
                )));
            }
        }
        Ok(Self { bits, bytes: out })
    }
}

impl TokenStatusList {
    /// Convert to in-memory form.
    pub fn decode(
        &self,
        byte_len_hint: Option<usize>,
    ) -> Result<TokenStatusListBytes, OidcError> {
        TokenStatusListBytes::from_encoded(&self.lst, self.bits, byte_len_hint)
    }

    /// Encode from in-memory form.
    pub fn encode(
        list: &TokenStatusListBytes,
    ) -> Result<Self, OidcError> {
        Ok(Self {
            bits: list.bits,
            lst: list.to_encoded()?,
        })
    }
}

/// Common slot semantics for 1-bit lists.
pub const STATUS_VALID: u8 = 0;
/// 1-bit revoked sentinel.
pub const STATUS_INVALID: u8 = 1;

/// Common slot semantics for 2-bit lists (IETF token-status-list).
pub const STATUS_SUSPENDED: u8 = 2;
/// 2-bit application-specific sentinel.
pub const STATUS_APPLICATION_SPECIFIC: u8 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_slot_roundtrip() {
        let mut l = TokenStatusListBytes::new(1, 1024).unwrap();
        l.set(0, 1).unwrap();
        l.set(7, 1).unwrap();
        l.set(513, 1).unwrap();
        let enc = TokenStatusList::encode(&l).unwrap();
        let back = enc.decode(Some(l.bytes.len())).unwrap();
        assert_eq!(back.get(0).unwrap(), 1);
        assert_eq!(back.get(7).unwrap(), 1);
        assert_eq!(back.get(513).unwrap(), 1);
        assert_eq!(back.get(1).unwrap(), 0);
    }

    #[test]
    fn two_bit_slots() {
        let mut l = TokenStatusListBytes::new(2, 16).unwrap();
        l.set(0, STATUS_INVALID).unwrap();
        l.set(1, STATUS_SUSPENDED).unwrap();
        l.set(2, STATUS_APPLICATION_SPECIFIC).unwrap();
        assert_eq!(l.get(0).unwrap(), STATUS_INVALID);
        assert_eq!(l.get(1).unwrap(), STATUS_SUSPENDED);
        assert_eq!(l.get(2).unwrap(), STATUS_APPLICATION_SPECIFIC);
        assert_eq!(l.get(3).unwrap(), STATUS_VALID);
    }

    #[test]
    fn rejects_invalid_bits() {
        assert!(TokenStatusListBytes::new(3, 8).is_err());
        assert!(TokenStatusListBytes::new(16, 8).is_err());
    }

    #[test]
    fn rejects_value_overflow() {
        let mut l = TokenStatusListBytes::new(1, 8).unwrap();
        assert!(l.set(0, 2).is_err());
    }

    #[test]
    fn rejects_idx_out_of_bounds() {
        let l = TokenStatusListBytes::new(1, 8).unwrap();
        assert!(l.get(9).is_err());
    }

    #[test]
    fn bitstring_compat_check() {
        // Spot-check that we still re-export the W3C helper.
        let bs = BitstringStatusList::new(16);
        let encoded = bs.to_encoded().unwrap();
        assert!(!check_bitstring_status(&encoded, 16, 0).unwrap());
    }
}
