//! ISO 8583 primary + secondary bitmaps.
//!
//! Each bitmap is 64 bits (8 raw bytes / 16 hex chars). Bit 1 (the
//! most-significant bit of the first byte) of the primary bitmap is a
//! **continuation flag**: when set, an 8-byte secondary bitmap follows,
//! covering data elements 65..=128. Together they address fields 1..=128.
//!
//! ISO 8583 numbers bits starting at 1, left-to-right within each byte
//! (MSB is bit 1). DE 1 itself is the "secondary bitmap present" flag;
//! decoders MUST NOT treat DE 1 as a separately decodable data element.
//!
//! ## On-the-wire forms
//!
//! - **Binary**: 8 raw bytes for the primary (and 8 more for secondary
//!   when present). Used by Visa Base I, Mastercard MDS, modern Discover,
//!   modern JCB.
//! - **Hex ASCII**: 16 ASCII hex characters per bitmap. Used by some
//!   legacy installations (notably older Amex GNS) and easier to log.
//!   The [`Bitmaps::decode_binary`] / [`Bitmaps::decode_hex`] entry
//!   points pick which.
//!
//! We model the bitmaps as `u128` (primary in the high 64 bits,
//! secondary in the low 64 bits). Bit-1-of-the-primary maps to bit 127
//! of the `u128`; bit-128-of-the-secondary maps to bit 0.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A pair of (primary, secondary) bitmaps representing which data
/// elements are present in a message.
///
/// `has(n)` is true iff data element `n` (1..=128) is present.
///
/// Internally the storage is one `u128`. Bit position for DE `n` is
/// `128 - n` (so DE 1 sits at the top, DE 128 at the bottom).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Bitmaps(u128);

impl Bitmaps {
    /// Empty bitmaps (no DEs present, secondary off).
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Construct directly from a `u128` representation (DE 1 → bit 127,
    /// DE 128 → bit 0).
    #[must_use]
    pub const fn from_bits(bits: u128) -> Self {
        Self(bits)
    }

    /// Raw bit representation. DE 1 is bit 127, DE 128 is bit 0.
    #[must_use]
    pub const fn bits(self) -> u128 {
        self.0
    }

    /// True iff DE `n` (1..=128) is marked present.
    ///
    /// DEs outside `1..=128` return `false` (no panic).
    #[must_use]
    pub const fn has(self, de: u8) -> bool {
        if de < 1 || de > 128 {
            return false;
        }
        let pos = 128 - de;
        (self.0 >> pos) & 1 == 1
    }

    /// Mark DE `n` present. No-op for `n` outside `1..=128`.
    pub const fn set(&mut self, de: u8) {
        if de < 1 || de > 128 {
            return;
        }
        let pos = 128 - de;
        self.0 |= 1_u128 << pos;
    }

    /// Mark DE `n` absent.
    pub const fn clear(&mut self, de: u8) {
        if de < 1 || de > 128 {
            return;
        }
        let pos = 128 - de;
        self.0 &= !(1_u128 << pos);
    }

    /// True iff any of DE 65..=128 are set (i.e. a secondary bitmap is
    /// required on the wire).
    #[must_use]
    pub const fn secondary_needed(self) -> bool {
        (self.0 & ((1_u128 << 64) - 1)) != 0
    }

    /// Encode to wire bytes (8 or 16 bytes depending on
    /// [`Self::secondary_needed`]).
    ///
    /// Sets DE 1 (bit 127) iff the secondary is present.
    #[must_use]
    pub fn encode_binary(self) -> Vec<u8> {
        let secondary = self.secondary_needed();
        let mut bits = self.0;
        if secondary {
            // Set DE 1 = bit 127.
            bits |= 1_u128 << 127;
        } else {
            bits &= !(1_u128 << 127);
        }
        let primary = (bits >> 64) as u64;
        if secondary {
            let secondary_bits = bits as u64;
            let mut out = Vec::with_capacity(16);
            out.extend_from_slice(&primary.to_be_bytes());
            out.extend_from_slice(&secondary_bits.to_be_bytes());
            out
        } else {
            primary.to_be_bytes().to_vec()
        }
    }

    /// Encode as 16 (or 32) ASCII hex characters, uppercase.
    #[must_use]
    pub fn encode_hex(self) -> String {
        let bin = self.encode_binary();
        let mut s = String::with_capacity(bin.len() * 2);
        for b in bin {
            // Lookup table; no allocations beyond the String we're building.
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            s.push(char::from(HEX[(b >> 4) as usize]));
            s.push(char::from(HEX[(b & 0x0F) as usize]));
        }
        s
    }

    /// Decode a binary bitmap starting at `offset`. Returns the
    /// `(bitmaps, new_offset)`. Consumes 8 bytes; if bit 1 of the
    /// primary is set, consumes 16 bytes total.
    ///
    /// # Errors
    /// [`Error::UnexpectedEof`] if the input is shorter than required.
    pub fn decode_binary(input: &[u8], offset: usize) -> Result<(Self, usize)> {
        if offset + 8 > input.len() {
            return Err(Error::UnexpectedEof {
                offset,
                needed: 8,
            });
        }
        let mut primary_bytes = [0_u8; 8];
        primary_bytes.copy_from_slice(&input[offset..offset + 8]);
        let primary = u64::from_be_bytes(primary_bytes);
        let secondary_present = primary & (1_u64 << 63) != 0; // bit 1 = MSB
        let mut bits = u128::from(primary) << 64;
        let mut new_offset = offset + 8;
        if secondary_present {
            if new_offset + 8 > input.len() {
                return Err(Error::UnexpectedEof {
                    offset: new_offset,
                    needed: 8,
                });
            }
            let mut sec_bytes = [0_u8; 8];
            sec_bytes.copy_from_slice(&input[new_offset..new_offset + 8]);
            let secondary = u64::from_be_bytes(sec_bytes);
            bits |= u128::from(secondary);
            new_offset += 8;
        }
        Ok((Self(bits), new_offset))
    }

    /// Decode a hex-ASCII bitmap (16 or 32 hex characters) starting at
    /// `offset` in `input`.
    ///
    /// # Errors
    /// [`Error::UnexpectedEof`] or [`Error::InvalidAscii`] for malformed input.
    pub fn decode_hex(input: &[u8], offset: usize) -> Result<(Self, usize)> {
        if offset + 16 > input.len() {
            return Err(Error::UnexpectedEof {
                offset,
                needed: 16,
            });
        }
        let primary_bytes = hex_to_bytes(&input[offset..offset + 16])?;
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(&primary_bytes);
        let primary = u64::from_be_bytes(buf);
        let mut bits = u128::from(primary) << 64;
        let secondary_present = primary & (1_u64 << 63) != 0;
        let mut new_offset = offset + 16;
        if secondary_present {
            if new_offset + 16 > input.len() {
                return Err(Error::UnexpectedEof {
                    offset: new_offset,
                    needed: 16,
                });
            }
            let sec_bytes = hex_to_bytes(&input[new_offset..new_offset + 16])?;
            let mut sbuf = [0_u8; 8];
            sbuf.copy_from_slice(&sec_bytes);
            let secondary = u64::from_be_bytes(sbuf);
            bits |= u128::from(secondary);
            new_offset += 16;
        }
        Ok((Self(bits), new_offset))
    }

    /// Iterate the data-element numbers that are present, in ascending
    /// order. Skips DE 1 because that bit is the secondary-bitmap flag,
    /// not a separately decodable data element.
    pub fn iter_des(self) -> impl Iterator<Item = u8> {
        (2_u8..=128_u8).filter(move |de| self.has(*de))
    }
}

fn hex_to_bytes(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() % 2 != 0 {
        return Err(Error::InvalidVarLength {
            prefix: input.len(),
            max: input.len() + 1,
        });
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    for (i, pair) in input.chunks(2).enumerate() {
        let hi = hex_nibble(pair[0]).ok_or(Error::InvalidAscii {
            byte: pair[0],
            offset: i * 2,
        })?;
        let lo = hex_nibble(pair[1]).ok_or(Error::InvalidAscii {
            byte: pair[1],
            offset: i * 2 + 1,
        })?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bitmap_has_no_des() {
        let b = Bitmaps::new();
        assert!(!b.has(1));
        assert!(!b.has(64));
        assert!(!b.has(128));
        assert!(!b.secondary_needed());
    }

    #[test]
    fn set_and_query_primary_only() {
        let mut b = Bitmaps::new();
        b.set(2);
        b.set(3);
        b.set(4);
        b.set(7);
        b.set(11);
        assert!(b.has(2));
        assert!(b.has(11));
        assert!(!b.has(8));
        assert!(!b.secondary_needed());
        // Encoded as 8 bytes only.
        assert_eq!(b.encode_binary().len(), 8);
    }

    #[test]
    fn setting_secondary_field_activates_secondary_bitmap() {
        let mut b = Bitmaps::new();
        b.set(2);
        b.set(90);
        assert!(b.secondary_needed());
        let bin = b.encode_binary();
        assert_eq!(bin.len(), 16);
        // First byte: bit 1 (MSB) set because secondary present,
        // bit 2 set because DE 2 present -> 0b1100_0000 = 0xC0.
        assert_eq!(bin[0], 0xC0);
    }

    #[test]
    fn round_trip_binary() {
        let mut b = Bitmaps::new();
        for de in [2_u8, 3, 4, 7, 11, 39, 41, 42, 55, 90, 128] {
            b.set(de);
        }
        let bin = b.encode_binary();
        let (decoded, off) = Bitmaps::decode_binary(&bin, 0).unwrap();
        assert_eq!(off, bin.len());
        // The decoded form should equal the original *with* the
        // continuation-flag bit (DE 1) accounted for. Recompute:
        let mut expected = b;
        expected.set(1);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn round_trip_hex() {
        let mut b = Bitmaps::new();
        b.set(2);
        b.set(11);
        b.set(70);
        let hex = b.encode_hex();
        let (decoded, _) = Bitmaps::decode_hex(hex.as_bytes(), 0).unwrap();
        let mut expected = b;
        expected.set(1);
        assert_eq!(decoded, expected);
    }

    #[test]
    fn clear_works() {
        let mut b = Bitmaps::new();
        b.set(7);
        b.set(11);
        b.clear(7);
        assert!(!b.has(7));
        assert!(b.has(11));
    }

    #[test]
    fn iter_des_skips_de1_continuation() {
        let mut b = Bitmaps::new();
        b.set(1);
        b.set(11);
        b.set(90);
        let des: Vec<u8> = b.iter_des().collect();
        assert_eq!(des, vec![11, 90]);
    }
}
