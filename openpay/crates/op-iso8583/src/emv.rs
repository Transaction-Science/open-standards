//! EMV ICC data carriage inside ISO 8583 DE 55.
//!
//! DE 55 is an LLLVAR binary field that holds a BER-TLV bundle of EMV
//! tags — the chip-card's response to the terminal's authorization
//! prompt. Typical tags carried:
//!
//! - `9F26` (8 bytes) — Application Cryptogram (AC).
//! - `9F27` (1 byte)  — Cryptogram Information Data (CID; ARQC/AAC/TC).
//! - `9F10` (var)     — Issuer Application Data.
//! - `9F37` (4 bytes) — Unpredictable Number.
//! - `9F36` (2 bytes) — Application Transaction Counter (ATC).
//! - `95`   (5 bytes) — Terminal Verification Results (TVR).
//! - `9A`   (3 bytes) — Transaction Date.
//! - `9C`   (1 byte)  — Transaction Type.
//! - `5F2A` (2 bytes) — Transaction Currency Code.
//! - `82`   (2 bytes) — Application Interchange Profile.
//! - `9F1A` (2 bytes) — Terminal Country Code.
//!
//! This module provides a minimal BER-TLV walker scoped to the subset
//! ISO 8583 DE 55 actually uses (single-byte and two-byte tags;
//! short-form lengths up to 127, long-form 1-byte length 128..=255).
//! It is intentionally narrower than `op-emv`: we don't take a
//! compile-time dependency on that crate (per the brief), but we
//! cover the cases DE 55 in production ISO 8583 emits.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// One EMV tag (1- or 2-byte) decoded from DE 55.
///
/// Stored as a `u16` for ergonomic match-arms: a 1-byte tag is the
/// low byte (e.g. `0x95` for TVR); a 2-byte tag has the leading byte
/// in the high octet (e.g. `0x9F26` for AC).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EmvTag(pub u16);

impl EmvTag {
    /// Application Cryptogram.
    pub const AC: Self = Self(0x9F26);
    /// Cryptogram Information Data.
    pub const CID: Self = Self(0x9F27);
    /// Issuer Application Data.
    pub const IAD: Self = Self(0x9F10);
    /// Unpredictable Number.
    pub const UN: Self = Self(0x9F37);
    /// Application Transaction Counter.
    pub const ATC: Self = Self(0x9F36);
    /// Terminal Verification Results.
    pub const TVR: Self = Self(0x95);
    /// Transaction Date.
    pub const TX_DATE: Self = Self(0x9A);
    /// Transaction Type.
    pub const TX_TYPE: Self = Self(0x9C);
    /// Transaction Currency Code.
    pub const TX_CURRENCY: Self = Self(0x5F2A);
    /// Application Interchange Profile.
    pub const AIP: Self = Self(0x82);
    /// Terminal Country Code.
    pub const TERMINAL_COUNTRY: Self = Self(0x9F1A);
}

/// One parsed TLV: tag + owned value bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmvTlv {
    /// Decoded tag.
    pub tag: EmvTag,
    /// Owned value bytes.
    pub value: Vec<u8>,
}

/// Walk a DE 55 byte string and decode every TLV at the top level.
///
/// We do NOT recurse into constructed TLVs; production DE 55 payloads
/// are flat sequences of primitive tags. If a constructed tag is
/// encountered, we store its raw value bytes verbatim and let the
/// caller decode further if needed.
///
/// # Errors
/// [`Error::InvalidEmvTlv`] for malformed length encoding, truncated
/// values, or invalid tag continuation bytes.
pub fn parse_de55(input: &[u8]) -> Result<Vec<EmvTlv>> {
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < input.len() {
        // Skip leading padding zeros (some terminals do this).
        if input[offset] == 0x00 {
            offset += 1;
            continue;
        }
        // ---- Tag ----
        let first = input[offset];
        offset += 1;
        let tag = if (first & 0x1F) == 0x1F {
            // Multi-byte tag — continuation while bit 8 of the next byte is set.
            if offset >= input.len() {
                return Err(Error::InvalidEmvTlv {
                    offset,
                    reason: "tag continuation byte missing",
                });
            }
            let second = input[offset];
            offset += 1;
            if (second & 0x80) != 0 {
                // We only support 2-byte tags (the EMV catalog used by
                // DE 55 fits inside that space). Three-byte tags would
                // require another continuation byte; reject.
                return Err(Error::InvalidEmvTlv {
                    offset,
                    reason: "3+ byte tags not supported in DE 55",
                });
            }
            EmvTag((u16::from(first) << 8) | u16::from(second))
        } else {
            EmvTag(u16::from(first))
        };
        // ---- Length ----
        if offset >= input.len() {
            return Err(Error::InvalidEmvTlv {
                offset,
                reason: "length byte missing",
            });
        }
        let len_byte = input[offset];
        offset += 1;
        let value_len = if (len_byte & 0x80) == 0 {
            usize::from(len_byte)
        } else {
            // Long form: 0x8N followed by N length bytes (big-endian).
            let n = usize::from(len_byte & 0x7F);
            if n == 0 {
                return Err(Error::InvalidEmvTlv {
                    offset,
                    reason: "indefinite-length form not allowed in EMV",
                });
            }
            if offset + n > input.len() {
                return Err(Error::InvalidEmvTlv {
                    offset,
                    reason: "long-form length bytes truncated",
                });
            }
            let mut acc: usize = 0;
            for byte in &input[offset..offset + n] {
                acc = (acc << 8) | usize::from(*byte);
            }
            offset += n;
            acc
        };
        // ---- Value ----
        if offset + value_len > input.len() {
            return Err(Error::InvalidEmvTlv {
                offset,
                reason: "value bytes truncated",
            });
        }
        let value = input[offset..offset + value_len].to_vec();
        offset += value_len;
        out.push(EmvTlv { tag, value });
    }
    Ok(out)
}

/// Encode a sequence of EMV TLVs into a DE 55 byte string (no LLLVAR
/// prefix — that is added by the field codec when this is set as
/// `FieldValue::Bytes` on DE 55).
///
/// # Errors
/// [`Error::InvalidEmvTlv`] if a value exceeds the long-form length we
/// can encode (we cap at 4 length bytes, matching the EMV practical
/// limit of values < 2^32 bytes).
pub fn encode_de55(tlvs: &[EmvTlv]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for tlv in tlvs {
        // Tag.
        let t = tlv.tag.0;
        if t > 0xFF {
            out.push((t >> 8) as u8);
            out.push(t as u8);
        } else {
            out.push(t as u8);
        }
        // Length.
        let n = tlv.value.len();
        if n < 0x80 {
            out.push(n as u8);
        } else if n < 0x100 {
            out.push(0x81);
            out.push(n as u8);
        } else if n < 0x1_0000 {
            out.push(0x82);
            out.push((n >> 8) as u8);
            out.push(n as u8);
        } else {
            return Err(Error::InvalidEmvTlv {
                offset: out.len(),
                reason: "value too large for 2-byte length encoding",
            });
        }
        out.extend_from_slice(&tlv.value);
    }
    Ok(out)
}

/// Build a tag→value map from a slice of TLVs. If the same tag appears
/// twice (rare in DE 55 but legal in BER-TLV), the last occurrence wins.
#[must_use]
pub fn tlv_map(tlvs: &[EmvTlv]) -> BTreeMap<EmvTag, Vec<u8>> {
    let mut m = BTreeMap::new();
    for tlv in tlvs {
        m.insert(tlv.tag, tlv.value.clone());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_byte_tag_short_length() {
        // Tag 95 (TVR), length 5, value 00..04
        let input = [0x95_u8, 0x05, 0x00, 0x01, 0x02, 0x03, 0x04];
        let tlvs = parse_de55(&input).unwrap();
        assert_eq!(tlvs.len(), 1);
        assert_eq!(tlvs[0].tag, EmvTag::TVR);
        assert_eq!(tlvs[0].value, vec![0x00, 0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn two_byte_tag_application_cryptogram() {
        let input = [
            0x9F, 0x26, 0x08, // 9F26 length 8
            1, 2, 3, 4, 5, 6, 7, 8,
        ];
        let tlvs = parse_de55(&input).unwrap();
        assert_eq!(tlvs.len(), 1);
        assert_eq!(tlvs[0].tag, EmvTag::AC);
        assert_eq!(tlvs[0].value.len(), 8);
    }

    #[test]
    fn multi_tlv_real_world_layout() {
        let mut input = Vec::new();
        // 9F26 8 bytes AC
        input.extend_from_slice(&[0x9F, 0x26, 0x08]);
        input.extend_from_slice(&[0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF]);
        // 9F27 1 byte CID
        input.extend_from_slice(&[0x9F, 0x27, 0x01, 0x80]);
        // 95 5 bytes TVR
        input.extend_from_slice(&[0x95, 0x05, 0, 0, 0, 0, 0]);
        // 9A 3 bytes date
        input.extend_from_slice(&[0x9A, 0x03, 0x26, 0x05, 0x23]);
        let tlvs = parse_de55(&input).unwrap();
        assert_eq!(tlvs.len(), 4);
        let m = tlv_map(&tlvs);
        assert_eq!(m[&EmvTag::AC].len(), 8);
        assert_eq!(m[&EmvTag::CID][0], 0x80);
        assert_eq!(m[&EmvTag::TX_DATE], vec![0x26, 0x05, 0x23]);
    }

    #[test]
    fn long_form_length_one_byte() {
        // Tag 70 (template) length 130 (long form 0x81 0x82)
        let mut input = vec![0x70_u8, 0x81, 130];
        input.extend(vec![0xAA_u8; 130]);
        let tlvs = parse_de55(&input).unwrap();
        assert_eq!(tlvs.len(), 1);
        assert_eq!(tlvs[0].value.len(), 130);
    }

    #[test]
    fn truncated_value_rejected() {
        let input = [0x95_u8, 0x05, 0x00]; // says 5 bytes, only 1 byte present
        let err = parse_de55(&input).unwrap_err();
        assert!(matches!(err, Error::InvalidEmvTlv { .. }));
    }

    #[test]
    fn encode_round_trip() {
        let tlvs = vec![
            EmvTlv {
                tag: EmvTag::AC,
                value: vec![1, 2, 3, 4, 5, 6, 7, 8],
            },
            EmvTlv {
                tag: EmvTag::TVR,
                value: vec![0; 5],
            },
            EmvTlv {
                tag: EmvTag::TX_DATE,
                value: vec![0x26, 0x05, 0x23],
            },
        ];
        let encoded = encode_de55(&tlvs).unwrap();
        let decoded = parse_de55(&encoded).unwrap();
        assert_eq!(decoded, tlvs);
    }

    #[test]
    fn padding_zeros_skipped() {
        // Leading zero pad before 95 05 00 00 00 00 00
        let input = [0x00_u8, 0x00, 0x95, 0x05, 0, 0, 0, 0, 0];
        let tlvs = parse_de55(&input).unwrap();
        assert_eq!(tlvs.len(), 1);
        assert_eq!(tlvs[0].tag, EmvTag::TVR);
    }
}
