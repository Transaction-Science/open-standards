//! BER-TLV tags.
//!
//! A tag is 1–4 bytes (EMV cap). The first byte carries:
//! - **Class** in bits 8–7 (universal, application, context-specific, private)
//! - **P/C bit** in bit 6 (0 = primitive, 1 = constructed)
//! - **Tag number** in bits 5–1 (0–30, or `0b11111` to signal multi-byte)
//!
//! When bits 5–1 of the first byte are all set (`0x1F`), the tag
//! continues. Each continuation byte uses bit 8 as a "more follows"
//! flag and bits 7–1 as 7 bits of tag number.
//!
//! We represent the whole tag as a `u32` because it fits and lets us
//! cheaply compare known tag constants (`0x9F02`, `0x5F2D`, etc.).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// BER tag class (bits 8–7 of the first byte).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TagClass {
    /// `00` — Universal. Reserved for ASN.1 primitive types.
    Universal,
    /// `01` — Application. Most EMV tags below `0x60` use this class.
    Application,
    /// `10` — Context-specific. Many EMV `0x9F..` tags fall here.
    ContextSpecific,
    /// `11` — Private. EMVCo-issued proprietary tags (`0xDF..`, `0xE1..`).
    Private,
}

impl TagClass {
    /// Extract the class from the first byte of a tag.
    #[must_use]
    pub const fn from_first_byte(b: u8) -> Self {
        match b >> 6 {
            0b00 => Self::Universal,
            0b01 => Self::Application,
            0b10 => Self::ContextSpecific,
            _ => Self::Private,
        }
    }
}

/// A complete BER-TLV tag (up to 4 bytes packed into a `u32`).
///
/// `Tag(0x9F02)` is `9F 02`. `Tag(0x6F)` is `6F`. The most-significant
/// bytes are the *first* bytes on the wire — this lets us write
/// constants like `Tag(0x9F02)` that read naturally.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tag(pub u32);

impl Tag {
    // ---- Common EMV / EMVCo tags (verified against EMV Book 3, Annex A) ----

    /// `6F` — File Control Information Template (constructed).
    pub const FCI_TEMPLATE: Self = Self(0x6F);
    /// `70` — Application Elementary File Record Template (constructed).
    pub const AEF_DATA_TEMPLATE: Self = Self(0x70);
    /// `77` — Response Message Template Format 2 (constructed).
    pub const RESPONSE_FMT2: Self = Self(0x77);
    /// `80` — Response Message Template Format 1 (primitive).
    pub const RESPONSE_FMT1: Self = Self(0x80);
    /// `84` — Dedicated File (DF) Name.
    pub const DF_NAME: Self = Self(0x84);
    /// `A5` — FCI Proprietary Template (constructed).
    pub const FCI_PROPRIETARY: Self = Self(0xA5);
    /// `5A` — Application Primary Account Number (PAN).
    pub const PAN: Self = Self(0x5A);
    /// `5F20` — Cardholder Name.
    pub const CARDHOLDER_NAME: Self = Self(0x5F20);
    /// `5F24` — Application Expiration Date (YYMMDD, BCD).
    pub const EXPIRY: Self = Self(0x5F24);
    /// `5F2A` — Transaction Currency Code (ISO 4217 numeric, BCD).
    pub const TXN_CURRENCY: Self = Self(0x5F2A);
    /// `5F2D` — Language Preference.
    pub const LANGUAGE: Self = Self(0x5F2D);
    /// `5F34` — Application PAN Sequence Number.
    pub const PAN_SEQ: Self = Self(0x5F34);
    /// `82` — Application Interchange Profile.
    pub const AIP: Self = Self(0x82);
    /// `88` — Short File Identifier (SFI).
    pub const SFI: Self = Self(0x88);
    /// `8A` — Authorisation Response Code.
    pub const ARC: Self = Self(0x8A);
    /// `9A` — Transaction Date (YYMMDD, BCD).
    pub const TXN_DATE: Self = Self(0x9A);
    /// `9C` — Transaction Type (00 = purchase, 09 = cashback, ...).
    pub const TXN_TYPE: Self = Self(0x9C);
    /// `9F02` — Amount, Authorised (Numeric, BCD, 12 digits).
    pub const AMOUNT_AUTHORISED: Self = Self(0x9F02);
    /// `9F03` — Amount, Other (Numeric, BCD, 12 digits).
    pub const AMOUNT_OTHER: Self = Self(0x9F03);
    /// `9F1A` — Terminal Country Code (ISO 3166-1 numeric, BCD).
    pub const TERMINAL_COUNTRY: Self = Self(0x9F1A);
    /// `9F26` — Application Cryptogram. The crown jewel — proves the card
    /// generated the response, not a clone.
    pub const CRYPTOGRAM: Self = Self(0x9F26);
    /// `9F27` — Cryptogram Information Data.
    pub const CID: Self = Self(0x9F27);
    /// `9F36` — Application Transaction Counter (ATC).
    pub const ATC: Self = Self(0x9F36);
    /// `9F37` — Unpredictable Number.
    pub const UNPREDICTABLE_NUMBER: Self = Self(0x9F37);

    /// The tag's first byte on the wire (top byte of the packed integer).
    #[must_use]
    pub const fn first_byte(self) -> u8 {
        let mut t = self.0;
        while t > 0xFF {
            t >>= 8;
        }
        // The loop exits only when `t <= 0xFF`, so this narrowing is
        // exact, not a truncation.
        #[allow(clippy::cast_possible_truncation)]
        let byte = t as u8;
        byte
    }

    /// Tag class.
    #[must_use]
    pub const fn class(self) -> TagClass {
        TagClass::from_first_byte(self.first_byte())
    }

    /// True if this tag's value is itself a sequence of TLVs.
    #[must_use]
    pub const fn is_constructed(self) -> bool {
        (self.first_byte() & 0x20) != 0
    }

    /// Number of bytes this tag occupies on the wire (1–4).
    #[must_use]
    pub const fn wire_len(self) -> usize {
        if self.0 > 0x00FF_FFFF {
            4
        } else if self.0 > 0x0000_FFFF {
            3
        } else if self.0 > 0x0000_00FF {
            2
        } else {
            1
        }
    }

    /// Read a tag from `buf[start..]`. Returns the tag and the byte
    /// count consumed.
    ///
    /// # Errors
    /// - `UnexpectedEof` if `buf` is too short.
    /// - `TagTooLong` if the multi-byte tag exceeds 4 total bytes.
    pub fn read(buf: &[u8], start: usize) -> Result<(Self, usize)> {
        if start >= buf.len() {
            return Err(Error::UnexpectedEof { offset: start });
        }
        let first = buf[start];
        // Short form if bits 5–1 are not all set.
        if first & 0x1F != 0x1F {
            return Ok((Self(u32::from(first)), 1));
        }
        // Multi-byte tag. Up to 3 more bytes; bit 8 = "more follows".
        let mut value = u32::from(first);
        let mut consumed = 1;
        loop {
            if start + consumed >= buf.len() {
                return Err(Error::UnexpectedEof {
                    offset: start + consumed,
                });
            }
            let b = buf[start + consumed];
            consumed += 1;
            value = (value << 8) | u32::from(b);
            if consumed > 4 {
                return Err(Error::TagTooLong { offset: start });
            }
            if b & 0x80 == 0 {
                return Ok((Self(value), consumed));
            }
        }
    }

    /// Write this tag's bytes into `buf`. Returns the count written.
    /// Used by encoders.
    ///
    /// # Errors
    /// `LengthExceedsBuffer` if `buf` is too small.
    pub fn write(self, buf: &mut [u8]) -> Result<usize> {
        let n = self.wire_len();
        if buf.len() < n {
            return Err(Error::LengthExceedsBuffer {
                declared: n,
                remaining: buf.len(),
                offset: 0,
            });
        }
        let bytes = self.0.to_be_bytes();
        buf[..n].copy_from_slice(&bytes[4 - n..]);
        Ok(n)
    }
}

impl core::fmt::Debug for Tag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Tag({:X})", self.0)
    }
}

impl core::fmt::Display for Tag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:X}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Single-byte tags ----

    #[test]
    fn fci_template_is_constructed() {
        assert!(Tag::FCI_TEMPLATE.is_constructed());
        assert_eq!(Tag::FCI_TEMPLATE.class(), TagClass::Application);
        assert_eq!(Tag::FCI_TEMPLATE.wire_len(), 1);
    }

    #[test]
    fn df_name_is_primitive() {
        assert!(!Tag::DF_NAME.is_constructed());
        // 0x84 = 1000_0100 -> bits 87 = 10 -> ContextSpecific.
        assert_eq!(Tag::DF_NAME.class(), TagClass::ContextSpecific);
    }

    #[test]
    fn fci_proprietary_is_constructed() {
        assert!(Tag::FCI_PROPRIETARY.is_constructed());
        // 0xA5 -> bits 87 = 10 (context-specific)
        assert_eq!(Tag::FCI_PROPRIETARY.class(), TagClass::ContextSpecific);
    }

    // ---- Multi-byte tags ----

    #[test]
    fn amount_tag_is_multibyte() {
        assert_eq!(Tag::AMOUNT_AUTHORISED.0, 0x9F02);
        assert_eq!(Tag::AMOUNT_AUTHORISED.wire_len(), 2);
        assert!(!Tag::AMOUNT_AUTHORISED.is_constructed());
        // 0x9F starts with bits 87 = 10
        assert_eq!(Tag::AMOUNT_AUTHORISED.class(), TagClass::ContextSpecific);
    }

    #[test]
    fn language_tag_first_byte() {
        // 0x5F2D — first byte 0x5F
        assert_eq!(Tag::LANGUAGE.first_byte(), 0x5F);
        assert_eq!(Tag::LANGUAGE.wire_len(), 2);
    }

    // ---- Tag::read ----

    #[test]
    fn read_single_byte_tag() {
        let (t, n) = Tag::read(&[0x6F, 0x10], 0).unwrap();
        assert_eq!(t.0, 0x6F);
        assert_eq!(n, 1);
    }

    #[test]
    fn read_two_byte_tag() {
        let (t, n) = Tag::read(&[0x9F, 0x02, 0x06], 0).unwrap();
        assert_eq!(t.0, 0x9F02);
        assert_eq!(n, 2);
    }

    #[test]
    fn read_tag_with_offset() {
        let buf = [0x00, 0x00, 0x9F, 0x37, 0x04, 0xAA];
        let (t, n) = Tag::read(&buf, 2).unwrap();
        assert_eq!(t.0, 0x9F37);
        assert_eq!(n, 2);
    }

    #[test]
    fn read_truncated_tag_errors() {
        // 0x1F means multi-byte; nothing follows.
        let buf = [0x1F];
        assert!(matches!(
            Tag::read(&buf, 0),
            Err(Error::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn read_too_long_tag_errors() {
        // Five bytes all with bit-8 set => >4 total bytes
        let buf = [0x1F, 0x80, 0x80, 0x80, 0x80, 0x01];
        assert!(matches!(Tag::read(&buf, 0), Err(Error::TagTooLong { .. })));
    }

    #[test]
    fn read_empty_buffer_errors() {
        assert!(matches!(
            Tag::read(&[], 0),
            Err(Error::UnexpectedEof { offset: 0 })
        ));
    }

    // ---- Tag::write ----

    #[test]
    fn write_round_trips_single_byte() {
        let mut buf = [0u8; 4];
        let n = Tag(0x6F).write(&mut buf).unwrap();
        assert_eq!(n, 1);
        assert_eq!(&buf[..1], &[0x6F]);
    }

    #[test]
    fn write_round_trips_two_byte() {
        let mut buf = [0u8; 4];
        let n = Tag(0x9F02).write(&mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&buf[..2], &[0x9F, 0x02]);
    }

    #[test]
    fn write_round_trips_three_byte() {
        let mut buf = [0u8; 4];
        let n = Tag(0xDF_AB_CD).write(&mut buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(&buf[..3], &[0xDF, 0xAB, 0xCD]);
    }

    #[test]
    fn write_fails_on_small_buffer() {
        let mut buf = [0u8; 1];
        assert!(Tag(0x9F02).write(&mut buf).is_err());
    }

    // ---- Read/write symmetry ----

    #[test]
    fn read_write_round_trip_all_known_tags() {
        let tags = [
            Tag::FCI_TEMPLATE,
            Tag::DF_NAME,
            Tag::FCI_PROPRIETARY,
            Tag::PAN,
            Tag::EXPIRY,
            Tag::TXN_CURRENCY,
            Tag::LANGUAGE,
            Tag::AMOUNT_AUTHORISED,
            Tag::AMOUNT_OTHER,
            Tag::TERMINAL_COUNTRY,
            Tag::CRYPTOGRAM,
            Tag::ATC,
            Tag::UNPREDICTABLE_NUMBER,
        ];
        let mut buf = [0u8; 4];
        for t in tags {
            let n = t.write(&mut buf).unwrap();
            let (parsed, consumed) = Tag::read(&buf[..n], 0).unwrap();
            assert_eq!(parsed, t, "round-trip failed for {t:?}");
            assert_eq!(consumed, n);
        }
    }
}
