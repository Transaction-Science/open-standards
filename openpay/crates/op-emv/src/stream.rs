//! Zero-allocation streaming TLV iterator.
//!
//! Parses BER-TLV one entry at a time, returning a [`TlvRef`] that
//! borrows from the input slice. Suitable for `no_std` and for the
//! secure-element interface where heap allocation isn't an option.
//!
//! Padding (`0x00` bytes between TLVs) is silently skipped per
//! EMV Book 3 §B.3.
//!
//! ## Usage
//!
//! ```
//! use op_emv::stream::TlvIter;
//! let bytes = [0x9Fu8, 0x02, 0x06, 0, 0, 0, 0, 1, 0];
//! for tlv in TlvIter::new(&bytes) {
//!     let tlv = tlv.unwrap();
//!     // tlv.tag, tlv.value, tlv.offset
//! }
//! ```

use crate::error::{Error, Result};
use crate::tag::Tag;

/// One TLV, borrowed from the input buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TlvRef<'a> {
    /// The tag.
    pub tag: Tag,
    /// The value bytes (length is `value.len()`).
    pub value: &'a [u8],
    /// Byte offset of this TLV's first (tag) byte within the original
    /// buffer. Useful for error reporting and reconstruction.
    pub offset: usize,
    /// Byte offset of the *value* within the original buffer, i.e.
    /// `offset` advanced past the tag and length fields. This is the
    /// correct base when descending into a constructed TLV's children
    /// so their offsets stay relative to the original input.
    pub value_offset: usize,
}

impl<'a> TlvRef<'a> {
    /// Iterate over the children of a constructed TLV.
    ///
    /// # Errors
    /// `NotConstructed` if `self.tag` is primitive.
    pub fn children(&self) -> Result<TlvIter<'a>> {
        if !self.tag.is_constructed() {
            return Err(Error::NotConstructed {
                tag: self.tag.0,
                offset: self.offset,
            });
        }
        Ok(TlvIter::with_base_offset(self.value, self.value_offset))
    }
}

/// Lending-iterator over a BER-TLV byte slice.
///
/// Implements `Iterator<Item = Result<TlvRef<'a>>>`. Failures terminate
/// iteration: once the iterator yields `Err(_)`, subsequent `next()`
/// calls return `None`.
#[derive(Clone, Debug)]
pub struct TlvIter<'a> {
    buf: &'a [u8],
    pos: usize,
    base_offset: usize,
    failed: bool,
}

impl<'a> TlvIter<'a> {
    /// Construct a fresh iterator over `buf`, treating the start of
    /// `buf` as absolute offset 0.
    #[must_use]
    pub const fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            base_offset: 0,
            failed: false,
        }
    }

    /// Construct an iterator over a sub-slice, preserving the absolute
    /// offset of the parent so error messages and `TlvRef.offset`
    /// values stay accurate across nested constructed TLVs.
    #[must_use]
    pub const fn with_base_offset(buf: &'a [u8], base_offset: usize) -> Self {
        Self {
            buf,
            pos: 0,
            base_offset,
            failed: false,
        }
    }

    /// Read a BER-TLV length field starting at `self.pos`.
    fn read_length(&mut self) -> Result<usize> {
        if self.pos >= self.buf.len() {
            return Err(Error::UnexpectedEof {
                offset: self.base_offset + self.pos,
            });
        }
        let first = self.buf[self.pos];
        self.pos += 1;
        if first & 0x80 == 0 {
            return Ok(first as usize);
        }
        let n_bytes = (first & 0x7F) as usize;
        if n_bytes == 0 {
            return Err(Error::IndefiniteLength {
                offset: self.base_offset + self.pos - 1,
            });
        }
        if n_bytes > 4 {
            return Err(Error::LengthTooLong {
                bytes: n_bytes,
                offset: self.base_offset + self.pos - 1,
            });
        }
        if self.pos + n_bytes > self.buf.len() {
            return Err(Error::UnexpectedEof {
                offset: self.base_offset + self.pos + n_bytes,
            });
        }
        let mut length: usize = 0;
        for _ in 0..n_bytes {
            length = (length << 8) | self.buf[self.pos] as usize;
            self.pos += 1;
        }
        Ok(length)
    }
}

impl<'a> Iterator for TlvIter<'a> {
    type Item = Result<TlvRef<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }

        // Skip 0x00 padding bytes between TLVs. EMV Book 3 §B.3 permits
        // arbitrary padding; we silently consume it.
        while self.pos < self.buf.len() && self.buf[self.pos] == 0x00 {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            return None;
        }

        let entry_start = self.pos;

        // Tag.
        let (tag, tag_len) = match Tag::read(self.buf, self.pos) {
            Ok(t) => t,
            Err(e) => {
                self.failed = true;
                return Some(Err(match e {
                    Error::UnexpectedEof { offset } => Error::UnexpectedEof {
                        offset: self.base_offset + offset,
                    },
                    Error::TagTooLong { offset } => Error::TagTooLong {
                        offset: self.base_offset + offset,
                    },
                    other => other,
                }));
            }
        };
        self.pos += tag_len;

        // Length.
        let length = match self.read_length() {
            Ok(l) => l,
            Err(e) => {
                self.failed = true;
                return Some(Err(e));
            }
        };

        // Value.
        let remaining = self.buf.len() - self.pos;
        if length > remaining {
            self.failed = true;
            return Some(Err(Error::LengthExceedsBuffer {
                declared: length,
                remaining,
                offset: self.base_offset + self.pos,
            }));
        }
        let value_start = self.base_offset + self.pos;
        let value = &self.buf[self.pos..self.pos + length];
        self.pos += length;

        Some(Ok(TlvRef {
            tag,
            value,
            offset: self.base_offset + entry_start,
            value_offset: value_start,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Length decoding ----

    #[test]
    fn short_form_length() {
        let mut it = TlvIter::new(&[0x6F, 0x05, 1, 2, 3, 4, 5]);
        let tlv = it.next().unwrap().unwrap();
        assert_eq!(tlv.value.len(), 5);
        assert_eq!(tlv.value, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn long_form_length_one_byte() {
        // 0x81 means "one more length byte follows"
        let mut buf = vec![0x6F, 0x81, 0x80]; // length 128
        buf.extend(std::iter::repeat_n(0xAB, 128));
        let mut it = TlvIter::new(&buf);
        let tlv = it.next().unwrap().unwrap();
        assert_eq!(tlv.value.len(), 128);
    }

    #[test]
    fn long_form_length_two_bytes() {
        // 0x82 0x01 0x00 = 256
        let mut buf = vec![0x6F, 0x82, 0x01, 0x00];
        buf.extend(std::iter::repeat_n(0xCD, 256));
        let mut it = TlvIter::new(&buf);
        let tlv = it.next().unwrap().unwrap();
        assert_eq!(tlv.value.len(), 256);
    }

    #[test]
    fn indefinite_length_rejected() {
        // 0x80 alone = indefinite, banned by EMV.
        let mut it = TlvIter::new(&[0x6F, 0x80, 0x01, 0x02, 0x00, 0x00]);
        let err = it.next().unwrap();
        assert!(matches!(err, Err(Error::IndefiniteLength { .. })));
    }

    #[test]
    fn length_too_long_rejected() {
        // 0x85 = "5 length bytes follow" — EMV max is 4
        let buf = [0x6F, 0x85, 0, 0, 0, 0, 1];
        let mut it = TlvIter::new(&buf);
        let err = it.next().unwrap();
        assert!(matches!(err, Err(Error::LengthTooLong { bytes: 5, .. })));
    }

    #[test]
    fn length_exceeds_buffer_rejected() {
        let buf = [0x6F, 0x10, 0x01, 0x02]; // declares 16, only 2 follow
        let mut it = TlvIter::new(&buf);
        let err = it.next().unwrap();
        assert!(matches!(
            err,
            Err(Error::LengthExceedsBuffer {
                declared: 16,
                remaining: 2,
                ..
            })
        ));
    }

    // ---- Multi-byte tags ----

    #[test]
    fn multibyte_tag_parsed() {
        let buf = [0x9F, 0x02, 0x06, 0, 0, 0, 0, 1, 0];
        let mut it = TlvIter::new(&buf);
        let tlv = it.next().unwrap().unwrap();
        assert_eq!(tlv.tag.0, 0x9F02);
        assert_eq!(tlv.value.len(), 6);
    }

    // ---- Padding ----

    #[test]
    fn padding_between_tlvs_skipped() {
        let buf = [0x00, 0x00, 0x9F, 0x02, 0x01, 0xAA, 0x00, 0x9A, 0x01, 0xBB];
        let parsed: Vec<TlvRef<'_>> = TlvIter::new(&buf).map(|t| t.unwrap()).collect();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].tag.0, 0x9F02);
        assert_eq!(parsed[0].value, &[0xAA]);
        assert_eq!(parsed[1].tag.0, 0x9A);
        assert_eq!(parsed[1].value, &[0xBB]);
    }

    #[test]
    fn trailing_padding_ends_cleanly() {
        let buf = [0x9F, 0x02, 0x01, 0xAA, 0x00, 0x00, 0x00];
        let parsed: Vec<TlvRef<'_>> = TlvIter::new(&buf).map(|t| t.unwrap()).collect();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn empty_buffer_yields_nothing() {
        let mut it = TlvIter::new(&[]);
        assert!(it.next().is_none());
    }

    #[test]
    fn all_padding_yields_nothing() {
        let mut it = TlvIter::new(&[0x00, 0x00, 0x00]);
        assert!(it.next().is_none());
    }

    // ---- Offsets ----

    #[test]
    fn offsets_track_position_in_input() {
        // Two TLVs back to back.
        let buf = [0x9A, 0x03, 0x01, 0x02, 0x03, 0x9C, 0x01, 0x00];
        let parsed: Vec<TlvRef<'_>> = TlvIter::new(&buf).map(|t| t.unwrap()).collect();
        assert_eq!(parsed[0].offset, 0);
        assert_eq!(parsed[1].offset, 5);
    }

    // ---- Constructed children ----

    #[test]
    fn constructed_children_iterate() {
        // 6F (constructed) wraps 84 (primitive) and A5 (constructed).
        let buf = [0x6F, 0x06, 0x84, 0x02, 0xAA, 0xBB, 0xA5, 0x00];
        let outer = TlvIter::new(&buf).next().unwrap().unwrap();
        assert_eq!(outer.tag.0, 0x6F);
        let children: Vec<TlvRef<'_>> = outer.children().unwrap().map(|t| t.unwrap()).collect();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].tag.0, 0x84);
        assert_eq!(children[0].value, &[0xAA, 0xBB]);
        assert_eq!(children[1].tag.0, 0xA5);
        assert!(children[1].value.is_empty());
    }

    #[test]
    fn child_offsets_relative_to_input() {
        // Wrapper at offset 0; children at 2 and 6 within input.
        let buf = [0x6F, 0x06, 0x84, 0x02, 0xAA, 0xBB, 0xA5, 0x00];
        let outer = TlvIter::new(&buf).next().unwrap().unwrap();
        let children: Vec<TlvRef<'_>> = outer.children().unwrap().map(|t| t.unwrap()).collect();
        assert_eq!(children[0].offset, 2); // 84 02 AA BB starts here
        assert_eq!(children[1].offset, 6); // A5 00 starts here
    }

    #[test]
    fn children_on_primitive_errors() {
        // 84 is primitive.
        let buf = [0x84, 0x02, 0xAA, 0xBB];
        let tlv = TlvIter::new(&buf).next().unwrap().unwrap();
        assert!(matches!(
            tlv.children(),
            Err(Error::NotConstructed { tag: 0x84, .. })
        ));
    }

    // ---- Iterator contract ----

    #[test]
    fn error_terminates_iteration() {
        let buf = [0x6F, 0x80]; // indefinite length error
        let mut it = TlvIter::new(&buf);
        assert!(it.next().unwrap().is_err());
        assert!(it.next().is_none());
        assert!(it.next().is_none()); // idempotent
    }
}
