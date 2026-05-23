//! Owned TLV tree.
//!
//! `Tlv` is a heap-allocated tree, useful for:
//! - Diagnostic logging (pretty-printing a full payload)
//! - Caching parsed messages
//! - The fraud-detection feature extractor in `op-fraud`, which needs
//!   to walk the tree multiple times
//!
//! For the hot path on the secure-element side, use [`crate::stream`]
//! directly — it doesn't allocate.

use alloc::boxed::Box;
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::stream::TlvIter;
use crate::tag::Tag;

/// An owned TLV with either a primitive byte payload or a list of
/// children.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tlv {
    /// Tag identifier.
    pub tag: Tag,
    /// Primitive value or constructed children.
    pub body: TlvBody,
}

/// Either a leaf byte payload or an internal node with children.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TlvBody {
    /// A primitive value — opaque byte payload.
    Primitive(Vec<u8>),
    /// Constructed — the children, in source order.
    Constructed(Vec<Tlv>),
}

impl Tlv {
    /// Parse a full byte slice into a vector of top-level TLVs.
    ///
    /// Recursively descends into every constructed TLV. Padding (`0x00`
    /// bytes between TLVs) is silently skipped.
    ///
    /// # Errors
    /// First TLV error in the stream.
    pub fn parse_all(buf: &[u8]) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for entry in TlvIter::new(buf) {
            let r = entry?;
            out.push(Self::from_ref(&r)?);
        }
        Ok(out)
    }

    /// Build an owned `Tlv` from a borrowed `TlvRef`.
    pub(crate) fn from_ref(r: &crate::stream::TlvRef<'_>) -> Result<Self> {
        let body = if r.tag.is_constructed() {
            let mut children = Vec::new();
            for child in TlvIter::with_base_offset(r.value, r.value_offset) {
                children.push(Self::from_ref(&child?)?);
            }
            TlvBody::Constructed(children)
        } else {
            TlvBody::Primitive(r.value.to_vec())
        };
        Ok(Self { tag: r.tag, body })
    }

    /// Find the first descendant with the given tag (depth-first).
    /// Returns `None` if no such tag exists in the tree.
    #[must_use]
    pub fn find(&self, tag: Tag) -> Option<&Self> {
        if self.tag == tag {
            return Some(self);
        }
        if let TlvBody::Constructed(children) = &self.body {
            for c in children {
                if let Some(hit) = c.find(tag) {
                    return Some(hit);
                }
            }
        }
        None
    }

    /// If this is a primitive TLV, return its bytes. None on constructed.
    #[must_use]
    pub fn primitive(&self) -> Option<&[u8]> {
        match &self.body {
            TlvBody::Primitive(b) => Some(b.as_slice()),
            TlvBody::Constructed(_) => None,
        }
    }

    /// Total encoded length (tag + length-of-length + length + value).
    /// Useful for buffer pre-sizing in encoders.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        let value_len = match &self.body {
            TlvBody::Primitive(b) => b.len(),
            TlvBody::Constructed(c) => c.iter().map(Self::encoded_len).sum(),
        };
        let length_len = if value_len < 0x80 {
            1
        } else if value_len < 0x100 {
            2
        } else if value_len < 0x10000 {
            3
        } else if value_len < 0x0100_0000 {
            4
        } else {
            5
        };
        self.tag.wire_len() + length_len + value_len
    }
}

/// Convenience: search a list of top-level TLVs.
pub trait TlvSliceExt {
    /// Find the first TLV (or descendant) with the given tag.
    fn find_tag(&self, tag: Tag) -> Option<&Tlv>;
}

impl TlvSliceExt for [Tlv] {
    fn find_tag(&self, tag: Tag) -> Option<&Tlv> {
        for t in self {
            if let Some(hit) = t.find(tag) {
                return Some(hit);
            }
        }
        None
    }
}

// We need Box import for some no_std arrangements; keep it to silence
// future warnings.
#[allow(dead_code)]
fn _ensure_box_imported() -> Box<u8> {
    Box::new(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_flat_primitives() {
        // Two primitives: 9A 03 26 05 17, 9C 01 00
        let buf = [0x9A, 0x03, 0x26, 0x05, 0x17, 0x9C, 0x01, 0x00];
        let tree = Tlv::parse_all(&buf).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree[0].tag, Tag::TXN_DATE);
        assert_eq!(tree[0].primitive(), Some(&[0x26, 0x05, 0x17][..]));
        assert_eq!(tree[1].tag, Tag::TXN_TYPE);
        assert_eq!(tree[1].primitive(), Some(&[0x00][..]));
    }

    #[test]
    fn parse_nested_constructed() {
        // 6F 06 84 02 AA BB A5 00
        let buf = [0x6F, 0x06, 0x84, 0x02, 0xAA, 0xBB, 0xA5, 0x00];
        let tree = Tlv::parse_all(&buf).unwrap();
        assert_eq!(tree.len(), 1);
        match &tree[0].body {
            TlvBody::Constructed(children) => {
                assert_eq!(children.len(), 2);
                assert_eq!(children[0].tag, Tag::DF_NAME);
                assert_eq!(children[0].primitive(), Some(&[0xAA, 0xBB][..]));
                assert_eq!(children[1].tag, Tag::FCI_PROPRIETARY);
            }
            TlvBody::Primitive(_) => panic!("expected constructed body for 6F"),
        }
    }

    #[test]
    fn find_descends_deep() {
        // 6F wraps A5 wraps 88 01 02 and 5F 2D 02 65 6E
        // Inner of A5: 88 01 02 (3) + 5F 2D 02 65 6E (5) = 8 bytes -> A5 08
        // Inner of 6F: A5 08 + 8 = 10 bytes -> 6F 0A
        let buf = [
            0x6F, 0x0A, 0xA5, 0x08, 0x88, 0x01, 0x02, 0x5F, 0x2D, 0x02, 0x65, 0x6E,
        ];
        let tree = Tlv::parse_all(&buf).unwrap();
        let language = tree[0].find(Tag::LANGUAGE).expect("should find 5F2D");
        assert_eq!(language.primitive(), Some(&[0x65, 0x6E][..]));
        let sfi = tree[0].find(Tag::SFI).expect("should find 88");
        assert_eq!(sfi.primitive(), Some(&[0x02][..]));
    }

    #[test]
    fn find_returns_none_for_absent_tag() {
        let buf = [0x9A, 0x03, 0x26, 0x05, 0x17];
        let tree = Tlv::parse_all(&buf).unwrap();
        assert!(tree[0].find(Tag::CRYPTOGRAM).is_none());
    }

    #[test]
    fn encoded_len_short_form() {
        // 9A 03 26 05 17 => 5 bytes total
        let buf = [0x9A, 0x03, 0x26, 0x05, 0x17];
        let tree = Tlv::parse_all(&buf).unwrap();
        assert_eq!(tree[0].encoded_len(), 5);
    }

    #[test]
    fn encoded_len_long_form() {
        // 0x5A (PAN) is a primitive tag, so a 256-byte opaque value is
        // legal. A constructed tag like 0x6F would require its value to
        // itself be valid nested TLV, which raw filler is not.
        let mut buf = vec![0x5A, 0x82, 0x01, 0x00];
        buf.extend(std::iter::repeat_n(0xAA, 256));
        let tree = Tlv::parse_all(&buf).unwrap();
        // tag(1) + length-field(3 = 0x82 + 2 bytes) + value(256) = 260
        assert_eq!(tree[0].encoded_len(), 260);
    }
}
