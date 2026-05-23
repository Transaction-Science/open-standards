//! Errors. Sealed enum, exhaustive match at every callsite.

use thiserror::Error;

/// Result alias for `op-emv`.
pub type Result<T> = core::result::Result<T, Error>;

/// All possible failure modes when decoding BER-TLV data.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// The buffer ended before we finished reading a TLV.
    #[error("unexpected end of input at offset {offset}")]
    UnexpectedEof {
        /// Byte offset at which the buffer was exhausted.
        offset: usize,
    },

    /// Tag continued past the maximum 4 bytes EMV permits.
    #[error("tag exceeds 4 bytes (starts at offset {offset})")]
    TagTooLong {
        /// Offset of the tag's first byte.
        offset: usize,
    },

    /// Length field uses indefinite form (`0x80` alone), which BER
    /// allows but EMV bans.
    #[error("indefinite length not allowed in EMV BER-TLV (offset {offset})")]
    IndefiniteLength {
        /// Offset of the length byte.
        offset: usize,
    },

    /// Long-form length declared more than 4 length bytes — EMV cap.
    #[error("length field uses {bytes} bytes (max 4) at offset {offset}")]
    LengthTooLong {
        /// Number of length bytes the encoding claims.
        bytes: usize,
        /// Offset of the length-of-length byte.
        offset: usize,
    },

    /// Declared value length exceeds the remaining buffer.
    #[error("declared value length {declared} exceeds remaining {remaining} at offset {offset}")]
    LengthExceedsBuffer {
        /// Length declared by the TLV.
        declared: usize,
        /// Bytes actually available after the length field.
        remaining: usize,
        /// Offset of the value's first byte.
        offset: usize,
    },

    /// Tried to descend into a primitive TLV as if it were constructed.
    #[error("tag {tag:#X} at offset {offset} is primitive; cannot iterate as constructed")]
    NotConstructed {
        /// The tag value.
        tag: u32,
        /// Offset of its first byte.
        offset: usize,
    },

    /// Tag class / structure violates EMV rules (reserved value, etc.).
    #[error("invalid tag {tag:#X} at offset {offset}: {reason}")]
    InvalidTag {
        /// Tag bytes interpreted as integer.
        tag: u32,
        /// Offset.
        offset: usize,
        /// Why.
        reason: &'static str,
    },
}
