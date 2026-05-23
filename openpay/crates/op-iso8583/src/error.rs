//! Typed errors for `op-iso8583`. One sealed enum per project convention.

use thiserror::Error;

/// Result alias for `op-iso8583`.
pub type Result<T> = core::result::Result<T, Error>;

/// Failure modes for ISO 8583 message handling.
#[derive(Debug, Error)]
pub enum Error {
    /// The byte slice ran out mid-decode.
    #[error("unexpected end of input at offset {offset} (needed {needed} bytes)")]
    UnexpectedEof {
        /// Offset into the input where the read began.
        offset: usize,
        /// Number of bytes the codec needed.
        needed: usize,
    },

    /// A byte was not a valid BCD nibble (0x0..=0x9) when one was expected.
    #[error("invalid BCD nibble 0x{nibble:02x} at offset {offset}")]
    InvalidBcd {
        /// Offending nibble (high or low).
        nibble: u8,
        /// Offset into the input.
        offset: usize,
    },

    /// A byte was not valid ASCII (or fell outside the 0x20..=0x7E printable
    /// range where the dialect required it).
    #[error("invalid ASCII byte 0x{byte:02x} at offset {offset}")]
    InvalidAscii {
        /// Offending byte.
        byte: u8,
        /// Offset into the input.
        offset: usize,
    },

    /// A variable-length field carried a length prefix outside the allowed
    /// range for its encoding (LL: 0..=99, LLL: 0..=999).
    #[error("invalid variable-length prefix {prefix} (max {max})")]
    InvalidVarLength {
        /// Decoded length value.
        prefix: usize,
        /// Maximum legal length for the encoding.
        max: usize,
    },

    /// The Message Type Indicator was not four BCD/ASCII numeric digits.
    #[error("invalid MTI: {0}")]
    InvalidMti(String),

    /// A bitmap bit was set referencing a data element with no decoder
    /// registered in this dialect.
    #[error("unknown data element {0}")]
    UnknownDataElement(u8),

    /// A required data element was missing from the bitmap.
    #[error("missing required data element {0}")]
    MissingDataElement(u8),

    /// A data element's decoded value did not satisfy its semantic
    /// constraints (e.g. DE 39 response code not 2 digits, DE 49
    /// currency not 3 numeric digits).
    #[error("invalid data element {de}: {reason}")]
    InvalidDataElement {
        /// Data-element number.
        de: u8,
        /// Human-readable reason.
        reason: String,
    },

    /// Track 2 / DE 35 was malformed (missing separator, bad PAN, bad
    /// expiration date, etc.).
    #[error("invalid track 2 data: {0}")]
    InvalidTrack2(&'static str),

    /// DE 55 EMV TLV could not be walked. Carries the failing offset
    /// (within the DE 55 value) and a static reason.
    #[error("invalid EMV TLV at offset {offset}: {reason}")]
    InvalidEmvTlv {
        /// Offset into the DE 55 value.
        offset: usize,
        /// Static reason ("length overflows value", "constructed before primitive", ...).
        reason: &'static str,
    },

    /// MAC verification failed: computed MAC did not match the value
    /// carried in DE 64 / DE 128.
    #[error("MAC mismatch")]
    MacMismatch,

    /// The dialect rejected a field for a network-specific reason
    /// (e.g. Mastercard MDS requires DE 22 sub-fields the message lacks).
    #[error("dialect {dialect} rejected: {reason}")]
    DialectViolation {
        /// Which dialect.
        dialect: &'static str,
        /// Why.
        reason: String,
    },

    /// Forwarded `op-core` error (currency / overflow during Money mapping).
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
