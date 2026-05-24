//! Error type shared across the quantization crate.

use thiserror::Error;

/// Errors returned by quantization primitives and format readers.
#[derive(Debug, Error)]
pub enum QuantError {
    /// Input slice was empty when a value was required.
    #[error("input slice was empty")]
    EmptyInput,

    /// Input length is not divisible by the requested group size.
    #[error("input length {len} is not divisible by group size {group}")]
    BadGroupSize {
        /// Length of the input slice.
        len: usize,
        /// Group size requested.
        group: usize,
    },

    /// Slice was shorter than required for parsing.
    #[error("buffer too short: needed {needed} bytes, got {got}")]
    Truncated {
        /// Required byte count.
        needed: usize,
        /// Actual byte count.
        got: usize,
    },

    /// Magic / version mismatch when reading a file format.
    #[error("invalid format: {0}")]
    InvalidFormat(&'static str),

    /// Value out of representable range for the target dtype.
    #[error("value {value} out of range for {dtype}")]
    OutOfRange {
        /// Offending value.
        value: f64,
        /// Target dtype name.
        dtype: &'static str,
    },
}
