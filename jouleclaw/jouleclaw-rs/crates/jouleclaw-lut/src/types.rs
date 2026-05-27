//! Public types for the LUT primitive.
//!
//! - [`LutKey`] — 128-bit BLAKE3-truncated hash of the normalised input.
//! - [`LutEntry`] — the stored value: output bytes, declared joule cost,
//!   source tag (registry attribution), and registration timestamp.
//! - [`LutHit`] — the lookup return shape: the same payload as an entry
//!   minus the timestamp, ready to hand to the cascade.
//! - [`LutError`] — IO + parse + serde errors from bulk-load paths.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 128-bit BLAKE3-truncated content hash of a normalised input.
///
/// The full BLAKE3 digest is 256 bits; we truncate to the leading 128
/// bits because the LUT is exact-match and 2^64 is comfortably more
/// than enough collision resistance for a per-runtime registered table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LutKey(pub u128);

impl LutKey {
    /// Compute the [`LutKey`] for an already-normalised input string.
    pub fn from_normalized(normalized: &str) -> Self {
        let digest = blake3::hash(normalized.as_bytes());
        let bytes = digest.as_bytes();
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&bytes[..16]);
        Self(u128::from_be_bytes(buf))
    }
}

/// A stored LUT entry — what `register` writes and `iter` yields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LutEntry {
    /// The pre-baked output bytes. Interpretation is up to the caller;
    /// the [`Tier`](jouleclaw_cascade::tier::Tier) impl surfaces them as
    /// `AnswerOutput::Text` when valid UTF-8 and
    /// `AnswerOutput::Structured` otherwise.
    pub output: Vec<u8>,
    /// Declared joule cost in microjoules. Mirrors the rest of the
    /// JouleClaw cost model where Tier estimates are stored in joules
    /// (this field is the per-entry attribution at the registration
    /// surface; the [`Tier`](jouleclaw_cascade::tier::Tier) impl
    /// converts µJ → J).
    pub declared_cost_uj: u64,
    /// Free-form provenance tag — typically the registry / dataset /
    /// table this entry came from (e.g. `"lawful:gcd"`,
    /// `"smartbyte:greeting/en"`).
    pub source_tag: String,
    /// Wall-clock timestamp at registration. Mostly observability —
    /// the LUT does not evict by age.
    pub registered_at: DateTime<Utc>,
}

/// The shape returned by [`Lut::try_lookup`] — the entry payload minus
/// the timestamp. Cheap to construct and clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LutHit {
    pub output: Vec<u8>,
    pub declared_cost_uj: u64,
    pub source_tag: String,
}

/// Errors from the LUT primitive — at the moment only the bulk-load
/// paths can fail.
#[derive(Debug, thiserror::Error)]
pub enum LutError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("toml parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),

    #[error("csv row missing required column `{0}`")]
    CsvMissingColumn(&'static str),

    #[error("csv row has invalid cost_uj `{0}`")]
    CsvBadCost(String),
}
