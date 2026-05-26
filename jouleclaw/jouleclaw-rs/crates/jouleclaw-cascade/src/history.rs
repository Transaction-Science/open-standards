//! The history layer — durable, queryable record of past answers.
//!
//! L0 (the exact-match cache) is the hot path over the history layer.
//! Same answer-lookup interface, but the history layer adds:
//!   - durability (survives runtime restarts when disk-backed)
//!   - time-indexed retrieval (recent queries by timestamp)
//!   - aggregate statistics (total joule spend, hit rate, by-tier counts)
//!   - hooks for semantic retrieval (filled by R5's embedding model)
//!
//! This module defines the `HistoryLayer` trait. Implementations live
//! in the `jouleclaw-history` crate.

use crate::types::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// A 256-bit content-addressed key for an entry.
pub type EntryKey = [u8; 32];

/// Errors from history-layer operations.
#[derive(Debug)]
pub enum HistoryError {
    Io(std::io::Error),
    Corrupt(String),
    /// The backend has no support for the requested operation. Used by
    /// the in-memory backend when asked for semantic lookup, etc.
    Unsupported(&'static str),
}

impl std::fmt::Display for HistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {}", e),
            Self::Corrupt(s) => write!(f, "corrupt: {}", s),
            Self::Unsupported(s) => write!(f, "unsupported: {}", s),
        }
    }
}

impl std::error::Error for HistoryError {}

impl From<std::io::Error> for HistoryError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

/// A stored answer plus indexing metadata. The history layer records
/// these and returns them on lookup.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub key: EntryKey,
    /// The original query input. Stored for replay and debugging.
    pub query_input: QueryInput,
    pub query_context: ContextRef,
    /// The cached answer. tier_used here is the ORIGINATING tier
    /// (whatever produced the answer the first time), not L0.
    pub answer: HistoryAnswer,
    /// Unix epoch seconds when recorded.
    pub timestamp_secs: u64,
    /// Embedding of the query, if available. Empty until R5.
    pub embedding: Vec<f32>,
}

/// The subset of `Answer` we durably store. Stream variants are
/// resolved to text before storage.
#[derive(Debug, Clone)]
pub struct HistoryAnswer {
    pub output: AnswerOutput,
    pub originating_tier: TierId,
    pub joules_spent: f64,
    pub confidence: f32,
}

/// Aggregate statistics about the history layer's contents and access
/// patterns.
#[derive(Debug, Clone, Default)]
pub struct HistoryStats {
    pub entry_count: usize,
    pub total_lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    /// Sum of `joules_spent` across all stored answers. The cumulative
    /// total cost the runtime has paid to populate the history.
    pub joules_recorded: f64,
}

impl HistoryStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 { 0.0 }
        else { self.hits as f64 / self.total_lookups as f64 }
    }
}

/// The history layer interface. Implementations may be in-memory,
/// disk-backed, or remote.
pub trait HistoryLayer: Send {
    /// Look up an answer by its exact key. The key is computed from
    /// the query via `EntryKey::for_query`.
    fn lookup_exact(&mut self, key: &EntryKey) -> Result<Option<HistoryAnswer>, HistoryError>;

    /// Look up answers semantically similar to the given embedding.
    /// R3 implementations return `Unsupported`. R5 (after embeddings
    /// land) implements this.
    fn lookup_semantic(
        &mut self,
        _embedding: &[f32],
        _k: usize,
        _min_sim: f32,
    ) -> Result<Vec<(HistoryAnswer, f32)>, HistoryError> {
        Err(HistoryError::Unsupported("semantic lookup arrives in R5"))
    }

    /// Record an answer keyed by its query. Idempotent: re-recording
    /// the same key overwrites.
    fn record(&mut self, q: &Query, a: &Answer) -> Result<EntryKey, HistoryError>;

    /// Estimate the joule cost of a lookup. Cheap (microseconds).
    fn estimate_lookup_cost(&self, q: &Query) -> f64;

    /// Current statistics.
    fn stats(&self) -> &HistoryStats;
}

/// Compute the canonical 256-bit key for a query. The key is content-
/// addressed: structurally-identical queries (same input, same context
/// fingerprint) produce the same key.
pub fn key_for(q: &Query) -> EntryKey {
    use jouleclaw_core::hash::Hasher256;
    let mut h = Hasher256::new();
    h.update(b"L0v1");  // domain separation; matches L0Cache::key_for
    match &q.input {
        QueryInput::Text(s) => {
            h.update(b"T:");
            h.update(s.as_bytes());
        }
        QueryInput::Structured(b) => {
            h.update(b"S:");
            h.update(b);
        }
        QueryInput::Binary(b) => {
            h.update(b"B:");
            h.update(b);
        }
        QueryInput::Image(b) => {
            h.update(b"I:");
            h.update(b);
        }
        QueryInput::Audio(b) => {
            h.update(b"A:");
            h.update(b);
        }
        QueryInput::Multimodal { text, images, audio } => {
            h.update(b"M:");
            h.update(text.as_bytes());
            h.update(b"|i:");
            for img in images {
                h.update(&(img.len() as u64).to_le_bytes());
                h.update(img);
            }
            h.update(b"|a:");
            for clip in audio {
                h.update(&(clip.len() as u64).to_le_bytes());
                h.update(clip);
            }
        }
    }
    h.update(b"|C:");
    h.update(&q.context.history_fingerprint.0);
    h.finalize()
}

/// Convert an `Answer` to a `HistoryAnswer` for storage. Drops the
/// trace (which is per-call) and keeps the durable parts.
pub fn answer_to_history(a: &Answer) -> HistoryAnswer {
    HistoryAnswer {
        output: a.output.clone(),
        originating_tier: a.tier_used,
        joules_spent: a.joules_spent,
        confidence: a.confidence,
    }
}

/// Current Unix epoch seconds. Used for entry timestamps.
pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}
