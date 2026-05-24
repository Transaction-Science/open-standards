//! Base [`Memory`] trait and shared identifier / item types.

use serde::{Deserialize, Serialize};

use crate::error::MemoryResult;

/// Stable identifier for a single episodic event.
///
/// Constructed from a BLAKE3 digest over `(timestamp_ms, payload)`
/// so identical events at identical timestamps collapse to the same
/// id — preserving determinism.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EpisodeId(pub [u8; 16]);

impl EpisodeId {
    /// Build an [`EpisodeId`] from a timestamp + payload bytes.
    #[must_use]
    pub fn from_event(timestamp_ms: u64, payload: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&timestamp_ms.to_le_bytes());
        hasher.update(payload);
        let h = hasher.finalize();
        let mut out = [0u8; 16];
        out.copy_from_slice(&h.as_bytes()[..16]);
        Self(out)
    }

    /// Lowercase hex form of the id (32 chars).
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(32);
        for b in &self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

/// Taxonomy of memory subsystems implemented by this crate.
///
/// The classification roughly follows Tulving (1972) — episodic vs
/// semantic — plus working / procedural / summary as engineering
/// niceties for LLM agents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind {
    /// Time-stamped events (conversations, observations).
    Episodic,
    /// Distilled facts as entity-relation triples.
    Semantic,
    /// Volatile scratchpad held inside the model's attention budget.
    Working,
    /// Rolling natural-language summary of older context.
    Summary,
    /// Sliding window of recent turns.
    Window,
}

/// A reference to a single memory record, used by retrieval and
/// injection routines.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryRef {
    /// Which subsystem produced this item.
    pub kind: MemoryKind,
    /// Stable opaque id (BLAKE3-16 hex) of the underlying record.
    pub id: String,
}

/// A retrieved memory item carrying enough text to be injected into
/// a prompt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Stable reference to the source record.
    pub reference: MemoryRef,
    /// Rendered natural-language text.
    pub text: String,
    /// Approximate token count (`text.split_whitespace().count()`).
    pub tokens: u32,
    /// Monotonic timestamp of the underlying event in ms.
    pub timestamp_ms: u64,
}

impl MemoryItem {
    /// Construct a new item; tokens are computed by whitespace count.
    #[must_use]
    pub fn new(reference: MemoryRef, text: String, timestamp_ms: u64) -> Self {
        let tokens = u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX);
        Self {
            reference,
            text,
            tokens,
            timestamp_ms,
        }
    }
}

/// Common contract for every memory subsystem.
///
/// Implementations must be deterministic given a fixed clock; that
/// is, calling [`Memory::recent`] with the same arguments after the
/// same sequence of writes must yield the same items in the same
/// order.
pub trait Memory {
    /// Identify which subsystem this implementation is.
    fn kind(&self) -> MemoryKind;

    /// Total number of records currently held.
    fn len(&self) -> usize;

    /// True iff `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Most-recent `n` items (newest first).
    ///
    /// Returns an error only if the store itself is broken.
    fn recent(&self, n: usize) -> MemoryResult<Vec<MemoryItem>>;
}
