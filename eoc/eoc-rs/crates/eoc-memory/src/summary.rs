//! Rolling summary buffer.
//!
//! When a conversation exceeds an in-context budget, older turns are
//! replaced by a natural-language summary. This module owns the
//! summary text plus the count of original turns it represents.
//!
//! The summarisation step itself is left to a caller-provided
//! closure so the crate stays dependency-light and deterministic
//! given the closure's output.

use serde::{Deserialize, Serialize};

use crate::error::MemoryResult;
use crate::memory::{Memory, MemoryItem, MemoryKind, MemoryRef};

/// Rolling summary of conversation history.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SummaryBuffer {
    summary: String,
    represented_turns: u32,
    updated_ms: u64,
}

impl SummaryBuffer {
    /// New empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the summary text with `new_summary`, incrementing the
    /// represented-turn count by `added_turns`.
    pub fn update(&mut self, new_summary: impl Into<String>, added_turns: u32, now_ms: u64) {
        self.summary = new_summary.into();
        self.represented_turns = self.represented_turns.saturating_add(added_turns);
        self.updated_ms = now_ms;
    }

    /// Merge another buffer into this one by re-summarising via the
    /// supplied closure.
    pub fn merge<F>(&mut self, other: &SummaryBuffer, now_ms: u64, summariser: F)
    where
        F: FnOnce(&str, &str) -> String,
    {
        let merged = summariser(&self.summary, &other.summary);
        self.summary = merged;
        self.represented_turns = self
            .represented_turns
            .saturating_add(other.represented_turns);
        self.updated_ms = now_ms;
    }

    /// Borrow the current summary text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.summary
    }

    /// Number of original turns this summary represents.
    #[must_use]
    pub fn represented_turns(&self) -> u32 {
        self.represented_turns
    }

    /// Timestamp (ms) of last update.
    #[must_use]
    pub fn updated_ms(&self) -> u64 {
        self.updated_ms
    }
}

impl Memory for SummaryBuffer {
    fn kind(&self) -> MemoryKind {
        MemoryKind::Summary
    }

    fn len(&self) -> usize {
        usize::from(!self.summary.is_empty())
    }

    fn recent(&self, _n: usize) -> MemoryResult<Vec<MemoryItem>> {
        if self.summary.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![MemoryItem::new(
            MemoryRef {
                kind: MemoryKind::Summary,
                id: "summary".into(),
            },
            self.summary.clone(),
            self.updated_ms,
        )])
    }
}
