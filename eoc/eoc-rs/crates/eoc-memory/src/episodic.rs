//! Episodic memory: a monotonic event log with a temporal index.
//!
//! Episodes are the "raw tape" of agentic interaction (user turns,
//! tool calls, observations). They are stamped with a monotonic
//! wall-clock millisecond and indexed for `[t0, t1)` range queries.

use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, MemoryResult};
use crate::memory::{EpisodeId, Memory, MemoryItem, MemoryKind, MemoryRef};

/// A single time-stamped event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    /// Stable id derived from `(timestamp_ms, payload)`.
    pub id: EpisodeId,
    /// Monotonic wall-clock time in ms.
    pub timestamp_ms: u64,
    /// Free-form actor label ("user", "assistant", "tool:web").
    pub actor: String,
    /// Natural-language payload of the event.
    pub payload: String,
    /// Optional embedding for similarity-based retrieval.
    pub embedding: Option<Vec<f32>>,
    /// Number of times this episode has been re-read by the retriever.
    pub access_count: u32,
}

impl Episode {
    /// Construct a new episode, computing its id.
    #[must_use]
    pub fn new(timestamp_ms: u64, actor: impl Into<String>, payload: impl Into<String>) -> Self {
        let payload = payload.into();
        let actor = actor.into();
        let id = EpisodeId::from_event(timestamp_ms, payload.as_bytes());
        Self {
            id,
            timestamp_ms,
            actor,
            payload,
            embedding: None,
            access_count: 0,
        }
    }

    /// Attach an embedding to this episode (builder-style).
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Render as a `MemoryItem` (for injection / retrieval).
    #[must_use]
    pub fn as_item(&self) -> MemoryItem {
        MemoryItem::new(
            MemoryRef {
                kind: MemoryKind::Episodic,
                id: self.id.to_hex(),
            },
            format!("[{}] {}", self.actor, self.payload),
            self.timestamp_ms,
        )
    }
}

/// Simple sorted temporal index over episode timestamps.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimeIndex {
    /// `(timestamp_ms, position_in_log)` sorted by `timestamp_ms`.
    entries: Vec<(u64, usize)>,
}

impl TimeIndex {
    /// Insert a new `(timestamp_ms, position)` pair. The index must
    /// be appended-to monotonically.
    pub fn insert(&mut self, timestamp_ms: u64, position: usize) -> MemoryResult<()> {
        if let Some(&(last_ms, _)) = self.entries.last() {
            if timestamp_ms < last_ms {
                return Err(MemoryError::NonMonotonic {
                    got_ms: timestamp_ms,
                    last_ms,
                });
            }
        }
        self.entries.push((timestamp_ms, position));
        Ok(())
    }

    /// Positions of episodes whose timestamp falls into `[t0, t1)`.
    #[must_use]
    pub fn range(&self, t0: u64, t1: u64) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|(ms, _)| *ms >= t0 && *ms < t1)
            .map(|(_, idx)| *idx)
            .collect()
    }

    /// Total number of indexed entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the index has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Append-only episodic event log with a [`TimeIndex`].
#[derive(Clone, Debug, Default)]
pub struct EpisodicLog {
    log: Vec<Episode>,
    index: TimeIndex,
}

impl EpisodicLog {
    /// Fresh empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new episode. Returns its id on success.
    pub fn append(&mut self, episode: Episode) -> MemoryResult<EpisodeId> {
        let position = self.log.len();
        self.index.insert(episode.timestamp_ms, position)?;
        let id = episode.id;
        self.log.push(episode);
        Ok(id)
    }

    /// Episodes in the half-open interval `[t0, t1)`, in chronological
    /// order.
    #[must_use]
    pub fn range(&self, t0: u64, t1: u64) -> Vec<&Episode> {
        self.index
            .range(t0, t1)
            .into_iter()
            .filter_map(|i| self.log.get(i))
            .collect()
    }

    /// Borrow a single episode by id.
    #[must_use]
    pub fn get(&self, id: &EpisodeId) -> Option<&Episode> {
        self.log.iter().find(|e| &e.id == id)
    }

    /// Mutable borrow of a single episode (used by the retriever to
    /// bump `access_count`).
    pub fn get_mut(&mut self, id: &EpisodeId) -> Option<&mut Episode> {
        self.log.iter_mut().find(|e| &e.id == id)
    }

    /// All episodes in insertion (chronological) order.
    #[must_use]
    pub fn all(&self) -> &[Episode] {
        &self.log
    }

    /// Borrow the temporal index.
    #[must_use]
    pub fn index(&self) -> &TimeIndex {
        &self.index
    }
}

impl Memory for EpisodicLog {
    fn kind(&self) -> MemoryKind {
        MemoryKind::Episodic
    }

    fn len(&self) -> usize {
        self.log.len()
    }

    fn recent(&self, n: usize) -> MemoryResult<Vec<MemoryItem>> {
        let take = n.min(self.log.len());
        Ok(self
            .log
            .iter()
            .rev()
            .take(take)
            .map(Episode::as_item)
            .collect())
    }
}
