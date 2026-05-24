//! MemGPT-style hierarchical memory.
//!
//! Packer et al. 2023 ("MemGPT: Towards LLMs as Operating Systems")
//! split agent memory into three tiers:
//!
//! 1. **Main context** — always-resident; small, hot, in-prompt.
//! 2. **Recall** — recent overflow; paged in by reference.
//! 3. **Archival** — cold, large, retrievable by query.
//!
//! This module ties together [`Scratchpad`], [`SlidingWindow`] and
//! [`EpisodicLog`] into that three-tier shape.

use serde::{Deserialize, Serialize};

use crate::episodic::{Episode, EpisodicLog};
use crate::error::MemoryResult;
use crate::memory::EpisodeId;
use crate::window::{SlidingWindow, Turn};
use crate::working::Scratchpad;

/// MemGPT tier discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    /// In-prompt working set.
    Main,
    /// Recent overflow.
    Recall,
    /// Cold long-term store.
    Archival,
}

/// Configuration knobs for a [`MemGpt`] instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemGptConfig {
    /// Token budget for the main context scratchpad.
    pub main_tokens: u32,
    /// Turn capacity of the recall sliding window.
    pub recall_turns: usize,
}

impl Default for MemGptConfig {
    fn default() -> Self {
        Self {
            main_tokens: 2048,
            recall_turns: 32,
        }
    }
}

/// Three-tier hierarchical memory bundle.
pub struct MemGpt {
    /// Main-context scratchpad.
    pub main: Scratchpad,
    /// Recall sliding window.
    pub recall: SlidingWindow,
    /// Archival episodic log.
    pub archival: EpisodicLog,
}

impl MemGpt {
    /// Build a new MemGpt with the supplied config.
    pub fn new(cfg: MemGptConfig) -> MemoryResult<Self> {
        Ok(Self {
            main: Scratchpad::new(cfg.main_tokens)?,
            recall: SlidingWindow::new(cfg.recall_turns)?,
            archival: EpisodicLog::new(),
        })
    }

    /// Record a fresh conversation turn:
    ///   * `recall` gains the turn (oldest evicted on overflow).
    ///   * The evicted-out turn (if any) is appended to `archival`.
    pub fn observe(&mut self, turn: Turn) -> MemoryResult<Option<EpisodeId>> {
        let actor = turn.speaker.clone();
        let text = turn.text.clone();
        let ts = turn.timestamp_ms;
        let evicted = self.recall.push(turn);
        if let Some(old) = evicted {
            let ep = Episode::new(old.timestamp_ms, old.speaker, old.text);
            let id = self.archival.append(ep)?;
            // The just-pushed turn metadata is also retained for the
            // caller to inspect.
            let _ = (actor, text, ts);
            return Ok(Some(id));
        }
        Ok(None)
    }

    /// Manually promote a turn into the archival log (e.g. when the
    /// agent calls `archive`).
    pub fn archive(
        &mut self,
        speaker: impl Into<String>,
        text: impl Into<String>,
        timestamp_ms: u64,
    ) -> MemoryResult<EpisodeId> {
        self.archival
            .append(Episode::new(timestamp_ms, speaker, text))
    }

    /// Which tier currently holds an episode id, if any.
    #[must_use]
    pub fn locate(&self, id: &EpisodeId) -> Option<Tier> {
        if self.archival.get(id).is_some() {
            return Some(Tier::Archival);
        }
        None
    }

    /// Lengths of each tier, useful for instrumentation.
    #[must_use]
    pub fn sizes(&self) -> (usize, usize, usize) {
        (
            self.main.slots().len(),
            self.recall.turns().len(),
            self.archival.all().len(),
        )
    }
}
