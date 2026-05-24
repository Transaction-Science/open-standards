//! Sliding-window conversational memory.
//!
//! Holds the last `N` turns of a conversation. The window evicts
//! oldest-first when its capacity is exceeded.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, MemoryResult};
use crate::memory::{Memory, MemoryItem, MemoryKind, MemoryRef};

/// One conversation turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    /// Speaker label ("user", "assistant", "tool:...").
    pub speaker: String,
    /// Natural-language content.
    pub text: String,
    /// Monotonic timestamp in ms.
    pub timestamp_ms: u64,
}

/// FIFO sliding-window memory of recent turns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlidingWindow {
    cap: usize,
    turns: VecDeque<Turn>,
}

impl SlidingWindow {
    /// New window with capacity `cap > 0`.
    pub fn new(cap: usize) -> MemoryResult<Self> {
        if cap == 0 {
            return Err(MemoryError::Config("window cap must be > 0".into()));
        }
        Ok(Self {
            cap,
            turns: VecDeque::with_capacity(cap),
        })
    }

    /// Push a new turn, evicting the oldest if at capacity.
    /// Returns the evicted turn if eviction happened.
    pub fn push(&mut self, turn: Turn) -> Option<Turn> {
        let evicted = if self.turns.len() == self.cap {
            self.turns.pop_front()
        } else {
            None
        };
        self.turns.push_back(turn);
        evicted
    }

    /// Borrow all in-window turns in chronological order.
    #[must_use]
    pub fn turns(&self) -> Vec<&Turn> {
        self.turns.iter().collect()
    }

    /// Window capacity.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }
}

impl Memory for SlidingWindow {
    fn kind(&self) -> MemoryKind {
        MemoryKind::Window
    }

    fn len(&self) -> usize {
        self.turns.len()
    }

    fn recent(&self, n: usize) -> MemoryResult<Vec<MemoryItem>> {
        let take = n.min(self.turns.len());
        Ok(self
            .turns
            .iter()
            .rev()
            .take(take)
            .map(|t| {
                MemoryItem::new(
                    MemoryRef {
                        kind: MemoryKind::Window,
                        id: format!("turn:{}", t.timestamp_ms),
                    },
                    format!("[{}] {}", t.speaker, t.text),
                    t.timestamp_ms,
                )
            })
            .collect())
    }
}
