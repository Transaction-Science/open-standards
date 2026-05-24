//! Working memory: a small scratchpad bounded by an attention budget.
//!
//! Working memory holds *active* items — what the agent is currently
//! "thinking about". It is bounded by a token cap (the in-context
//! attention budget) and evicts the lowest-salience item when the
//! cap would be exceeded.

use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, MemoryResult};
use crate::memory::{Memory, MemoryItem, MemoryKind, MemoryRef};

/// One slot in the scratchpad.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Slot {
    /// Free-form key (e.g. "current_goal", "tool_result").
    pub key: String,
    /// Natural-language content.
    pub value: String,
    /// Salience, `>= 0.0`. Higher = more important.
    pub salience: f32,
    /// Approximate token cost.
    pub tokens: u32,
    /// Insertion timestamp in ms.
    pub timestamp_ms: u64,
}

impl Slot {
    fn approx_tokens(s: &str) -> u32 {
        u32::try_from(s.split_whitespace().count()).unwrap_or(u32::MAX)
    }
}

/// Bounded scratchpad with attention budget.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scratchpad {
    slots: Vec<Slot>,
    used_tokens: u32,
    cap_tokens: u32,
}

impl Scratchpad {
    /// New scratchpad with `cap_tokens` attention budget.
    pub fn new(cap_tokens: u32) -> MemoryResult<Self> {
        if cap_tokens == 0 {
            return Err(MemoryError::Config(
                "scratchpad cap_tokens must be > 0".into(),
            ));
        }
        Ok(Self {
            slots: Vec::new(),
            used_tokens: 0,
            cap_tokens,
        })
    }

    /// Insert a new key/value. If the budget would be exceeded, the
    /// lowest-salience slots are evicted until it fits. If the new
    /// slot itself does not fit the empty budget, returns
    /// [`MemoryError::BudgetExhausted`].
    pub fn write(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
        salience: f32,
        timestamp_ms: u64,
    ) -> MemoryResult<()> {
        let key = key.into();
        let value = value.into();
        let tokens = Slot::approx_tokens(&value);
        if tokens > self.cap_tokens {
            return Err(MemoryError::BudgetExhausted {
                used: tokens,
                cap: self.cap_tokens,
            });
        }
        // Replace existing same-key slot first.
        if let Some(pos) = self.slots.iter().position(|s| s.key == key) {
            let old = self.slots.remove(pos);
            self.used_tokens = self.used_tokens.saturating_sub(old.tokens);
        }
        while self.used_tokens + tokens > self.cap_tokens {
            // Evict the lowest-salience, oldest slot.
            let Some(victim_idx) = self.lowest_salience_idx() else {
                return Err(MemoryError::BudgetExhausted {
                    used: self.used_tokens + tokens,
                    cap: self.cap_tokens,
                });
            };
            let victim = self.slots.remove(victim_idx);
            self.used_tokens = self.used_tokens.saturating_sub(victim.tokens);
        }
        self.slots.push(Slot {
            key,
            value,
            salience,
            tokens,
            timestamp_ms,
        });
        self.used_tokens += tokens;
        Ok(())
    }

    fn lowest_salience_idx(&self) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.salience
                    .partial_cmp(&b.salience)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.timestamp_ms.cmp(&b.timestamp_ms))
            })
            .map(|(i, _)| i)
    }

    /// Read a slot by key.
    #[must_use]
    pub fn read(&self, key: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.key == key)
    }

    /// Tokens currently held.
    #[must_use]
    pub fn used_tokens(&self) -> u32 {
        self.used_tokens
    }

    /// Total budget.
    #[must_use]
    pub fn cap_tokens(&self) -> u32 {
        self.cap_tokens
    }

    /// All slots (for inspection).
    #[must_use]
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }
}

impl Memory for Scratchpad {
    fn kind(&self) -> MemoryKind {
        MemoryKind::Working
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn recent(&self, n: usize) -> MemoryResult<Vec<MemoryItem>> {
        let mut sorted: Vec<&Slot> = self.slots.iter().collect();
        sorted.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
        Ok(sorted
            .into_iter()
            .take(n)
            .map(|s| {
                MemoryItem::new(
                    MemoryRef {
                        kind: MemoryKind::Working,
                        id: s.key.clone(),
                    },
                    format!("{}: {}", s.key, s.value),
                    s.timestamp_ms,
                )
            })
            .collect())
    }
}
