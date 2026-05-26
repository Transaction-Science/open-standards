//! `InMemoryHistory` — a HashMap-backed `HistoryLayer`.
//!
//! Same semantics as the previous in-place `L0Cache`, lifted to the
//! trait. Lookup is O(1), record is O(1), no durability. Useful when:
//!   - the deployment has no persistent storage
//!   - a test wants determinism without filesystem effects
//!   - the workload is short-lived (one-shot CLI invocation)

use jouleclaw_cascade::*;
use std::collections::HashMap;
use std::time::Duration;

pub struct InMemoryHistory {
    entries: HashMap<EntryKey, HistoryEntry>,
    stats: HistoryStats,
    /// Joule cost baseline per lookup.
    c_lookup: f64,
    /// Joule cost per byte of input hashed.
    c_hash_per_byte: f64,
}

impl InMemoryHistory {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            stats: HistoryStats::default(),
            c_lookup: 6e-9,
            c_hash_per_byte: 5e-11,
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Borrow all entries (for diagnostic tools, R10 benchmarks).
    pub fn entries(&self) -> impl Iterator<Item = &HistoryEntry> {
        self.entries.values()
    }

    /// Remove an entry by key, returning it if present. Used by tiered
    /// memory to demote entries to a colder layer.
    pub fn remove(&mut self, key: &EntryKey) -> Option<HistoryEntry> {
        let evicted = self.entries.remove(key);
        if evicted.is_some() && self.stats.entry_count > 0 {
            self.stats.entry_count -= 1;
        }
        evicted
    }

    /// Directly insert a fully-formed entry without going through the
    /// query-record path. Used by tiered memory to promote entries
    /// from a colder layer.
    pub fn insert_entry(&mut self, entry: HistoryEntry) {
        let was_present = self.entries.contains_key(&entry.key);
        self.entries.insert(entry.key, entry);
        if !was_present {
            self.stats.writes += 1;
            self.stats.entry_count = self.entries.len();
        }
    }
}

impl crate::semantic::IndexedHistory for InMemoryHistory {
    fn iter_entries(&self) -> Box<dyn Iterator<Item = HistoryEntry> + '_> {
        Box::new(self.entries.values().cloned())
    }

    fn set_embedding(&mut self, key: &EntryKey, embedding: Vec<f32>)
        -> Result<(), HistoryError>
    {
        if let Some(e) = self.entries.get_mut(key) {
            e.embedding = embedding;
        }
        Ok(())
    }
}

impl Default for InMemoryHistory {
    fn default() -> Self { Self::new() }
}

impl HistoryLayer for InMemoryHistory {
    fn lookup_exact(&mut self, key: &EntryKey) -> Result<Option<HistoryAnswer>, HistoryError> {
        self.stats.total_lookups += 1;
        match self.entries.get(key) {
            Some(e) => {
                self.stats.hits += 1;
                Ok(Some(e.answer.clone()))
            }
            None => {
                self.stats.misses += 1;
                Ok(None)
            }
        }
    }

    fn record(&mut self, q: &Query, a: &Answer) -> Result<EntryKey, HistoryError> {
        let key = key_for(q);
        let already_present = self.entries.contains_key(&key);
        let entry = HistoryEntry {
            key,
            query_input: q.input.clone(),
            query_context: q.context,
            answer: answer_to_history(a),
            timestamp_secs: now_secs(),
            embedding: Vec::new(),
        };
        self.stats.joules_recorded += a.joules_spent;
        if !already_present {
            self.stats.writes += 1;
            self.stats.entry_count += 1;
        }
        self.entries.insert(key, entry);
        Ok(key)
    }

    fn estimate_lookup_cost(&self, q: &Query) -> f64 {
        let len = match &q.input {
            QueryInput::Text(s) => s.len(),
            QueryInput::Structured(b) | QueryInput::Binary(b) => b.len(),
            QueryInput::Image(b) | QueryInput::Audio(b) => b.len(),
            QueryInput::Multimodal { text, images, audio } => {
                text.len()
                    + images.iter().map(|v| v.len()).sum::<usize>()
                    + audio.iter().map(|v| v.len()).sum::<usize>()
            }
        };
        self.c_lookup + self.c_hash_per_byte * (len as f64)
    }

    fn stats(&self) -> &HistoryStats {
        &self.stats
    }
}

/// A `Tier` adapter over a `HistoryLayer`. This is what becomes the new
/// L0Cache: an L0Cache is really just "expose the history layer as a
/// tier."
pub struct L0Tier<H: HistoryLayer> {
    pub history: H,
}

impl<H: HistoryLayer> L0Tier<H> {
    pub fn new(history: H) -> Self {
        Self { history }
    }

    pub fn into_inner(self) -> H { self.history }
}

impl<H: HistoryLayer + 'static> Tier for L0Tier<H> {
    fn id(&self) -> TierId { TierId::L0 }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        Some(TierEstimate {
            joules: self.history.estimate_lookup_cost(q),
            latency: Duration::from_nanos(100),
            confidence_floor: 1.0,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget: f64,
    ) -> Result<Answer, AnswerError> {
        let key = key_for(q);
        let cost = self.history.estimate_lookup_cost(q);
        let lookup = self.history.lookup_exact(&key)
            .map_err(|e| AnswerError::TierFailed {
                tier: TierId::L0,
                cause: format!("history: {}", e),
            })?;
        match lookup {
            Some(ha) => {
                let mut trace = ExecutionTrace::default();
                trace.attempts.push(TraceEntry {
                    tier: TierId::L0,
                    outcome: TraceOutcome::Hit,
                    joules: cost,
                });
                Ok(Answer {
                    output: ha.output,
                    tier_used: TierId::L0,
                    joules_spent: cost,
                    confidence: ha.confidence,
                    trace,
                    verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                })
            }
            None => {
                let mut trace = ExecutionTrace::default();
                trace.attempts.push(TraceEntry {
                    tier: TierId::L0,
                    outcome: TraceOutcome::Refused(RefusalReason::Inapplicable),
                    joules: cost,
                });
                Ok(Answer {
                    output: AnswerOutput::Refused(RefusalReason::Inapplicable),
                    tier_used: TierId::L0,
                    joules_spent: cost,
                    confidence: 0.0,
                    trace,
                    verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                })
            }
        }
    }
}
