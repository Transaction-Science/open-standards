//! `TieredMemory` — composite memory hierarchy.
//!
//! Hot (bounded HashMap) → Warm (bounded LRU) → Cold (disk-backed).
//!
//! Access pattern:
//!   1. Lookup queries Hot first, then Warm, then Cold.
//!   2. A Warm hit promotes the entry back to Hot (LRU-style heating).
//!   3. A Cold hit promotes the entry through Warm to Hot (two-step).
//!   4. Hot eviction (capacity pressure) demotes to Warm.
//!   5. Warm eviction (capacity pressure) demotes to Cold.
//!
//! Joule cost per transition is recorded and returned in the lookup
//! result. This makes the cost of memory motion observable —
//! consistent with the "joule-priced everything" principle.
//!
//! `TieredMemory` implements `HistoryLayer` so it drops into the
//! existing `Runtime::new_with_history()` slot. The transitions are
//! invisible to the cascade walker except via the trace.

use crate::{InMemoryHistory, WarmCache, DiskHistory};
use jouleclaw_cascade::*;

pub struct TieredMemory {
    /// The smallest, fastest tier — bounded HashMap. Always in-process,
    /// always RAM. ~6 nJ per lookup.
    hot: InMemoryHistory,
    hot_capacity: usize,
    /// Bounded LRU. ~10-20 nJ per lookup. Holds items that recently
    /// fell out of Hot.
    warm: WarmCache,
    /// Disk-backed durable layer. ~1 µJ per lookup. Holds all entries
    /// that have ever been recorded.
    cold: DiskHistory,

    /// Monotonic sequence number assigned to entries on insertion into
    /// Hot. Used to pick the LRU candidate for demotion deterministically
    /// (timestamp_secs has only second-resolution; with rapid inserts,
    /// many entries share the same value).
    hot_lru: std::collections::HashMap<EntryKey, u64>,
    next_seq: u64,

    stats: HistoryStats,
    /// Where the most recent lookup hit. Useful for tests and traces.
    pub last_hit_tier: Option<MemoryTier>,
    /// Promotion/demotion cost accumulator.
    pub total_transition_joules: f64,
}

impl TieredMemory {
    /// Build a tiered memory with `hot_capacity` items in Hot,
    /// `warm_capacity` in Warm, unbounded in Cold (on disk at
    /// `cold_path`).
    pub fn open(
        hot_capacity: usize,
        warm_capacity: usize,
        cold_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, HistoryError> {
        Ok(Self {
            hot: InMemoryHistory::new(),
            hot_capacity,
            warm: WarmCache::new(warm_capacity),
            cold: DiskHistory::open(cold_path)?,
            hot_lru: std::collections::HashMap::new(),
            next_seq: 0,
            stats: HistoryStats::default(),
            last_hit_tier: None,
            total_transition_joules: 0.0,
        })
    }

    pub fn hot_len(&self) -> usize { self.hot.len() }
    pub fn warm_len(&self) -> usize { self.warm.len() }
    pub fn cold_len(&self) -> usize { self.cold.len() }

    /// Pretty-print the contents-by-tier for tests and demos.
    pub fn tier_sizes(&self) -> (usize, usize, usize) {
        (self.hot.len(), self.warm.len(), self.cold.len())
    }

    /// Cost of promoting a single entry from one tier up to another.
    fn promotion_cost(&self, from: MemoryTier, to: MemoryTier) -> f64 {
        // Cold → Warm: disk read + warm insert. ~1.1 µJ.
        // Warm → Hot:  hashmap insert. ~10 nJ.
        // Cold → Hot:  double promotion.
        match (from, to) {
            (MemoryTier::Cold, MemoryTier::Warm)
            | (MemoryTier::Cold, MemoryTier::Hot) => 1e-6,
            (MemoryTier::Warm, MemoryTier::Hot) => 1e-8,
            _ => 0.0,
        }
    }

    /// Cost of demoting a single entry. Demotion happens on eviction
    /// pressure.
    fn demotion_cost(&self, from: MemoryTier, to: MemoryTier) -> f64 {
        match (from, to) {
            (MemoryTier::Hot, MemoryTier::Warm) => 1e-8,
            (MemoryTier::Warm, MemoryTier::Cold) => 1e-6,   // disk write
            _ => 0.0,
        }
    }

    /// Insert an entry into Hot, demoting whatever falls out (if any)
    /// down through the hierarchy. Returns the total joule cost of
    /// any demotions performed.
    fn insert_hot_with_demotion(&mut self, entry: HistoryEntry) -> f64 {
        let mut demotion_cost = 0.0;

        // If Hot is at capacity, evict the LRU entry (lowest seq).
        if self.hot.len() >= self.hot_capacity {
            let evict_key = self.hot_lru.iter()
                .min_by_key(|(_, seq)| *seq)
                .map(|(k, _)| *k);
            if let Some(k) = evict_key {
                self.hot_lru.remove(&k);
                if let Some(evicted) = self.hot.remove(&k) {
                    demotion_cost += self.demotion_cost(MemoryTier::Hot, MemoryTier::Warm);
                    let (_, warm_evicted) = self.warm.put(evicted);
                    if let Some(we) = warm_evicted {
                        demotion_cost += self.demotion_cost(MemoryTier::Warm, MemoryTier::Cold);
                        if let Ok(None) = self.cold.lookup_exact(&we.key) {
                            let q = Query {
                                input: we.query_input.clone(),
                                budget: JouleBudget::standard(),
                                quality: QualityFloor::any(),
                                context: we.query_context,
                                deadline: None,
                            };
                            let a = Answer {
                                output: we.answer.output.clone(),
                                tier_used: we.answer.originating_tier,
                                joules_spent: we.answer.joules_spent,
                                confidence: we.answer.confidence,
                                trace: ExecutionTrace::default(),
                                verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                            };
                            let _ = self.cold.record(&q, &a);
                        }
                    }
                }
            }
        }

        // Assign new seq for the inserted entry and stamp it as MRU.
        let key = entry.key;
        self.hot.insert_entry(entry);
        self.next_seq += 1;
        self.hot_lru.insert(key, self.next_seq);

        demotion_cost
    }

    /// Touch a hot key — bump its LRU seq.
    fn touch_hot(&mut self, key: &EntryKey) {
        if self.hot_lru.contains_key(key) {
            self.next_seq += 1;
            self.hot_lru.insert(*key, self.next_seq);
        }
    }
}

impl HistoryLayer for TieredMemory {
    fn lookup_exact(&mut self, key: &EntryKey) -> Result<Option<HistoryAnswer>, HistoryError> {
        self.stats.total_lookups += 1;

        // Try Hot.
        if let Some(answer) = self.hot.lookup_exact(key)? {
            self.last_hit_tier = Some(MemoryTier::Hot);
            self.stats.hits += 1;
            self.touch_hot(key);
            return Ok(Some(answer));
        }

        // Try Warm. On hit, promote to Hot.
        if let Some(answer) = self.warm.lookup_exact(key)? {
            self.last_hit_tier = Some(MemoryTier::Warm);
            self.stats.hits += 1;
            // Promote: copy the warm entry up to Hot.
            let warm_entry = self.warm.get_entry(key);
            if let Some(e) = warm_entry {
                let promote_cost = self.promotion_cost(MemoryTier::Warm, MemoryTier::Hot);
                self.total_transition_joules += promote_cost;
                self.total_transition_joules += self.insert_hot_with_demotion(e);
            }
            return Ok(Some(answer));
        }

        // Try Cold. On hit, promote through Warm to Hot.
        if let Some(answer) = self.cold.lookup_exact(key)? {
            self.last_hit_tier = Some(MemoryTier::Cold);
            self.stats.hits += 1;
            // Reconstruct an entry from Cold and promote it. Collect
            // first to release the borrow on self.cold.
            let cold_entry = self.cold.entries()
                .find(|e| &e.key == key).cloned();
            if let Some(e) = cold_entry {
                let promote_cost = self.promotion_cost(MemoryTier::Cold, MemoryTier::Hot);
                self.total_transition_joules += promote_cost;
                self.total_transition_joules += self.insert_hot_with_demotion(e);
            }
            return Ok(Some(answer));
        }

        self.last_hit_tier = None;
        self.stats.misses += 1;
        Ok(None)
    }

    fn record(&mut self, q: &Query, a: &Answer) -> Result<EntryKey, HistoryError> {
        // Always write to Cold (the durable layer) AND Hot.
        let key = self.cold.record(q, a)?;
        let entry = HistoryEntry {
            key,
            query_input: q.input.clone(),
            query_context: q.context,
            answer: answer_to_history(a),
            timestamp_secs: now_secs(),
            embedding: Vec::new(),
        };
        self.total_transition_joules += self.insert_hot_with_demotion(entry);
        self.stats.joules_recorded += a.joules_spent;
        self.stats.writes += 1;
        self.stats.entry_count = self.hot.len() + self.warm.len() + self.cold.len();
        Ok(key)
    }

    fn estimate_lookup_cost(&self, q: &Query) -> f64 {
        // Upper bound: lookup-cost-of-all-tiers if everything misses.
        self.hot.estimate_lookup_cost(q)
            + self.warm.estimate_lookup_cost(q)
            + self.cold.estimate_lookup_cost(q)
    }

    fn stats(&self) -> &HistoryStats { &self.stats }
}
