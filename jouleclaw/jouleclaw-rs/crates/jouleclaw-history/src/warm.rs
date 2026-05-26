//! `WarmCache` — bounded LRU cache. Middle tier between Hot (small,
//! always-resident) and Cold (disk-backed).
//!
//! Invariants:
//!   - Capacity-bounded: never more than `max_entries`. On insert past
//!     capacity, the least-recently-used entry is evicted.
//!   - On lookup, an entry is "touched" to become MRU. Touching itself
//!     costs joules — the LRU update is the dominant overhead vs the
//!     Hot hashmap.
//!   - Evicted entries do NOT vanish: they're returned from `put()` so
//!     the caller can demote them to Cold.

use jouleclaw_cascade::*;
use std::collections::HashMap;

pub struct WarmCache {
    /// The actual entries keyed by EntryKey.
    entries: HashMap<EntryKey, HistoryEntry>,
    /// Access order, oldest first. The back is MRU.
    access_order: Vec<EntryKey>,
    max_entries: usize,
    stats: HistoryStats,
    c_lookup_base: f64,
    c_lookup_per_byte: f64,
    /// LRU bookkeeping is non-trivial — a vec shuffle. Cost per op.
    c_touch: f64,
}

impl WarmCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: Vec::with_capacity(max_entries),
            max_entries,
            stats: HistoryStats::default(),
            c_lookup_base: 1e-8,         // ~10 nJ
            c_lookup_per_byte: 5e-11,
            c_touch: 5e-9,               // ~5 nJ for LRU update
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn capacity(&self) -> usize { self.max_entries }

    /// Get a full entry by key (not just the answer). Used by tiered
    /// memory to promote entries to Hot.
    pub fn get_entry(&self, key: &EntryKey) -> Option<HistoryEntry> {
        self.entries.get(key).cloned()
    }

    /// Touch a key — move it to MRU position. Returns the joule cost
    /// of the touch.
    fn touch(&mut self, key: &EntryKey) -> f64 {
        // Remove the key from its current position, append to the end.
        // O(n) on the vec, but n ≤ max_entries (small for Warm tiers).
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
            self.access_order.push(*key);
        }
        self.c_touch
    }

    /// Insert an entry. Returns the joule cost of the insert and any
    /// evicted entry (None if capacity wasn't full).
    pub fn put(&mut self, entry: HistoryEntry) -> (f64, Option<HistoryEntry>) {
        let key = entry.key;
        let already_present = self.entries.contains_key(&key);
        let mut cost = self.c_lookup_base;

        if already_present {
            // Update value, touch.
            cost += self.touch(&key);
            self.entries.insert(key, entry);
            return (cost, None);
        }

        // New entry. If at capacity, evict oldest.
        let mut evicted = None;
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.access_order.first().copied() {
                self.access_order.remove(0);
                evicted = self.entries.remove(&oldest);
                cost += self.c_lookup_base;  // eviction cost
            }
        }

        self.entries.insert(key, entry);
        self.access_order.push(key);
        self.stats.writes += 1;
        self.stats.entry_count = self.entries.len();
        cost += self.c_touch;
        (cost, evicted)
    }
}

impl Default for WarmCache {
    fn default() -> Self { Self::new(128) }
}

impl HistoryLayer for WarmCache {
    fn lookup_exact(&mut self, key: &EntryKey) -> Result<Option<HistoryAnswer>, HistoryError> {
        self.stats.total_lookups += 1;
        let answer = self.entries.get(key).map(|e| e.answer.clone());
        if answer.is_some() {
            self.stats.hits += 1;
            self.touch(key);
        } else {
            self.stats.misses += 1;
        }
        Ok(answer)
    }

    fn record(&mut self, q: &Query, a: &Answer) -> Result<EntryKey, HistoryError> {
        let key = key_for(q);
        let entry = HistoryEntry {
            key,
            query_input: q.input.clone(),
            query_context: q.context,
            answer: answer_to_history(a),
            timestamp_secs: now_secs(),
            embedding: Vec::new(),
        };
        let (_cost, _evicted) = self.put(entry);
        self.stats.joules_recorded += a.joules_spent;
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
        self.c_lookup_base + self.c_lookup_per_byte * (len as f64) + self.c_touch
    }

    fn stats(&self) -> &HistoryStats { &self.stats }
}
