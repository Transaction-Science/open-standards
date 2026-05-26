//! L0 — exact-match cache.
//!
//! The first tier. Hash of (normalized query + context fingerprint) →
//! stored `Answer`. Hit returns the cached answer in microseconds at a
//! ~picojoule cost. Miss falls through to subsequent tiers.
//!
//! For R1 the backing store is an in-memory `HashMap`. R3 introduces a
//! durable `HistoryLayer` and L0 becomes the hot cache over it.
//!
//! Cache key design:
//!   key = Hasher256(normalize(input) || context_fingerprint)
//! The normalization is currently identity for text (trim/lowercase
//! later if measurements show it helps the hit rate). Structured and
//! binary inputs hash their bytes directly.

use crate::tier::{Tier, TierEstimate};
use crate::types::*;
use jouleclaw_core::hash::Hasher256;
use std::collections::HashMap;
use std::time::Duration;

/// L0 — exact-match cache, in-memory.
///
/// Energy model:
///   estimate.joules = C_lookup + C_hash * input_len
///   actual.joules   = same (deterministic — no neural inference)
///
/// The constants are calibrated for typical hardware. Order of
/// magnitude is what matters here: lookups are picojoules to nanojoules,
/// not joules.
pub struct L0Cache {
    entries: HashMap<[u8; 32], CachedAnswer>,
    stats: L0Stats,
    /// Joule cost per cache lookup, independent of input size. ~1 ns
    /// hash setup + ~5 ns HashMap lookup ≈ 6 nJ on a typical CPU at
    /// 100 W and 1 GHz effective throughput. Conservative.
    c_lookup: f64,
    /// Joule cost per input byte hashed. FNV-1a is ~0.5 ns/byte;
    /// at 100 W that's 0.05 nJ/byte = 5e-11 J/byte.
    c_hash_per_byte: f64,
}

#[derive(Debug, Clone)]
struct CachedAnswer {
    output: AnswerOutput,
    confidence: f32,
    /// The tier that originally produced this answer. Will be surfaced
    /// in the trace by future phases (R4+ router uses this to short-
    /// circuit re-routing for cached answers).
    #[allow(dead_code)]
    original_tier: TierId,
}

#[derive(Debug, Clone, Default)]
pub struct L0Stats {
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
}

impl L0Cache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            stats: L0Stats::default(),
            c_lookup: 6e-9,           // ~6 nJ per lookup
            c_hash_per_byte: 5e-11,   // ~50 pJ per byte hashed
        }
    }

    pub fn stats(&self) -> &L0Stats {
        &self.stats
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Record an answer in the cache for the given query. The next
    /// identical query (same input, same context fingerprint) hits.
    /// Idempotent: writing the same key twice is fine.
    pub fn put(&mut self, q: &Query, a: &Answer) {
        let key = Self::key_for(q);
        let entry = CachedAnswer {
            output: a.output.clone(),
            confidence: a.confidence,
            original_tier: a.tier_used,
        };
        let already_present = self.entries.contains_key(&key);
        self.entries.insert(key, entry);
        if !already_present {
            self.stats.writes += 1;
        }
    }

    /// Compute the cache key for a query.
    pub fn key_for(q: &Query) -> [u8; 32] {
        let mut h = Hasher256::new();
        // Domain separation tag.
        h.update(b"L0v1");
        // Input variant tag + payload.
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
        // Context fingerprint — same query in different sessions can
        // produce different answers, so the key includes the session
        // context. A "fresh" context (all zeros) collapses all
        // sessionless queries to the same key.
        h.update(b"|C:");
        h.update(&q.context.history_fingerprint.0);
        h.finalize()
    }

    /// Estimate the joule cost of a lookup. Cheap by construction.
    fn estimate_joules(&self, q: &Query) -> f64 {
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
}

impl Default for L0Cache {
    fn default() -> Self { Self::new() }
}

impl Tier for L0Cache {
    fn id(&self) -> TierId {
        TierId::L0
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // L0 is universally applicable (any query can in principle be
        // cached) but its confidence is 1.0 on hit and undefined on
        // miss. We claim confidence_floor = 1.0: if the tier returns
        // an answer at all, that answer is the previously-recorded
        // one, exactly. On miss it refuses, and refusal doesn't
        // produce an answer.
        Some(TierEstimate {
            joules: self.estimate_joules(q),
            latency: Duration::from_nanos(50),
            confidence_floor: 1.0,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let key = Self::key_for(q);
        let joules = self.estimate_joules(q);
        match self.entries.get(&key) {
            Some(cached) => {
                self.stats.hits += 1;
                let mut trace = ExecutionTrace::default();
                trace.attempts.push(TraceEntry {
                    tier: TierId::L0,
                    outcome: TraceOutcome::Hit,
                    joules,
                });
                Ok(Answer {
                    output: cached.output.clone(),
                    tier_used: TierId::L0,
                    joules_spent: joules,
                    confidence: cached.confidence,
                    trace,
                    verification: crate::verification::VerificationStatus::Resolved,
                })
            }
            None => {
                self.stats.misses += 1;
                // Miss: refuse with Inapplicable. The runtime moves to
                // the next tier.
                Ok(Answer {
                    output: AnswerOutput::Refused(RefusalReason::Inapplicable),
                    tier_used: TierId::L0,
                    joules_spent: joules,
                    confidence: 0.0,
                    trace: ExecutionTrace::default(),
                    verification: crate::verification::VerificationStatus::Resolved,
                })
            }
        }
    }

    fn coord(&self) -> Option<crate::coord::Coord> {
        Some(crate::coord::prebuilt::l0_cache())
    }

    fn cost_estimate(&self, q: &Query) -> Option<crate::cost::CostEstimate> {
        Some(crate::cost::CostEstimate::flat(
            self.estimate_joules(q),
            Duration::from_nanos(50),
            1.0,
        ))
    }
}
