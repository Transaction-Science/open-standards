//! Phasor-fingerprint router.
//!
//! A "phasor" here is a fixed-width bit fingerprint of a query. The
//! default [`HashPhasorEmbedder`] derives it deterministically from the
//! token set via FNV-1a so the crate needs no model. A consumer with
//! real phasor embeddings (e.g. from JouleDB) implements
//! [`PhasorEmbedder`] and plugs it in; routing then votes by Hamming
//! proximity over learned embeddings instead of token overlap.

use jouleclaw_cascade::router::{Router, RoutingPlan};
use jouleclaw_cascade::types::{Query, TierId};
use std::collections::HashMap;

/// A 64-bit query fingerprint. Hamming distance approximates query
/// dissimilarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Phasor(pub u64);

impl Phasor {
    /// Number of differing bits between two phasors (0..=64).
    pub fn hamming(self, other: Phasor) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Similarity in `[0, 64]` — bits in common.
    pub fn similarity(self, other: Phasor) -> u32 {
        64 - self.hamming(other)
    }
}

/// Maps a query to a [`Phasor`]. Implement this to plug in real
/// embeddings; the default is a deterministic hash.
pub trait PhasorEmbedder: Send + Sync {
    fn embed(&self, q: &Query) -> Phasor;
}

/// Default deterministic embedder: FNV-1a over the sorted token set,
/// folded so each token sets a few bits. No model, no allocation beyond
/// the token split.
#[derive(Debug, Default, Clone, Copy)]
pub struct HashPhasorEmbedder;

impl PhasorEmbedder for HashPhasorEmbedder {
    fn embed(&self, q: &Query) -> Phasor {
        let mut tokens = crate::learned::tokenize(q);
        tokens.sort();
        let mut bits: u64 = 0;
        for tok in &tokens {
            // FNV-1a 64.
            let mut h: u64 = 0xcbf29ce484222325;
            for b in tok.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            // Set the bit the hash points at, plus a fold for density.
            bits |= 1u64 << (h % 64);
            bits |= 1u64 << ((h >> 6) % 64);
        }
        Phasor(bits)
    }
}

#[derive(Clone)]
struct PhasorEpisode {
    phasor: Phasor,
    winning_tier: TierId,
    joules_spent: f64,
}

/// Router that orders tiers by the winning tier of phasor-nearest past
/// queries.
pub struct PhasorRouter<E: PhasorEmbedder = HashPhasorEmbedder> {
    embedder: E,
    episodes: Vec<PhasorEpisode>,
    capacity: usize,
    k: usize,
    /// Minimum bits-in-common for an episode to count as a neighbour.
    min_similarity: u32,
}

impl PhasorRouter<HashPhasorEmbedder> {
    /// New router with the default hash embedder.
    pub fn with_defaults() -> Self {
        Self::with_embedder(HashPhasorEmbedder, 256, 8, 40)
    }
}

impl<E: PhasorEmbedder> PhasorRouter<E> {
    /// New router with a custom embedder, capacity, neighbourhood size,
    /// and similarity floor (bits-in-common out of 64).
    pub fn with_embedder(embedder: E, capacity: usize, k: usize, min_similarity: u32) -> Self {
        Self {
            embedder,
            episodes: Vec::new(),
            capacity: capacity.max(1),
            k: k.max(1),
            min_similarity: min_similarity.min(64),
        }
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Record a dispatch outcome.
    pub fn record_outcome(&mut self, q: &Query, winning_tier: TierId, joules_spent: f64) {
        let phasor = self.embedder.embed(q);
        if phasor.0 == 0 {
            return; // empty / non-text query
        }
        if self.episodes.len() >= self.capacity {
            self.episodes.remove(0);
        }
        self.episodes.push(PhasorEpisode {
            phasor,
            winning_tier,
            joules_spent,
        });
    }
}

impl<E: PhasorEmbedder> Router for PhasorRouter<E> {
    fn route(&self, q: &Query) -> RoutingPlan {
        let phasor = self.embedder.embed(q);
        if phasor.0 == 0 || self.episodes.is_empty() {
            return RoutingPlan::fallback(
                crate::ROUTING_JOULES,
                "L5/phasor: cold start or non-text query",
            );
        }

        let mut scored: Vec<(&PhasorEpisode, u32)> = self
            .episodes
            .iter()
            .map(|e| (e, phasor.similarity(e.phasor)))
            .filter(|(_, s)| *s >= self.min_similarity)
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(self.k);

        if scored.is_empty() {
            return RoutingPlan::fallback(
                crate::ROUTING_JOULES,
                "L5/phasor: no episodes within similarity floor",
            );
        }

        let mut wins: HashMap<TierId, (u64, f64, u32)> = HashMap::new();
        for (ep, sim) in &scored {
            let entry = wins.entry(ep.winning_tier).or_insert((0, 0.0, 0));
            entry.0 += *sim as u64;
            entry.1 += ep.joules_spent;
            entry.2 += 1;
        }
        let mut ranked: Vec<(TierId, u64, f64)> = wins
            .into_iter()
            .map(|(t, (v, j, n))| (t, v, j / n as f64))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.0.wire_tag().cmp(b.0.wire_tag()))
        });

        let tier_order: Vec<TierId> = ranked.iter().map(|(t, _, _)| *t).collect();
        RoutingPlan {
            tier_order,
            router_joules: crate::ROUTING_JOULES,
            reasoning: format!("L5/phasor: {} neighbours voted", scored.len()),
        }
    }

    fn estimate_overhead(&self, _q: &Query) -> f64 {
        crate::ROUTING_JOULES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{ContextRef, JouleBudget, QualityFloor, QueryInput};

    fn q(text: &str) -> Query {
        Query {
            input: QueryInput::Text(text.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn embedder_deterministic() {
        let e = HashPhasorEmbedder;
        assert_eq!(e.embed(&q("hello world")), e.embed(&q("world hello")));
    }

    #[test]
    fn embedder_empty_is_zero() {
        let e = HashPhasorEmbedder;
        assert_eq!(e.embed(&q("   ")).0, 0);
    }

    #[test]
    fn hamming_and_similarity_complement() {
        let a = Phasor(0b1010);
        let b = Phasor(0b1001);
        assert_eq!(a.hamming(b) + a.similarity(b), 64);
    }

    #[test]
    fn cold_start_fallback() {
        let r = PhasorRouter::with_defaults();
        assert!(r.route(&q("anything")).is_fallback());
    }

    #[test]
    fn learns_winner() {
        let mut r = PhasorRouter::with_defaults();
        for _ in 0..5 {
            r.record_outcome(&q("capital of france is paris"), TierId::L0_1FactLut, 5e-6);
        }
        let plan = r.route(&q("capital of france is paris"));
        assert!(!plan.is_fallback());
        assert_eq!(plan.tier_order.first().copied(), Some(TierId::L0_1FactLut));
    }

    #[test]
    fn non_text_not_recorded() {
        let mut r = PhasorRouter::with_defaults();
        let query = Query {
            input: QueryInput::Binary(vec![9, 9]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        r.record_outcome(&query, TierId::L0, 1.0);
        assert!(r.is_empty());
    }

    #[test]
    fn capacity_bounds_memory() {
        let mut r = PhasorRouter::with_embedder(HashPhasorEmbedder, 2, 8, 0);
        r.record_outcome(&q("one"), TierId::L0, 1.0);
        r.record_outcome(&q("two"), TierId::L0, 1.0);
        r.record_outcome(&q("three"), TierId::L0, 1.0);
        assert_eq!(r.len(), 2);
    }
}
