//! Token-Jaccard nearest-episode router.

use jouleclaw_cascade::router::{Router, RoutingPlan};
use jouleclaw_cascade::types::{Query, QueryInput, TierId};
use std::collections::HashMap;

/// One recorded outcome: a query (reduced to its token set) and the
/// tier that resolved it, plus the joules that tier spent.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Lowercased, deduplicated token set of the query text.
    pub tokens: Vec<String>,
    /// The tier that produced the accepted answer.
    pub winning_tier: TierId,
    /// Joules that tier spent. Lower is better when two tiers tie on
    /// win count.
    pub joules_spent: f64,
}

/// Learned router. Keeps a bounded ring of [`Episode`]s and, on each
/// query, orders tiers by how often they won for the `k` most similar
/// past queries (ties broken by lower mean joules).
pub struct LearnedRouter {
    episodes: Vec<Episode>,
    capacity: usize,
    k: usize,
}

impl LearnedRouter {
    /// New router with the given episode-memory capacity and
    /// neighbourhood size `k`. A `capacity` of 0 is treated as 1.
    pub fn new(capacity: usize, k: usize) -> Self {
        Self {
            episodes: Vec::new(),
            capacity: capacity.max(1),
            k: k.max(1),
        }
    }

    /// A router with sensible defaults (256 episodes, k=8).
    pub fn with_defaults() -> Self {
        Self::new(256, 8)
    }

    /// Number of episodes currently remembered.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Record the outcome of a dispatch so future similar queries
    /// prefer the tier that worked. Call this after the cascade accepts
    /// an answer.
    pub fn record_outcome(&mut self, q: &Query, winning_tier: TierId, joules_spent: f64) {
        let tokens = tokenize(q);
        if tokens.is_empty() {
            return;
        }
        if self.episodes.len() >= self.capacity {
            // Evict the oldest. Ring semantics, FIFO.
            self.episodes.remove(0);
        }
        self.episodes.push(Episode {
            tokens,
            winning_tier,
            joules_spent,
        });
    }

    /// The `k` nearest episodes to `tokens` by token-set Jaccard, paired
    /// with their similarity (q16.16-ish integer score to keep ordering
    /// deterministic: `intersection * 10_000 / union`).
    fn nearest(&self, tokens: &[String]) -> Vec<(&Episode, u32)> {
        let mut scored: Vec<(&Episode, u32)> = self
            .episodes
            .iter()
            .map(|e| (e, jaccard_milli(tokens, &e.tokens)))
            .filter(|(_, s)| *s > 0)
            .collect();
        // Highest similarity first; stable so insertion order breaks ties.
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(self.k);
        scored
    }
}

impl Router for LearnedRouter {
    fn route(&self, q: &Query) -> RoutingPlan {
        let tokens = tokenize(q);
        if tokens.is_empty() || self.episodes.is_empty() {
            return RoutingPlan::fallback(
                crate::ROUTING_JOULES,
                "L5: cold start — no episode memory, walking registration order",
            );
        }

        let neighbours = self.nearest(&tokens);
        if neighbours.is_empty() {
            return RoutingPlan::fallback(
                crate::ROUTING_JOULES,
                "L5: no similar episodes — walking registration order",
            );
        }

        // Tally wins per tier, weighted by similarity, and accumulate
        // joules so ties prefer the cheaper tier.
        let mut wins: HashMap<TierId, (u64, f64, u32)> = HashMap::new();
        for (ep, sim) in &neighbours {
            let entry = wins.entry(ep.winning_tier).or_insert((0, 0.0, 0));
            entry.0 += *sim as u64; // similarity-weighted vote
            entry.1 += ep.joules_spent;
            entry.2 += 1;
        }

        let mut ranked: Vec<(TierId, u64, f64)> = wins
            .into_iter()
            .map(|(tier, (vote, joules, n))| (tier, vote, joules / n as f64))
            .collect();
        // Highest vote first; ties → lower mean joules first; final tie
        // → stable by wire tag for determinism.
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.0.wire_tag().cmp(b.0.wire_tag()))
        });

        let tier_order: Vec<TierId> = ranked.iter().map(|(t, _, _)| *t).collect();
        let top = tier_order
            .first()
            .map(|t| t.wire_tag())
            .unwrap_or("none");
        RoutingPlan {
            tier_order,
            router_joules: crate::ROUTING_JOULES,
            reasoning: format!(
                "L5: {} similar episodes voted; leading tier {}",
                neighbours.len(),
                top
            ),
        }
    }

    fn estimate_overhead(&self, _q: &Query) -> f64 {
        crate::ROUTING_JOULES
    }
}

/// Lowercased, deduplicated token set of the query text. Non-text
/// queries reduce to the empty set (the router has no opinion).
pub(crate) fn tokenize(q: &Query) -> Vec<String> {
    let text = match &q.input {
        QueryInput::Text(t) => t.as_str(),
        QueryInput::Multimodal { text, .. } => text.as_str(),
        _ => return Vec::new(),
    };
    let mut seen = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        let tok = raw.to_lowercase();
        if !seen.contains(&tok) {
            seen.push(tok);
        }
    }
    seen
}

/// Jaccard similarity of two token sets, scaled to `[0, 10_000]`
/// (per-mille × 10). Integer-valued so ordering is deterministic.
pub(crate) fn jaccard_milli(a: &[String], b: &[String]) -> u32 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let inter = a.iter().filter(|t| b.contains(t)).count();
    if inter == 0 {
        return 0;
    }
    // |A ∪ B| = |A| + |B| - |A ∩ B|
    let union = a.len() + b.len() - inter;
    ((inter as u64 * 10_000) / union as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, L3ModelId, QualityFloor, TierId,
    };

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
    fn cold_start_is_fallback() {
        let r = LearnedRouter::with_defaults();
        let plan = r.route(&q("what is the capital of france"));
        assert!(plan.is_fallback());
        assert_eq!(plan.router_joules, crate::ROUTING_JOULES);
    }

    #[test]
    fn non_text_is_fallback() {
        let r = LearnedRouter::with_defaults();
        let query = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(r.route(&query).is_fallback());
    }

    #[test]
    fn learns_winning_tier() {
        let mut r = LearnedRouter::new(64, 8);
        for _ in 0..5 {
            r.record_outcome(&q("capital of france paris"), TierId::L0_1FactLut, 5e-6);
        }
        let plan = r.route(&q("capital of france"));
        assert!(!plan.is_fallback());
        assert_eq!(plan.tier_order.first().copied(), Some(TierId::L0_1FactLut));
    }

    #[test]
    fn prefers_more_frequent_winner() {
        let mut r = LearnedRouter::new(64, 16);
        for _ in 0..6 {
            r.record_outcome(&q("translate hello to french"), TierId::L3(L3ModelId(0)), 2.0);
        }
        for _ in 0..2 {
            r.record_outcome(&q("translate hello to french"), TierId::L0_1FactLut, 5e-6);
        }
        let plan = r.route(&q("translate hello to french"));
        assert_eq!(plan.tier_order.first().copied(), Some(TierId::L3(L3ModelId(0))));
    }

    #[test]
    fn dissimilar_query_gets_no_neighbours() {
        let mut r = LearnedRouter::new(64, 8);
        r.record_outcome(&q("capital of france"), TierId::L0_1FactLut, 5e-6);
        let plan = r.route(&q("integrate x squared dx"));
        assert!(plan.is_fallback());
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut r = LearnedRouter::new(3, 8);
        r.record_outcome(&q("alpha one"), TierId::L0, 1.0);
        r.record_outcome(&q("beta two"), TierId::L0, 1.0);
        r.record_outcome(&q("gamma three"), TierId::L0, 1.0);
        r.record_outcome(&q("delta four"), TierId::L0, 1.0);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn empty_query_not_recorded() {
        let mut r = LearnedRouter::new(8, 8);
        r.record_outcome(&q("   "), TierId::L0, 1.0);
        assert!(r.is_empty());
    }

    #[test]
    fn jaccard_basics() {
        let a = vec!["x".to_string(), "y".to_string()];
        let b = vec!["x".to_string(), "y".to_string()];
        assert_eq!(jaccard_milli(&a, &b), 10_000);
        let c = vec!["z".to_string()];
        assert_eq!(jaccard_milli(&a, &c), 0);
    }

    #[test]
    fn tie_break_prefers_cheaper_tier() {
        let mut r = LearnedRouter::new(64, 16);
        // Equal vote weight (same query, same count), different joules.
        for _ in 0..3 {
            r.record_outcome(&q("foo bar baz"), TierId::L4_5Proof, 60e-6);
        }
        for _ in 0..3 {
            r.record_outcome(&q("foo bar baz"), TierId::L3(L3ModelId(0)), 2.0);
        }
        let plan = r.route(&q("foo bar baz"));
        // Both have equal similarity-weighted votes; cheaper (Proof) wins.
        assert_eq!(plan.tier_order.first().copied(), Some(TierId::L4_5Proof));
    }

    #[test]
    fn determinism_same_state_same_plan() {
        let mut r = LearnedRouter::new(64, 8);
        for _ in 0..4 {
            r.record_outcome(&q("sort these numbers"), TierId::L0_5ToolCompute, 15e-6);
        }
        let p1 = r.route(&q("sort these numbers please"));
        let p2 = r.route(&q("sort these numbers please"));
        assert_eq!(p1.tier_order, p2.tier_order);
    }
}
