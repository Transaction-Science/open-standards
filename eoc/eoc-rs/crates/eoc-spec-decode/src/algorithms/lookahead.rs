//! Lookahead decoding (Fu, Bailis, Stoica & Zhang 2024).
//!
//! Lookahead does *not* use a separate draft model. Instead it runs
//! Jacobi iteration over a *guess set*: a small set of candidate
//! continuations sampled from an n-gram cache built on the fly from
//! the target's own past outputs. The target verifies all guesses in
//! parallel; surviving n-grams are admitted to the cache and used to
//! seed the next round.
//!
//! ## What this module implements
//!
//! The n-gram cache + lookup machinery is pure-Rust, deterministic,
//! and useful on its own — for tooling, for tests, and as a building
//! block that real backends can wire into the lookahead-decoding step
//! in vLLM / TGI-style runners.
//!
//! The full Jacobi parallel-verification loop is left to the
//! orchestrator: the orchestrator runs the target with a multi-token
//! verification window (`window` tokens) and uses
//! [`LookaheadDecoding::propose_from_cache`] to seed the candidates.
//! When no n-gram cache hit is available the algorithm degrades
//! gracefully to single-token target decoding.

use std::collections::HashMap;

use crate::draft::TokenId;

/// Lookahead-decoding state — an n-gram cache that grows monotonically
/// as the target emits tokens.
#[derive(Debug, Clone)]
pub struct LookaheadDecoding {
    /// Verification window — how many tokens the target verifies in a
    /// single forward pass.
    pub window: usize,
    /// N-gram width used by the cache (typical: 3–5).
    pub n: usize,
    cache: HashMap<Vec<TokenId>, Vec<Vec<TokenId>>>,
}

impl LookaheadDecoding {
    /// Construct an empty lookahead state.
    pub fn new(window: usize, n: usize) -> Self {
        Self {
            window: window.max(1),
            n: n.max(2),
            cache: HashMap::new(),
        }
    }

    /// Number of distinct n-gram prefixes currently cached.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Walk a freshly-emitted sequence and admit every length-`n`
    /// n-gram into the cache. The key is the first `n-1` tokens, the
    /// value is the continuation we observed after them.
    pub fn ingest(&mut self, sequence: &[TokenId]) {
        if sequence.len() < self.n {
            return;
        }
        for start in 0..=sequence.len() - self.n {
            let prefix: Vec<TokenId> = sequence[start..start + self.n - 1].to_vec();
            let continuation: Vec<TokenId> =
                sequence[start + self.n - 1..start + self.n].to_vec();
            self.cache
                .entry(prefix)
                .or_default()
                .push(continuation);
        }
    }

    /// Look up candidate continuations for the last `n-1` tokens of
    /// `prefix`. Returns up to `window` distinct candidates ordered by
    /// the order they were first observed.
    pub fn propose_from_cache(&self, prefix: &[TokenId]) -> Vec<TokenId> {
        if prefix.len() < self.n - 1 {
            return Vec::new();
        }
        let key: Vec<TokenId> = prefix[prefix.len() - (self.n - 1)..].to_vec();
        let Some(candidates) = self.cache.get(&key) else {
            return Vec::new();
        };
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for cand in candidates {
            for &tok in cand {
                if seen.insert(tok) {
                    out.push(tok);
                    if out.len() >= self.window {
                        return out;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_and_propose_simple_sequence() {
        // n = 3 -> we key on (a, b) and store c. Sequence "1 2 3 4 5"
        // yields keys [1,2]->3, [2,3]->4, [3,4]->5.
        let mut la = LookaheadDecoding::new(4, 3);
        la.ingest(&[1, 2, 3, 4, 5]);
        assert_eq!(la.cache_size(), 3);
        // Looking up "ends with 3, 4" should yield 5.
        assert_eq!(la.propose_from_cache(&[10, 3, 4]), vec![5]);
        // Looking up "ends with 1, 2" should yield 3.
        assert_eq!(la.propose_from_cache(&[10, 1, 2]), vec![3]);
    }

    #[test]
    fn propose_returns_empty_on_cache_miss() {
        let la = LookaheadDecoding::new(4, 3);
        assert!(la.propose_from_cache(&[1, 2, 3]).is_empty());
    }

    #[test]
    fn ingest_too_short_is_noop() {
        let mut la = LookaheadDecoding::new(4, 3);
        la.ingest(&[1, 2]);
        assert_eq!(la.cache_size(), 0);
    }

    #[test]
    fn proposals_respect_window() {
        let mut la = LookaheadDecoding::new(2, 3);
        // Multiple continuations from the same key.
        la.ingest(&[1, 2, 3, 4, 5, 6, 7]);
        // After ingesting we have (5,6)->7. Now manually inject more
        // continuations for (5,6) via further ingests.
        la.ingest(&[5, 6, 8]);
        la.ingest(&[5, 6, 9]);
        let p = la.propose_from_cache(&[10, 5, 6]);
        assert_eq!(p.len(), 2); // window = 2
    }

    #[test]
    fn window_minimum_is_one() {
        assert_eq!(LookaheadDecoding::new(0, 3).window, 1);
    }

    #[test]
    fn n_minimum_is_two() {
        assert_eq!(LookaheadDecoding::new(4, 0).n, 2);
    }
}
