//! Router-shaped cases — 30 prompts deliberately chosen so the cascade
//! has to use *all four* stages to answer them efficiently.
//!
//! The case ids carry a hint of which stage a well-configured cascade
//! should route them to:
//!
//! * `rt-cache-*` — short, repeated, deterministic prompts that should
//!   memoize after the first run.
//! * `rt-kv-*`    — exact-key lookups (system facts, well-known
//!   constants) that a populated KV store should resolve.
//! * `rt-graph-*` — structured, triple-shaped questions
//!   (`subject — predicate — ?`).
//! * `rt-neural-*` — open-ended generation / reasoning that no
//!   pre-populated store can answer.
//!
//! These are *hints*, not contracts. The point of the harness is to
//! measure how well a deployment's router and warmup hit those targets;
//! the case file itself does not enforce a stage.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use eoc_bench_runner::BenchCase;

/// Raw JSON for the router suite, embedded at compile time.
pub const CASES_JSON: &str = include_str!("../data/cases.json");

/// Load and parse the bundled router-shaped cases.
pub fn load() -> Vec<BenchCase> {
    serde_json::from_str(CASES_JSON).expect("eoc-bench-router: data/cases.json is malformed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_parse() {
        let cases = load();
        assert!(!cases.is_empty(), "case set must not be empty");
        for c in &cases {
            assert!(!c.id.is_empty(), "case id must not be empty");
            assert!(!c.prompt.is_empty(), "case prompt must not be empty");
            assert_eq!(c.dataset, "router", "all cases must be tagged dataset=router");
        }
    }

    #[test]
    fn case_ids_are_unique() {
        let cases = load();
        let mut ids: Vec<&str> = cases.iter().map(|c| c.id.as_str()).collect();
        ids.sort_unstable();
        let total = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), total, "case ids must be unique");
    }

    #[test]
    fn case_count_matches_spec() {
        // The harness pins at 30 router cases.
        assert_eq!(load().len(), 30);
    }

    #[test]
    fn all_four_stage_hints_are_represented() {
        let cases = load();
        let n_cache = cases.iter().filter(|c| c.id.starts_with("rt-cache-")).count();
        let n_kv = cases.iter().filter(|c| c.id.starts_with("rt-kv-")).count();
        let n_graph = cases.iter().filter(|c| c.id.starts_with("rt-graph-")).count();
        let n_neural = cases.iter().filter(|c| c.id.starts_with("rt-neural-")).count();
        assert!(n_cache >= 4, "expected at least 4 cache-shaped cases");
        assert!(n_kv >= 4, "expected at least 4 kv-shaped cases");
        assert!(n_graph >= 4, "expected at least 4 graph-shaped cases");
        assert!(n_neural >= 4, "expected at least 4 neural-shaped cases");
        assert_eq!(n_cache + n_kv + n_graph + n_neural, cases.len());
    }
}
