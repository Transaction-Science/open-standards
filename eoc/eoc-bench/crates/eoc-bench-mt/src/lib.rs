//! MT-Bench-style cases bundled with the EOC benchmark harness.
//!
//! The cases live in `data/cases.json` next to this crate. They are
//! embedded at compile time so the published crate is self-contained and
//! reproducible across environments. The schema is the JSON form of
//! [`eoc_bench_runner::BenchCase`]:
//!
//! ```json
//! {
//!   "id": "mt-arith-01",
//!   "prompt": "What is 2+2?",
//!   "expected": "4",
//!   "dataset": "mt"
//! }
//! ```
//!
//! The set deliberately mixes single-turn, multi-turn (concatenated with
//! `\nFollow-up:`), math, reasoning, code, and writing cases so the
//! cascade exercises all four stages.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use eoc_bench_runner::BenchCase;

/// Raw JSON for the MT-Bench-style suite, embedded at compile time.
pub const CASES_JSON: &str = include_str!("../data/cases.json");

/// Load and parse the bundled MT-Bench-style cases.
///
/// Panics if the embedded JSON fails to parse — that would be a build
/// error, not a runtime condition.
pub fn load() -> Vec<BenchCase> {
    serde_json::from_str(CASES_JSON).expect("eoc-bench-mt: data/cases.json is malformed")
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
            assert_eq!(c.dataset, "mt", "all cases must be tagged dataset=mt");
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
        // The harness pins at 20 MT cases.
        assert_eq!(load().len(), 20);
    }
}
