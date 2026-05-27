//! Zero-shot Normalised Compression Distance — pure-Rust stand-in.
//!
//! The donor's `crate::mdl::concept_ncd` depended on a compression backend
//! (`zstd`/`gzip` via the `mdl` crate inside `verity-cascade`). That backend
//! is not portable to a `forbid(unsafe_code)`-clean, zero-extra-dep
//! JouleClaw crate.
//!
//! ## What we replace it with
//!
//! A deterministic distance that approximates NCD for short strings using
//! character n-gram (trigram) Jaccard distance:
//!
//! ```text
//! ncd(a, b) ≈ 1 - |trigrams(a) ∩ trigrams(b)| / |trigrams(a) ∪ trigrams(b)|
//! ```
//!
//! Trigram Jaccard has the same load-bearing property as compression-based
//! NCD for the formula tier's purpose: it is `0.0` for identical strings,
//! approaches `1.0` for strings that share no substring structure, and is
//! monotonic in shared substructure. The formula tier only uses NCD as a
//! tie-breaker confidence signal when no entities resolve; we lose absolute
//! calibration (zstd-derived NCD has different numerical values) but
//! preserve the ordering signal that the donor relied on.
//!
//! ## Why not pull in `zstd`?
//!
//! `zstd` requires a C dependency. JouleClaw open-standard crates are
//! pure-Rust, `forbid(unsafe_code)`. The donor's NCD path is a graceful-
//! degradation fallback, not the hot path — substituting trigram Jaccard
//! keeps the behaviour shape without dragging a C compiler into the
//! standard.

use std::collections::HashSet;

/// Normalised Compression Distance approximation via trigram Jaccard.
///
/// Returns a value in `[0.0, 1.0]`. `0.0` means "structurally identical",
/// `1.0` means "no shared substrings of length ≥ 3".
///
/// Inputs SHOULD be lower-cased and whitespace-collapsed by the caller; the
/// function itself does not normalise. Empty inputs → `1.0`.
pub fn concept_ncd(a: &str, b: &str) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    if a == b {
        return 0.0;
    }
    let ta = trigrams(a);
    let tb = trigrams(b);
    if ta.is_empty() || tb.is_empty() {
        // Strings shorter than 3 chars: fall back to character overlap.
        let ca: HashSet<char> = a.chars().collect();
        let cb: HashSet<char> = b.chars().collect();
        let inter = ca.intersection(&cb).count();
        let union = ca.union(&cb).count();
        if union == 0 {
            return 1.0;
        }
        return 1.0 - (inter as f64 / union as f64);
    }
    let inter = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    if union == 0 {
        return 1.0;
    }
    1.0 - (inter as f64 / union as f64)
}

fn trigrams(s: &str) -> HashSet<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return HashSet::new();
    }
    let mut out = HashSet::with_capacity(chars.len().saturating_sub(2));
    for i in 0..=(chars.len() - 3) {
        let g: String = chars[i..i + 3].iter().collect();
        out.insert(g);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings_zero_distance() {
        assert_eq!(concept_ncd("fire", "fire"), 0.0);
    }

    #[test]
    fn disjoint_strings_high_distance() {
        let d = concept_ncd("aaaa", "zzzz");
        assert!(d > 0.9, "expected high distance, got {d}");
    }

    #[test]
    fn similar_strings_low_distance() {
        let d_close = concept_ncd("hydrogen", "hydrogenate");
        let d_far = concept_ncd("hydrogen", "tomato");
        assert!(
            d_close < d_far,
            "close pair should rank closer: {d_close} vs {d_far}"
        );
    }

    #[test]
    fn empty_string_returns_one() {
        assert_eq!(concept_ncd("", "anything"), 1.0);
    }

    #[test]
    fn short_strings_fall_back_to_char_overlap() {
        // "ab" and "ba" share both chars but have no shared trigram.
        let d = concept_ncd("ab", "ba");
        assert!(d < 1.0);
    }
}
