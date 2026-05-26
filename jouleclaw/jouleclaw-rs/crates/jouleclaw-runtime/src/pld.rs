//! Prompt Lookup Decoding — speculative decoding with the prompt as drafter.
//!
//! ## What
//!
//! Standard speculative decoding needs a small "draft" model to propose
//! tokens that a big "target" model verifies in parallel. PLD throws out
//! the draft model and uses the prompt itself: if the last few tokens of
//! the generated sequence match an n-gram seen earlier in the history,
//! the K tokens that followed in the history become the draft.
//!
//! One forward pass with `[last_token, draft_1, ..., draft_K]` produces
//! `K+1` logit rows. Sample each row; the model's prediction at row 0
//! is the "real" next token (this is what no-PLD would have produced).
//! For each draft `d_i` whose prediction `s_{i-1}` from the previous row
//! matches it, the draft is verified — we can use the model's row-i
//! prediction `s_i` as the *next* token, and so on until a mismatch.
//!
//! ## When it pays off
//!
//! Echo-heavy outputs: RAG, code completion, "summarize this paragraph",
//! agentic tool-call replay. Typical 1.3–1.6× wall-clock speedup.
//!
//! For purely novel generation, no n-gram match → degrades to standard
//! single-token decode (one extra empty-draft forward per step), so the
//! cost of PLD on a bad workload is negligible.
//!
//! ## KV-cache correctness
//!
//! Each PLD forward advances the in-place cache by `K+1`. After
//! verification we accept `A ∈ [1, K+1]` tokens; the cache has stale
//! K/V entries at positions `accepted..K+1` that came from rejected
//! drafts. We rewind `cache.current_seq` by `(K+1) - A`. The stale
//! slots are masked out by the attention mask (which is driven by
//! `current_seq`) and overwritten by the next forward pass.
//!
//! ## Why not just modify `ConversationStream`?
//!
//! `ConversationStream::next()` is a one-token-per-call iterator —
//! every PLD step can emit between 1 and K+1 tokens, which doesn't
//! fit the iterator contract without a pending buffer. The non-
//! streaming entry point `extend_pld_tokens` returns a `Vec<u32>` of
//! generated tokens along with a per-step acceptance histogram so
//! callers can measure how much PLD actually helped their workload.

// PldConfig / find_draft / PldOutcome — pure algorithm. The
// `extend_pld_tokens` method lives in streaming.rs where it can
// touch Conversation's private state.

/// Configuration for Prompt Lookup Decoding.
#[derive(Debug, Clone, Copy)]
pub struct PldConfig {
    /// Length of the n-gram suffix to match against the history.
    /// Smaller (2–3) → more matches, more false positives.
    /// Larger (4–5) → fewer matches, higher precision.
    pub ngram_size: usize,
    /// Maximum draft length per forward pass. The forward runs with
    /// `K+1` tokens (1 verified + K drafts). The KV cache advances by
    /// `K+1`; rewinding by `(K+1) - accepted` is automatic.
    pub max_lookahead: usize,
    /// Maximum lookback window over the full history. 0 = unbounded.
    pub max_lookback: usize,
}

impl Default for PldConfig {
    fn default() -> Self {
        // 3-gram match + 3-token lookahead matches what the original
        // PLD paper used. Per-step worst-case compute is 4× single-
        // token decode (one real + three drafts).
        Self { ngram_size: 3, max_lookahead: 3, max_lookback: 0 }
    }
}

/// Result of [`Conversation::extend_pld_tokens`].
pub struct PldOutcome {
    /// Generated tokens (does NOT include the prompt). Length is
    /// bounded by `cfg.max_new_tokens`.
    pub tokens: Vec<u32>,
    /// Sum of `KernelResult.joules` across the prefill + every PLD
    /// forward pass. The Runtime's calibration ledger sees this as the
    /// `actual` reading.
    pub joules: f64,
    /// Per forward pass: how many tokens were accepted. Sum equals
    /// `tokens.len()`. The mean of this vector divided by 1 is the
    /// effective speedup-per-pass; e.g., 2.0 means PLD averaged 2
    /// tokens per pass vs no-PLD's 1.
    pub accepted_per_step: Vec<usize>,
}

impl PldOutcome {
    /// Mean tokens accepted per forward pass. 1.0 means PLD landed no
    /// hits (every draft was rejected). Higher values are wall-clock
    /// speedup over no-PLD.
    pub fn mean_acceptance(&self) -> f64 {
        if self.accepted_per_step.is_empty() {
            return 1.0;
        }
        let sum: usize = self.accepted_per_step.iter().sum();
        sum as f64 / self.accepted_per_step.len() as f64
    }
}

/// Find the most recent occurrence of the last `cfg.ngram_size` tokens
/// of `history` earlier in `history`, and return the up-to-K tokens
/// that followed it. Empty when no match.
///
/// "Most recent" so the draft reflects the most local pattern; "K
/// tokens that followed" because that's the chunk the model is
/// likely about to emit if it's echoing the same context.
pub fn find_draft(history: &[u32], cfg: &PldConfig) -> Vec<u32> {
    let n = cfg.ngram_size;
    if history.len() <= n || cfg.max_lookahead == 0 {
        return Vec::new();
    }
    let suffix_start = history.len() - n;
    let suffix = &history[suffix_start..];

    let lookback_floor = if cfg.max_lookback > 0 && history.len() > cfg.max_lookback {
        history.len() - cfg.max_lookback
    } else {
        0
    };
    let last_match_start = suffix_start; // search must be STRICTLY before the suffix

    // Scan backwards so the most recent match wins.
    if last_match_start < n {
        return Vec::new();
    }
    let mut best: Option<usize> = None;
    let mut start = last_match_start - n;
    loop {
        if start < lookback_floor { break; }
        if &history[start..start + n] == suffix {
            best = Some(start + n);
            break;
        }
        if start == 0 { break; }
        start -= 1;
    }
    let Some(idx) = best else { return Vec::new(); };
    let take = cfg.max_lookahead.min(history.len() - idx);
    history[idx..idx + take].to_vec()
}

// `extend_pld_tokens` lives in streaming.rs where it can touch
// Conversation's private fields directly.

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ngram: usize, lookahead: usize) -> PldConfig {
        PldConfig { ngram_size: ngram, max_lookahead: lookahead, max_lookback: 0 }
    }

    #[test]
    fn no_match_returns_empty_draft() {
        // History too short for any n-gram suffix to repeat.
        let h = vec![1, 2, 3];
        assert!(find_draft(&h, &cfg(3, 3)).is_empty());
    }

    #[test]
    fn echo_history_drafts_the_continuation() {
        // History: [the cat sat on the mat the cat] — the last
        // 2-gram is "the cat"; "the cat" appeared earlier followed
        // by "sat on the mat". Draft should be those 3 tokens.
        let h = vec![1, 2, 3, 4, 1, 5, 1, 2];
        // suffix [1, 2] = "the cat"; earliest match starts at idx 0;
        // 3 tokens following are [3, 4, 1].
        let draft = find_draft(&h, &cfg(2, 3));
        assert_eq!(draft, vec![3, 4, 1]);
    }

    #[test]
    fn most_recent_match_wins() {
        // History: [a b X Y a b Z W a b]. Suffix "a b" appears twice
        // before the final occurrence. We want the most recent match,
        // which followed by [Z, W] (the latest occurrence prior to
        // the suffix).
        let h = vec![10, 20, 30, 40, 10, 20, 50, 60, 10, 20];
        let draft = find_draft(&h, &cfg(2, 2));
        assert_eq!(draft, vec![50, 60]);
    }

    #[test]
    fn max_lookahead_caps_draft_length() {
        let h = vec![1, 2, 3, 4, 5, 6, 7, 1, 2];
        let draft = find_draft(&h, &cfg(2, 2));
        assert_eq!(draft.len(), 2);
        assert_eq!(draft, vec![3, 4]);
    }

    #[test]
    fn ngram_size_zero_treated_as_no_lookup() {
        // Edge: ngram=0 would always "match"; we treat as no-op.
        let h = vec![1, 2, 3];
        // With ngram=0 and history=[1,2,3], suffix is empty slice — it
        // would match at every position, but max_lookahead=0 yields
        // empty draft anyway. Either short-circuit is acceptable; we
        // guarantee the result is empty.
        assert!(find_draft(&h, &cfg(0, 3)).is_empty());
    }

    #[test]
    fn mean_acceptance_default_is_one() {
        let outcome = PldOutcome { tokens: vec![], joules: 0.0, accepted_per_step: vec![] };
        assert_eq!(outcome.mean_acceptance(), 1.0);
    }

    #[test]
    fn mean_acceptance_averages_per_step() {
        let outcome = PldOutcome {
            tokens: vec![1, 2, 3, 4, 5],
            joules: 0.0,
            accepted_per_step: vec![1, 2, 2],
        };
        // (1 + 2 + 2) / 3 = 5/3 ≈ 1.667 — i.e., 1.67× tokens per pass
        // on average.
        assert!((outcome.mean_acceptance() - 5.0 / 3.0).abs() < 1e-9);
    }
}

