//! Trained-drafter speculative decoding — config + outcome types.
//! The executor (`streaming::extend_with_drafter`) lives in
//! streaming.rs where it can touch Conversation's private fields
//! directly. Same pattern as PLD.
//!
//! ## What it does
//!
//! Same accept/reject logic as PLD but the draft comes from a small
//! paired model instead of an n-gram match in the prompt. The
//! drafter generates `K` tokens autoregressively (`K` cheap forwards
//! on the small model); the target verifies them in one forward
//! pass of `K+1` positions on the big model. Each step emits 1..=K+1
//! tokens for the cost of one target forward plus K drafter forwards.
//!
//! ## When it pays off
//!
//! `cost_per_step = K × drafter_per_token + 1 × target_per_token`.
//! Speedup = `target_per_token / cost_per_step × accepted_count`.
//! For Bonsai-1.7B drafting Bonsai-4B (≈3× param ratio):
//!
//!   drafter_per_token ≈ target_per_token / 3
//!   K=4 → cost ≈ 2.33× target_per_token
//!   accepted=3 average → 1.29× speedup
//!   accepted=5 (full K+1) → 2.15× speedup
//!
//! Per the edge-architecture notes survey: "1.5-2× on echo/code"
//! roughly matches expected acceptance rates.
//!
//! ## Tokenizer requirement
//!
//! Target and drafter MUST use the same tokenizer. The Bonsai family
//! (1.7B / 4B / 8B, all qwen3 arch) qualifies. The mlx-serve survey
//! noted Gemma 4 ships dedicated drafters for the same reason —
//! Google trained them to match the target's vocabulary.

/// Configuration for trained-drafter speculative decoding.
#[derive(Debug, Clone, Copy)]
pub struct DrafterConfig {
    /// Number of tokens the drafter proposes per step. The target
    /// verifies K+1 positions in one forward. Larger K → higher
    /// peak speedup but more compute wasted on a rejected branch.
    pub max_lookahead: usize,
}

impl Default for DrafterConfig {
    fn default() -> Self {
        // 4 is the typical sweet spot (Leviathan et al., "Fast
        // Inference from Transformers via Speculative Decoding").
        // At K=4 with realistic acceptance of 2.5-3 average, expect
        // ~1.5× wall-clock with modest drafter overhead.
        Self { max_lookahead: 4 }
    }
}

/// Result of a drafter-extended generation.
pub struct DrafterOutcome {
    pub tokens: Vec<u32>,
    /// Cumulative target-model joules across prefill + every verify
    /// forward.
    pub target_joules: f64,
    /// Cumulative drafter joules across prefill + every draft forward.
    pub drafter_joules: f64,
    /// Per verify-step: how many tokens were accepted (1..=K+1).
    pub accepted_per_step: Vec<usize>,
}

impl DrafterOutcome {
    pub fn total_joules(&self) -> f64 {
        self.target_joules + self.drafter_joules
    }
    pub fn mean_acceptance(&self) -> f64 {
        if self.accepted_per_step.is_empty() {
            return 1.0;
        }
        let sum: usize = self.accepted_per_step.iter().sum();
        sum as f64 / self.accepted_per_step.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_config_default_k4() {
        let cfg = DrafterConfig::default();
        assert_eq!(cfg.max_lookahead, 4);
    }

    #[test]
    fn drafter_outcome_math() {
        let out = DrafterOutcome {
            tokens: vec![1, 2, 3, 4, 5, 6],
            target_joules: 100.0, drafter_joules: 30.0,
            accepted_per_step: vec![2, 3, 1],
        };
        assert!((out.mean_acceptance() - 2.0).abs() < 1e-9);
        assert!((out.total_joules() - 130.0).abs() < 1e-9);
    }

    #[test]
    fn drafter_outcome_empty_default_acceptance() {
        let out = DrafterOutcome {
            tokens: vec![], target_joules: 0.0, drafter_joules: 0.0,
            accepted_per_step: vec![],
        };
        assert_eq!(out.mean_acceptance(), 1.0);
    }
}
