//! The speculative-decoding driver loop.
//!
//! The orchestrator owns the top-level decode loop:
//!
//! 1. Ask the draft for `k` tokens.
//! 2. Hand them to the target in a *single* verification call.
//! 3. Append the accepted prefix and either the target's bonus token
//!    (on full acceptance) or its replacement token (on rejection).
//! 4. Repeat until `max_new_tokens` is reached.
//!
//! Acceptance rate is the headline efficiency metric: K = 4 with 80%
//! acceptance is ~3.2× the per-token throughput of plain autoregressive
//! decoding on the target alone, because each verification call only
//! ever runs the target once (rather than `K` times).

use std::convert::TryFrom;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::algorithms::SpeculativeAlgorithm;
use crate::draft::{DraftModel, TokenId};
use crate::error::{SpecDecodeError, SpecDecodeResult};
use crate::sampler::Sampler;
use crate::target::TargetModel;

/// End-to-end output of one speculative-decoding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    /// Rendered text. The reference implementation just concatenates
    /// `TokenId`s as decimal strings separated by spaces — production
    /// deployments wire a real detokenizer at the wrapper layer.
    pub text: String,
    /// Sequence of generated token ids in emission order.
    pub tokens: Vec<TokenId>,
    /// Number of new tokens generated. Always equals `tokens.len()`
    /// and always equals the orchestrator's `max_new_tokens` (unless
    /// the target signaled early stop — not modelled here).
    pub total_new_tokens: usize,
    /// Fraction of draft proposals the target accepted. `1.0` means
    /// every draft token was accepted; `0.0` means none were.
    pub acceptance_rate: f32,
    /// Sum of draft + target joules.
    pub total_joules: u64,
    /// Joules spent inside draft `propose` calls.
    pub draft_joules: u64,
    /// Joules spent inside target `verify` calls.
    pub target_joules: u64,
    /// Number of target forward passes. Speculative decoding's whole
    /// point is to keep this much lower than `total_new_tokens`.
    pub target_forward_passes: u32,
}

/// Speculative-decoding driver.
///
/// Owns the draft, the target, the algorithm hyperparameters and the
/// sampler. `Arc<dyn ...>` so callers can share the same backend
/// across multiple decoders (e.g. one decoder per request, all
/// pointing at the same underlying model).
pub struct SpeculativeDecoder {
    /// Draft model producing the per-round proposal.
    pub draft: Arc<dyn DraftModel>,
    /// Target model performing single-pass verification.
    pub target: Arc<dyn TargetModel>,
    /// Strategy hyperparameters.
    pub algorithm: SpeculativeAlgorithm,
    /// Hard cap on the number of new tokens emitted.
    pub max_new_tokens: usize,
    /// Terminal sampler used when the target needs to emit a token
    /// itself (e.g. bonus token on full acceptance, replacement on
    /// rejection). The reference implementation delegates this to the
    /// target's `verify` return value, but a sampler is kept on the
    /// struct so future algorithms (lookahead, EAGLE-2) can use it.
    pub sampler: Box<dyn Sampler>,
}

impl SpeculativeDecoder {
    /// Construct a decoder. `max_new_tokens` must be non-zero.
    pub fn new(
        draft: Arc<dyn DraftModel>,
        target: Arc<dyn TargetModel>,
        algorithm: SpeculativeAlgorithm,
        max_new_tokens: usize,
        sampler: Box<dyn Sampler>,
    ) -> SpecDecodeResult<Self> {
        if max_new_tokens == 0 {
            return Err(SpecDecodeError::Config(
                "max_new_tokens must be non-zero".into(),
            ));
        }
        Ok(Self {
            draft,
            target,
            algorithm,
            max_new_tokens,
            sampler,
        })
    }

    /// Run the speculative-decoding loop end-to-end. Returns a
    /// [`Generation`] with separate draft / target joule attribution
    /// and a target-forward-pass count.
    pub async fn generate(&self, prompt: &str) -> SpecDecodeResult<Generation> {
        let k = self.algorithm.k().max(1);

        let mut emitted: Vec<TokenId> = Vec::with_capacity(self.max_new_tokens);
        let mut draft_joules: u64 = 0;
        let mut target_joules: u64 = 0;
        let mut target_forward_passes: u32 = 0;

        let mut proposed_total: usize = 0;
        let mut accepted_total: usize = 0;

        let mut prefix = prompt.to_string();

        while emitted.len() < self.max_new_tokens {
            let remaining = self.max_new_tokens - emitted.len();
            let request_k = k.min(remaining);

            // Draft proposes.
            let draft_seq = self.draft.propose(&prefix, request_k).await?;
            if draft_seq.is_empty() {
                return Err(SpecDecodeError::EmptyDraft {
                    requested: request_k,
                });
            }
            draft_joules = draft_joules.saturating_add(draft_seq.draft_joules);

            // Target verifies in a single pass.
            let verification = self.target.verify(&prefix, &draft_seq).await?;
            target_joules = target_joules.saturating_add(verification.target_joules);
            target_forward_passes = target_forward_passes.saturating_add(1);

            // Update acceptance counters. We count the draft tokens
            // we actually asked about (full block size of `draft_seq`)
            // against the prefix the target accepted.
            proposed_total += draft_seq.len();
            accepted_total += verification.accepted_tokens.len();

            // Emit accepted prefix, capping at max_new_tokens.
            for tok in verification.accepted_tokens.iter() {
                if emitted.len() >= self.max_new_tokens {
                    break;
                }
                emitted.push(*tok);
                push_token(&mut prefix, *tok);
            }

            // Emit replacement / bonus token (one extra token per
            // round) if we still have headroom.
            if emitted.len() < self.max_new_tokens
                && let Some(repl) = verification.replacement_token
            {
                emitted.push(repl);
                push_token(&mut prefix, repl);
            }
        }

        // Defensive truncation — push_token + capped loop should
        // never overshoot, but a real backend could return a larger
        // accepted prefix than the orchestrator requested.
        emitted.truncate(self.max_new_tokens);

        let acceptance_rate = if proposed_total > 0 {
            (accepted_total as f32) / (proposed_total as f32)
        } else {
            0.0
        };

        let total_joules = draft_joules.saturating_add(target_joules);
        let text = render_tokens(&emitted);
        let total_new_tokens = emitted.len();

        Ok(Generation {
            text,
            tokens: emitted,
            total_new_tokens,
            acceptance_rate,
            total_joules,
            draft_joules,
            target_joules,
            target_forward_passes,
        })
    }
}

fn push_token(prefix: &mut String, tok: TokenId) {
    if !prefix.is_empty() {
        prefix.push(' ');
    }
    prefix.push_str(&tok.to_string());
}

fn render_tokens(tokens: &[TokenId]) -> String {
    let mut out = String::with_capacity(tokens.len() * 4);
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&t.to_string());
    }
    out
}

/// Run a non-speculative *baseline* over the same target, used by
/// tests to compute the speedup factor. The baseline asks the target
/// to verify single-token "drafts" — one target forward pass per
/// emitted token — so the forward-pass count equals the new-token
/// count.
pub async fn baseline_generate(
    target: Arc<dyn TargetModel>,
    prompt: &str,
    max_new_tokens: usize,
    bonus_token: TokenId,
) -> SpecDecodeResult<(usize, u32, u64)> {
    use crate::draft::DraftSequence;

    let mut emitted = 0usize;
    let mut forward_passes = 0u32;
    let mut target_joules = 0u64;
    let mut prefix = prompt.to_string();
    while emitted < max_new_tokens {
        let dummy = DraftSequence::new(vec![bonus_token], vec![vec![10.0, 0.0]], 0);
        let v = target.verify(&prefix, &dummy).await?;
        forward_passes = forward_passes.saturating_add(1);
        target_joules = target_joules.saturating_add(v.target_joules);
        let added: usize = v.accepted_tokens.len() + usize::from(v.replacement_token.is_some());
        let to_take = added.min(max_new_tokens - emitted);
        let mut taken = 0usize;
        for tok in v.accepted_tokens.iter() {
            if taken >= to_take {
                break;
            }
            push_token(&mut prefix, *tok);
            taken += 1;
        }
        if taken < to_take
            && let Some(repl) = v.replacement_token
        {
            push_token(&mut prefix, repl);
            taken += 1;
        }
        emitted += taken;
        // Safety net — if the target returns zero new tokens, break
        // to avoid an infinite loop.
        if added == 0 {
            break;
        }
    }
    Ok((emitted, forward_passes, target_joules))
}

// `try_from(u32)` rules out platforms where usize < 4 bytes, which
// the workspace toolchain doesn't target. Kept as a const assertion.
const _USIZE_AT_LEAST_U32: bool = {
    if usize::BITS < u32::BITS {
        panic!("eoc-spec-decode requires usize >= 32 bits");
    }
    true
};

// Suppress unused-import warnings on platforms where the `TryFrom`
// import is only used by future helpers.
#[allow(dead_code)]
fn _try_from_anchor(x: u32) -> usize {
    usize::try_from(x).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::vanilla::VanillaSpeculative;
    use crate::sampler::GreedySampler;
    use crate::synthetic::{SyntheticDraft, SyntheticTarget};

    #[tokio::test]
    async fn end_to_end_emits_exactly_max_new_tokens() {
        let draft = Arc::new(SyntheticDraft::new("d", vec![1, 2, 3, 4], 16, 100));
        let target = Arc::new(SyntheticTarget::new("t", 1.0, 50_000, 16, 1));
        let dec = SpeculativeDecoder::new(
            draft,
            target,
            SpeculativeAlgorithm::Vanilla(VanillaSpeculative::new(4)),
            12,
            Box::new(GreedySampler),
        )
        .expect("ok");
        let g = dec.generate("hello").await.expect("ok");
        assert_eq!(g.total_new_tokens, 12);
        assert_eq!(g.tokens.len(), 12);
    }

    #[tokio::test]
    async fn full_acceptance_yields_minimum_forward_passes() {
        // K = 4, target always accepts, plus one bonus token per
        // pass -> 5 emitted per forward pass. 20 tokens / 5 = 4
        // forward passes.
        let draft = Arc::new(SyntheticDraft::new("d", vec![7; 32], 16, 100));
        let target = Arc::new(SyntheticTarget::new("t", 1.0, 50_000, 16, 1));
        let dec = SpeculativeDecoder::new(
            draft,
            target.clone(),
            SpeculativeAlgorithm::Vanilla(VanillaSpeculative::new(4)),
            20,
            Box::new(GreedySampler),
        )
        .expect("ok");
        let g = dec.generate("hello").await.expect("ok");
        assert_eq!(g.total_new_tokens, 20);
        assert_eq!(target.forward_pass_count(), 4);
        assert_eq!(g.target_forward_passes, 4);
        assert!((g.acceptance_rate - 1.0).abs() < 1e-6);
    }
}
