//! Speculative-decoding algorithm variants.
//!
//! Each algorithm is a *strategy* the orchestrator selects via the
//! [`SpeculativeAlgorithm`] enum. The orchestrator owns the loop
//! structure; the algorithm owns the per-step decision logic (how many
//! tokens to propose, what acceptance criterion to apply, when to
//! short-circuit).

pub mod eagle;
pub mod lookahead;
pub mod medusa;
pub mod sps_with_temperature;
pub mod vanilla;

use crate::draft::DraftSequence;
use crate::error::SpecDecodeResult;
use crate::target::VerificationResult;

/// Acceptance decision for one position of a draft proposal under a
/// given algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceDecision {
    /// Accept the draft's proposed token and move on.
    Accept,
    /// Reject. The orchestrator substitutes the target's preferred
    /// token (or, in SpS, samples from the corrected distribution) and
    /// stops walking the proposal.
    Reject,
}

/// Strategy enum chosen by the orchestrator at construction time.
///
/// Each variant carries its hyperparameters so the orchestrator can be
/// constructed without further configuration. Variants line up 1:1
/// with the modules below.
#[derive(Debug, Clone)]
pub enum SpeculativeAlgorithm {
    /// Classic Leviathan greedy: accept iff the draft's token matches
    /// the target's argmax.
    Vanilla(vanilla::VanillaSpeculative),
    /// Chen et al. probabilistic speculative sampling.
    SpsWithTemperature(sps_with_temperature::SpsWithTemperature),
    /// Lookahead decoding (Fu et al. 2024) — needs no separate draft
    /// model; runs Jacobi iteration over an n-gram cache.
    Lookahead(lookahead::LookaheadDecoding),
    /// EAGLE skeleton — requires backend-internal access.
    Eagle(eagle::LocalEagleDraft),
    /// Medusa skeleton — requires parallel decoding heads.
    Medusa(medusa::LocalMedusaDraft),
}

impl SpeculativeAlgorithm {
    /// Maximum number of tokens to propose per round. The orchestrator
    /// uses this to size the draft call.
    pub fn k(&self) -> usize {
        match self {
            Self::Vanilla(a) => a.k,
            Self::SpsWithTemperature(a) => a.k,
            Self::Lookahead(a) => a.window,
            Self::Eagle(a) => a.k,
            Self::Medusa(a) => a.heads,
        }
    }

    /// Apply the algorithm's per-position acceptance criterion. Called
    /// by the orchestrator inside the verification loop.
    ///
    /// `draft_logits` and `target_logits` are the logit vectors the
    /// draft and target produced at the same position. `rng` provides
    /// the entropy SpS needs for its probabilistic test. `proposed`
    /// is the draft's chosen token at this position.
    pub fn accept(
        &self,
        proposed: crate::draft::TokenId,
        draft_logits: &[f32],
        target_logits: &[f32],
        rng: &mut crate::sampler::SplitMix64,
    ) -> SpecDecodeResult<AcceptanceDecision> {
        match self {
            Self::Vanilla(a) => a.accept(proposed, draft_logits, target_logits),
            Self::SpsWithTemperature(a) => {
                a.accept(proposed, draft_logits, target_logits, rng)
            }
            Self::Lookahead(_) | Self::Eagle(_) | Self::Medusa(_) => {
                // These three drive the orchestrator through different
                // paths; the orchestrator dispatches on the variant
                // before reaching the per-position decision.
                Ok(AcceptanceDecision::Accept)
            }
        }
    }
}

/// Convenience helper: walk a draft proposal under an algorithm and
/// return the prefix of accepted tokens plus the index at which
/// rejection occurred (if any). Used by the orchestrator to build a
/// [`VerificationResult`] from raw target logits.
pub fn walk_acceptance(
    algorithm: &SpeculativeAlgorithm,
    draft: &DraftSequence,
    target_logits: &[Vec<f32>],
    rng: &mut crate::sampler::SplitMix64,
) -> SpecDecodeResult<(Vec<crate::draft::TokenId>, Option<usize>)> {
    let mut accepted = Vec::with_capacity(draft.len());
    for (i, &tok) in draft.tokens.iter().enumerate() {
        let dl = &draft.logits[i];
        let tl = target_logits
            .get(i)
            .ok_or(crate::error::SpecDecodeError::EmptyVerification)?;
        match algorithm.accept(tok, dl, tl, rng)? {
            AcceptanceDecision::Accept => accepted.push(tok),
            AcceptanceDecision::Reject => return Ok((accepted, Some(i))),
        }
    }
    Ok((accepted, None))
}

/// Plumbing-only helper used by the orchestrator to build a
/// [`VerificationResult`] from a walk + target logits when the
/// algorithm doesn't do its own verification end-to-end.
pub fn finalize_verification(
    accepted: Vec<crate::draft::TokenId>,
    rejected_at: Option<usize>,
    target_logits: &[Vec<f32>],
    target_joules: u64,
    rng: &mut crate::sampler::SplitMix64,
    algorithm: &SpeculativeAlgorithm,
) -> SpecDecodeResult<VerificationResult> {
    // Position from which to draw the replacement / bonus token.
    let pos = match rejected_at {
        Some(i) => i,
        None => accepted.len(),
    };
    let replacement = if let Some(tl) = target_logits.get(pos) {
        Some(match algorithm {
            SpeculativeAlgorithm::SpsWithTemperature(a) => {
                if let Some(i) = rejected_at {
                    a.sample_adjusted(&draft_logits_at(target_logits, i)?, tl, rng)?
                } else {
                    sample_greedy(tl)?
                }
            }
            _ => sample_greedy(tl)?,
        })
    } else {
        None
    };

    Ok(VerificationResult {
        accepted_tokens: accepted,
        rejected_at,
        replacement_token: replacement,
        target_joules,
    })
}

fn sample_greedy(logits: &[f32]) -> SpecDecodeResult<crate::draft::TokenId> {
    if logits.is_empty() {
        return Err(crate::error::SpecDecodeError::EmptyVerification);
    }
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            return Err(crate::error::SpecDecodeError::Sampling(
                "NaN in target logits".into(),
            ));
        }
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    Ok(best as crate::draft::TokenId)
}

// Placeholder that's only invoked in tests by the SpS path through
// finalize_verification. We don't have the draft logits at this site,
// so we conservatively fall back to the target's distribution when
// SpS's adjusted-distribution sampler asks for them. The orchestrator
// avoids this path by computing SpS rejection sampling itself with
// proper access to both logit vectors.
fn draft_logits_at(
    target_logits: &[Vec<f32>],
    i: usize,
) -> SpecDecodeResult<Vec<f32>> {
    target_logits
        .get(i)
        .cloned()
        .ok_or(crate::error::SpecDecodeError::EmptyVerification)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SplitMix64;

    #[test]
    fn walk_acceptance_full_accept() {
        let alg = SpeculativeAlgorithm::Vanilla(vanilla::VanillaSpeculative { k: 3 });
        let draft = DraftSequence::new(
            vec![1, 2, 3],
            vec![
                vec![0.0, 10.0, 0.0, 0.0],
                vec![0.0, 0.0, 10.0, 0.0],
                vec![0.0, 0.0, 0.0, 10.0],
            ],
            10,
        );
        let target_logits = vec![
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0],
            vec![0.0, 0.0, 0.0, 10.0],
        ];
        let mut rng = SplitMix64::new(1);
        let (accepted, rejected_at) =
            walk_acceptance(&alg, &draft, &target_logits, &mut rng).expect("walk");
        assert_eq!(accepted, vec![1, 2, 3]);
        assert!(rejected_at.is_none());
    }

    #[test]
    fn walk_acceptance_reject_midway() {
        let alg = SpeculativeAlgorithm::Vanilla(vanilla::VanillaSpeculative { k: 3 });
        let draft = DraftSequence::new(
            vec![1, 2, 3],
            vec![
                vec![0.0, 10.0, 0.0, 0.0],
                vec![0.0, 0.0, 10.0, 0.0],
                vec![0.0, 0.0, 0.0, 10.0],
            ],
            10,
        );
        // Target disagrees at position 1: argmax is 3, not 2.
        let target_logits = vec![
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0],
        ];
        let mut rng = SplitMix64::new(1);
        let (accepted, rejected_at) =
            walk_acceptance(&alg, &draft, &target_logits, &mut rng).expect("walk");
        assert_eq!(accepted, vec![1]);
        assert_eq!(rejected_at, Some(1));
    }
}
