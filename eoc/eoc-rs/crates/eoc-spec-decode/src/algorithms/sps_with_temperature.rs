//! Probabilistic speculative sampling (Chen et al. 2023).
//!
//! Where vanilla speculative decoding uses a greedy acceptance test,
//! SpS uses a probabilistic one that preserves the *exact*
//! distribution of the target model — sampling from `p_target` via
//! draft proposals is unbiased.
//!
//! ### Acceptance criterion
//!
//! At position `i`, accept the draft's token `t` with probability
//!
//! ```text
//! min(1, p_target(t) / p_draft(t))
//! ```
//!
//! ### Rejection sampling
//!
//! On reject at position `i`, sample the replacement token from the
//! *adjusted* distribution
//!
//! ```text
//! p_adjusted(t) ∝ max(0, p_target(t) - p_draft(t))
//! ```
//!
//! which is the residual probability mass after the (failed) draft
//! proposal. Together these two rules make the joint distribution of
//! emitted tokens identical to running the target directly — no
//! quality loss, just an energy / latency win.

use crate::algorithms::AcceptanceDecision;
use crate::draft::TokenId;
use crate::error::{SpecDecodeError, SpecDecodeResult};
use crate::sampler::{SplitMix64, softmax};

/// Chen-et-al. probabilistic SpS with temperature scaling on both
/// draft and target before computing acceptance probabilities.
#[derive(Debug, Clone, Copy)]
pub struct SpsWithTemperature {
    /// Number of tokens the draft proposes per round.
    pub k: usize,
    /// Temperature applied to *both* draft and target logits before
    /// softmax. Set to 1.0 to leave logits untouched.
    pub temperature: f32,
}

impl SpsWithTemperature {
    /// Construct with a draft block size and a temperature.
    pub fn new(k: usize, temperature: f32) -> Self {
        Self {
            k: k.max(1),
            temperature: if temperature <= 0.0 { 1.0 } else { temperature },
        }
    }

    fn probs(&self, logits: &[f32]) -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|x| x / self.temperature).collect();
        softmax(&scaled)
    }

    /// Apply the SpS acceptance test.
    pub fn accept(
        &self,
        proposed: TokenId,
        draft_logits: &[f32],
        target_logits: &[f32],
        rng: &mut SplitMix64,
    ) -> SpecDecodeResult<AcceptanceDecision> {
        if draft_logits.is_empty() || target_logits.is_empty() {
            return Err(SpecDecodeError::EmptyVerification);
        }
        if draft_logits.len() != target_logits.len() {
            return Err(SpecDecodeError::VocabMismatch {
                draft: draft_logits.len(),
                target: target_logits.len(),
            });
        }
        let idx = proposed as usize;
        if idx >= target_logits.len() {
            return Err(SpecDecodeError::Sampling(
                "proposed token id out of range".into(),
            ));
        }
        let p_draft = self.probs(draft_logits)[idx];
        let p_target = self.probs(target_logits)[idx];
        if p_draft <= 0.0 {
            // Defensive — draft assigned zero mass to its own token.
            // Treat as accept to avoid divide-by-zero blow-ups; the
            // target's argmax will still drive the next round.
            return Ok(AcceptanceDecision::Accept);
        }
        let ratio = (p_target / p_draft).min(1.0);
        let r = rng.next_f32();
        Ok(if r <= ratio {
            AcceptanceDecision::Accept
        } else {
            AcceptanceDecision::Reject
        })
    }

    /// Sample from the adjusted distribution `max(0, p_target -
    /// p_draft) / Z`. Called by the orchestrator on rejection to pick
    /// the replacement token.
    pub fn sample_adjusted(
        &self,
        draft_logits: &[f32],
        target_logits: &[f32],
        rng: &mut SplitMix64,
    ) -> SpecDecodeResult<TokenId> {
        if draft_logits.len() != target_logits.len() {
            return Err(SpecDecodeError::VocabMismatch {
                draft: draft_logits.len(),
                target: target_logits.len(),
            });
        }
        let p_draft = self.probs(draft_logits);
        let p_target = self.probs(target_logits);
        let mut adjusted: Vec<f32> = p_target
            .iter()
            .zip(p_draft.iter())
            .map(|(pt, pd)| (pt - pd).max(0.0))
            .collect();
        let z: f32 = adjusted.iter().sum();
        if z <= 0.0 {
            // Distributions agree on every token — fall back to the
            // target's argmax.
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in target_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            return Ok(best as TokenId);
        }
        for x in &mut adjusted {
            *x /= z;
        }
        let r = rng.next_f32();
        let mut cum = 0.0f32;
        for (i, &p) in adjusted.iter().enumerate() {
            cum += p;
            if r <= cum {
                return Ok(i as TokenId);
            }
        }
        Ok((adjusted.len() - 1) as TokenId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_certain_when_target_dominates_draft() {
        // p_target(0) >> p_draft(0)  ->  ratio = 1, always accept.
        let sps = SpsWithTemperature::new(4, 1.0);
        let draft = vec![0.0, 5.0, 5.0]; // p_draft(0) ~ tiny
        let target = vec![10.0, 0.0, 0.0]; // p_target(0) ~ 1
        let mut rng = SplitMix64::new(1);
        for _ in 0..16 {
            assert_eq!(
                sps.accept(0, &draft, &target, &mut rng).expect("non-empty"),
                AcceptanceDecision::Accept
            );
        }
    }

    #[test]
    fn reject_likely_when_draft_dominates_target() {
        // p_draft(0) >> p_target(0) -> ratio tiny -> mostly reject.
        let sps = SpsWithTemperature::new(4, 1.0);
        let draft = vec![10.0, 0.0, 0.0];
        let target = vec![0.0, 5.0, 5.0];
        let mut rng = SplitMix64::new(1);
        let mut rejects = 0;
        for _ in 0..200 {
            if sps.accept(0, &draft, &target, &mut rng).expect("non-empty")
                == AcceptanceDecision::Reject
            {
                rejects += 1;
            }
        }
        assert!(rejects > 150, "expected mostly rejects, got {rejects}/200");
    }

    #[test]
    fn vocab_mismatch_is_error() {
        let sps = SpsWithTemperature::new(4, 1.0);
        let mut rng = SplitMix64::new(1);
        assert!(sps.accept(0, &[0.0, 1.0], &[0.0, 1.0, 2.0], &mut rng).is_err());
    }

    #[test]
    fn sample_adjusted_avoids_zero_mass_tokens() {
        let sps = SpsWithTemperature::new(4, 1.0);
        // Target and draft agree on tokens 1 and 2; target prefers 0,
        // draft prefers 3 -> adjusted should pick 0 most of the time.
        let draft = vec![0.0, 1.0, 1.0, 10.0];
        let target = vec![10.0, 1.0, 1.0, 0.0];
        let mut rng = SplitMix64::new(7);
        let mut hits_zero = 0;
        for _ in 0..200 {
            if sps.sample_adjusted(&draft, &target, &mut rng).expect("ok") == 0 {
                hits_zero += 1;
            }
        }
        assert!(hits_zero > 150, "expected mostly token 0, got {hits_zero}/200");
    }
}
