//! Synthetic draft & target models for tests, demos, and CI.
//!
//! The "real" backends in `eoc-vendor-api` and `eoc-local` need
//! credentials, GPUs, or both. To exercise the orchestration layer
//! deterministically we ship a pair of cheap synthetic models:
//!
//! * [`SyntheticDraft`] proposes tokens according to a configurable
//!   policy. It always emits its `proposed_token` as the argmax of a
//!   one-hot logit vector, charging `per_token_microjoules` per
//!   proposed token.
//!
//! * [`SyntheticTarget`] verifies a proposal by deciding,
//!   per-position, whether it agrees with the draft. The decision is
//!   driven by a seeded PRNG with a configurable
//!   [`acceptance_rate`](SyntheticTarget::acceptance_rate). Every
//!   verification charges one fixed `per_pass_microjoules` cost — the
//!   single forward pass — regardless of how many tokens the draft
//!   proposed. This is the headline energy property that makes
//!   speculative decoding work.
//!
//! Both backends are fully deterministic given a seed.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::draft::{DraftModel, DraftSequence, TokenId};
use crate::error::{SpecDecodeError, SpecDecodeResult};
use crate::sampler::SplitMix64;
use crate::target::{TargetModel, VerificationResult};

/// Synthetic draft model — proposes tokens from a pre-seeded sequence.
pub struct SyntheticDraft {
    /// Token sequence the draft proposes, cycling on wrap-around.
    pub script: Vec<TokenId>,
    /// Vocabulary size — width of the one-hot logit vectors.
    pub vocab: usize,
    /// Energy per proposed token.
    pub per_token_microjoules: u64,
    /// Identifier surfaced via [`DraftModel::name`].
    pub name: String,
    /// Mutex-guarded cursor into `script`; kept inside the type so the
    /// trait can take `&self`.
    cursor: Mutex<usize>,
}

impl SyntheticDraft {
    /// Build a draft that cycles through `script`.
    pub fn new(
        name: impl Into<String>,
        script: Vec<TokenId>,
        vocab: usize,
        per_token_microjoules: u64,
    ) -> Self {
        Self {
            script,
            vocab: vocab.max(1),
            per_token_microjoules,
            name: name.into(),
            cursor: Mutex::new(0),
        }
    }

    fn one_hot(&self, tok: TokenId) -> Vec<f32> {
        let mut v = vec![0.0f32; self.vocab];
        let idx = (tok as usize) % self.vocab;
        v[idx] = 10.0;
        v
    }
}

#[async_trait]
impl DraftModel for SyntheticDraft {
    async fn propose(&self, _prefix: &str, k: usize) -> SpecDecodeResult<DraftSequence> {
        if k == 0 {
            return Err(SpecDecodeError::Config("k must be non-zero".into()));
        }
        if self.script.is_empty() {
            return Err(SpecDecodeError::Config(
                "SyntheticDraft script is empty".into(),
            ));
        }
        let mut cursor = self
            .cursor
            .lock()
            .map_err(|e| SpecDecodeError::Backend(format!("cursor poisoned: {e}")))?;
        let mut tokens = Vec::with_capacity(k);
        let mut logits = Vec::with_capacity(k);
        for _ in 0..k {
            let tok = self.script[*cursor % self.script.len()];
            *cursor += 1;
            tokens.push(tok);
            logits.push(self.one_hot(tok));
        }
        Ok(DraftSequence::new(
            tokens,
            logits,
            self.per_token_microjoules.saturating_mul(k as u64),
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Synthetic target model — verifies a proposal by Bernoulli-flipping
/// at each position with probability `acceptance_rate`.
pub struct SyntheticTarget {
    /// Per-position probability of accepting the draft's token.
    pub acceptance_rate: f32,
    /// Energy of one verification forward pass — independent of `k`.
    pub per_pass_microjoules: u64,
    /// Vocabulary size — width of the logit vectors returned for
    /// replacement / bonus tokens.
    pub vocab: usize,
    /// Replacement / bonus token (always emitted as the argmax of a
    /// one-hot vector). When `replacement_token == proposed`, the
    /// vanilla algorithm sees the draft's token as the target's
    /// argmax and accepts.
    pub replacement_token: TokenId,
    /// Identifier surfaced via [`TargetModel::name`].
    pub name: String,
    rng: Mutex<SplitMix64>,
    forward_passes: Mutex<u32>,
}

impl SyntheticTarget {
    /// Build a synthetic target with a fixed acceptance rate and seed.
    pub fn new(
        name: impl Into<String>,
        acceptance_rate: f32,
        per_pass_microjoules: u64,
        vocab: usize,
        seed: u64,
    ) -> Self {
        Self {
            acceptance_rate: acceptance_rate.clamp(0.0, 1.0),
            per_pass_microjoules,
            vocab: vocab.max(1),
            replacement_token: 0,
            name: name.into(),
            rng: Mutex::new(SplitMix64::new(seed)),
            forward_passes: Mutex::new(0),
        }
    }

    /// Override the replacement / bonus token (default: 0).
    pub fn with_replacement_token(mut self, tok: TokenId) -> Self {
        self.replacement_token = tok;
        self
    }

    /// Number of times `verify` has been called — i.e. the number of
    /// target forward passes. Used by tests to assert the
    /// orchestration achieves the expected speedup.
    pub fn forward_pass_count(&self) -> u32 {
        *self.forward_passes.lock().expect("not poisoned")
    }

    fn logits_for(&self, tok: TokenId) -> Vec<f32> {
        let mut v = vec![0.0f32; self.vocab];
        let idx = (tok as usize) % self.vocab;
        v[idx] = 10.0;
        v
    }
}

#[async_trait]
impl TargetModel for SyntheticTarget {
    async fn verify(
        &self,
        _prefix: &str,
        proposed: &DraftSequence,
    ) -> SpecDecodeResult<VerificationResult> {
        let mut rng = self
            .rng
            .lock()
            .map_err(|e| SpecDecodeError::Backend(format!("rng poisoned: {e}")))?;
        {
            let mut fp = self
                .forward_passes
                .lock()
                .map_err(|e| SpecDecodeError::Backend(format!("fp counter poisoned: {e}")))?;
            *fp += 1;
        }

        let mut accepted = Vec::with_capacity(proposed.len());
        let mut rejected_at = None;
        for (i, &tok) in proposed.tokens.iter().enumerate() {
            let r = rng.next_f32();
            if r < self.acceptance_rate {
                accepted.push(tok);
            } else {
                rejected_at = Some(i);
                break;
            }
        }

        let replacement = Some(self.replacement_token);

        Ok(VerificationResult {
            accepted_tokens: accepted,
            rejected_at,
            replacement_token: replacement,
            target_joules: self.per_pass_microjoules,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Public access to the logits a `SyntheticTarget` would produce at a
/// given position — used by tests / synthetic orchestrators that need
/// to walk the full per-position logit grid (vanilla, SpS). Real
/// backends carry this implicitly inside `verify`.
impl SyntheticTarget {
    /// Build a `vocab`-wide one-hot logit vector for the synthetic
    /// target's replacement token. Useful when constructing
    /// hand-shaped scenarios for tests.
    pub fn replacement_logits(&self) -> Vec<f32> {
        self.logits_for(self.replacement_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn synthetic_draft_proposes_script() {
        let d = SyntheticDraft::new("d", vec![1, 2, 3, 4], 16, 100);
        let p = d.propose("prefix", 6).await.expect("ok");
        // Script cycles: 1,2,3,4,1,2.
        assert_eq!(p.tokens, vec![1, 2, 3, 4, 1, 2]);
        assert_eq!(p.draft_joules, 600);
        assert_eq!(p.logits.len(), 6);
        for (i, tok) in p.tokens.iter().enumerate() {
            let argmax = p.logits[i]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN"))
                .map(|(j, _)| j as TokenId)
                .expect("non-empty");
            assert_eq!(argmax, *tok);
        }
    }

    #[tokio::test]
    async fn synthetic_target_acceptance_rate_is_deterministic() {
        let t = SyntheticTarget::new("t", 1.0, 50_000, 16, 42);
        let d = SyntheticDraft::new("d", vec![5; 4], 16, 100);
        let p = d.propose("prefix", 4).await.expect("ok");
        let v = t.verify("prefix", &p).await.expect("ok");
        assert_eq!(v.accepted_tokens.len(), 4);
        assert!(v.rejected_at.is_none());
        assert_eq!(v.target_joules, 50_000);
        assert_eq!(t.forward_pass_count(), 1);
    }

    #[tokio::test]
    async fn synthetic_target_zero_acceptance_rejects_first() {
        let t = SyntheticTarget::new("t", 0.0, 50_000, 16, 42);
        let d = SyntheticDraft::new("d", vec![5; 4], 16, 100);
        let p = d.propose("prefix", 4).await.expect("ok");
        let v = t.verify("prefix", &p).await.expect("ok");
        assert_eq!(v.accepted_tokens.len(), 0);
        assert_eq!(v.rejected_at, Some(0));
    }
}
