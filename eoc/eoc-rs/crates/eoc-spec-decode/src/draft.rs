//! Draft model trait.
//!
//! A *draft* is a cheap autoregressive model that proposes the next `K`
//! tokens given a prefix. Quality is allowed to be mediocre — the
//! target is the source of truth and corrects the draft on rejection —
//! but throughput must be much higher than the target's, otherwise the
//! technique loses energy instead of saving it.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::SpecDecodeResult;

/// Vocabulary identifier for one token. We use `u32` to match the
/// sampler trait in `eoc-local` and the conventions of `tokenizers`,
/// llama.cpp, and ONNX Runtime.
pub type TokenId = u32;

/// A draft proposal — the tokens the draft model thinks come next and
/// the per-position logits it used to pick them. The orchestrator hands
/// both to the target so the target can apply Leviathan's acceptance
/// criterion (greedy or probabilistic) without re-running the draft.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftSequence {
    /// Proposed tokens in order. Length is at most `k`; a draft is
    /// allowed to give up early (e.g. it hit an EOS token).
    pub tokens: Vec<TokenId>,
    /// Per-position logit vectors as produced by the draft. Same length
    /// as `tokens`. Each inner vector has length `vocab_size`.
    pub logits: Vec<Vec<f32>>,
    /// Energy attributable to producing this proposal, in micro-joules.
    /// Backends with a hardware counter report a measured reading;
    /// vendor-API backends report an estimate.
    pub draft_joules: u64,
}

impl DraftSequence {
    /// Build a sequence and assert the vector lengths match.
    pub fn new(tokens: Vec<TokenId>, logits: Vec<Vec<f32>>, draft_joules: u64) -> Self {
        debug_assert_eq!(
            tokens.len(),
            logits.len(),
            "DraftSequence: tokens and logits must have matching length"
        );
        Self {
            tokens,
            logits,
            draft_joules,
        }
    }

    /// Number of proposed tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// True if no tokens were proposed.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// A pluggable draft model.
///
/// Implementations are `Send + Sync` so the orchestrator can call into
/// them from a Tokio task. Backends that wrap a non-`Send` runtime
/// should put a mutex / oneshot around it; this crate doesn't impose a
/// concurrency story on the draft.
#[async_trait]
pub trait DraftModel: Send + Sync {
    /// Propose `k` tokens following `prefix`. Implementations are free
    /// to return fewer (e.g. they hit a stop token), but must never
    /// return more.
    async fn propose(&self, prefix: &str, k: usize) -> SpecDecodeResult<DraftSequence>;

    /// Stable identifier for logging / receipts (e.g. `"llama-3-1b"`).
    fn name(&self) -> &str;
}
