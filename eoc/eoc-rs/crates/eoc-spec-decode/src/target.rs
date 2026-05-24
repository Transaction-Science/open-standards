//! Target model trait.
//!
//! The *target* is the large, expensive model whose distribution we
//! actually want to sample from. It receives a prefix plus the draft's
//! proposal and runs a single forward pass that produces logits at each
//! position; the orchestrator then walks left-to-right and either
//! accepts a draft token (when it agrees with the target's argmax /
//! passes the SpS acceptance test) or rejects it and substitutes the
//! target's own choice.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::draft::{DraftSequence, TokenId};
use crate::error::SpecDecodeResult;

/// Outcome of verifying a draft proposal against the target.
///
/// `accepted_tokens` is always a prefix of the draft. If the draft is
/// fully accepted, `rejected_at` is `None` and `replacement_token`
/// carries the target's *bonus* token — the one extra token the target
/// can emit thanks to running a single forward pass over `prefix +
/// proposal`. If the draft is rejected at position `i`,
/// `replacement_token` is the corrected token the target wants there.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Draft tokens the target accepted, in order.
    pub accepted_tokens: Vec<TokenId>,
    /// Position (within the draft proposal) at which the draft was
    /// rejected, or `None` if the entire proposal was accepted.
    pub rejected_at: Option<usize>,
    /// On rejection: the target's preferred token at `rejected_at`.
    /// On full acceptance: the target's bonus token, picked from the
    /// logits at position `k`.
    pub replacement_token: Option<TokenId>,
    /// Energy attributable to this verification pass, in micro-joules.
    pub target_joules: u64,
}

impl VerificationResult {
    /// Total number of *new* tokens this verification contributes to
    /// the output: the accepted prefix plus the replacement / bonus.
    pub fn new_token_count(&self) -> usize {
        self.accepted_tokens.len() + usize::from(self.replacement_token.is_some())
    }
}

/// A pluggable target model.
#[async_trait]
pub trait TargetModel: Send + Sync {
    /// Verify `proposed` against the target's own distribution given
    /// `prefix`. Implementations should run a *single* forward pass
    /// over `prefix ++ proposed.tokens` to keep the energy savings
    /// real; running one forward pass per draft token defeats the
    /// purpose.
    async fn verify(
        &self,
        prefix: &str,
        proposed: &DraftSequence,
    ) -> SpecDecodeResult<VerificationResult>;

    /// Stable identifier for logging / receipts.
    fn name(&self) -> &str;
}
