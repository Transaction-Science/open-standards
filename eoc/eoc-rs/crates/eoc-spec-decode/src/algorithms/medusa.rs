//! Medusa — parallel decoding heads on a frozen target.
//!
//! Medusa (Cai, Li, Geng, Peng, Lee, Zhang, Dao 2024, *Medusa: Simple
//! LLM Inference Acceleration Framework with Multiple Decoding Heads*)
//! attaches several small MLP heads to the final layer of a frozen
//! target model. Each head predicts the token at a *different* future
//! offset (head `i` predicts the token at position `t + i`). Verification
//! uses a tree-attention pass over the cartesian product of the heads'
//! top-k candidates and accepts the longest matching prefix.
//!
//! ## Why this is a stub
//!
//! Medusa's heads live *inside* the target model's compute graph —
//! they share its embedding matrix and run on the same activation
//! buffers. The orchestration trait surface in this crate
//! deliberately doesn't expose those internals, so a real Medusa
//! implementation lives behind a local backend that ships its own
//! pre-trained heads (e.g. the Medusa-2 finetune of Llama-3-8B).
//!
//! The skeleton below pins the configuration the orchestrator needs
//! to dispatch correctly so a future implementation slots in without
//! touching the orchestrator.

use async_trait::async_trait;

use crate::draft::{DraftModel, DraftSequence};
use crate::error::{SpecDecodeError, SpecDecodeResult};

/// Configuration / skeleton for a Medusa draft head pack.
#[derive(Debug, Clone)]
pub struct LocalMedusaDraft {
    /// Number of Medusa heads — also the maximum block size per
    /// verification pass.
    pub heads: usize,
    /// Top-k candidates kept per head before tree-attention
    /// verification. Larger values raise acceptance rates but
    /// quadratically grow the verification budget.
    pub top_k_per_head: usize,
    /// Maximum tree-attention candidates evaluated in one verification
    /// pass.
    pub max_tree_candidates: usize,
    /// Stable identifier used in logs / receipts.
    pub name: String,
}

impl LocalMedusaDraft {
    /// Construct a Medusa config skeleton matching the Medusa-1
    /// reference paper (5 heads, top-10 per head, 42-candidate tree).
    pub fn new(name: impl Into<String>, heads: usize) -> Self {
        Self {
            heads: heads.max(1),
            top_k_per_head: 10,
            max_tree_candidates: 42,
            name: name.into(),
        }
    }

    /// Override the per-head top-k.
    pub fn with_top_k_per_head(mut self, k: usize) -> Self {
        self.top_k_per_head = k.max(1);
        self
    }

    /// Override the verification-tree size.
    pub fn with_max_tree_candidates(mut self, m: usize) -> Self {
        self.max_tree_candidates = m.max(1);
        self
    }
}

#[async_trait]
impl DraftModel for LocalMedusaDraft {
    async fn propose(&self, _prefix: &str, _k: usize) -> SpecDecodeResult<DraftSequence> {
        Err(SpecDecodeError::StubAlgorithm("medusa"))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_propose_errors_with_named_algorithm() {
        let m = LocalMedusaDraft::new("medusa-llama3", 5);
        let err = m.propose("hello", 5).await.expect_err("stub must error");
        assert!(matches!(err, SpecDecodeError::StubAlgorithm("medusa")));
    }

    #[test]
    fn defaults_match_medusa_1() {
        let m = LocalMedusaDraft::new("medusa", 5);
        assert_eq!(m.heads, 5);
        assert_eq!(m.top_k_per_head, 10);
        assert_eq!(m.max_tree_candidates, 42);
    }

    #[test]
    fn builder_overrides_clamp_to_one() {
        let m = LocalMedusaDraft::new("medusa", 5)
            .with_top_k_per_head(0)
            .with_max_tree_candidates(0);
        assert_eq!(m.top_k_per_head, 1);
        assert_eq!(m.max_tree_candidates, 1);
    }
}
