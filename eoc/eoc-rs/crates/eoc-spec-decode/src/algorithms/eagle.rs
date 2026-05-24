//! EAGLE — Efficient Auto-aggressive Generation with Layer Extrapolation.
//!
//! EAGLE (Li, Wei, Zhang & Zhang 2024, *EAGLE: Speculative Sampling
//! Requires Rethinking Feature Uncertainty*) trains a tiny draft head
//! that consumes the *target model's own penultimate-layer features*
//! and predicts the next token from them. Because the head sees the
//! target's activations directly, the draft and target distributions
//! agree on most tokens — published acceptance rates run 75–85% even
//! at `k = 6`, well above what an independent draft can achieve.
//!
//! ## Why this is a stub
//!
//! EAGLE requires read access to a layer of the target's residual
//! stream. The [`crate::draft::DraftModel`] /
//! [`crate::target::TargetModel`] traits intentionally hide every
//! backend's internals — they take a `&str` prefix and return token
//! ids — so an EAGLE draft head can't read the target's activations
//! through them. A real EAGLE implementation has two options:
//!
//! 1. Plug into a *local* backend (llama.cpp, MLX, ONNX) that exposes
//!    hidden-state buffers between layers, and run the EAGLE head as
//!    an extra forward pass against those buffers.
//! 2. Co-locate the draft head inside the target backend itself
//!    (Medusa-style) and have the backend return both target tokens
//!    and EAGLE drafts in one call.
//!
//! Both options are out of scope for the orchestration layer; the
//! [`LocalEagleDraft`] struct below pins down the configuration
//! surface so a future implementation can slot in without touching
//! the orchestrator. Calling `propose` on it today returns
//! [`SpecDecodeError::StubAlgorithm`].

use async_trait::async_trait;

use crate::draft::{DraftModel, DraftSequence};
use crate::error::{SpecDecodeError, SpecDecodeResult};

/// Configuration / skeleton for an EAGLE draft head.
///
/// Fields mirror the hyperparameters in the reference EAGLE-2
/// implementation. The struct deliberately holds no native handles
/// (an `Arc<TargetModel>` or feature-tap channel) so it compiles on
/// every platform the `default` feature set targets; backends that
/// implement it for real should wrap it in their own newtype.
#[derive(Debug, Clone)]
pub struct LocalEagleDraft {
    /// Block size — how many tokens the head proposes per round.
    pub k: usize,
    /// Index of the target-model layer whose hidden state the head
    /// reads. Negative values count from the end (`-1` = penultimate
    /// layer, the EAGLE-1 default).
    pub feature_layer: i32,
    /// Whether to use EAGLE-2's tree-attention verification. When
    /// `false`, falls back to chain (vanilla) verification.
    pub tree_attention: bool,
    /// Stable identifier used in logs / receipts.
    pub name: String,
}

impl LocalEagleDraft {
    /// Construct an EAGLE config skeleton.
    pub fn new(name: impl Into<String>, k: usize) -> Self {
        Self {
            k: k.max(1),
            feature_layer: -1,
            tree_attention: true,
            name: name.into(),
        }
    }

    /// Set the feature-layer index (default `-1` = penultimate).
    pub fn with_feature_layer(mut self, layer: i32) -> Self {
        self.feature_layer = layer;
        self
    }

    /// Enable / disable EAGLE-2 tree attention.
    pub fn with_tree_attention(mut self, on: bool) -> Self {
        self.tree_attention = on;
        self
    }
}

#[async_trait]
impl DraftModel for LocalEagleDraft {
    async fn propose(&self, _prefix: &str, _k: usize) -> SpecDecodeResult<DraftSequence> {
        Err(SpecDecodeError::StubAlgorithm("eagle"))
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
        let e = LocalEagleDraft::new("eagle-1b", 4);
        let err = e.propose("hello", 4).await.expect_err("stub must error");
        assert!(matches!(err, SpecDecodeError::StubAlgorithm("eagle")));
    }

    #[test]
    fn config_defaults_match_eagle_1() {
        let e = LocalEagleDraft::new("eagle", 4);
        assert_eq!(e.feature_layer, -1);
        assert!(e.tree_attention);
    }

    #[test]
    fn config_builder_overrides() {
        let e = LocalEagleDraft::new("eagle", 4)
            .with_feature_layer(-2)
            .with_tree_attention(false);
        assert_eq!(e.feature_layer, -2);
        assert!(!e.tree_attention);
    }
}
