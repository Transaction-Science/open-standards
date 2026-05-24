//! `NeuralBackend` wrapper around a [`SpeculativeDecoder`].
//!
//! This is the piece that lets speculative decoding plug into the EOC
//! cascade as a drop-in replacement for stage 4 (neural inference).
//! The wrapper hides the draft / target / algorithm machinery behind
//! the same trait every other `eoc-neural` backend implements; the
//! cascade keeps treating "neural" as a single black box, but its
//! joule cost drops by the speculative-decoding factor.
//!
//! Joule attribution sums draft and target costs:
//!
//! ```text
//! cost(query) = sum(draft.propose calls) + sum(target.verify calls)
//! ```
//!
//! Because [`JouleCost`](eoc_core::JouleCost) carries a `source` field,
//! the wrapper has to pick one when the two backends disagree. The
//! rule:
//!
//! * Both `Measured` → `Measured`
//! * Otherwise → `Estimated`
//!
//! This matches the cascade's general convention that a single
//! estimated component poisons the whole reading.

use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage as StageKind};
use eoc_neural::NeuralBackend;

use crate::algorithms::SpeculativeAlgorithm;
use crate::draft::DraftModel;
use crate::orchestrator::SpeculativeDecoder;
use crate::sampler::{GreedySampler, Sampler};
use crate::target::TargetModel;

/// Wraps a [`SpeculativeDecoder`] as a [`NeuralBackend`].
///
/// The wrapper is generic over the draft and target traits so callers
/// retain access to the concrete types after construction (handy for
/// inspection, metrics, hot-swap). The cascade only ever sees it as
/// `&dyn NeuralBackend`.
pub struct SpeculativeBackend<D: DraftModel + 'static, T: TargetModel + 'static> {
    /// The underlying decoder.
    pub decoder: SpeculativeDecoder,
    /// Joule-source policy: when either backend reports an estimate,
    /// the wrapper downgrades the combined cost to `Estimated`. Most
    /// deployments leave this `true`; flip it off only if you know
    /// every backend you're wiring in carries a hardware reading.
    pub coerce_estimated_on_mixed: bool,
    _draft: std::marker::PhantomData<D>,
    _target: std::marker::PhantomData<T>,
}

impl<D: DraftModel + 'static, T: TargetModel + 'static> SpeculativeBackend<D, T> {
    /// Construct a wrapper from an existing [`SpeculativeDecoder`].
    pub fn new(decoder: SpeculativeDecoder) -> Self {
        Self {
            decoder,
            coerce_estimated_on_mixed: true,
            _draft: std::marker::PhantomData,
            _target: std::marker::PhantomData,
        }
    }

    /// Convenience builder that wires the components together.
    pub fn build(
        draft: Arc<D>,
        target: Arc<T>,
        algorithm: SpeculativeAlgorithm,
        max_new_tokens: usize,
    ) -> crate::error::SpecDecodeResult<Self> {
        let sampler: Box<dyn Sampler> = Box::new(GreedySampler);
        let decoder = SpeculativeDecoder::new(
            draft as Arc<dyn DraftModel>,
            target as Arc<dyn TargetModel>,
            algorithm,
            max_new_tokens,
            sampler,
        )?;
        Ok(Self::new(decoder))
    }
}

#[async_trait]
impl<D: DraftModel + 'static, T: TargetModel + 'static> NeuralBackend
    for SpeculativeBackend<D, T>
{
    async fn infer(&self, q: &Query) -> Response {
        match self.decoder.generate(&q.prompt).await {
            Ok(g) => {
                // We don't have per-call provenance for the synthetic /
                // vendor backends, so apply the documented coercion
                // policy: assume `Estimated` unless we know better.
                let source = if self.coerce_estimated_on_mixed {
                    JouleSource::Estimated
                } else {
                    JouleSource::Measured
                };
                let cost = JouleCost {
                    microjoules: g.total_joules,
                    source,
                };
                Response::new(q.id, g.text, StageKind::Neural, cost)
            }
            Err(e) => {
                // The neural stage is the answerer of last resort —
                // returning an error would crash the cascade. Surface
                // the failure as a payload and zero-cost reading.
                Response::new(
                    q.id,
                    format!("[spec-decode error] {e}"),
                    StageKind::Neural,
                    JouleCost::estimated(0),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::vanilla::VanillaSpeculative;
    use crate::synthetic::{SyntheticDraft, SyntheticTarget};

    #[tokio::test]
    async fn wrapper_is_a_neural_backend() {
        let draft = Arc::new(SyntheticDraft::new("d", vec![1, 2, 3, 4], 16, 100));
        let target = Arc::new(SyntheticTarget::new("t", 1.0, 50_000, 16, 1));
        let wrapper = SpeculativeBackend::build(
            draft,
            target,
            SpeculativeAlgorithm::Vanilla(VanillaSpeculative::new(4)),
            8,
        )
        .expect("ok");
        // Coerce to `&dyn NeuralBackend` to prove the trait bound.
        let backend: &dyn NeuralBackend = &wrapper;
        let q = Query::new("hello world");
        let r = backend.infer(&q).await;
        assert_eq!(r.stage, StageKind::Neural);
        assert!(!r.payload.is_empty());
        assert!(r.joule_cost.microjoules > 0);
    }
}
