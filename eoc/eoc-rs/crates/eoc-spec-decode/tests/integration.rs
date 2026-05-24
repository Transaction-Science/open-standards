//! Integration tests for `eoc-spec-decode`.
//!
//! These complement the unit tests in each module by exercising the
//! crate's public surface end-to-end: a real `SpeculativeDecoder`, a
//! real `SpeculativeBackend`, and a property test against the
//! orchestrator's invariants.

use std::sync::Arc;

use eoc_core::{JouleSource, Query, Stage as StageKind};
use eoc_neural::NeuralBackend;
use eoc_spec_decode::{
    Generation, GreedySampler, LookaheadDecoding, SpeculativeAlgorithm, SpeculativeBackend,
    SpeculativeDecoder, SpsWithTemperature, SyntheticDraft, SyntheticTarget, VanillaSpeculative,
};
use eoc_spec_decode::orchestrator::baseline_generate;
use proptest::prelude::*;

fn build_decoder(
    acceptance: f32,
    k: usize,
    max_new_tokens: usize,
    seed: u64,
) -> (SpeculativeDecoder, Arc<SyntheticTarget>) {
    let draft = Arc::new(SyntheticDraft::new("d", vec![7; 32], 16, 100));
    let target = Arc::new(SyntheticTarget::new("t", acceptance, 50_000, 16, seed));
    let dec = SpeculativeDecoder::new(
        draft,
        target.clone(),
        SpeculativeAlgorithm::Vanilla(VanillaSpeculative::new(k)),
        max_new_tokens,
        Box::new(GreedySampler),
    )
    .expect("ok");
    (dec, target)
}

#[tokio::test]
async fn vanilla_synthetic_full_acceptance_minimum_forward_passes() {
    let (dec, target) = build_decoder(1.0, 4, 25, 1);
    let g: Generation = dec.generate("hello").await.expect("ok");
    assert_eq!(g.total_new_tokens, 25);
    // K=4 with full acceptance + 1 bonus token per pass = 5 per
    // round. ceil(25 / 5) = 5 forward passes.
    assert_eq!(g.target_forward_passes, 5);
    assert_eq!(target.forward_pass_count(), 5);
    assert!((g.acceptance_rate - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn vanilla_synthetic_80pct_acceptance_yields_speedup() {
    // K = 4, acceptance ~ 0.8, max_new_tokens = 200. With 80%
    // per-position acceptance, expected accepted prefix length per
    // round ≈ 0.8 + 0.8^2 + 0.8^3 + 0.8^4 ≈ 2.36, plus one bonus =
    // ~3.36 new tokens per target forward pass. Speedup vs baseline
    // (~1 token per forward pass) is ~3.36×.
    let (dec, _) = build_decoder(0.8, 4, 200, 42);
    let g = dec.generate("hello").await.expect("ok");
    assert_eq!(g.total_new_tokens, 200);

    let baseline_target = Arc::new(SyntheticTarget::new("t", 0.0, 50_000, 16, 99));
    let (baseline_emitted, baseline_passes, _) =
        baseline_generate(baseline_target, "hello", 200, 0)
            .await
            .expect("ok");
    assert_eq!(baseline_emitted, 200);
    // Baseline is 1 emitted token per pass (replacement only).
    assert_eq!(baseline_passes, 200);

    let speedup = baseline_passes as f32 / g.target_forward_passes as f32;
    assert!(
        speedup >= 2.5,
        "expected ≥2.5× speedup, got {speedup}× ({} vs {} passes)",
        baseline_passes,
        g.target_forward_passes
    );
}

#[tokio::test]
async fn joule_attribution_sums_components() {
    let (dec, _) = build_decoder(0.6, 4, 40, 7);
    let g = dec.generate("hi").await.expect("ok");
    assert_eq!(g.total_joules, g.draft_joules + g.target_joules);
    // Both components must be non-zero in a non-trivial run.
    assert!(g.draft_joules > 0);
    assert!(g.target_joules > 0);
}

#[tokio::test]
async fn wrapper_plugs_into_cascade_as_neural_backend() {
    let draft = Arc::new(SyntheticDraft::new("d", vec![3; 16], 16, 100));
    let target = Arc::new(SyntheticTarget::new("t", 0.7, 50_000, 16, 11));
    let wrapper = SpeculativeBackend::build(
        draft,
        target,
        SpeculativeAlgorithm::Vanilla(VanillaSpeculative::new(4)),
        16,
    )
    .expect("ok");
    let backend: &dyn NeuralBackend = &wrapper;
    let q = Query::new("hi there");
    let r = backend.infer(&q).await;
    assert_eq!(r.stage, StageKind::Neural);
    assert_eq!(r.joule_cost.source, JouleSource::Estimated);
    assert!(r.joule_cost.microjoules > 0);
    assert_eq!(r.query_id, q.id);
}

#[tokio::test]
async fn sps_with_temperature_runs_end_to_end() {
    // We can't drive SpS-with-temperature through the orchestrator's
    // synthetic target (the target uses its own acceptance Bernoulli,
    // not the SpS test), but we can verify the algorithm's
    // `accept` + `sample_adjusted` methods compose cleanly under
    // realistic conditions.
    let sps = SpsWithTemperature::new(4, 1.0);
    let draft = vec![0.0, 5.0, 1.0, 0.0];
    let target = vec![3.0, 3.0, 1.0, 0.0];
    let mut rng = eoc_spec_decode::sampler::SplitMix64::new(13);
    let mut accepts = 0;
    let mut rejects = 0;
    for _ in 0..500 {
        match sps.accept(1, &draft, &target, &mut rng).expect("ok") {
            eoc_spec_decode::algorithms::AcceptanceDecision::Accept => accepts += 1,
            eoc_spec_decode::algorithms::AcceptanceDecision::Reject => rejects += 1,
        }
    }
    // p_target(1) / p_draft(1) ≈ softmax(target)[1] / softmax(draft)[1]
    // is well under 1, so we expect more rejects than accepts but
    // some of both.
    assert!(accepts > 0);
    assert!(rejects > 0);
}

#[test]
fn lookahead_pure_rust_decode_no_draft_needed() {
    let mut la = LookaheadDecoding::new(4, 3);
    la.ingest(&[10, 20, 30, 40, 50]);
    let proposals = la.propose_from_cache(&[10, 20, 30]);
    // Last n-1 = (20, 30) -> we observed 40 next.
    assert_eq!(proposals, vec![40]);

    // After ingesting another continuation for (20, 30), we should
    // see both candidates within the window.
    la.ingest(&[20, 30, 60]);
    let proposals = la.propose_from_cache(&[10, 20, 30]);
    assert_eq!(proposals, vec![40, 60]);
}

// ---------------------------------------------------------------------
// Property test: regardless of the per-position accept/reject decisions
// the synthetic target makes, the decoder must emit exactly
// `max_new_tokens` tokens. No off-by-one in either direction.

proptest! {
    #[test]
    fn decoder_always_emits_exactly_max_new_tokens(
        acceptance in 0.0f32..=1.0f32,
        k in 1usize..=8,
        max_new_tokens in 1usize..=64,
        seed in any::<u64>(),
    ) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        let result: eoc_spec_decode::SpecDecodeResult<Generation> = runtime.block_on(async {
            let (dec, _) = build_decoder(acceptance, k, max_new_tokens, seed);
            dec.generate("p").await
        });
        let g = result.expect("ok");
        prop_assert_eq!(g.total_new_tokens, max_new_tokens);
        prop_assert_eq!(g.tokens.len(), max_new_tokens);
    }
}
