//! End-to-end oracle: LFM2-350M (Liquid AI hybrid arch — alternating
//! attention + shortconv recurrent layers) dispatched as a `PrismTier`
//! in the cascade.
//!
//! What this exercises that no prior test did:
//!   * the shortconv decode-mode dispatch in
//!     `build_decode_step_graph_inplace_const_block`
//!   * the per-layer `ShortConvStateCache` take/replace dance through
//!     `run_inplace_step_cached`
//!   * LFM2's `token_embd_norm.weight` final norm (not `output_norm`)
//!   * streaming over a hybrid graph where each layer's input/output
//!     name set depends on whether the layer is recurrent or attention
//!
//! Smoke gate — not caption-quality. If the streaming shortconv state
//! is wrong, the model emits garbage (low-vocab IDs, NaN logits, or
//! catastrophic divergence after a few tokens). We assert the easier
//! invariants here and let the deeper correctness oracle live in
//! `lmm`'s `real_lfm2_vl.rs`.
//!
//! `#[ignore]` — needs the model on disk.
//!
//!   cargo test --release -p prism --test real_lfm2 \
//!     -- --ignored --nocapture

use jouleclaw_cascade::{
    AnswerOutput, ContextRef, JouleBudget, QualityFloor,
    Query, QueryInput, Tier,
};
use jouleclaw_prism::PrismTier;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_LFM2")
        .unwrap_or_else(|_| "../../models/lfm2-350m-q8_0.gguf".to_string())
}

#[test]
#[ignore]
fn lfm2_350m_dispatches_through_prism_tier() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    let mut tier = PrismTier::from_gguf(11, &path, 128)
        .expect("PrismTier::from_gguf on LFM2-350M")
        .with_max_new_tokens(16);
    eprintln!("  loaded + parsed in {:?}", t0.elapsed());

    let coord = tier.coord().expect("tier must report a coord");
    eprintln!("  coord: {:?}", coord);

    let q = Query {
        input: QueryInput::Text("The capital of France is".into()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };
    let est = tier.estimate_cost(&q).expect("estimate_cost on Text");
    eprintln!("  estimate: {:.3} mJ", est.joules * 1e3);
    assert!(est.joules.is_finite() && est.joules > 0.0);

    let t_dispatch = Instant::now();
    let ans = tier.try_answer(&q, q.budget.hard_limit).expect("try_answer");
    let elapsed = t_dispatch.elapsed();

    match &ans.output {
        AnswerOutput::Text(s) => {
            eprintln!("  >>> joules_spent={:.3} mJ wall={:?}",
                ans.joules_spent * 1e3, elapsed);
            eprintln!("  >>> {:?}", s);
            // Real shortconv-state or FFN-dispatch bugs produce
            // degenerate repetition ("Reg Reg Reg regardless...") or
            // empty strings. A model that knows the capital of France
            // should mention "Paris" — the same gate Bonsai's cascade
            // test uses.
            assert!(s.to_lowercase().contains("paris"),
                "LFM2 streaming-decode did not produce 'Paris' — \
                 likely a shortconv/state bug. got {:?}", s);
        }
        other => panic!("expected Text answer, got {:?}", other),
    }

    assert!(ans.joules_spent.is_finite() && ans.joules_spent > 0.0,
        "joule receipt must be positive and finite");

    eprintln!("VERDICT: LFM2-350M (hybrid attn+shortconv) ran end-to-end \
        through PrismTier. Streaming shortconv state cache is live: \
        prefill → decode loop produced finite logits, sampled in-vocab \
        ids, decoded to text. Measured {:.3} mJ in {:?}.",
        ans.joules_spent * 1e3, elapsed);
}
