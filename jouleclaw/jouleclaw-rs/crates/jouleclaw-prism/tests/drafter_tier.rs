//! Production-surface test: `PrismTier::with_drafter(...)` glues
//! Bonsai-1.7B as drafter onto Bonsai-4B target, dispatched through
//! the cascade as a single tier.
//!
//! Validates:
//!   - The builder loads both models and rejects mismatched vocab
//!     sizes at build time.
//!   - `try_answer` routes through `jouleclaw_runtime::extend_with_drafter`
//!     when a drafter is configured, producing a correct answer.
//!   - The drafter path produces semantically-correct output (contains
//!     "Paris" for the canonical capital query).
//!
//! `#[ignore]` — needs Bonsai 1.7B and 4B GGUFs on disk.
//!
//! Run:
//!   cargo test --release -p prism --test drafter_tier
//!     -- --ignored --nocapture

use jouleclaw_cascade::{
    AnswerOutput, ContextRef, JouleBudget, QualityFloor, Query, QueryInput, Tier,
};
use jouleclaw_prism::PrismTier;
use std::time::Instant;

fn bonsai_path(n: &str) -> String {
    format!("../../models/{n}")
}

fn text_query(s: &str) -> Query {
    Query {
        input: QueryInput::Text(s.to_string()),
        // Drafter path runs BOTH models — total joules scales up.
        // expensive() is the right ceiling.
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

#[test]
#[ignore]
fn prism_tier_with_drafter_answers_correctly() {
    let target_path = bonsai_path("ternary-bonsai-4b-q2_0.gguf");
    let drafter_path = bonsai_path("ternary-bonsai-1.7b-q2_0.gguf");
    eprintln!("target:  {target_path}");
    eprintln!("drafter: {drafter_path}");

    let t0 = Instant::now();
    let mut tier = PrismTier::from_gguf(42, &target_path, 128)
        .expect("target load")
        .with_max_new_tokens(16)
        .with_drafter(&drafter_path, 128)
        .expect("drafter load")
        .with_drafter_lookahead(4);
    eprintln!("loaded both models in {:?}", t0.elapsed());

    let q = text_query("The capital of France is");
    let t = Instant::now();
    let ans = tier.try_answer(&q, q.budget.hard_limit).expect("answer");
    let wall = t.elapsed();

    let text = match &ans.output {
        AnswerOutput::Text(s) => s.clone(),
        other => panic!("expected Text, got {:?}", other),
    };
    eprintln!(
        "VERDICT: wall={:.3}s  joules_spent={:.1} mJ  text={:?}",
        wall.as_secs_f64(), ans.joules_spent * 1e3, text,
    );

    assert!(text.to_lowercase().contains("paris"),
        "PrismTier with drafter must reach 'Paris': {text:?}");
    assert!(ans.joules_spent > 1e-3,
        "drafter path should spend > 1 mJ (target+drafter compute); got {} J",
        ans.joules_spent);
}

#[test]
#[ignore]
fn with_drafter_rejects_mismatched_vocab() {
    // Bonsai-1.7B (qwen3, vocab ~152K) and lfm2-350M (different vocab
    // family) must not pair. The build-time vocab-size check catches
    // it before any inference attempt.
    let target_path = bonsai_path("ternary-bonsai-1.7b-q2_0.gguf");
    let mismatch_path = bonsai_path("lfm2-350m-q8_0.gguf");

    // Both files must exist for this test to be meaningful — skip if
    // either is missing.
    if !std::path::Path::new(&mismatch_path).exists() {
        eprintln!("skipping: {mismatch_path} not present");
        return;
    }

    let tier = PrismTier::from_gguf(1, &target_path, 128)
        .expect("target load")
        .with_drafter(&mismatch_path, 128);
    assert!(tier.is_err(),
        "expected vocab-mismatch rejection when pairing Bonsai with LFM2");
    eprintln!("vocab mismatch correctly rejected");
}
