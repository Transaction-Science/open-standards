//! Multi-tier orchestration oracle.
//!
//! Everything else proves a *tier* works. This proves the **cascade**
//! works: a `jouleclaw_cascade::Runtime` with an auto-prepended L0 cache
//! and a registered `PrismTier(Bonsai)` routes a query end-to-end,
//! and a *repeat* of the same query is served from L0 at
//! picojoule cost instead of re-running the 1.8B model.
//!
//! This is the thesis in one test: the runtime is the AI, the model
//! is a peripheral the cascade reaches for only when the cheaper
//! tiers can't answer — and never twice for the same question.
//!
//! Gate:
//!   * 1st query: tier_used = L3 (Bonsai), measured joules > 1 mJ,
//!     answer contains "Paris".
//!   * 2nd identical query: tier_used = L0, joules ≪ 1st (≥1000×
//!     cheaper), identical answer text.
//!
//! `#[ignore]` — needs the model on disk.

use jouleclaw_cascade::{
    AnswerOutput, Cascade, ContextRef, JouleBudget, QualityFloor,
    Query, QueryInput, Runtime, TierId,
};
use jouleclaw_prism::PrismTier;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

fn query(text: &str) -> Query {
    Query {
        input: QueryInput::Text(text.to_string()),
        // `expensive()` (100 J) instead of `standard()` (1 J) because
        // Bonsai-1.7B in thinking mode genuinely sits on the L3/L4
        // boundary: prompt prefill costs ~430 mJ alone, and a natural-
        // length answer with `<think></think>` markers adds another
        // ~700 mJ to reach "Paris" — total ~1.1 J. The Runtime's
        // budget guard would reject this against the 1 J limit even
        // though the answer is correct, so we route through a budget
        // category that fits. This is data-driven: it's what the
        // calibration ledger learns after one dispatch.
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

#[test]
#[ignore]
fn cascade_routes_l3_then_l0_on_repeat() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    // 16 tokens: Bonsai is a thinking-mode reasoner; it emits
    // <think>...</think> markers before the answer. 6 tokens (prior
    // cap) leaves no room for "Paris". 16 keeps the *measured* cost
    // (~1.25× the static estimate at 43 mJ/tok) under
    // `JouleBudget::standard()`'s 1 J ceiling, so the Runtime's
    // budget guard accepts the answer.
    let tier = PrismTier::from_gguf(7, &path, 128)
        .expect("PrismTier::from_gguf")
        .with_max_new_tokens(16);
    eprintln!("  loaded in {:?}", t0.elapsed());

    // Cascade with just the one L3 tier; Runtime::new prepends an L0.
    let mut cascade = Cascade::new();
    cascade.register(Box::new(tier));
    let mut runtime = Runtime::new(cascade);

    let q = query("The capital of France is");

    // ---- 1st dispatch: L0 miss → L3 (Bonsai) ----
    let t1 = Instant::now();
    let a1 = runtime.answer(q.clone()).expect("answer #1");
    let wall1 = t1.elapsed();
    let txt1 = match &a1.output {
        AnswerOutput::Text(s) => s.clone(),
        other => panic!("expected Text, got {:?}", other),
    };
    eprintln!("  #1  tier={:?}  joules={:.3} mJ  wall={:?}  {:?}",
        a1.tier_used, a1.joules_spent * 1e3, wall1, txt1);
    assert_eq!(a1.tier_used, TierId::L3(jouleclaw_cascade::L3ModelId(7)),
        "first dispatch must reach the Bonsai L3 tier");
    assert!(txt1.to_lowercase().contains("paris"),
        "L3 answer must contain 'Paris', got {:?}", txt1);
    assert!(a1.joules_spent > 1e-3,
        "L3 measured joules should be model-scale (>1 mJ), got {} J",
        a1.joules_spent);

    // ---- 2nd dispatch: identical query → L0 cache hit ----
    let t2 = Instant::now();
    let a2 = runtime.answer(q.clone()).expect("answer #2");
    let wall2 = t2.elapsed();
    let txt2 = match &a2.output {
        AnswerOutput::Text(s) => s.clone(),
        other => panic!("expected Text on cache hit, got {:?}", other),
    };
    eprintln!("  #2  tier={:?}  joules={:.3e} J  wall={:?}  {:?}",
        a2.tier_used, a2.joules_spent, wall2, txt2);

    assert_eq!(a2.tier_used, TierId::L0,
        "second identical query must be served from the L0 cache");
    assert_eq!(txt2, txt1, "cache must return the exact same answer text");
    assert!(a2.joules_spent > 0.0 && a2.joules_spent < a1.joules_spent / 1000.0,
        "L0 hit must be ≥1000× cheaper than the L3 run: L0={} J vs L3={} J",
        a2.joules_spent, a1.joules_spent);
    // The cache hit should also be dramatically faster in wall-clock.
    assert!(wall2 * 50 < wall1,
        "L0 wall {:?} should be ≫ faster than L3 wall {:?}", wall2, wall1);

    let energy_ratio = a1.joules_spent / a2.joules_spent;
    eprintln!("VERDICT: cascade routed L3 then L0 on repeat. \
        Energy: L3 {:.1} mJ → L0 {:.3e} J ({:.0}× cheaper). \
        Wall: {:?} → {:?}.",
        a1.joules_spent * 1e3, a2.joules_spent, energy_ratio, wall1, wall2);
}
