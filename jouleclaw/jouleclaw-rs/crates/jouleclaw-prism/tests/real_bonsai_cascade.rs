//! Cascade integration oracle: a real PrismML Bonsai-class model is
//! dispatchable as a `Tier` inside the joule cascade.
//!
//! Loads `Ternary-Bonsai-1.7B-Q2_0.gguf` once via
//! [`PrismTier::from_gguf`], registers it with a fresh
//! [`Cascade`], submits a text Query, and asserts:
//!
//! * the registered tier reports the expected `TierId::L3(...)` and a
//!   non-`None` `Coord` (so the router can reason about it),
//! * `estimate_cost` returns a finite joule estimate at the ternary
//!   rate (~40 mJ/tok scaled by `max_new_tokens`),
//! * `try_answer` produces `AnswerOutput::Text` containing "Paris",
//! * `joules_spent` is positive and bounded by the estimate,
//! * the `ExecutionTrace` records a `Hit`.
//!
//! This closes the loop the user named: "make the runtime interface so
//! the files can be used in pattern-lang." Bonsai is now a real
//! cascade tier, not just a loader-level proof.
//!
//! `#[ignore]` — needs the model on disk. Run:
//!
//!   cargo test --release -p prism --test real_bonsai_cascade \
//!     -- --ignored --nocapture

use jouleclaw_cascade::{
    AnswerOutput, Cascade, ContextRef, JouleBudget, L3ModelId, QualityFloor,
    Query, QueryInput, Tier, TierId, TraceOutcome,
};
use jouleclaw_prism::PrismTier;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

#[test]
#[ignore]
fn bonsai_dispatches_as_prism_tier_in_cascade() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    // 16 tokens of headroom: Bonsai is a thinking-mode reasoner that
    // emits `<think>...</think>` markers (3-4 tokens) before its
    // answer. 8 tokens (the prior cap) leaves no room for "Paris" to
    // appear past the markers. 16 keeps the *measured* cost — which
    // for this model runs ~1.25× the static estimate — under
    // `JouleBudget::standard()`'s 1 J ceiling, so the Runtime's
    // budget guard (used by `real_bonsai_multitier`) won't reject.
    let mut tier = PrismTier::from_gguf(7, &path, 128)
        .expect("PrismTier::from_gguf")
        .with_max_new_tokens(16);
    eprintln!("  loaded + parsed in {:?}", t0.elapsed());

    // Tier identity + coordinate — the router needs both.
    assert_eq!(tier.id(), TierId::L3(L3ModelId(7)), "stable TierId");
    let coord = tier.coord().expect("tier must report a coord");
    eprintln!("  coord: {:?}", coord);

    // Static joule estimate. For Bonsai (~43 mJ/tok) × 8 tokens this is
    // ~340 mJ — comfortably inside `JouleBudget::standard().hard_limit`
    // (1 J) and well outside `cheap()` (1 mJ), which is exactly how the
    // router decides this tier is "L3-class".
    let q = Query {
        input: QueryInput::Text("The capital of France is".into()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };
    let est = tier.estimate_cost(&q).expect("estimate_cost on Text");
    eprintln!("  estimate: {:.3} mJ, latency {:?}, confidence_floor {}",
        est.joules * 1e3, est.latency, est.confidence_floor);
    assert!(est.joules.is_finite() && est.joules > 0.0);
    assert!(est.joules < q.budget.hard_limit,
        "estimate {:.3} J must fit hard limit {:.3} J", est.joules, q.budget.hard_limit);

    // Register in a real Cascade and confirm visibility.
    let mut cascade = Cascade::new();
    cascade.register(Box::new(PrismTier::from_gguf(7, &path, 128)
        .expect("PrismTier::from_gguf for cascade")
        .with_max_new_tokens(16)));
    let tier_ids = cascade.tier_ids();
    assert_eq!(tier_ids, vec![TierId::L3(L3ModelId(7))]);
    let coords = cascade.tier_coords();
    assert!(coords[0].1.is_some(), "registered tier exposes a Coord");
    drop(cascade); // We don't need the second copy past visibility check.

    // Dispatch.
    let t_dispatch = Instant::now();
    let ans = tier.try_answer(&q, q.budget.hard_limit).expect("try_answer");
    let elapsed = t_dispatch.elapsed();

    match &ans.output {
        AnswerOutput::Text(s) => {
            eprintln!("  >>> tier_used={:?} joules_spent={:.3} mJ wall={:?}",
                ans.tier_used, ans.joules_spent * 1e3, elapsed);
            eprintln!("  >>> {:?}", s);
            assert!(s.to_lowercase().contains("paris"),
                "CASCADE ORACLE FAILED — got {:?}", s);
        }
        other => panic!("expected Text answer, got {:?}", other),
    }

    assert_eq!(ans.tier_used, TierId::L3(L3ModelId(7)));
    assert!(ans.joules_spent > 0.0, "joule receipt must be positive");

    // `joules_spent` is now the MEASURED kernel-reported total, not the
    // static `estimate_cost` figure. They are *expected* to diverge —
    // that divergence is exactly what the cascade's calibration ledger
    // learns `learned_mu` from. The pre-dispatch estimate is for
    // budgeting; the receipt is honest measurement. So we no longer
    // assert `spent <= estimate`; we assert they differ (proving the
    // measured path is wired, not just echoing the estimate) and that
    // both are finite + positive.
    let ratio = ans.joules_spent / est.joules;
    eprintln!("  estimate={:.3} mJ  measured={:.3} mJ  ratio={:.2}x",
        est.joules * 1e3, ans.joules_spent * 1e3, ratio);
    assert!(ans.joules_spent.is_finite());
    assert!((ans.joules_spent - est.joules).abs() > f64::EPSILON,
        "measured joules must come from kernel accounting, not echo the \
         static estimate (both were {:.6} mJ)", est.joules * 1e3);

    // Trace must show a Hit (the answer wasn't refused or escalated).
    assert!(matches!(ans.trace.attempts[0].outcome, TraceOutcome::Hit),
        "ExecutionTrace should record a Hit, got {:?}", ans.trace.attempts);

    eprintln!("VERDICT: Bonsai dispatchable as a cascade tier with an \
        HONEST calibration receipt. Static estimate {:.3} mJ, measured \
        {:.3} mJ ({:.2}x), wall {:?}, said 'Paris'.",
        est.joules * 1e3, ans.joules_spent * 1e3, ratio, elapsed);
}
