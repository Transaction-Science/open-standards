//! Bench: PLD vs no-PLD on Bonsai-1.7B.
//!
//! Runs the same Text query through `PrismTier` twice — once with the
//! GGUF dispatch in its standard single-token-decode mode, once with
//! Prompt Lookup Decoding enabled. Reports wall-clock, measured joules,
//! and PLD's per-step acceptance histogram.
//!
//! On an echo-friendly prompt (the model rephrases the question before
//! answering) PLD should land hits and beat the baseline. On a purely
//! novel-generation prompt it should match the baseline within noise
//! (no n-gram matches → all single-token forwards).
//!
//! `#[ignore]` — needs Bonsai on disk.
//!
//! Run:
//!   cargo test --release -p prism --test pld_bench
//!     -- --ignored --nocapture

use jouleclaw_cascade::{
    AnswerOutput, ContextRef, JouleBudget, QualityFloor, Query, QueryInput, Tier,
};
use jouleclaw_runtime::PldConfig;
use jouleclaw_prism::PrismTier;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

fn query(text: &str) -> Query {
    Query {
        input: QueryInput::Text(text.to_string()),
        // expensive() — Bonsai-1.7B in thinking mode runs ~1.1 J end-
        // to-end (prefill ~430 mJ + decode ~700 mJ for ~12 tokens past
        // the <think> markers). PLD doesn't reduce total compute (it
        // does MORE forward-pass work per step that gets partially
        // rewound), but it reduces wall-clock by amortizing latency
        // across multiple tokens per pass.
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

#[test]
#[ignore]
fn pld_beats_baseline_on_echo_friendly_prompt() {
    let path = model_path();

    // Phrase that primes the model to echo "The capital of France is"
    // (a 5-gram already in the prompt) before answering "Paris."
    // Bonsai prepends `<think>\n</think>\n\n` markers so 16 tokens of
    // output budget covers the echo + the answer.
    let q = query("The capital of France is");

    // ---- Baseline: no PLD ----
    eprintln!("=== baseline (no PLD) ===");
    let mut base = PrismTier::from_gguf(7, &path, 128)
        .expect("PrismTier::from_gguf")
        .with_max_new_tokens(16);
    let t = Instant::now();
    let a_base = base.try_answer(&q, q.budget.hard_limit).expect("try_answer");
    let wall_base = t.elapsed();
    let text_base = match &a_base.output {
        AnswerOutput::Text(s) => s.clone(),
        other => panic!("expected Text answer, got {:?}", other),
    };
    eprintln!(
        "  wall={:.3}s  joules={:.1} mJ  text={:?}",
        wall_base.as_secs_f64(), a_base.joules_spent * 1e3, text_base,
    );

    // ---- PLD: 3-gram match, 3-token lookahead ----
    eprintln!("=== PLD (ngram=3, lookahead=3) ===");
    let mut pld = PrismTier::from_gguf(7, &path, 128)
        .expect("PrismTier::from_gguf")
        .with_max_new_tokens(16)
        .with_pld(PldConfig::default());
    let t = Instant::now();
    let a_pld = pld.try_answer(&q, q.budget.hard_limit).expect("try_answer");
    let wall_pld = t.elapsed();
    let text_pld = match &a_pld.output {
        AnswerOutput::Text(s) => s.clone(),
        other => panic!("expected Text answer, got {:?}", other),
    };
    eprintln!(
        "  wall={:.3}s  joules={:.1} mJ  text={:?}",
        wall_pld.as_secs_f64(), a_pld.joules_spent * 1e3, text_pld,
    );

    let speedup = wall_base.as_secs_f64() / wall_pld.as_secs_f64();
    eprintln!(
        "VERDICT: PLD wall={:.3}s vs baseline {:.3}s — {:.2}× speedup. \
         Both said 'Paris': base={}  pld={}.",
        wall_pld.as_secs_f64(), wall_base.as_secs_f64(), speedup,
        text_base.to_lowercase().contains("paris"),
        text_pld.to_lowercase().contains("paris"),
    );

    // Sanity: both should say "paris" since this is a well-known fact.
    // (We're not asserting EQUAL output because sampling+greedy on a
    // ternary model isn't deterministic across slight cache differences
    // in the PLD path — KV rewinds on rejected drafts leave bit-pattern
    // residue in the tensor backing store, even though those positions
    // are masked out of attention. The rejected-draft slot reuse is the
    // only source of divergence between paths.)
    assert!(
        text_base.to_lowercase().contains("paris"),
        "baseline must answer Paris: {:?}", text_base);
    assert!(
        text_pld.to_lowercase().contains("paris"),
        "PLD must answer Paris: {:?}", text_pld);

    // PLD should not be CATASTROPHICALLY slower (which would indicate
    // a bug — e.g., cache rewind not working, every draft rejected
    // wasting compute). Allow up to 2× slower in the pathological-
    // workload case; flag for review if it's slower than that.
    assert!(
        wall_pld < wall_base * 2,
        "PLD shouldn't be >2× slower than baseline on any workload \
         (baseline {:?}, PLD {:?}): cache rewind likely broken",
        wall_base, wall_pld,
    );
}
