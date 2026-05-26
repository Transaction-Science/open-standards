//! Bench: KV cache fp32 vs int8 on Bonsai-1.7B.
//!
//! Three things to verify:
//!   1. Cold-storage cache footprint shrinks ~4× (the lever's purpose).
//!   2. Output still contains "Paris" (correctness within quant noise).
//!   3. Wall-clock latency is within a measured bound: int8 trades
//!      latency for memory, and at Bonsai's K/V shape this runs
//!      ~20-40% slower than fp32 (scalar Rust quant/dequant on the
//!      full max_seq buffer per step). The trade-off pays off when
//!      cache memory is the bottleneck (long contexts, multi-model
//!      servers) — not when latency is.
//!
//! `#[ignore]` — needs the model on disk.
//!
//! Run:
//!   cargo test --release -p prism --test kv_quant_bench
//!     -- --ignored --nocapture

use jouleclaw_cascade::{
    AnswerOutput, ContextRef, JouleBudget, QualityFloor, Query, QueryInput, Tier,
};
use jouleclaw_loader_gguf::kv_cache_inplace::KvQuant;
use jouleclaw_prism::PrismTier;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

fn query(text: &str) -> Query {
    Query {
        input: QueryInput::Text(text.to_string()),
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

#[test]
#[ignore]
fn kv_quant_int8_shrinks_cache_and_preserves_answer() {
    let path = model_path();
    let q = query("The capital of France is");

    // ---- Baseline: fp32 KV cache ----
    eprintln!("=== KV fp32 (baseline) ===");
    let mut base = PrismTier::from_gguf(7, &path, 128)
        .expect("PrismTier::from_gguf")
        .with_max_new_tokens(16);
    let t = Instant::now();
    let a_base = base.try_answer(&q, q.budget.hard_limit).expect("try_answer");
    let wall_base = t.elapsed();
    let text_base = match &a_base.output {
        AnswerOutput::Text(s) => s.clone(),
        other => panic!("expected Text, got {:?}", other),
    };
    eprintln!(
        "  wall={:.3}s  joules={:.1} mJ  text={:?}",
        wall_base.as_secs_f64(), a_base.joules_spent * 1e3, text_base,
    );

    // ---- KV int8 ----
    eprintln!("=== KV int8 ===");
    let mut q8 = PrismTier::from_gguf(7, &path, 128)
        .expect("PrismTier::from_gguf")
        .with_max_new_tokens(16)
        .with_kv_quant(KvQuant::Int8);
    let t = Instant::now();
    let a_q8 = q8.try_answer(&q, q.budget.hard_limit).expect("try_answer");
    let wall_q8 = t.elapsed();
    let text_q8 = match &a_q8.output {
        AnswerOutput::Text(s) => s.clone(),
        other => panic!("expected Text, got {:?}", other),
    };
    eprintln!(
        "  wall={:.3}s  joules={:.1} mJ  text={:?}",
        wall_q8.as_secs_f64(), a_q8.joules_spent * 1e3, text_q8,
    );

    // ---- Cache footprint comparison (synthetic — we don't expose the
    //      cache out of the live Conversation, so we construct standalone
    //      caches with the same shape to demonstrate the ratio). ----
    use jouleclaw_loader_gguf::kv_cache_inplace::InPlaceKvCache;
    use jouleclaw_loader_gguf::read_gguf_file;
    let model = read_gguf_file(&path).expect("model");
    let cache_fp32 = InPlaceKvCache::for_model(&model, 128).expect("cache fp32");
    let cache_int8 = InPlaceKvCache::for_model_with_quant(&model, 128, KvQuant::Int8)
        .expect("cache int8");
    let bytes_fp32 = cache_fp32.cache_bytes();
    let bytes_int8 = cache_int8.cache_bytes();
    let ratio = bytes_fp32 as f64 / bytes_int8 as f64;
    eprintln!(
        "cache_bytes (max_seq=128): fp32={} ({:.2} MB), int8={} ({:.2} MB), ratio={:.2}×",
        bytes_fp32, bytes_fp32 as f64 / (1024.0 * 1024.0),
        bytes_int8, bytes_int8 as f64 / (1024.0 * 1024.0),
        ratio,
    );

    // Same demonstration at max_seq=2048 — where quant savings really
    // pay off because the cache grows linearly with seq length while
    // the model size stays put.
    let cache_fp32_long = InPlaceKvCache::for_model(&model, 2048).expect("cache fp32 long");
    let cache_int8_long = InPlaceKvCache::for_model_with_quant(&model, 2048, KvQuant::Int8)
        .expect("cache int8 long");
    eprintln!(
        "cache_bytes (max_seq=2048): fp32={:.2} MB, int8={:.2} MB, ratio={:.2}×",
        cache_fp32_long.cache_bytes() as f64 / (1024.0 * 1024.0),
        cache_int8_long.cache_bytes() as f64 / (1024.0 * 1024.0),
        cache_fp32_long.cache_bytes() as f64 / cache_int8_long.cache_bytes() as f64,
    );

    let wall_ratio = wall_q8.as_secs_f64() / wall_base.as_secs_f64();
    eprintln!(
        "VERDICT: KV int8 wall={:.3}s vs fp32 {:.3}s ({:.2}× wall ratio). \
         Cache footprint at max_seq=128: {:.2}× saved. Both said 'Paris': base={}  int8={}.",
        wall_q8.as_secs_f64(), wall_base.as_secs_f64(), wall_ratio,
        ratio,
        text_base.to_lowercase().contains("paris"),
        text_q8.to_lowercase().contains("paris"),
    );

    // Correctness: both should answer "Paris" (quant noise shouldn't
    // flip the model's prediction on a strong fact).
    assert!(
        text_base.to_lowercase().contains("paris"),
        "baseline must say Paris: {:?}", text_base);
    assert!(
        text_q8.to_lowercase().contains("paris"),
        "KV int8 must say Paris: {:?}", text_q8);

    // Memory: at least 3× savings (theoretical 4× minus scale overhead).
    assert!(ratio >= 3.0,
        "KV int8 should save ≥3× cache memory, got {ratio}×");

    // Latency: int8 path now routes through `run_inplace_step_sequential`,
    // which runs n_layers + 2 separate execute() calls per decode step
    // (one per layer + embed + head) instead of one monolithic call.
    // The trade buys the actual memory-savings goal (5.1 MB peak vs
    // fp32's 12 MB at Bonsai max_seq=128) — the old monolithic int8
    // path used MORE RAM than fp32 (16.6 MB) because each step
    // allocated 24 fresh fp32 working buffers.
    //
    // Measured wall ratio: 1.48-1.69× across runs (mean ~1.55×).
    // Threshold 1.9× to absorb run-to-run variance without flapping.
    assert!(wall_ratio < 1.9,
        "KV int8 wall {:?} >90% slower than fp32 {:?}: sequential \
         dispatch regression suspected — typical overhead is 50-70%",
         wall_q8, wall_base);
}
