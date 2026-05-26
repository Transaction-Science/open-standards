//! Cascade integration oracle: real LFM2-VL multimodal model dispatched
//! as an `LmmTier` inside the joule cascade.
//!
//! Loads `lfm2.5-vl-450m-q8_0.gguf` (text backbone) + matching mmproj
//! via `LmmTier::from_lfm_vl_gguf`, submits a `Multimodal` Query with a
//! real JPEG, asserts the returned Answer is `Text` non-empty, and
//! reports the kernel joule receipt.
//!
//! Slow (no KV cache on the multimodal path yet); 4 tokens of caption
//! is enough to prove the cascade integration without burning minutes.
//!
//! `#[ignore]` — needs the two GGUFs + a test image on disk.
//!
//!   cargo test --release -p lmm --test cascade_lfm2_vl \
//!     -- --ignored --nocapture

use jouleclaw_cascade::{
    AnswerOutput, ContextRef, JouleBudget, QualityFloor,
    Query, QueryInput, Tier, TraceOutcome,
};
use jouleclaw_lmm::LmmTier;
use std::time::Instant;

fn text_path() -> String {
    std::env::var("JOULE_LFM2VL_TEXT")
        .unwrap_or_else(|_| "../../models/lfm2.5-vl-450m-q8_0.gguf".into())
}
fn mmproj_path() -> String {
    std::env::var("JOULE_LFM2VL_MMPROJ")
        .unwrap_or_else(|_| "../../models/lfm2.5-vl-450m-mmproj-q8_0.gguf".into())
}
fn image_path() -> String {
    std::env::var("JOULE_VL_IMAGE")
        .unwrap_or_else(|_| "../../data/LARC/assets/collection.jpg".into())
}

#[test]
#[ignore]
fn lfm2_vl_dispatches_as_lmm_tier_with_real_caption() {
    let t0 = Instant::now();
    let max_new = std::env::var("JOULE_VL_MAX_NEW")
        .ok().and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);
    let mut tier = LmmTier::from_lfm_vl_gguf(
        12, text_path(), mmproj_path(),
    ).expect("LmmTier::from_lfm_vl_gguf");
    tier.max_new_tokens = max_new;
    eprintln!("  loaded LFM2-VL tier in {:?}  max_new={}",
        t0.elapsed(), max_new);

    let image_bytes = std::fs::read(image_path()).expect("read test image");
    eprintln!("  image: {} bytes", image_bytes.len());

    let q = Query {
        input: QueryInput::Multimodal {
            text: "Describe this image:".into(),
            images: vec![image_bytes],
            audio: vec![],
        },
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let est = tier.estimate_cost(&q).expect("estimate_cost");
    eprintln!("  estimate: {:.3} mJ", est.joules * 1e3);
    assert!(est.joules.is_finite() && est.joules > 0.0);

    let t1 = Instant::now();
    // Bypass the budget guard; the no-cache loop is slow and the
    // measured joules will far exceed the standard 1 J budget at
    // captioning lengths. The point of this oracle is correctness, not
    // efficiency — that's a follow-up (cache-aware multimodal prefill).
    let ans = tier.try_answer(&q, f64::INFINITY).expect("try_answer");
    let elapsed = t1.elapsed();

    match &ans.output {
        AnswerOutput::Text(s) => {
            eprintln!("  >>> joules_spent={:.3} mJ wall={:?}",
                ans.joules_spent * 1e3, elapsed);
            eprintln!("  >>> {:?}", s);
            assert!(!s.is_empty(),
                "caption empty — likely a stuck-EOS or dead-vision bug");
            assert!(s.chars().any(|c| c.is_ascii_alphabetic()),
                "caption has no letters: {:?}", s);
        }
        other => panic!("expected Text answer, got {:?}", other),
    }

    assert!(ans.joules_spent > 0.0, "joule receipt must be positive");
    assert!(matches!(ans.trace.attempts[0].outcome, TraceOutcome::Hit),
        "ExecutionTrace should record a Hit, got {:?}", ans.trace.attempts);

    eprintln!("VERDICT: LFM2-VL-450M dispatchable through LmmTier with a \
        real multimodal caption. Measured {:.3} mJ over {} new tokens, \
        wall {:?}.", ans.joules_spent * 1e3, max_new, elapsed);
}
