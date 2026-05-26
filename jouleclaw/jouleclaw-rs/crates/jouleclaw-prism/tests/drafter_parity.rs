//! Drafter spec-decode tests against real Bonsai weights.
//!
//! ## Self-drafting parity (the substrate check)
//!
//! When drafter == target, every draft the drafter proposes must be
//! the same token the target predicts (same weights, same sampling).
//! Acceptance should be 100% — every step accepts all K+1 tokens.
//! Output should be identical to the non-spec-decode run.
//!
//! If acceptance is anything less than full, the spec-decode
//! substrate has a bug (cache rewinding, history tracking, sampler
//! penalty handling — any of these can diverge the drafter's view
//! from the target's). This test catches it.
//!
//! ## Real drafter (separate test, ignored)
//!
//! Smaller Bonsai (1.7B) drafts for bigger (4B). Acceptance < 100%
//! but should average 2-3 out of K=4. Wall-clock should improve.
//!
//! `#[ignore]` on both — need models on disk.
//!
//! Run:
//!   cargo test --release -p prism --test drafter_parity
//!     -- --ignored --nocapture

use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{
    extend_with_drafter, ChatTemplate, Conversation, DrafterConfig,
    GenerateConfig, KvCacheKind, Runtime, TokenizerKind,
};
use jouleclaw_loader_gguf::sample::SamplingConfig;
use std::time::Instant;

fn bonsai_path(n: &str) -> String {
    format!("../../models/{n}")
}

fn cfg(max_new: usize) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens: max_new,
        sampling: SamplingConfig::greedy(),
        add_bos: false,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: Some(128),
        stop_strings: vec![],
    }
}

#[test]
#[ignore]
fn self_drafting_matches_single_token_baseline() {
    // Drafter == target: Bonsai-1.7B drafting for itself. Same
    // weights, same sampling — the drafter's prediction at each
    // step must equal the target's row-i sample. Every draft should
    // be accepted (modulo EOS truncation on the final step).
    //
    // The strongest substrate check: output token sequences from
    // spec-decode and single-token-decode must be IDENTICAL. Any
    // divergence indicates a hist / cache / acceptance bug.
    let path = bonsai_path("ternary-bonsai-1.7b-q2_0.gguf");
    eprintln!("loading {path}");
    let model = jouleclaw_loader_gguf::read_gguf_file(&path).expect("model");
    let vocab = Vocab::from_gguf(&model).expect("vocab");
    let chat_template = ChatTemplate::detect_from_model(&model);

    let max_seq = 128;
    let text = "The capital of France is";
    let prompt_tokens: Vec<u32> = match chat_template {
        Some(template) => jouleclaw_runtime::encode_user_turn(template, text, &vocab, true),
        None => vocab.encode_bpe_regex(text, true),
    };
    eprintln!("prompt tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);
    let gc = cfg(16);

    // ── Baseline: single-token decode ──
    let mut baseline_conv = Conversation::with_runtime(&model, &vocab, max_seq, Runtime::boot())
        .expect("baseline conv");
    let baseline_stream = baseline_conv.extend_tokens(prompt_tokens.clone(), &gc)
        .expect("baseline extend");
    let mut baseline_tokens = Vec::new();
    for st in baseline_stream {
        baseline_tokens.push(st.expect("baseline tok").id);
    }
    let baseline_text = vocab.decode_bpe(&baseline_tokens);
    eprintln!("baseline ({} tok): {:?}", baseline_tokens.len(), baseline_text);
    eprintln!("baseline tokens: {:?}", baseline_tokens);

    // ── Self-drafting ──
    let mut target = Conversation::with_runtime(&model, &vocab, max_seq, Runtime::boot())
        .expect("target conv");
    let mut drafter = Conversation::with_runtime(&model, &vocab, max_seq, Runtime::boot())
        .expect("drafter conv");
    let spec = DrafterConfig { max_lookahead: 4 };
    let t = Instant::now();
    let out = extend_with_drafter(&mut target, &mut drafter,
        prompt_tokens, &gc, &spec).expect("drafter run");
    let wall = t.elapsed();
    let text_out = vocab.decode_bpe(&out.tokens);
    eprintln!("self-drafting ({} tok): {:?}", out.tokens.len(), text_out);
    eprintln!("self-drafting tokens: {:?}", out.tokens);
    eprintln!("accepted_per_step: {:?}", out.accepted_per_step);
    eprintln!(
        "mean_acceptance: {:.2}/{} target_joules: {:.1} mJ  drafter_joules: {:.1} mJ  wall: {:?}",
        out.mean_acceptance(), spec.max_lookahead + 1,
        out.target_joules * 1e3, out.drafter_joules * 1e3, wall,
    );

    // The bit-equality check: same model, same sampling. The
    // generated token sequence MUST be identical between baseline
    // and self-drafting. Any divergence flags a substrate bug.
    assert_eq!(out.tokens, baseline_tokens,
        "self-drafting tokens diverged from single-token baseline.\n\
         baseline: {baseline_tokens:?}\n\
         self-draft: {:?}", out.tokens);

    // All non-final steps must hit K+1 acceptance (with same weights,
    // the drafter never "guesses wrong"). The final step may emit
    // fewer if EOS lands inside the accepted block.
    let n_steps = out.accepted_per_step.len();
    for (i, &acc) in out.accepted_per_step.iter().enumerate() {
        if i < n_steps - 1 {
            assert_eq!(acc, spec.max_lookahead + 1,
                "step {i} (non-final) accepted {acc}/{} — drafter and \
                 target should agree perfectly given identical weights",
                spec.max_lookahead + 1);
        }
        // Final step: no assertion. EOS may truncate it.
    }
}

#[test]
#[ignore]
fn bonsai_17b_drafting_bonsai_4b() {
    // Real-drafter bench: smaller Bonsai (1.7B) drafts for bigger
    // Bonsai (4B). Both qwen3 arch, same tokenizer family.
    let target_path = bonsai_path("ternary-bonsai-4b-q2_0.gguf");
    let drafter_path = bonsai_path("ternary-bonsai-1.7b-q2_0.gguf");
    eprintln!("target:  {target_path}");
    eprintln!("drafter: {drafter_path}");

    let target_model = jouleclaw_loader_gguf::read_gguf_file(&target_path).expect("target");
    let target_vocab = Vocab::from_gguf(&target_model).expect("target vocab");
    let drafter_model = jouleclaw_loader_gguf::read_gguf_file(&drafter_path).expect("drafter");
    let drafter_vocab = Vocab::from_gguf(&drafter_model).expect("drafter vocab");

    // Sanity: vocab sizes match (qwen3 family uses the same
    // tokenizer across all sizes).
    assert_eq!(target_vocab.len(), drafter_vocab.len(),
        "tokenizer size mismatch: target {} drafter {} — these aren't \
         a paired drafter family", target_vocab.len(), drafter_vocab.len());

    let chat_template = ChatTemplate::detect_from_model(&target_model);
    let max_seq = 128;
    let mut target = Conversation::with_runtime(
        &target_model, &target_vocab, max_seq, Runtime::boot()).expect("target conv");
    let mut drafter = Conversation::with_runtime(
        &drafter_model, &drafter_vocab, max_seq, Runtime::boot()).expect("drafter conv");

    let text = "The capital of France is";
    let prompt_tokens: Vec<u32> = match chat_template {
        Some(template) => jouleclaw_runtime::encode_user_turn(template, text, &target_vocab, true),
        None => target_vocab.encode_bpe_regex(text, true),
    };

    let gc = cfg(16);
    let spec = DrafterConfig { max_lookahead: 4 };

    // ── Baseline: target alone, no spec decode ──
    eprintln!("=== baseline: Bonsai-4B alone ===");
    let mut target_alone = Conversation::with_runtime(
        &target_model, &target_vocab, max_seq, Runtime::boot()).expect("alone");
    let t = Instant::now();
    let alone_stream = target_alone.extend_tokens(prompt_tokens.clone(), &gc)
        .expect("alone extend");
    let mut alone_tokens = Vec::new();
    for st in alone_stream {
        match st {
            Ok(t) => alone_tokens.push(t.id),
            Err(e) => panic!("baseline error: {e}"),
        }
    }
    let alone_wall = t.elapsed();
    let alone_text = target_vocab.decode_bpe(&alone_tokens);
    eprintln!("  baseline ({} tok): wall={:?}  joules={:.1} mJ  text={:?}",
        alone_tokens.len(), alone_wall,
        target_alone.cumulative_joules() * 1e3, alone_text);

    // ── Drafter spec decode ──
    eprintln!("=== drafter: Bonsai-1.7B → Bonsai-4B (K=4) ===");
    let t = Instant::now();
    let out = extend_with_drafter(&mut target, &mut drafter,
        prompt_tokens, &gc, &spec).expect("drafter run");
    let drafter_wall = t.elapsed();
    let drafter_text = target_vocab.decode_bpe(&out.tokens);
    eprintln!(
        "  drafter ({} tok): wall={:?}  target={:.1} mJ  drafter={:.1} mJ  text={:?}",
        out.tokens.len(), drafter_wall,
        out.target_joules * 1e3, out.drafter_joules * 1e3, drafter_text,
    );
    eprintln!(
        "  accepted_per_step: {:?}  mean: {:.2}/{} (K+1)",
        out.accepted_per_step, out.mean_acceptance(), spec.max_lookahead + 1,
    );

    let speedup = alone_wall.as_secs_f64() / drafter_wall.as_secs_f64();
    eprintln!(
        "VERDICT: speedup={:.2}× (drafter wall={:.3}s vs baseline {:.3}s). \
         Both contain 'Paris': baseline={}  drafter={}.",
        speedup, drafter_wall.as_secs_f64(), alone_wall.as_secs_f64(),
        alone_text.to_lowercase().contains("paris"),
        drafter_text.to_lowercase().contains("paris"),
    );

    // Correctness: both should reach "Paris".
    assert!(alone_text.to_lowercase().contains("paris"),
        "baseline (Bonsai-4B) must say Paris: {alone_text:?}");
    assert!(drafter_text.to_lowercase().contains("paris"),
        "drafter must say Paris: {drafter_text:?}");

    // Sanity: drafter mean acceptance is > 1 (i.e., spec decode is
    // doing SOMETHING, not just falling back to single-token).
    assert!(out.mean_acceptance() > 1.0,
        "drafter accepted {:.2} avg — drafts aren't being verified. \
         Acceptance broken?", out.mean_acceptance());
}
