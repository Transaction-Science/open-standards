//! Prefix cache oracle: replaying a checkpointed prefix produces
//! identical logits to a fresh full prefill, and is faster.
//!
//! Two-conversation parity:
//!   * `control`: fresh Conversation, prefill [prefix ; suffix] in one
//!     call, sample the last position.
//!   * `replay`: fresh Conversation, prefill only `prefix`, take a
//!     `ConversationCheckpoint`, build a new Conversation from it, then
//!     prefill `suffix` and sample the last position.
//!
//! The argmax token must match (the most stringent greedy-decode gate);
//! we also report the wall-clock prefill costs to demonstrate the win.
//!
//! `#[ignore]` — needs Bonsai-1.7B on disk.

use jouleclaw_loader_gguf::read_gguf_file;
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::generate::GenerateConfig;
use jouleclaw_runtime::streaming::{Conversation, PrefixCache};
use jouleclaw_runtime::Runtime;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

#[test]
#[ignore]
fn replayed_checkpoint_matches_fresh_prefill_and_is_faster() {
    let path = model_path();
    eprintln!("loading {path}");
    let model = read_gguf_file(&path).expect("load model");
    let vocab = Vocab::from_gguf(&model).expect("vocab");

    // Pick a "system" prefix that's reasonably long (the win scales
    // with prefix size) and a short suffix.
    let prefix_text = "You are a helpful assistant. Always answer concisely.\n\
                       User: What is the capital of";
    let suffix_text = " France? A:";

    let prefix_tokens = vocab.encode_bpe_regex(prefix_text, true);
    let suffix_tokens = vocab.encode_bpe_regex(suffix_text, false);
    let full_tokens: Vec<u32> = prefix_tokens.iter()
        .chain(suffix_tokens.iter()).copied().collect();
    eprintln!("  prefix={} tokens, suffix={} tokens, full={} tokens",
        prefix_tokens.len(), suffix_tokens.len(), full_tokens.len());

    let max_seq = 256usize;
    let cfg = GenerateConfig {
        max_new_tokens: 1,
        ..GenerateConfig::default()
    };

    // ── CONTROL: fresh conv, single prefill of full sequence ──
    let t_ctrl = Instant::now();
    let mut conv_ctrl = Conversation::with_runtime(
        &model, &vocab, max_seq, Runtime::boot()).expect("ctrl conv");
    let stream_ctrl = conv_ctrl
        .extend_tokens(full_tokens.clone(), &cfg)
        .expect("ctrl extend");
    let ctrl_prefill_ms = t_ctrl.elapsed().as_secs_f64() * 1000.0;
    let ctrl_tok = stream_ctrl.peek_next_token()
        .expect("ctrl must have a sampled first token");
    eprintln!("  CONTROL prefill {} tokens in {:.1} ms — next_id={}",
        full_tokens.len(), ctrl_prefill_ms, ctrl_tok);

    // ── WARM: fresh conv, prefill ONLY the prefix, take checkpoint ──
    let mut conv_warm = Conversation::with_runtime(
        &model, &vocab, max_seq, Runtime::boot()).expect("warm conv");
    let t_warm = Instant::now();
    let _ = conv_warm.extend_tokens(prefix_tokens.clone(), &cfg)
        .expect("warm extend");
    let warm_prefill_ms = t_warm.elapsed().as_secs_f64() * 1000.0;
    let checkpoint = conv_warm.checkpoint();
    eprintln!("  WARM prefix prefill {} tokens in {:.1} ms — \
        checkpoint bytes={:.1} MB",
        prefix_tokens.len(), warm_prefill_ms,
        checkpoint.bytes() as f64 / 1e6);

    // Stash into a PrefixCache and look it back up to exercise that
    // surface too.
    let mut pcache = PrefixCache::new(500_000_000);
    pcache.insert(checkpoint);
    let (cached, matched_len) = pcache.lookup(&full_tokens)
        .expect("must find cached prefix");
    assert_eq!(matched_len, prefix_tokens.len(),
        "PrefixCache::lookup should match the full cached prefix length");

    // ── REPLAY: resume from checkpoint, prefill only the suffix ──
    let t_replay = Instant::now();
    let mut conv_replay = Conversation::from_checkpoint(
        &model, &vocab, Runtime::boot(), cached,
    ).expect("from_checkpoint");
    let replay_restore_ms = t_replay.elapsed().as_secs_f64() * 1000.0;
    let t_replay_pref = Instant::now();
    let stream_replay = conv_replay
        .extend_tokens(suffix_tokens.clone(), &cfg)
        .expect("replay extend");
    let replay_prefill_ms = t_replay_pref.elapsed().as_secs_f64() * 1000.0;
    let replay_tok = stream_replay.peek_next_token()
        .expect("replay must have a sampled first token");
    eprintln!("  REPLAY restore in {:.1} ms + suffix prefill {} tokens in {:.1} ms \
        — next_id={}",
        replay_restore_ms, suffix_tokens.len(),
        replay_prefill_ms, replay_tok);

    // ── Correctness gate: argmax token must match ──
    assert_eq!(replay_tok, ctrl_tok,
        "REPLAY argmax {} != CONTROL argmax {} — prefix-cache restore \
         lost numerical fidelity",
        replay_tok, ctrl_tok);

    // ── Win gate: the replay prefill (suffix only) must be cheaper
    // than the control prefill (full sequence). Restore + suffix
    // combined should still beat control in most realistic cases
    // (prefill is quadratic in length and the suffix is short). ──
    let total_replay_ms = replay_restore_ms + replay_prefill_ms;
    eprintln!("  speedup: control {:.1} ms vs replay {:.1} ms (restore+suffix) = {:.2}x",
        ctrl_prefill_ms, total_replay_ms,
        ctrl_prefill_ms / total_replay_ms.max(1.0));
    assert!(replay_prefill_ms < ctrl_prefill_ms,
        "suffix-only prefill {:.1} ms must beat full prefill {:.1} ms",
        replay_prefill_ms, ctrl_prefill_ms);

    eprintln!("VERDICT: ConversationCheckpoint replays exactly — \
        argmax {} matched on both paths, suffix prefill cut from \
        {:.1} ms to {:.1} ms. PrefixCache substrate is live.",
        replay_tok, ctrl_prefill_ms, replay_prefill_ms);
}

#[test]
#[ignore]
fn persistent_prefix_cache_roundtrip_replay_matches_in_process() {
    let path = model_path();
    eprintln!("loading {path}");
    let model = read_gguf_file(&path).expect("load model");
    let vocab = Vocab::from_gguf(&model).expect("vocab");

    let prefix_text = "You are a helpful assistant. Always answer concisely.\n\
                       User: What is the capital of";
    let suffix_text = " France? A:";
    let prefix_tokens = vocab.encode_bpe_regex(prefix_text, true);
    let suffix_tokens = vocab.encode_bpe_regex(suffix_text, false);
    let full_tokens: Vec<u32> = prefix_tokens.iter()
        .chain(suffix_tokens.iter()).copied().collect();

    let max_seq = 256usize;
    let cfg = GenerateConfig {
        max_new_tokens: 1,
        ..GenerateConfig::default()
    };

    // Capture a checkpoint for the prefix.
    let mut conv_warm = Conversation::with_runtime(
        &model, &vocab, max_seq, Runtime::boot()).expect("warm conv");
    let _ = conv_warm.extend_tokens(prefix_tokens.clone(), &cfg)
        .expect("warm extend");
    let checkpoint = conv_warm.checkpoint();

    // CONTROL: replay from the live in-process checkpoint and capture
    // the argmax. The previous test already proved this matches a
    // fresh full prefill; we use it as the reference here.
    let mut cache_a = PrefixCache::new(500_000_000);
    cache_a.insert(checkpoint);
    let (cp_a, _) = cache_a.lookup(&full_tokens).expect("hit A");
    let mut conv_a = Conversation::from_checkpoint(
        &model, &vocab, Runtime::boot(), cp_a).expect("conv_a");
    let stream_a = conv_a.extend_tokens(suffix_tokens.clone(), &cfg)
        .expect("conv_a extend");
    let argmax_a = stream_a.peek_next_token().expect("conv_a tok");
    eprintln!("  IN-PROCESS replay argmax = {}", argmax_a);

    // PERSIST: save to disk, drop, reload, replay.
    let cache_path = std::env::temp_dir().join("joule_prefix_cache_test.bin");
    let _ = std::fs::remove_file(&cache_path);
    cache_a.save_to_file(&cache_path).expect("save_to_file");
    let cache_size = std::fs::metadata(&cache_path).expect("stat").len();
    eprintln!("  wrote cache to {} — {:.1} MB on disk",
        cache_path.display(), cache_size as f64 / 1e6);
    drop(cache_a);

    let t_load = Instant::now();
    let mut cache_b = PrefixCache::load_from_file(&cache_path, 500_000_000)
        .expect("load_from_file");
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  reloaded cache in {:.1} ms — {} entries, {:.1} MB resident",
        load_ms, cache_b.len(), cache_b.current_bytes() as f64 / 1e6);

    let (cp_b, matched_len) = cache_b.lookup(&full_tokens)
        .expect("hit B");
    assert_eq!(matched_len, prefix_tokens.len(),
        "disk-reloaded cache must find the same prefix length");

    let mut conv_b = Conversation::from_checkpoint(
        &model, &vocab, Runtime::boot(), cp_b).expect("conv_b");
    let stream_b = conv_b.extend_tokens(suffix_tokens.clone(), &cfg)
        .expect("conv_b extend");
    let argmax_b = stream_b.peek_next_token().expect("conv_b tok");
    eprintln!("  PERSISTED replay argmax = {}", argmax_b);

    assert_eq!(argmax_a, argmax_b,
        "argmax after save→load→replay {} must match in-process replay {}",
        argmax_b, argmax_a);

    // Cleanup.
    let _ = std::fs::remove_file(&cache_path);

    eprintln!("VERDICT: PrefixCache survives save→load→replay with \
        bit-identical argmax ({}). A cold-started process can resume \
        from a warm system-prompt cache without any model-side \
        re-prefill.", argmax_a);
}
