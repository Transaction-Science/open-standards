//! End-to-end streaming oracle: multi-token generation of
//! `Ternary-Bonsai-1.7B-Q2_0` through the actual KV-cache decode path
//! (`Conversation::extend` → `run_inplace_step` →
//! `build_decode_step_graph_inplace` → ternary kernels).
//!
//! `real_bonsai.rs` validates *prefill* only (one forward, one logit
//! row, one greedy token). This validates *generation* — repeated
//! decode steps consuming the cache and the ternary routing inside the
//! in-place block builder. If the decode path's `wmm` / `lookup_w`
//! routing is wrong, the continuation will be incoherent.
//!
//! `#[ignore]` — needs the model on disk. Run:
//!
//!   cargo test --release -p jouleclaw-runtime --test real_bonsai_streaming \
//!     -- --ignored --nocapture

use jouleclaw_loader_gguf::read_gguf_file;
use jouleclaw_loader_gguf::sample::SamplingConfig;
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::generate::{GenerateConfig, KvCacheKind, TokenizerKind};
use jouleclaw_runtime::streaming::Conversation;
use jouleclaw_runtime::Runtime;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

#[test]
#[ignore]
fn real_bonsai_streams_paris() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    let model = read_gguf_file(&path).expect("read_gguf_file");
    eprintln!("  parsed GGUF in {:?}", t0.elapsed());
    let vocab = Vocab::from_gguf(&model).expect("Vocab::from_gguf");

    // Reference-only runtime: the AppleAmx f32 drift is a documented
    // pre-existing bug; the ternary kernels live only in the reference
    // backend anyway, so this is the path that's actually under test.
    let mut conv = Conversation::with_runtime(
        &model, &vocab, 64, Runtime::reference_only(),
    ).expect("Conversation::with_runtime");

    let cfg = GenerateConfig {
        max_new_tokens: 8,
        add_bos: false,                   // qwen2 BPE; no BOS
        tokenizer_kind: TokenizerKind::Bpe,
        cache_kind: KvCacheKind::InPlace,
        sampling: SamplingConfig::greedy(),
        max_seq: Some(64),
        stop_strings: vec![],
    };

    let prompt = "The capital of France is";
    let t_gen = Instant::now();
    let mut tokens: Vec<u32> = Vec::new();
    {
        let stream = conv.extend(prompt, &cfg).expect("extend");
        for st in stream {
            let st = st.expect("decode step");
            tokens.push(st.id);
        }
    }
    let gen_time = t_gen.elapsed();
    let cont = vocab.decode_bpe(&tokens);
    eprintln!("  >>> generated {} tokens in {:?} ({:?}/tok)",
        tokens.len(), gen_time, gen_time / tokens.len().max(1) as u32);
    eprintln!("  >>> tokens = {:?}", tokens);
    eprintln!("  >>> {:?}{}", prompt, cont);

    // Streaming oracle: greedy generation must include "Paris" early.
    // Same bar as the prefill test, applied to the cache-driven path.
    assert!(
        cont.to_lowercase().contains("paris"),
        "STREAMING ORACLE FAILED — got {:?}. Decode/streaming path \
         (build_decode_step_graph_inplace + ternary wmm/lookup_w) is \
         incorrect.", cont);

    let tok_per_s = tokens.len() as f64 / gen_time.as_secs_f64();
    eprintln!("VERDICT: Bonsai streaming via in-place KV-cache decode \
        produces 'Paris'. Throughput {:.2} tok/s.", tok_per_s);
}
