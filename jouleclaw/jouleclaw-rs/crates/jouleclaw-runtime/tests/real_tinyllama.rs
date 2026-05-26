//! Empirical real-model harness.
//!
//! Loads a genuine TinyLlama-1.1B-Chat Q4_K_M GGUF, runs ONE real
//! forward pass through the existing jouleclaw-loader-gguf + jouleclaw-runtime
//! pipeline, samples the next token, and reports wall-clock timing.
//!
//! `#[ignore]` because it needs the model file on disk (not in CI).
//! Run explicitly:
//!
//!   cargo test -p jouleclaw-runtime --test real_tinyllama -- --ignored --nocapture
//!
//! Model path: `JOULE_TINYLLAMA` env var, else
//! `../../models/tinyllama-1.1b-chat-q4km.gguf` relative to the crate.

use jouleclaw_loader_gguf::llama::{build_llama_graph, LlamaConfig};
use jouleclaw_loader_gguf::sample::{sample_logits, SamplingConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::read_gguf_file;
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_TINYLLAMA")
        .unwrap_or_else(|_| "../../models/tinyllama-1.1b-chat-q4km.gguf".to_string())
}

#[test]
#[ignore]
fn real_tinyllama_one_forward_pass() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    let model = read_gguf_file(&path).expect("read_gguf_file");
    eprintln!("  parsed GGUF in {:?}", t0.elapsed());
    eprintln!("  arch = {:?}", model.metadata_string("general.architecture"));

    let cfg = LlamaConfig::from_metadata(&model).expect("LlamaConfig::from_metadata");
    eprintln!(
        "  config: vocab={} dim={} layers={} ff={} heads={} kv_heads={} ctx={} rope_base={}",
        cfg.vocab_size,
        cfg.embedding_length,
        cfg.block_count,
        cfg.feed_forward_length,
        cfg.head_count,
        cfg.head_count_kv,
        cfg.context_length,
        cfg.rope_base,
    );

    let vocab = Vocab::from_gguf(&model).expect("Vocab::from_gguf");
    eprintln!("  vocab tokens = {}", cfg.vocab_size);

    // A short prompt. TinyLlama-Chat uses a Zephyr-ish template; we keep
    // it simple — raw text, BOS prepended by the tokenizer.
    let prompt = "The capital of France is";
    // TinyLlama (Llama-2 lineage) uses SentencePiece, not GPT-2 BPE.
    // tokenizer.ggml.model == "llama" → SPM encoder.
    let token_ids_u32 = vocab.encode_spm(prompt, true);
    eprintln!("  prompt {:?} → {} tokens: {:?}", prompt, token_ids_u32.len(), token_ids_u32);
    let seq_len = token_ids_u32.len();
    assert!(seq_len > 0, "tokenizer produced no tokens");

    let t1 = Instant::now();
    let llama = build_llama_graph(&model, seq_len).expect("build_llama_graph");
    eprintln!("  built graph: {} nodes in {:?}", llama.graph.nodes.len(), t1.elapsed());

    let t2 = Instant::now();
    // Reference backend only: the AppleAmx matmul kernel has a known
    // preexisting batched inner-dim bug (also fails joule-l2 +
    // boot_validation). The reference backend is the determinism oracle
    // and handles real attention correctly — slowly.
    let runtime = Runtime::reference_only();
    let compiled = compile(llama.graph, &runtime.kernels).expect("compile");
    eprintln!("  compiled: {} plan entries in {:?}", compiled.plan.len(), t2.elapsed());

    // Bind token_ids as an I32 tensor [seq_len].
    let bytes: Vec<u8> = token_ids_u32
        .iter()
        .flat_map(|&id| (id as i32).to_le_bytes())
        .collect();
    let token_tensor = jouleclaw_core::tensor::Tensor {
        meta: jouleclaw_core::tensor::TensorMeta::new(jouleclaw_core::tensor::Dtype::I32, &[seq_len]),
        storage: std::sync::Arc::new(jouleclaw_core::tensor::TensorStorage { bytes, mapped: None }),
    };
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".to_string(), token_tensor);

    let t3 = Instant::now();
    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");
    let fwd = t3.elapsed();
    eprintln!("  >>> ONE forward pass ({} tokens): {:?}", seq_len, fwd);
    eprintln!("  >>> per-token: {:?}", fwd / seq_len as u32);

    let logits = res.outputs.get("logits").expect("logits output");
    assert_eq!(
        logits.meta.shape,
        vec![seq_len, cfg.vocab_size],
        "logits shape"
    );
    let l = logits.as_f32_vec();
    // Next-token logits = last row.
    let last = &l[(seq_len - 1) * cfg.vocab_size..seq_len * cfg.vocab_size];
    let next = sample_logits(last, &SamplingConfig::greedy());
    let tok = vocab.id_to_token(next).unwrap_or("<unk>");
    eprintln!("  >>> greedy next token: id={} {:?}", next, tok);
    eprintln!("  >>> continuation: {:?}{}", prompt, vocab.decode_bpe(&[next]));

    // Sanity: softmax sums to 1, logits finite.
    let max = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = last.iter().map(|x| (x - max).exp()).sum();
    assert!(sum.is_finite() && sum > 0.0, "logits must be finite");

    eprintln!(
        "VERDICT: real 1.1B model loaded + ran via the existing pipeline. \
         Forward {:?} for {} tokens ({:?}/tok).",
        fwd, seq_len, fwd / seq_len as u32
    );
}
