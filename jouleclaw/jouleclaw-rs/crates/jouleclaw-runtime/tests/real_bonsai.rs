//! Empirical real-model coherence oracle — PrismML Ternary-Bonsai-1.7B.
//!
//! Loads the genuine `Ternary-Bonsai-1.7B-Q2_0.gguf` (qwen3 arch,
//! 1.58-bit ternary weights, Q2_0 g128 packing) and runs ONE real
//! forward pass through the jouleclaw-loader-gguf + jouleclaw-runtime pipeline
//! exercising the *new* code paths:
//!
//!   * `dequant::dequantize_q2_0` (128-elem / 34-byte ternary blocks)
//!   * the qwen3 arch adapter: per-head QK-RMSNorm, decoupled head_dim,
//!     rope_base 1e6, tied input/output embeddings
//!   * the gpt2/qwen2 byte-level BPE tokenizer path
//!
//! This is the GATE. No part of the Bonsai/Q2_0 work is claimed correct
//! until this prints a coherent continuation. The structural unit tests
//! (`dequant::q2_0_tests`) prove the block math; this proves the whole
//! stack produces language.
//!
//! `#[ignore]` — needs the model on disk (not in CI). Run:
//!
//!   cargo test -p jouleclaw-runtime --test real_bonsai -- --ignored --nocapture
//!
//! Model path: `JOULE_BONSAI` env var, else
//! `../../models/ternary-bonsai-1.7b-q2_0.gguf` relative to the crate.

use jouleclaw_loader_gguf::llama::{build_llama_graph, LlamaConfig};
use jouleclaw_loader_gguf::read_gguf_file;
use jouleclaw_loader_gguf::sample::{sample_logits, SamplingConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

#[test]
#[ignore]
fn real_bonsai_coherence_oracle() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    let model = read_gguf_file(&path).expect("read_gguf_file");
    eprintln!("  parsed GGUF in {:?}", t0.elapsed());
    eprintln!("  arch = {:?}", model.metadata_string("general.architecture"));

    let cfg = LlamaConfig::from_metadata(&model).expect("LlamaConfig::from_metadata");
    eprintln!(
        "  config: arch={} vocab={} dim={} layers={} ff={} heads={} kv={} head_dim={} \
         qk_norm={} ctx={} rope_base={}",
        cfg.arch, cfg.vocab_size, cfg.embedding_length, cfg.block_count,
        cfg.feed_forward_length, cfg.head_count, cfg.head_count_kv, cfg.head_dim,
        cfg.qk_norm, cfg.context_length, cfg.rope_base,
    );
    assert_eq!(cfg.arch, "qwen3", "expected qwen3 architecture");
    assert!(cfg.qk_norm, "Bonsai must carry QK-RMSNorm weights");
    assert_eq!(cfg.head_dim, 128, "qwen3 key_length");

    let vocab = Vocab::from_gguf(&model).expect("Vocab::from_gguf");

    // Qwen byte-level BPE (gpt2 model / qwen2 pre); no BOS.
    let prompt = "The capital of France is";
    let token_ids_u32 = vocab.encode_bpe_regex(prompt, false);
    eprintln!(
        "  prompt {:?} → {} tokens: {:?}",
        prompt, token_ids_u32.len(), token_ids_u32
    );
    let seq_len = token_ids_u32.len();
    assert!(seq_len > 0, "tokenizer produced no tokens");

    let t1 = Instant::now();
    let llama = build_llama_graph(&model, seq_len).expect("build_llama_graph");
    eprintln!(
        "  built graph: {} nodes in {:?}",
        llama.graph.nodes.len(), t1.elapsed()
    );

    let t2 = Instant::now();
    let runtime = Runtime::reference_only();
    let compiled = compile(llama.graph, &runtime.kernels).expect("compile");
    eprintln!(
        "  compiled: {} plan entries in {:?}",
        compiled.plan.len(), t2.elapsed()
    );

    let bytes: Vec<u8> = token_ids_u32
        .iter()
        .flat_map(|&id| (id as i32).to_le_bytes())
        .collect();
    let token_tensor = jouleclaw_core::tensor::Tensor {
        meta: jouleclaw_core::tensor::TensorMeta::new(
            jouleclaw_core::tensor::Dtype::I32, &[seq_len]),
        storage: std::sync::Arc::new(jouleclaw_core::tensor::TensorStorage { bytes, mapped: None }),
    };
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".to_string(), token_tensor);

    let t3 = Instant::now();
    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");
    let fwd = t3.elapsed();
    eprintln!("  >>> ONE forward pass ({} tokens): {:?}", seq_len, fwd);

    let logits = res.outputs.get("logits").expect("logits output");
    assert_eq!(
        logits.meta.shape, vec![seq_len, cfg.vocab_size],
        "logits shape"
    );
    let l = logits.as_f32_vec();
    let last = &l[(seq_len - 1) * cfg.vocab_size..seq_len * cfg.vocab_size];

    // Logits must be finite.
    let max = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = last.iter().map(|x| (x - max).exp()).sum();
    assert!(sum.is_finite() && sum > 0.0, "logits must be finite");

    // Greedy decode several tokens for a readable continuation.
    let mut continuation_ids = Vec::new();
    {
        // Top-1 of the prefill's last row.
        let next = sample_logits(last, &SamplingConfig::greedy());
        continuation_ids.push(next);
    }
    let first = continuation_ids[0];
    let first_tok = vocab.id_to_token(first).unwrap_or("<unk>");
    let cont = vocab.decode_bpe(&continuation_ids);
    eprintln!("  >>> greedy next: id={} token={:?} decoded={:?}", first, first_tok, cont);
    eprintln!("  >>> {:?}{}", prompt, cont);

    // THE ORACLE: a competent model completes "The capital of France is"
    // with Paris. This is the same falsifiable bar TinyLlama passed
    // ("Paris"). If this fails, the Q2_0 dequant or the qwen3 adapter is
    // wrong — do not claim the Bonsai path works.
    let said_paris = cont.to_lowercase().contains("paris")
        || first_tok.to_lowercase().contains("paris");
    assert!(
        said_paris,
        "COHERENCE ORACLE FAILED: expected 'Paris' in continuation, got {:?} \
         (token {:?}). Q2_0 dequant or qwen3 adapter is incorrect.",
        cont, first_tok
    );

    eprintln!(
        "VERDICT: PrismML Ternary-Bonsai-1.7B (Q2_0 g128, qwen3) loaded + ran \
         + produced 'Paris'. Forward {:?} for {} tokens.",
        fwd, seq_len
    );
}
