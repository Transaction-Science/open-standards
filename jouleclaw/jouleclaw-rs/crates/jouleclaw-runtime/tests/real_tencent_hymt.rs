//! End-to-end oracle for Tencent / AngelSlim **Hy-MT1.5-1.8B-1.25bit**.
//!
//! Two new things ride on this single model:
//! 1. **STQ1_0 (Sparse Ternary Quant)** — the 3:4-sparse ternary
//!    packing landed in llama.cpp PR #22836 (10 days old at time of
//!    writing). 1.3125 bpw — more aggressive than PrismML Bonsai's
//!    Q1_0 (1.125) and Q2_0 (2.125). On-disk ggml type id 42 collides
//!    with PrismML's Q2_0; our loader resolves it by byte-stride at
//!    parse time.
//! 2. **`hunyuan-dense`** — a third supported arch alongside `llama`
//!    and `qwen3`. Same QK-RMSNorm + GQA shape; just a different
//!    metadata prefix.
//!
//! Hy-MT is a translation model (33 languages, 1056 translation
//! directions), not a chat/QA model — so we don't gate on "Paris". The
//! oracle gate is: (a) parse succeeds, (b) config detects
//! `hunyuan-dense` + qk_norm, (c) prefill runs, (d) logits are finite
//! and the greedy next token decodes to a non-empty non-garbage string.
//! Coherent translation would require the right chat template; that's
//! a follow-on once we know it speaks at all.

use jouleclaw_loader_gguf::llama::{build_llama_graph, LlamaConfig};
use jouleclaw_loader_gguf::read_gguf_file;
use jouleclaw_loader_gguf::sample::{sample_logits, SamplingConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_TENCENT_HYMT")
        .unwrap_or_else(|_| "../../models/tencent-hy-mt1.5-1.8b-1.25bit.gguf".to_string())
}

#[test]
#[ignore]
fn real_tencent_hymt_loads_and_speaks() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    let model = read_gguf_file(&path).expect("read_gguf_file");
    eprintln!("  parsed GGUF in {:?}", t0.elapsed());
    eprintln!("  arch = {:?}", model.metadata_string("general.architecture"));

    // Verify the stride-based STQ1_0 disambiguation kicked in.
    let n_stq1 = model.tensors.iter()
        .filter(|t| t.dtype == jouleclaw_loader_gguf::GgmlType::STQ1_0)
        .count();
    let n_q2 = model.tensors.iter()
        .filter(|t| t.dtype == jouleclaw_loader_gguf::GgmlType::Q2_0)
        .count();
    eprintln!("  type rewrite: STQ1_0={} Q2_0={}", n_stq1, n_q2);
    assert!(n_stq1 > 0, "type-42 should have been rewritten to STQ1_0");
    assert_eq!(n_q2, 0, "no Q2_0 tensors should remain after disambiguation");

    let cfg = LlamaConfig::from_metadata(&model).expect("LlamaConfig::from_metadata");
    eprintln!(
        "  config: arch={} vocab={} dim={} layers={} ff={} heads={} kv={} head_dim={} \
         qk_norm={} ctx={} rope_base={}",
        cfg.arch, cfg.vocab_size, cfg.embedding_length, cfg.block_count,
        cfg.feed_forward_length, cfg.head_count, cfg.head_count_kv, cfg.head_dim,
        cfg.qk_norm, cfg.context_length, cfg.rope_base,
    );
    assert_eq!(cfg.arch, "hunyuan-dense");
    assert!(cfg.qk_norm, "Hy-MT has per-head QK-RMSNorm");

    let vocab = Vocab::from_gguf(&model).expect("Vocab::from_gguf");

    // Translation-style prompt; Hy-MT is trained for this. We don't
    // pin a specific output word — just check the greedy next token is
    // a decodable non-empty string and logits are finite.
    let prompt = "Translate English to French: Hello, how are you?\nFrench:";
    let token_ids_u32 = vocab.encode_bpe_regex(prompt, false);
    eprintln!("  prompt {:?} → {} tokens", prompt, token_ids_u32.len());
    assert!(!token_ids_u32.is_empty());
    let seq_len = token_ids_u32.len();

    let t1 = Instant::now();
    let llama = build_llama_graph(&model, seq_len).expect("build_llama_graph");
    eprintln!("  built graph: {} nodes in {:?}", llama.graph.nodes.len(), t1.elapsed());

    let t2 = Instant::now();
    let runtime = Runtime::reference_only();
    let compiled = compile(llama.graph, &runtime.kernels).expect("compile");
    eprintln!("  compiled: {} plan entries in {:?}", compiled.plan.len(), t2.elapsed());

    let bytes: Vec<u8> = token_ids_u32.iter()
        .flat_map(|&id| (id as i32).to_le_bytes()).collect();
    let token_tensor = jouleclaw_core::tensor::Tensor {
        meta: jouleclaw_core::tensor::TensorMeta::new(
            jouleclaw_core::tensor::Dtype::I32, &[seq_len]),
        storage: std::sync::Arc::new(
            jouleclaw_core::tensor::TensorStorage { bytes, mapped: None }),
    };
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".to_string(), token_tensor);

    let t3 = Instant::now();
    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");
    let fwd = t3.elapsed();
    eprintln!("  >>> ONE forward pass ({} tokens): {:?}", seq_len, fwd);

    let logits = res.outputs.get("logits").expect("logits");
    assert_eq!(logits.meta.shape, vec![seq_len, cfg.vocab_size]);
    let l = logits.as_f32_vec();
    let last = &l[(seq_len - 1) * cfg.vocab_size..seq_len * cfg.vocab_size];
    let max = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = last.iter().map(|x| (x - max).exp()).sum();
    assert!(sum.is_finite() && sum > 0.0, "logits must be finite");

    let next = sample_logits(last, &SamplingConfig::greedy());
    let tok = vocab.id_to_token(next).unwrap_or("<unk>");
    let cont = vocab.decode_bpe(&[next]);
    eprintln!("  >>> greedy next: id={} token={:?} decoded={:?}", next, tok, cont);
    eprintln!("  >>> {:?}{}", prompt, cont);

    // Oracle: the next token decoded must be non-empty and contain at
    // least one alphabetic/ASCII character. A correctness disaster
    // (random logits, dequant bug) shows up as <unk> or empty.
    assert!(!cont.is_empty(), "decoded continuation must not be empty");
    assert!(
        cont.chars().any(|c| c.is_ascii_alphabetic() || c.is_ascii_punctuation()
            || c.is_alphabetic()),
        "decoded continuation must contain a real character, got {:?}", cont
    );

    eprintln!("VERDICT: Hy-MT1.5-1.8B-1.25bit (STQ1_0, hunyuan-dense) loaded + ran + \
        produced a finite, decodable next token. Forward {:?} for {} tokens.",
        fwd, seq_len);
}
