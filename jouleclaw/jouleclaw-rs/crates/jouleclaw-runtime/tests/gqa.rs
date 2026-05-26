//! Grouped-Query Attention (GQA) end-to-end correctness tests.
//!
//! These tests construct synthetic models where `head_count_kv != head_count`
//! and verify two things:
//!
//! 1. The one-shot prefill graph runs and produces logits of the correct shape.
//! 2. Incremental decode with the KV cache matches one-shot prefill within
//!    tolerance — the same correctness property as the standard MHA case,
//!    but with the cache stored at the smaller `n_heads_kv` shape.
//!
//! The defining benefit of GQA — smaller KV cache memory — is verified by
//! checking the actual cache tensor shape: `[n_heads_kv, seq, d_head]`,
//! not `[n_heads, seq, d_head]`.

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_loader_gguf::decode::build_decode_step_graph;
use jouleclaw_loader_gguf::kv_cache::KvCache;
use jouleclaw_loader_gguf::llama::build_llama_graph;
use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::io::Cursor;

fn make_token_ids(seq_len: usize, vocab_size: usize, seed: u64) -> Tensor {
    let mut s = seed;
    let bytes: Vec<u8> = (0..seq_len).flat_map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let id = ((s >> 32) as u32 % vocab_size as u32) as i32;
        id.to_le_bytes().to_vec()
    }).collect();
    Tensor {
        meta: TensorMeta::new(Dtype::I32, &[seq_len]),
        storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
    }
}

fn token_ids_slice(full: &Tensor, start: usize, len: usize) -> Tensor {
    let bytes = full.storage.bytes[start * 4..(start + len) * 4].to_vec();
    Tensor {
        meta: TensorMeta::new(Dtype::I32, &[len]),
        storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
    }
}

fn run_decode_step(
    model: &jouleclaw_loader_gguf::GgufModel,
    cache: &mut KvCache,
    new_token_ids: Tensor,
    new_seq: usize,
) -> Vec<f32> {
    let step = build_decode_step_graph(model, cache, new_seq).expect("build decode step");
    let runtime = Runtime::boot();
    let compiled = compile(step.graph, &runtime.kernels).expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), new_token_ids);
    if cache.current_seq > 0 {
        for layer in 0..step.config.block_count {
            let k = cache.k_for(layer).expect("K present").clone();
            let v = cache.v_for(layer).expect("V present").clone();
            inputs.insert(step.k_input_names[layer].clone(), k);
            inputs.insert(step.v_input_names[layer].clone(), v);
        }
    }
    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");
    for layer in 0..step.config.block_count {
        let k = res.outputs.get(&step.k_output_names[layer]).expect("kv_out_k").clone();
        let v = res.outputs.get(&step.v_output_names[layer]).expect("kv_out_v").clone();
        cache.put(layer, k, v);
    }
    res.outputs.get(&step.logits_output_name).expect("logits").as_f32_vec()
}

fn cmp_logits(actual: &[f32], expected: &[f32], rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(),
        "{}: length mismatch {} vs {}", label, actual.len(), expected.len());
    let mut max_rel = 0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let abs = (a - e).abs();
        let denom = e.abs().max(1e-6);
        let rel = abs / denom;
        if rel > max_rel { max_rel = rel; }
    }
    assert!(max_rel < rtol,
        "{}: max relative diff {:.3e} exceeds tolerance {}", label, max_rel, rtol);
}

/// GQA prefill: model with head_count=4, head_count_kv=2 (group_size=2) loads
/// and runs through the one-shot prefill graph.
#[test]
fn gqa_prefill_runs_end_to_end() {
    let cfg = SyntheticConfig {
        vocab_size: 32,
        embedding_length: 16,
        block_count: 2,
        feed_forward_length: 32,
        head_count: 4,
        head_count_kv: 2,        // GQA: 2 KV heads, 4 Q heads → group_size = 2
        rms_eps: 1e-6,
        seed: 11,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let seq_len = 4;
    let llama = build_llama_graph(&model, seq_len).expect("build_llama_graph for GQA");
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();

    let toks = make_token_ids(seq_len, cfg.vocab_size, 7);
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), toks);

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let logits = res.outputs.get("logits").unwrap();
    assert_eq!(logits.meta.shape, vec![seq_len, cfg.vocab_size],
        "GQA prefill logits should have shape [seq_len, vocab_size]");
}

/// GQA decode: KV cache stores K/V at the smaller `n_heads_kv` shape.
/// After a decode step with head_count_kv=2, the cache shape should be
/// `[2, current_seq, d_head]`, not `[4, current_seq, d_head]`.
#[test]
fn gqa_cache_uses_smaller_kv_head_shape() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 16,
        block_count: 1, feed_forward_length: 16,
        head_count: 4, head_count_kv: 2,
        rms_eps: 1e-6, seed: 13,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let mut cache = KvCache::empty(cfg.block_count);

    // Single decode step with 3 tokens.
    let toks = make_token_ids(3, cfg.vocab_size, 1);
    let _ = run_decode_step(&model, &mut cache, toks, 3);

    // Cache K should be [head_count_kv=2, seq=3, d_head=4], NOT
    // [head_count=4, seq=3, d_head=4].
    let k0 = cache.k_for(0).expect("K populated");
    let d_head = cfg.embedding_length / cfg.head_count;  // 16/4 = 4
    assert_eq!(k0.meta.shape, vec![cfg.head_count_kv, 3, d_head],
        "GQA cache should store K at n_heads_kv shape, got {:?}", k0.meta.shape);
}

/// GQA incremental decode matches one-shot prefill within tolerance.
/// Same correctness property as the MHA case but with group_size > 1.
#[test]
fn gqa_incremental_decode_matches_one_shot_prefill() {
    let cfg = SyntheticConfig {
        vocab_size: 32, embedding_length: 16,
        block_count: 2, feed_forward_length: 32,
        head_count: 4, head_count_kv: 2,  // GQA
        rms_eps: 1e-6, seed: 21,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let total_seq = 6;
    let full_tokens = make_token_ids(total_seq, cfg.vocab_size, 1);

    // One-shot prefill.
    let llama = build_llama_graph(&model, total_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), full_tokens.clone());
    let one_shot_res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_logits = one_shot_res.outputs.get("logits").unwrap().as_f32_vec();

    // Incremental decode: [3, 1, 1, 1] step sizes summing to 6.
    let step_sizes = [3, 1, 1, 1];
    let mut cache = KvCache::empty(cfg.block_count);
    let mut all_logits: Vec<f32> = Vec::new();
    let mut consumed = 0;
    for size in step_sizes.iter() {
        let new_tokens = token_ids_slice(&full_tokens, consumed, *size);
        let logits = run_decode_step(&model, &mut cache, new_tokens, *size);
        all_logits.extend(logits);
        consumed += size;
    }
    assert_eq!(consumed, total_seq);
    assert_eq!(cache.current_seq, total_seq);

    cmp_logits(&all_logits, &one_shot_logits, 1e-3,
        "GQA incremental decode vs one-shot prefill");
}

/// MQA (Multi-Query Attention) is the limit case head_count_kv = 1.
/// All Q heads share a single KV head.
#[test]
fn mqa_extreme_case_works() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 16,
        block_count: 1, feed_forward_length: 16,
        head_count: 4, head_count_kv: 1,  // MQA — group_size = 4
        rms_eps: 1e-6, seed: 17,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let total_seq = 4;
    let full_tokens = make_token_ids(total_seq, cfg.vocab_size, 2);

    // One-shot prefill should run.
    let llama = build_llama_graph(&model, total_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), full_tokens.clone());
    let one_shot = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_logits = one_shot.outputs.get("logits").unwrap().as_f32_vec();
    assert_eq!(one_shot.outputs.get("logits").unwrap().meta.shape,
        vec![total_seq, cfg.vocab_size]);

    // First decode step on the same tokens should match one-shot.
    let mut cache = KvCache::empty(cfg.block_count);
    let step_logits = run_decode_step(&model, &mut cache, full_tokens, total_seq);
    cmp_logits(&step_logits, &one_shot_logits, 1e-3,
        "MQA first decode step vs one-shot prefill");

    // Cache should be [1, seq, d_head].
    let k0 = cache.k_for(0).unwrap();
    let d_head = cfg.embedding_length / cfg.head_count;
    assert_eq!(k0.meta.shape, vec![1, total_seq, d_head],
        "MQA cache should store K at single-head shape");
}

/// MHA (head_count_kv == head_count) — the pre-GQA path — still works correctly
/// after the GQA refactor. group_size = 1 short-circuits the repeat.
#[test]
fn mha_unchanged_after_gqa_refactor() {
    let cfg = SyntheticConfig {
        vocab_size: 32, embedding_length: 16,
        block_count: 2, feed_forward_length: 32,
        head_count: 4, head_count_kv: 4,  // standard MHA
        rms_eps: 1e-6, seed: 25,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let total_seq = 5;
    let full_tokens = make_token_ids(total_seq, cfg.vocab_size, 3);

    let llama = build_llama_graph(&model, total_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), full_tokens.clone());
    let one_shot = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_logits = one_shot.outputs.get("logits").unwrap().as_f32_vec();

    // Decode through full sequence in two steps.
    let mut cache = KvCache::empty(cfg.block_count);
    let mut all_logits = Vec::new();
    for sizes in &[3, 2] {
        let new_t = token_ids_slice(&full_tokens, all_logits.len() / cfg.vocab_size, *sizes);
        let l = run_decode_step(&model, &mut cache, new_t, *sizes);
        all_logits.extend(l);
    }
    cmp_logits(&all_logits, &one_shot_logits, 1e-3,
        "MHA path (head_count_kv == head_count) preserved");
}
