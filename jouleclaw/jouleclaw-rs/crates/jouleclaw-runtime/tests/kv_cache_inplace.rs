//! In-place KV cache correctness tests.
//!
//! The defining property: in-place decode with the preallocated buffer
//! produces logits equivalent to one-shot prefill, just like the
//! concat-based cache. Plus: the buffer size stays constant across
//! decode steps (the actual memory benefit).

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_loader_gguf::kv_cache_inplace::{
    build_decode_step_graph_inplace, InPlaceKvCache,
};
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

fn run_inplace_step(
    model: &jouleclaw_loader_gguf::GgufModel,
    cache: &mut InPlaceKvCache,
    new_token_ids: Tensor,
    new_seq: usize,
) -> Vec<f32> {
    let step = build_decode_step_graph_inplace(model, cache, new_seq)
        .expect("build inplace decode step");
    let runtime = Runtime::boot();
    let compiled = compile(step.graph, &runtime.kernels).expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), new_token_ids);
    for layer in 0..step.config.block_count {
        inputs.insert(step.k_input_names[layer].clone(), cache.k_bufs[layer].clone());
        inputs.insert(step.v_input_names[layer].clone(), cache.v_bufs[layer].clone());
    }
    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");

    // Stash updated buffers back into cache.
    for layer in 0..step.config.block_count {
        let k = res.outputs.get(&step.k_output_names[layer]).expect("kv_out_k").clone();
        let v = res.outputs.get(&step.v_output_names[layer]).expect("kv_out_v").clone();
        cache.put(layer, k, v);
    }
    cache.advance(new_seq);

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

/// Buffer construction: shape is [n_heads_kv, max_seq, d_head] from the start
/// and contains zeros.
#[test]
fn inplace_cache_initial_state() {
    let n_layers = 2;
    let n_heads_kv = 2;
    let max_seq = 16;
    let d_head = 4;
    let cache = InPlaceKvCache::new(n_layers, n_heads_kv, max_seq, d_head);
    assert_eq!(cache.current_seq, 0);
    assert_eq!(cache.k_bufs.len(), n_layers);
    for layer in 0..n_layers {
        assert_eq!(cache.k_bufs[layer].meta.shape, vec![n_heads_kv, max_seq, d_head]);
        let data = cache.k_bufs[layer].as_f32_vec();
        for v in &data { assert_eq!(*v, 0.0); }
    }
}

/// In-place decode: first step on a sequence produces same logits as one-shot.
#[test]
fn inplace_first_step_matches_one_shot() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8,
        block_count: 1, feed_forward_length: 16,
        head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 5,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let seq = 3;
    let toks = make_token_ids(seq, cfg.vocab_size, 11);

    // One-shot prefill.
    let llama = build_llama_graph(&model, seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), toks.clone());
    let one_shot = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_l = one_shot.outputs.get("logits").unwrap().as_f32_vec();

    // In-place first step.
    let max_seq = 16;
    let mut cache = InPlaceKvCache::for_model(&model, max_seq).unwrap();
    let step_l = run_inplace_step(&model, &mut cache, toks, seq);
    assert_eq!(cache.current_seq, seq);

    cmp_logits(&step_l, &one_shot_l, 1e-3,
        "in-place first decode step vs one-shot prefill");
}

/// THE in-place KV cache correctness proof.
///
/// Run 6 tokens through one-shot prefill, then through in-place incremental
/// decode in step sizes [3, 1, 1, 1]. The logits at the last token must match
/// within tolerance. This is the same property the concat-based cache test
/// verifies, now for the in-place implementation.
#[test]
fn inplace_incremental_decode_matches_one_shot_prefill() {
    let cfg = SyntheticConfig {
        vocab_size: 32, embedding_length: 16,
        block_count: 2, feed_forward_length: 32,
        head_count: 4, head_count_kv: 4,
        rms_eps: 1e-6, seed: 7,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let total_seq = 6;
    let full_tokens = make_token_ids(total_seq, cfg.vocab_size, 1);

    // One-shot.
    let llama = build_llama_graph(&model, total_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), full_tokens.clone());
    let one_shot = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_l = one_shot.outputs.get("logits").unwrap().as_f32_vec();

    // In-place incremental.
    let max_seq = 16;
    let mut cache = InPlaceKvCache::for_model(&model, max_seq).unwrap();
    let step_sizes = [3, 1, 1, 1];
    let mut all_logits = Vec::new();
    let mut consumed = 0;
    for size in &step_sizes {
        let new = token_ids_slice(&full_tokens, consumed, *size);
        let l = run_inplace_step(&model, &mut cache, new, *size);
        all_logits.extend(l);
        consumed += size;
    }
    assert_eq!(cache.current_seq, total_seq);

    // Cache buffer shape unchanged.
    for layer in 0..cfg.block_count {
        assert_eq!(cache.k_bufs[layer].meta.shape,
            vec![cfg.head_count_kv, max_seq, cfg.embedding_length / cfg.head_count]);
    }

    cmp_logits(&all_logits, &one_shot_l, 1e-3,
        "in-place incremental decode vs one-shot prefill");
}

/// In-place GQA: head_count=4, head_count_kv=2 (group_size=2).
#[test]
fn inplace_gqa_matches_one_shot_prefill() {
    let cfg = SyntheticConfig {
        vocab_size: 32, embedding_length: 16,
        block_count: 2, feed_forward_length: 32,
        head_count: 4, head_count_kv: 2,  // GQA
        rms_eps: 1e-6, seed: 13,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let total_seq = 5;
    let full_tokens = make_token_ids(total_seq, cfg.vocab_size, 2);

    let llama = build_llama_graph(&model, total_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), full_tokens.clone());
    let one_shot = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_l = one_shot.outputs.get("logits").unwrap().as_f32_vec();

    let max_seq = 12;
    let mut cache = InPlaceKvCache::for_model(&model, max_seq).unwrap();
    let mut all_logits = Vec::new();
    for sizes in &[2, 1, 1, 1] {
        let new = token_ids_slice(&full_tokens, all_logits.len() / cfg.vocab_size, *sizes);
        let l = run_inplace_step(&model, &mut cache, new, *sizes);
        all_logits.extend(l);
    }

    // Buffer shape is at the n_heads_kv shape, not n_heads — the GQA benefit.
    assert_eq!(cache.k_bufs[0].meta.shape,
        vec![cfg.head_count_kv, max_seq, cfg.embedding_length / cfg.head_count]);

    cmp_logits(&all_logits, &one_shot_l, 1e-3,
        "in-place GQA incremental vs one-shot");
}

/// Buffer size stays constant across all decode steps — the actual memory
/// benefit of in-place. Compare to the concat-based cache where shape grows.
#[test]
fn inplace_buffer_size_stays_constant() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8,
        block_count: 1, feed_forward_length: 16,
        head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 3,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let max_seq = 10;
    let mut cache = InPlaceKvCache::for_model(&model, max_seq).unwrap();
    let expected_shape = vec![cfg.head_count_kv, max_seq,
        cfg.embedding_length / cfg.head_count];

    for step in 1..=4 {
        let toks = make_token_ids(1, cfg.vocab_size, step);
        let _ = run_inplace_step(&model, &mut cache, toks, 1);
        assert_eq!(cache.k_bufs[0].meta.shape, expected_shape,
            "after step {}: buffer shape should stay {:?}, got {:?}",
            step, expected_shape, cache.k_bufs[0].meta.shape);
    }
    assert_eq!(cache.current_seq, 4);
}

/// Reset clears the buffers.
#[test]
fn inplace_reset_clears_buffers() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8,
        block_count: 1, feed_forward_length: 16,
        head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 1,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let mut cache = InPlaceKvCache::for_model(&model, 8).unwrap();
    let toks = make_token_ids(2, cfg.vocab_size, 1);
    let _ = run_inplace_step(&model, &mut cache, toks, 2);
    assert_eq!(cache.current_seq, 2);

    cache.reset();
    assert_eq!(cache.current_seq, 0);
    let data = cache.k_bufs[0].as_f32_vec();
    for v in &data { assert_eq!(*v, 0.0, "reset should zero the K buffer"); }
}
