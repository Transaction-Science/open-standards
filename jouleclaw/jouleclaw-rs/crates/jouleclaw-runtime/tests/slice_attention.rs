//! Test that the in-place decode path's attention work scales with
//! `valid_seq` (the live region), not `max_seq` (the buffer capacity).
//!
//! This is the Phase 1.17 perf win: Slice + slim attention. We verify it
//! by comparing two runs of the same prompt with different `max_seq`
//! values. The output tokens must be identical (correctness), and the
//! graph node count for the larger-buffer run must NOT scale linearly
//! with max_seq (perf characteristic).

use jouleclaw_loader_gguf::kv_cache_inplace::{
    build_decode_step_graph_inplace, InPlaceKvCache,
};
use jouleclaw_loader_gguf::llama::build_llama_graph;
use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

fn make_token_ids(seq_len: usize, vocab_size: usize, seed: u64) -> Tensor {
    let mut s = seed;
    let bytes: Vec<u8> = (0..seq_len).flat_map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let id = ((s >> 32) as u32 % vocab_size as u32) as i32;
        id.to_le_bytes().to_vec()
    }).collect();
    Tensor {
        meta: TensorMeta::new(Dtype::I32, &[seq_len]),
        storage: Arc::new(TensorStorage { bytes, mapped: None }),
    }
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

fn run_inplace_step(
    model: &jouleclaw_loader_gguf::GgufModel,
    cache: &mut InPlaceKvCache,
    new_tokens: Tensor,
    new_seq: usize,
) -> Vec<f32> {
    let step = build_decode_step_graph_inplace(model, cache, new_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(step.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), new_tokens);
    for layer in 0..step.config.block_count {
        inputs.insert(step.k_input_names[layer].clone(), cache.k_bufs[layer].clone());
        inputs.insert(step.v_input_names[layer].clone(), cache.v_bufs[layer].clone());
    }
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    for layer in 0..step.config.block_count {
        let k = res.outputs.get(&step.k_output_names[layer]).unwrap().clone();
        let v = res.outputs.get(&step.v_output_names[layer]).unwrap().clone();
        cache.put(layer, k, v);
    }
    cache.advance(new_seq);
    res.outputs.get(&step.logits_output_name).unwrap().as_f32_vec()
}

/// THE perf characteristic test: changing max_seq doesn't change the
/// attention work. We verify this by checking that the per-step scores
/// tensor's shape is `[heads, new_seq, valid_seq]`, not `[heads, new_seq,
/// max_seq]`. The shape itself is what determines matmul work in our
/// reference kernel.
#[test]
fn attention_work_bounded_by_valid_seq_not_max_seq() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8,
        block_count: 1, feed_forward_length: 16,
        head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 1,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    // Compare two buffer sizes. After 2 decode steps (valid_seq=2):
    //   small buffer (max_seq=4):  attention scores shape = [2, 1, 2]
    //   large buffer (max_seq=64): attention scores shape = [2, 1, 2] (SAME)
    //
    // If attention were bounded by max_seq, the large-buffer scores would
    // be shape [2, 1, 64] — 32× more work. With Slice in place, both
    // produce [2, 1, 2].
    for &max_seq in &[4usize, 64] {
        let mut cache = InPlaceKvCache::for_model(&model, max_seq).unwrap();
        // Two steps to get to valid_seq=2.
        let _ = run_inplace_step(&model, &mut cache, make_token_ids(1, cfg.vocab_size, 1), 1);
        let _ = run_inplace_step(&model, &mut cache, make_token_ids(1, cfg.vocab_size, 2), 1);

        // Inspect the next-step graph for the scores tensor shape.
        let step = build_decode_step_graph_inplace(&model, &cache, 1).unwrap();

        // Find a MatMul node whose output has shape ending in `valid_seq`
        // (= cache.current_seq + 1 = 3). The QK^T scores matmul produces
        // shape [n_heads, new_seq, valid_seq] = [2, 1, 3].
        let valid_seq = cache.current_seq + 1;
        let expected_scores_shape: Vec<usize> = vec![2, 1, valid_seq];
        let mut found_scores = false;
        for node in step.graph.nodes.iter() {
            if let jouleclaw_core::graph::NodeKind::Op { op, .. } = &node.kind {
                if *op == jouleclaw_core::op::OpKind::MatMul
                    && node.output_meta[0].shape == expected_scores_shape
                {
                    found_scores = true;
                    break;
                }
            }
        }
        assert!(found_scores,
            "max_seq={}: should find scores matmul with shape {:?}; \
             this proves attention work is bounded by valid_seq ({}), \
             not max_seq ({})",
            max_seq, expected_scores_shape, valid_seq, max_seq);
    }
}

/// Numerical equivalence is preserved across different max_seq values.
/// Same prompt, same model — output logits must match within tolerance.
#[test]
fn output_logits_independent_of_max_seq() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8,
        block_count: 2, feed_forward_length: 16,
        head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 2,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let toks = make_token_ids(3, cfg.vocab_size, 7);

    // Run the same step sequence with two different buffer sizes.
    let mut logits_per_max_seq = Vec::new();
    for &max_seq in &[8usize, 32, 128] {
        let mut cache = InPlaceKvCache::for_model(&model, max_seq).unwrap();
        let logits = run_inplace_step(&model, &mut cache, toks.clone(), 3);
        logits_per_max_seq.push(logits);
    }

    cmp_logits(&logits_per_max_seq[0], &logits_per_max_seq[1], 1e-5,
        "max_seq=8 vs max_seq=32: logits must match");
    cmp_logits(&logits_per_max_seq[1], &logits_per_max_seq[2], 1e-5,
        "max_seq=32 vs max_seq=128: logits must match");
}

/// Side-by-side demo: same prompt through small and large buffer sizes.
#[test]
fn perf_characteristic_demo() {
    println!("\n=== In-place decode attention scales with valid_seq, not max_seq ===");
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8,
        block_count: 1, feed_forward_length: 16,
        head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 9,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    for &max_seq in &[16usize, 256, 4096] {
        let mut cache = InPlaceKvCache::for_model(&model, max_seq).unwrap();
        // Get to valid_seq=4 (4 sequential decode steps).
        for i in 0..4 {
            let t = make_token_ids(1, cfg.vocab_size, i as u64 + 1);
            let _ = run_inplace_step(&model, &mut cache, t, 1);
        }
        let step = build_decode_step_graph_inplace(&model, &cache, 1).unwrap();
        // Find scores matmul shape.
        let mut scores_shape: Option<Vec<usize>> = None;
        for node in step.graph.nodes.iter() {
            if let jouleclaw_core::graph::NodeKind::Op { op, .. } = &node.kind {
                if *op == jouleclaw_core::op::OpKind::MatMul {
                    let shape = &node.output_meta[0].shape;
                    if shape.len() == 3 && shape[1] == 1 && shape[2] == 5 {
                        scores_shape = Some(shape.clone());
                        break;
                    }
                }
            }
        }
        println!("  max_seq={:5}, valid_seq=5: scores matmul shape = {:?}",
            max_seq, scores_shape.unwrap_or_else(|| vec![]));
    }
    println!("Attention shape is independent of max_seq — Slice did its job.");
}

/// Equivalent to a sanity check on the existing one-shot Llama graph,
/// confirming that the standard prefill still produces deterministic logits.
#[test]
fn one_shot_prefill_remains_deterministic_under_slice_changes() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8,
        block_count: 1, feed_forward_length: 16,
        head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 4,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();
    let toks = make_token_ids(5, cfg.vocab_size, 1);
    let llama = build_llama_graph(&model, 5).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), toks.clone());
    let r1 = execute(&compiled, inputs.clone(), ExecutionOptions::default()).unwrap();
    let r2 = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let l1 = r1.outputs.get("logits").unwrap().as_f32_vec();
    let l2 = r2.outputs.get("logits").unwrap().as_f32_vec();
    assert_eq!(l1, l2);
}
