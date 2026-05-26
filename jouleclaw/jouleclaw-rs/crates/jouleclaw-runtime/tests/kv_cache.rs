//! KV cache integration test.
//!
//! The defining correctness property: given the same total input sequence,
//! incremental decode (prefill then step-by-step) must produce the same
//! final logits as one-shot prefill. The test runs both pipelines and
//! compares the logits at each shared token position.

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_loader_gguf::decode::build_decode_step_graph;
use jouleclaw_loader_gguf::kv_cache::KvCache;
use jouleclaw_loader_gguf::llama::build_llama_graph;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::read_gguf;
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

/// Run one decode step. Updates `cache` in place; returns logits for the new tokens.
fn run_decode_step(
    model: &jouleclaw_loader_gguf::GgufModel,
    cache: &mut KvCache,
    new_token_ids: Tensor,
    new_seq: usize,
) -> Vec<f32> {
    // Build the step graph for the current cache state.
    let step = build_decode_step_graph(model, cache, new_seq).expect("build decode step");

    let runtime = Runtime::boot();
    let compiled = compile(step.graph, &runtime.kernels).expect("compile");

    // Bind inputs: token_ids + per-layer cached K and V (when present).
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), new_token_ids);
    if cache.current_seq > 0 {
        for layer in 0..step.config.block_count {
            let k_name = &step.k_input_names[layer];
            let v_name = &step.v_input_names[layer];
            let k_tensor = cache.k_for(layer)
                .expect("cache K should be filled when current_seq > 0").clone();
            let v_tensor = cache.v_for(layer)
                .expect("cache V should be filled when current_seq > 0").clone();
            inputs.insert(k_name.clone(), k_tensor);
            inputs.insert(v_name.clone(), v_tensor);
        }
    }

    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");

    // Pull the updated K and V for each layer out of the result and stash
    // them back into the cache.
    for layer in 0..step.config.block_count {
        let k_out_name = &step.k_output_names[layer];
        let v_out_name = &step.v_output_names[layer];
        let k_full = res.outputs.get(k_out_name).expect("kv_out_k").clone();
        let v_full = res.outputs.get(v_out_name).expect("kv_out_v").clone();
        cache.put(layer, k_full, v_full);
    }

    let logits = res.outputs.get(&step.logits_output_name).expect("logits");
    logits.as_f32_vec()
}

fn cmp_logits(actual: &[f32], expected: &[f32], rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(),
        "{}: length mismatch {} vs {}", label, actual.len(), expected.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let abs = (a - e).abs();
        if abs > max_abs { max_abs = abs; }
        let denom = e.abs().max(1e-6);
        let rel = abs / denom;
        if rel > max_rel { max_rel = rel; }
    }
    assert!(max_rel < rtol,
        "{}: max relative diff {:.3e} exceeds tolerance {} (max abs diff {:.3e})",
        label, max_rel, rtol, max_abs);
}

/// Core correctness test: incremental decode matches one-shot prefill.
///
/// Setup: 8-token sequence. One-shot path runs the full prefill graph.
/// Incremental path runs decode steps of various sizes that sum to 8.
/// At the end, the logits for the last token must match between paths
/// (within numerical tolerance, since the ops are floating-point).
#[test]
fn incremental_decode_matches_one_shot_prefill() {
    let cfg = SyntheticConfig {
        vocab_size: 32,
        embedding_length: 16,
        block_count: 2,
        feed_forward_length: 32,
        head_count: 4,
        head_count_kv: 4,
        rms_eps: 1e-6,
        seed: 7,
        vocab: None,
        merges: None,
        bos_id: None,
        eos_id: None,
        unk_id: None, chat_template: None,
    };

    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let total_seq = 6;
    let full_tokens = make_token_ids(total_seq, cfg.vocab_size, 1);

    // -------- One-shot prefill path --------
    let llama = build_llama_graph(&model, total_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), full_tokens.clone());
    let one_shot_res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_logits = one_shot_res.outputs.get("logits").unwrap().as_f32_vec();

    // -------- Incremental decode path --------
    // Step sizes: [3, 1, 1, 1] sums to 6.
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

    // Sanity: cache final state should reflect total_seq.
    assert_eq!(cache.current_seq, total_seq);

    // Compare. Floating-point reductions will produce small differences
    // because the order of operations is slightly different (concat
    // changes how the same accumulations are arranged). Tolerance: ~1e-3
    // relative. In practice the magnitudes are much closer than this.
    cmp_logits(&all_logits, &one_shot_logits, 1e-3,
        "incremental decode vs one-shot");

    println!("incremental decode total_seq={} matches one-shot prefill within tolerance", total_seq);
}

/// Cache state evolves correctly across multiple decode steps.
#[test]
fn kv_cache_state_grows_correctly() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8, block_count: 1,
        feed_forward_length: 16, head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 3,
        vocab: None, merges: None,
        bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let mut cache = KvCache::empty(cfg.block_count);
    assert_eq!(cache.current_seq, 0);
    assert!(cache.k_for(0).is_none());

    // Step 1: prefill 3 tokens.
    let toks = make_token_ids(3, cfg.vocab_size, 1);
    let _ = run_decode_step(&model, &mut cache, toks, 3);
    assert_eq!(cache.current_seq, 3);
    let k0 = cache.k_for(0).expect("K should be populated after first step");
    assert_eq!(k0.meta.shape, vec![cfg.head_count, 3, cfg.embedding_length / cfg.head_count]);

    // Step 2: append 1 token.
    let toks = make_token_ids(1, cfg.vocab_size, 2);
    let _ = run_decode_step(&model, &mut cache, toks, 1);
    assert_eq!(cache.current_seq, 4);
    let k0 = cache.k_for(0).unwrap();
    assert_eq!(k0.meta.shape[1], 4, "K seq dim should be 4 after one append");

    // Step 3: append 2 more.
    let toks = make_token_ids(2, cfg.vocab_size, 3);
    let _ = run_decode_step(&model, &mut cache, toks, 2);
    assert_eq!(cache.current_seq, 6);
    assert_eq!(cache.k_for(0).unwrap().meta.shape[1], 6);

    // Reset clears it.
    cache.reset();
    assert_eq!(cache.current_seq, 0);
    assert!(cache.k_for(0).is_none());
}

/// First decode step (cached_seq = 0) should produce identical logits to
/// one-shot prefill on the same tokens.
#[test]
fn first_decode_step_matches_one_shot() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8, block_count: 1,
        feed_forward_length: 16, head_count: 2, head_count_kv: 2,
        rms_eps: 1e-6, seed: 5,
        vocab: None, merges: None,
        bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let seq = 3;
    let toks = make_token_ids(seq, cfg.vocab_size, 11);

    // One-shot.
    let llama = build_llama_graph(&model, seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), toks.clone());
    let one_shot = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let one_shot_l = one_shot.outputs.get("logits").unwrap().as_f32_vec();

    // First decode step (no cache).
    let mut cache = KvCache::empty(cfg.block_count);
    let step_l = run_decode_step(&model, &mut cache, toks, seq);

    // These should match exactly: zero cached_seq means causal_offset = 0,
    // which is identical to standard causal mask. Both paths run the same
    // ops in the same order.
    cmp_logits(&step_l, &one_shot_l, 1e-3,
        "first decode step vs one-shot prefill (same seq, no cache)");
}
