//! Test that the storage-aliasing optimization actually fires in the
//! decode path used by `generate()` and `generate_stream()`.
//!
//! The alias is correctness-preserving by design (the executor falls
//! back to plain Scatter when the Arc is shared), so the existing 197
//! tests already cover correctness. What's new here is verifying the
//! optimization actually fires — i.e., the K/V buffer Arc pointer
//! survives across decode steps (storage is stolen, mutated in place,
//! and returned).

use jouleclaw_loader_gguf::kv_cache_inplace::{
    build_decode_step_graph_inplace, InPlaceKvCache,
};
use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

fn build_test_model() -> jouleclaw_loader_gguf::GgufModel {
    let cfg = SyntheticConfig {
        vocab_size: 16,
        embedding_length: 8,
        block_count: 1,
        feed_forward_length: 16,
        head_count: 2,
        head_count_kv: 2,
        rms_eps: 1e-6,
        seed: 50,
        vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    read_gguf(Cursor::new(bytes)).unwrap()
}

fn token_tensor(ids: &[u32]) -> Tensor {
    let bytes: Vec<u8> = ids.iter().flat_map(|&id| (id as i32).to_le_bytes()).collect();
    Tensor {
        meta: TensorMeta::new(Dtype::I32, &[ids.len()]),
        storage: Arc::new(TensorStorage { bytes, mapped: None }),
    }
}

/// Execute one decode step using the take/replace pattern that the
/// decode path uses. Return the K-buffer storage pointer before and
/// after for layer 0.
fn one_step_with_addrs(
    model: &jouleclaw_loader_gguf::GgufModel,
    cache: &mut InPlaceKvCache,
    tokens: &[u32],
    new_seq: usize,
) -> (usize, usize) {
    // Pointer before step.
    let before = Arc::as_ptr(&cache.k_bufs[0].storage) as usize;

    // Build the step graph and execute, using take/replace just like
    // run_inplace_step does internally.
    let step = build_decode_step_graph_inplace(model, cache, new_seq).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(step.graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), token_tensor(tokens));
    for layer in 0..step.config.block_count {
        let (k, v) = cache.take_buffers(layer);
        inputs.insert(step.k_input_names[layer].clone(), k);
        inputs.insert(step.v_input_names[layer].clone(), v);
    }
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    for layer in 0..step.config.block_count {
        let k = res.outputs.get(&step.k_output_names[layer]).unwrap().clone();
        let v = res.outputs.get(&step.v_output_names[layer]).unwrap().clone();
        cache.replace_buffers(layer, k, v);
    }
    cache.advance(new_seq);
    let after = Arc::as_ptr(&cache.k_bufs[0].storage) as usize;
    (before, after)
}

/// THE proof: the K buffer's storage Arc has the same address before
/// and after a decode step. This means the executor stole the input's
/// storage, the Scatter kernel wrote into it in-place, and the result
/// was put back into the cache — all sharing one allocation.
#[test]
fn alias_fires_through_full_decode_path() {
    let model = build_test_model();
    let mut cache = InPlaceKvCache::for_model(&model, 32).unwrap();

    // First decode step: cached_seq=0, so the in-place block builder may
    // skip the K_prev input. Skip that step and check the second step
    // where the cache is actually used as input.
    let _ = one_step_with_addrs(&model, &mut cache, &[1], 1);

    // Second decode step: cached_seq=1. K_prev IS an input. This is the
    // step where aliasing matters.
    let (before, after) = one_step_with_addrs(&model, &mut cache, &[2], 1);

    assert_eq!(before, after,
        "K storage Arc address should survive a decode step \
         when aliasing fires. before=0x{:x}, after=0x{:x}",
        before, after);
}

/// Across many decode steps, the same storage Arc persists. The cache
/// truly reuses one allocation for the lifetime of the conversation.
#[test]
fn one_allocation_persists_across_many_steps() {
    let model = build_test_model();
    let mut cache = InPlaceKvCache::for_model(&model, 32).unwrap();
    let _ = one_step_with_addrs(&model, &mut cache, &[1], 1);  // priming step

    let initial = Arc::as_ptr(&cache.k_bufs[0].storage) as usize;
    let mut all_addrs = vec![initial];
    for tok in 2..8u32 {
        let (_, after) = one_step_with_addrs(&model, &mut cache, &[tok], 1);
        all_addrs.push(after);
    }

    let distinct: std::collections::HashSet<_> = all_addrs.iter().collect();
    assert_eq!(distinct.len(), 1,
        "expected one persistent allocation across all steps, \
         got {} distinct addrs: {:x?}",
        distinct.len(), all_addrs);
}

/// Demo: visible side-by-side of the storage addresses across a real
/// decode loop.
#[test]
fn alias_decode_demo() {
    println!("\n=== Storage aliasing fires through the decode path ===");
    let model = build_test_model();
    let mut cache = InPlaceKvCache::for_model(&model, 24).unwrap();

    // Priming step (cached_seq=0 has no K_prev input).
    let _ = one_step_with_addrs(&model, &mut cache, &[1], 1);

    println!("After priming, K buffer at: 0x{:012x}",
        Arc::as_ptr(&cache.k_bufs[0].storage) as usize);
    println!();
    println!("Each subsequent decode step:");

    for (i, tok) in (2..7u32).enumerate() {
        let (before, after) = one_step_with_addrs(&model, &mut cache, &[tok], 1);
        let aliased = if before == after { "ALIASED ✓" } else { "fresh alloc" };
        println!("  step {}: before=0x{:012x}  after=0x{:012x}  {}",
            i + 1, before, after, aliased);
    }

    println!();
    println!("The K/V storage allocations are reused for the lifetime");
    println!("of the conversation. No per-step max_seq-sized memcpy.");
}
