//! End-to-end Llama loader test.
//!
//! Full pipeline:
//!   1. Synthesize a tiny Llama-style GGUF buffer (2 blocks, d=16, dff=32, vocab=32)
//!   2. Parse it via the GGUF parser
//!   3. Build the inference graph via the Llama loader
//!   4. Compile + execute with deterministic random token IDs
//!   5. Verify: output shape is [seq, vocab], probabilities sum to 1, runs
//!      twice produce bit-identical output

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
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

#[test]
fn synthesize_parse_load_execute_end_to_end() {
    let cfg = SyntheticConfig::default();
    let bytes = synthesize_llama_gguf(&cfg);
    println!("Synthetic GGUF size: {} bytes", bytes.len());

    // Parse the GGUF.
    let model = read_gguf(Cursor::new(bytes)).expect("GGUF parse");
    assert_eq!(model.metadata_string("general.architecture"), Some("llama"));
    assert_eq!(model.metadata_u64("llama.block_count"), Some(cfg.block_count as u64));
    println!("Parsed model: {} tensors, {} metadata entries",
        model.tensors.len(), model.metadata.len());

    // Build the inference graph.
    let seq_len = 4;
    let llama = build_llama_graph(&model, seq_len).expect("build llama graph");
    println!("Built graph: {} nodes (config: {:?})",
        llama.graph.nodes.len(), llama.config);

    // Compile.
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).expect("compile");
    println!("Compiled: {} plan entries", compiled.plan.len());

    // Bind tokens and execute.
    let token_ids = make_token_ids(seq_len, cfg.vocab_size, 7);
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), token_ids.clone());

    let res1 = execute(&compiled, inputs.clone(), ExecutionOptions::default())
        .expect("execute 1");
    let res2 = execute(&compiled, inputs, ExecutionOptions::default())
        .expect("execute 2");

    // Output shape check.
    let logits = res1.outputs.get("logits").expect("logits output");
    assert_eq!(logits.meta.shape, vec![seq_len, cfg.vocab_size],
        "logits should be [seq_len, vocab_size]");

    // Bit-identical determinism check.
    assert_eq!(
        res1.outputs.get("logits").unwrap().storage.bytes,
        res2.outputs.get("logits").unwrap().storage.bytes,
        "logits must be byte-identical across runs"
    );
    assert_eq!(res1.trace.output_hashes, res2.trace.output_hashes);

    // Probability check: softmax over vocab dim should sum to 1.
    let l = logits.as_f32_vec();
    let v = cfg.vocab_size;
    for row in 0..seq_len {
        let row_logits = &l[row * v..(row + 1) * v];
        let max = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row_logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let p_sum: f32 = probs.iter().sum();
        assert!((p_sum - 1.0).abs() < 1e-5,
            "softmax over logits row {} should sum to 1.0, got {}", row, p_sum);
    }

    // Joule accounting: should be non-trivial — many MatMuls, plus the
    // other primitives.
    let acc = &res1.trace.joule_accounting;
    assert!(acc.total_joules > 0.0);
    println!("\n--- Forward pass joule accounting ---");
    println!("total: {:.6e} J", acc.total_joules);
    let mut by_op: Vec<_> = acc.by_op_kind.iter().collect();
    by_op.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (op, j) in by_op {
        println!("  {:>10?}: {:.6e} J", op, j);
    }

    // First few logits, for sanity.
    println!("\nLogits[0, 0..8] = {:?}", &l[0..8.min(l.len())]);
    println!("Wall clock: {:?}", res1.trace.wall_clock);
    println!("\nOK: end-to-end Llama load + forward pass + determinism");
}

/// Smaller, faster test variant — single block, d=8, vocab=16, seq=2.
/// Purpose: quick smoke test with minimal compute.
#[test]
fn smoke_test_minimum_viable_model() {
    let cfg = SyntheticConfig {
        vocab_size: 16, embedding_length: 8, block_count: 1,
        feed_forward_length: 16, head_count: 1, head_count_kv: 1, rms_eps: 1e-6, seed: 1, vocab: None, merges: None, bos_id: None, eos_id: None, unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();
    let llama = build_llama_graph(&model, 2).unwrap();
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).unwrap();

    let token_ids = make_token_ids(2, cfg.vocab_size, 1);
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), token_ids);

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let logits = res.outputs.get("logits").unwrap();
    assert_eq!(logits.meta.shape, vec![2, cfg.vocab_size]);
}

/// Multi-head variant: 4 heads, d_model=16 (so d_head=4), 2 blocks, vocab=32.
/// This exercises the full multi_head_attention path through the Llama loader.
#[test]
fn multi_head_llama_end_to_end() {
    let cfg = SyntheticConfig {
        vocab_size: 32,
        embedding_length: 16,
        block_count: 2,
        feed_forward_length: 32,
        head_count: 4,
        head_count_kv: 4,// 16 / 4 = 4 d_head
        rms_eps: 1e-6,
        seed: 99,
        vocab: None,
        merges: None,
        bos_id: None,
        eos_id: None,
        unk_id: None, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();

    let seq_len = 4;
    let llama = build_llama_graph(&model, seq_len).expect("multi-head graph build");
    println!("Multi-head graph: {} nodes, head_count={}, d_head={}",
        llama.graph.nodes.len(),
        llama.config.head_count,
        llama.config.embedding_length / llama.config.head_count);

    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).expect("compile");

    let token_ids = make_token_ids(seq_len, cfg.vocab_size, 7);
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), token_ids);

    let res1 = execute(&compiled, inputs.clone(), ExecutionOptions::default())
        .expect("execute 1");
    let res2 = execute(&compiled, inputs, ExecutionOptions::default())
        .expect("execute 2");

    let logits = res1.outputs.get("logits").unwrap();
    assert_eq!(logits.meta.shape, vec![seq_len, cfg.vocab_size]);

    // Bit-identical determinism check (with reshape + transpose + batched
    // matmul in the path, this is a stronger statement than for single-head).
    assert_eq!(
        res1.outputs.get("logits").unwrap().storage.bytes,
        res2.outputs.get("logits").unwrap().storage.bytes,
        "multi-head Llama must be deterministic"
    );

    // Logits softmax sums to 1 across vocab.
    let l = logits.as_f32_vec();
    for row in 0..seq_len {
        let row_logits = &l[row * cfg.vocab_size..(row + 1) * cfg.vocab_size];
        let max = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row_logits.iter().map(|x| (x - max).exp()).sum();
        let probs: Vec<f32> = row_logits.iter().map(|x| (x - max).exp() / sum).collect();
        let p_sum: f32 = probs.iter().sum();
        assert!((p_sum - 1.0).abs() < 1e-5,
            "multi-head logits row {} softmax should sum to 1.0, got {}", row, p_sum);
    }

    println!("Multi-head joules: {:.6e} J, wall: {:?}",
        res1.trace.joule_accounting.total_joules,
        res1.trace.wall_clock);
}
