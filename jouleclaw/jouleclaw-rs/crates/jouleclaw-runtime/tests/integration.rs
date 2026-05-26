//! Integration tests for the Phase 1.1 reference path.

use jouleclaw_core::determinism::DeterminismMode;
use jouleclaw_core::error::Error;
use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::op::SamplerKind;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

/// Lookup: index a 4-row, 3-col F32 table with known indices.
#[test]
fn lookup_returns_correct_rows() {
    let mut g = GraphBuilder::new();
    let idx = g.input("idx", TensorMeta::new(Dtype::I32, &[3]));
    let table = g.input("table", TensorMeta::new(Dtype::F32, &[4, 3]));
    let out = g.lookup(idx, table);
    g.output("y", out);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    // Build I32 indices [2, 0, 3].
    let idx_meta = TensorMeta::new(Dtype::I32, &[3]);
    let idx_bytes: Vec<u8> = [2i32, 0, 3].iter()
        .flat_map(|v| v.to_le_bytes()).collect();
    let idx_tensor = Tensor {
        meta: idx_meta,
        storage: std::sync::Arc::new(jouleclaw_core::tensor::TensorStorage { bytes: idx_bytes, mapped: None }),
    };

    let table_data: Vec<f32> = vec![
        0.1, 0.2, 0.3,   // row 0
        1.1, 1.2, 1.3,   // row 1
        2.1, 2.2, 2.3,   // row 2
        3.1, 3.2, 3.3,   // row 3
    ];
    let table_tensor = Tensor::from_f32(TensorMeta::new(Dtype::F32, &[4, 3]), &table_data);

    let mut inputs = HashMap::new();
    inputs.insert("idx".into(), idx_tensor);
    inputs.insert("table".into(), table_tensor);

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let y = res.outputs.get("y").unwrap().as_f32_vec();

    // Expected: rows [2, 0, 3] in order.
    let expected = vec![2.1, 2.2, 2.3,  0.1, 0.2, 0.3,  3.1, 3.2, 3.3];
    assert_eq!(y, expected);
}

/// Greedy sample picks the argmax with low-index tie-break.
#[test]
fn sample_greedy_picks_argmax() {
    let mut g = GraphBuilder::new();
    let logits = g.input("logits", TensorMeta::new(Dtype::F32, &[5]));
    let out = g.sample(logits, SamplerKind::Greedy, None);
    g.output("y", out);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    // Logits with the max at index 3.
    let l = vec![0.1f32, -1.0, 0.5, 2.5, 1.7];
    let mut inputs = HashMap::new();
    inputs.insert("logits".into(),
        Tensor::from_f32(TensorMeta::new(Dtype::F32, &[5]), &l));

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let y = res.outputs.get("y").unwrap();
    let mut b = [0u8; 4];
    b.copy_from_slice(&y.storage.bytes[..4]);
    let id = i32::from_le_bytes(b);
    assert_eq!(id, 3);
}

/// Seeded TopK is deterministic given the same seed, different from the
/// argmax in general.
#[test]
fn sample_topk_is_seed_deterministic() {
    let mut g = GraphBuilder::new();
    let logits = g.input("logits", TensorMeta::new(Dtype::F32, &[8]));
    let out = g.sample(logits, SamplerKind::TopK { k: 4 }, Some(42));
    g.output("y", out);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let l = vec![0.1f32, -1.0, 0.5, 2.5, 1.7, 1.6, 0.0, -2.0];
    let mut inputs = HashMap::new();
    inputs.insert("logits".into(),
        Tensor::from_f32(TensorMeta::new(Dtype::F32, &[8]), &l));

    let res1 = execute(&compiled, inputs.clone(), ExecutionOptions::default()).unwrap();
    let res2 = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    assert_eq!(res1.outputs.get("y").unwrap().storage.bytes,
               res2.outputs.get("y").unwrap().storage.bytes,
               "seeded TopK must produce identical results across runs");
}

/// Strict-deterministic mode rejects an unseeded sampler.
#[test]
fn strict_mode_rejects_unseeded_sampler() {
    let mut g = GraphBuilder::new();
    let logits = g.input("logits", TensorMeta::new(Dtype::F32, &[4]));
    // No seed — this op's determinism class becomes Stochastic.
    let out = g.sample(logits, SamplerKind::TopK { k: 2 }, None);
    g.output("y", out);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let l = vec![0.1f32, -1.0, 0.5, 2.5];
    let mut inputs = HashMap::new();
    inputs.insert("logits".into(),
        Tensor::from_f32(TensorMeta::new(Dtype::F32, &[4]), &l));

    // Strict mode — the kernel itself reports SeededStochastic, but it will
    // fail at execute time when it sees no seed in attrs. Either rejection
    // is acceptable; what matters is that strict mode produces an error
    // rather than a non-deterministic sample.
    let result = execute(&compiled, inputs, ExecutionOptions {
        determinism: DeterminismMode::Strict,
        seed: None,
    });
    assert!(result.is_err(), "strict mode must reject unseeded sampling");
    if let Err(Error::Execution(e)) = result {
        let msg = format!("{:?}", e);
        assert!(msg.contains("seed") || msg.contains("Sample"),
            "error should mention seed or Sample: {}", msg);
    }
}
