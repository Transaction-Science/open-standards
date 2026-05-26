//! Tests for Concat (used to grow the KV cache) and offset-causal softmax
//! (used in decode steps where the query slice is shorter than the key slice).

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_concat(
    a_data: Vec<f32>, a_shape: &[usize],
    b_data: Vec<f32>, b_shape: &[usize],
    axis: i32,
) -> Tensor {
    let mut g = GraphBuilder::new();
    let a_meta = TensorMeta::new(Dtype::F32, a_shape);
    let b_meta = TensorMeta::new(Dtype::F32, b_shape);
    let a = g.input("a", a_meta.clone());
    let b = g.input("b", b_meta.clone());
    let c = g.concat(a, b, axis);
    g.output("c", c);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("a".into(), Tensor::from_f32(a_meta, &a_data));
    inputs.insert("b".into(), Tensor::from_f32(b_meta, &b_data));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("c").unwrap().clone()
}

/// 1D concat: simple element append.
#[test]
fn concat_1d_appends_elements() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0];
    let out = run_concat(a, &[3], b, &[2], 0);
    assert_eq!(out.meta.shape, vec![5]);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// 2D concat along axis 0: append rows.
#[test]
fn concat_2d_axis_0_appends_rows() {
    // a: [[1,2,3],[4,5,6]] (2x3)
    // b: [[7,8,9]] (1x3)
    // result: 3x3
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0];
    let out = run_concat(a, &[2, 3], b, &[1, 3], 0);
    assert_eq!(out.meta.shape, vec![3, 3]);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

/// 2D concat along axis 1: append columns. Hand-verified.
/// a: [[1,2],[3,4]]   b: [[5,6,7],[8,9,10]]  axis=1
/// result row 0: [1,2,5,6,7]; row 1: [3,4,8,9,10]
#[test]
fn concat_2d_axis_1_appends_columns() {
    let a = vec![1.0, 2.0,  3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0,  8.0, 9.0, 10.0];
    let out = run_concat(a, &[2, 2], b, &[2, 3], 1);
    assert_eq!(out.meta.shape, vec![2, 5]);
    let expected = vec![1.0, 2.0, 5.0, 6.0, 7.0,  3.0, 4.0, 8.0, 9.0, 10.0];
    assert_eq!(out.as_f32_vec(), expected);
}

/// 3D concat along middle axis (the seq axis in [n_heads, seq, d_head]).
/// Used for KV cache: append new K/V slice along seq axis.
#[test]
fn concat_3d_middle_axis_for_kv_cache() {
    let n_heads = 2;
    let seq_a = 3;
    let seq_b = 1;
    let d_head = 2;

    // K_prev: [2, 3, 2] = 12 elements
    let a: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    // K_new: [2, 1, 2] = 4 elements
    let b: Vec<f32> = (0..4).map(|i| i as f32 + 100.0).collect();

    let out = run_concat(a.clone(), &[n_heads, seq_a, d_head],
                          b.clone(), &[n_heads, seq_b, d_head], 1);
    assert_eq!(out.meta.shape, vec![n_heads, seq_a + seq_b, d_head]);

    let result = out.as_f32_vec();
    // Head 0: a's seq_a rows then b's seq_b rows
    // Head 0 a-block starts at 0, b-block at offset 6 in result
    // Head 0 result: a[0..6] + b[0..2] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 100, 101]
    let head0: &[f32] = &result[..(seq_a + seq_b) * d_head];
    assert_eq!(head0, &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 100.0, 101.0]);

    // Head 1: a's [6..12] then b's [2..4]
    // Head 1 result: [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 102, 103]
    let head1: &[f32] = &result[(seq_a + seq_b) * d_head..];
    let expected_head1 = vec![0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 102.0, 103.0];
    for (i, (a_v, b_v)) in head1.iter().zip(expected_head1.iter()).enumerate() {
        assert!((a_v - b_v).abs() < 1e-5,
            "head1[{}] expected {}, got {}", i, b_v, a_v);
    }
}

/// Concat with negative axis index (-1 = last axis).
#[test]
fn concat_negative_axis() {
    let a = vec![1.0, 2.0,  3.0, 4.0];
    let b = vec![5.0,  6.0];
    let out = run_concat(a, &[2, 2], b, &[2, 1], -1);
    assert_eq!(out.meta.shape, vec![2, 3]);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 5.0,  3.0, 4.0, 6.0]);
}

/// Concat is deterministic.
#[test]
fn concat_is_deterministic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0];
    let out1 = run_concat(a.clone(), &[4], b.clone(), &[2], 0);
    let out2 = run_concat(a, &[4], b, &[2], 0);
    assert_eq!(out1.storage.bytes, out2.storage.bytes);
}

// =====================================================================
// Offset-causal softmax (for KV-cache decode)
// =====================================================================

fn run_softmax_offset(input: Vec<f32>, shape: &[usize], offset: i32) -> Vec<f32> {
    let mut g = GraphBuilder::new();
    let in_meta = TensorMeta::new(Dtype::F32, shape);
    let x = g.input("x", in_meta.clone());
    let y = g.softmax_causal_offset(x, -1, offset);
    g.output("y", y);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(in_meta, &input));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("y").unwrap().as_f32_vec()
}

/// Decode step with one new query against four total keys.
/// Shape: [seq_q=1, seq_k=4]. The single query is at absolute position 3
/// (causal_offset = 3), so it can attend to all 4 keys.
#[test]
fn softmax_offset_decode_one_query_all_visible() {
    // shape [1, 4]: one query row, four keys.
    let logits = vec![1.0, 2.0, 0.5, 1.5];
    // causal_offset = 3 means query at relative pos 0 attends to keys 0..=3.
    let out = run_softmax_offset(logits, &[1, 4], 3);
    // All 4 keys visible; standard softmax.
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax row should sum to 1");
    for v in &out { assert!(*v > 0.0); }
}

/// Decode step with one new query at absolute position 0.
/// causal_offset = 0, seq_q = 1: query attends to keys 0..=0 only.
/// All other keys should be masked even though they exist in the row.
#[test]
fn softmax_offset_zero_offset_first_decode_step() {
    let logits = vec![1.0, 999.0, 999.0, 999.0];  // huge logits at masked positions
    let out = run_softmax_offset(logits, &[1, 4], 0);
    // Only key 0 visible.
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!(out[1].abs() < 1e-6);
    assert!(out[2].abs() < 1e-6);
    assert!(out[3].abs() < 1e-6);
}

/// Decode step with two new queries against five keys.
/// Shape [2, 5], causal_offset = 3.
/// Query 0 (relative pos 0) at abs position 3: attends to keys 0..=3 (mask key 4).
/// Query 1 (relative pos 1) at abs position 4: attends to keys 0..=4 (no mask).
#[test]
fn softmax_offset_two_new_queries_partial_mask() {
    let logits = vec![
        1.0, 1.0, 1.0, 1.0, 999.0,    // q0: should mask key 4
        1.0, 1.0, 1.0, 1.0, 1.0,      // q1: no mask
    ];
    let out = run_softmax_offset(logits, &[2, 5], 3);

    // q0 row: key 4 masked. probs[0..4] should be uniform 0.25, probs[4] = 0.
    let q0 = &out[..5];
    for i in 0..4 {
        assert!((q0[i] - 0.25).abs() < 1e-5,
            "q0[{}] expected 0.25, got {}", i, q0[i]);
    }
    assert!(q0[4].abs() < 1e-6, "q0[4] should be masked, got {}", q0[4]);

    // q1 row: no mask, uniform 0.2.
    let q1 = &out[5..];
    for i in 0..5 {
        assert!((q1[i] - 0.2).abs() < 1e-5,
            "q1[{}] expected 0.2, got {}", i, q1[i]);
    }
}

/// Multi-head offset-causal softmax: shape [n_heads, seq_q, seq_k].
/// Verifies the mask is applied per-head independently.
#[test]
fn softmax_offset_multihead() {
    let n_heads = 2;
    let seq_q = 1;
    let seq_k = 3;
    // All logits = 1.0. With offset=2, query at rel pos 0 = abs 2 attends to all 3 keys.
    let total = n_heads * seq_q * seq_k;
    let logits = vec![1.0_f32; total];
    let out = run_softmax_offset(logits, &[n_heads, seq_q, seq_k], 2);

    for h in 0..n_heads {
        let row = &out[h * seq_k..(h + 1) * seq_k];
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        for v in row { assert!((v - 1.0 / 3.0).abs() < 1e-5); }
    }
}
