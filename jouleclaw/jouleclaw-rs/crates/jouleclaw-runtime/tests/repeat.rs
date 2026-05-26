//! Tests for the Repeat primitive.
//!
//! Repeat semantics: `output[..., i*N + j, ...] = input[..., j, ...]`
//! for `i in 0..repeats`, where N is the input's size on the chosen axis.
//! This is a "tile" pattern (repeats whole-axis blocks contiguously);
//! it is NOT a "repeat-interleave" pattern (which the GQA block builds
//! via reshape+repeat+reshape).

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_repeat(data: Vec<f32>, shape: &[usize], axis: i32, repeats: usize) -> Tensor {
    let mut g = GraphBuilder::new();
    let meta = TensorMeta::new(Dtype::F32, shape);
    let x = g.input("x", meta.clone());
    let y = g.repeat(x, axis, repeats);
    g.output("y", y);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(meta, &data));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("y").unwrap().clone()
}

/// 1D repeat: tile the whole sequence twice.
#[test]
fn repeat_1d_tiles_axis() {
    let out = run_repeat(vec![1.0, 2.0, 3.0], &[3], 0, 2);
    assert_eq!(out.meta.shape, vec![6]);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

/// repeats=1 is a no-op (same shape, copy of data).
#[test]
fn repeat_one_is_identity() {
    let out = run_repeat(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], 0, 1);
    assert_eq!(out.meta.shape, vec![2, 2]);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// 2D repeat on axis 0: tile rows.
#[test]
fn repeat_2d_axis_0() {
    // 2x3 input → repeat 3 times along axis 0 → 6x3.
    let data = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
    let out = run_repeat(data, &[2, 3], 0, 3);
    assert_eq!(out.meta.shape, vec![6, 3]);
    let expected = vec![
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
    ];
    assert_eq!(out.as_f32_vec(), expected);
}

/// 2D repeat on axis 1: tile within each row.
#[test]
fn repeat_2d_axis_1() {
    let data = vec![1.0, 2.0,  3.0, 4.0];
    let out = run_repeat(data, &[2, 2], 1, 3);
    assert_eq!(out.meta.shape, vec![2, 6]);
    // Each row: [1,2,1,2,1,2] / [3,4,3,4,3,4]
    let expected = vec![
        1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
    ];
    assert_eq!(out.as_f32_vec(), expected);
}

/// Negative axis indexing.
#[test]
fn repeat_negative_axis() {
    let data = vec![1.0, 2.0,  3.0, 4.0];
    let out = run_repeat(data, &[2, 2], -1, 2);
    assert_eq!(out.meta.shape, vec![2, 4]);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 1.0, 2.0,  3.0, 4.0, 3.0, 4.0]);
}

/// 3D repeat on middle axis (the GQA-broadcast pattern):
/// `[n_kv, 1, seq*d_head] → [n_kv, group_size, seq*d_head]` then reshape
/// produces the repeat-interleaved K/V. Just test the raw repeat here.
#[test]
fn repeat_3d_middle_axis_for_gqa_broadcast() {
    // Input [2, 1, 3]: two kv-heads, singleton group axis, 3 features.
    // Repeat axis=1 with repeats=2: result [2, 2, 3] with each kv-head's
    // content duplicated along the new group axis.
    let data = vec![
        1.0, 2.0, 3.0,    // kv-head 0
        4.0, 5.0, 6.0,    // kv-head 1
    ];
    let out = run_repeat(data, &[2, 1, 3], 1, 2);
    assert_eq!(out.meta.shape, vec![2, 2, 3]);
    // kv-head 0 row 0: [1,2,3]; row 1: [1,2,3]
    // kv-head 1 row 0: [4,5,6]; row 1: [4,5,6]
    // After reshape to [4, 3] this gives [1,2,3, 1,2,3, 4,5,6, 4,5,6]
    // — the repeat-interleave pattern needed for GQA.
    let expected = vec![
        1.0, 2.0, 3.0,  1.0, 2.0, 3.0,    // kv-head 0 ×2
        4.0, 5.0, 6.0,  4.0, 5.0, 6.0,    // kv-head 1 ×2
    ];
    assert_eq!(out.as_f32_vec(), expected);
}

/// Repeat is deterministic.
#[test]
fn repeat_is_deterministic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let out1 = run_repeat(data.clone(), &[5], 0, 4);
    let out2 = run_repeat(data, &[5], 0, 4);
    assert_eq!(out1.storage.bytes, out2.storage.bytes);
}
