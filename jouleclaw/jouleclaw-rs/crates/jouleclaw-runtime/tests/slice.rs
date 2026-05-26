//! Tests for the Slice primitive.
//!
//! Slice semantics: `output = input[..., start:start+length, ...]` along
//! the given axis; output shape's axis is `length`, everything else matches.
//! Slice is the inverse of Scatter in the sense that:
//!   slice(scatter(zero, src, axis, offset), axis, offset, src.len) == src

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_slice(data: Vec<f32>, shape: &[usize], axis: i32, start: usize, length: usize) -> Tensor {
    let mut g = GraphBuilder::new();
    let meta = TensorMeta::new(Dtype::F32, shape);
    let x = g.input("x", meta.clone());
    let y = g.slice(x, axis, start, length);
    g.output("y", y);
    let graph = g.build();
    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(meta, &data));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("y").unwrap().clone()
}

#[test]
fn slice_1d_basic() {
    let out = run_slice(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5], 0, 1, 3);
    assert_eq!(out.meta.shape, vec![3]);
    assert_eq!(out.as_f32_vec(), vec![20.0, 30.0, 40.0]);
}

#[test]
fn slice_1d_full_extent() {
    let out = run_slice(vec![1.0, 2.0, 3.0], &[3], 0, 0, 3);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn slice_1d_single_element() {
    let out = run_slice(vec![10.0, 20.0, 30.0], &[3], 0, 1, 1);
    assert_eq!(out.as_f32_vec(), vec![20.0]);
}

#[test]
fn slice_2d_rows() {
    // 4x3 matrix; slice rows 1..3.
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let out = run_slice(data, &[4, 3], 0, 1, 2);
    assert_eq!(out.meta.shape, vec![2, 3]);
    assert_eq!(out.as_f32_vec(), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn slice_2d_columns() {
    // 2x5; slice columns 1..4.
    let data = vec![
        0.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0,
    ];
    let out = run_slice(data, &[2, 5], 1, 1, 3);
    assert_eq!(out.meta.shape, vec![2, 3]);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 3.0,  6.0, 7.0, 8.0]);
}

#[test]
fn slice_3d_kv_cache_pattern() {
    // [n_heads=2, max_seq=4, d_head=3]; slice middle axis 0..2 → live region.
    let n_heads = 2; let max_seq = 4; let d_head = 3;
    let data: Vec<f32> = (0..n_heads * max_seq * d_head).map(|i| i as f32).collect();
    let out = run_slice(data, &[n_heads, max_seq, d_head], 1, 0, 2);
    assert_eq!(out.meta.shape, vec![2, 2, 3]);
    // Head 0: rows 0 and 1: [0,1,2, 3,4,5]
    // Head 1: rows 0 and 1: [12,13,14, 15,16,17]
    assert_eq!(out.as_f32_vec(), vec![
        0.0, 1.0, 2.0,   3.0, 4.0, 5.0,
        12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    ]);
}

#[test]
fn slice_negative_axis() {
    let out = run_slice(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], -1, 1, 3);
    assert_eq!(out.as_f32_vec(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn slice_is_deterministic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let a = run_slice(data.clone(), &[5], 0, 1, 3);
    let b = run_slice(data, &[5], 0, 1, 3);
    assert_eq!(a.storage.bytes, b.storage.bytes);
}

/// Slice is the inverse of Scatter into a zero buffer.
#[test]
fn slice_inverts_scatter() {
    // Build: dst = scatter(zeros[5], src=[10,20,30], axis=0, offset=1)
    // Then: live = slice(dst, axis=0, start=1, length=3)
    // Expect: live == src == [10,20,30]
    let mut g = GraphBuilder::new();
    let dst_meta = TensorMeta::new(Dtype::F32, &[5]);
    let src_meta = TensorMeta::new(Dtype::F32, &[3]);
    let dst = g.input("dst", dst_meta.clone());
    let src = g.input("src", src_meta.clone());
    let scattered = g.scatter(dst, src, 0, 1);
    let sliced = g.slice(scattered, 0, 1, 3);
    g.output("sliced", sliced);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("dst".into(), Tensor::from_f32(dst_meta, &[0.0; 5]));
    inputs.insert("src".into(), Tensor::from_f32(src_meta, &[10.0, 20.0, 30.0]));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let result = res.outputs.get("sliced").unwrap().as_f32_vec();
    assert_eq!(result, vec![10.0, 20.0, 30.0],
        "slice(scatter(zero, src, axis, off), axis, off, src.len) should equal src");
}
