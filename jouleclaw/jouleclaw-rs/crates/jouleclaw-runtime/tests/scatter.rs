//! Tests for the Scatter primitive.
//!
//! Scatter semantics: `output[..., offset:offset+src_len, ...] = src`
//! along the given axis; everything else copied from `dst` unchanged.
//! Output has dst's shape.

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_scatter(
    dst_data: Vec<f32>, dst_shape: &[usize],
    src_data: Vec<f32>, src_shape: &[usize],
    axis: i32, offset: usize,
) -> Tensor {
    let mut g = GraphBuilder::new();
    let dst_meta = TensorMeta::new(Dtype::F32, dst_shape);
    let src_meta = TensorMeta::new(Dtype::F32, src_shape);
    let dst = g.input("dst", dst_meta.clone());
    let src = g.input("src", src_meta.clone());
    let out = g.scatter(dst, src, axis, offset);
    g.output("out", out);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("dst".into(), Tensor::from_f32(dst_meta, &dst_data));
    inputs.insert("src".into(), Tensor::from_f32(src_meta, &src_data));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("out").unwrap().clone()
}

/// 1D scatter: write src at offset, leave the rest of dst alone.
#[test]
fn scatter_1d_writes_at_offset() {
    let dst = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let src = vec![1.0, 2.0, 3.0];
    let out = run_scatter(dst, &[5], src, &[3], 0, 1);
    assert_eq!(out.meta.shape, vec![5]);
    // Positions 1..4 become src; positions 0 and 4 are unchanged from dst.
    assert_eq!(out.as_f32_vec(), vec![0.0, 1.0, 2.0, 3.0, 0.0]);
}

/// Scatter preserves dst at positions outside the scatter region.
#[test]
fn scatter_preserves_unwritten_positions() {
    let dst = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let src = vec![99.0, 99.0];
    let out = run_scatter(dst, &[5], src, &[2], 0, 2);
    assert_eq!(out.as_f32_vec(), vec![10.0, 20.0, 99.0, 99.0, 50.0]);
}

/// Scatter at offset 0 with full-length src equals just copying src.
#[test]
fn scatter_full_overlap_is_src_copy() {
    let dst = vec![10.0, 20.0, 30.0];
    let src = vec![1.0, 2.0, 3.0];
    let out = run_scatter(dst, &[3], src, &[3], 0, 0);
    assert_eq!(out.as_f32_vec(), vec![1.0, 2.0, 3.0]);
}

/// 2D scatter along axis 0 (write rows).
#[test]
fn scatter_2d_axis_0() {
    // dst: 4x3, src: 2x3, offset 1 → rows 1 and 2 get overwritten.
    let dst = vec![
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let src = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ];
    let out = run_scatter(dst, &[4, 3], src, &[2, 3], 0, 1);
    assert_eq!(out.meta.shape, vec![4, 3]);
    let expected = vec![
        0.0, 0.0, 0.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        0.0, 0.0, 0.0,
    ];
    assert_eq!(out.as_f32_vec(), expected);
}

/// 2D scatter along axis 1 (write columns within each row).
#[test]
fn scatter_2d_axis_1() {
    // dst: 2x5, src: 2x2, offset 2 → columns 2..4 get overwritten in each row.
    let dst = vec![
        0.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0,
    ];
    let src = vec![
        100.0, 200.0,
        300.0, 400.0,
    ];
    let out = run_scatter(dst, &[2, 5], src, &[2, 2], 1, 2);
    let expected = vec![
        0.0, 1.0, 100.0, 200.0, 4.0,
        5.0, 6.0, 300.0, 400.0, 9.0,
    ];
    assert_eq!(out.as_f32_vec(), expected);
}

/// 3D scatter on middle axis — the KV cache pattern. Write a new K slice
/// (shape `[n_heads, new_seq, d_head]`) into a preallocated buffer (shape
/// `[n_heads, max_seq, d_head]`) at position `pos`.
#[test]
fn scatter_3d_kv_cache_pattern() {
    let n_heads = 2;
    let max_seq = 6;
    let d_head = 3;
    let new_seq = 2;
    let pos = 3;

    // Preallocated buffer initialized to zeros (in a real cache, only the
    // positions before `pos` are meaningful from previous steps).
    let dst: Vec<f32> = vec![0.0; n_heads * max_seq * d_head];
    // New slice: small distinctive values.
    let src: Vec<f32> = (0..n_heads * new_seq * d_head).map(|i| i as f32 + 1.0).collect();

    let out = run_scatter(dst, &[n_heads, max_seq, d_head],
                          src.clone(), &[n_heads, new_seq, d_head], 1, pos);
    assert_eq!(out.meta.shape, vec![n_heads, max_seq, d_head]);

    let result = out.as_f32_vec();
    // Head 0, positions pos..pos+new_seq should match src head 0.
    for s in 0..new_seq {
        for d in 0..d_head {
            let out_idx = 0 * (max_seq * d_head) + (pos + s) * d_head + d;
            let src_idx = 0 * (new_seq * d_head) + s * d_head + d;
            assert_eq!(result[out_idx], src[src_idx],
                "head 0 pos {} d {}: out {} != src {}",
                pos + s, d, result[out_idx], src[src_idx]);
        }
    }
    // Head 0, positions outside [pos, pos+new_seq) should remain zero.
    for s in 0..max_seq {
        if (pos..pos + new_seq).contains(&s) { continue; }
        for d in 0..d_head {
            let out_idx = s * d_head + d;
            assert_eq!(result[out_idx], 0.0,
                "head 0 pos {} should remain zero, got {}", s, result[out_idx]);
        }
    }
    // Head 1, same check.
    let head1_off = max_seq * d_head;
    for s in 0..new_seq {
        for d in 0..d_head {
            let out_idx = head1_off + (pos + s) * d_head + d;
            let src_idx = new_seq * d_head + s * d_head + d;
            assert_eq!(result[out_idx], src[src_idx]);
        }
    }
}

/// Negative axis indexing.
#[test]
fn scatter_negative_axis() {
    let dst = vec![0.0, 0.0, 0.0, 0.0];
    let src = vec![7.0];
    let out = run_scatter(dst, &[4], src, &[1], -1, 2);
    assert_eq!(out.as_f32_vec(), vec![0.0, 0.0, 7.0, 0.0]);
}

/// Scatter is deterministic.
#[test]
fn scatter_is_deterministic() {
    let dst = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let src = vec![9.0, 9.0];
    let out1 = run_scatter(dst.clone(), &[5], src.clone(), &[2], 0, 1);
    let out2 = run_scatter(dst, &[5], src, &[2], 0, 1);
    assert_eq!(out1.storage.bytes, out2.storage.bytes);
}

/// Sequential scatter writes simulate a growing KV cache: write at pos=0,
/// then pos=1, then pos=2. After all three writes, positions 0..3 should
/// reflect the cumulative writes.
#[test]
fn scatter_sequential_writes_simulate_kv_growth() {
    let max_seq = 5;
    let d_head = 2;

    // Step 1: write [1, 2] at pos 0.
    let dst: Vec<f32> = vec![0.0; max_seq * d_head];
    let s1: Vec<f32> = vec![1.0, 2.0];
    let after_1 = run_scatter(dst, &[max_seq, d_head], s1, &[1, d_head], 0, 0);
    assert_eq!(after_1.as_f32_vec(),
        vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Step 2: write [3, 4] at pos 1.
    let s2: Vec<f32> = vec![3.0, 4.0];
    let after_2 = run_scatter(
        after_1.as_f32_vec(), &[max_seq, d_head],
        s2, &[1, d_head], 0, 1);
    assert_eq!(after_2.as_f32_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Step 3: write [5, 6] at pos 2.
    let s3: Vec<f32> = vec![5.0, 6.0];
    let after_3 = run_scatter(
        after_2.as_f32_vec(), &[max_seq, d_head],
        s3, &[1, d_head], 0, 2);
    assert_eq!(after_3.as_f32_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]);
}
