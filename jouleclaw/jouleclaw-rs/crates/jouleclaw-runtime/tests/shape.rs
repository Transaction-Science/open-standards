//! Tests for Reshape and Transpose primitives.

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_single_op_graph(
    input_meta: TensorMeta,
    input_data: Vec<f32>,
    build: impl FnOnce(&mut GraphBuilder, jouleclaw_core::graph::NodeId) -> jouleclaw_core::graph::NodeId,
) -> Tensor {
    let mut g = GraphBuilder::new();
    let x = g.input("x", input_meta.clone());
    let y = build(&mut g, x);
    g.output("y", y);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(input_meta, &input_data));

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("y").unwrap().clone()
}

/// Reshape preserves all element values; only the shape metadata changes.
#[test]
fn reshape_preserves_element_values() {
    let input_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let in_meta = TensorMeta::new(Dtype::F32, &[2, 3, 4]);

    let out = run_single_op_graph(
        in_meta,
        input_data.clone(),
        |g, x| g.reshape(x, &[6, 4]),
    );

    assert_eq!(out.meta.shape, vec![6, 4]);
    assert_eq!(out.as_f32_vec(), input_data,
        "reshape must preserve element values");
}

/// Reshape to a different rank with same total element count.
#[test]
fn reshape_changes_rank() {
    let input_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let in_meta = TensorMeta::new(Dtype::F32, &[12]);

    // [12] -> [3, 4]
    let out = run_single_op_graph(
        in_meta.clone(),
        input_data.clone(),
        |g, x| g.reshape(x, &[3, 4]),
    );
    assert_eq!(out.meta.shape, vec![3, 4]);
    assert_eq!(out.as_f32_vec(), input_data);

    // [12] -> [2, 2, 3]
    let out = run_single_op_graph(
        in_meta,
        input_data.clone(),
        |g, x| g.reshape(x, &[2, 2, 3]),
    );
    assert_eq!(out.meta.shape, vec![2, 2, 3]);
    assert_eq!(out.as_f32_vec(), input_data);
}

/// Transpose 2D: swap rows and columns. Hand-verified.
///
/// Input [2,3]:
///   1 2 3
///   4 5 6
/// Permutation [1, 0] -> output [3, 2]:
///   1 4
///   2 5
///   3 6
#[test]
fn transpose_2d_hand_verified() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let in_meta = TensorMeta::new(Dtype::F32, &[2, 3]);

    let out = run_single_op_graph(
        in_meta,
        input_data,
        |g, x| g.transpose(x, &[1, 0]),
    );

    assert_eq!(out.meta.shape, vec![3, 2]);
    let expected = vec![1.0, 4.0,  2.0, 5.0,  3.0, 6.0];
    assert_eq!(out.as_f32_vec(), expected);
}

/// Transpose 3D: permute [0,1,2] -> [1,0,2].
///
/// Input [2,2,3]:
///   batch 0: [[1,2,3],[4,5,6]]
///   batch 1: [[7,8,9],[10,11,12]]
/// Permute axes [1,0,2] -> output shape [2,2,3]:
///   for output[b][s][d] = input[s][b][d]
///   row 0: [[1,2,3],[7,8,9]]
///   row 1: [[4,5,6],[10,11,12]]
#[test]
fn transpose_3d_axes_swap() {
    let input_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let in_meta = TensorMeta::new(Dtype::F32, &[2, 2, 3]);

    let out = run_single_op_graph(
        in_meta,
        input_data,
        |g, x| g.transpose(x, &[1, 0, 2]),
    );

    assert_eq!(out.meta.shape, vec![2, 2, 3]);
    let expected = vec![
        1.0, 2.0, 3.0,    // input[0][0]
        7.0, 8.0, 9.0,    // input[1][0]
        4.0, 5.0, 6.0,    // input[0][1]
        10.0, 11.0, 12.0, // input[1][1]
    ];
    assert_eq!(out.as_f32_vec(), expected);
}

/// Transpose round-trip: applying a permutation followed by its inverse
/// returns the original tensor.
#[test]
fn transpose_round_trip_returns_original() {
    let input_data: Vec<f32> = (0..60).map(|i| i as f32 * 0.1).collect();
    let in_meta = TensorMeta::new(Dtype::F32, &[3, 4, 5]);

    let mut g = GraphBuilder::new();
    let x = g.input("x", in_meta.clone());
    // Apply [2,0,1]: output shape [5,3,4]
    let y = g.transpose(x, &[2, 0, 1]);
    // Inverse permutation: if forward maps i -> permutation[i],
    // inverse[permutation[i]] = i. For [2,0,1]: inverse[2]=0, inverse[0]=1, inverse[1]=2 -> [1,2,0]
    let z = g.transpose(y, &[1, 2, 0]);
    g.output("z", z);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(in_meta, &input_data));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let z = res.outputs.get("z").unwrap();

    assert_eq!(z.meta.shape, vec![3, 4, 5]);
    assert_eq!(z.as_f32_vec(), input_data,
        "transpose followed by inverse must recover original");
}

/// Reshape + transpose composition: simulate the multi-head split that
/// real attention requires. Given x: [seq=4, d_model=6] and n_heads=2,
/// d_head=3, produce [n_heads=2, seq=4, d_head=3].
#[test]
fn reshape_then_transpose_for_multihead_split() {
    let seq = 4usize;
    let d_model = 6usize;
    let n_heads = 2usize;
    let d_head = 3usize;
    assert_eq!(d_model, n_heads * d_head);

    // Create x with values 0..24 so we can hand-verify.
    let input_data: Vec<f32> = (0..(seq * d_model)).map(|i| i as f32).collect();
    let in_meta = TensorMeta::new(Dtype::F32, &[seq, d_model]);

    let mut g = GraphBuilder::new();
    let x = g.input("x", in_meta.clone());
    // [4, 6] -> [4, 2, 3]
    let r = g.reshape(x, &[seq, n_heads, d_head]);
    // [4, 2, 3] -> [2, 4, 3]  (axes [1, 0, 2])
    let t = g.transpose(r, &[1, 0, 2]);
    g.output("y", t);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(in_meta, &input_data));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let y = res.outputs.get("y").unwrap();

    assert_eq!(y.meta.shape, vec![n_heads, seq, d_head]);
    let y_vec = y.as_f32_vec();

    // Verify: for token t and head h, the d_head values at output[h, t, :]
    // should equal input[t, h*d_head .. (h+1)*d_head].
    for t in 0..seq {
        for h in 0..n_heads {
            for k in 0..d_head {
                let out_idx = h * (seq * d_head) + t * d_head + k;
                let in_idx = t * d_model + h * d_head + k;
                assert_eq!(y_vec[out_idx], input_data[in_idx],
                    "multi-head split mismatch at h={}, t={}, k={}", h, t, k);
            }
        }
    }
}

/// Determinism: reshape and transpose must be deterministic-tagged kernels.
#[test]
fn shape_ops_are_deterministic() {
    let runtime = Runtime::boot();
    let mut found_reshape = false;
    let mut found_transpose = false;
    for k in runtime.kernels.iter() {
        match k.op_kind() {
            jouleclaw_core::op::OpKind::Reshape => {
                assert_eq!(k.determinism(), jouleclaw_core::determinism::DeterminismClass::Deterministic);
                found_reshape = true;
            }
            jouleclaw_core::op::OpKind::Transpose => {
                assert_eq!(k.determinism(), jouleclaw_core::determinism::DeterminismClass::Deterministic);
                found_transpose = true;
            }
            _ => {}
        }
    }
    assert!(found_reshape, "Reshape kernel must be registered");
    assert!(found_transpose, "Transpose kernel must be registered");
}
