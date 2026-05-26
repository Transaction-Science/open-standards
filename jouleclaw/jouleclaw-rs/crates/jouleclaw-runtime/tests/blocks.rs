//! Tests for Phase 1.3 block constructors.

use jouleclaw_core::blocks;
use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::op::{ActivationKind, NormKind, OpAttrs, OpKind};
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn det_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (s >> 40) as u32;
        (bits as f32) * (1.0 / (1u32 << 24) as f32) - 0.5
    }).collect()
}

fn bind(name: &str, shape: &[usize], seed: u64) -> (String, Tensor) {
    let n = shape.iter().product::<usize>();
    let data = det_random(n, seed);
    (name.into(), Tensor::from_f32(TensorMeta::new(Dtype::F32, shape), &data))
}

/// Single-head attention with proper QK^T computes softmax-normalized
/// probabilities. Each row of the attention matrix must sum to 1.
///
/// Strategy: build the attention block, but instead of completing through
/// the residual, branch out at the softmax to expose the probabilities,
/// and verify each row sums to 1 (within float precision).
#[test]
fn attention_softmax_rows_sum_to_one() {
    let seq = 8;
    let d = 16;

    let mut g = GraphBuilder::new();
    let x = g.input("x", TensorMeta::new(Dtype::F32, &[seq, d]));
    let w_norm = g.input("w_norm", TensorMeta::new(Dtype::F32, &[d]));
    let w_q = g.input("w_q", TensorMeta::new(Dtype::F32, &[d, d]));
    let w_k = g.input("w_k", TensorMeta::new(Dtype::F32, &[d, d]));

    let xn = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let q = g.matmul(xn, w_q);
    let k = g.matmul(xn, w_k);
    let scores = g.matmul_bt(q, k);
    let probs = g.softmax(scores, -1);
    g.output("probs", probs);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let mut inputs = HashMap::new();
    let (name, t) = bind("x", &[seq, d], 1); inputs.insert(name, t);
    let (name, t) = bind("w_norm", &[d], 2); inputs.insert(name, t);
    let (name, t) = bind("w_q", &[d, d], 3); inputs.insert(name, t);
    let (name, t) = bind("w_k", &[d, d], 4); inputs.insert(name, t);

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let probs = res.outputs.get("probs").unwrap();
    assert_eq!(probs.meta.shape, vec![seq, seq],
        "attention probs must be [seq, seq] for single-head");

    let p = probs.as_f32_vec();
    for row in 0..seq {
        let row_sum: f32 = (0..seq).map(|j| p[row * seq + j]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5,
            "row {} of attention probabilities should sum to 1.0, got {}", row, row_sum);
        for j in 0..seq {
            let v = p[row * seq + j];
            assert!(v >= 0.0 && v <= 1.0,
                "attention prob [{},{}] out of [0,1]: {}", row, j, v);
        }
    }
}

/// The transformer_block helper produces the expected number of nodes:
/// attention has 1 Norm + 6 MatMul (Q, K, V, QK^T, attn*V, Wo) + 1 Softmax + 1 Add = 9
/// FFN has 1 Norm + 3 MatMul (gate, up, down) + 1 Activation + 1 Mul + 1 Add = 7
/// Total ops = 16.
#[test]
fn transformer_block_node_count() {
    let mut g = GraphBuilder::new();
    let d = 8;
    let dff = 16;
    let x = g.input("x", TensorMeta::new(Dtype::F32, &[4, d]));
    let w_an = g.input("w_an", TensorMeta::new(Dtype::F32, &[d]));
    let w_q = g.input("w_q", TensorMeta::new(Dtype::F32, &[d, d]));
    let w_k = g.input("w_k", TensorMeta::new(Dtype::F32, &[d, d]));
    let w_v = g.input("w_v", TensorMeta::new(Dtype::F32, &[d, d]));
    let w_o = g.input("w_o", TensorMeta::new(Dtype::F32, &[d, d]));
    let w_fn = g.input("w_fn", TensorMeta::new(Dtype::F32, &[d]));
    let w_g = g.input("w_g", TensorMeta::new(Dtype::F32, &[d, dff]));
    let w_u = g.input("w_u", TensorMeta::new(Dtype::F32, &[d, dff]));
    let w_d = g.input("w_d", TensorMeta::new(Dtype::F32, &[dff, d]));

    let out = blocks::transformer_block(&mut g, x, w_an, w_q, w_k, w_v, w_o, w_fn, w_g, w_u, w_d);
    g.output("y", out);
    let graph = g.build();

    let mut op_count = 0;
    let mut by_kind: std::collections::HashMap<OpKind, usize> = Default::default();
    for node in &graph.nodes {
        if let jouleclaw_core::graph::NodeKind::Op { op, .. } = &node.kind {
            op_count += 1;
            *by_kind.entry(*op).or_insert(0) += 1;
        }
    }
    assert_eq!(op_count, 16, "transformer block should have 16 op nodes");
    assert_eq!(by_kind.get(&OpKind::MatMul), Some(&9),
        "9 MatMul: 6 in attention (Q/K/V/QK^T/attn*V/Wo), 3 in FFN (gate/up/down)");
    assert_eq!(by_kind.get(&OpKind::Norm), Some(&2));
    assert_eq!(by_kind.get(&OpKind::Add), Some(&2));
    assert_eq!(by_kind.get(&OpKind::Softmax), Some(&1));
    assert_eq!(by_kind.get(&OpKind::Activation), Some(&1));
    assert_eq!(by_kind.get(&OpKind::Mul), Some(&1));
}

/// matmul_bt produces correct shapes regardless of input shapes:
/// A[m,k] x B[n,k]^T -> [m,n].
#[test]
fn matmul_bt_shape_inference() {
    let mut g = GraphBuilder::new();
    let a = g.input("a", TensorMeta::new(Dtype::F32, &[7, 13]));
    let b = g.input("b", TensorMeta::new(Dtype::F32, &[5, 13]));
    let c = g.matmul_bt(a, b);
    g.output("c", c);
    let graph = g.build();

    let c_node = &graph.nodes[c.0 as usize];
    assert_eq!(c_node.output_meta[0].shape, vec![7, 5],
        "A[7,13] x B[5,13]^T should yield [7,5]");
    if let jouleclaw_core::graph::NodeKind::Op { attrs: OpAttrs::MatMul { transpose_a, transpose_b, .. }, .. } = &c_node.kind {
        assert_eq!(*transpose_a, false);
        assert_eq!(*transpose_b, true);
    } else {
        panic!("expected MatMul node");
    }
}

/// Verifying the activation kinds export properly and the FFN block doesn't
/// silently lose ActivationKind variants.
#[test]
fn ffn_block_uses_silu() {
    let mut g = GraphBuilder::new();
    let d = 4; let dff = 8;
    let x = g.input("x", TensorMeta::new(Dtype::F32, &[2, d]));
    let w_n = g.input("w_n", TensorMeta::new(Dtype::F32, &[d]));
    let w_g = g.input("w_g", TensorMeta::new(Dtype::F32, &[d, dff]));
    let w_u = g.input("w_u", TensorMeta::new(Dtype::F32, &[d, dff]));
    let w_d = g.input("w_d", TensorMeta::new(Dtype::F32, &[dff, d]));
    let out = blocks::ffn_gated_silu(&mut g, x, w_n, w_g, w_u, w_d);
    g.output("y", out);
    let graph = g.build();

    let mut found_silu = false;
    for node in &graph.nodes {
        if let jouleclaw_core::graph::NodeKind::Op {
            attrs: OpAttrs::Activation { kind: ActivationKind::SiLU }, .. } = &node.kind {
            found_silu = true;
            break;
        }
    }
    assert!(found_silu, "FFN block must use SiLU activation");
}
