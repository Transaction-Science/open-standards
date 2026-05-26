//! Tests for RoPE (rotary position embedding) and scaled MatMul.

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_rope(
    input: Vec<f32>,
    shape: &[usize],
    base: f32,
    pos_offset: u32,
) -> Vec<f32> {
    let mut g = GraphBuilder::new();
    let in_meta = TensorMeta::new(Dtype::F32, shape);
    let x = g.input("x", in_meta.clone());
    let y = g.rope(x, base, pos_offset);
    g.output("y", y);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(in_meta, &input));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("y").unwrap().as_f32_vec()
}

/// At position 0 the angle is 0, so cos=1, sin=0, rotation = identity.
#[test]
fn rope_at_position_zero_is_identity() {
    let input = vec![1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0];  // [seq=2, d=4]
    // But we want position 0 only — seq=1.
    let input = vec![1.0, 2.0, 3.0, 4.0];  // [seq=1, d=4]
    let out = run_rope(input.clone(), &[1, 4], 10000.0, 0);
    // Position 0 always gives identity for any base.
    assert_eq!(out, input);
}

/// Hand-computed test: d=2, base=10000, pos=1.
/// theta_0 = 10000^0 = 1, m = 1*1 = 1 radian.
/// Input [1, 0]: out = [cos(1), sin(1)] ≈ [0.5403, 0.8415]
/// Input [0, 1]: out = [-sin(1), cos(1)] ≈ [-0.8415, 0.5403]
#[test]
fn rope_d2_position_1_hand_computed() {
    // Two-token sequence: token at pos 0 is identity, token at pos 1 rotates.
    let input = vec![1.0, 0.0,  0.0, 1.0];  // [seq=2, d=2]
    let out = run_rope(input, &[2, 2], 10000.0, 0);

    // Token 0 at pos 0: identity.
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);

    // Token 1 at pos 1: rotation by 1 radian.
    let cos1 = 1.0_f32.cos();
    let sin1 = 1.0_f32.sin();
    assert!((out[2] - 0.0 * cos1 + 1.0 * sin1).abs() < 1e-6,
        "out[2] = 0*cos(1) - 1*sin(1) = -sin(1), got {}", out[2]);
    assert!((out[3] - (0.0 * sin1 + 1.0 * cos1)).abs() < 1e-6,
        "out[3] = 0*sin(1) + 1*cos(1) = cos(1), got {}", out[3]);
}

/// Position offset shifts the effective sequence position.
#[test]
fn rope_position_offset_works() {
    let input = vec![1.0, 0.0];  // [seq=1, d=2]

    // No offset: pos=0, identity.
    let out0 = run_rope(input.clone(), &[1, 2], 10000.0, 0);
    assert!((out0[0] - 1.0).abs() < 1e-6);
    assert!((out0[1] - 0.0).abs() < 1e-6);

    // Offset 1: same input behaves as if at pos 1, rotation by 1 radian.
    let out1 = run_rope(input, &[1, 2], 10000.0, 1);
    assert!((out1[0] - 1.0_f32.cos()).abs() < 1e-6,
        "expected cos(1), got {}", out1[0]);
    assert!((out1[1] - 1.0_f32.sin()).abs() < 1e-6,
        "expected sin(1), got {}", out1[1]);
}

/// Multi-head shape works: [n_heads, seq, d_head] with d_head even.
#[test]
fn rope_handles_multihead_shape() {
    let n_heads = 2;
    let seq = 3;
    let d_head = 4;
    let total = n_heads * seq * d_head;
    let input: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
    let out = run_rope(input.clone(), &[n_heads, seq, d_head], 10000.0, 0);

    // Check: position-0 row of each head should be identity.
    // Head 0, token 0 starts at offset 0.
    assert!((out[0] - input[0]).abs() < 1e-6);
    assert!((out[1] - input[1]).abs() < 1e-6);
    assert!((out[2] - input[2]).abs() < 1e-6);
    assert!((out[3] - input[3]).abs() < 1e-6);

    // Head 1, token 0 starts at offset seq * d_head = 12.
    let h1t0 = n_heads * d_head;  // wait, that's 8. Recompute: head 1 starts at seq*d_head = 12.
    let h1t0 = seq * d_head;
    assert!((out[h1t0 + 0] - input[h1t0 + 0]).abs() < 1e-6);
    assert!((out[h1t0 + 1] - input[h1t0 + 1]).abs() < 1e-6);
    assert!((out[h1t0 + 2] - input[h1t0 + 2]).abs() < 1e-6);
    assert!((out[h1t0 + 3] - input[h1t0 + 3]).abs() < 1e-6);

    // Output shape is preserved.
    assert_eq!(out.len(), total);
}

/// Determinism: running RoPE twice must produce byte-identical results.
#[test]
fn rope_is_deterministic() {
    let input = vec![0.5, -0.3, 0.7, 0.1];
    let out1 = run_rope(input.clone(), &[2, 2], 10000.0, 0);
    let out2 = run_rope(input, &[2, 2], 10000.0, 0);
    assert_eq!(out1, out2, "rope must be byte-identical across runs");
}

/// Each pair rotates by a different angle: theta_i = base^(-2i/d).
/// For d=4, base=100, theta_0 = 1, theta_1 = 100^(-1/2) = 0.1.
/// At pos=1: pair (0,1) rotates by 1 rad, pair (2,3) rotates by 0.1 rad.
#[test]
fn rope_different_pairs_use_different_angles() {
    // Two tokens, d=4. Token 0 at pos 0 is identity; token 1 at pos 1
    // rotates each pair by its own theta.
    let input = vec![
        1.0, 0.0, 1.0, 0.0,  // token 0
        1.0, 0.0, 1.0, 0.0,  // token 1
    ];
    let out = run_rope(input, &[2, 4], 100.0, 0);

    // Token 0: identity.
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);
    assert!((out[2] - 1.0).abs() < 1e-6);
    assert!((out[3] - 0.0).abs() < 1e-6);

    // Token 1, pair (0,1): rotation by 1 rad.
    // x=[1,0] -> [cos(1), sin(1)]
    let cos1 = 1.0_f32.cos();
    let sin1 = 1.0_f32.sin();
    assert!((out[4] - cos1).abs() < 1e-6);
    assert!((out[5] - sin1).abs() < 1e-6);

    // Token 1, pair (2,3): theta = 100^(-2/4) = 100^(-0.5) = 0.1.
    // angle = 1 * 0.1 = 0.1 rad.
    // x=[1,0] -> [cos(0.1), sin(0.1)]
    let cos01 = 0.1_f32.cos();
    let sin01 = 0.1_f32.sin();
    assert!((out[6] - cos01).abs() < 1e-6,
        "expected cos(0.1) = {}, got {}", cos01, out[6]);
    assert!((out[7] - sin01).abs() < 1e-6,
        "expected sin(0.1) = {}, got {}", sin01, out[7]);
}

// =====================================================================
// Attention scaling (alpha in MatMul)
// =====================================================================

/// matmul_bt_scaled multiplies the output by alpha.
#[test]
fn matmul_bt_scaled_applies_alpha() {
    let mut g = GraphBuilder::new();
    let a = g.input("a", TensorMeta::new(Dtype::F32, &[2, 4]));
    let b = g.input("b", TensorMeta::new(Dtype::F32, &[3, 4]));
    let scaled = g.matmul_bt_scaled(a, b, 0.5);
    let unscaled = g.matmul_bt(a, b);
    g.output("scaled", scaled);
    g.output("unscaled", unscaled);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let a_data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
    let b_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();

    let mut inputs = HashMap::new();
    inputs.insert("a".into(),
        Tensor::from_f32(TensorMeta::new(Dtype::F32, &[2, 4]), &a_data));
    inputs.insert("b".into(),
        Tensor::from_f32(TensorMeta::new(Dtype::F32, &[3, 4]), &b_data));

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let scaled = res.outputs.get("scaled").unwrap().as_f32_vec();
    let unscaled = res.outputs.get("unscaled").unwrap().as_f32_vec();

    assert_eq!(scaled.len(), unscaled.len());
    for (s, u) in scaled.iter().zip(unscaled.iter()) {
        assert!((s - u * 0.5).abs() < 1e-5,
            "scaled output should be 0.5 * unscaled; got {} vs {}", s, u);
    }
}

/// Alpha = 1.0 is the default and matches matmul (without scaling).
#[test]
fn alpha_one_matches_default_matmul() {
    let mut g = GraphBuilder::new();
    let a = g.input("a", TensorMeta::new(Dtype::F32, &[3, 5]));
    let b = g.input("b", TensorMeta::new(Dtype::F32, &[5, 4]));
    let default = g.matmul(a, b);
    let alpha_one = g.matmul_with_alpha(a, b, false, false, 1.0);
    g.output("default", default);
    g.output("alpha_one", alpha_one);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let a_data: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.05).collect();

    let mut inputs = HashMap::new();
    inputs.insert("a".into(),
        Tensor::from_f32(TensorMeta::new(Dtype::F32, &[3, 5]), &a_data));
    inputs.insert("b".into(),
        Tensor::from_f32(TensorMeta::new(Dtype::F32, &[5, 4]), &b_data));

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let default = res.outputs.get("default").unwrap().as_f32_vec();
    let alpha_one = res.outputs.get("alpha_one").unwrap().as_f32_vec();

    assert_eq!(default, alpha_one,
        "alpha=1.0 must produce bit-identical output to default matmul");
}
