//! Tests for causal softmax (autoregressive attention masking).

use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

fn run_softmax_causal(input: Vec<f32>, shape: &[usize]) -> Vec<f32> {
    let mut g = GraphBuilder::new();
    let in_meta = TensorMeta::new(Dtype::F32, shape);
    let x = g.input("x", in_meta.clone());
    let y = g.softmax_causal(x, -1);
    g.output("y", y);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(in_meta, &input));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    res.outputs.get("y").unwrap().as_f32_vec()
}

/// 2x2 attention scores. Row 0 sees only key 0; row 1 sees both.
/// After causal softmax:
/// - probs[0, 0] = 1.0, probs[0, 1] = 0.0
/// - probs[1, 0..2] forms a valid distribution that sums to 1.
#[test]
fn causal_softmax_2x2_basic() {
    let input = vec![
        1.0, 2.0,    // row 0: key 1 has higher score, but it's masked
        0.5, 1.5,    // row 1: key 1 has higher score, both visible
    ];
    let out = run_softmax_causal(input, &[2, 2]);

    // Row 0: only key 0 should have probability mass.
    assert!((out[0] - 1.0).abs() < 1e-6, "row 0 key 0 should be 1.0, got {}", out[0]);
    assert!((out[1]).abs() < 1e-6, "row 0 key 1 should be 0.0 (masked), got {}", out[1]);

    // Row 1: both keys visible, full softmax.
    let row1_sum = out[2] + out[3];
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 should sum to 1, got {}", row1_sum);
    // Higher logit gets higher prob.
    assert!(out[3] > out[2]);
}

/// 4x4 causal softmax: each row i should have non-zero values only in columns 0..=i.
#[test]
fn causal_softmax_upper_triangle_is_zero() {
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let out = run_softmax_causal(input, &[4, 4]);

    for row in 0..4 {
        for col in 0..4 {
            let v = out[row * 4 + col];
            if col > row {
                assert!(v.abs() < 1e-6,
                    "upper triangle [{},{}] should be 0, got {}", row, col, v);
            } else {
                assert!(v > 0.0,
                    "lower triangle [{},{}] should be > 0, got {}", row, col, v);
            }
        }
        // Each row's lower-triangle entries should sum to 1.
        let row_sum: f32 = (0..=row).map(|c| out[row * 4 + c]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5,
            "row {} should sum to 1.0, got {}", row, row_sum);
    }
}

/// Causal softmax over multi-head shape `[n_heads, seq, seq]` works per-head:
/// each head's rows have the same triangular structure independently.
#[test]
fn causal_softmax_multihead_shape() {
    let n_heads = 3;
    let seq = 5;
    let total = n_heads * seq * seq;
    // Distinct values per head so masking can be verified independently.
    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let out = run_softmax_causal(input, &[n_heads, seq, seq]);

    for h in 0..n_heads {
        for row in 0..seq {
            for col in 0..seq {
                let idx = h * (seq * seq) + row * seq + col;
                if col > row {
                    assert!(out[idx].abs() < 1e-6,
                        "head {}: upper triangle [{},{}] should be 0", h, row, col);
                }
            }
            // Each row sums to 1.
            let row_sum: f32 = (0..=row)
                .map(|c| out[h * seq * seq + row * seq + c]).sum();
            assert!((row_sum - 1.0).abs() < 1e-5,
                "head {} row {} should sum to 1.0, got {}", h, row, row_sum);
        }
    }
}

/// First-row case: only key 0 is visible, regardless of input values.
#[test]
fn causal_softmax_first_row_always_one_hot() {
    // Causal masking requires square last two dims. Use [3, 3] and check
    // that row 0 is one-hot (only key 0 visible).
    let input = vec![
        100.0, 200.0, 300.0,
          1.0,   1.0,   1.0,
          1.0,   1.0,   1.0,
    ];
    let out = run_softmax_causal(input, &[3, 3]);
    assert!((out[0] - 1.0).abs() < 1e-6, "row 0 col 0 should be 1.0");
    assert!(out[1].abs() < 1e-6);
    assert!(out[2].abs() < 1e-6);
}

/// Causal softmax is deterministic across runs.
#[test]
fn causal_softmax_is_deterministic() {
    let input: Vec<f32> = (0..36).map(|i| ((i as f32) - 18.0) * 0.1).collect();
    let out1 = run_softmax_causal(input.clone(), &[6, 6]);
    let out2 = run_softmax_causal(input, &[6, 6]);
    assert_eq!(out1, out2, "causal softmax must be byte-identical across runs");
}

/// Non-causal softmax (the default) doesn't mask — both halves contribute.
#[test]
fn non_causal_softmax_includes_full_row() {
    let input = vec![1.0, 2.0,  0.5, 1.5];
    let mut g = GraphBuilder::new();
    let in_meta = TensorMeta::new(Dtype::F32, &[2, 2]);
    let x = g.input("x", in_meta.clone());
    let y = g.softmax(x, -1);  // non-causal default
    g.output("y", y);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(in_meta, &input));
    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let out = res.outputs.get("y").unwrap().as_f32_vec();

    // Both rows: nothing should be exactly zero.
    for v in &out { assert!(*v > 0.0, "non-causal softmax shouldn't produce zeros"); }
    // Both rows sum to 1.
    let r0 = out[0] + out[1];
    let r1 = out[2] + out[3];
    assert!((r0 - 1.0).abs() < 1e-5);
    assert!((r1 - 1.0).abs() < 1e-5);
}

/// Multi-head Llama with causal masking: each token's attention only looks
/// at past + present tokens, never future. Verifying via the existing
/// end-to-end pipeline that determinism still holds.
#[test]
fn causal_attention_in_multi_head_is_deterministic() {
    use jouleclaw_core::blocks;

    let seq = 5usize;
    let n_heads = 2usize;
    let d_head = 4usize;
    let d_model = n_heads * d_head;

    let mut g = GraphBuilder::new();
    let x = g.input("x", TensorMeta::new(Dtype::F32, &[seq, d_model]));
    let w_norm = g.input("w_norm", TensorMeta::new(Dtype::F32, &[d_model]));
    let w_q = g.input("w_q", TensorMeta::new(Dtype::F32, &[d_model, d_model]));
    let w_k = g.input("w_k", TensorMeta::new(Dtype::F32, &[d_model, d_model]));
    let w_v = g.input("w_v", TensorMeta::new(Dtype::F32, &[d_model, d_model]));
    let w_o = g.input("w_o", TensorMeta::new(Dtype::F32, &[d_model, d_model]));

    // The with-rope variant uses causal softmax internally.
    let out = blocks::multi_head_attention_with_rope(
        &mut g, x, w_norm, w_q, w_k, w_v, w_o,
        seq, n_heads, d_head, 10000.0,
    );
    g.output("y", out);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    fn det_random(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (s >> 40) as u32;
            ((bits as f32) * (1.0 / (1u32 << 24) as f32) - 0.5) * 0.5
        }).collect()
    }
    fn bind(name: &str, shape: &[usize], seed: u64) -> (String, Tensor) {
        let n = shape.iter().product::<usize>();
        (name.into(),
         Tensor::from_f32(TensorMeta::new(Dtype::F32, shape), &det_random(n, seed)))
    }

    let mut inputs = HashMap::new();
    let (k, v) = bind("x", &[seq, d_model], 1); inputs.insert(k, v);
    let (k, v) = bind("w_norm", &[d_model], 2); inputs.insert(k, v);
    let (k, v) = bind("w_q", &[d_model, d_model], 3); inputs.insert(k, v);
    let (k, v) = bind("w_k", &[d_model, d_model], 4); inputs.insert(k, v);
    let (k, v) = bind("w_v", &[d_model, d_model], 5); inputs.insert(k, v);
    let (k, v) = bind("w_o", &[d_model, d_model], 6); inputs.insert(k, v);

    let r1 = execute(&compiled, inputs.clone(), ExecutionOptions::default()).unwrap();
    let r2 = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();

    let y = r1.outputs.get("y").unwrap();
    assert_eq!(y.meta.shape, vec![seq, d_model]);
    assert_eq!(
        r1.outputs.get("y").unwrap().storage.bytes,
        r2.outputs.get("y").unwrap().storage.bytes,
        "multi-head with RoPE + causal mask must be deterministic"
    );
}
