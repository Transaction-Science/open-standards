//! Tests for batched MatMul and multi-head attention.

use jouleclaw_core::blocks;
use jouleclaw_core::graph::GraphBuilder;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

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

/// Batched 3D × 3D matmul: A[B,M,K] × B[B,K,N] = C[B,M,N], where each batch
/// is computed independently. Hand-verified by also computing each batch
/// as a separate 2D matmul and comparing.
#[test]
fn batched_matmul_matches_per_batch_2d() {
    let batch = 3usize;
    let m = 4usize;
    let k = 5usize;
    let n = 6usize;

    let a_data = det_random(batch * m * k, 100);
    let b_data = det_random(batch * k * n, 200);

    // Run batched.
    let batched_out = {
        let mut g = GraphBuilder::new();
        let a = g.input("a", TensorMeta::new(Dtype::F32, &[batch, m, k]));
        let b = g.input("b", TensorMeta::new(Dtype::F32, &[batch, k, n]));
        let c = g.matmul(a, b);
        g.output("c", c);
        let graph = g.build();

        let runtime = Runtime::boot();
        let compiled = compile(graph, &runtime.kernels).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("a".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[batch, m, k]), &a_data));
        inputs.insert("b".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[batch, k, n]), &b_data));
        execute(&compiled, inputs, ExecutionOptions::default()).unwrap()
    };
    let bytes_batched = batched_out.outputs.get("c").unwrap().as_f32_vec();
    assert_eq!(batched_out.outputs.get("c").unwrap().meta.shape,
        vec![batch, m, n]);

    // For each batch, run a 2D matmul and compare bit-for-bit.
    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let a_slice = &a_data[a_off..a_off + m * k];
        let b_slice = &b_data[b_off..b_off + k * n];

        let mut g = GraphBuilder::new();
        let a = g.input("a", TensorMeta::new(Dtype::F32, &[m, k]));
        let b = g.input("b", TensorMeta::new(Dtype::F32, &[k, n]));
        let c = g.matmul(a, b);
        g.output("c", c);
        let graph = g.build();

        let runtime = Runtime::boot();
        let compiled = compile(graph, &runtime.kernels).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("a".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[m, k]), a_slice));
        inputs.insert("b".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[k, n]), b_slice));
        let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
        let single = res.outputs.get("c").unwrap().as_f32_vec();

        let c_off = bi * m * n;
        let batched_slice = &bytes_batched[c_off..c_off + m * n];
        assert_eq!(single.as_slice(), batched_slice,
            "batch {} should equal a standalone 2D matmul", bi);
    }
}

/// Batched matmul_bt: A[B,M,K] × B[B,N,K]^T = C[B,M,N]. Hand-verified
/// against per-batch 2D matmul_bt.
#[test]
fn batched_matmul_bt_matches_per_batch() {
    let batch = 2usize;
    let m = 3usize;
    let k = 4usize;
    let n = 3usize;

    let a_data = det_random(batch * m * k, 11);
    let b_data = det_random(batch * n * k, 13);

    // Batched.
    let batched_out = {
        let mut g = GraphBuilder::new();
        let a = g.input("a", TensorMeta::new(Dtype::F32, &[batch, m, k]));
        let b = g.input("b", TensorMeta::new(Dtype::F32, &[batch, n, k]));
        let c = g.matmul_bt(a, b);
        g.output("c", c);
        let graph = g.build();
        let runtime = Runtime::boot();
        let compiled = compile(graph, &runtime.kernels).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("a".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[batch, m, k]), &a_data));
        inputs.insert("b".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[batch, n, k]), &b_data));
        execute(&compiled, inputs, ExecutionOptions::default()).unwrap()
    };
    let batched = batched_out.outputs.get("c").unwrap().as_f32_vec();
    assert_eq!(batched_out.outputs.get("c").unwrap().meta.shape,
        vec![batch, m, n]);

    for bi in 0..batch {
        let a_slice = &a_data[bi * m * k..(bi + 1) * m * k];
        let b_slice = &b_data[bi * n * k..(bi + 1) * n * k];

        let mut g = GraphBuilder::new();
        let a = g.input("a", TensorMeta::new(Dtype::F32, &[m, k]));
        let b = g.input("b", TensorMeta::new(Dtype::F32, &[n, k]));
        let c = g.matmul_bt(a, b);
        g.output("c", c);
        let graph = g.build();
        let runtime = Runtime::boot();
        let compiled = compile(graph, &runtime.kernels).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("a".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[m, k]), a_slice));
        inputs.insert("b".into(),
            Tensor::from_f32(TensorMeta::new(Dtype::F32, &[n, k]), b_slice));
        let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
        let single = res.outputs.get("c").unwrap().as_f32_vec();

        let bs = &batched[bi * m * n..(bi + 1) * m * n];
        assert_eq!(single.as_slice(), bs);
    }
}

/// Multi-head attention produces softmax-normalized probabilities at the
/// per-head, per-token level. Output shape and softmax-validity are the
/// load-bearing checks.
#[test]
fn multi_head_attention_softmax_per_head_sums_to_one() {
    let seq = 6usize;
    let n_heads = 4usize;
    let d_head = 8usize;
    let d_model = n_heads * d_head;

    // Build a graph that exposes the attention probabilities, not just the
    // post-residual output. To do this we call the multi_head_attention
    // helper but also separately build the same shape pipeline up to
    // softmax so we can output the probs.
    //
    // Easier path: build the multi-head pipeline by hand, expose probs.
    let mut g = GraphBuilder::new();
    let x = g.input("x", TensorMeta::new(Dtype::F32, &[seq, d_model]));
    let w_norm = g.input("w_norm", TensorMeta::new(Dtype::F32, &[d_model]));
    let w_q = g.input("w_q", TensorMeta::new(Dtype::F32, &[d_model, d_model]));
    let w_k = g.input("w_k", TensorMeta::new(Dtype::F32, &[d_model, d_model]));

    let xn = g.norm(x, w_norm, jouleclaw_core::op::NormKind::Rms, 1e-6);
    let q = g.matmul_bt(xn, w_q);  // [seq, d_model]
    let k = g.matmul_bt(xn, w_k);

    let q_h = {
        let r = g.reshape(q, &[seq, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])  // [n_heads, seq, d_head]
    };
    let k_h = {
        let r = g.reshape(k, &[seq, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };

    let scores = g.matmul_bt(q_h, k_h);   // [n_heads, seq, seq]
    let probs = g.softmax(scores, -1);
    g.output("probs", probs);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let mut inputs = HashMap::new();
    let (name, t) = bind("x", &[seq, d_model], 1); inputs.insert(name, t);
    let (name, t) = bind("w_norm", &[d_model], 2); inputs.insert(name, t);
    let (name, t) = bind("w_q", &[d_model, d_model], 3); inputs.insert(name, t);
    let (name, t) = bind("w_k", &[d_model, d_model], 4); inputs.insert(name, t);

    let res = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();
    let probs = res.outputs.get("probs").unwrap();

    assert_eq!(probs.meta.shape, vec![n_heads, seq, seq],
        "multi-head probs must be [n_heads, seq, seq]");

    let p = probs.as_f32_vec();
    for h in 0..n_heads {
        for t in 0..seq {
            let row_sum: f32 = (0..seq)
                .map(|s| p[h * (seq * seq) + t * seq + s])
                .sum();
            assert!((row_sum - 1.0).abs() < 1e-5,
                "head {} token {}: probs should sum to 1.0, got {}",
                h, t, row_sum);
        }
    }
}

/// The full multi_head_attention block runs end-to-end with residual.
#[test]
fn multi_head_attention_block_runs() {
    let seq = 4usize;
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

    let out = blocks::multi_head_attention(
        &mut g, x, w_norm, w_q, w_k, w_v, w_o,
        seq, n_heads, d_head,
    );
    g.output("y", out);
    let graph = g.build();

    let runtime = Runtime::boot();
    let compiled = compile(graph, &runtime.kernels).unwrap();

    let mut inputs = HashMap::new();
    let (name, t) = bind("x", &[seq, d_model], 1); inputs.insert(name, t);
    let (name, t) = bind("w_norm", &[d_model], 2); inputs.insert(name, t);
    let (name, t) = bind("w_q", &[d_model, d_model], 3); inputs.insert(name, t);
    let (name, t) = bind("w_k", &[d_model, d_model], 4); inputs.insert(name, t);
    let (name, t) = bind("w_v", &[d_model, d_model], 5); inputs.insert(name, t);
    let (name, t) = bind("w_o", &[d_model, d_model], 6); inputs.insert(name, t);

    let res1 = execute(&compiled, inputs.clone(), ExecutionOptions::default()).unwrap();
    let res2 = execute(&compiled, inputs, ExecutionOptions::default()).unwrap();

    let y = res1.outputs.get("y").unwrap();
    assert_eq!(y.meta.shape, vec![seq, d_model],
        "multi-head attention output must preserve [seq, d_model] shape");

    // Determinism.
    assert_eq!(
        res1.outputs.get("y").unwrap().storage.bytes,
        res2.outputs.get("y").unwrap().storage.bytes,
        "multi-head attention must be deterministic"
    );
}
