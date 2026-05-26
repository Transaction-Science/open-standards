//! # Example: attention block from primitives
//!
//! Builds a single transformer attention block as a graph using only the
//! eight math primitives. Demonstrates that no special "attention op" is
//! needed — the block is just composition.
//!
//! Run with: `cargo run --example attention_block -p jouleclaw-core`
//! (Phase 0: graph builds and prints; execution lands in Phase 1.)

use jouleclaw_core::graph::{Graph, GraphBuilder};
use jouleclaw_core::op::{ActivationKind, NormKind};
use jouleclaw_core::tensor::{Dtype, LifetimeTier, TensorMeta};

/// Build an attention block:
///
/// ```text
///   x_norm  = Norm(x, w_norm)
///   q       = MatMul(x_norm, W_q)
///   k       = MatMul(x_norm, W_k)
///   v       = MatMul(x_norm, W_v)
///   scores  = MatMul(q, K^T)            // K^T not yet exposed; placeholder via attrs
///   probs   = Softmax(scores, axis=-1)
///   ctx     = MatMul(probs, v)
///   y       = MatMul(ctx, W_o)
///   out     = Add(x, y)                  // residual
/// ```
fn build_attention_block(
    g: &mut GraphBuilder,
    x: jouleclaw_core::graph::NodeId,
    w_norm: jouleclaw_core::graph::NodeId,
    w_q: jouleclaw_core::graph::NodeId,
    w_k: jouleclaw_core::graph::NodeId,
    w_v: jouleclaw_core::graph::NodeId,
    w_o: jouleclaw_core::graph::NodeId,
) -> jouleclaw_core::graph::NodeId {
    let x_norm = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let q = g.matmul(x_norm, w_q);
    let k = g.matmul(x_norm, w_k);
    let v = g.matmul(x_norm, w_v);
    // For Phase 0, we approximate (q · k^T) with a single matmul; Phase 1 will
    // add explicit transpose attrs to MatMul (`transpose_b: true`).
    let scores = g.matmul(q, k);
    let probs = g.softmax(scores, -1);
    let ctx = g.matmul(probs, v);
    let y = g.matmul(ctx, w_o);
    g.add(x, y)
}

/// Build a feed-forward block (gated SiLU variant, as in Llama):
///
/// ```text
///   x_norm = Norm(x, w_norm)
///   gate   = Activation(MatMul(x_norm, W_gate), SiLU)
///   up     = MatMul(x_norm, W_up)
///   hidden = Mul(gate, up)
///   y      = MatMul(hidden, W_down)
///   out    = Add(x, y)
/// ```
fn build_ffn_block(
    g: &mut GraphBuilder,
    x: jouleclaw_core::graph::NodeId,
    w_norm: jouleclaw_core::graph::NodeId,
    w_gate: jouleclaw_core::graph::NodeId,
    w_up: jouleclaw_core::graph::NodeId,
    w_down: jouleclaw_core::graph::NodeId,
) -> jouleclaw_core::graph::NodeId {
    let x_norm = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let gate = g.matmul(x_norm, w_gate);
    let gate = g.activation(gate, ActivationKind::SiLU);
    let up = g.matmul(x_norm, w_up);
    let hidden = g.mul(gate, up);
    let y = g.matmul(hidden, w_down);
    g.add(x, y)
}

fn build_transformer_block_graph() -> Graph {
    let mut g = GraphBuilder::new();

    // Hyperparameters (representative of a small model).
    let seq_len = 512;
    let d_model = 1024;
    let d_ff = 4096;

    // Input.
    let x = g.input(
        "x",
        TensorMeta::new(Dtype::F16, &[seq_len, d_model]),
    );

    // Constants (weights). In a real graph these are loaded from a file
    // and tagged with LifetimeTier::Cold; here we use placeholder shapes.
    let w_attn_norm = g.input("w_attn_norm",
        TensorMeta::new(Dtype::F16, &[d_model]).with_tier(LifetimeTier::Cold));
    let w_q = g.input("w_q",
        TensorMeta::new(Dtype::F16, &[d_model, d_model]).with_tier(LifetimeTier::Cold));
    let w_k = g.input("w_k",
        TensorMeta::new(Dtype::F16, &[d_model, d_model]).with_tier(LifetimeTier::Cold));
    let w_v = g.input("w_v",
        TensorMeta::new(Dtype::F16, &[d_model, d_model]).with_tier(LifetimeTier::Cold));
    let w_o = g.input("w_o",
        TensorMeta::new(Dtype::F16, &[d_model, d_model]).with_tier(LifetimeTier::Cold));

    let w_ffn_norm = g.input("w_ffn_norm",
        TensorMeta::new(Dtype::F16, &[d_model]).with_tier(LifetimeTier::Cold));
    let w_gate = g.input("w_gate",
        TensorMeta::new(Dtype::F16, &[d_model, d_ff]).with_tier(LifetimeTier::Cold));
    let w_up = g.input("w_up",
        TensorMeta::new(Dtype::F16, &[d_model, d_ff]).with_tier(LifetimeTier::Cold));
    let w_down = g.input("w_down",
        TensorMeta::new(Dtype::F16, &[d_ff, d_model]).with_tier(LifetimeTier::Cold));

    let after_attn = build_attention_block(&mut g, x, w_attn_norm, w_q, w_k, w_v, w_o);
    let after_ffn = build_ffn_block(&mut g, after_attn, w_ffn_norm, w_gate, w_up, w_down);

    g.output("y", after_ffn);

    g.build()
}

fn main() {
    let graph = build_transformer_block_graph();
    println!("Built transformer block graph with {} nodes", graph.nodes.len());
    println!("Inputs:  {}", graph.inputs.len());
    println!("Outputs: {}", graph.outputs.len());

    // Count ops by kind to confirm composition.
    use std::collections::HashMap;
    use jouleclaw_core::graph::NodeKind;
    let mut counts: HashMap<&'static str, u32> = HashMap::new();
    for node in &graph.nodes {
        let key = match &node.kind {
            NodeKind::Input { .. } => "Input",
            NodeKind::Output { .. } => "Output",
            NodeKind::Constant { .. } => "Constant",
            NodeKind::Op { op, .. } => match op {
                jouleclaw_core::op::OpKind::MatMul => "MatMul",
                jouleclaw_core::op::OpKind::Softmax => "Softmax",
                jouleclaw_core::op::OpKind::Norm => "Norm",
                jouleclaw_core::op::OpKind::Activation => "Activation",
                jouleclaw_core::op::OpKind::Add => "Add",
                jouleclaw_core::op::OpKind::Mul => "Mul",
                jouleclaw_core::op::OpKind::Lookup => "Lookup",
                jouleclaw_core::op::OpKind::Sample => "Sample",
                _ => "Other",
            },
        };
        *counts.entry(key).or_insert(0) += 1;
    }

    let mut entries: Vec<_> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\nNode breakdown:");
    for (kind, count) in entries {
        println!("  {:>10}: {}", kind, count);
    }
}
