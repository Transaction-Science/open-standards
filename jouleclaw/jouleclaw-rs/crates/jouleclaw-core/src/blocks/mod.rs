//! Block constructors.
//!
//! Higher-level graph composition helpers built from the eight math
//! primitives. These are not new ops; they are graph builders that produce
//! subgraphs of the existing primitives.
//!
//! Phase 1.3 implements single-head attention (proper QK^T via transpose)
//! and gated SiLU FFN. Multi-head attention requires reshape + transpose
//! ops which arrive in Phase 2.

use crate::graph::{GraphBuilder, NodeId};
use crate::op::{ActivationKind, NormKind};

/// Single-head attention block.
///
/// ```text
///   xn      = Norm(x, w_norm)
///   q       = MatMul(xn, W_q)        // [seq, d]
///   k       = MatMul(xn, W_k)        // [seq, d]
///   v       = MatMul(xn, W_v)        // [seq, d]
///   scores  = MatMul(q, K^T)         // [seq, seq]
///   probs   = Softmax(scores, -1)    // [seq, seq], rows sum to 1
///   ctx     = MatMul(probs, v)       // [seq, d]
///   y       = MatMul(ctx, W_o)       // [seq, d]
///   out     = Add(x, y)              // residual
/// ```
///
/// All weight tensors are `[d, d]` for single-head. For multi-head see
/// [`multi_head_attention`].
pub fn attention(
    g: &mut GraphBuilder,
    x: NodeId,
    w_norm: NodeId,
    w_q: NodeId, w_k: NodeId, w_v: NodeId, w_o: NodeId,
) -> NodeId {
    let xn = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let q = g.matmul(xn, w_q);
    let k = g.matmul(xn, w_k);
    let v = g.matmul(xn, w_v);
    let scores = g.matmul_bt(q, k);
    let probs = g.softmax(scores, -1);
    let ctx = g.matmul(probs, v);
    let y = g.matmul(ctx, w_o);
    g.add(x, y)
}

/// Multi-head attention with RoPE positional encoding and `1/sqrt(d_head)`
/// score scaling. This is the production form — `multi_head_attention`
/// (above) is the simpler structural variant retained for tests.
///
/// Differences from `multi_head_attention`:
/// - Applies `Rope(base, position_offset=0)` to Q and K after the multi-head
///   split, so attention becomes position-aware.
/// - Multiplies QK^T scores by `1/sqrt(d_head)` (via `matmul_bt_scaled`)
///   so softmax sees properly-scaled values.
///
/// Position offset is fixed at 0 here (prefill); decoding will need a way
/// to thread the current step's position through.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_with_rope(
    g: &mut GraphBuilder,
    x: NodeId,
    w_norm: NodeId,
    w_q: NodeId, w_k: NodeId, w_v: NodeId, w_o: NodeId,
    seq_len: usize,
    n_heads: usize,
    d_head: usize,
    rope_base: f32,
) -> NodeId {
    let d_model = n_heads * d_head;

    let xn = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let q = g.matmul_bt(xn, w_q);
    let k = g.matmul_bt(xn, w_k);
    let v = g.matmul_bt(xn, w_v);

    // Split heads: [seq, d_model] -> [seq, n_heads, d_head] -> [n_heads, seq, d_head]
    let q_h = {
        let r = g.reshape(q, &[seq_len, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let k_h = {
        let r = g.reshape(k, &[seq_len, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let v_h = {
        let r = g.reshape(v, &[seq_len, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };

    // RoPE on Q and K. V is not rotated.
    let q_rot = g.rope(q_h, rope_base, 0);
    let k_rot = g.rope(k_h, rope_base, 0);

    // Scaled QK^T: 1/sqrt(d_head).
    let inv_sqrt_d = 1.0 / (d_head as f32).sqrt();
    let scores = g.matmul_bt_scaled(q_rot, k_rot, inv_sqrt_d);
    // Causal mask: future tokens excluded from each query's attention.
    let probs = g.softmax_causal(scores, -1);
    let ctx_h = g.matmul(probs, v_h);

    // Merge heads.
    let ctx = {
        let t = g.transpose(ctx_h, &[1, 0, 2]);
        g.reshape(t, &[seq_len, d_model])
    };

    let y = g.matmul_bt(ctx, w_o);
    g.add(x, y)
}

/// Grouped-Query Attention (GQA) variant of `multi_head_attention_with_rope`.
///
/// Used by Llama 2 / 3, Mistral, Qwen, and most modern LLMs. The Q projection
/// has `n_heads_q` heads (each of size `d_head`); the K and V projections
/// have `n_heads_kv` heads (also each of size `d_head`), where
/// `n_heads_kv <= n_heads_q` and `n_heads_q % n_heads_kv == 0`. Each K/V
/// head is shared by `group_size = n_heads_q / n_heads_kv` Q heads.
///
/// Implementation: after the multi-head split, K and V are repeat-interleaved
/// along the head axis by `group_size` so each K/V head broadcasts to the
/// `group_size` consecutive Q heads that share it. The rest of attention
/// runs as standard MHA at `n_heads_q`.
///
/// When `n_heads_kv == n_heads_q`, this collapses to plain MHA (group_size=1).
///
/// Tensor shapes:
/// - W_q: `[n_heads_q * d_head, d_model]`
/// - W_k, W_v: `[n_heads_kv * d_head, d_model]`
/// - W_o: `[d_model, n_heads_q * d_head]` (typically `d_model = n_heads_q * d_head`)
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_gqa(
    g: &mut GraphBuilder,
    x: NodeId,
    w_norm: NodeId,
    w_q: NodeId, w_k: NodeId, w_v: NodeId, w_o: NodeId,
    seq_len: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    d_head: usize,
    rope_base: f32,
) -> NodeId {
    assert!(n_heads_q % n_heads_kv == 0,
        "GQA: n_heads_q ({}) must be divisible by n_heads_kv ({})",
        n_heads_q, n_heads_kv);
    let group_size = n_heads_q / n_heads_kv;
    let d_model_q = n_heads_q * d_head;

    let xn = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let q = g.matmul_bt(xn, w_q);  // [seq, d_model_q]
    let k = g.matmul_bt(xn, w_k);  // [seq, n_heads_kv * d_head]
    let v = g.matmul_bt(xn, w_v);  // [seq, n_heads_kv * d_head]

    // Multi-head split (different head counts for Q vs K/V).
    let q_h = {
        let r = g.reshape(q, &[seq_len, n_heads_q, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let k_h = {
        let r = g.reshape(k, &[seq_len, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let v_h = {
        let r = g.reshape(v, &[seq_len, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])
    };

    // RoPE on Q and K (V is not rotated).
    let q_rot = g.rope(q_h, rope_base, 0);
    let k_rot = g.rope(k_h, rope_base, 0);

    // GQA broadcast: repeat-interleave each K/V head `group_size` times.
    // Reshape to add a singleton group axis, repeat along it, then merge:
    //   [n_heads_kv, seq, d_head] -> [n_heads_kv, 1, seq, d_head]
    //   -> repeat axis=1 -> [n_heads_kv, group_size, seq, d_head]
    //   -> reshape [n_heads_q, seq, d_head]
    // This places kv-head i at output positions [i*group_size .. (i+1)*group_size],
    // which is the correct mapping for GQA (Q head j attends against
    // KV head j / group_size).
    let k_broad = if group_size == 1 { k_rot } else {
        let r = g.reshape(k_rot, &[n_heads_kv, 1, seq_len, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads_q, seq_len, d_head])
    };
    let v_broad = if group_size == 1 { v_h } else {
        let r = g.reshape(v_h, &[n_heads_kv, 1, seq_len, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads_q, seq_len, d_head])
    };

    let inv_sqrt_d = 1.0 / (d_head as f32).sqrt();
    let scores = g.matmul_bt_scaled(q_rot, k_broad, inv_sqrt_d);
    let probs = g.softmax_causal(scores, -1);
    let ctx_h = g.matmul(probs, v_broad);

    let ctx = {
        let t = g.transpose(ctx_h, &[1, 0, 2]);
        g.reshape(t, &[seq_len, d_model_q])
    };
    let y = g.matmul_bt(ctx, w_o);
    g.add(x, y)
}

/// GQA with qwen3-style per-head QK-RMSNorm.
///
/// Identical to [`multi_head_attention_gqa`] except that, after the
/// head split and *before* RoPE, Q and K are RMS-normalised per head
/// using learned `[d_head]` weights `w_q_norm` / `w_k_norm`. This is
/// the defining qwen3 attention modification; without it qwen3 weights
/// produce incoherent output. `d_head` is taken as given (qwen3
/// decouples it from `d_model / n_heads`).
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_gqa_qknorm(
    g: &mut GraphBuilder,
    x: NodeId,
    w_norm: NodeId,
    w_q: NodeId, w_k: NodeId, w_v: NodeId, w_o: NodeId,
    w_q_norm: NodeId, w_k_norm: NodeId,
    seq_len: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    d_head: usize,
    rope_base: f32,
    rms_eps: f32,
) -> NodeId {
    assert!(n_heads_q % n_heads_kv == 0,
        "GQA: n_heads_q ({}) must be divisible by n_heads_kv ({})",
        n_heads_q, n_heads_kv);
    let group_size = n_heads_q / n_heads_kv;
    let d_model_q = n_heads_q * d_head;

    let xn = g.norm(x, w_norm, NormKind::Rms, rms_eps);
    let q = g.matmul_bt(xn, w_q);
    let k = g.matmul_bt(xn, w_k);
    let v = g.matmul_bt(xn, w_v);

    let q_h = {
        let r = g.reshape(q, &[seq_len, n_heads_q, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let k_h = {
        let r = g.reshape(k, &[seq_len, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let v_h = {
        let r = g.reshape(v, &[seq_len, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])
    };

    // Per-head QK-RMSNorm over the last axis (d_head), weight broadcast
    // across [n_heads, seq].
    let q_h = g.norm(q_h, w_q_norm, NormKind::Rms, rms_eps);
    let k_h = g.norm(k_h, w_k_norm, NormKind::Rms, rms_eps);

    let q_rot = g.rope(q_h, rope_base, 0);
    let k_rot = g.rope(k_h, rope_base, 0);

    let k_broad = if group_size == 1 { k_rot } else {
        let r = g.reshape(k_rot, &[n_heads_kv, 1, seq_len, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads_q, seq_len, d_head])
    };
    let v_broad = if group_size == 1 { v_h } else {
        let r = g.reshape(v_h, &[n_heads_kv, 1, seq_len, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads_q, seq_len, d_head])
    };

    let inv_sqrt_d = 1.0 / (d_head as f32).sqrt();
    let scores = g.matmul_bt_scaled(q_rot, k_broad, inv_sqrt_d);
    let probs = g.softmax_causal(scores, -1);
    let ctx_h = g.matmul(probs, v_broad);

    let ctx = {
        let t = g.transpose(ctx_h, &[1, 0, 2]);
        g.reshape(t, &[seq_len, d_model_q])
    };
    let y = g.matmul_bt(ctx, w_o);
    g.add(x, y)
}

/// Phase 1.5b limitation: this block does not include scaling by
/// `1/sqrt(d_head)`. Real attention requires it; we'll add a `Scale` op
/// (or `alpha` to MatMul attrs) in a follow-on. For now the softmax over
/// raw scores still produces a valid probability distribution — just less
/// well-calibrated.
///
/// Phase 1.5b limitation: no RoPE positional encoding. Position is purely
/// content-based here.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    g: &mut GraphBuilder,
    x: NodeId,
    w_norm: NodeId,
    w_q: NodeId, w_k: NodeId, w_v: NodeId, w_o: NodeId,
    seq_len: usize,
    n_heads: usize,
    d_head: usize,
) -> NodeId {
    let d_model = n_heads * d_head;

    let xn = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let q = g.matmul_bt(xn, w_q);
    let k = g.matmul_bt(xn, w_k);
    let v = g.matmul_bt(xn, w_v);

    // Split heads: [seq, d_model] -> [seq, n_heads, d_head] -> [n_heads, seq, d_head]
    let q_h = {
        let r = g.reshape(q, &[seq_len, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let k_h = {
        let r = g.reshape(k, &[seq_len, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let v_h = {
        let r = g.reshape(v, &[seq_len, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };

    // Per-head attention. Both inputs are 3D with matching leading dim
    // (n_heads), so the kernel runs in batched mode.
    let scores = g.matmul_bt(q_h, k_h);    // [n_heads, seq, seq]
    let probs = g.softmax(scores, -1);     // [n_heads, seq, seq]
    let ctx_h = g.matmul(probs, v_h);      // [n_heads, seq, d_head]

    // Merge heads back: [n_heads, seq, d_head] -> [seq, n_heads, d_head] -> [seq, d_model]
    let ctx = {
        let t = g.transpose(ctx_h, &[1, 0, 2]);
        g.reshape(t, &[seq_len, d_model])
    };

    let y = g.matmul_bt(ctx, w_o);
    g.add(x, y)
}

/// Gated SiLU feed-forward block (Llama-style).
///
/// ```text
///   xn      = Norm(x, w_norm)
///   gate    = Activation(MatMul(xn, W_gate), SiLU)
///   up      = MatMul(xn, W_up)
///   hidden  = Mul(gate, up)
///   y       = MatMul(hidden, W_down)
///   out     = Add(x, y)
/// ```
pub fn ffn_gated_silu(
    g: &mut GraphBuilder,
    x: NodeId,
    w_norm: NodeId,
    w_gate: NodeId, w_up: NodeId, w_down: NodeId,
) -> NodeId {
    let xn = g.norm(x, w_norm, NormKind::Rms, 1e-6);
    let gate = g.matmul(xn, w_gate);
    let gate = g.activation(gate, ActivationKind::SiLU);
    let up = g.matmul(xn, w_up);
    let hidden = g.mul(gate, up);
    let y = g.matmul(hidden, w_down);
    g.add(x, y)
}

/// One full transformer block (attention + FFN with residuals).
pub fn transformer_block(
    g: &mut GraphBuilder,
    x: NodeId,
    w_attn_norm: NodeId,
    w_q: NodeId, w_k: NodeId, w_v: NodeId, w_o: NodeId,
    w_ffn_norm: NodeId,
    w_gate: NodeId, w_up: NodeId, w_down: NodeId,
) -> NodeId {
    let after_attn = attention(g, x, w_attn_norm, w_q, w_k, w_v, w_o);
    ffn_gated_silu(g, after_attn, w_ffn_norm, w_gate, w_up, w_down)
}
