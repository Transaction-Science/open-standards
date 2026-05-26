//! Dataflow graph representation.
//!
//! A `Graph` is a DAG of operations. Built once, executable many times.
//! See spec 02.

use crate::op::{ActivationKind, NormKind, OpAttrs, OpKind, SamplerKind};
use crate::tensor::{Dtype, TensorMeta, TensorRef};

/// Stable identifier for a node within a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Node within a graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub inputs: Vec<NodeId>,
    pub output_meta: Vec<TensorMeta>,
    /// In-place storage hint. When `Some(i)`, the executor is permitted
    /// (but not required) to reuse input `i`'s storage as the output's
    /// storage rather than allocating fresh bytes. This is safe iff:
    /// (a) the output's shape, dtype, and total byte size match input i's,
    /// (b) no downstream node consumes input i directly (only its
    ///     transitive uses go through this node's output), AND
    /// (c) the kernel honors the alias (writes the necessary bytes;
    ///     reads input i's pre-write content first if it matters).
    /// Today this is only set by GraphBuilder::scatter_inplace; the
    /// executor checks the storage Arc's strong-count and falls back to
    /// allocation if the input is shared.
    pub aliases_input: Option<u32>,
}

/// What a node represents.
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// External input. Bound at execution time.
    Input { name: String, meta: TensorMeta },
    /// External output.
    Output { name: String },
    /// Constant tensor embedded in the graph (e.g., a weight).
    Constant { tensor: TensorRef },
    /// Operation node.
    Op { op: OpKind, attrs: OpAttrs },
}

/// A built graph. Immutable once constructed.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
}

impl Graph {
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }
}

/// Builder for constructing graphs.
pub struct GraphBuilder {
    nodes: Vec<Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    fn add_node(&mut self, kind: NodeKind, inputs: Vec<NodeId>, output_meta: Vec<TensorMeta>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node { id, kind, inputs, output_meta, aliases_input: None });
        id
    }

    pub fn input(&mut self, name: &str, meta: TensorMeta) -> NodeId {
        let id = self.add_node(
            NodeKind::Input { name: name.to_string(), meta: meta.clone() },
            Vec::new(),
            vec![meta],
        );
        self.inputs.push(id);
        id
    }

    pub fn constant(&mut self, tensor: TensorRef) -> NodeId {
        let meta = tensor.meta.clone();
        self.add_node(
            NodeKind::Constant { tensor },
            Vec::new(),
            vec![meta],
        )
    }

    pub fn output(&mut self, name: &str, source: NodeId) {
        let meta = self.nodes[source.0 as usize].output_meta[0].clone();
        let id = self.add_node(
            NodeKind::Output { name: name.to_string() },
            vec![source],
            vec![meta],
        );
        self.outputs.push(id);
    }

    // ---- Op constructors ----
    // These compute output metadata via the op's signature; Phase 0 uses
    // simple inference where possible and `unimplemented` placeholders elsewhere.

    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.matmul_with(a, b, false, false)
    }

    /// MatMul with optional transposes on either operand.
    ///
    /// Logical shapes:
    /// - A: `[..., m, k]` (or `[..., k, m]` if `transpose_a`)
    /// - B: `[k, n]`      — broadcast across A's batch dims
    /// - or B: `[..., k, n]` (or `[..., n, k]` if `transpose_b`) — batched, A's
    ///   and B's leading batch dims must match.
    /// - Output: `[..., m, n]`
    pub fn matmul_with(
        &mut self, a: NodeId, b: NodeId,
        transpose_a: bool, transpose_b: bool,
    ) -> NodeId {
        self.matmul_with_alpha(a, b, transpose_a, transpose_b, 1.0)
    }

    /// MatMul with explicit output scale `alpha`. Used for attention's
    /// `1/sqrt(d_head)` score scaling.
    pub fn matmul_with_alpha(
        &mut self, a: NodeId, b: NodeId,
        transpose_a: bool, transpose_b: bool, alpha: f32,
    ) -> NodeId {
        let a_meta = &self.nodes[a.0 as usize].output_meta[0];
        let b_meta = &self.nodes[b.0 as usize].output_meta[0];
        let a_shape = &a_meta.shape;
        let b_shape = &b_meta.shape;
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let m = if transpose_a { a_shape[a_rank - 1] } else { a_shape[a_rank - 2] };
        let n = if transpose_b { b_shape[b_rank - 2] } else { b_shape[b_rank - 1] };

        let mut out_shape = a_shape.clone();
        out_shape[a_rank - 2] = m;
        *out_shape.last_mut().unwrap() = n;

        let out_meta = TensorMeta::new(a_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::MatMul,
                attrs: OpAttrs::MatMul { transpose_a, transpose_b, alpha, b_n_valid: None },
            },
            vec![a, b],
            vec![out_meta],
        )
    }

    /// `MatMul(A, B^T)`. Shorthand for `matmul_with(a, b, false, true)`.
    /// Used for the QK^T product in attention.
    pub fn matmul_bt(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.matmul_with(a, b, false, true)
    }

    /// `A @ W^T` where `b` is a ternary-packed weight constant
    /// (PrismML Q2_0). `a` is dense f32 `[..., m, k]`; `b` is the raw
    /// packed byte blob (U8, shape `[packed_len]`); output is dense f32
    /// `[..., m, out]`. `out`/`k` are passed explicitly because the
    /// packed operand's shape is its byte length, not `[out, k]`.
    ///
    /// Semantically identical to `matmul_bt(a, dequantize(b))` but the
    /// kernel never materialises the f32 weight: it sign-selects and
    /// accumulates against the 2-bit codes, applying the per-128-block
    /// f16 scale once per block. Same numeric result (ternary is exact
    /// on its three code points), far lower energy and memory traffic.
    pub fn matmul_bt_ternary(
        &mut self, a: NodeId, b: NodeId, out: usize, k: usize,
    ) -> NodeId {
        let a_meta = &self.nodes[a.0 as usize].output_meta[0];
        let a_shape = &a_meta.shape;
        let a_rank = a_shape.len();
        debug_assert_eq!(
            a_shape[a_rank - 1], k,
            "matmul_bt_ternary: A's last dim ({}) must equal k ({})",
            a_shape[a_rank - 1], k);
        let mut out_shape = a_shape.clone();
        *out_shape.last_mut().unwrap() = out;
        let out_meta = TensorMeta::new(a_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::MatMulTernary,
                attrs: OpAttrs::MatMulTernary { out, k },
            },
            vec![a, b],
            vec![out_meta],
        )
    }

    /// `A @ W^T` where `b` is a Q8_0 packed weight (32-element blocks,
    /// 1 fp16 scale per block, 8.5 bits/weight on disk). Same
    /// shape/output contract as [`Self::matmul_bt_ternary`]; the
    /// kernel keeps the bytes packed and runs an int8×int8→fp32
    /// fused matmul (NEON dot product + per-block scale fold-in).
    /// Bypasses the dequant-to-fp32 + sgemm round-trip on edge
    /// targets where AMX isn't available.
    pub fn matmul_bt_q8_0(
        &mut self, a: NodeId, b: NodeId, out: usize, k: usize,
    ) -> NodeId {
        let a_meta = &self.nodes[a.0 as usize].output_meta[0];
        let a_shape = &a_meta.shape;
        let a_rank = a_shape.len();
        debug_assert_eq!(
            a_shape[a_rank - 1], k,
            "matmul_bt_q8_0: A's last dim ({}) must equal k ({})",
            a_shape[a_rank - 1], k);
        debug_assert_eq!(
            k % 32, 0,
            "matmul_bt_q8_0: k ({}) must be a multiple of the Q8_0 block size 32", k);
        let mut out_shape = a_shape.clone();
        *out_shape.last_mut().unwrap() = out;
        let out_meta = TensorMeta::new(a_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::MatMulQ80,
                attrs: OpAttrs::MatMulQ80 { out, k },
            },
            vec![a, b],
            vec![out_meta],
        )
    }

    /// `A @ W^T` where `b` is a Tencent STQ1_0 packed weight (3:4
    /// sparse ternary g256, 1.3125 bpw). Same shape/output contract as
    /// [`Self::matmul_bt_ternary`]; the kernel decodes per-group via
    /// the 32-entry codebook LUT.
    pub fn matmul_bt_stq1_0(
        &mut self, a: NodeId, b: NodeId, out: usize, k: usize,
    ) -> NodeId {
        let a_meta = &self.nodes[a.0 as usize].output_meta[0];
        let a_shape = &a_meta.shape;
        let a_rank = a_shape.len();
        debug_assert_eq!(
            a_shape[a_rank - 1], k,
            "matmul_bt_stq1_0: A's last dim ({}) must equal k ({})",
            a_shape[a_rank - 1], k);
        let mut out_shape = a_shape.clone();
        *out_shape.last_mut().unwrap() = out;
        let out_meta = TensorMeta::new(a_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::MatMulSTQ1_0,
                attrs: OpAttrs::MatMulSTQ1_0 { out, k },
            },
            vec![a, b],
            vec![out_meta],
        )
    }

    /// `A @ W^T` where `b` is a 1-bit packed weight constant
    /// (PrismML Q1_0 g128). Same shape/output contract as
    /// [`Self::matmul_bt_ternary`], but the kernel operates over 1-bit
    /// codes — `bit==1 → +d, bit==0 → −d` — with no zero case to skip.
    pub fn matmul_bt_bit(
        &mut self, a: NodeId, b: NodeId, out: usize, k: usize,
    ) -> NodeId {
        let a_meta = &self.nodes[a.0 as usize].output_meta[0];
        let a_shape = &a_meta.shape;
        let a_rank = a_shape.len();
        debug_assert_eq!(
            a_shape[a_rank - 1], k,
            "matmul_bt_bit: A's last dim ({}) must equal k ({})",
            a_shape[a_rank - 1], k);
        let mut out_shape = a_shape.clone();
        *out_shape.last_mut().unwrap() = out;
        let out_meta = TensorMeta::new(a_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::MatMulBit,
                attrs: OpAttrs::MatMulBit { out, k },
            },
            vec![a, b],
            vec![out_meta],
        )
    }

    /// `alpha * MatMul(A, B^T)`. Used for scaled attention scores.
    pub fn matmul_bt_scaled(&mut self, a: NodeId, b: NodeId, alpha: f32) -> NodeId {
        self.matmul_with_alpha(a, b, false, true, alpha)
    }

    /// `alpha * MatMul(A, B^T)` with a logical truncation of B's "N" axis
    /// to `b_n_valid` positions. Used as a perf optimization for in-place
    /// KV cache attention: the K buffer is preallocated full-sized but
    /// only the first `valid_seq` positions are live. Output shape's last
    /// dim is `b_n_valid` instead of B's actual N-axis size.
    ///
    /// Correctness note: this is a perf hint, not a correctness requirement
    /// for in-place KV. The default `matmul_bt` over the full buffer also
    /// produces correct attention output (the offset-causal softmax masks
    /// any "junk" positions to zero, suppressing V's content at those
    /// positions in the subsequent `probs @ V` product). The sliced form
    /// avoids computing scores that would just be masked anyway.
    pub fn matmul_bt_scaled_sliced(
        &mut self, a: NodeId, b: NodeId, alpha: f32, b_n_valid: usize,
    ) -> NodeId {
        let a_meta = &self.nodes[a.0 as usize].output_meta[0];
        let b_meta = &self.nodes[b.0 as usize].output_meta[0];
        let a_shape = &a_meta.shape;
        let b_shape = &b_meta.shape;
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();
        assert!(b_n_valid <= b_shape[b_rank - 2],
            "b_n_valid {} exceeds B's N axis size {}", b_n_valid, b_shape[b_rank - 2]);
        let m = a_shape[a_rank - 2];
        let mut out_shape = a_shape.clone();
        out_shape[a_rank - 2] = m;
        *out_shape.last_mut().unwrap() = b_n_valid;
        let out_meta = TensorMeta::new(a_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::MatMul,
                attrs: OpAttrs::MatMul {
                    transpose_a: false, transpose_b: true, alpha,
                    b_n_valid: Some(b_n_valid),
                },
            },
            vec![a, b],
            vec![out_meta],
        )
    }

    pub fn softmax(&mut self, x: NodeId, axis: i32) -> NodeId {
        self.softmax_with(x, axis, false, 0)
    }

    /// Softmax with a causal (upper-triangular) mask: positions `j > i` are
    /// excluded from the row-i softmax. Required for autoregressive attention.
    pub fn softmax_causal(&mut self, x: NodeId, axis: i32) -> NodeId {
        self.softmax_with(x, axis, true, 0)
    }

    /// Softmax with a causal mask plus a key offset, for KV-cache decode
    /// where the query slice is shorter than the key slice. Query at
    /// relative position `i` attends to keys `0..=i + causal_offset`.
    pub fn softmax_causal_offset(&mut self, x: NodeId, axis: i32, causal_offset: i32) -> NodeId {
        self.softmax_with(x, axis, true, causal_offset)
    }

    fn softmax_with(&mut self, x: NodeId, axis: i32, causal: bool, causal_offset: i32) -> NodeId {
        let meta = self.nodes[x.0 as usize].output_meta[0].clone();
        self.add_node(
            NodeKind::Op { op: OpKind::Softmax, attrs: OpAttrs::Softmax { axis, causal, causal_offset } },
            vec![x],
            vec![meta],
        )
    }

    /// Concatenate two tensors along the given axis. Both must have the
    /// same rank and matching shapes on all axes except `axis`. Negative
    /// axis indices count from the end (`-1` = last axis).
    pub fn concat(&mut self, a: NodeId, b: NodeId, axis: i32) -> NodeId {
        let a_meta = self.nodes[a.0 as usize].output_meta[0].clone();
        let b_meta = &self.nodes[b.0 as usize].output_meta[0];
        let rank = a_meta.shape.len();
        let axis_resolved = if axis < 0 { (rank as i32 + axis) as usize } else { axis as usize };
        assert!(axis_resolved < rank,
            "concat axis {} out of range for rank {}", axis, rank);
        assert_eq!(a_meta.shape.len(), b_meta.shape.len(),
            "concat: rank mismatch");
        for d in 0..rank {
            if d != axis_resolved {
                assert_eq!(a_meta.shape[d], b_meta.shape[d],
                    "concat: shape mismatch on non-concat axis {}", d);
            }
        }
        let mut out_shape = a_meta.shape.clone();
        out_shape[axis_resolved] += b_meta.shape[axis_resolved];
        let out_meta = TensorMeta::new(a_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op { op: OpKind::Concat, attrs: OpAttrs::Concat { axis } },
            vec![a, b],
            vec![out_meta],
        )
    }

    /// Repeat (tile) a tensor along the given axis: the axis's size is
    /// multiplied by `repeats`, with each repetition being a copy of the
    /// original axis. Used for GQA head broadcasting.
    pub fn repeat(&mut self, x: NodeId, axis: i32, repeats: usize) -> NodeId {
        let in_meta = self.nodes[x.0 as usize].output_meta[0].clone();
        let rank = in_meta.shape.len();
        let axis_resolved = if axis < 0 { (rank as i32 + axis) as usize } else { axis as usize };
        assert!(axis_resolved < rank,
            "repeat axis {} out of range for rank {}", axis, rank);
        assert!(repeats >= 1, "repeats must be >= 1, got {}", repeats);
        let mut out_shape = in_meta.shape.clone();
        out_shape[axis_resolved] *= repeats;
        let out_meta = TensorMeta::new(in_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::Repeat,
                attrs: OpAttrs::Repeat { axis, repeats },
            },
            vec![x],
            vec![out_meta],
        )
    }

    /// Write `src` into `dst` at `offset` along `axis`. Output has dst's
    /// shape and dtype; positions outside the scatter region are copied
    /// from dst unchanged.
    pub fn scatter(&mut self, dst: NodeId, src: NodeId, axis: i32, offset: usize) -> NodeId {
        self.scatter_impl(dst, src, axis, offset, false)
    }

    /// Same as `scatter`, but tags the output to alias `dst`'s storage.
    /// The executor may then mutate `dst`'s bytes directly rather than
    /// allocating fresh output and copying. Safe iff no other graph node
    /// consumes `dst` after this point.
    pub fn scatter_inplace(&mut self, dst: NodeId, src: NodeId, axis: i32, offset: usize) -> NodeId {
        self.scatter_impl(dst, src, axis, offset, true)
    }

    fn scatter_impl(&mut self, dst: NodeId, src: NodeId, axis: i32, offset: usize, in_place: bool) -> NodeId {
        let dst_meta = self.nodes[dst.0 as usize].output_meta[0].clone();
        let src_meta = &self.nodes[src.0 as usize].output_meta[0];
        let rank = dst_meta.shape.len();
        let axis_resolved = if axis < 0 { (rank as i32 + axis) as usize } else { axis as usize };
        assert!(axis_resolved < rank,
            "scatter axis {} out of range for rank {}", axis, rank);
        assert_eq!(dst_meta.shape.len(), src_meta.shape.len(),
            "scatter: rank mismatch");
        for d in 0..rank {
            if d != axis_resolved {
                assert_eq!(dst_meta.shape[d], src_meta.shape[d],
                    "scatter: shape mismatch on non-scatter axis {}", d);
            }
        }
        assert!(offset + src_meta.shape[axis_resolved] <= dst_meta.shape[axis_resolved],
            "scatter out of bounds: offset {} + src_len {} > dst_len {} on axis {}",
            offset, src_meta.shape[axis_resolved], dst_meta.shape[axis_resolved], axis_resolved);
        let id = self.add_node(
            NodeKind::Op {
                op: OpKind::Scatter,
                attrs: OpAttrs::Scatter { axis, offset },
            },
            vec![dst, src],
            vec![dst_meta],
        );
        if in_place {
            self.nodes[id.0 as usize].aliases_input = Some(0);  // alias dst (input 0)
        }
        id
    }

    /// Extract a contiguous slice along `axis` starting at `start` with
    /// `length` elements. Output shape matches input except `axis` becomes
    /// `length`.
    pub fn slice(&mut self, x: NodeId, axis: i32, start: usize, length: usize) -> NodeId {
        let in_meta = self.nodes[x.0 as usize].output_meta[0].clone();
        let rank = in_meta.shape.len();
        let axis_resolved = if axis < 0 { (rank as i32 + axis) as usize } else { axis as usize };
        assert!(axis_resolved < rank,
            "slice axis {} out of range for rank {}", axis, rank);
        assert!(start + length <= in_meta.shape[axis_resolved],
            "slice out of bounds: start {} + length {} > dim {} on axis {}",
            start, length, in_meta.shape[axis_resolved], axis_resolved);
        let mut out_shape = in_meta.shape.clone();
        out_shape[axis_resolved] = length;
        let out_meta = TensorMeta::new(in_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::Slice,
                attrs: OpAttrs::Slice { axis, start, length },
            },
            vec![x],
            vec![out_meta],
        )
    }

    pub fn norm(&mut self, x: NodeId, weight: NodeId, kind: NormKind, eps: f32) -> NodeId {
        let meta = self.nodes[x.0 as usize].output_meta[0].clone();
        self.add_node(
            NodeKind::Op { op: OpKind::Norm, attrs: OpAttrs::Norm { kind, eps } },
            vec![x, weight],
            vec![meta],
        )
    }

    pub fn activation(&mut self, x: NodeId, kind: ActivationKind) -> NodeId {
        let meta = self.nodes[x.0 as usize].output_meta[0].clone();
        self.add_node(
            NodeKind::Op { op: OpKind::Activation, attrs: OpAttrs::Activation { kind } },
            vec![x],
            vec![meta],
        )
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let meta = self.nodes[a.0 as usize].output_meta[0].clone();
        self.add_node(
            NodeKind::Op { op: OpKind::Add, attrs: OpAttrs::Add },
            vec![a, b],
            vec![meta],
        )
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let meta = self.nodes[a.0 as usize].output_meta[0].clone();
        self.add_node(
            NodeKind::Op { op: OpKind::Mul, attrs: OpAttrs::Mul },
            vec![a, b],
            vec![meta],
        )
    }

    pub fn lookup(&mut self, idx: NodeId, table: NodeId) -> NodeId {
        let idx_meta = &self.nodes[idx.0 as usize].output_meta[0];
        let tbl_meta = &self.nodes[table.0 as usize].output_meta[0];
        // [n] indexing into [V, d] -> [n, d]
        let d = tbl_meta.shape[tbl_meta.shape.len() - 1];
        let mut out_shape = idx_meta.shape.clone();
        out_shape.push(d);
        let out_meta = TensorMeta::new(tbl_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::Lookup,
                attrs: OpAttrs::Lookup { scale_by_inv_sqrt_d: false },
            },
            vec![idx, table],
            vec![out_meta],
        )
    }

    /// Embedding lookup against a ternary-packed table (PrismML Q2_0).
    /// `table` is the raw packed byte blob (U8); only the rows named by
    /// `idx` are decoded to f32. Output `[..idx.., d]` f32. Same result
    /// as `lookup(idx, dequantize(table))` without ever materialising
    /// the full f32 table.
    pub fn lookup_ternary(
        &mut self, idx: NodeId, table: NodeId, v: usize, d: usize,
    ) -> NodeId {
        let idx_meta = &self.nodes[idx.0 as usize].output_meta[0];
        let mut out_shape = idx_meta.shape.clone();
        out_shape.push(d);
        let out_meta = TensorMeta::new(Dtype::F32, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::LookupTernary,
                attrs: OpAttrs::LookupTernary { v, d },
            },
            vec![idx, table],
            vec![out_meta],
        )
    }

    /// Depthwise causal 1-D convolution over the sequence axis. `x`
    /// is `[seq, d_model]` f32, `w` is `[taps, d_model]` f32. Output
    /// is `[seq, d_model]` f32. Each channel is convolved
    /// independently with its own kernel; positions before the start
    /// of the sequence are treated as zero (causal left-pad). Used by
    /// LFM2's `shortconv` recurrent block.
    pub fn conv1d_depthwise_causal(
        &mut self, x: NodeId, w: NodeId, taps: usize,
    ) -> NodeId {
        let x_meta = &self.nodes[x.0 as usize].output_meta[0];
        let out_meta = TensorMeta::new(x_meta.dtype, &x_meta.shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::Conv1DDepthwiseCausal,
                attrs: OpAttrs::Conv1DDepthwise { taps },
            },
            vec![x, w],
            vec![out_meta],
        )
    }

    /// Embedding lookup against a 1-bit packed table (PrismML Q1_0).
    /// Same row-on-demand decode semantics as [`Self::lookup_ternary`]
    /// — never materialises the full f32 table.
    pub fn lookup_bit(
        &mut self, idx: NodeId, table: NodeId, v: usize, d: usize,
    ) -> NodeId {
        let idx_meta = &self.nodes[idx.0 as usize].output_meta[0];
        let mut out_shape = idx_meta.shape.clone();
        out_shape.push(d);
        let out_meta = TensorMeta::new(Dtype::F32, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::LookupBit,
                attrs: OpAttrs::LookupBit { v, d },
            },
            vec![idx, table],
            vec![out_meta],
        )
    }

    pub fn sample(&mut self, logits: NodeId, kind: SamplerKind, seed: Option<u64>) -> NodeId {
        let out_meta = TensorMeta::new(Dtype::I32, &[1]);
        self.add_node(
            NodeKind::Op { op: OpKind::Sample, attrs: OpAttrs::Sample { kind, seed } },
            vec![logits],
            vec![out_meta],
        )
    }

    /// Reshape: change the tensor's shape without changing its element count.
    pub fn reshape(&mut self, x: NodeId, new_shape: &[usize]) -> NodeId {
        let in_meta = self.nodes[x.0 as usize].output_meta[0].clone();
        let in_numel: usize = in_meta.shape.iter().product();
        let out_numel: usize = new_shape.iter().product();
        assert_eq!(in_numel, out_numel,
            "reshape element count mismatch: {} vs {}", in_numel, out_numel);
        let out_meta = TensorMeta::new(in_meta.dtype, new_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::Reshape,
                attrs: OpAttrs::Reshape { new_shape: new_shape.to_vec() },
            },
            vec![x],
            vec![out_meta],
        )
    }

    /// Transpose: permute axes. `permutation[i]` is the source axis that
    /// becomes axis `i` in the output. Must be a valid permutation of
    /// `0..rank`.
    pub fn transpose(&mut self, x: NodeId, permutation: &[usize]) -> NodeId {
        let in_meta = self.nodes[x.0 as usize].output_meta[0].clone();
        assert_eq!(permutation.len(), in_meta.shape.len(),
            "transpose permutation length must equal tensor rank");
        let out_shape: Vec<usize> = permutation.iter().map(|&p| in_meta.shape[p]).collect();
        let out_meta = TensorMeta::new(in_meta.dtype, &out_shape);
        self.add_node(
            NodeKind::Op {
                op: OpKind::Transpose,
                attrs: OpAttrs::Transpose { permutation: permutation.to_vec() },
            },
            vec![x],
            vec![out_meta],
        )
    }

    /// Apply rotary position embedding. Input shape: `[..., seq, d]` where
    /// `d` is even (each consecutive pair is rotated). Position-axis is `-2`,
    /// rotation-axis is `-1`.
    pub fn rope(&mut self, x: NodeId, base: f32, position_offset: u32) -> NodeId {
        let in_meta = self.nodes[x.0 as usize].output_meta[0].clone();
        assert!(in_meta.shape.len() >= 2,
            "rope requires at least 2D input ([seq, d])");
        let d = *in_meta.shape.last().unwrap();
        assert!(d % 2 == 0, "rope rotation dim must be even, got {}", d);
        // Output shape == input shape.
        self.add_node(
            NodeKind::Op {
                op: OpKind::Rope,
                attrs: OpAttrs::Rope { base, position_offset },
            },
            vec![x],
            vec![in_meta],
        )
    }

    /// RoPE with a **runtime** position offset supplied as a second
    /// input (`pos`, an I32 `[1]` tensor). Identical to [`Self::rope`]
    /// except the per-token base position is read from `pos[0]` at
    /// execute time instead of baked into the op attrs at build time.
    /// This is what lets a single compiled decode graph be reused for
    /// every streaming step (the position is the only thing that
    /// changes token-to-token). The attr `position_offset` stays 0 and
    /// is unused when the extra input is present.
    pub fn rope_dyn(&mut self, x: NodeId, base: f32, pos: NodeId) -> NodeId {
        let in_meta = self.nodes[x.0 as usize].output_meta[0].clone();
        assert!(in_meta.shape.len() >= 2, "rope requires >=2D input");
        let d = *in_meta.shape.last().unwrap();
        assert!(d % 2 == 0, "rope rotation dim must be even, got {}", d);
        self.add_node(
            NodeKind::Op {
                op: OpKind::Rope,
                attrs: OpAttrs::Rope { base, position_offset: 0 },
            },
            vec![x, pos],
            vec![in_meta],
        )
    }

    /// In-place scatter with a **runtime** offset (`pos`, I32 `[1]`)
    /// read from `pos[0]` at execute time. Build-time can't bounds-
    /// check the dynamic offset, so it only asserts `src` fits in
    /// `dst` along the scatter axis at all; the runtime offset must
    /// keep `offset + src_len <= dst_len` (the decode loop guarantees
    /// this via `max_seq`).
    pub fn scatter_inplace_dyn(
        &mut self, dst: NodeId, src: NodeId, axis: i32, pos: NodeId,
    ) -> NodeId {
        let dst_meta = self.nodes[dst.0 as usize].output_meta[0].clone();
        let src_meta = &self.nodes[src.0 as usize].output_meta[0];
        let rank = dst_meta.shape.len();
        let axis_resolved = if axis < 0 { (rank as i32 + axis) as usize } else { axis as usize };
        assert!(axis_resolved < rank, "scatter axis out of range");
        assert_eq!(dst_meta.shape.len(), src_meta.shape.len(), "scatter rank mismatch");
        for d in 0..rank {
            if d != axis_resolved {
                assert_eq!(dst_meta.shape[d], src_meta.shape[d],
                    "scatter shape mismatch on non-scatter axis {}", d);
            }
        }
        assert!(src_meta.shape[axis_resolved] <= dst_meta.shape[axis_resolved],
            "scatter src longer than dst on scatter axis");
        let id = self.add_node(
            NodeKind::Op {
                op: OpKind::Scatter,
                attrs: OpAttrs::Scatter { axis, offset: 0 },
            },
            vec![dst, src, pos],
            vec![dst_meta],
        );
        self.nodes[id.0 as usize].aliases_input = Some(0);
        id
    }

    /// Causal softmax with a **runtime** causal offset (`pos`, I32
    /// `[1]`) read from `pos[0]`. Query relative position `q` attends
    /// keys `0..=q + pos[0]`. With full-buffer attention this also
    /// implicitly masks the not-yet-written tail of the KV buffer
    /// (those positions are `> q + cached_seq`).
    pub fn softmax_causal_offset_dyn(
        &mut self, x: NodeId, axis: i32, pos: NodeId,
    ) -> NodeId {
        let meta = self.nodes[x.0 as usize].output_meta[0].clone();
        self.add_node(
            NodeKind::Op {
                op: OpKind::Softmax,
                attrs: OpAttrs::Softmax { axis, causal: true, causal_offset: 0 },
            },
            vec![x, pos],
            vec![meta],
        )
    }

    pub fn build(self) -> Graph {
        Graph { nodes: self.nodes, inputs: self.inputs, outputs: self.outputs }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self { Self::new() }
}
