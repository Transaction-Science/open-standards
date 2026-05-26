//! Operation primitives.
//!
//! Every transformer runtime converges on the same eight math operations.
//! Plus deterministic operations resolved without invoking the model.

use crate::backend::BackendId;
use crate::determinism::DeterminismClass;
use crate::energy::JouleEstimate;
use crate::error::TypeError;
use crate::tensor::TensorMeta;

/// The kind of operation. Discriminator for `Op` implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    // ---- The eight math primitives ----
    /// Matrix multiplication. `(A: [..,m,k], B: [..,k,n]) -> [..,m,n]`
    MatMul,
    /// `A @ W^T` where W is a ternary-packed weight (PrismML Q2_0 g128:
    /// {-1,0,+1} 2-bit codes + per-128-block f16 scale). A is dense f32
    /// `[..,m,k]`; W is the raw packed byte blob `[packed_len]` (U8);
    /// output is dense f32 `[..,m,out]`. The inner product is
    /// sign-select + accumulate (no per-weight FP multiply) — the
    /// energy/bandwidth win that makes ternary weights worthwhile.
    /// Logical `out`/`k` travel in `OpAttrs::MatMulTernary` because the
    /// packed operand's shape is byte-length, not `[out,k]`.
    MatMulTernary,
    /// `A @ W^T` where W is a 1-bit packed weight (PrismML Q1_0 g128:
    /// `bit==1 → +d`, `bit==0 → −d`, per-128-block f16 scale). Even
    /// strictly stronger than `MatMulTernary` for the thesis — every
    /// weight contribution is pure sign-flip + accumulate, no zero
    /// case to skip and no `q==3` outlier. Logical `out`/`k` in
    /// `OpAttrs::MatMulBit`.
    MatMulBit,
    /// `A @ W^T` where W is the Tencent/AngelSlim "Sherry" 3:4-sparse
    /// ternary packing (STQ1_0 g256): 256-elem / 42-byte blocks of
    /// `qs[32] + sign[8] + f16 d`. Decoded via a 32-entry codebook
    /// LUT; 3:4 sparsity guarantees exactly one of every four lanes
    /// is zero. Effective **1.3125 bpw** — the most aggressive
    /// packing in the substrate.
    MatMulSTQ1_0,
    /// `A @ W^T` where W is a Q8_0 packed weight (block of 32 elements,
    /// 1 fp16 scale per block, 8.5 bits/weight). The packed bytes
    /// stream through unchanged; the kernel quantizes A per-row to
    /// int8 + scale and runs a fused int8×int8→int32→fp32 dot
    /// product (NEON `vmull_s8` + `vpadalq_s16` on aarch64; scalar
    /// fallback elsewhere). Skips the fp32 dequant + sgemm round-trip
    /// that the standard `MatMul` path does for Q8_0-stored weights —
    /// a real win on edge targets where cblas_sgemm (AMX) isn't
    /// available.
    ///
    /// Logical `out`/`k` in `OpAttrs::MatMulQ80`.
    MatMulQ80,
    /// Softmax along last dim. `(X: [..,d]) -> [..,d]`
    Softmax,
    /// RMS or LayerNorm. `(X: [..,d], weight: [d]) -> [..,d]`
    Norm,
    /// Element-wise activation (SiLU, GELU, ReLU, Tanh).
    Activation,
    /// Element-wise add (residual connections).
    Add,
    /// Element-wise multiply (gating).
    Mul,
    /// Sparse indexing into a table. `(idx: [n], table: [V,d]) -> [n,d]`
    Lookup,
    /// Sparse indexing into a **ternary-packed** table (PrismML Q2_0
    /// g128). `(idx: [n], table: packed [V·rowbytes]) -> [n,d]` f32.
    /// Only the `n` requested rows are decoded — the full `[V,d]` table
    /// is never materialised as f32 (e.g. a 151669×2048 embedding is
    /// ~40 MB packed vs ~1.2 GB dequantised). Logical `v`/`d` travel in
    /// `OpAttrs::LookupTernary`.
    LookupTernary,
    /// Sparse indexing into a **1-bit packed** table (PrismML Q1_0 g128).
    /// Same on-demand decode semantics as `LookupTernary`, ~17 MB
    /// packed for Bonsai's 151669×2048 vs 1.2 GB dequantised.
    LookupBit,
    /// Categorical sampling from logits. `(logits: [V]) -> i32`
    Sample,

    // ---- Shape / layout primitives (Phase 1.5) ----
    /// Change tensor shape without changing data layout.
    /// Total element count must match.
    Reshape,
    /// Permute tensor axes. Physical reorder of memory.
    Transpose,
    /// Concatenate two tensors along a given axis. All other dims must match.
    Concat,
    /// Write a source tensor into a destination tensor at a given offset
    /// along a given axis. Conceptually `dst[..., offset:offset+src_len, ...] = src`.
    /// In the dataflow model this produces a new tensor; the runtime may
    /// elide the copy when the destination's lifetime allows in-place writes
    /// (see `Node::aliases_input` and `GraphBuilder::scatter_inplace`).
    Scatter,
    /// Extract a sub-region of a tensor along an axis. Conceptually
    /// `output = x[..., start:start+length, ...]`. The inverse of Scatter
    /// in the sense that `slice(scatter(zero, src, axis, offset), axis,
    /// offset, src.shape[axis]) == src`.
    Slice,
    /// Repeat (tile) a tensor along a given axis, producing
    /// `output[..., i*size..(i+1)*size, ...] = input[..., :, ...]` for each
    /// repeat index `i`. Used for GQA broadcast (K/V heads shared across Q
    /// groups) and other broadcast patterns.
    Repeat,

    // ---- Positional encoding primitives (Phase 1.7) ----
    /// Rotary position embedding (RoPE). Applied to Q and K in attention.
    Rope,
    /// Depthwise causal 1-D convolution. Used by LFM2's `shortconv`
    /// recurrent block — each channel is convolved independently with
    /// its own `taps`-long kernel, with left-padding by zeros so that
    /// `y[t]` depends only on `x[≤t]`. Inputs: `x` shape `[seq,
    /// d_model]` f32, `w` shape `[taps, d_model]` f32. Output: `y`
    /// shape `[seq, d_model]` f32. `taps` travels in
    /// `OpAttrs::Conv1DDepthwise`.
    Conv1DDepthwiseCausal,

    // ---- Deterministic primitives ----
    /// Tokenize text to token ids.
    Tokenize,
    /// Detokenize token ids back to text.
    Detokenize,
    /// Regular expression match.
    Regex,
    /// Structured parse.
    Parse,
    /// Vector-store retrieval.
    Retrieve,
    /// Template slot-filling.
    TemplateFill,
    /// Content-addressable cache lookup.
    CacheLookup,
    /// Sandboxed code execution.
    Execute,
}

impl OpKind {
    /// Whether this op is intrinsically deterministic, regardless of impl.
    pub fn intrinsic_determinism(self) -> DeterminismClass {
        match self {
            // Math primitives are deterministic given fixed reduction order
            // and deterministic kernels.
            Self::MatMul
            | Self::MatMulTernary
            | Self::MatMulBit
            | Self::MatMulSTQ1_0
            | Self::MatMulQ80
            | Self::LookupTernary
            | Self::LookupBit
            | Self::Conv1DDepthwiseCausal
            | Self::Softmax
            | Self::Norm
            | Self::Activation
            | Self::Add
            | Self::Mul
            | Self::Lookup => DeterminismClass::Deterministic,

            // Sample is the only stochastic math primitive, and it requires
            // a seed in deterministic mode.
            Self::Sample => DeterminismClass::SeededStochastic,

            // Shape primitives are pure data movement.
            Self::Reshape | Self::Transpose | Self::Concat | Self::Repeat | Self::Scatter
            | Self::Slice
                => DeterminismClass::Deterministic,

            // Positional encoding is deterministic given fixed base and offset.
            Self::Rope => DeterminismClass::Deterministic,

            // Deterministic primitives are deterministic by construction.
            Self::Tokenize
            | Self::Detokenize
            | Self::Regex
            | Self::Parse
            | Self::Retrieve
            | Self::TemplateFill
            | Self::CacheLookup => DeterminismClass::Deterministic,

            // Execute calls into a sandbox; non-deterministic by default.
            Self::Execute => DeterminismClass::Stochastic,
        }
    }
}

/// Per-operation attributes (op-kind-specific).
///
/// Phase 0 uses a discriminated enum; Phase 2+ may switch to a more compact
/// flat representation if profiling shows it matters.
#[derive(Debug, Clone)]
pub enum OpAttrs {
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
        /// Output scale factor: `C = alpha * A * B`. Default 1.0.
        /// Used for attention's `1/sqrt(d_head)` score scaling.
        alpha: f32,
        /// Optional logical truncation of operand B's "N" axis (the axis
        /// that becomes the output's last dim).
        ///
        /// When `Some(n_valid)`, the kernel uses only the first `n_valid`
        /// positions along B's N axis. Used for in-place KV cache:
        /// the K buffer is preallocated at `[heads, max_seq, d_head]` but
        /// only the first `valid_seq` positions are meaningful; setting
        /// `b_n_valid = Some(valid_seq)` makes `matmul_bt(Q, K)` produce
        /// output `[heads, new_seq, valid_seq]` instead of `[heads,
        /// new_seq, max_seq]`.
        ///
        /// When `None`, B is used in full. Default.
        b_n_valid: Option<usize>,
    },
    /// Attributes for [`OpKind::MatMulTernary`]. The packed weight
    /// operand carries its bytes (not `[out,k]`), so the logical
    /// dimensions are recorded here at graph-build time from the GGUF
    /// tensor info. Weight `W` is `[out, k]` row-major ternary; the op
    /// computes `Y[..,m,o] = Σ_k A[..,m,k] · W[o,k]`.
    MatMulTernary {
        /// Output feature count (rows of W).
        out: usize,
        /// Reduction dim (cols of W, must equal A's last dim).
        k: usize,
    },
    /// Attributes for [`OpKind::MatMulBit`]. Shares the structure of
    /// `MatMulTernary` — the packed operand carries bytes, dims are
    /// recorded here at graph-build time.
    MatMulBit { out: usize, k: usize },
    /// Attributes for [`OpKind::MatMulSTQ1_0`]. `k` must be a multiple
    /// of the STQ1_0 block size (256).
    MatMulSTQ1_0 { out: usize, k: usize },
    /// Attributes for [`OpKind::MatMulQ80`]. `k` must be a multiple
    /// of the Q8_0 block size (32). The packed weight operand
    /// carries `n * (k/32) * 34` bytes, so logical `out`/`k` are
    /// recorded here at graph-build time.
    MatMulQ80 { out: usize, k: usize },
    /// Attributes for [`OpKind::LookupTernary`]. The packed table
    /// operand carries its bytes, so the logical table dimensions are
    /// recorded here at graph-build time.
    LookupTernary {
        /// Vocabulary / table row count.
        v: usize,
        /// Embedding width (decoded f32 columns per row).
        d: usize,
    },
    /// Attributes for [`OpKind::LookupBit`].
    LookupBit { v: usize, d: usize },
    /// Attributes for [`OpKind::Conv1DDepthwiseCausal`].
    Conv1DDepthwise {
        /// Filter length (kernel taps), e.g. 3 for LFM2 shortconv.
        taps: usize,
    },
    Softmax {
        axis: i32,
        /// If true, apply a causal (upper-triangular) mask before softmax:
        /// `scores[..., i, j] = -inf for j > i + causal_offset`.
        ///
        /// Required for autoregressive attention. Only meaningful when
        /// the key axis (last dim) is at least `seq_q + causal_offset`.
        ///
        /// During prefill (`new_seq == total_seq`), set `causal_offset = 0`,
        /// so position `i` attends to keys 0..=i. During decode with KV
        /// cache (`new_seq < total_seq`), `causal_offset = total_seq -
        /// new_seq`, so the first new query (relative pos 0) attends to
        /// all `total_seq` past keys, the second to `total_seq - new_seq + 1`,
        /// etc.
        causal: bool,
        causal_offset: i32,
    },
    Norm {
        kind: NormKind,
        eps: f32,
    },
    Activation {
        kind: ActivationKind,
    },
    Add,
    Mul,
    Lookup {
        // Optional: scale by 1/sqrt(d) for embedding lookups (RoPE-style).
        scale_by_inv_sqrt_d: bool,
    },
    Sample {
        kind: SamplerKind,
        seed: Option<u64>,
    },

    // Shape / layout primitives.
    Reshape {
        /// New shape. Total element count must match input.
        new_shape: Vec<usize>,
    },
    Transpose {
        /// Permutation of axes. `permutation[i]` is the source axis that
        /// becomes axis `i` in the output. Must be a valid permutation of
        /// `0..rank`.
        permutation: Vec<usize>,
    },

    /// Concatenate two tensors along a given axis. Both must have the same
    /// rank and matching shapes on all axes except `axis`.
    Concat {
        axis: i32,
    },

    /// Repeat (tile) a tensor along an axis. The axis's size is multiplied
    /// by `repeats`; new output[..., i*N + j, ...] = input[..., j, ...]
    /// where N is the input's size on that axis and 0 <= i < repeats.
    /// Used for GQA head broadcasting.
    Repeat {
        axis: i32,
        repeats: usize,
    },

    /// Write `src` into `dst` at a given offset along an axis.
    /// `dst.shape[axis]` must satisfy `offset + src.shape[axis] <= dst.shape[axis]`.
    /// All other dims must match.
    Scatter {
        axis: i32,
        offset: usize,
    },

    /// Extract a contiguous slice along an axis. Output shape matches input
    /// except `axis` is reduced from `dim` to `length`.
    Slice {
        axis: i32,
        start: usize,
        length: usize,
    },

    /// Rotary position embedding. Applied to Q or K projections in attention.
    /// Rotates pairs `(x[2i], x[2i+1])` by an angle that depends on
    /// position and pair index.
    Rope {
        /// Base for the geometric series of frequencies. Llama uses 10000.0;
        /// Llama 3 uses 500000.0; some long-context models use larger.
        base: f32,
        /// Position of the first token in the input sequence. For prefill
        /// always 0; for decoding equals the current step.
        position_offset: u32,
    },

    // Deterministic primitive attrs intentionally minimal in Phase 0.
    Tokenize {
        tokenizer_id: u32,
    },
    Detokenize {
        tokenizer_id: u32,
    },
    Regex {
        pattern_id: u32,
    },
    Parse {
        grammar_id: u32,
    },
    Retrieve {
        store_id: u32,
        top_k: u32,
    },
    TemplateFill {
        template_id: u32,
    },
    CacheLookup {
        cache_id: u32,
    },
    Execute {
        sandbox_id: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    Layer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    SiLU,
    GELU,
    ReLU,
    Tanh,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    Greedy,
    TopK { k: u32 },
    TopP { p_thousandths: u32 }, // Use millionths to keep integer-deterministic.
    Temperature { temp_thousandths: u32 },
}

/// The contract every op implementation must satisfy.
pub trait Op: Send + Sync {
    /// Stable identifier for this operation kind.
    fn kind(&self) -> OpKind;

    /// Attributes (op-kind-specific configuration).
    fn attrs(&self) -> &OpAttrs;

    /// Determinism class of this op as configured.
    /// May be more conservative than `OpKind::intrinsic_determinism`
    /// (e.g., a `Sample` op without a seed is `Stochastic`).
    fn determinism(&self) -> DeterminismClass {
        match (self.kind().intrinsic_determinism(), self.attrs()) {
            (DeterminismClass::SeededStochastic, OpAttrs::Sample { seed: None, .. }) => {
                DeterminismClass::Stochastic
            }
            (cls, _) => cls,
        }
    }

    /// Backends that have a kernel registered for this op.
    fn supported_backends(&self) -> &[BackendId];

    /// Estimate joules to execute given input shapes/types and a target backend.
    fn estimate_joules(
        &self,
        inputs: &[TensorMeta],
        backend: BackendId,
    ) -> JouleEstimate;

    /// Type-check inputs and produce output metadata.
    /// Errors if inputs do not match this op's signature.
    fn signature(&self, inputs: &[TensorMeta]) -> Result<Vec<TensorMeta>, TypeError>;
}
