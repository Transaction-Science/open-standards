//! KV cache for autoregressive decoding.
//!
//! The KV cache holds persistent K and V tensors per transformer layer.
//! On each decode step:
//! 1. The new tokens are projected through W_q/W_k/W_v to get the new
//!    Q, K, V slices.
//! 2. RoPE is applied to Q and the new K (with the correct position offset).
//! 3. The new K and V are concatenated onto the cached K and V.
//! 4. Attention is computed: Q × cached_K^T, scaled, masked with the
//!    correct causal_offset (since query slice is shorter than key slice
//!    after concat), softmax, × cached_V, output projection, residual.
//!
//! This is a *functional* KV cache: each step rebuilds the full K and V
//! tensors via Concat. A future allocator-backed version will use
//! preallocated `[max_seq, d_head]` tensors with in-place writes; that
//! requires `LifetimeTier::Persistent` plumbing through the engine, which
//! is Phase 2 work.

use jouleclaw_core::tensor::Tensor;

/// Per-layer KV cache. Entries are populated incrementally during
/// decode; on prefill the entire prompt fills them in one shot.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Per-layer K cache. Shape: `[n_heads, current_seq, d_head]`.
    /// Empty (zero-length seq) before any prefill.
    pub k: Vec<Option<Tensor>>,
    /// Per-layer V cache. Same shape as k.
    pub v: Vec<Option<Tensor>>,
    /// Number of positions currently filled. Equal to `k[0].shape[1]`
    /// (and same for all layers); 0 before any prefill.
    pub current_seq: usize,
    /// Architectural constant: number of layers. Set at construction.
    pub n_layers: usize,
}

impl KvCache {
    /// Construct an empty cache for `n_layers`.
    pub fn empty(n_layers: usize) -> Self {
        Self {
            k: vec![None; n_layers],
            v: vec![None; n_layers],
            current_seq: 0,
            n_layers,
        }
    }

    /// Update cache for one layer: store the layer's full-context K and V.
    /// After this call, `current_seq` reflects the new length.
    ///
    /// `layer` is the block index (0..n_layers).
    /// `k`, `v` are tensors of shape `[n_heads, total_seq, d_head]` —
    /// these are the values produced by `concat(prev_K, new_K)` and
    /// similarly for V, after RoPE has been applied to the new K slice.
    pub fn put(&mut self, layer: usize, k: Tensor, v: Tensor) {
        assert!(layer < self.n_layers, "layer {} out of range", layer);
        // Verify shape consistency. K and V must be 3D `[n_heads, seq, d_head]`.
        assert_eq!(k.meta.shape.len(), 3, "K must be 3D");
        assert_eq!(v.meta.shape.len(), 3, "V must be 3D");
        let total_seq = k.meta.shape[1];
        assert_eq!(v.meta.shape[1], total_seq, "K and V must have same seq length");

        // After the first layer's update, all subsequent layers must agree
        // on current_seq (otherwise the graph is malformed).
        if layer == 0 {
            self.current_seq = total_seq;
        } else {
            assert_eq!(total_seq, self.current_seq,
                "layer {}: total_seq {} doesn't match current_seq {}",
                layer, total_seq, self.current_seq);
        }
        self.k[layer] = Some(k);
        self.v[layer] = Some(v);
    }

    /// Read the cached K for a layer, or None if unfilled.
    pub fn k_for(&self, layer: usize) -> Option<&Tensor> {
        self.k[layer].as_ref()
    }

    /// Read the cached V for a layer, or None if unfilled.
    pub fn v_for(&self, layer: usize) -> Option<&Tensor> {
        self.v[layer].as_ref()
    }

    /// Reset the cache for a new generation.
    pub fn reset(&mut self) {
        for slot in self.k.iter_mut() { *slot = None; }
        for slot in self.v.iter_mut() { *slot = None; }
        self.current_seq = 0;
    }
}
