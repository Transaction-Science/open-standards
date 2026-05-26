//! In-place KV cache for autoregressive decoding.
//!
//! Unlike `KvCache` (which holds variable-length K/V tensors that grow via
//! Concat each step), `InPlaceKvCache` preallocates `[n_heads_kv, max_seq,
//! d_head]` buffers per layer at construction time. Each decode step
//! scatters the new K and V at the current position; the cache buffer's
//! shape stays constant.
//!
//! This is the design real production runtimes use. Memory traffic per
//! decode step drops from O(seq_len) (Concat rebuild) to O(new_seq)
//! (Scatter write).
//!
//! The Scatter operation is still functionally pure in the graph (produces
//! a new tensor); a future allocator with persistent-tensor lifetime hooks
//! will elide the copy and write directly into the host-provided buffer.
//! For correctness today, the host can choose to overwrite its own buffer
//! with the kernel's output and pass the same buffer back next step.
//!
//! ## KV cache quantization
//!
//! Opt-in via [`KvQuant::Int8`] at construction time. The cache stores
//! K/V as int8 with one fp32 scale per `(head, position)` row. The
//! forward pass still sees fp32 tensors — quantize/dequantize happens
//! transparently at the `take_buffers` / `replace_buffers` boundary.
//! 4× cold-storage savings; the working set during a forward is
//! unchanged (the executor still gets a fresh fp32 buffer). Per-token
//! cost: O(n_heads_kv × max_seq × d_head) for each of quant and
//! dequant — sub-millisecond on Apple Silicon for typical sizes.
//!
//! Per the edge-architecture notes survey, this is "format-agnostic;
//! both formats can do it equally well" — the cache is downstream of
//! the model format, and the quant scheme is symmetric per-row int8
//! (≈ what llama.cpp's `--cache-type-k q8_0 --cache-type-v q8_0`
//! delivers).

use crate::decode::{
    build_decode_step_graph_inplace_block,
    embed_constant_pub,
};
use crate::llama::{LlamaConfig, LoadError};
use crate::GgufModel;
use jouleclaw_core::graph::{Graph, GraphBuilder};
use jouleclaw_core::op::NormKind;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};

/// KV cache storage precision. `None` is the historical fp32 path;
/// `Int8` halves-twice the per-position memory at the cost of a fast
/// per-step quant + dequant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuant {
    /// Store K/V as F32 (current default). No quant overhead; full
    /// numerical precision; 4 bytes per element.
    None,
    /// Store K/V as I8 with one fp32 scale per `(head, position)`
    /// row. 1 byte per element + 4 bytes per row of scales ≈ 4×
    /// cold-storage savings.
    Int8,
}

impl KvQuant {
    fn bytes_per_elem(self) -> usize {
        match self {
            Self::None => 4,
            Self::Int8 => 1,
        }
    }
}

/// Per-layer rolling-window state for LFM2 shortconv (recurrent)
/// layers. Each recurrent layer remembers the previous `taps - 1`
/// tokens of its gated input (`bx`) so streaming decode produces
/// the same output as a fresh prefill over the full history.
///
/// Attention layers in LFM2's hybrid arch have empty (None) slots
/// here; KV cache lives separately in [`InPlaceKvCache`].
///
/// For LFM2, `taps = 3` (per the `shortconv.l_cache` metadata),
/// so each recurrent layer's state is shape `[2, embedding_length]`
/// f32 — tiny compared to KV cache (and constant per layer, doesn't
/// grow with sequence length).
#[derive(Debug, Clone)]
pub struct ShortConvStateCache {
    /// One slot per layer. `Some(tensor)` for shortconv layers
    /// (shape `[taps - 1, embedding_length]`, f32, zero on init).
    /// `None` for attention layers — they don't need it.
    pub states: Vec<Option<Tensor>>,
    /// `taps - 1` (window size minus the current-token slot).
    pub window: usize,
    /// `embedding_length` — the d dim each state row has.
    pub d: usize,
}

impl ShortConvStateCache {
    /// Build a cache sized to an LFM2 model. For non-LFM2 archs
    /// (no shortconv layers), `states` is all-None and the cache is
    /// effectively unused.
    pub fn for_model(model: &GgufModel) -> Result<Self, LoadError> {
        let config = LlamaConfig::from_metadata(model)?;
        let d = config.embedding_length;
        let taps = config.shortconv_l_cache.max(3);
        let window = taps.saturating_sub(1);
        let n_layers = config.block_count;

        let zero_bytes: Vec<u8> = vec![0u8; window * d * 4];
        let mut states: Vec<Option<Tensor>> = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let is_recurrent = config.arch == "lfm2"
                && config.per_layer_head_count_kv.get(layer).copied() == Some(0);
            let slot = if is_recurrent {
                Some(Tensor {
                    meta: TensorMeta::new(Dtype::F32, &[window, d]),
                    storage: std::sync::Arc::new(TensorStorage { bytes: zero_bytes.clone(), mapped: None }),
                })
            } else {
                None
            };
            states.push(slot);
        }
        Ok(Self { states, window, d })
    }

    /// Move the state buffer for a layer OUT, leaving a zero-byte
    /// placeholder. Mirrors `InPlaceKvCache::take_buffers` so the
    /// executor's `scatter_inplace`-style aliasing can fire when
    /// the input/output Arcs are strong_count == 1.
    ///
    /// Returns `None` for attention layers (which don't have state).
    pub fn take_state(&mut self, layer: usize) -> Option<Tensor> {
        let slot = self.states.get_mut(layer)?;
        slot.take().map(|t| {
            // Leave a zero-byte placeholder marker — replace_state
            // will put a real tensor back.
            // We don't put a placeholder Some; the slot is None
            // between take/replace.
            t
        })
    }

    /// Put the updated state back for a layer. Caller's responsibility
    /// to call this after each decode step for every layer that had
    /// `take_state` return `Some`.
    pub fn replace_state(&mut self, layer: usize, state: Tensor) {
        if let Some(slot) = self.states.get_mut(layer) {
            *slot = Some(state);
        }
    }

    /// Reset every shortconv state to zeros. Same shape, just clears
    /// the rolling window.
    pub fn reset(&mut self) {
        let zero_bytes: Vec<u8> = vec![0u8; self.window * self.d * 4];
        for slot in self.states.iter_mut() {
            if slot.is_some() {
                *slot = Some(Tensor {
                    meta: TensorMeta::new(Dtype::F32, &[self.window, self.d]),
                    storage: std::sync::Arc::new(TensorStorage { bytes: zero_bytes.clone(), mapped: None }),
                });
            }
        }
    }

    /// Total resident bytes across all shortconv layers.
    pub fn cache_bytes(&self) -> usize {
        self.states.iter()
            .filter(|s| s.is_some())
            .count() * self.window * self.d * 4
    }

    pub fn n_recurrent_layers(&self) -> usize {
        self.states.iter().filter(|s| s.is_some()).count()
    }

    /// Capture the full shortconv state as owned bytes, suitable for
    /// stuffing into a [`PrefixCache`]. The returned snapshot is
    /// immutable — restoring it copies bytes into a fresh cache so the
    /// snapshot can be shared across many resumed conversations.
    pub fn snapshot(&self) -> ShortConvStateSnapshot {
        let states: Vec<Option<Vec<u8>>> = self.states.iter()
            .map(|slot| slot.as_ref().map(|t| t.storage.view_bytes().to_vec()))
            .collect();
        ShortConvStateSnapshot { states, window: self.window, d: self.d }
    }

    /// Restore a cache from a snapshot. Layer count, window size, and
    /// d-dim must match — a snapshot is only meaningful for the model
    /// that produced it.
    pub fn from_snapshot(snap: &ShortConvStateSnapshot) -> Self {
        let states: Vec<Option<Tensor>> = snap.states.iter().map(|opt| {
            opt.as_ref().map(|bytes| Tensor {
                meta: TensorMeta::new(Dtype::F32, &[snap.window, snap.d]),
                storage: std::sync::Arc::new(TensorStorage::from_bytes(bytes.clone())),
            })
        }).collect();
        Self { states, window: snap.window, d: snap.d }
    }
}

/// Immutable snapshot of a [`ShortConvStateCache`]. Cheaply `Clone`-able
/// (no buffer duplication: the inner `Vec<u8>` is plain owned bytes that
/// users typically wrap in `Arc<ShortConvStateSnapshot>` when sharing).
#[derive(Debug, Clone)]
pub struct ShortConvStateSnapshot {
    /// One slot per layer; `Some(bytes)` for shortconv layers, `None`
    /// for attention layers. `bytes.len() == window * d * 4`.
    pub states: Vec<Option<Vec<u8>>>,
    pub window: usize,
    pub d: usize,
}

/// Immutable snapshot of an [`InPlaceKvCache`] — everything needed to
/// reconstruct an identical cache later. Used by prefix-caching to
/// resume a conversation at a previously-prefilled point without
/// re-running prefill on the shared prefix tokens.
///
/// The K/V buffers are full [n_heads_kv, max_seq, d_head] arrays,
/// matching the original cache shape. Positions beyond `current_seq`
/// are zero-padded buffer space (same invariant the live cache
/// maintains). When restored, the cache reads the entire buffer for
/// attention but masks tail positions via `softmax_causal_offset_dyn`,
/// so the zero pad never leaks into logits.
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    pub n_layers: usize,
    pub n_heads_kv: usize,
    pub max_seq: usize,
    pub d_head: usize,
    /// Number of positions that hold real K/V data (0..current_seq).
    pub current_seq: usize,
    pub quant: KvQuant,
    /// Per-layer K buffer bytes. For `KvQuant::None`, fp32 little-endian;
    /// for `KvQuant::Int8`, raw i8. Length = `n_heads_kv * max_seq *
    /// d_head * quant.bytes_per_elem()` per layer.
    pub k_bufs: Vec<Vec<u8>>,
    pub v_bufs: Vec<Vec<u8>>,
    /// Per-layer K scales (one fp32 per `(head, position)` row).
    /// Empty when `quant == KvQuant::None`.
    pub k_scales: Vec<Vec<f32>>,
    pub v_scales: Vec<Vec<f32>>,
}

impl KvSnapshot {
    /// Resident-bytes accounting — used by `PrefixCache` for eviction.
    pub fn bytes(&self) -> usize {
        let k_v: usize = self.k_bufs.iter().chain(self.v_bufs.iter())
            .map(|b| b.len()).sum();
        let scales: usize = (self.k_scales.iter().chain(self.v_scales.iter()))
            .map(|s| s.len() * 4).sum();
        k_v + scales
    }
}

/// Per-layer preallocated KV cache.
#[derive(Debug, Clone)]
pub struct InPlaceKvCache {
    /// Per-layer K buffer. Shape `[n_heads_kv, max_seq, d_head]`.
    /// Dtype is F32 when `quant == KvQuant::None`, I8 when `Int8`.
    pub k_bufs: Vec<Tensor>,
    /// Per-layer V buffer. Same shape/dtype rules as `k_bufs`.
    pub v_bufs: Vec<Tensor>,
    /// Per-layer K scales — one fp32 per `(head, position)` row, so
    /// length is `n_heads_kv * max_seq`. Empty when `quant ==
    /// KvQuant::None`.
    pub k_scales: Vec<Vec<f32>>,
    /// Per-layer V scales — same shape rules as `k_scales`.
    pub v_scales: Vec<Vec<f32>>,
    /// Number of positions currently filled. 0 before any step.
    pub current_seq: usize,
    /// Architectural constants.
    pub n_layers: usize,
    pub n_heads_kv: usize,
    pub max_seq: usize,
    pub d_head: usize,
    /// Storage precision (fp32 vs int8).
    pub quant: KvQuant,
}

impl InPlaceKvCache {
    /// Construct an empty fp32 cache with preallocated buffers — the
    /// historical no-quant path.
    pub fn new(n_layers: usize, n_heads_kv: usize, max_seq: usize, d_head: usize) -> Self {
        Self::new_with_quant(n_layers, n_heads_kv, max_seq, d_head, KvQuant::None)
    }

    /// Construct an empty cache with the chosen quant scheme.
    pub fn new_with_quant(
        n_layers: usize,
        n_heads_kv: usize,
        max_seq: usize,
        d_head: usize,
        quant: KvQuant,
    ) -> Self {
        let buf_elems = n_heads_kv * max_seq * d_head;
        let (k_bufs, v_bufs, k_scales, v_scales) = match quant {
            KvQuant::None => {
                let zero_bytes: Vec<u8> = vec![0u8; buf_elems * 4];
                let make_buf = || Tensor {
                    meta: TensorMeta::new(Dtype::F32, &[n_heads_kv, max_seq, d_head]),
                    storage: std::sync::Arc::new(TensorStorage { bytes: zero_bytes.clone(), mapped: None }),
                };
                (
                    (0..n_layers).map(|_| make_buf()).collect(),
                    (0..n_layers).map(|_| make_buf()).collect(),
                    vec![Vec::new(); n_layers],
                    vec![Vec::new(); n_layers],
                )
            }
            KvQuant::Int8 => {
                let zero_bytes: Vec<u8> = vec![0u8; buf_elems];
                let make_buf = || Tensor {
                    meta: TensorMeta::new(Dtype::I8, &[n_heads_kv, max_seq, d_head]),
                    storage: std::sync::Arc::new(TensorStorage { bytes: zero_bytes.clone(), mapped: None }),
                };
                let zero_scales = vec![0.0_f32; n_heads_kv * max_seq];
                (
                    (0..n_layers).map(|_| make_buf()).collect(),
                    (0..n_layers).map(|_| make_buf()).collect(),
                    vec![zero_scales.clone(); n_layers],
                    vec![zero_scales.clone(); n_layers],
                )
            }
        };

        Self {
            k_bufs, v_bufs, k_scales, v_scales,
            current_seq: 0,
            n_layers, n_heads_kv, max_seq, d_head, quant,
        }
    }

    /// Construct a cache sized to a model's configuration.
    pub fn for_model(model: &GgufModel, max_seq: usize) -> Result<Self, LoadError> {
        Self::for_model_with_quant(model, max_seq, KvQuant::None)
    }

    /// Like [`Self::for_model`] but with an explicit quant scheme.
    pub fn for_model_with_quant(
        model: &GgufModel,
        max_seq: usize,
        quant: KvQuant,
    ) -> Result<Self, LoadError> {
        let config = LlamaConfig::from_metadata(model)?;
        let n_heads_kv = config.head_count_kv.max(1);
        let d_head = config.head_dim;
        Ok(Self::new_with_quant(config.block_count, n_heads_kv, max_seq, d_head, quant))
    }

    /// Capture the entire cache as a [`KvSnapshot`] suitable for
    /// stashing in a prefix-cache. The K/V buffers are copied into
    /// owned `Vec<u8>` bytes so the snapshot is independent of any
    /// future mutation here. Includes the int8 scales when relevant.
    pub fn snapshot(&self) -> KvSnapshot {
        let k_bufs: Vec<Vec<u8>> = self.k_bufs.iter()
            .map(|t| t.storage.view_bytes().to_vec()).collect();
        let v_bufs: Vec<Vec<u8>> = self.v_bufs.iter()
            .map(|t| t.storage.view_bytes().to_vec()).collect();
        KvSnapshot {
            n_layers: self.n_layers,
            n_heads_kv: self.n_heads_kv,
            max_seq: self.max_seq,
            d_head: self.d_head,
            current_seq: self.current_seq,
            quant: self.quant,
            k_bufs, v_bufs,
            k_scales: self.k_scales.clone(),
            v_scales: self.v_scales.clone(),
        }
    }

    /// Restore a cache from a snapshot. The returned cache holds
    /// independent (owned) bytes so the original snapshot may be
    /// reused for other resumed conversations.
    pub fn from_snapshot(snap: &KvSnapshot) -> Self {
        let dtype = match snap.quant {
            KvQuant::None => Dtype::F32,
            KvQuant::Int8 => Dtype::I8,
        };
        let shape = [snap.n_heads_kv, snap.max_seq, snap.d_head];
        let make_buf = |bytes: &Vec<u8>| Tensor {
            meta: TensorMeta::new(dtype, &shape),
            storage: std::sync::Arc::new(TensorStorage::from_bytes(bytes.clone())),
        };
        let k_bufs = snap.k_bufs.iter().map(make_buf).collect();
        let v_bufs = snap.v_bufs.iter().map(make_buf).collect();
        Self {
            k_bufs, v_bufs,
            k_scales: snap.k_scales.clone(),
            v_scales: snap.v_scales.clone(),
            current_seq: snap.current_seq,
            n_layers: snap.n_layers,
            n_heads_kv: snap.n_heads_kv,
            max_seq: snap.max_seq,
            d_head: snap.d_head,
            quant: snap.quant,
        }
    }

    /// Cold-storage byte footprint of the K + V buffers across all
    /// layers. Doesn't count the working-set memory during a forward
    /// (which is a fresh per-layer fp32 allocation regardless of
    /// quant). Useful for benchmarks demonstrating quant savings.
    pub fn cache_bytes(&self) -> usize {
        let per_layer_buf = self.n_heads_kv * self.max_seq * self.d_head * self.quant.bytes_per_elem();
        let per_layer_scales = match self.quant {
            KvQuant::None => 0,
            KvQuant::Int8 => self.n_heads_kv * self.max_seq * 4, // f32 scales
        };
        // K + V per layer.
        self.n_layers * 2 * (per_layer_buf + per_layer_scales)
    }

    /// Update K/V buffer for a layer with the new scattered version.
    /// `k` and `v` must have the same shape as the existing buffers.
    pub fn put(&mut self, layer: usize, k: Tensor, v: Tensor) {
        assert_eq!(k.meta.shape, self.k_bufs[layer].meta.shape,
            "in-place K must keep the buffer shape");
        assert_eq!(v.meta.shape, self.v_bufs[layer].meta.shape,
            "in-place V must keep the buffer shape");
        self.k_bufs[layer] = k;
        self.v_bufs[layer] = v;
    }

    /// Hand the K and V buffers for the given layer to a forward pass
    /// as **fp32 tensors**, whether or not the cache is quantized:
    ///
    /// - `KvQuant::None`: same dance as before — move the cache's
    ///   fp32 Tensors out, leaving zero-byte placeholders. The
    ///   executor sees `strong_count == 1` storage and can alias-
    ///   write its output into it.
    /// - `KvQuant::Int8`: dequantize the cached i8 buffers into
    ///   fresh fp32 Tensors. The int8 storage stays in place; the
    ///   executor gets clean strong_count == 1 fp32 tensors that
    ///   it owns. No aliasing optimization is available on this
    ///   path (the int8 cache and fp32 working buffer are different
    ///   allocations), but the dequant cost is sub-ms on Apple
    ///   Silicon for typical sizes.
    pub fn take_buffers(&mut self, layer: usize) -> (Tensor, Tensor) {
        match self.quant {
            KvQuant::None => {
                let placeholder = Tensor {
                    meta: TensorMeta::new(Dtype::F32, &[0]),
                    storage: std::sync::Arc::new(TensorStorage { bytes: Vec::new(), mapped: None }),
                };
                let k = std::mem::replace(&mut self.k_bufs[layer], placeholder.clone());
                let v = std::mem::replace(&mut self.v_bufs[layer], placeholder);
                (k, v)
            }
            KvQuant::Int8 => {
                let k_fp32 = dequantize_per_row(
                    &self.k_bufs[layer].storage.bytes,
                    &self.k_scales[layer],
                    self.n_heads_kv * self.max_seq,
                    self.d_head,
                );
                let v_fp32 = dequantize_per_row(
                    &self.v_bufs[layer].storage.bytes,
                    &self.v_scales[layer],
                    self.n_heads_kv * self.max_seq,
                    self.d_head,
                );
                let make_fp32 = |bytes: Vec<u8>| Tensor {
                    meta: TensorMeta::new(
                        Dtype::F32,
                        &[self.n_heads_kv, self.max_seq, self.d_head],
                    ),
                    storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
                };
                (make_fp32(k_fp32), make_fp32(v_fp32))
            }
        }
    }

    /// Put the step's K and V outputs back into the cache. Quantizes
    /// transparently when `quant == KvQuant::Int8`.
    pub fn replace_buffers(&mut self, layer: usize, k: Tensor, v: Tensor) {
        match self.quant {
            KvQuant::None => {
                self.k_bufs[layer] = k;
                self.v_bufs[layer] = v;
            }
            KvQuant::Int8 => {
                let (k_q, k_s) = quantize_per_row(
                    &k.storage.bytes,
                    self.n_heads_kv * self.max_seq,
                    self.d_head,
                );
                let (v_q, v_s) = quantize_per_row(
                    &v.storage.bytes,
                    self.n_heads_kv * self.max_seq,
                    self.d_head,
                );
                self.k_bufs[layer] = Tensor {
                    meta: TensorMeta::new(
                        Dtype::I8,
                        &[self.n_heads_kv, self.max_seq, self.d_head],
                    ),
                    storage: std::sync::Arc::new(TensorStorage { bytes: k_q, mapped: None }),
                };
                self.v_bufs[layer] = Tensor {
                    meta: TensorMeta::new(
                        Dtype::I8,
                        &[self.n_heads_kv, self.max_seq, self.d_head],
                    ),
                    storage: std::sync::Arc::new(TensorStorage { bytes: v_q, mapped: None }),
                };
                self.k_scales[layer] = k_s;
                self.v_scales[layer] = v_s;
            }
        }
    }

    /// Set the new `current_seq` after a step. Caller advances by `new_seq`
    /// once per step.
    pub fn advance(&mut self, new_seq: usize) {
        self.current_seq += new_seq;
        assert!(self.current_seq <= self.max_seq,
            "in-place cache overflow: current_seq {} > max_seq {}",
            self.current_seq, self.max_seq);
    }

    pub fn reset(&mut self) {
        let buf_elems = self.n_heads_kv * self.max_seq * self.d_head;
        match self.quant {
            KvQuant::None => {
                let zero_bytes: Vec<u8> = vec![0u8; buf_elems * 4];
                let make_buf = || Tensor {
                    meta: TensorMeta::new(Dtype::F32, &[self.n_heads_kv, self.max_seq, self.d_head]),
                    storage: std::sync::Arc::new(TensorStorage { bytes: zero_bytes.clone(), mapped: None }),
                };
                for slot in self.k_bufs.iter_mut() { *slot = make_buf(); }
                for slot in self.v_bufs.iter_mut() { *slot = make_buf(); }
            }
            KvQuant::Int8 => {
                let zero_bytes: Vec<u8> = vec![0u8; buf_elems];
                let make_buf = || Tensor {
                    meta: TensorMeta::new(Dtype::I8, &[self.n_heads_kv, self.max_seq, self.d_head]),
                    storage: std::sync::Arc::new(TensorStorage { bytes: zero_bytes.clone(), mapped: None }),
                };
                let zero_scales = vec![0.0_f32; self.n_heads_kv * self.max_seq];
                for slot in self.k_bufs.iter_mut() { *slot = make_buf(); }
                for slot in self.v_bufs.iter_mut() { *slot = make_buf(); }
                for s in self.k_scales.iter_mut() { *s = zero_scales.clone(); }
                for s in self.v_scales.iter_mut() { *s = zero_scales.clone(); }
            }
        }
        self.current_seq = 0;
    }
}

// ── per-row symmetric int8 quant / dequant ─────────────────────────
// `n_rows = n_heads_kv * max_seq`, `d_head` = row length. One fp32
// scale per row. Mirrors `joule_deberta::int8::quantize_activation_per_row`
// but lives here to avoid the cache crate depending on joule-deberta.

fn quantize_per_row(fp32_bytes: &[u8], n_rows: usize, d_head: usize) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(fp32_bytes.len(), n_rows * d_head * 4);

    // Reinterpret the fp32 byte buffer as &[f32]. Vec<u8> from the
    // global allocator is 8-byte-aligned for our buffer sizes (KB),
    // and `vld1q_f32` on aarch64 needs only 4-byte alignment.
    let fp32: &[f32] = unsafe {
        std::slice::from_raw_parts(fp32_bytes.as_ptr() as *const f32, n_rows * d_head)
    };

    let mut q = vec![0_i8; n_rows * d_head];
    let mut scales = vec![0.0_f32; n_rows];

    for r in 0..n_rows {
        let row = &fp32[r * d_head..(r + 1) * d_head];
        let q_row = &mut q[r * d_head..(r + 1) * d_head];
        scales[r] = quantize_row(row, q_row);
    }

    // Reinterpret Vec<i8> → Vec<u8>. Same byte pattern, zero copy.
    let q_u8 = unsafe {
        let len = q.len();
        let cap = q.capacity();
        let ptr = q.as_mut_ptr();
        std::mem::forget(q);
        Vec::from_raw_parts(ptr as *mut u8, len, cap)
    };
    (q_u8, scales)
}

fn dequantize_per_row(q_bytes: &[u8], scales: &[f32], n_rows: usize, d_head: usize) -> Vec<u8> {
    debug_assert_eq!(q_bytes.len(), n_rows * d_head);
    debug_assert_eq!(scales.len(), n_rows);

    let mut out = vec![0.0_f32; n_rows * d_head];
    for r in 0..n_rows {
        let q_row = unsafe {
            std::slice::from_raw_parts(
                q_bytes.as_ptr().add(r * d_head) as *const i8,
                d_head,
            )
        };
        let out_row = &mut out[r * d_head..(r + 1) * d_head];
        dequantize_row(q_row, scales[r], out_row);
    }

    // Reinterpret Vec<f32> → Vec<u8>. Apple Silicon, x86_64, aarch64-
    // linux, and Android arm64 are all LE; the bytes match what
    // `f32::to_le_bytes` would produce. Big-endian targets aren't
    // supported by GGUF anyway.
    let mut out_v = std::mem::ManuallyDrop::new(out);
    unsafe {
        let len = out_v.len() * 4;
        let cap = out_v.capacity() * 4;
        let ptr = out_v.as_mut_ptr() as *mut u8;
        Vec::from_raw_parts(ptr, len, cap)
    }
}

// ── Per-row inner kernels ──────────────────────────────────────────────
//
// Quantize: 2 passes — find max|x|, then `round(x / scale).clamp(-127, 127)`.
// Dequantize: 1 pass — `out = q * scale`.
//
// NEON path processes 4 f32 per cycle via `vmaxq_f32` / `vmulq_f32` /
// `vcvtq_s32_f32` / saturating narrow. Available on every target we
// ship to (Apple Silicon, Pi 4/5, Android arm64). The scalar fallback
// path is used on non-aarch64 hosts and for the tail of a row when
// `d_head` isn't a multiple of 4.

#[inline]
fn quantize_row(row: &[f32], q_out: &mut [i8]) -> f32 {
    debug_assert_eq!(row.len(), q_out.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        quantize_row_neon(row, q_out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    quantize_row_scalar(row, q_out)
}

#[inline]
fn dequantize_row(q: &[i8], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(q.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        dequantize_row_neon(q, scale, out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    dequantize_row_scalar(q, scale, out)
}

/// Pure-scalar reference. Always available — used directly off-aarch64,
/// and as the parity oracle for the NEON path's correctness test.
#[allow(dead_code)] // referenced by tests + non-aarch64 path
pub(super) fn quantize_row_scalar(row: &[f32], q_out: &mut [i8]) -> f32 {
    let mut max_abs = 0.0_f32;
    for &v in row {
        let a = v.abs();
        if a > max_abs { max_abs = a; }
    }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inv = 1.0 / scale;
    for (slot, &v) in q_out.iter_mut().zip(row) {
        *slot = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

#[allow(dead_code)] // referenced by tests + non-aarch64 path
pub(super) fn dequantize_row_scalar(q: &[i8], scale: f32, out: &mut [f32]) {
    for (dst, &qb) in out.iter_mut().zip(q) {
        *dst = qb as f32 * scale;
    }
}

/// NEON-accelerated quantize. 4 f32 per iteration via SIMD; scalar
/// tail for rows whose `d_head` isn't a multiple of 4.
///
/// SAFETY: `row` and `q_out` must be the same length. The pointer
/// arithmetic uses `add(chunk*4)` bounded by `chunks` derived from the
/// length, so it stays within both slices.
#[cfg(target_arch = "aarch64")]
unsafe fn quantize_row_neon(row: &[f32], q_out: &mut [i8]) -> f32 {
    use std::arch::aarch64::*;

    let len = row.len();
    let chunks = len / 4;

    // Pass 1: max-abs reduction.
    let mut max_v = vdupq_n_f32(0.0);
    for c in 0..chunks {
        let v = vld1q_f32(row.as_ptr().add(c * 4));
        let av = vabsq_f32(v);
        max_v = vmaxq_f32(max_v, av);
    }
    let mut max_abs = vmaxvq_f32(max_v);
    for i in (chunks * 4)..len {
        let a = row[i].abs();
        if a > max_abs { max_abs = a; }
    }

    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inv = 1.0 / scale;

    // Pass 2: round-and-clamp via SIMD, narrow i32 → i8 with saturation.
    let inv_v = vdupq_n_f32(inv);
    let p127 = vdupq_n_f32(127.0);
    let n127 = vdupq_n_f32(-127.0);
    for c in 0..chunks {
        let v = vld1q_f32(row.as_ptr().add(c * 4));
        let scaled = vmulq_f32(v, inv_v);
        // Round to nearest, ties-to-even — matches `f32::round` only
        // for non-tie values; ties differ. For typical attention K/V
        // distributions ties are statistically negligible. (Exact-tie
        // discrepancy is bounded by 1 ULP per element and tested
        // against the scalar reference in the parity test below.)
        let rounded = vrndnq_f32(scaled);
        let clamped = vminq_f32(vmaxq_f32(rounded, n127), p127);
        let i32x4 = vcvtq_s32_f32(clamped);
        let i16x4 = vqmovn_s32(i32x4);
        let i16x8 = vcombine_s16(i16x4, vdup_n_s16(0));
        let i8x8 = vqmovn_s16(i16x8);
        // Store the low 4 bytes (the lane representing our 4 i8 values).
        let ptr = q_out.as_mut_ptr().add(c * 4) as *mut u32;
        ptr.write_unaligned(std::mem::transmute::<int8x8_t, [u32; 2]>(i8x8)[0]);
    }
    for i in (chunks * 4)..len {
        q_out[i] = (row[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// NEON-accelerated dequantize. Loads 4 i8 → widens i8→i16→i32→f32
/// → multiplies by scale → stores 4 f32. The narrowing reverse path
/// of the quantize kernel.
#[cfg(target_arch = "aarch64")]
unsafe fn dequantize_row_neon(q: &[i8], scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = q.len();
    let chunks = len / 4;
    let scale_v = vdupq_n_f32(scale);

    for c in 0..chunks {
        // Load 4 i8 packed into the low 32 bits.
        let bytes = (q.as_ptr().add(c * 4) as *const u32).read_unaligned();
        // Splat into an i8x8 lane vector, then widen low half: i8x8 →
        // i16x8 → take low half i16x4 → widen to i32x4 → cvt to f32x4.
        let i8x8 = vreinterpret_s8_u32(vdup_n_u32(bytes));
        let i16x8 = vmovl_s8(i8x8);
        let i32x4 = vmovl_s16(vget_low_s16(i16x8));
        let f32x4 = vcvtq_f32_s32(i32x4);
        let scaled = vmulq_f32(f32x4, scale_v);
        vst1q_f32(out.as_mut_ptr().add(c * 4), scaled);
    }
    for i in (chunks * 4)..len {
        out[i] = q[i] as f32 * scale;
    }
}

#[cfg(test)]
mod kv_quant_tests {
    use super::*;

    #[test]
    fn quant_dequant_roundtrip_is_bounded() {
        // Synthetic K-like values in [-2, 2] over 4 rows of d_head=8.
        let n_rows = 4;
        let d_head = 8;
        let mut fp32_bytes = Vec::with_capacity(n_rows * d_head * 4);
        for r in 0..n_rows {
            for i in 0..d_head {
                let v = ((r * d_head + i) as f32) * 0.13 - 1.5;
                fp32_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }

        let (q, scales) = quantize_per_row(&fp32_bytes, n_rows, d_head);
        let back = dequantize_per_row(&q, &scales, n_rows, d_head);

        // Error per element <= scale/2 (theoretical bound on symmetric
        // round-to-nearest quant).
        let mut max_abs_err = 0.0_f32;
        for r in 0..n_rows {
            for i in 0..d_head {
                let off = (r * d_head + i) * 4;
                let orig = f32::from_le_bytes([
                    fp32_bytes[off], fp32_bytes[off + 1],
                    fp32_bytes[off + 2], fp32_bytes[off + 3],
                ]);
                let deq = f32::from_le_bytes([
                    back[off], back[off + 1], back[off + 2], back[off + 3],
                ]);
                let err = (orig - deq).abs();
                if err > max_abs_err { max_abs_err = err; }
                assert!(err <= scales[r] / 2.0 + 1e-7,
                    "row {} col {}: err {} > bound {}", r, i, err, scales[r] / 2.0);
            }
        }
        eprintln!("quant roundtrip max_abs_err = {max_abs_err:.6}");
    }

    #[test]
    fn empty_cache_take_replace_roundtrip_int8_path() {
        // Zero cache initially. take_buffers should yield zero fp32
        // tensors; replace_buffers with non-zero values should
        // quantize+store; next take_buffers should dequant to ~the
        // same values (within bound).
        let mut cache = InPlaceKvCache::new_with_quant(1, 2, 4, 4, KvQuant::Int8);
        let (k0, v0) = cache.take_buffers(0);
        // All zero on a fresh cache.
        for &b in k0.storage.bytes.iter() { assert_eq!(b, 0); }
        for &b in v0.storage.bytes.iter() { assert_eq!(b, 0); }

        // Build a non-trivial fp32 K to put back.
        let elems = 2 * 4 * 4;
        let mut k_bytes = Vec::with_capacity(elems * 4);
        for i in 0..elems {
            let v = (i as f32) * 0.05 - 0.5;
            k_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let k_new = Tensor {
            meta: TensorMeta::new(Dtype::F32, &[2, 4, 4]),
            storage: std::sync::Arc::new(TensorStorage { bytes: k_bytes.clone(), mapped: None }),
        };
        cache.replace_buffers(0, k_new, v0);

        // Take again, dequant, compare.
        let (k_back, _) = cache.take_buffers(0);
        let mut max_err = 0.0_f32;
        for i in 0..elems {
            let off = i * 4;
            let orig = f32::from_le_bytes([
                k_bytes[off], k_bytes[off + 1], k_bytes[off + 2], k_bytes[off + 3],
            ]);
            let deq = f32::from_le_bytes([
                k_back.storage.bytes[off], k_back.storage.bytes[off + 1],
                k_back.storage.bytes[off + 2], k_back.storage.bytes[off + 3],
            ]);
            let e = (orig - deq).abs();
            if e > max_err { max_err = e; }
        }
        eprintln!("int8 cache roundtrip max_err = {max_err:.6}");
        // Per-row max_abs/127 bound; rows here have max_abs <= 1, so
        // err per element <= 1/254 ≈ 0.004.
        assert!(max_err < 0.01, "max_err too high: {max_err}");
    }

    #[test]
    fn cache_bytes_reports_4x_savings() {
        let fp32 = InPlaceKvCache::new(24, 8, 128, 64);
        let int8 = InPlaceKvCache::new_with_quant(24, 8, 128, 64, KvQuant::Int8);
        let ratio = fp32.cache_bytes() as f64 / int8.cache_bytes() as f64;
        eprintln!(
            "cache_bytes: fp32={} int8={} ratio={:.2}x",
            fp32.cache_bytes(), int8.cache_bytes(), ratio,
        );
        // ~4× — int8 trades 4× data for tiny scale overhead.
        assert!(ratio > 3.5 && ratio < 4.5, "ratio {ratio} not ~4×");
    }

    /// Dequantize is pure integer-to-float multiply — NEON and scalar
    /// must agree bit-for-bit (same f32 multiply, just lane-batched).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dequantize_row_neon_matches_scalar_bitwise() {
        let d_head = 64;
        let scale = 0.0123_f32;
        // Cover the full int8 range a few times, plus tail elements
        // that don't fit a 4-wide chunk.
        let q: Vec<i8> = (0..d_head).map(|i| ((i as i32 * 7 - 3) & 0xFF) as i8).collect();
        let mut out_neon = vec![0.0_f32; d_head];
        let mut out_scalar = vec![0.0_f32; d_head];
        unsafe { super::dequantize_row_neon(&q, scale, &mut out_neon); }
        super::dequantize_row_scalar(&q, scale, &mut out_scalar);
        for (i, (&n, &s)) in out_neon.iter().zip(&out_scalar).enumerate() {
            assert_eq!(n.to_bits(), s.to_bits(),
                "row[{i}]: NEON {n} bits={:08x} vs scalar {s} bits={:08x}",
                n.to_bits(), s.to_bits());
        }
    }

    /// Standalone kernel bench: NEON vs scalar quant + dequant, no
    /// allocator / cache / executor in the loop. Confirms the inner
    /// SIMD kernel does the work it should — if this ratio is <1 (NEON
    /// faster) but the full Bonsai bench is barely affected, the
    /// bottleneck is somewhere else (likely allocation churn).
    ///
    /// Run: cargo test --release -p jouleclaw-loader-gguf
    ///       bench_quant_kernel -- --ignored --nocapture
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bench_quant_kernel() {
        use std::time::Instant;
        // Match Bonsai-1.7B: d_head=64, n_heads_kv * max_seq = 8 * 128 = 1024 rows.
        let d_head = 64;
        let n_rows = 1024;
        let row_count = n_rows;

        // Pre-generate the input.
        let mut rows = Vec::with_capacity(row_count * d_head);
        for r in 0..row_count {
            for i in 0..d_head {
                let v = ((r * d_head + i) as f32) * 0.001 - 0.5;
                rows.push(v);
            }
        }
        let mut q_scalar = vec![0_i8; row_count * d_head];
        let mut q_neon = vec![0_i8; row_count * d_head];

        // Warmup.
        for r in 0..row_count {
            let row = &rows[r * d_head..(r + 1) * d_head];
            let _ = super::quantize_row_scalar(row, &mut q_scalar[r * d_head..(r + 1) * d_head]);
        }
        for r in 0..row_count {
            let row = &rows[r * d_head..(r + 1) * d_head];
            unsafe {
                let _ = super::quantize_row_neon(row, &mut q_neon[r * d_head..(r + 1) * d_head]);
            }
        }

        let iters = 100;

        // Scalar quantize.
        let t = Instant::now();
        for _ in 0..iters {
            for r in 0..row_count {
                let row = &rows[r * d_head..(r + 1) * d_head];
                let _ = super::quantize_row_scalar(row, &mut q_scalar[r * d_head..(r + 1) * d_head]);
            }
        }
        let t_scalar_q = t.elapsed();

        // NEON quantize.
        let t = Instant::now();
        for _ in 0..iters {
            for r in 0..row_count {
                let row = &rows[r * d_head..(r + 1) * d_head];
                unsafe {
                    let _ = super::quantize_row_neon(row, &mut q_neon[r * d_head..(r + 1) * d_head]);
                }
            }
        }
        let t_neon_q = t.elapsed();

        // Scalar dequantize.
        let scale = 0.01;
        let mut out_scalar = vec![0.0_f32; row_count * d_head];
        let mut out_neon = vec![0.0_f32; row_count * d_head];
        let t = Instant::now();
        for _ in 0..iters {
            for r in 0..row_count {
                super::dequantize_row_scalar(
                    &q_scalar[r * d_head..(r + 1) * d_head], scale,
                    &mut out_scalar[r * d_head..(r + 1) * d_head]);
            }
        }
        let t_scalar_d = t.elapsed();

        // NEON dequantize.
        let t = Instant::now();
        for _ in 0..iters {
            for r in 0..row_count {
                unsafe {
                    super::dequantize_row_neon(
                        &q_scalar[r * d_head..(r + 1) * d_head], scale,
                        &mut out_neon[r * d_head..(r + 1) * d_head]);
                }
            }
        }
        let t_neon_d = t.elapsed();

        let q_ratio = t_neon_q.as_secs_f64() / t_scalar_q.as_secs_f64();
        let d_ratio = t_neon_d.as_secs_f64() / t_scalar_d.as_secs_f64();
        eprintln!(
            "[d_head={d_head}, n_rows={n_rows}, iters={iters}]  \
             quant: scalar={:.3}ms  neon={:.3}ms  neon/scalar={:.2}x  \
             dequant: scalar={:.3}ms  neon={:.3}ms  neon/scalar={:.2}x",
            t_scalar_q.as_secs_f64() * 1000.0,
            t_neon_q.as_secs_f64() * 1000.0,
            q_ratio,
            t_scalar_d.as_secs_f64() * 1000.0,
            t_neon_d.as_secs_f64() * 1000.0,
            d_ratio,
        );
    }

    /// Quantize differs in rounding semantics — `vrndnq_f32` is round-
    /// to-nearest-ties-to-even, `f32::round` is round-to-nearest-ties-
    /// away-from-zero. They agree on every non-tie value; ties can
    /// differ by ±1 in the i8 output. For typical fp values this
    /// happens essentially never; we test that NEON is within ±1 of
    /// scalar everywhere, and bit-exact on non-tie cases.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn quantize_row_neon_within_one_ulp_of_scalar() {
        let d_head = 64;
        // Synthetic fp32 row with values that span [-2, 2], a mix of
        // tie and non-tie quantize points.
        let row: Vec<f32> = (0..d_head)
            .map(|i| (i as f32) * 0.061 - 1.95)
            .collect();
        let mut q_neon = vec![0_i8; d_head];
        let mut q_scalar = vec![0_i8; d_head];
        let s_neon = unsafe { super::quantize_row_neon(&row, &mut q_neon) };
        let s_scalar = super::quantize_row_scalar(&row, &mut q_scalar);
        assert!((s_neon - s_scalar).abs() < 1e-6, "scales differ");
        let mut max_diff: i32 = 0;
        for (i, (&n, &s)) in q_neon.iter().zip(&q_scalar).enumerate() {
            let d = (n as i32 - s as i32).abs();
            if d > max_diff { max_diff = d; }
            assert!(d <= 1,
                "row[{i}]: NEON {n} vs scalar {s} differs by {d} (>1 ulp)");
        }
        eprintln!("quantize_row_neon vs scalar max diff = {max_diff} ULP");
    }
}

/// In-place decode-step graph. Inputs are token IDs + per-layer K/V buffers
/// (always present, always full-sized); outputs are per-layer updated K/V
/// buffers (also full-sized) and logits for the new tokens.
pub struct InPlaceDecodeStepGraph {
    pub graph: Graph,
    pub config: LlamaConfig,
    pub new_seq: usize,
    pub cached_seq: usize,
    pub k_output_names: Vec<String>,
    pub v_output_names: Vec<String>,
    pub logits_output_name: String,
    pub k_input_names: Vec<String>,
    pub v_input_names: Vec<String>,
    /// Per-layer LFM2 shortconv state input names. `Some(name)` for
    /// shortconv layers, `None` for attention layers. Empty Vec on
    /// non-LFM2 models.
    pub shortconv_state_input_names: Vec<Option<String>>,
    /// Per-layer LFM2 shortconv state output names. Same shape rules
    /// as `shortconv_state_input_names`.
    pub shortconv_state_output_names: Vec<Option<String>>,
}

/// Build the in-place decode-step graph.
pub fn build_decode_step_graph_inplace(
    model: &GgufModel,
    cache: &InPlaceKvCache,
    new_seq: usize,
) -> Result<InPlaceDecodeStepGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let cached_seq = cache.current_seq;
    let n_heads = config.head_count.max(1);
    let n_heads_kv = cache.n_heads_kv;
    let d_head = cache.d_head;
    let max_seq = cache.max_seq;
    let group_size = n_heads / n_heads_kv;
    if group_size * n_heads_kv != n_heads {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "head_count ({}) not divisible by head_count_kv ({})",
            n_heads, n_heads_kv)));
    }

    let mut g = GraphBuilder::new();
    let token_ids = g.input("token_ids", TensorMeta::new(Dtype::I32, &[new_seq]));

    // Per-layer K/V buffer inputs — always present, always [n_heads_kv, max_seq, d_head].
    let mut k_input_names = Vec::with_capacity(config.block_count);
    let mut v_input_names = Vec::with_capacity(config.block_count);
    let mut k_input_nodes = Vec::with_capacity(config.block_count);
    let mut v_input_nodes = Vec::with_capacity(config.block_count);
    for layer in 0..config.block_count {
        let k_name = format!("kv_buf_k_{}", layer);
        let v_name = format!("kv_buf_v_{}", layer);
        let k_node = g.input(&k_name,
            TensorMeta::new(Dtype::F32, &[n_heads_kv, max_seq, d_head]));
        let v_node = g.input(&v_name,
            TensorMeta::new(Dtype::F32, &[n_heads_kv, max_seq, d_head]));
        k_input_nodes.push(k_node);
        v_input_nodes.push(v_node);
        k_input_names.push(k_name);
        v_input_names.push(v_name);
    }

    // Token embedding + (tied) LM head — Q2_0-aware. Bonsai's packed
    // table is reused for both, never expanded to f32.
    use crate::llama::{embed_weight, lookup_w, wmm, Weight as LWeight};
    let te_info = model.tensor_by_name("token_embd.weight")
        .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
    let token_embd_w: Option<LWeight> = if matches!(
        te_info.dtype, crate::GgmlType::Q2_0 | crate::GgmlType::Q1_0,
    ) {
        Some(embed_weight(&mut g, model, "token_embd.weight")?)
    } else {
        None
    };
    let mut x = match &token_embd_w {
        Some(w) => lookup_w(&mut g, token_ids, w),
        None => {
            let te = embed_constant_pub(&mut g, model, "token_embd.weight")?;
            g.lookup(token_ids, te)
        }
    };

    // Per-layer K/V outputs (the buffers after scatter).
    let mut k_output_names = Vec::with_capacity(config.block_count);
    let mut v_output_names = Vec::with_capacity(config.block_count);

    for layer in 0..config.block_count {
        let result = build_decode_step_graph_inplace_block(
            &mut g, model, &config, layer, x,
            new_seq, cached_seq, max_seq,
            n_heads, n_heads_kv, d_head, group_size,
            k_input_nodes[layer], v_input_nodes[layer],
        )?;
        x = result.x_out;

        let k_out_name = format!("kv_out_k_{}", layer);
        let v_out_name = format!("kv_out_v_{}", layer);
        g.output(&k_out_name, result.k_buf_new);
        g.output(&v_out_name, result.v_buf_new);
        k_output_names.push(k_out_name);
        v_output_names.push(v_out_name);
    }

    // Final norm + lm_head.
    let output_norm = embed_constant_pub(&mut g, model, "output_norm.weight")?;
    let xn = g.norm(x, output_norm, NormKind::Rms, config.rms_eps);

    let logits = if model.tensor_by_name("output.weight").is_some() {
        let w = embed_weight(&mut g, model, "output.weight")?;
        wmm(&mut g, xn, &w)
    } else {
        match &token_embd_w {
            Some(w) => wmm(&mut g, xn, w),
            None => {
                let te = embed_constant_pub(&mut g, model, "token_embd.weight")?;
                g.matmul_bt(xn, te)
            }
        }
    };
    g.output("logits", logits);

    let block_count = config.block_count;
    Ok(InPlaceDecodeStepGraph {
        graph: g.build(),
        config,
        new_seq, cached_seq,
        k_output_names, v_output_names,
        logits_output_name: "logits".into(),
        k_input_names, v_input_names,
        // The non-const variant doesn't currently support LFM2's
        // shortconv state plumbing. It's used by tests and earlier
        // paths; LFM2 routes through the const variant only.
        shortconv_state_input_names: vec![None; block_count],
        shortconv_state_output_names: vec![None; block_count],
    })
}

/// Runtime input name for the dynamic KV position (= `cached_seq`,
/// I32 `[1]`) consumed by the constant-topology decode graph.
pub const KV_POS_INPUT: &str = "kv_pos";

/// **Constant-topology** counterpart of [`build_decode_step_graph_inplace`].
/// The graph has fixed shape for a given `(model, new_seq, max_seq)` —
/// `cached_seq` enters as the runtime input [`KV_POS_INPUT`] rather
/// than being baked into op attrs, and attention runs over the full
/// `max_seq` buffer (no `valid_seq` slice). Compile it once and reuse
/// it for every decode step (see `run_inplace_step`'s cache).
///
/// `cached_seq` in the returned struct is set to 0 and is meaningless
/// for the const graph — the real position is bound at execute time
/// via `KV_POS_INPUT`.
pub fn build_decode_step_graph_inplace_const(
    model: &GgufModel,
    cache: &InPlaceKvCache,
    new_seq: usize,
) -> Result<InPlaceDecodeStepGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let n_heads = config.head_count.max(1);
    let n_heads_kv = cache.n_heads_kv;
    let d_head = cache.d_head;
    let max_seq = cache.max_seq;
    let group_size = n_heads / n_heads_kv;
    if group_size * n_heads_kv != n_heads {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "head_count ({}) not divisible by head_count_kv ({})",
            n_heads, n_heads_kv)));
    }

    let mut g = GraphBuilder::new();
    let token_ids = g.input("token_ids", TensorMeta::new(Dtype::I32, &[new_seq]));
    let kv_pos = g.input(KV_POS_INPUT, TensorMeta::new(Dtype::I32, &[1]));

    let mut k_input_names = Vec::with_capacity(config.block_count);
    let mut v_input_names = Vec::with_capacity(config.block_count);
    let mut k_input_nodes = Vec::with_capacity(config.block_count);
    let mut v_input_nodes = Vec::with_capacity(config.block_count);
    // Per-layer LFM2 shortconv state inputs (None for attention layers).
    let mut shortconv_state_input_names: Vec<Option<String>> =
        Vec::with_capacity(config.block_count);
    let mut shortconv_state_input_nodes: Vec<Option<jouleclaw_core::graph::NodeId>> =
        Vec::with_capacity(config.block_count);
    let shortconv_window = config.shortconv_l_cache.max(3).saturating_sub(1);
    for layer in 0..config.block_count {
        let k_name = format!("kv_buf_k_{}", layer);
        let v_name = format!("kv_buf_v_{}", layer);
        let k_node = g.input(&k_name,
            TensorMeta::new(Dtype::F32, &[n_heads_kv, max_seq, d_head]));
        let v_node = g.input(&v_name,
            TensorMeta::new(Dtype::F32, &[n_heads_kv, max_seq, d_head]));
        k_input_nodes.push(k_node);
        v_input_nodes.push(v_node);
        k_input_names.push(k_name);
        v_input_names.push(v_name);

        let is_recurrent = config.arch == "lfm2"
            && config.per_layer_head_count_kv.get(layer).copied() == Some(0);
        if is_recurrent {
            let name = format!("shortconv_state_{}", layer);
            let node = g.input(&name,
                TensorMeta::new(Dtype::F32, &[shortconv_window, config.embedding_length]));
            shortconv_state_input_names.push(Some(name));
            shortconv_state_input_nodes.push(Some(node));
        } else {
            shortconv_state_input_names.push(None);
            shortconv_state_input_nodes.push(None);
        }
    }

    use crate::llama::{embed_weight, lookup_w, wmm, Weight as LWeight};
    let te_info = model.tensor_by_name("token_embd.weight")
        .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
    let token_embd_w: Option<LWeight> = if matches!(
        te_info.dtype, crate::GgmlType::Q2_0 | crate::GgmlType::Q1_0,
    ) {
        Some(embed_weight(&mut g, model, "token_embd.weight")?)
    } else {
        None
    };
    let mut x = match &token_embd_w {
        Some(w) => lookup_w(&mut g, token_ids, w),
        None => {
            let te = embed_constant_pub(&mut g, model, "token_embd.weight")?;
            g.lookup(token_ids, te)
        }
    };

    let mut k_output_names = Vec::with_capacity(config.block_count);
    let mut v_output_names = Vec::with_capacity(config.block_count);
    let mut shortconv_state_output_names: Vec<Option<String>> =
        Vec::with_capacity(config.block_count);
    for layer in 0..config.block_count {
        let result = crate::decode::build_decode_step_graph_inplace_const_block(
            &mut g, model, &config, layer, x,
            new_seq, kv_pos, max_seq,
            n_heads, n_heads_kv, d_head, group_size,
            k_input_nodes[layer], v_input_nodes[layer],
            shortconv_state_input_nodes[layer],
        )?;
        x = result.x_out;
        let k_out_name = format!("kv_out_k_{}", layer);
        let v_out_name = format!("kv_out_v_{}", layer);
        g.output(&k_out_name, result.k_buf_new);
        g.output(&v_out_name, result.v_buf_new);
        k_output_names.push(k_out_name);
        v_output_names.push(v_out_name);

        if let Some(state_out) = result.shortconv_state_out {
            let name = format!("shortconv_state_out_{}", layer);
            g.output(&name, state_out);
            shortconv_state_output_names.push(Some(name));
        } else {
            shortconv_state_output_names.push(None);
        }
    }

    // LFM2's final norm uses `token_embd_norm.weight` (per Liquid's
    // Transformers config); every other arch uses
    // `output_norm.weight`. Pick whichever exists.
    let final_norm_name = if config.arch == "lfm2"
        && model.tensor_by_name("token_embd_norm.weight").is_some()
    {
        "token_embd_norm.weight"
    } else {
        "output_norm.weight"
    };
    let output_norm = embed_constant_pub(&mut g, model, final_norm_name)?;
    let xn = g.norm(x, output_norm, NormKind::Rms, config.rms_eps);
    let logits = if model.tensor_by_name("output.weight").is_some() {
        let w = embed_weight(&mut g, model, "output.weight")?;
        wmm(&mut g, xn, &w)
    } else {
        match &token_embd_w {
            Some(w) => wmm(&mut g, xn, w),
            None => {
                let te = embed_constant_pub(&mut g, model, "token_embd.weight")?;
                g.matmul_bt(xn, te)
            }
        }
    };
    g.output("logits", logits);

    Ok(InPlaceDecodeStepGraph {
        graph: g.build(),
        config,
        new_seq, cached_seq: 0,
        k_output_names, v_output_names,
        logits_output_name: "logits".into(),
        k_input_names, v_input_names,
        shortconv_state_input_names,
        shortconv_state_output_names,
    })
}

// ──────────────────────────────────────────────────────────────────────
// Sequential per-piece decode graphs
//
// The monolithic `build_decode_step_graph_inplace_const` above packs
// all `n_layers` transformer blocks into one compiled graph that
// requires every layer's K/V tensors alive simultaneously in the
// executor's inputs HashMap. For int8 KV quant, that means
// dequantizing all 24 layers' K/V into fresh fp32 buffers at once —
// the allocation churn / page-fault cost that pool int8 was meant
// to eliminate would just re-emerge if the pool had to hold all 24
// layers' worth of fp32.
//
// The three builders below split the monolithic graph into pieces:
//
//   1. `build_embed_only_graph`:  token_ids → x
//   2. `build_layer_only_graph`:  x_in, k_buf_in, v_buf_in, kv_pos →
//                                  x_out, k_buf_out, v_buf_out
//   3. `build_head_only_graph`:   x_in → logits
//
// Orchestrated by `joule_runtime::generate::run_inplace_step_sequential`,
// only one layer's K/V is in fp32 working memory at any moment, so
// the cache only needs a single shared fp32 working buffer (~512 KB
// at Bonsai's max_seq=128) instead of n_layers × that.
//
// Each layer graph compiles to a separate `CompiledGraph` because the
// layer's per-block weights (attn_q, attn_k, …) are baked in as
// constants — so we cache `n_layers + 2` compiled graphs per
// `new_seq` instead of 1. Compile cost is ~10 ms/graph at Bonsai
// scale, paid once at first use.

/// Embed-only piece: tokens → hidden state.
pub struct EmbedOnlyGraph {
    pub graph: Graph,
    pub config: LlamaConfig,
    pub new_seq: usize,
    pub token_ids_input_name: String,
    pub x_output_name: String,
}

pub fn build_embed_only_graph(
    model: &GgufModel,
    new_seq: usize,
) -> Result<EmbedOnlyGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let mut g = GraphBuilder::new();
    let token_ids = g.input("token_ids", TensorMeta::new(Dtype::I32, &[new_seq]));

    use crate::llama::{embed_weight, lookup_w, Weight as LWeight};
    let te_info = model.tensor_by_name("token_embd.weight")
        .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
    let token_embd_w: Option<LWeight> = if matches!(
        te_info.dtype, crate::GgmlType::Q2_0 | crate::GgmlType::Q1_0,
    ) {
        Some(embed_weight(&mut g, model, "token_embd.weight")?)
    } else {
        None
    };
    let x = match &token_embd_w {
        Some(w) => lookup_w(&mut g, token_ids, w),
        None => {
            let te = embed_constant_pub(&mut g, model, "token_embd.weight")?;
            g.lookup(token_ids, te)
        }
    };
    g.output("x", x);

    Ok(EmbedOnlyGraph {
        graph: g.build(),
        config,
        new_seq,
        token_ids_input_name: "token_ids".into(),
        x_output_name: "x".into(),
    })
}

/// One transformer layer piece: hidden state in + K/V buffers →
/// hidden state out + K/V buffers out.
pub struct LayerOnlyGraph {
    pub graph: Graph,
    pub layer: usize,
    pub new_seq: usize,
    pub x_input_name: String,
    pub k_input_name: String,
    pub v_input_name: String,
    pub x_output_name: String,
    pub k_output_name: String,
    pub v_output_name: String,
}

pub fn build_layer_only_graph(
    model: &GgufModel,
    cache: &InPlaceKvCache,
    layer: usize,
    new_seq: usize,
) -> Result<LayerOnlyGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let n_heads = config.head_count.max(1);
    let n_heads_kv = cache.n_heads_kv;
    let d_head = cache.d_head;
    let max_seq = cache.max_seq;
    let group_size = n_heads / n_heads_kv;
    if group_size * n_heads_kv != n_heads {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "head_count ({}) not divisible by head_count_kv ({})",
            n_heads, n_heads_kv)));
    }

    let mut g = GraphBuilder::new();
    let kv_pos = g.input(KV_POS_INPUT, TensorMeta::new(Dtype::I32, &[1]));
    let hidden_size = config.embedding_length;
    let x_in = g.input("x_in", TensorMeta::new(Dtype::F32, &[new_seq, hidden_size]));
    let k_buf_in = g.input(
        "k_buf_in",
        TensorMeta::new(Dtype::F32, &[n_heads_kv, max_seq, d_head]),
    );
    let v_buf_in = g.input(
        "v_buf_in",
        TensorMeta::new(Dtype::F32, &[n_heads_kv, max_seq, d_head]),
    );

    let result = crate::decode::build_decode_step_graph_inplace_const_block(
        &mut g, model, &config, layer, x_in,
        new_seq, kv_pos, max_seq,
        n_heads, n_heads_kv, d_head, group_size,
        k_buf_in, v_buf_in,
        // Sequential per-layer graph: LFM2 shortconv state plumbing
        // through this path is a follow-up. The const-monolithic path
        // (build_decode_step_graph_inplace_const above) has the
        // shortconv dispatch; that's what Conversation::extend uses.
        None,
    )?;

    g.output("x_out", result.x_out);
    g.output("k_buf_out", result.k_buf_new);
    g.output("v_buf_out", result.v_buf_new);

    Ok(LayerOnlyGraph {
        graph: g.build(),
        layer,
        new_seq,
        x_input_name: "x_in".into(),
        k_input_name: "k_buf_in".into(),
        v_input_name: "v_buf_in".into(),
        x_output_name: "x_out".into(),
        k_output_name: "k_buf_out".into(),
        v_output_name: "v_buf_out".into(),
    })
}

/// Head-only piece: final norm + LM head, hidden state → logits.
pub struct HeadOnlyGraph {
    pub graph: Graph,
    pub new_seq: usize,
    pub x_input_name: String,
    pub logits_output_name: String,
}

pub fn build_head_only_graph(
    model: &GgufModel,
    new_seq: usize,
) -> Result<HeadOnlyGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let hidden_size = config.embedding_length;

    let mut g = GraphBuilder::new();
    let x_in = g.input("x_in", TensorMeta::new(Dtype::F32, &[new_seq, hidden_size]));

    let output_norm = embed_constant_pub(&mut g, model, "output_norm.weight")?;
    let xn = g.norm(x_in, output_norm, NormKind::Rms, config.rms_eps);

    use crate::llama::{embed_weight, wmm};
    let logits = if model.tensor_by_name("output.weight").is_some() {
        let w = embed_weight(&mut g, model, "output.weight")?;
        wmm(&mut g, xn, &w)
    } else {
        // Tied embedding: reuse token_embd.weight as the LM head.
        // Same dispatch logic as the monolithic builder.
        let te_info = model.tensor_by_name("token_embd.weight")
            .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
        if matches!(
            te_info.dtype, crate::GgmlType::Q2_0 | crate::GgmlType::Q1_0,
        ) {
            let w = embed_weight(&mut g, model, "token_embd.weight")?;
            wmm(&mut g, xn, &w)
        } else {
            let te = embed_constant_pub(&mut g, model, "token_embd.weight")?;
            g.matmul_bt(xn, te)
        }
    };
    g.output("logits", logits);

    Ok(HeadOnlyGraph {
        graph: g.build(),
        new_seq,
        x_input_name: "x_in".into(),
        logits_output_name: "logits".into(),
    })
}
