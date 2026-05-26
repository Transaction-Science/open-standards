//! Pure-Rust correctness-first Gemma 4 (E2B-class) text forward.
//!
//! Follows the established codebase pattern (`lmm::vision::VisionTower`,
//! `lmm::vl_real::LfmVl`): load the adapted [`GgufModel`] (via
//! [`crate::gemma::load_gemma_dir`] — config→metadata, HF→GGUF name
//! remap), dequantise the needed text tensors to
//! `f32`, and run a plain, faithful forward. Gemma 4 needs ops the
//! joule graph does not yet have (per-layer-type partial/proportional
//! RoPE, sliding-window mask, PLE, GeGLU-tanh, logit softcap, MQA
//! broadcast), so correctness lands here first and is oracle-gated
//! against HF `transformers` reference logits before any tier/graph
//! integration is claimed.
//!
//! Spec is reverse-engineered from `transformers` `modular_gemma4.py`
//! and **verified against the real E2B checkpoint tensor shapes**:
//! per-layer-type head_dim (sliding 256 / full_attention 512),
//! double-wide MLP on the kv-shared tail (layers ≥ `L-num_kv_shared`),
//! MQA (`n_kv=1`), 4-norm sandwich + PLE residual + `layer_scalar`,
//! tied+unscaled LM head, `softcap·tanh(logits/softcap)`.

use std::collections::HashMap;
use std::path::Path;

use crate::{tensor_from_gguf, GgufModel, ParseError};

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,        // sliding-layer head_dim
    pub global_head_dim: usize, // full-attention head_dim
    pub ffn: usize,
    pub vocab: usize,
    pub ple_dim: usize,
    pub n_kv_shared: usize,
    pub sliding_window: usize,
    pub rms_eps: f32,
    pub rope_theta_sliding: f32,
    pub rope_theta_full: f32,
    pub partial_rotary_factor: f32,
    pub softcap: f32,
    pub layer_types: Vec<String>,
    /// Gemma-2-only: tanh softcap applied to attention logits before softmax.
    /// `logits = softcap_attn * tanh(logits / softcap_attn)`. Default 0.0 = disabled
    /// (Gemma-4 path; q-norm + k-norm handle the equivalent role).
    pub attn_softcap: f32,
    /// Gemma-2-only: Q is scaled by `1/sqrt(q_pre_attn_scalar)` instead of
    /// `1/sqrt(head_dim)`. Default 0.0 = disabled (Gemma-4 uses `scaling=1.0`).
    pub q_pre_attn_scalar: f32,
}

fn mu(m: &GgufModel, k: &str) -> Option<u64> {
    m.metadata_u64(k)
}
fn mf(m: &GgufModel, k: &str) -> Option<f32> {
    m.metadata_f32(k)
}

impl Gemma4Config {
    fn from_model(m: &GgufModel) -> Result<Self, ParseError> {
        let g = |k: &str| -> Result<u64, ParseError> {
            mu(m, k).ok_or_else(|| {
                ParseError::Safetensors(format!("gemma4: missing {k}"))
            })
        };
        let layer_types = m
            .metadata
            .get("gemma.attention.layer_types")
            .and_then(|v| v.as_string_array())
            .map(|v| v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .unwrap_or_default();
        Ok(Self {
            d_model: g("gemma.embedding_length")? as usize,
            n_layers: g("gemma.block_count")? as usize,
            n_heads: g("gemma.attention.head_count")? as usize,
            n_kv: g("gemma.attention.head_count_kv")? as usize,
            head_dim: g("gemma.attention.key_length")? as usize,
            global_head_dim: mu(m, "gemma.attention.global_head_dim")
                .unwrap_or_else(|| g("gemma.attention.key_length").unwrap())
                as usize,
            ffn: g("gemma.feed_forward_length")? as usize,
            vocab: g("gemma.vocab_size")? as usize,
            ple_dim: mu(m, "gemma.per_layer_input_length").unwrap_or(0)
                as usize,
            n_kv_shared: mu(m, "gemma.attention.num_kv_shared_layers")
                .unwrap_or(0) as usize,
            sliding_window: mu(m, "gemma.attention.sliding_window")
                .unwrap_or(0) as usize,
            rms_eps: mf(m, "gemma.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6),
            rope_theta_sliding: mf(m, "gemma.rope.freq_base")
                .unwrap_or(10000.0),
            rope_theta_full: mf(m, "gemma.rope.freq_base_global")
                .unwrap_or(1_000_000.0),
            partial_rotary_factor: mf(m, "gemma.rope.partial_rotary_factor")
                .unwrap_or(1.0),
            softcap: mf(m, "gemma.final_logit_softcapping").unwrap_or(0.0),
            layer_types,
            attn_softcap: mf(m, "gemma.attention.attn_logit_softcapping").unwrap_or(0.0),
            q_pre_attn_scalar: mf(m, "gemma.attention.query_pre_attn_scalar").unwrap_or(0.0),
        })
    }
    pub fn is_full(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .map(|s| s == "full_attention")
            .unwrap_or(false)
    }
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        if self.is_full(layer) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }
    pub fn kv_shared(&self, layer: usize) -> bool {
        self.n_kv_shared > 0
            && layer >= self.n_layers - self.n_kv_shared
    }
}

pub struct Gemma4 {
    pub cfg: Gemma4Config,
    pub(crate) w: HashMap<String, Vec<f32>>,
}

/// Intermediates captured for staged oracle comparison against the HF
/// reference dump (embedding, selected layer outputs, final norm,
/// pre/post-softcap logits for the last position).
pub struct ForwardOut {
    pub embed: Vec<f32>,      // [seq, d]
    pub layer0: Vec<f32>,     // [seq, d]
    pub layer14: Vec<f32>,    // [seq, d]
    pub layer_last: Vec<f32>, // [seq, d]
    pub final_norm: Vec<f32>, // [seq, d]
    pub logits_pre: Vec<f32>, // [vocab] last position
    pub logits_post: Vec<f32>,
    // layer-0 sub-steps for localizing the first divergence
    pub l0_attn: Vec<f32>,
    pub l0_post_attn_norm: Vec<f32>,
    pub l0_mlp: Vec<f32>,
    pub l0_post_ffw_norm: Vec<f32>,
    pub l0_post_ple_norm: Vec<f32>,
}

pub(crate) fn gelu_tanh(x: f32) -> f32 {
    // gelu_pytorch_tanh
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * x * (1.0 + ((C * (x + 0.044715 * x * x * x)) as f32).tanh())
}

/// `y[o] = Σ_i x[i] * W[o,i]` for the decode hot path (seq=1).
/// Parallel over rows of W; NEON FMA on aarch64. All Gemma 4 inner
/// dims are multiples of 4, so the 4-lane main loop covers everything
/// with no tail handling. This is the engine behind every per-token
/// matvec in [`Gemma4::generate_cached`] (projections, MLP, LM head).
fn matvec_fast(x: &[f32], w: &[f32], y: &mut [f32], in_d: usize, out_d: usize) {
    debug_assert_eq!(x.len(), in_d);
    debug_assert_eq!(w.len(), out_d * in_d);
    debug_assert_eq!(y.len(), out_d);
    debug_assert!(in_d % 4 == 0, "in_d must be multiple of 4");

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(out_d.max(1));
    // small enough to not bother threading
    if out_d < 64 || nthreads <= 1 {
        for o in 0..out_d {
            y[o] = row_dot(x, &w[o * in_d..(o + 1) * in_d]);
        }
        return;
    }

    let chunk = out_d.div_ceil(nthreads);
    std::thread::scope(|sc| {
        for (i, ys) in y.chunks_mut(chunk).enumerate() {
            let row_start = i * chunk;
            let n = ys.len();
            let w_chunk = &w[row_start * in_d..(row_start + n) * in_d];
            sc.spawn(move || {
                for o in 0..n {
                    ys[o] = row_dot(x, &w_chunk[o * in_d..(o + 1) * in_d]);
                }
            });
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn row_dot(x: &[f32], w_row: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    debug_assert_eq!(n % 4, 0);
    // SAFETY: NEON is baseline-mandatory on aarch64; n%4==0 and both
    // slices have ≥n f32; pointers walk in-bounds 4 lanes at a time.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let px = x.as_ptr();
        let pw = w_row.as_ptr();
        let mut i = 0;
        // 16-wide main loop (Gemma 4 dims are all multiples of 16 too)
        while i + 16 <= n {
            let x0 = vld1q_f32(px.add(i));
            let x1 = vld1q_f32(px.add(i + 4));
            let x2 = vld1q_f32(px.add(i + 8));
            let x3 = vld1q_f32(px.add(i + 12));
            let w0 = vld1q_f32(pw.add(i));
            let w1 = vld1q_f32(pw.add(i + 4));
            let w2 = vld1q_f32(pw.add(i + 8));
            let w3 = vld1q_f32(pw.add(i + 12));
            acc0 = vfmaq_f32(acc0, x0, w0);
            acc1 = vfmaq_f32(acc1, x1, w1);
            acc2 = vfmaq_f32(acc2, x2, w2);
            acc3 = vfmaq_f32(acc3, x3, w3);
            i += 16;
        }
        // tail 4-wide
        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= n {
            let xv = vld1q_f32(px.add(i));
            let wv = vld1q_f32(pw.add(i));
            acc = vfmaq_f32(acc, xv, wv);
            i += 4;
        }
        vaddvq_f32(acc)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn row_dot(x: &[f32], w_row: &[f32]) -> f32 {
    let mut acc = 0f32;
    for i in 0..x.len() {
        acc += x[i] * w_row[i];
    }
    acc
}

/// Convenience: same shape as the original `linear` but seq=1 and
/// using [`matvec_fast`].
fn matvec(x: &[f32], in_d: usize, w: &[f32], out_d: usize) -> Vec<f32> {
    let mut y = vec![0f32; out_d];
    matvec_fast(x, w, &mut y, in_d, out_d);
    y
}

/// y[s,o] = sum_i x[s,i] * W[o,i]  (W is HF row-major [out,in]).
///
/// `seq==1` (the decode hot path) is dispatched to [`matvec_fast`]
/// (parallel + NEON FMA). `seq>1` (prefill in `forward`) keeps the
/// scalar reference loop — byte-for-byte identical to the verified
/// path, no surprise from reordering. The KV-cache parity oracle is
/// the cross-check that the fast and reference paths agree.
pub(crate) fn linear(x: &[f32], seq: usize, in_d: usize, w: &[f32], out_d: usize) -> Vec<f32> {
    if seq == 1 {
        return matvec(x, in_d, w, out_d);
    }
    let mut y = vec![0f32; seq * out_d];
    for s in 0..seq {
        let xr = &x[s * in_d..s * in_d + in_d];
        for o in 0..out_d {
            let wr = &w[o * in_d..o * in_d + in_d];
            let mut acc = 0f32;
            for i in 0..in_d {
                acc += xr[i] * wr[i];
            }
            y[s * out_d + o] = acc;
        }
    }
    y
}

/// RMSNorm over the last `d` dims. `weight` already carries Gemma's
/// `(1+w)` (folded by the adapter). `weight=None` → no-scale variant
/// (Gemma4 `v_norm`, `with_scale=False`).
pub(crate) fn rmsnorm(x: &mut [f32], rows: usize, d: usize, weight: Option<&[f32]>, eps: f32) {
    for r in 0..rows {
        let row = &mut x[r * d..r * d + d];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        match weight {
            Some(w) => {
                for i in 0..d {
                    row[i] = row[i] * inv * w[i];
                }
            }
            None => {
                for i in 0..d {
                    row[i] *= inv;
                }
            }
        }
    }
}

/// In-place rotate-half RoPE (NeoX/HF convention) on `[heads, seq, hd]`.
/// `inv_freq` has length `hd/2`; entries may be 0 (NoPE tail of Gemma 4
/// "proportional" rope → those dims pass through unchanged). Pairs
/// `(i, hd/2 + i)` over the full head, matching HF `rotate_half` with
/// duplicated cos/sin.
fn rope(x: &mut [f32], heads: usize, seq: usize, hd: usize, inv_freq: &[f32]) {
    let half = hd / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for h in 0..heads {
        for p in 0..seq {
            let base = (h * seq + p) * hd;
            for i in 0..half {
                let ang = p as f32 * inv_freq[i];
                let (sn, cs) = ang.sin_cos();
                let a = x[base + i];
                let b = x[base + half + i];
                x[base + i] = a * cs - b * sn;
                x[base + half + i] = b * cs + a * sn;
            }
        }
    }
}

/// Apply rotate-half RoPE to a single `[hd]` vector at absolute
/// position `pos`. Used by the incremental (KV-cached) decode path.
pub(crate) fn rope_one(x: &mut [f32], hd: usize, inv_freq: &[f32], pos: usize) {
    let half = hd / 2;
    let p = pos as f32;
    for i in 0..half {
        let (sn, cs) = (p * inv_freq[i]).sin_cos();
        let a = x[i];
        let b = x[half + i];
        x[i] = a * cs - b * sn;
        x[half + i] = b * cs + a * sn;
    }
}

/// Per-layer-type `inv_freq` of length `hd/2`, matching transformers
/// `_compute_default_rope_parameters` (sliding) and
/// `_compute_proportional_rope_parameters` (full-attention).
pub(crate) fn build_inv_freq(cfg: &Gemma4Config, layer: usize, hd: usize) -> Vec<f32> {
    let half = hd / 2;
    if cfg.is_full(layer) {
        // proportional: rope_angles = int(prf * head_dim // 2) real
        // freqs over base θ_full, then (half - rope_angles) zeros.
        let base = cfg.rope_theta_full;
        let ra = (((cfg.partial_rotary_factor * cfg.global_head_dim as f32)
            as i64)
            / 2) as usize;
        (0..half)
            .map(|i| {
                if i < ra {
                    base.powf(-(2.0 * i as f32) / hd as f32)
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        // default: full rotary over θ_sliding.
        let base = cfg.rope_theta_sliding;
        (0..half)
            .map(|i| base.powf(-(2.0 * i as f32) / hd as f32))
            .collect()
    }
}

impl Gemma4 {
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, ParseError> {
        let model = crate::gemma::load_gemma_dir(dir)?;
        let cfg = Gemma4Config::from_model(&model)?;
        let mut w = HashMap::new();
        for info in &model.tensors {
            let t = tensor_from_gguf(&model, info)?;
            w.insert(info.name.clone(), t.as_f32_vec());
        }
        Ok(Self { cfg, w })
    }

    fn g(&self, name: &str) -> &[f32] {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("gemma4: missing tensor {name}"))
    }

    pub fn forward(&self, ids: &[u32]) -> ForwardOut {
        let c = &self.cfg;
        let s = ids.len();
        let d = c.d_model;

        // 1. Scaled token embedding.
        let te = self.g("token_embd.weight"); // [vocab, d]
        let emb_scale = (d as f32).sqrt();
        let mut h = vec![0f32; s * d];
        for (t, &id) in ids.iter().enumerate() {
            let row = &te[id as usize * d..id as usize * d + d];
            for i in 0..d {
                h[t * d + i] = row[i] * emb_scale;
            }
        }
        let embed_dump = h.clone();

        // 2. Per-Layer Embeddings (PLE): [seq, n_layers, ple_dim].
        let nl = c.n_layers;
        let pd = c.ple_dim;
        let combined_ple: Vec<f32> = if pd > 0 {
            let tpl = self.g("per_layer_token_embd.weight"); // [vocab, nl*pd]
            let tpl_scale = (pd as f32).sqrt();
            // token-identity component
            let mut tok_id = vec![0f32; s * nl * pd];
            for (t, &id) in ids.iter().enumerate() {
                let row = &tpl[id as usize * nl * pd..id as usize * nl * pd + nl * pd];
                for j in 0..nl * pd {
                    tok_id[t * nl * pd + j] = row[j] * tpl_scale;
                }
            }
            // context-aware: proj(embed_scaled) * 1/sqrt(d), then RMS-norm
            let pmp = self.g("per_layer_model_proj.weight"); // [nl*pd, d]
            let mut ctx = linear(&embed_dump, s, d, pmp, nl * pd);
            let pscale = 1.0 / (d as f32).sqrt();
            for v in &mut ctx {
                *v *= pscale;
            }
            // per (token, layer) RMSNorm over pd with per_layer_proj_norm
            let pln = self.g("per_layer_proj_norm.weight"); // [pd] (1+w folded)
            rmsnorm(&mut ctx, s * nl, pd, Some(pln), c.rms_eps);
            // combine
            let comb_scale = 1.0 / 2f32.sqrt();
            let mut comb = vec![0f32; s * nl * pd];
            for k in 0..s * nl * pd {
                comb[k] = (ctx[k] + tok_id[k]) * comb_scale;
            }
            comb
        } else {
            Vec::new()
        };

        // KV-share store: last computed (k,v) per attention type.
        let mut shared_kv: HashMap<String, (Vec<f32>, Vec<f32>, usize)> =
            HashMap::new();

        let mut layer0 = Vec::new();
        let mut layer14 = Vec::new();
        let mut layer_last = Vec::new();
        let mut l0_attn = Vec::new();
        let mut l0_post_attn_norm = Vec::new();
        let mut l0_mlp = Vec::new();
        let mut l0_post_ffw_norm = Vec::new();
        let mut l0_post_ple_norm = Vec::new();

        for l in 0..nl {
            let is_full = c.is_full(l);
            let hd = c.layer_head_dim(l);
            let inv_freq = build_inv_freq(c, l, hd);
            let ltype = c.layer_types[l].clone();
            let double_wide = c.kv_shared(l);
            let inter = if double_wide { c.ffn * 2 } else { c.ffn };

            // ---- attention ----
            let res = h.clone();
            let mut xn = h.clone();
            rmsnorm(
                &mut xn,
                s,
                d,
                Some(self.g(&format!("blk.{l}.attn_norm.weight"))),
                c.rms_eps,
            );

            // q
            let q = linear(
                &xn,
                s,
                d,
                self.g(&format!("blk.{l}.attn_q.weight")),
                c.n_heads * hd,
            );
            // -> [heads, seq, hd]
            let mut qh = vec![0f32; c.n_heads * s * hd];
            for t in 0..s {
                for hh in 0..c.n_heads {
                    for i in 0..hd {
                        qh[(hh * s + t) * hd + i] = q[t * c.n_heads * hd + hh * hd + i];
                    }
                }
            }
            rmsnorm(
                &mut qh,
                c.n_heads * s,
                hd,
                Some(self.g(&format!("blk.{l}.attn_q_norm.weight"))),
                c.rms_eps,
            );
            rope(&mut qh, c.n_heads, s, hd, &inv_freq);
            // Gemma-2 q pre-attention scaling: Q /= sqrt(q_pre_attn_scalar).
            // Gemma-4 path: q_pre_attn_scalar=0 → no-op (q-norm handles it).
            if c.q_pre_attn_scalar > 0.0 {
                let q_scale = 1.0 / c.q_pre_attn_scalar.sqrt();
                for x in qh.iter_mut() { *x *= q_scale; }
            }

            // k,v (reuse if kv-shared)
            let (kh, vh) = if c.kv_shared(l) {
                let (k, v, _) = shared_kv
                    .get(&ltype)
                    .expect("kv-shared layer before any stored kv")
                    .clone();
                (k, v)
            } else {
                let k = linear(
                    &xn,
                    s,
                    d,
                    self.g(&format!("blk.{l}.attn_k.weight")),
                    c.n_kv * hd,
                );
                let v = linear(
                    &xn,
                    s,
                    d,
                    self.g(&format!("blk.{l}.attn_v.weight")),
                    c.n_kv * hd,
                );
                // [n_kv, seq, hd]
                let mut kh = vec![0f32; c.n_kv * s * hd];
                let mut vh = vec![0f32; c.n_kv * s * hd];
                for t in 0..s {
                    for kk in 0..c.n_kv {
                        for i in 0..hd {
                            kh[(kk * s + t) * hd + i] = k[t * c.n_kv * hd + kk * hd + i];
                            vh[(kk * s + t) * hd + i] = v[t * c.n_kv * hd + kk * hd + i];
                        }
                    }
                }
                rmsnorm(
                    &mut kh,
                    c.n_kv * s,
                    hd,
                    Some(self.g(&format!("blk.{l}.attn_k_norm.weight"))),
                    c.rms_eps,
                );
                rope(&mut kh, c.n_kv, s, hd, &inv_freq);
                // v_norm: with_scale=False (no weight)
                rmsnorm(&mut vh, c.n_kv * s, hd, None, c.rms_eps);
                (kh, vh)
            };
            // store kv for the last non-shared layer of this type
            if !c.kv_shared(l) {
                shared_kv.insert(ltype.clone(), (kh.clone(), vh.clone(), l));
            }

            // attention (MQA: kv head 0 shared across q heads), scaling=1.0
            let win = if is_full { 0 } else { c.sliding_window };
            let mut attn_out = vec![0f32; s * c.n_heads * hd];
            for hh in 0..c.n_heads {
                for i in 0..s {
                    // scores over j<=i (causal), sliding window if set
                    let j0 = if win > 0 && i + 1 > win { i + 1 - win } else { 0 };
                    let mut sc = vec![f32::NEG_INFINITY; s];
                    let mut mx = f32::NEG_INFINITY;
                    for j in j0..=i {
                        let mut acc = 0f32;
                        for x in 0..hd {
                            acc += qh[(hh * s + i) * hd + x] * kh[(0 * s + j) * hd + x];
                        }
                        // Gemma-2 attn-logit softcap: logits = sc·tanh(logits/sc).
                        // Gemma-4 path: attn_softcap=0 → no-op.
                        if c.attn_softcap > 0.0 {
                            acc = c.attn_softcap * (acc / c.attn_softcap).tanh();
                        }
                        sc[j] = acc; // scaling = 1.0 (q already pre-scaled for Gemma-2)
                        if acc > mx {
                            mx = acc;
                        }
                    }
                    let mut den = 0f32;
                    for j in j0..=i {
                        sc[j] = (sc[j] - mx).exp();
                        den += sc[j];
                    }
                    for x in 0..hd {
                        let mut acc = 0f32;
                        for j in j0..=i {
                            acc += sc[j] / den * vh[(0 * s + j) * hd + x];
                        }
                        attn_out[(i * c.n_heads + hh) * hd + x] = acc;
                    }
                }
            }
            // o_proj: [n_heads*hd -> d]
            let ao = linear(
                &attn_out,
                s,
                c.n_heads * hd,
                self.g(&format!("blk.{l}.attn_output.weight")),
                d,
            );
            // post_attention_norm then residual
            let mut ao = ao;
            if l == 0 {
                l0_attn = ao.clone();
            }
            rmsnorm(
                &mut ao,
                s,
                d,
                Some(self.g(&format!("blk.{l}.post_attention_norm.weight"))),
                c.rms_eps,
            );
            if l == 0 {
                l0_post_attn_norm = ao.clone();
            }
            for k in 0..s * d {
                h[k] = res[k] + ao[k];
            }

            // ---- MLP (GeGLU-tanh) ----
            let res2 = h.clone();
            let mut xn2 = h.clone();
            rmsnorm(
                &mut xn2,
                s,
                d,
                Some(self.g(&format!("blk.{l}.ffn_norm.weight"))),
                c.rms_eps,
            );
            let gate = linear(
                &xn2,
                s,
                d,
                self.g(&format!("blk.{l}.ffn_gate.weight")),
                inter,
            );
            let up = linear(
                &xn2,
                s,
                d,
                self.g(&format!("blk.{l}.ffn_up.weight")),
                inter,
            );
            let mut act = vec![0f32; s * inter];
            for k in 0..s * inter {
                act[k] = gelu_tanh(gate[k]) * up[k];
            }
            let mut down = linear(
                &act,
                s,
                inter,
                self.g(&format!("blk.{l}.ffn_down.weight")),
                d,
            );
            if l == 0 {
                l0_mlp = down.clone();
            }
            rmsnorm(
                &mut down,
                s,
                d,
                Some(self.g(&format!("blk.{l}.post_ffw_norm.weight"))),
                c.rms_eps,
            );
            if l == 0 {
                l0_post_ffw_norm = down.clone();
            }
            for k in 0..s * d {
                h[k] = res2[k] + down[k];
            }

            // ---- PLE residual block ----
            if pd > 0 {
                let res3 = h.clone();
                let mut gp = linear(
                    &h,
                    s,
                    d,
                    self.g(&format!("blk.{l}.per_layer_gate.weight")),
                    pd,
                );
                for v in &mut gp {
                    *v = gelu_tanh(*v);
                }
                // * this layer's PLE vector
                for t in 0..s {
                    for i in 0..pd {
                        gp[t * pd + i] *= combined_ple[(t * nl + l) * pd + i];
                    }
                }
                let mut pp = linear(
                    &gp,
                    s,
                    pd,
                    self.g(&format!("blk.{l}.per_layer_proj.weight")),
                    d,
                );
                rmsnorm(
                    &mut pp,
                    s,
                    d,
                    Some(self.g(&format!("blk.{l}.post_per_layer_norm.weight"))),
                    c.rms_eps,
                );
                if l == 0 {
                    l0_post_ple_norm = pp.clone();
                }
                for k in 0..s * d {
                    h[k] = res3[k] + pp[k];
                }
            }

            // ---- layer_scalar ----
            let ls = self.g(&format!("blk.{l}.layer_scalar"))[0];
            for v in &mut h {
                *v *= ls;
            }

            if l == 0 {
                layer0 = h.clone();
            }
            if l == 14 {
                layer14 = h.clone();
            }
            if l == nl - 1 {
                layer_last = h.clone();
            }
        }

        // final norm
        rmsnorm(
            &mut h,
            s,
            d,
            Some(self.g("output_norm.weight")),
            c.rms_eps,
        );
        let final_norm = h.clone();

        // tied, unscaled LM head — last position only
        let last = &h[(s - 1) * d..s * d];
        let mut logits_pre = vec![0f32; c.vocab];
        for o in 0..c.vocab {
            let wr = &te[o * d..o * d + d];
            let mut acc = 0f32;
            for i in 0..d {
                acc += last[i] * wr[i];
            }
            logits_pre[o] = acc;
        }
        let logits_post: Vec<f32> = if c.softcap > 0.0 {
            logits_pre
                .iter()
                .map(|&x| c.softcap * (x / c.softcap).tanh())
                .collect()
        } else {
            logits_pre.clone()
        };

        ForwardOut {
            embed: embed_dump,
            layer0,
            layer14,
            layer_last,
            final_norm,
            logits_pre,
            logits_post,
            l0_attn,
            l0_post_attn_norm,
            l0_mlp,
            l0_post_ffw_norm,
            l0_post_ple_norm,
        }
    }

    /// Encoder-style forward: returns just the final RMSNorm hidden state
    /// (pre-LM-head), shape `[seq, d_model]`. Used by SANA-WM and other
    /// downstream models that treat Gemma as a text encoder. Avoids the LM
    /// head + softcap computation entirely (we throw them away anyway).
    pub fn forward_hidden(&self, ids: &[u32]) -> Vec<f32> {
        self.forward(ids).final_norm
    }

    /// Greedy decode: append the post-softcap argmax `max_new` times,
    /// recomputing the full prefill each step (correctness-first; no KV
    /// cache yet). `argmax` is invariant under the monotonic
    /// `softcap·tanh` so this matches HF `generate(do_sample=False)`.
    /// Returns the newly generated token ids.
    pub fn generate(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let mut ids = prompt.to_vec();
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let o = self.forward(&ids);
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in o.logits_post.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            ids.push(best as u32);
            out.push(best as u32);
        }
        out
    }

    /// KV-cached greedy decode. Prefill and decode share one
    /// single-token loop (lower bug surface); per-layer K/V (post
    /// q/k-norm + RoPE, post v-norm) is cached so each step is O(1) in
    /// sequence length instead of an O(n) re-prefill. Shared layers
    /// (the last `num_kv_shared`) reference the cache of the last
    /// non-shared layer of the same attention type, matching HF
    /// `shared_kv_states`. Numerically identical to [`Self::generate`];
    /// oracle-gated by token-identity against it and HF.
    pub fn generate_cached(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let c = &self.cfg;
        let d = c.d_model;
        let nl = c.n_layers;
        let pd = c.ple_dim;
        let te = self.g("token_embd.weight");
        let emb_scale = (d as f32).sqrt();
        let first_shared = nl - c.n_kv_shared;

        // last non-shared layer index per attention type.
        let mut store_layer: HashMap<String, usize> = HashMap::new();
        for (idx, t) in c.layer_types.iter().take(first_shared).enumerate() {
            store_layer.insert(t.clone(), idx);
        }

        // per-layer K/V caches (only non-shared layers store).
        let mut kc: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let mut vc: Vec<Vec<f32>> = vec![Vec::new(); nl];

        // precompute per-layer constants
        let inv: Vec<Vec<f32>> =
            (0..nl).map(|l| build_inv_freq(c, l, c.layer_head_dim(l))).collect();

        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let total = prompt.len() + max_new;
        for pos in 0..total {
            if out.len() == max_new {
                break;
            }
            let id = if pos < prompt.len() {
                prompt[pos]
            } else {
                out[pos - prompt.len()]
            } as usize;

            // scaled embedding
            let mut h: Vec<f32> = te[id * d..id * d + d]
                .iter()
                .map(|v| v * emb_scale)
                .collect();

            // PLE vector for this token: [nl, pd]
            let ple: Vec<f32> = if pd > 0 {
                let tpl = self.g("per_layer_token_embd.weight");
                let tps = (pd as f32).sqrt();
                let tok_id: Vec<f32> = tpl[id * nl * pd..id * nl * pd + nl * pd]
                    .iter()
                    .map(|v| v * tps)
                    .collect();
                let pmp = self.g("per_layer_model_proj.weight");
                let mut ctx = linear(&h, 1, d, pmp, nl * pd);
                let psc = 1.0 / (d as f32).sqrt();
                for v in &mut ctx {
                    *v *= psc;
                }
                rmsnorm(&mut ctx, nl, pd, Some(self.g("per_layer_proj_norm.weight")), c.rms_eps);
                let cs = 1.0 / 2f32.sqrt();
                (0..nl * pd).map(|k| (ctx[k] + tok_id[k]) * cs).collect()
            } else {
                Vec::new()
            };

            for l in 0..nl {
                let is_full = c.is_full(l);
                let hd = c.layer_head_dim(l);
                let ltype = &c.layer_types[l];
                let inter = if c.kv_shared(l) { c.ffn * 2 } else { c.ffn };

                let res = h.clone();
                let mut xn = h.clone();
                rmsnorm(&mut xn, 1, d, Some(self.g(&format!("blk.{l}.attn_norm.weight"))), c.rms_eps);

                // q [n_heads, hd]
                let mut qh = linear(&xn, 1, d, self.g(&format!("blk.{l}.attn_q.weight")), c.n_heads * hd);
                rmsnorm(&mut qh, c.n_heads, hd, Some(self.g(&format!("blk.{l}.attn_q_norm.weight"))), c.rms_eps);
                for hh in 0..c.n_heads {
                    rope_one(&mut qh[hh * hd..hh * hd + hd], hd, &inv[l], pos);
                }
                // Gemma-2 q pre-attention scaling (cached path).
                if c.q_pre_attn_scalar > 0.0 {
                    let q_scale = 1.0 / c.q_pre_attn_scalar.sqrt();
                    for x in qh.iter_mut() { *x *= q_scale; }
                }

                // k/v: compute+cache for non-shared; reference store for shared
                let src = if l < first_shared {
                    let mut k = linear(&xn, 1, d, self.g(&format!("blk.{l}.attn_k.weight")), hd);
                    rmsnorm(&mut k, 1, hd, Some(self.g(&format!("blk.{l}.attn_k_norm.weight"))), c.rms_eps);
                    rope_one(&mut k, hd, &inv[l], pos);
                    let mut v = linear(&xn, 1, d, self.g(&format!("blk.{l}.attn_v.weight")), hd);
                    rmsnorm(&mut v, 1, hd, None, c.rms_eps); // v_norm: no scale
                    kc[l].extend_from_slice(&k);
                    vc[l].extend_from_slice(&v);
                    l
                } else {
                    *store_layer.get(ltype).expect("shared layer type stored")
                };
                let kk = &kc[src];
                let vv = &vc[src];
                let clen = kk.len() / hd;

                let win = if is_full { 0 } else { c.sliding_window };
                let j0 = if win > 0 && pos + 1 > win { pos + 1 - win } else { 0 };

                let mut attn = vec![0f32; c.n_heads * hd];
                for hh in 0..c.n_heads {
                    let qv = &qh[hh * hd..hh * hd + hd];
                    let mut sc = vec![f32::NEG_INFINITY; clen];
                    let mut mx = f32::NEG_INFINITY;
                    for j in j0..clen {
                        let kr = &kk[j * hd..j * hd + hd];
                        let mut a = 0f32;
                        for x in 0..hd {
                            a += qv[x] * kr[x];
                        }
                        // Gemma-2 attn-logit softcap (cached path).
                        if c.attn_softcap > 0.0 {
                            a = c.attn_softcap * (a / c.attn_softcap).tanh();
                        }
                        sc[j] = a;
                        if a > mx {
                            mx = a;
                        }
                    }
                    let mut den = 0f32;
                    for j in j0..clen {
                        sc[j] = (sc[j] - mx).exp();
                        den += sc[j];
                    }
                    let o = &mut attn[hh * hd..hh * hd + hd];
                    for j in j0..clen {
                        let w = sc[j] / den;
                        let vr = &vv[j * hd..j * hd + hd];
                        for x in 0..hd {
                            o[x] += w * vr[x];
                        }
                    }
                }
                let mut ao = linear(&attn, 1, c.n_heads * hd, self.g(&format!("blk.{l}.attn_output.weight")), d);
                rmsnorm(&mut ao, 1, d, Some(self.g(&format!("blk.{l}.post_attention_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res[k] + ao[k];
                }

                let res2 = h.clone();
                let mut xn2 = h.clone();
                rmsnorm(&mut xn2, 1, d, Some(self.g(&format!("blk.{l}.ffn_norm.weight"))), c.rms_eps);
                let gate = linear(&xn2, 1, d, self.g(&format!("blk.{l}.ffn_gate.weight")), inter);
                let up = linear(&xn2, 1, d, self.g(&format!("blk.{l}.ffn_up.weight")), inter);
                let act: Vec<f32> =
                    (0..inter).map(|k| gelu_tanh(gate[k]) * up[k]).collect();
                let mut down = linear(&act, 1, inter, self.g(&format!("blk.{l}.ffn_down.weight")), d);
                rmsnorm(&mut down, 1, d, Some(self.g(&format!("blk.{l}.post_ffw_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res2[k] + down[k];
                }

                if pd > 0 {
                    let res3 = h.clone();
                    let mut gp = linear(&h, 1, d, self.g(&format!("blk.{l}.per_layer_gate.weight")), pd);
                    for (i, v) in gp.iter_mut().enumerate() {
                        *v = gelu_tanh(*v) * ple[l * pd + i];
                    }
                    let mut pp = linear(&gp, 1, pd, self.g(&format!("blk.{l}.per_layer_proj.weight")), d);
                    rmsnorm(&mut pp, 1, d, Some(self.g(&format!("blk.{l}.post_per_layer_norm.weight"))), c.rms_eps);
                    for k in 0..d {
                        h[k] = res3[k] + pp[k];
                    }
                }

                let ls = self.g(&format!("blk.{l}.layer_scalar"))[0];
                for v in &mut h {
                    *v *= ls;
                }
            }

            // emit once we're at/after the last prompt position
            if pos + 1 >= prompt.len() {
                rmsnorm(&mut h, 1, d, Some(self.g("output_norm.weight")), c.rms_eps);
                let logits = matvec(&h, d, te, c.vocab);
                // argmax invariant under monotonic softcap
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (o, &a) in logits.iter().enumerate() {
                    if a > bv {
                        bv = a;
                        best = o;
                    }
                }
                out.push(best as u32);
            }
        }
        out.truncate(max_new);
        out
    }
}

