//! Per-row symmetric int8 quantization of dense Linear weights, plus
//! the matching [`Gemma4Q8`] model: same arithmetic as the verified
//! [`crate::gemma4::Gemma4`] forward, every heavy linear (q/k/v/o,
//! gate/up/down, per-layer gate/proj, per-layer model proj, the tied
//! LM head, and the PLE token-identity table) routed through
//! [`matvec_q8`]. Norms and `layer_scalar` stay f32 (tiny tensors).
//!
//!   q[o,i] = round(W[o,i] / scale[o]).clamp(-127, 127)
//!   scale[o] = max_i |W[o,i]| / 127
//!   W[o,i]  ≈ q[o,i] * scale[o]
//!
//! Memory: ~10 GB → ~3 GB on E2B. Compute: 16 i8 per NEON cycle (vs 4
//! f32) and 4× less memory bandwidth. Oracle: argmax of the first
//! generated token still matches HF (id 9079 = " Paris") despite per-
//! row rounding — the margin to runner-up is wide (22.51 vs 21.80).

/// A Linear weight stored as i8 rows + per-row f32 scales.
pub struct Q8Weight {
    pub q: Vec<i8>,     // [out_d * in_d] row-major
    pub scale: Vec<f32>, // [out_d]
    pub in_d: usize,
    pub out_d: usize,
}

impl Q8Weight {
    /// Quantize a row-major f32 weight `[out_d, in_d]`. Empty rows
    /// (all zero) get scale 0 → dequant produces all-zero, faithful.
    pub fn quantize(w: &[f32], in_d: usize, out_d: usize) -> Self {
        debug_assert_eq!(w.len(), in_d * out_d);
        let mut q = vec![0i8; in_d * out_d];
        let mut scale = vec![0f32; out_d];
        for o in 0..out_d {
            let row = &w[o * in_d..(o + 1) * in_d];
            let mx = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if mx > 0.0 { mx / 127.0 } else { 0.0 };
            scale[o] = s;
            if s > 0.0 {
                let inv = 1.0 / s;
                let dst = &mut q[o * in_d..(o + 1) * in_d];
                for i in 0..in_d {
                    let r = (row[i] * inv).round();
                    dst[i] = r.clamp(-127.0, 127.0) as i8;
                }
            }
        }
        Self { q, scale, in_d, out_d }
    }
}

/// `y[o] = scale[o] · Σ_i x[i] · (q[o,i] as f32)`. Parallel over
/// rows; per-row dequant into a small scratch, then f32 dot via the
/// same NEON path the f32 matvec uses (here inlined as a tight FMA
/// loop). Inner dim must be a multiple of 4 (all Gemma 4 dims are).
pub fn matvec_q8(x: &[f32], w: &Q8Weight, y: &mut [f32]) {
    let in_d = w.in_d;
    let out_d = w.out_d;
    debug_assert_eq!(x.len(), in_d);
    debug_assert_eq!(y.len(), out_d);
    debug_assert!(in_d % 4 == 0);

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(out_d.max(1));
    if out_d < 64 || nthreads <= 1 {
        for o in 0..out_d {
            y[o] = row_dot_q8(x, &w.q[o * in_d..(o + 1) * in_d], w.scale[o]);
        }
        return;
    }
    let chunk = out_d.div_ceil(nthreads);
    std::thread::scope(|sc| {
        for (i, ys) in y.chunks_mut(chunk).enumerate() {
            let row_start = i * chunk;
            let n = ys.len();
            let q_chunk = &w.q[row_start * in_d..(row_start + n) * in_d];
            let s_chunk = &w.scale[row_start..row_start + n];
            sc.spawn(move || {
                for o in 0..n {
                    ys[o] = row_dot_q8(
                        x,
                        &q_chunk[o * in_d..(o + 1) * in_d],
                        s_chunk[o],
                    );
                }
            });
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn row_dot_q8(x: &[f32], q_row: &[i8], scale: f32) -> f32 {
    use std::arch::aarch64::*;
    if scale == 0.0 {
        return 0.0;
    }
    let n = x.len();
    debug_assert_eq!(q_row.len(), n);
    debug_assert_eq!(n % 4, 0);
    // SAFETY: NEON baseline on aarch64; n%4==0; pointers in-bounds.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let px = x.as_ptr();
        let pq = q_row.as_ptr();
        let mut i = 0;
        // 16-wide main loop: one vld1q_s8 (16 i8), widen to two i16x8,
        // each to two i32x4 then to f32x4; FMA with x.
        while i + 16 <= n {
            let b = vld1q_s8(pq.add(i));
            let lo16 = vmovl_s8(vget_low_s8(b));   // i16x8
            let hi16 = vmovl_s8(vget_high_s8(b));
            let q0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            let q1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
            let q2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            let q3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
            acc0 = vfmaq_f32(acc0, vld1q_f32(px.add(i)),      q0);
            acc1 = vfmaq_f32(acc1, vld1q_f32(px.add(i + 4)),  q1);
            acc2 = vfmaq_f32(acc2, vld1q_f32(px.add(i + 8)),  q2);
            acc3 = vfmaq_f32(acc3, vld1q_f32(px.add(i + 12)), q3);
            i += 16;
        }
        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        // 4-wide tail
        while i + 4 <= n {
            let q = vld1_s8(pq.add(i));               // 8 i8 (only low 4 used)
            let q16 = vget_low_s16(vmovl_s8(q));      // 4 i16
            let qf = vcvtq_f32_s32(vmovl_s16(q16));   // 4 f32
            acc = vfmaq_f32(acc, vld1q_f32(px.add(i)), qf);
            i += 4;
        }
        vaddvq_f32(acc) * scale
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn row_dot_q8(x: &[f32], q_row: &[i8], scale: f32) -> f32 {
    if scale == 0.0 {
        return 0.0;
    }
    let mut acc = 0f32;
    for i in 0..x.len() {
        acc += x[i] * q_row[i] as f32;
    }
    acc * scale
}

// ============================================================
// Gemma4Q8 — int8-quantized Gemma 4 with mirroring cached forward.
// ============================================================

use std::collections::HashMap;
use std::path::Path;

use crate::gemma4::{
    build_inv_freq, gelu_tanh, rmsnorm, rope_one, Gemma4, Gemma4Config,
};
use crate::ParseError;

/// Shape `(out_d, in_d)` of every linear we quantize, derived from
/// `cfg`. `None` ⇒ the named tensor is not a heavy linear (norm,
/// scalar) and stays f32.
fn linear_shape(name: &str, cfg: &Gemma4Config) -> Option<(usize, usize)> {
    let d = cfg.d_model;
    let pd = cfg.ple_dim;
    let nl = cfg.n_layers;
    match name {
        "token_embd.weight" => return Some((cfg.vocab, d)),
        "per_layer_token_embd.weight" => return Some((cfg.vocab, nl * pd)),
        "per_layer_model_proj.weight" => return Some((nl * pd, d)),
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("blk.") {
        let dot = rest.find('.')?;
        let l: usize = rest[..dot].parse().ok()?;
        let tail = &rest[dot + 1..];
        let hd = cfg.layer_head_dim(l);
        let inter = if cfg.kv_shared(l) { cfg.ffn * 2 } else { cfg.ffn };
        return match tail {
            "attn_q.weight" => Some((cfg.n_heads * hd, d)),
            "attn_k.weight" | "attn_v.weight" => Some((cfg.n_kv * hd, d)),
            "attn_output.weight" => Some((d, cfg.n_heads * hd)),
            "ffn_gate.weight" | "ffn_up.weight" => Some((inter, d)),
            "ffn_down.weight" => Some((d, inter)),
            "per_layer_gate.weight" => Some((pd, d)),
            "per_layer_proj.weight" => Some((d, pd)),
            _ => None,
        };
    }
    None
}

pub struct Gemma4Q8 {
    pub cfg: Gemma4Config,
    qw: HashMap<String, Q8Weight>,
    fw: HashMap<String, Vec<f32>>,
}

impl Gemma4Q8 {
    /// Convenience: load from a Gemma 4 HF snapshot directory, then
    /// in-memory quantize the heavy linears.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, ParseError> {
        let g = Gemma4::load(dir)?;
        Ok(Self::from_gemma4(&g))
    }

    /// In-memory int8 quantize the heavy linears of an already-loaded
    /// f32 [`Gemma4`]. The original `g` is untouched.
    pub fn from_gemma4(g: &Gemma4) -> Self {
        let cfg = g.cfg.clone();
        let mut qw = HashMap::new();
        let mut fw = HashMap::new();
        for (name, w) in &g.w {
            if let Some((out_d, in_d)) = linear_shape(name, &cfg) {
                qw.insert(name.clone(), Q8Weight::quantize(w, in_d, out_d));
            } else {
                fw.insert(name.clone(), w.clone());
            }
        }
        Self { cfg, qw, fw }
    }

    fn qg(&self, n: &str) -> &Q8Weight {
        self.qw
            .get(n)
            .unwrap_or_else(|| panic!("gemma4_q8: missing q-weight {n}"))
    }
    fn fg(&self, n: &str) -> &[f32] {
        self.fw
            .get(n)
            .unwrap_or_else(|| panic!("gemma4_q8: missing f-weight {n}"))
    }

    /// Dequant one row of a Q8 weight at `id` into a fresh `Vec<f32>`.
    /// Used for token embedding lookups (Q8 `token_embd` /
    /// `per_layer_token_embd`).
    fn dequant_row(&self, w: &Q8Weight, id: usize) -> Vec<f32> {
        let s = w.scale[id];
        let row = &w.q[id * w.in_d..(id + 1) * w.in_d];
        if s == 0.0 {
            return vec![0f32; w.in_d];
        }
        row.iter().map(|&q| q as f32 * s).collect()
    }

    /// KV-cached greedy decode (Q8). Mirrors
    /// [`Gemma4::generate_cached`](crate::gemma4::Gemma4::generate_cached)
    /// exactly; the only change is every heavy linear routes through
    /// [`matvec_q8`].
    pub fn generate_cached(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let c = &self.cfg;
        let d = c.d_model;
        let nl = c.n_layers;
        let pd = c.ple_dim;
        let emb_scale = (d as f32).sqrt();
        let first_shared = nl - c.n_kv_shared;

        let mut store_layer: HashMap<String, usize> = HashMap::new();
        for (idx, t) in c.layer_types.iter().take(first_shared).enumerate() {
            store_layer.insert(t.clone(), idx);
        }

        let mut kc: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let mut vc: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let inv: Vec<Vec<f32>> =
            (0..nl).map(|l| build_inv_freq(c, l, c.layer_head_dim(l))).collect();

        let te_q = self.qg("token_embd.weight");
        let tpl_q = if pd > 0 { Some(self.qg("per_layer_token_embd.weight")) } else { None };
        let pmp_q = if pd > 0 { Some(self.qg("per_layer_model_proj.weight")) } else { None };

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

            // scaled embedding via Q8 row lookup
            let mut h: Vec<f32> = self.dequant_row(te_q, id);
            for v in &mut h {
                *v *= emb_scale;
            }

            // PLE for this token
            let ple: Vec<f32> = if pd > 0 {
                let tps = (pd as f32).sqrt();
                let mut tok_id = self.dequant_row(tpl_q.unwrap(), id);
                for v in &mut tok_id {
                    *v *= tps;
                }
                let mut ctx = vec![0f32; nl * pd];
                matvec_q8(&h, pmp_q.unwrap(), &mut ctx);
                let psc = 1.0 / (d as f32).sqrt();
                for v in &mut ctx {
                    *v *= psc;
                }
                rmsnorm(&mut ctx, nl, pd, Some(self.fg("per_layer_proj_norm.weight")), c.rms_eps);
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
                rmsnorm(&mut xn, 1, d, Some(self.fg(&format!("blk.{l}.attn_norm.weight"))), c.rms_eps);

                let mut qh = vec![0f32; c.n_heads * hd];
                matvec_q8(&xn, self.qg(&format!("blk.{l}.attn_q.weight")), &mut qh);
                rmsnorm(&mut qh, c.n_heads, hd, Some(self.fg(&format!("blk.{l}.attn_q_norm.weight"))), c.rms_eps);
                for hh in 0..c.n_heads {
                    rope_one(&mut qh[hh * hd..hh * hd + hd], hd, &inv[l], pos);
                }

                let src = if l < first_shared {
                    let mut k = vec![0f32; hd];
                    matvec_q8(&xn, self.qg(&format!("blk.{l}.attn_k.weight")), &mut k);
                    rmsnorm(&mut k, 1, hd, Some(self.fg(&format!("blk.{l}.attn_k_norm.weight"))), c.rms_eps);
                    rope_one(&mut k, hd, &inv[l], pos);
                    let mut v = vec![0f32; hd];
                    matvec_q8(&xn, self.qg(&format!("blk.{l}.attn_v.weight")), &mut v);
                    rmsnorm(&mut v, 1, hd, None, c.rms_eps);
                    kc[l].extend_from_slice(&k);
                    vc[l].extend_from_slice(&v);
                    l
                } else {
                    *store_layer.get(ltype).expect("kv-shared layer type stored")
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
                        sc[j] = a;
                        if a > mx { mx = a; }
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
                let mut ao = vec![0f32; d];
                matvec_q8(&attn, self.qg(&format!("blk.{l}.attn_output.weight")), &mut ao);
                rmsnorm(&mut ao, 1, d, Some(self.fg(&format!("blk.{l}.post_attention_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res[k] + ao[k];
                }

                let res2 = h.clone();
                let mut xn2 = h.clone();
                rmsnorm(&mut xn2, 1, d, Some(self.fg(&format!("blk.{l}.ffn_norm.weight"))), c.rms_eps);
                let mut gate = vec![0f32; inter];
                matvec_q8(&xn2, self.qg(&format!("blk.{l}.ffn_gate.weight")), &mut gate);
                let mut up = vec![0f32; inter];
                matvec_q8(&xn2, self.qg(&format!("blk.{l}.ffn_up.weight")), &mut up);
                let act: Vec<f32> =
                    (0..inter).map(|k| gelu_tanh(gate[k]) * up[k]).collect();
                let mut down = vec![0f32; d];
                matvec_q8(&act, self.qg(&format!("blk.{l}.ffn_down.weight")), &mut down);
                rmsnorm(&mut down, 1, d, Some(self.fg(&format!("blk.{l}.post_ffw_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res2[k] + down[k];
                }

                if pd > 0 {
                    let res3 = h.clone();
                    let mut gp = vec![0f32; pd];
                    matvec_q8(&h, self.qg(&format!("blk.{l}.per_layer_gate.weight")), &mut gp);
                    for (i, v) in gp.iter_mut().enumerate() {
                        *v = gelu_tanh(*v) * ple[l * pd + i];
                    }
                    let mut pp = vec![0f32; d];
                    matvec_q8(&gp, self.qg(&format!("blk.{l}.per_layer_proj.weight")), &mut pp);
                    rmsnorm(&mut pp, 1, d, Some(self.fg(&format!("blk.{l}.post_per_layer_norm.weight"))), c.rms_eps);
                    for k in 0..d {
                        h[k] = res3[k] + pp[k];
                    }
                }

                let ls = self.fg(&format!("blk.{l}.layer_scalar"))[0];
                for v in &mut h {
                    *v *= ls;
                }
            }

            if pos + 1 >= prompt.len() {
                rmsnorm(&mut h, 1, d, Some(self.fg("output_norm.weight")), c.rms_eps);
                // tied unscaled LM head, Q8
                let mut logits = vec![0f32; c.vocab];
                matvec_q8(&h, te_q, &mut logits);
                // argmax invariant under monotonic softcap
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (o, &a) in logits.iter().enumerate() {
                    if a > bv { bv = a; best = o; }
                }
                out.push(best as u32);
            }
        }
        out.truncate(max_new);
        out
    }

    /// Streaming variant of [`Self::generate_cached`]. Identical
    /// arithmetic; invokes `cb(id)` for each newly generated token
    /// **as it lands**, before the next decode step starts. Returns
    /// `false` from `cb` to stop early. Used by `jq --stream`.
    pub fn generate_cached_stream<F>(
        &self,
        prompt: &[u32],
        max_new: usize,
        mut cb: F,
    ) -> Vec<u32>
    where
        F: FnMut(u32) -> bool,
    {
        // Reuse generate_cached's logic verbatim — the only delta is
        // an `if let Some(&last) = out.last() { if !cb(last) { break; } }`
        // after each push. Implemented as a fresh inline loop to keep
        // the verified `generate_cached` untouched.
        let c = &self.cfg;
        let d = c.d_model;
        let nl = c.n_layers;
        let pd = c.ple_dim;
        let emb_scale = (d as f32).sqrt();
        let first_shared = nl - c.n_kv_shared;

        let mut store_layer: HashMap<String, usize> = HashMap::new();
        for (idx, t) in c.layer_types.iter().take(first_shared).enumerate() {
            store_layer.insert(t.clone(), idx);
        }

        let mut kc: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let mut vc: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let inv: Vec<Vec<f32>> = (0..nl)
            .map(|l| build_inv_freq(c, l, c.layer_head_dim(l)))
            .collect();

        let te_q = self.qg("token_embd.weight");
        let tpl_q = if pd > 0 { Some(self.qg("per_layer_token_embd.weight")) } else { None };
        let pmp_q = if pd > 0 { Some(self.qg("per_layer_model_proj.weight")) } else { None };

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

            let mut h: Vec<f32> = self.dequant_row(te_q, id);
            for v in &mut h {
                *v *= emb_scale;
            }

            let ple: Vec<f32> = if pd > 0 {
                let tps = (pd as f32).sqrt();
                let mut tok_id = self.dequant_row(tpl_q.unwrap(), id);
                for v in &mut tok_id {
                    *v *= tps;
                }
                let mut ctx = vec![0f32; nl * pd];
                matvec_q8(&h, pmp_q.unwrap(), &mut ctx);
                let psc = 1.0 / (d as f32).sqrt();
                for v in &mut ctx {
                    *v *= psc;
                }
                rmsnorm(&mut ctx, nl, pd, Some(self.fg("per_layer_proj_norm.weight")), c.rms_eps);
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
                rmsnorm(&mut xn, 1, d, Some(self.fg(&format!("blk.{l}.attn_norm.weight"))), c.rms_eps);

                let mut qh = vec![0f32; c.n_heads * hd];
                matvec_q8(&xn, self.qg(&format!("blk.{l}.attn_q.weight")), &mut qh);
                rmsnorm(&mut qh, c.n_heads, hd, Some(self.fg(&format!("blk.{l}.attn_q_norm.weight"))), c.rms_eps);
                for hh in 0..c.n_heads {
                    rope_one(&mut qh[hh * hd..hh * hd + hd], hd, &inv[l], pos);
                }

                let src = if l < first_shared {
                    let mut k = vec![0f32; hd];
                    matvec_q8(&xn, self.qg(&format!("blk.{l}.attn_k.weight")), &mut k);
                    rmsnorm(&mut k, 1, hd, Some(self.fg(&format!("blk.{l}.attn_k_norm.weight"))), c.rms_eps);
                    rope_one(&mut k, hd, &inv[l], pos);
                    let mut v = vec![0f32; hd];
                    matvec_q8(&xn, self.qg(&format!("blk.{l}.attn_v.weight")), &mut v);
                    rmsnorm(&mut v, 1, hd, None, c.rms_eps);
                    kc[l].extend_from_slice(&k);
                    vc[l].extend_from_slice(&v);
                    l
                } else {
                    *store_layer.get(ltype).expect("kv-shared layer type stored")
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
                        sc[j] = a;
                        if a > mx { mx = a; }
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
                let mut ao = vec![0f32; d];
                matvec_q8(&attn, self.qg(&format!("blk.{l}.attn_output.weight")), &mut ao);
                rmsnorm(&mut ao, 1, d, Some(self.fg(&format!("blk.{l}.post_attention_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res[k] + ao[k];
                }

                let res2 = h.clone();
                let mut xn2 = h.clone();
                rmsnorm(&mut xn2, 1, d, Some(self.fg(&format!("blk.{l}.ffn_norm.weight"))), c.rms_eps);
                let mut gate = vec![0f32; inter];
                matvec_q8(&xn2, self.qg(&format!("blk.{l}.ffn_gate.weight")), &mut gate);
                let mut up = vec![0f32; inter];
                matvec_q8(&xn2, self.qg(&format!("blk.{l}.ffn_up.weight")), &mut up);
                let act: Vec<f32> =
                    (0..inter).map(|k| gelu_tanh(gate[k]) * up[k]).collect();
                let mut down = vec![0f32; d];
                matvec_q8(&act, self.qg(&format!("blk.{l}.ffn_down.weight")), &mut down);
                rmsnorm(&mut down, 1, d, Some(self.fg(&format!("blk.{l}.post_ffw_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res2[k] + down[k];
                }

                if pd > 0 {
                    let res3 = h.clone();
                    let mut gp = vec![0f32; pd];
                    matvec_q8(&h, self.qg(&format!("blk.{l}.per_layer_gate.weight")), &mut gp);
                    for (i, v) in gp.iter_mut().enumerate() {
                        *v = gelu_tanh(*v) * ple[l * pd + i];
                    }
                    let mut pp = vec![0f32; d];
                    matvec_q8(&gp, self.qg(&format!("blk.{l}.per_layer_proj.weight")), &mut pp);
                    rmsnorm(&mut pp, 1, d, Some(self.fg(&format!("blk.{l}.post_per_layer_norm.weight"))), c.rms_eps);
                    for k in 0..d {
                        h[k] = res3[k] + pp[k];
                    }
                }

                let ls = self.fg(&format!("blk.{l}.layer_scalar"))[0];
                for v in &mut h {
                    *v *= ls;
                }
            }

            if pos + 1 >= prompt.len() {
                rmsnorm(&mut h, 1, d, Some(self.fg("output_norm.weight")), c.rms_eps);
                let mut logits = vec![0f32; c.vocab];
                matvec_q8(&h, te_q, &mut logits);
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (o, &a) in logits.iter().enumerate() {
                    if a > bv { bv = a; best = o; }
                }
                let best_u32 = best as u32;
                out.push(best_u32);
                if !cb(best_u32) {
                    break;
                }
            }
        }
        out.truncate(max_new);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_then_matvec_round_trips_small() {
        // Synthetic 8×64 matrix, random-ish but deterministic.
        let out = 8usize;
        let inn = 64usize;
        let w: Vec<f32> = (0..out * inn)
            .map(|i| (i as f32 * 0.131).sin() * 0.5)
            .collect();
        let x: Vec<f32> = (0..inn).map(|i| (i as f32 * 0.17).cos()).collect();

        // f32 reference
        let mut yref = vec![0f32; out];
        for o in 0..out {
            let mut a = 0f32;
            for i in 0..inn {
                a += x[i] * w[o * inn + i];
            }
            yref[o] = a;
        }

        let q = Q8Weight::quantize(&w, inn, out);
        let mut yq = vec![0f32; out];
        matvec_q8(&x, &q, &mut yq);

        // Per-row max-abs ≤ 1.0/127 of the row's max-abs scale.
        for o in 0..out {
            let err = (yref[o] - yq[o]).abs();
            let bound = q.scale[o] * (inn as f32).sqrt() * 0.6; // generous
            assert!(
                err < bound.max(1e-3),
                "row {o}: yref={} yq={} err={} bound={}",
                yref[o], yq[o], err, bound
            );
        }
    }
}
