//! Per-row, per-group symmetric int8 quantization (G=32). The
//! per-group sibling of `gemma4_q8` (per-row only). Same i8 storage,
//! but each row is sliced into `in_d / G` groups with its own f32
//! scale. Tightens dynamic range per group → less rounding error on
//! columns whose magnitude is dwarfed by a per-row max-abs. Closes
//! the documented Q8 limitation (8th-token drift on the longest
//! prompt at max_new=8) and is the architecturally correct primitive
//! for any future GPTQ-style calibrated int4.
//!
//!   q[o, g, k] = round(W[o, g·G + k] / scale[o, g]).clamp(-127, 127)
//!   scale[o, g] = max_k |W[o, g·G + k]| / 127
//!   W[o, i]    ≈ q[o, i] · scale[o, i / G]
//!
//! Storage: 1 B/param + 4 B/scale. At G=32, in_d=1536 → ~1.083 B/param
//! (vs 1 B for per-row Q8, 4 B for f32). Same RAM as per-row Q8 to
//! within 8%, but visibly better numerical behaviour on long
//! generations.

const G: usize = 32;

pub struct Q8GroupWeight {
    /// `[out_d * in_d]` row-major i8.
    pub q: Vec<i8>,
    /// Per-group scales, `[out_d * (in_d / G)]`.
    pub scale: Vec<f32>,
    pub in_d: usize,
    pub out_d: usize,
}

impl Q8GroupWeight {
    pub fn quantize(w: &[f32], in_d: usize, out_d: usize) -> Self {
        debug_assert_eq!(w.len(), in_d * out_d);
        debug_assert!(in_d % G == 0);
        let gpr = in_d / G;
        let mut q = vec![0i8; in_d * out_d];
        let mut scale = vec![0f32; out_d * gpr];
        for o in 0..out_d {
            let row = &w[o * in_d..(o + 1) * in_d];
            for g in 0..gpr {
                let blk = &row[g * G..(g + 1) * G];
                let mx = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let s = if mx > 0.0 { mx / 127.0 } else { 0.0 };
                scale[o * gpr + g] = s;
                if s > 0.0 {
                    let inv = 1.0 / s;
                    let dst = &mut q[o * in_d + g * G..o * in_d + (g + 1) * G];
                    for k in 0..G {
                        dst[k] = (blk[k] * inv)
                            .round()
                            .clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }
        Self { q, scale, in_d, out_d }
    }
}

pub fn matvec_q8g(x: &[f32], w: &Q8GroupWeight, y: &mut [f32]) {
    let in_d = w.in_d;
    let out_d = w.out_d;
    let gpr = in_d / G;
    debug_assert_eq!(x.len(), in_d);
    debug_assert_eq!(y.len(), out_d);
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(out_d.max(1));
    if out_d < 64 || nthreads <= 1 {
        for o in 0..out_d {
            y[o] = row_dot(
                x,
                &w.q[o * in_d..(o + 1) * in_d],
                &w.scale[o * gpr..(o + 1) * gpr],
            );
        }
        return;
    }
    let chunk = out_d.div_ceil(nthreads);
    std::thread::scope(|sc| {
        for (i, ys) in y.chunks_mut(chunk).enumerate() {
            let row_start = i * chunk;
            let n = ys.len();
            let q_chunk = &w.q[row_start * in_d..(row_start + n) * in_d];
            let s_chunk = &w.scale[row_start * gpr..(row_start + n) * gpr];
            sc.spawn(move || {
                for o in 0..n {
                    ys[o] = row_dot(
                        x,
                        &q_chunk[o * in_d..(o + 1) * in_d],
                        &s_chunk[o * gpr..(o + 1) * gpr],
                    );
                }
            });
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn row_dot(x: &[f32], q_row: &[i8], scales: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    let gpr = scales.len();
    debug_assert_eq!(q_row.len(), n);
    debug_assert_eq!(n, gpr * G);
    let px = x.as_ptr();
    let pq = q_row.as_ptr();
    // Single global f32 accumulator across all groups: each chunk's
    // i8s are widened to f32, multiplied by the group's scale (so the
    // FMA chain sees pre-scaled `q*s` weights), then accumulated into
    // `acc`. One horizontal sum at the end — vs 48 per row in the
    // earlier per-group-then-add formulation.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        for g in 0..gpr {
            let s = scales[g];
            if s == 0.0 {
                continue;
            }
            let s_v = vdupq_n_f32(s);
            let base = g * G;

            // 16-byte chunk 0 (elements 0..16 of the group)
            let b0 = vld1q_s8(pq.add(base));
            let lo16_0 = vmovl_s8(vget_low_s8(b0));
            let hi16_0 = vmovl_s8(vget_high_s8(b0));
            let q00 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))), s_v);
            let q01 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0))), s_v);
            let q02 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))), s_v);
            let q03 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_0))), s_v);
            acc0 = vfmaq_f32(acc0, vld1q_f32(px.add(base)), q00);
            acc1 = vfmaq_f32(acc1, vld1q_f32(px.add(base + 4)), q01);
            acc2 = vfmaq_f32(acc2, vld1q_f32(px.add(base + 8)), q02);
            acc3 = vfmaq_f32(acc3, vld1q_f32(px.add(base + 12)), q03);

            // 16-byte chunk 1 (elements 16..32 of the group)
            let b1 = vld1q_s8(pq.add(base + 16));
            let lo16_1 = vmovl_s8(vget_low_s8(b1));
            let hi16_1 = vmovl_s8(vget_high_s8(b1));
            let q10 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))), s_v);
            let q11 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1))), s_v);
            let q12 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))), s_v);
            let q13 = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_1))), s_v);
            acc0 = vfmaq_f32(acc0, vld1q_f32(px.add(base + 16)), q10);
            acc1 = vfmaq_f32(acc1, vld1q_f32(px.add(base + 20)), q11);
            acc2 = vfmaq_f32(acc2, vld1q_f32(px.add(base + 24)), q12);
            acc3 = vfmaq_f32(acc3, vld1q_f32(px.add(base + 28)), q13);
        }
        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(acc)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn row_dot(x: &[f32], q_row: &[i8], scales: &[f32]) -> f32 {
    let n = x.len();
    let gpr = scales.len();
    debug_assert_eq!(n, gpr * G);
    let mut total = 0f64;
    for g in 0..gpr {
        let s = scales[g];
        if s == 0.0 {
            continue;
        }
        let mut acc = 0f32;
        for k in 0..G {
            let i = g * G + k;
            acc += x[i] * q_row[i] as f32;
        }
        total += (acc * s) as f64;
    }
    total as f32
}

// ============================================================
// Gemma4Q8G — per-group Q8 Gemma 4 with mirroring cached forward.
// ============================================================

use std::collections::HashMap;
use std::path::Path;

use crate::gemma4::{
    build_inv_freq, gelu_tanh, rmsnorm, rope_one, Gemma4, Gemma4Config,
};
use crate::ParseError;

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

pub struct Gemma4Q8G {
    pub cfg: Gemma4Config,
    qw: HashMap<String, Q8GroupWeight>,
    fw: HashMap<String, Vec<f32>>,
}

impl Gemma4Q8G {
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, ParseError> {
        let g = Gemma4::load(dir)?;
        Ok(Self::from_gemma4(&g))
    }
    pub fn from_gemma4(g: &Gemma4) -> Self {
        let cfg = g.cfg.clone();
        let mut qw = HashMap::new();
        let mut fw = HashMap::new();
        for (name, w) in &g.w {
            if let Some((out_d, in_d)) = linear_shape(name, &cfg) {
                qw.insert(name.clone(), Q8GroupWeight::quantize(w, in_d, out_d));
            } else {
                fw.insert(name.clone(), w.clone());
            }
        }
        Self { cfg, qw, fw }
    }
    pub(crate) fn qg(&self, n: &str) -> &Q8GroupWeight {
        self.qw.get(n).unwrap_or_else(|| panic!("missing q-weight {n}"))
    }
    pub(crate) fn fg(&self, n: &str) -> &[f32] {
        self.fw.get(n).unwrap_or_else(|| panic!("missing f-weight {n}"))
    }
    pub(crate) fn dequant_row(&self, w: &Q8GroupWeight, id: usize) -> Vec<f32> {
        let gpr = w.in_d / G;
        let q_row = &w.q[id * w.in_d..(id + 1) * w.in_d];
        let scales = &w.scale[id * gpr..(id + 1) * gpr];
        let mut out = vec![0f32; w.in_d];
        for g in 0..gpr {
            let s = scales[g];
            if s == 0.0 {
                continue;
            }
            for k in 0..G {
                out[g * G + k] = q_row[g * G + k] as f32 * s;
            }
        }
        out
    }

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

            let mut h = self.dequant_row(te_q, id);
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
                matvec_q8g(&h, pmp_q.unwrap(), &mut ctx);
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
                matvec_q8g(&xn, self.qg(&format!("blk.{l}.attn_q.weight")), &mut qh);
                rmsnorm(&mut qh, c.n_heads, hd, Some(self.fg(&format!("blk.{l}.attn_q_norm.weight"))), c.rms_eps);
                for hh in 0..c.n_heads {
                    rope_one(&mut qh[hh * hd..hh * hd + hd], hd, &inv[l], pos);
                }

                let src = if l < first_shared {
                    let mut k = vec![0f32; hd];
                    matvec_q8g(&xn, self.qg(&format!("blk.{l}.attn_k.weight")), &mut k);
                    rmsnorm(&mut k, 1, hd, Some(self.fg(&format!("blk.{l}.attn_k_norm.weight"))), c.rms_eps);
                    rope_one(&mut k, hd, &inv[l], pos);
                    let mut v = vec![0f32; hd];
                    matvec_q8g(&xn, self.qg(&format!("blk.{l}.attn_v.weight")), &mut v);
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
                matvec_q8g(&attn, self.qg(&format!("blk.{l}.attn_output.weight")), &mut ao);
                rmsnorm(&mut ao, 1, d, Some(self.fg(&format!("blk.{l}.post_attention_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res[k] + ao[k];
                }

                let res2 = h.clone();
                let mut xn2 = h.clone();
                rmsnorm(&mut xn2, 1, d, Some(self.fg(&format!("blk.{l}.ffn_norm.weight"))), c.rms_eps);
                let mut gate = vec![0f32; inter];
                matvec_q8g(&xn2, self.qg(&format!("blk.{l}.ffn_gate.weight")), &mut gate);
                let mut up = vec![0f32; inter];
                matvec_q8g(&xn2, self.qg(&format!("blk.{l}.ffn_up.weight")), &mut up);
                let act: Vec<f32> =
                    (0..inter).map(|k| gelu_tanh(gate[k]) * up[k]).collect();
                let mut down = vec![0f32; d];
                matvec_q8g(&act, self.qg(&format!("blk.{l}.ffn_down.weight")), &mut down);
                rmsnorm(&mut down, 1, d, Some(self.fg(&format!("blk.{l}.post_ffw_norm.weight"))), c.rms_eps);
                for k in 0..d {
                    h[k] = res2[k] + down[k];
                }

                if pd > 0 {
                    let res3 = h.clone();
                    let mut gp = vec![0f32; pd];
                    matvec_q8g(&h, self.qg(&format!("blk.{l}.per_layer_gate.weight")), &mut gp);
                    for (i, v) in gp.iter_mut().enumerate() {
                        *v = gelu_tanh(*v) * ple[l * pd + i];
                    }
                    let mut pp = vec![0f32; d];
                    matvec_q8g(&gp, self.qg(&format!("blk.{l}.per_layer_proj.weight")), &mut pp);
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
                matvec_q8g(&h, te_q, &mut logits);
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
}
