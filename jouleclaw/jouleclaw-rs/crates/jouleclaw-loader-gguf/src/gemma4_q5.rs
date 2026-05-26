//! **Per-row per-group asymmetric int5** (G=32, signed zero-point).
//! One extra bit of weight resolution past Q4 — 32 codes per group
//! vs 16. Direct attack on the empirical Gemma 4 E2B int4 floor.
//!
//! Storage: split low-4-bits + high-1-bit. Per group of 32:
//!   • 16 bytes of low nibbles (`q & 0x0F`, identical to Q4 packing)
//!   • 4 bytes of high bits, bit k = `(q >> 4) & 1` for element k
//!   • 1 f32 scale, 1 i8 zero-point
//! Total: 20 weight bytes + 4 (scale) + 1 (zp) = 25 B per 32 weights
//! ≈ 0.781 B/param — between Q4 (0.6) and Q8 (1.0).
//!
//! `q_u5 ∈ [0, 31]`, `(q − zp) · scale ≈ w`.

use std::collections::HashMap;
use std::path::Path;

use crate::gemma4::{
    build_inv_freq, gelu_tanh, linear as f32_linear, rmsnorm, rope_one, Gemma4,
    Gemma4Config,
};
use crate::ParseError;

const G: usize = 32;

pub struct Q5Weight {
    /// Low 4 bits per weight, 2 weights per byte. Length `out_d·in_d/2`.
    pub q_lo: Vec<u8>,
    /// Bit k of byte `g·4 + k/8` = high bit of weight at position
    /// `g·G + k` (within row). Length `out_d · (in_d / 8)`.
    pub q_hi: Vec<u8>,
    /// Per-row per-group scale.
    pub scale: Vec<f32>,
    /// Per-row per-group signed zero-point.
    pub zero_point: Vec<i8>,
    pub in_d: usize,
    pub out_d: usize,
}

impl Q5Weight {
    pub fn quantize(w: &[f32], in_d: usize, out_d: usize) -> Self {
        debug_assert_eq!(w.len(), in_d * out_d);
        debug_assert!(in_d % G == 0);
        let gpr = in_d / G;
        let mut q_lo = vec![0u8; in_d * out_d / 2];
        let mut q_hi = vec![0u8; in_d * out_d / 8];
        let mut scale = vec![0f32; out_d * gpr];
        let mut zp = vec![0i8; out_d * gpr];

        for o in 0..out_d {
            let row = &w[o * in_d..(o + 1) * in_d];
            for g in 0..gpr {
                let blk = &row[g * G..(g + 1) * G];
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for &v in blk {
                    if v < mn { mn = v; }
                    if v > mx { mx = v; }
                }
                let s = if mx > mn { (mx - mn) / 31.0 } else { 0.0 };
                let z = if s > 0.0 {
                    (-mn / s).round().clamp(-128.0, 127.0) as i8
                } else { 0i8 };
                scale[o * gpr + g] = s;
                zp[o * gpr + g] = z;
                if s > 0.0 {
                    let inv = 1.0 / s;
                    let zf = z as f32;
                    for k in 0..G {
                        let qv = (blk[k] * inv + zf).round().clamp(0.0, 31.0) as u8;
                        // low 4 bits
                        let global_k = o * in_d + g * G + k;
                        let byte_idx = global_k / 2;
                        let lo = qv & 0x0F;
                        if global_k % 2 == 0 {
                            q_lo[byte_idx] = (q_lo[byte_idx] & 0xF0) | lo;
                        } else {
                            q_lo[byte_idx] = (q_lo[byte_idx] & 0x0F) | (lo << 4);
                        }
                        // high bit at position k within group (0..32)
                        let hi = (qv >> 4) & 0x01;
                        let hi_byte = (o * in_d / 8) + g * 4 + k / 8;
                        let hi_bit = k % 8;
                        q_hi[hi_byte] |= hi << hi_bit;
                    }
                }
            }
        }
        Self { q_lo, q_hi, scale, zero_point: zp, in_d, out_d }
    }
}

pub fn matvec_q5(x: &[f32], w: &Q5Weight, y: &mut [f32]) {
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
                &w.q_lo[o * in_d / 2..(o + 1) * in_d / 2],
                &w.q_hi[o * in_d / 8..(o + 1) * in_d / 8],
                &w.scale[o * gpr..(o + 1) * gpr],
                &w.zero_point[o * gpr..(o + 1) * gpr],
            );
        }
        return;
    }
    let chunk = out_d.div_ceil(nthreads);
    std::thread::scope(|sc| {
        for (i, ys) in y.chunks_mut(chunk).enumerate() {
            let row_start = i * chunk;
            let n = ys.len();
            let lo_chunk = &w.q_lo[row_start * in_d / 2..(row_start + n) * in_d / 2];
            let hi_chunk = &w.q_hi[row_start * in_d / 8..(row_start + n) * in_d / 8];
            let s_chunk = &w.scale[row_start * gpr..(row_start + n) * gpr];
            let z_chunk = &w.zero_point[row_start * gpr..(row_start + n) * gpr];
            sc.spawn(move || {
                for o in 0..n {
                    ys[o] = row_dot(
                        x,
                        &lo_chunk[o * in_d / 2..(o + 1) * in_d / 2],
                        &hi_chunk[o * in_d / 8..(o + 1) * in_d / 8],
                        &s_chunk[o * gpr..(o + 1) * gpr],
                        &z_chunk[o * gpr..(o + 1) * gpr],
                    );
                }
            });
        }
    });
}

#[inline]
fn row_dot(
    x: &[f32],
    q_lo: &[u8],
    q_hi: &[u8],
    scales: &[f32],
    zps: &[i8],
) -> f32 {
    let mut total = 0f64;
    let gpr = scales.len();
    for g in 0..gpr {
        let s = scales[g];
        if s == 0.0 { continue; }
        let z = zps[g] as f32;
        let mut acc = 0f32;
        for k in (0..G).step_by(2) {
            let lo_byte = q_lo[g * G / 2 + k / 2];
            let lo_a = (lo_byte & 0x0F) as u8;
            let lo_b = (lo_byte >> 4) as u8;
            let hi_byte_a = q_hi[g * 4 + k / 8];
            let hi_byte_b = q_hi[g * 4 + (k + 1) / 8];
            let hi_a = (hi_byte_a >> (k % 8)) & 1;
            let hi_b = (hi_byte_b >> ((k + 1) % 8)) & 1;
            let qv_a = (lo_a | (hi_a << 4)) as f32;
            let qv_b = (lo_b | (hi_b << 4)) as f32;
            let i = g * G + k;
            acc += x[i] * (qv_a - z) + x[i + 1] * (qv_b - z);
        }
        total += (acc * s) as f64;
    }
    total as f32
}

// ============================================================
// Gemma4Q5 — full int5 Gemma 4 with mirroring cached forward.
// ============================================================

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

pub struct Gemma4Q5 {
    pub cfg: Gemma4Config,
    qw: HashMap<String, Q5Weight>,
    fw: HashMap<String, Vec<f32>>,
}

impl Gemma4Q5 {
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, ParseError> {
        let g = Gemma4::load(dir)?;
        Ok(Self::from_gemma4(&g))
    }
    pub fn from_gemma4(g: &Gemma4) -> Self {
        let cfg = g.cfg.clone();
        let mut qw = HashMap::new();
        let mut fw = HashMap::new();
        for (name, w) in &g.w {
            if name == "token_embd.weight" {
                fw.insert(name.clone(), w.clone());
                continue;
            }
            if let Some((out_d, in_d)) = linear_shape(name, &cfg) {
                qw.insert(name.clone(), Q5Weight::quantize(w, in_d, out_d));
            } else {
                fw.insert(name.clone(), w.clone());
            }
        }
        Self { cfg, qw, fw }
    }
    fn qg(&self, n: &str) -> &Q5Weight {
        self.qw.get(n).unwrap_or_else(|| panic!("missing q-weight {n}"))
    }
    fn fg(&self, n: &str) -> &[f32] {
        self.fw.get(n).unwrap_or_else(|| panic!("missing f-weight {n}"))
    }
    fn dequant_row(&self, w: &Q5Weight, id: usize) -> Vec<f32> {
        let gpr = w.in_d / G;
        let q_lo = &w.q_lo[id * w.in_d / 2..(id + 1) * w.in_d / 2];
        let q_hi = &w.q_hi[id * w.in_d / 8..(id + 1) * w.in_d / 8];
        let scales = &w.scale[id * gpr..(id + 1) * gpr];
        let zps = &w.zero_point[id * gpr..(id + 1) * gpr];
        let mut out = vec![0f32; w.in_d];
        for g in 0..gpr {
            let s = scales[g];
            if s == 0.0 { continue; }
            let z = zps[g] as f32;
            for k in (0..G).step_by(2) {
                let lo_byte = q_lo[g * G / 2 + k / 2];
                let lo_a = lo_byte & 0x0F;
                let lo_b = lo_byte >> 4;
                let hi_a = (q_hi[g * 4 + k / 8] >> (k % 8)) & 1;
                let hi_b = (q_hi[g * 4 + (k + 1) / 8] >> ((k + 1) % 8)) & 1;
                let qv_a = (lo_a | (hi_a << 4)) as f32;
                let qv_b = (lo_b | (hi_b << 4)) as f32;
                out[g * G + k] = (qv_a - z) * s;
                out[g * G + k + 1] = (qv_b - z) * s;
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
            .map(|l| build_inv_freq(c, l, c.layer_head_dim(l))).collect();
        let te_f32 = self.fg("token_embd.weight");
        let tpl_q = if pd > 0 { Some(self.qg("per_layer_token_embd.weight")) } else { None };
        let pmp_q = if pd > 0 { Some(self.qg("per_layer_model_proj.weight")) } else { None };

        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let total = prompt.len() + max_new;
        for pos in 0..total {
            if out.len() == max_new { break; }
            let id = if pos < prompt.len() {
                prompt[pos]
            } else { out[pos - prompt.len()] } as usize;

            let mut h: Vec<f32> = te_f32[id * d..(id + 1) * d]
                .iter().map(|v| v * emb_scale).collect();

            let ple: Vec<f32> = if pd > 0 {
                let tps = (pd as f32).sqrt();
                let mut tok_id = self.dequant_row(tpl_q.unwrap(), id);
                for v in &mut tok_id { *v *= tps; }
                let mut ctx = vec![0f32; nl * pd];
                matvec_q5(&h, pmp_q.unwrap(), &mut ctx);
                let psc = 1.0 / (d as f32).sqrt();
                for v in &mut ctx { *v *= psc; }
                rmsnorm(&mut ctx, nl, pd, Some(self.fg("per_layer_proj_norm.weight")), c.rms_eps);
                let cs = 1.0 / 2f32.sqrt();
                (0..nl * pd).map(|k| (ctx[k] + tok_id[k]) * cs).collect()
            } else { Vec::new() };

            for l in 0..nl {
                let is_full = c.is_full(l);
                let hd = c.layer_head_dim(l);
                let ltype = &c.layer_types[l];
                let inter = if c.kv_shared(l) { c.ffn * 2 } else { c.ffn };

                let res = h.clone();
                let mut xn = h.clone();
                rmsnorm(&mut xn, 1, d, Some(self.fg(&format!("blk.{l}.attn_norm.weight"))), c.rms_eps);

                let mut qh = vec![0f32; c.n_heads * hd];
                matvec_q5(&xn, self.qg(&format!("blk.{l}.attn_q.weight")), &mut qh);
                rmsnorm(&mut qh, c.n_heads, hd, Some(self.fg(&format!("blk.{l}.attn_q_norm.weight"))), c.rms_eps);
                for hh in 0..c.n_heads {
                    rope_one(&mut qh[hh * hd..hh * hd + hd], hd, &inv[l], pos);
                }

                let src = if l < first_shared {
                    let mut k = vec![0f32; hd];
                    matvec_q5(&xn, self.qg(&format!("blk.{l}.attn_k.weight")), &mut k);
                    rmsnorm(&mut k, 1, hd, Some(self.fg(&format!("blk.{l}.attn_k_norm.weight"))), c.rms_eps);
                    rope_one(&mut k, hd, &inv[l], pos);
                    let mut v = vec![0f32; hd];
                    matvec_q5(&xn, self.qg(&format!("blk.{l}.attn_v.weight")), &mut v);
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
                        for x in 0..hd { a += qv[x] * kr[x]; }
                        sc[j] = a;
                        if a > mx { mx = a; }
                    }
                    let mut den = 0f32;
                    for j in j0..clen { sc[j] = (sc[j] - mx).exp(); den += sc[j]; }
                    let o = &mut attn[hh * hd..hh * hd + hd];
                    for j in j0..clen {
                        let w = sc[j] / den;
                        let vr = &vv[j * hd..j * hd + hd];
                        for x in 0..hd { o[x] += w * vr[x]; }
                    }
                }
                let mut ao = vec![0f32; d];
                matvec_q5(&attn, self.qg(&format!("blk.{l}.attn_output.weight")), &mut ao);
                rmsnorm(&mut ao, 1, d, Some(self.fg(&format!("blk.{l}.post_attention_norm.weight"))), c.rms_eps);
                for k in 0..d { h[k] = res[k] + ao[k]; }

                let res2 = h.clone();
                let mut xn2 = h.clone();
                rmsnorm(&mut xn2, 1, d, Some(self.fg(&format!("blk.{l}.ffn_norm.weight"))), c.rms_eps);
                let mut gate = vec![0f32; inter];
                matvec_q5(&xn2, self.qg(&format!("blk.{l}.ffn_gate.weight")), &mut gate);
                let mut up = vec![0f32; inter];
                matvec_q5(&xn2, self.qg(&format!("blk.{l}.ffn_up.weight")), &mut up);
                let act: Vec<f32> = (0..inter).map(|k| gelu_tanh(gate[k]) * up[k]).collect();
                let mut down = vec![0f32; d];
                matvec_q5(&act, self.qg(&format!("blk.{l}.ffn_down.weight")), &mut down);
                rmsnorm(&mut down, 1, d, Some(self.fg(&format!("blk.{l}.post_ffw_norm.weight"))), c.rms_eps);
                for k in 0..d { h[k] = res2[k] + down[k]; }

                if pd > 0 {
                    let res3 = h.clone();
                    let mut gp = vec![0f32; pd];
                    matvec_q5(&h, self.qg(&format!("blk.{l}.per_layer_gate.weight")), &mut gp);
                    for (i, v) in gp.iter_mut().enumerate() {
                        *v = gelu_tanh(*v) * ple[l * pd + i];
                    }
                    let mut pp = vec![0f32; d];
                    matvec_q5(&gp, self.qg(&format!("blk.{l}.per_layer_proj.weight")), &mut pp);
                    rmsnorm(&mut pp, 1, d, Some(self.fg(&format!("blk.{l}.post_per_layer_norm.weight"))), c.rms_eps);
                    for k in 0..d { h[k] = res3[k] + pp[k]; }
                }

                let ls = self.fg(&format!("blk.{l}.layer_scalar"))[0];
                for v in &mut h { *v *= ls; }
            }

            if pos + 1 >= prompt.len() {
                rmsnorm(&mut h, 1, d, Some(self.fg("output_norm.weight")), c.rms_eps);
                let logits = f32_linear(&h, 1, d, te_f32, c.vocab);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q5_roundtrip_small_error() {
        let inn = 64usize;
        let out = 4usize;
        let w: Vec<f32> = (0..out * inn)
            .map(|i| ((i as f32) * 0.131).sin() * 0.5).collect();
        let q5 = Q5Weight::quantize(&w, inn, out);
        let x: Vec<f32> = (0..inn).map(|i| ((i as f32) * 0.17).cos()).collect();
        let mut y_q5 = vec![0f32; out];
        matvec_q5(&x, &q5, &mut y_q5);
        let mut y_ref = vec![0f32; out];
        for o in 0..out {
            for i in 0..inn {
                y_ref[o] += x[i] * w[o * inn + i];
            }
        }
        for o in 0..out {
            let err = (y_ref[o] - y_q5[o]).abs();
            assert!(err < 0.15, "row {o}: err {err}");
        }
    }
}
