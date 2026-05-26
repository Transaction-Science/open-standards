//! LFM2.5-VL SigLIP vision tower + LFM2 multimodal projector.
//!
//! Pure-Rust f32 forward (the mmproj weights are tiny — 12 blocks of
//! 768 — so a direct implementation is correct-first and avoids adding
//! Conv2d / bias-matmul / LayerNorm-bias graph ops). Mirrors
//! `clip_graph_siglip` + the `PROJECTOR_TYPE_LFM2` head in
//! `ggml-org/llama.cpp@master:tools/mtmd/clip.cpp`.
//!
//! Flow:
//!   image[256×256×3] (SigLIP-normalised, −1..1)
//!     → Conv2d patch-embed (16×16 s16) + bias            → [256, 768]
//!     → + learned position-embed                          → [256, 768]
//!     → 12 × { LN+b; bidir MHA (12h×64, q/k/v/o +b); +res;
//!              LN+b; GELU FFN (up/down +b); +res }
//!     → post-LN+b                                         → [256, 768]
//!     → 2×2 pixel-unshuffle (scale_factor=2)              → [64, 3072]
//!     → projector  mm.1 (3072→2048) → GELU → mm.2 (→1024) → [64, 1024]
//!
//! `[64, 1024]` image tokens splice straight into the LFM2 text
//! embedding stream (1024 = lfm2 embedding_length).

use std::collections::HashMap;
use std::path::Path;

use jouleclaw_loader_gguf::{read_gguf_file, tensor_from_gguf, GgufModel, ParseError};

use crate::preprocess::{
    bilinear_resize, decode_image_bytes, normalize, patchify, ImageError, RgbImage,
};

#[derive(Debug)]
pub enum VisionError {
    Parse(ParseError),
    MissingTensor(String),
    Image(ImageError),
    Shape(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "mmproj parse: {:?}", e),
            Self::MissingTensor(n) => write!(f, "missing mmproj tensor: {}", n),
            Self::Image(e) => write!(f, "image: {}", e),
            Self::Shape(s) => write!(f, "shape: {}", s),
        }
    }
}
impl std::error::Error for VisionError {}

/// Exact (erf) GELU — SigLIP/LFM2 use the precise form, not tanh.
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Abramowitz–Stegun 7.1.26 erf approximation (max abs err ~1.5e-7).
#[inline]
fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736)
            * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

/// Loaded mmproj weights as dense f32, keyed by tensor name.
pub struct VisionTower {
    w: HashMap<String, (Vec<f32>, Vec<usize>)>, // name -> (data, logical_shape)
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub scale_factor: usize,
    pub ln_eps: f32,
}

impl VisionTower {
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self, VisionError> {
        let model = read_gguf_file(path.as_ref()).map_err(VisionError::Parse)?;
        let mu = |k: &str| model.metadata_u64(k);
        let n_layers = mu("clip.vision.block_count").unwrap_or(12) as usize;
        let n_embd = mu("clip.vision.embedding_length").unwrap_or(768) as usize;
        let n_head = mu("clip.vision.attention.head_count").unwrap_or(12) as usize;
        let patch_size = mu("clip.vision.patch_size").unwrap_or(16) as usize;
        let image_size = mu("clip.vision.image_size").unwrap_or(256) as usize;
        let scale_factor = mu("clip.vision.projector.scale_factor").unwrap_or(2) as usize;
        let ln_eps = model
            .metadata_f32("clip.vision.attention.layer_norm_epsilon")
            .unwrap_or(1e-6);

        let mut w = HashMap::new();
        Self::load_all(&model, &mut w)?;
        Ok(Self {
            w, n_layers, n_embd, n_head, patch_size, image_size,
            scale_factor, ln_eps,
        })
    }

    fn load_all(
        model: &GgufModel,
        w: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<(), VisionError> {
        for info in &model.tensors {
            let t = tensor_from_gguf(model, info).map_err(VisionError::Parse)?;
            // logical shape = GGUF ne-order reversed (loader convention)
            let shape: Vec<usize> = t.meta.shape.clone();
            w.insert(info.name.clone(), (t.as_f32_vec(), shape));
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Result<&(Vec<f32>, Vec<usize>), VisionError> {
        self.w.get(name).ok_or_else(|| VisionError::MissingTensor(name.into()))
    }

    /// Decode + preprocess raw image bytes the way LFM2.5-VL expects:
    /// resize to `image_size²`, SigLIP normalise to [−1, 1].
    pub fn preprocess(&self, bytes: &[u8]) -> Result<RgbImage, VisionError> {
        let img = decode_image_bytes(bytes).map_err(VisionError::Image)?;
        let mut r = bilinear_resize(&img, self.image_size, self.image_size)
            .map_err(VisionError::Image)?;
        // SigLIP: x → (x − 0.5) / 0.5  ⇒  [0,1] → [−1,1]
        normalize(&mut r, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        Ok(r)
    }

    /// Full vision forward → `[n_img_tokens, projector_out]` image
    /// tokens, ready to splice into the LFM2 text embedding stream.
    pub fn encode(&self, img: &RgbImage) -> Result<Vec<Vec<f32>>, VisionError> {
        let d = self.n_embd;
        let ps = self.patch_size;
        let grid = self.image_size / ps; // 16
        let n_patches = grid * grid; // 256

        // ---- Conv2d patch embedding ----
        // patchify gives [n_patches][ps*ps*3] in (iy, ix, c) order.
        let patches = patchify(img, ps).map_err(VisionError::Image)?;
        let (pe_w, _pe_shape) = self.get("v.patch_embd.weight")?;
        let (pe_b, _) = self.get("v.patch_embd.bias")?;
        // conv weight logical [out=768, in_ch=3, kh=16, kw=16]
        // flat idx = ((o*3 + ic)*ps + ky)*ps + kx
        let mut x = vec![0f32; n_patches * d]; // [n_patches, d] row-major
        for (p, patch) in patches.iter().enumerate() {
            for o in 0..d {
                let mut acc = pe_b[o];
                for ic in 0..3 {
                    for ky in 0..ps {
                        for kx in 0..ps {
                            let wv = pe_w[((o * 3 + ic) * ps + ky) * ps + kx];
                            // patch order (iy, ix, c)
                            let pv = patch[(ky * ps + kx) * 3 + ic];
                            acc += wv * pv;
                        }
                    }
                }
                x[p * d + o] = acc;
            }
        }

        // ---- + position embedding [n_patches, d] ----
        let (pos, _) = self.get("v.position_embd.weight")?;
        for i in 0..n_patches * d {
            x[i] += pos[i];
        }

        // ---- 12 transformer blocks ----
        let n_head = self.n_head;
        let d_head = d / n_head;
        let scale = 1.0 / (d_head as f32).sqrt();
        for l in 0..self.n_layers {
            let pfx = format!("v.blk.{}", l);
            // LN1
            let h = self.layernorm(&x, n_patches, d,
                &self.get(&format!("{pfx}.ln1.weight"))?.0,
                &self.get(&format!("{pfx}.ln1.bias"))?.0)?;
            // q,k,v projections (+bias). weights logical [d,d].
            let q = self.linear(&h, n_patches, d, d,
                &self.get(&format!("{pfx}.attn_q.weight"))?.0,
                Some(&self.get(&format!("{pfx}.attn_q.bias"))?.0));
            let k = self.linear(&h, n_patches, d, d,
                &self.get(&format!("{pfx}.attn_k.weight"))?.0,
                Some(&self.get(&format!("{pfx}.attn_k.bias"))?.0));
            let v = self.linear(&h, n_patches, d, d,
                &self.get(&format!("{pfx}.attn_v.weight"))?.0,
                Some(&self.get(&format!("{pfx}.attn_v.bias"))?.0));
            // bidirectional MHA, per head
            let mut ctx = vec![0f32; n_patches * d];
            for hd in 0..n_head {
                let off = hd * d_head;
                for i in 0..n_patches {
                    // scores over all positions (no causal mask)
                    let mut s = vec![0f32; n_patches];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..n_patches {
                        let mut dot = 0f32;
                        for c in 0..d_head {
                            dot += q[i * d + off + c] * k[j * d + off + c];
                        }
                        dot *= scale;
                        s[j] = dot;
                        if dot > mx { mx = dot; }
                    }
                    let mut sum = 0f32;
                    for sj in s.iter_mut() { *sj = (*sj - mx).exp(); sum += *sj; }
                    let inv = 1.0 / sum;
                    for c in 0..d_head {
                        let mut acc = 0f32;
                        for j in 0..n_patches {
                            acc += s[j] * inv * v[j * d + off + c];
                        }
                        ctx[i * d + off + c] = acc;
                    }
                }
            }
            // out proj (+bias) then residual
            let ao = self.linear(&ctx, n_patches, d, d,
                &self.get(&format!("{pfx}.attn_out.weight"))?.0,
                Some(&self.get(&format!("{pfx}.attn_out.bias"))?.0));
            for i in 0..n_patches * d { x[i] += ao[i]; }
            // LN2 → FFN (gelu) → residual
            let h2 = self.layernorm(&x, n_patches, d,
                &self.get(&format!("{pfx}.ln2.weight"))?.0,
                &self.get(&format!("{pfx}.ln2.bias"))?.0)?;
            let (up_w, up_s) = self.get(&format!("{pfx}.ffn_up.weight"))?;
            let ff_hidden = up_s[0]; // logical [ff, d]
            let mut up = self.linear(&h2, n_patches, d, ff_hidden, up_w,
                Some(&self.get(&format!("{pfx}.ffn_up.bias"))?.0));
            for u in up.iter_mut() { *u = gelu(*u); }
            let down = self.linear(&up, n_patches, ff_hidden, d,
                &self.get(&format!("{pfx}.ffn_down.weight"))?.0,
                Some(&self.get(&format!("{pfx}.ffn_down.bias"))?.0));
            for i in 0..n_patches * d { x[i] += down[i]; }
        }

        // ---- post-LN ----
        let x = self.layernorm(&x, n_patches, d,
            &self.get("v.post_ln.weight")?.0,
            &self.get("v.post_ln.bias")?.0)?;

        // ---- 2×2 pixel-unshuffle (scale_factor) ----
        // merged token (gx,gy) for gx,gy in 0..grid/sf, with the
        // 768-chunk order derived from ggml's permute sequence:
        //   [(2gx,2gy), (2gx+1,2gy), (2gx,2gy+1), (2gx+1,2gy+1)]
        // patch index = py*grid + px (patchify py-outer/px-inner).
        let sf = self.scale_factor;
        let mg = grid / sf; // 8
        let merged_dim = d * sf * sf; // 3072
        let mut merged = vec![0f32; mg * mg * merged_dim];
        for gy in 0..mg {
            for gx in 0..mg {
                let tok = gy * mg + gx;
                let mut off = 0usize;
                for (sx, sy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let px = 2 * gx + sx;
                    let py = 2 * gy + sy;
                    let pidx = py * grid + px;
                    merged[tok * merged_dim + off..tok * merged_dim + off + d]
                        .copy_from_slice(&x[pidx * d..pidx * d + d]);
                    off += d;
                }
            }
        }
        let n_tok = mg * mg;

        // ---- projector: mm.1 (→2048) → GELU → mm.2 (→1024) ----
        let (m1w, m1s) = self.get("mm.1.weight")?;
        let m1_out = m1s[0];
        let mut p1 = self.linear(&merged, n_tok, merged_dim, m1_out, m1w,
            Some(&self.get("mm.1.bias")?.0));
        for u in p1.iter_mut() { *u = gelu(*u); }
        let (m2w, m2s) = self.get("mm.2.weight")?;
        let m2_out = m2s[0];
        let p2 = self.linear(&p1, n_tok, m1_out, m2_out, m2w,
            Some(&self.get("mm.2.bias")?.0));

        Ok((0..n_tok)
            .map(|t| p2[t * m2_out..(t + 1) * m2_out].to_vec())
            .collect())
    }

    /// `Y[n, out] = X[n, in] @ W[out, in]^T (+ bias[out])`.
    fn linear(
        &self, x: &[f32], n: usize, in_d: usize, out_d: usize,
        w: &[f32], bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut y = vec![0f32; n * out_d];
        for i in 0..n {
            for o in 0..out_d {
                let mut acc = bias.map(|b| b[o]).unwrap_or(0.0);
                let wr = &w[o * in_d..o * in_d + in_d];
                let xr = &x[i * in_d..i * in_d + in_d];
                for c in 0..in_d {
                    acc += xr[c] * wr[c];
                }
                y[i * out_d + o] = acc;
            }
        }
        y
    }

    /// LayerNorm with weight + bias over the last `d` axis.
    fn layernorm(
        &self, x: &[f32], n: usize, d: usize, w: &[f32], b: &[f32],
    ) -> Result<Vec<f32>, VisionError> {
        if w.len() != d || b.len() != d {
            return Err(VisionError::Shape(format!(
                "layernorm w/b len {}/{} != d {}", w.len(), b.len(), d)));
        }
        let mut y = vec![0f32; n * d];
        for i in 0..n {
            let row = &x[i * d..i * d + d];
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>()
                / d as f32;
            let inv = 1.0 / (var + self.ln_eps).sqrt();
            for c in 0..d {
                y[i * d + c] = (row[c] - mean) * inv * w[c] + b[c];
            }
        }
        Ok(y)
    }
}
