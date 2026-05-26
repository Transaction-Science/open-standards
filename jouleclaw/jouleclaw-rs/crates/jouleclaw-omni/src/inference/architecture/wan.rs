//! Wan2.1 Video Diffusion Transformer architecture.
//!
//! Architecture: 30 transformer blocks with per-block modulation.
//! Each block has self-attention + QK-norm, cross-attention + QK-norm,
//! and a SiLU-gated FFN. Timestep conditioning via learned modulation
//! parameters per block (6 vectors: shift/scale/gate for SA and FFN).
//!
//! Key differences from PixArt-Sigma:
//!   - 3D patch embedding (T=2, H=2, W=2) for video
//!   - QK-norm (RMSNorm on Q and K separately) in attention
//!   - Per-block modulation (not global AdaLN-Single)
//!   - SiLU FFN (not GEGLU)
//!   - Text from UmT5-XXL (4096-dim) projected to 1536-dim

use crate::core::Result;
use crate::hal::DeviceId;
use crate::inference::model::Model;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::MetalCompute;
#[cfg(feature = "metal")]
use crate::hal::metal::LazyTensor;
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};

/// Wan2.1 video DiT configuration.
#[derive(Debug, Clone)]
pub struct WanConfig {
    /// Hidden dimension (1536 for 1.3B).
    pub dim: usize,
    /// Number of attention heads (12).
    pub num_heads: usize,
    /// FFN inner dimension (8960).
    pub ffn_dim: usize,
    /// Number of transformer blocks (30).
    pub num_layers: usize,
    /// Frequency embedding dimension (256).
    pub freq_dim: usize,
    /// Input/output latent channels (16).
    pub in_dim: usize,
    /// Text encoder output dimension (4096 for UmT5-XXL).
    pub text_dim: usize,
    /// Maximum text sequence length (512).
    pub text_len: usize,
    /// Layer norm epsilon.
    pub eps: f32,
}

impl Default for WanConfig {
    fn default() -> Self {
        Self {
            dim: 1536,
            num_heads: 12,
            ffn_dim: 8960,
            num_layers: 30,
            freq_dim: 256,
            in_dim: 16,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
        }
    }
}

impl WanConfig {
    /// Parse from config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| crate::core::Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| crate::core::Error::internal(format!("failed to parse config: {}", e)))?;

        let mut config = Self::default();
        if let Some(v) = json.get("dim").and_then(|v| v.as_u64()) { config.dim = v as usize; }
        if let Some(v) = json.get("num_heads").and_then(|v| v.as_u64()) { config.num_heads = v as usize; }
        if let Some(v) = json.get("ffn_dim").and_then(|v| v.as_u64()) { config.ffn_dim = v as usize; }
        if let Some(v) = json.get("num_layers").and_then(|v| v.as_u64()) { config.num_layers = v as usize; }
        if let Some(v) = json.get("freq_dim").and_then(|v| v.as_u64()) { config.freq_dim = v as usize; }
        if let Some(v) = json.get("in_dim").and_then(|v| v.as_u64()) { config.in_dim = v as usize; }
        if let Some(v) = json.get("eps").and_then(|v| v.as_f64()) { config.eps = v as f32; }
        if let Some(v) = json.get("text_len").and_then(|v| v.as_u64()) { config.text_len = v as usize; }
        Ok(config)
    }
}

/// Compiled kernel pipelines for Wan DiT.
#[cfg(feature = "metal")]
struct WanKernels {
    common: gpu_ops::CommonKernels,
    rms_norm: Arc<crate::hal::metal::ComputePipeline>,
    silu: Arc<crate::hal::metal::ComputePipeline>,
    gelu: Arc<crate::hal::metal::ComputePipeline>,
    mul: Arc<crate::hal::metal::ComputePipeline>,
    adaln_modulate: Arc<crate::hal::metal::ComputePipeline>,
    adaln_gate: Arc<crate::hal::metal::ComputePipeline>,
    // Mixed-precision attention: Q@K^T -> F32 scores, scale+softmax in F32.
    batched_linear_out_f32: Arc<crate::hal::metal::ComputePipeline>,
    softmax_scale_f32_to_f16: Arc<crate::hal::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
impl WanKernels {
    fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            common: gpu_ops::CommonKernels::new(compute)?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ADALN, "adaln_gate_f16")?,
            batched_linear_out_f32: compute.compile_pipeline(
                "batched_linear_f16_out_f32",
                sources::LINEAR,
                "batched_linear_f16_out_f32",
            )?,
            softmax_scale_f32_to_f16: compute.compile_pipeline(
                "row_softmax_scale_f32_to_f16",
                sources::LINEAR,
                "row_softmax_scale_f32_to_f16",
            )?,
        })
    }
}

/// Diagnostic helper: when `WAN_DUMP_DIR` env var is set, write the tensor's
/// f32 contents to `{WAN_DUMP_DIR}/{name}.bin` and shape to `{name}.shape`.
/// Same format as the Python reference dump script.
#[cfg(feature = "metal")]
fn dump_tensor_if_env(name: &str, t: &Tensor) {
    let dir = match std::env::var("WAN_DUMP_DIR") {
        Ok(d) => d,
        Err(_) => return,
    };
    let _ = std::fs::create_dir_all(&dir);
    let v: Vec<half::f16> = match t.to_vec() {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut bin = Vec::with_capacity(v.len() * 4);
    for x in &v {
        bin.extend_from_slice(&x.to_f32().to_le_bytes());
    }
    let _ = std::fs::write(format!("{}/{}.bin", dir, name), &bin);
    let shape_str = t.shape().dims().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    let _ = std::fs::write(format!("{}/{}.shape", dir, name), shape_str);
    let n = v.len() as f64;
    let sum_sq: f64 = v.iter().map(|x| { let f = x.to_f32() as f64; f * f }).sum();
    let rms = (sum_sq / n).sqrt();
    eprintln!("    [DUMP] {}: rms={:.4}", name, rms);
}

/// Wan2.1 Video DiT — full forward pass on Metal GPU.
#[cfg(feature = "metal")]
pub struct WanDiT {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: WanConfig,
    kernels: WanKernels,
    /// 3D RoPE frequency tables (cos, sin) per axis, precomputed once.
    /// Wan-spec: head_dim is split as (d_t, d_h, d_w) where
    ///   d_t = head_dim - 4 * (head_dim / 6)        // time axis  (44 reals for 1.3B)
    ///   d_h = 2 * (head_dim / 6)                   // height axis (42 reals)
    ///   d_w = 2 * (head_dim / 6)                   // width axis  (42 reals)
    /// Each axis stores `max_seq_len * (d_axis / 2)` (cos, sin) pairs.
    /// theta = 10000 (Wan default).
    rope_t: Vec<(f32, f32)>,  // shape [max_seq_len, d_t/2]
    rope_h: Vec<(f32, f32)>,  // shape [max_seq_len, d_h/2]
    rope_w: Vec<(f32, f32)>,  // shape [max_seq_len, d_w/2]
    rope_d_t_pairs: usize,    // d_t / 2
    rope_d_h_pairs: usize,    // d_h / 2
    rope_d_w_pairs: usize,    // d_w / 2
    rope_max_seq_len: usize,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for WanDiT {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl WanDiT {
    /// Create a new Wan DiT transformer.
    pub fn new(model: Arc<Model>, config: WanConfig, compute: &Arc<MetalCompute>) -> Result<Self> {
        let kernels = WanKernels::new(compute)?;

        // Precompute 3D RoPE frequency tables (theta=10000, max_seq_len=1024 per axis).
        let head_dim = config.dim / config.num_heads;
        let d_t = head_dim - 4 * (head_dim / 6);
        let d_h = 2 * (head_dim / 6);
        let d_w = 2 * (head_dim / 6);
        debug_assert_eq!(d_t + d_h + d_w, head_dim);
        let max_seq_len = 1024usize;
        let theta: f64 = 10000.0;
        let make_axis = |d_axis: usize| -> Vec<(f32, f32)> {
            let pairs = d_axis / 2;
            let mut out = Vec::with_capacity(max_seq_len * pairs);
            for s in 0..max_seq_len {
                for k in 0..pairs {
                    // freq = 1 / theta^(2k / d_axis); phase = s * freq
                    let freq = 1.0 / theta.powf((2 * k) as f64 / d_axis as f64);
                    let phase = (s as f64) * freq;
                    out.push((phase.cos() as f32, phase.sin() as f32));
                }
            }
            out
        };

        Ok(Self {
            model,
            compute: Arc::clone(compute),
            config,
            kernels,
            rope_t: make_axis(d_t),
            rope_h: make_axis(d_h),
            rope_w: make_axis(d_w),
            rope_d_t_pairs: d_t / 2,
            rope_d_h_pairs: d_h / 2,
            rope_d_w_pairs: d_w / 2,
            rope_max_seq_len: max_seq_len,
        })
    }

    /// CPU LayerNorm without learnable affine (Wan's norm1, norm2, head.norm).
    ///
    /// Computes `(x - mean) / sqrt(var + eps)` per row of `[seq_len, dim]`.
    /// Wan trains its blocks with `WanLayerNorm(dim, elementwise_affine=False)`
    /// for norm1 (pre-self-attn) and norm2 (pre-FFN). The Rust port previously
    /// applied AdaLN modulation directly to `h` without this normalization,
    /// which destroys the variance the modulation expects to operate on.
    fn cpu_layernorm_no_affine(&self, x: &Tensor, seq_len: usize, dim: usize) -> Result<Tensor> {
        let data: Vec<half::f16> = x.to_vec()?;
        debug_assert_eq!(data.len(), seq_len * dim);
        let eps = 1e-6f32;
        let mut out = vec![half::f16::ZERO; data.len()];
        for row in 0..seq_len {
            let off = row * dim;
            // Mean
            let mut sum = 0.0f32;
            for j in 0..dim { sum += data[off + j].to_f32(); }
            let mean = sum / (dim as f32);
            // Variance (population, biased — matches PyTorch LayerNorm)
            let mut var_sum = 0.0f32;
            for j in 0..dim {
                let d = data[off + j].to_f32() - mean;
                var_sum += d * d;
            }
            let inv_std = 1.0 / (var_sum / (dim as f32) + eps).sqrt();
            for j in 0..dim {
                let v = (data[off + j].to_f32() - mean) * inv_std;
                out[off + j] = half::f16::from_f32(v);
            }
        }
        Tensor::from_slice(&out, x.shape().clone(), DType::F16, x.device())
    }

    /// Apply Wan-spec 3D RoPE to a flat [seq_len, n_heads * head_dim] F16 tensor.
    ///
    /// `seq_len = grid_t * grid_h * grid_w`. The function pairs adjacent reals
    /// `(x[2k], x[2k+1])` as a complex number, then multiplies by per-axis
    /// rotation `(cos, sin)` chosen from `(rope_t, rope_h, rope_w)` based on
    /// the position's `(it, ih, iw)` index and the pair index `k`.
    ///
    /// Pair allocation per head:  [0 .. d_t/2) → time;  [d_t/2 .. d_t/2+d_h/2) → height;
    /// [d_t/2+d_h/2 .. head_dim/2) → width.
    fn apply_rope_3d_cpu(
        &self,
        qk: &Tensor,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let seq_len = grid_t * grid_h * grid_w;
        debug_assert_eq!(qk.shape().dim(0).unwrap_or(0), seq_len);
        debug_assert_eq!(qk.shape().dim(1).unwrap_or(0), n_heads * head_dim);
        let mut data: Vec<half::f16> = qk.to_vec()?;

        let pairs_t = self.rope_d_t_pairs;
        let pairs_h = self.rope_d_h_pairs;
        let pairs_w = self.rope_d_w_pairs;
        let pairs_total = pairs_t + pairs_h + pairs_w;
        debug_assert_eq!(pairs_total * 2, head_dim);

        // For each position (it, ih, iw), each head, each complex pair k,
        // rotate the (real, imag) by the appropriate axis phase.
        for it in 0..grid_t {
            let f_t = &self.rope_t[it * pairs_t .. (it + 1) * pairs_t];
            for ih in 0..grid_h {
                let f_h = &self.rope_h[ih * pairs_h .. (ih + 1) * pairs_h];
                for iw in 0..grid_w {
                    let f_w = &self.rope_w[iw * pairs_w .. (iw + 1) * pairs_w];
                    let pos = (it * grid_h + ih) * grid_w + iw;
                    for h in 0..n_heads {
                        let head_off = (pos * n_heads + h) * head_dim;
                        for k in 0..pairs_total {
                            let (cos, sin) = if k < pairs_t {
                                f_t[k]
                            } else if k < pairs_t + pairs_h {
                                f_h[k - pairs_t]
                            } else {
                                f_w[k - pairs_t - pairs_h]
                            };
                            let i_re = head_off + 2 * k;
                            let i_im = head_off + 2 * k + 1;
                            let re = data[i_re].to_f32();
                            let im = data[i_im].to_f32();
                            data[i_re] = half::f16::from_f32(re * cos - im * sin);
                            data[i_im] = half::f16::from_f32(re * sin + im * cos);
                        }
                    }
                }
            }
        }

        let device_id = qk.device();
        Tensor::from_slice(
            &data,
            Shape::from([seq_len, n_heads * head_dim]),
            DType::F16,
            device_id,
        )
    }

    /// Full forward pass: noise prediction from noisy latent + text + timestep.
    ///
    /// `latents`: [in_dim, T, H, W] noisy latent (e.g., [16, 5, 30, 52])
    /// `text_embeds`: [text_seq, dim] pre-projected text embeddings
    /// `timestep_embed`: [dim] timestep embedding (after time MLP)
    /// Returns: [in_dim, T, H, W] noise prediction
    pub fn forward(
        &self,
        latents: &Tensor,
        text_embeds: &Tensor,
        timestep_embed: &Tensor,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let config = &self.config;
        let dim = config.dim;
        let num_heads = config.num_heads;
        let head_dim = dim / num_heads;
        let text_seq = text_embeds.shape().dim(0).unwrap_or(1);

        // Get latent dimensions
        let (in_ch, t_dim, h_dim, w_dim) = latents.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("latents must be [C, T, H, W]"))?;

        // 1. 3D patch embedding: [in_dim, T, H, W] → [num_patches, dim]
        // Wan2.1 patch kernel is (kT=1, kH=2, kW=2): no temporal downsampling.
        let num_patches_t = t_dim;
        let num_patches_h = (h_dim + 1) / 2;
        let num_patches_w = (w_dim + 1) / 2;
        let num_patches = num_patches_t * num_patches_h * num_patches_w;

        let patches = self.patch_embed_3d_cpu(latents, in_ch, t_dim, h_dim, w_dim)?;

        // Linear projection: [num_patches, in_dim * 4] → [num_patches, dim]
        let patch_proj_dim = in_ch * 1 * 2 * 2;
        let cb = compute.new_command_buffer();
        let mut h = self.gpu_linear_biased(&cb, &patches, "patch_embedding",
            num_patches, patch_proj_dim, dim, compute)?;
        cb.commit();
        cb.wait_until_completed();
        dump_tensor_if_env("patch_embedding", &h);

        // 2. Compute per-block modulation from timestep.
        // Wan-spec: time_projection = nn.Sequential(SiLU, Linear(dim, 6*dim)).
        // The SiLU before the linear is critical — without it the modulation
        // MLP operates on un-activated input and produces wrong shift/scale/
        // gate vectors for every one of 30 blocks.
        let cb = compute.new_command_buffer();
        let timestep_silu = self.activation(&cb, &self.kernels.silu, timestep_embed);
        let time_proj = self.gpu_linear_biased(&cb, &timestep_silu, "time_projection.1",
            1, dim, 6 * dim, compute)?;
        cb.commit();
        cb.wait_until_completed();

        let mod_data = time_proj.to_f32_vec()?;

        // 3. Transformer blocks
        for layer in 0..config.num_layers {
            let prefix = format!("blocks.{}", layer);

            // Extract 6 modulation vectors for this block from the learned modulation weight.
            // BUG FIX: `mod_w.size()` is the raw on-disk byte size (F32 = numel*4).
            // `mod_w.buffer()` returns the F16-converted buffer of length numel*2.
            // Use the converted buffer's length to size the slice, not the raw size.
            let mod_w = self.w(&format!("{}.modulation", prefix))?;
            let mod_buf = mod_w.buffer();
            let mod_numel = (mod_buf.length() as usize) / std::mem::size_of::<half::f16>();
            let mod_learned: Vec<half::f16> = unsafe {
                let ptr = mod_buf.contents() as *const half::f16;
                std::slice::from_raw_parts(ptr, mod_numel).to_vec()
            };

            // Combine learned modulation + time-dependent modulation
            let device_id = compute.device().info().id;
            let shape_dim = Shape::from([dim]);
            let mut mod_vecs: Vec<Tensor> = Vec::with_capacity(6);
            for m in 0..6 {
                let mut vec = vec![half::f16::ZERO; dim];
                for d in 0..dim {
                    let learned = mod_learned[m * dim + d].to_f32();
                    let time_dep = mod_data[m * dim + d];
                    vec[d] = half::f16::from_f32(learned + time_dep);
                }
                mod_vecs.push(Tensor::from_slice(&vec, shape_dim.clone(), DType::F16, device_id)?);
            }
            // mod_vecs: [shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn]

            let cb = compute.new_command_buffer();

            // === Self-attention with modulation ===
            // Wan-spec: y = self_attn(LayerNorm(h, no_affine) * (1+scale) + shift).
            // Pre-norm (no learnable params) MUST happen before the modulation.
            cb.commit();
            cb.wait_until_completed();
            let h_norm1 = self.cpu_layernorm_no_affine(&h, num_patches, dim)?;
            let cb = compute.new_command_buffer();
            let modulated_sa = self.gpu_adaln_modulate(
                &cb, &h_norm1, &mod_vecs[1], &mod_vecs[0],
                num_patches, dim, compute,
            )?;

            // Q/K/V projections
            let q = self.gpu_linear_biased(&cb, &modulated_sa, &format!("{}.self_attn.q", prefix),
                num_patches, dim, dim, compute)?;
            let k = self.gpu_linear_biased(&cb, &modulated_sa, &format!("{}.self_attn.k", prefix),
                num_patches, dim, dim, compute)?;
            let v = self.gpu_linear_biased(&cb, &modulated_sa, &format!("{}.self_attn.v", prefix),
                num_patches, dim, dim, compute)?;

            // QK-norm (RMSNorm on Q and K)
            let q = self.gpu_rms_norm(&cb, &q, &format!("{}.self_attn.norm_q", prefix),
                num_patches, dim, compute)?;
            let k = self.gpu_rms_norm(&cb, &k, &format!("{}.self_attn.norm_k", prefix),
                num_patches, dim, compute)?;

            // Commit so q/k buffers are filled before we read them on CPU for RoPE.
            cb.commit();
            cb.wait_until_completed();

            if layer == 0 || layer == 29 {
                dump_tensor_if_env(&format!("block{:03}_norm1", layer), &h_norm1);
            }

            // 3D RoPE on Q and K (Wan-spec — head_dim split as time/h/w).
            // Cross-attention does NOT use RoPE; only self-attention.
            let q = self.apply_rope_3d_cpu(&q,
                num_patches_t, num_patches_h, num_patches_w,
                num_heads, head_dim)?;
            let k = self.apply_rope_3d_cpu(&k,
                num_patches_t, num_patches_h, num_patches_w,
                num_heads, head_dim)?;

            // New command buffer for attention onward.
            let cb = compute.new_command_buffer();
            // Multi-head attention via batched matmul
            let attn_out = self.gpu_multi_head_attention(
                &cb, &q, &k, &v, num_patches, num_patches,
                num_heads, head_dim, compute,
            )?;

            // Output projection
            let sa_out = self.gpu_linear_biased(&cb, &attn_out, &format!("{}.self_attn.o", prefix),
                num_patches, dim, dim, compute)?;

            // Gated residual: h = h + gate * sa_out
            let gated_sa = self.gpu_adaln_gate(
                &cb, &h, &sa_out, &mod_vecs[2],
                num_patches, dim, compute,
            )?;

            // === Cross-attention (text conditioning) ===
            // Wan applies LayerNorm WITH affine (norm3) to gated_sa before
            // cross-attention. Previously this norm was applied later (and
            // wrongly used as the FFN pre-norm with norm3's params).
            let ln3_w = self.w(&format!("{}.norm3.weight", prefix))?;
            let ln3_b = self.w(&format!("{}.norm3.bias", prefix))?;
            let pre_cross = self.gpu_layer_norm(&cb, &gated_sa, ln3_w, ln3_b,
                num_patches, dim, compute)?;
            let cq = self.gpu_linear_biased(&cb, &pre_cross, &format!("{}.cross_attn.q", prefix),
                num_patches, dim, dim, compute)?;
            let ck = self.gpu_linear_biased(&cb, text_embeds, &format!("{}.cross_attn.k", prefix),
                text_seq, dim, dim, compute)?;
            let cv = self.gpu_linear_biased(&cb, text_embeds, &format!("{}.cross_attn.v", prefix),
                text_seq, dim, dim, compute)?;

            // QK-norm on cross-attention
            let cq = self.gpu_rms_norm(&cb, &cq, &format!("{}.cross_attn.norm_q", prefix),
                num_patches, dim, compute)?;
            let ck = self.gpu_rms_norm(&cb, &ck, &format!("{}.cross_attn.norm_k", prefix),
                text_seq, dim, compute)?;

            let cross_out = self.gpu_multi_head_attention(
                &cb, &cq, &ck, &cv, num_patches, text_seq,
                num_heads, head_dim, compute,
            )?;
            let cross_proj = self.gpu_linear_biased(&cb, &cross_out, &format!("{}.cross_attn.o", prefix),
                num_patches, dim, dim, compute)?;

            // Cross-attention residual (no gating).
            // Diagnostic: WAN_DISABLE_CROSS=1 skips the cross-attn add entirely
            // (lets us see what the unconditional "image-prior-only" denoising
            // looks like).
            let h_after_cross = if std::env::var("WAN_DISABLE_CROSS").ok().is_some() {
                gated_sa.clone()
            } else {
                self.gpu_add(&cb, &gated_sa, &cross_proj, num_patches * dim, compute)?
            };

            // === FFN with modulation ===
            // Wan-spec: y = ffn(LayerNorm(h, no_affine) * (1+scale_ffn) + shift_ffn).
            // norm2 has NO learnable affine. Previously the Rust port used
            // norm3's weight/bias here, which is the cross-attn norm — wrong tensor.
            cb.commit();
            cb.wait_until_completed();

            if layer == 0 || layer == 29 {
                dump_tensor_if_env(&format!("block{:03}_self_attn", layer), &sa_out);
                dump_tensor_if_env(&format!("block{:03}_norm3", layer), &pre_cross);
                dump_tensor_if_env(&format!("block{:03}_cross_attn", layer), &cross_proj);
            }

            let pre_ffn_norm = self.cpu_layernorm_no_affine(&h_after_cross, num_patches, dim)?;
            let cb = compute.new_command_buffer();
            let modulated_ffn = self.gpu_adaln_modulate(
                &cb, &pre_ffn_norm, &mod_vecs[4], &mod_vecs[3],
                num_patches, dim, compute,
            )?;

            // FFN: linear → GELU(tanh approximate) → linear.
            // Wan-spec is GELU, not SiLU. Previously used SiLU here; that's a
            // wrong nonlinearity in every one of 30 blocks — the dominant single
            // bug responsible for activations not landing where the trained
            // weights expect them.
            let ffn_h = self.gpu_linear_biased(&cb, &modulated_ffn, &format!("{}.ffn.0", prefix),
                num_patches, dim, config.ffn_dim, compute)?;
            let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_h);
            let ffn_out = self.gpu_linear_biased(&cb, &ffn_act, &format!("{}.ffn.2", prefix),
                num_patches, config.ffn_dim, dim, compute)?;

            // Gated FFN residual
            h = self.gpu_adaln_gate(
                &cb, &h_after_cross, &ffn_out, &mod_vecs[5],
                num_patches, dim, compute,
            )?;

            cb.commit();
            cb.wait_until_completed();

            if layer == 0 || layer == 29 {
                dump_tensor_if_env(&format!("block{:03}_norm2", layer), &pre_ffn_norm);
                dump_tensor_if_env(&format!("block{:03}_ffn", layer), &ffn_out);
            }
        }

        // 4. Output head: norm + AdaLN modulation + linear.
        // Wan-spec Head.forward:
        //   e = (head.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        //   x = head.head(head.norm(x) * (1 + e[1]) + e[0])
        // head.modulation shape [1, 2, dim]; head.norm has no affine.
        // Without this, every patch comes out of the linear with no per-position
        // de-modulation — output looks like a flat per-patch tile (checkerboard).
        let head_mod_w = self.w("head.modulation")?;
        let head_mod_buf = head_mod_w.buffer();
        let head_mod_numel = (head_mod_buf.length() as usize) / std::mem::size_of::<half::f16>();
        let head_mod_learned: Vec<half::f16> = unsafe {
            let ptr = head_mod_buf.contents() as *const half::f16;
            std::slice::from_raw_parts(ptr, head_mod_numel).to_vec()
        };
        let time_vec = timestep_embed.to_f32_vec()?;
        let device_id = compute.device().info().id;
        let shape_dim = Shape::from([dim]);
        let mut shift_h = vec![half::f16::ZERO; dim];
        let mut scale_h = vec![half::f16::ZERO; dim];
        for d in 0..dim {
            // head.modulation is [1, 2, dim] flat → [shift; scale]
            shift_h[d] = half::f16::from_f32(head_mod_learned[d].to_f32() + time_vec[d]);
            scale_h[d] = half::f16::from_f32(head_mod_learned[dim + d].to_f32() + time_vec[d]);
        }
        let head_shift = Tensor::from_slice(&shift_h, shape_dim.clone(), DType::F16, device_id)?;
        let head_scale = Tensor::from_slice(&scale_h, shape_dim, DType::F16, device_id)?;

        let h_head_norm = self.cpu_layernorm_no_affine(&h, num_patches, dim)?;
        let cb = compute.new_command_buffer();
        let h_head_mod = self.gpu_adaln_modulate(
            &cb, &h_head_norm, &head_scale, &head_shift,
            num_patches, dim, compute,
        )?;
        // Patch kernel is (1,2,2) → output dim per patch = in_ch * 1 * 2 * 2.
        // Wan stores the head linear under `head.head.{weight,bias}` (not `head.*`).
        let output_flat = self.gpu_linear_biased(&cb, &h_head_mod, "head.head",
            num_patches, dim, in_ch * 1 * 2 * 2, compute)?;
        cb.commit();
        cb.wait_until_completed();
        dump_tensor_if_env("head_input_h", &h);
        dump_tensor_if_env("head_norm_out", &h_head_norm);
        dump_tensor_if_env("head_module", &output_flat);

        // Unpatchify: [num_patches, in_dim*8] → [in_dim, T, H, W]
        self.unpatchify_3d_cpu(&output_flat, in_ch, t_dim, h_dim, w_dim,
            num_patches_t, num_patches_h, num_patches_w)
    }

    /// Compute timestep embedding: sinusoidal → MLP(freq_dim → dim → dim).
    pub fn timestep_embedding(&self, timestep: f32, compute: &MetalCompute) -> Result<Tensor> {
        let config = &self.config;
        let device_id = compute.device().info().id;

        // Sinusoidal frequency embedding.
        // Wan-spec (sinusoidal_embedding_1d): freq_i = 10000^(-i/half).
        // Previously used base 4 (LN_2 * 2 = ln(4)), which produced a
        // time embedding nearly orthogonal to the reference (cos≈0.14)
        // — a constant 4× magnitude error and roughly random direction
        // for every block's modulation, throughout every step.
        let half_dim = config.freq_dim / 2;
        let mut freq_embed = vec![half::f16::ZERO; config.freq_dim];
        let log_base = (10000.0_f32).ln();
        for i in 0..half_dim {
            let freq = (-(i as f32) * log_base / half_dim as f32).exp();
            let angle = timestep * freq;
            freq_embed[i] = half::f16::from_f32(angle.cos());
            freq_embed[half_dim + i] = half::f16::from_f32(angle.sin());
        }
        let freq_tensor = Tensor::from_slice(&freq_embed, Shape::from([1, config.freq_dim]), DType::F16, device_id)?;

        // MLP: linear(freq_dim→dim) → SiLU → linear(dim→dim)
        let cb = compute.new_command_buffer();
        let h = self.gpu_linear_biased(&cb, &freq_tensor, "time_embedding.0",
            1, config.freq_dim, config.dim, compute)?;
        let h = self.activation(&cb, &self.kernels.silu, &h);
        let h = self.gpu_linear_biased(&cb, &h, "time_embedding.2",
            1, config.dim, config.dim, compute)?;
        cb.commit();
        cb.wait_until_completed();

        h.reshape([config.dim])
    }

    /// Project text embeddings: [text_seq, 4096] → [text_seq, dim].
    /// Wan-spec text_embedding is Linear → GELU(approximate='tanh') → Linear.
    /// Previously used SiLU between linears, which produces subtly-wrong text
    /// vectors and shows up as text-conditioning failure (output looks textless,
    /// cross-attention K/V operate on wrong embeddings).
    pub fn project_text(&self, text_embeds: &Tensor, text_seq: usize, compute: &MetalCompute) -> Result<Tensor> {
        let config = &self.config;
        let cb = compute.new_command_buffer();
        let h = self.gpu_linear_biased(&cb, text_embeds, "text_embedding.0",
            text_seq, config.text_dim, config.dim, compute)?;
        let h = self.activation(&cb, &self.kernels.gelu, &h);
        let projected = self.gpu_linear_biased(&cb, &h, "text_embedding.2",
            text_seq, config.dim, config.dim, compute)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(projected)
    }

    // ==================== 3D PATCH OPERATIONS (CPU) ====================

    /// 3D patch embedding on CPU.
    /// Input: [C, T, H, W], patch size 2×2×2.
    /// Output: [num_patches, C*8] (flattened patches, row-major).
    fn patch_embed_3d_cpu(
        &self, input: &Tensor,
        c: usize, t: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let data: Vec<half::f16> = input.to_vec()?;
        // Wan2.1 patch kernel = (kT=1, kH=2, kW=2). Temporal axis is NOT
        // downsampled — every latent timestep becomes one patch row.
        let pt = 1usize;
        let ph = 2usize;
        let pw = 2usize;
        let nt = (t + pt - 1) / pt;
        let nh = (h + ph - 1) / ph;
        let nw = (w + pw - 1) / pw;
        let num_patches = nt * nh * nw;
        let patch_dim = c * pt * ph * pw;

        let mut patches = vec![half::f16::ZERO; num_patches * patch_dim];

        for it in 0..nt {
            for ih in 0..nh {
                for iw in 0..nw {
                    let patch_idx = (it * nh + ih) * nw + iw;
                    for ic in 0..c {
                        for dt in 0..pt {
                            for dh in 0..ph {
                                for dw in 0..pw {
                                    let gt = it * pt + dt;
                                    let gh = ih * ph + dh;
                                    let gw = iw * pw + dw;
                                    let local_idx = ((ic * pt + dt) * ph + dh) * pw + dw;
                                    if gt < t && gh < h && gw < w {
                                        let src_idx = ((ic * t + gt) * h + gh) * w + gw;
                                        patches[patch_idx * patch_dim + local_idx] = data[src_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let device_id = input.device();
        Tensor::from_slice(&patches, Shape::from([num_patches, patch_dim]), DType::F16, device_id)
    }

    /// Reverse 3D patch embedding on CPU.
    /// Input: [num_patches, C*8], Output: [C, T, H, W].
    fn unpatchify_3d_cpu(
        &self, input: &Tensor,
        c: usize, t: usize, h: usize, w: usize,
        nt: usize, nh: usize, nw: usize,
    ) -> Result<Tensor> {
        let data: Vec<half::f16> = input.to_vec()?;
        // Wan-spec unpatchify: head.head outputs 64 elements per patch,
        // interpreted by Wan as `view(F, H_p, W_p, p, q, r, c)` with c
        // INNERMOST (channel-fastest). The trained head linear's 64 outputs
        // are ordered (p, q, r, c) → flat_idx = (dt*ph + dh)*pw*c + dw*c + ic.
        //
        // Previously we used the c-OUTERMOST (c*pt*ph*pw + ...) layout that
        // mirrored patch_embed_3d_cpu — but patch_embed comes from a Conv3d
        // weight (which IS c-outermost), while head is a plain Linear that
        // Wan reshapes with c-innermost. This asymmetry produced cosine
        // similarity 0.07 to the reference v_final even though every
        // intermediate block was cos=1.0.
        let pt = 1usize;
        let ph = 2usize;
        let pw = 2usize;
        let patch_dim = c * pt * ph * pw;

        let mut output = vec![half::f16::ZERO; c * t * h * w];

        for it in 0..nt {
            for ih in 0..nh {
                for iw in 0..nw {
                    let patch_idx = (it * nh + ih) * nw + iw;
                    for ic in 0..c {
                        for dt in 0..pt {
                            for dh in 0..ph {
                                for dw in 0..pw {
                                    let gt = it * pt + dt;
                                    let gh = ih * ph + dh;
                                    let gw = iw * pw + dw;
                                    if gt < t && gh < h && gw < w {
                                        // c-innermost: ((dt*ph + dh)*pw + dw)*c + ic
                                        let local_idx = ((dt * ph + dh) * pw + dw) * c + ic;
                                        let dst_idx = ((ic * t + gt) * h + gh) * w + gw;
                                        output[dst_idx] = data[patch_idx * patch_dim + local_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let device_id = input.device();
        Tensor::from_slice(&output, Shape::from([c, t, h, w]), DType::F16, device_id)
    }

    // ==================== GPU HELPERS ====================

    fn w(&self, name: &str) -> Result<&LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("Wan weight not found: {}", name)))
    }

    fn gpu_linear_biased(
        &self, cb: &metal::CommandBufferRef, input: &Tensor, prefix: &str,
        m: usize, k: usize, n: usize, compute: &MetalCompute,
    ) -> Result<Tensor> {
        let weight = self.w(&format!("{}.weight", prefix))?;
        let bias = self.model.get_weight(&format!("{}.bias", prefix));
        let device = compute.device().raw();
        let output_size = m * n * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let tile: usize = 16;
        let grid_x = (n + tile - 1) / tile;
        let grid_y = (m + tile - 1) / tile;

        compute.dispatch(
            cb, &self.kernels.common.linear,
            (grid_x, grid_y, 1), (tile, tile, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                if let Some(b) = &bias {
                    set_lazy_buffer(encoder, 2, b);
                } else {
                    encoder.set_buffer(2, Some(&output_buffer), 0);
                }
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let m_u32 = m as u32;
                let n_u32 = n as u32;
                let k_u32 = k as u32;
                let has_bias: u32 = if bias.is_some() { 1 } else { 0 };
                encoder.set_bytes(4, 4, &m_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
                encoder.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            },
        );

        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, compute.device().info().id))
    }

    fn gpu_rms_norm(
        &self, cb: &metal::CommandBufferRef, input: &Tensor, prefix: &str,
        num_rows: usize, dim: usize, compute: &MetalCompute,
    ) -> Result<Tensor> {
        let weight = self.w(&format!("{}.weight", prefix))?;
        let device = compute.device().raw();
        let output_buffer = device.new_buffer((num_rows * dim * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        // rms_norm_f16 kernel: input(0), weight(1), output(2), N(3), D(4), eps(5)
        let c_n = num_rows as u32;
        let c_d = dim as u32;
        let eps = self.config.eps;
        compute.dispatch(
            cb, &self.kernels.rms_norm,
            (num_rows, 1, 1), (1, 1, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_bytes(3, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_d as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            },
        );

        Ok(Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, compute.device().info().id))
    }

    fn gpu_layer_norm(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        weight: &LazyTensor, bias: &LazyTensor,
        n: usize, d: usize, compute: &MetalCompute,
    ) -> Result<Tensor> {
        let device = compute.device().raw();
        let output_buffer = device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        compute.dispatch_1d(
            cb, &self.kernels.common.layer_norm, n,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                set_lazy_buffer(encoder, 2, bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let n_u32 = n as u32;
                let d_u32 = d as u32;
                let eps = self.config.eps;
                encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
            },
        );

        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([n, d]), DType::F16, compute.device().info().id))
    }

    fn gpu_add(
        &self, cb: &metal::CommandBufferRef, a: &Tensor, b: &Tensor,
        count: usize, compute: &MetalCompute,
    ) -> Result<Tensor> {
        let device = compute.device().raw();
        let output_buffer = device.new_buffer((count * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        compute.dispatch_1d(
            cb, &self.kernels.common.add, count,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, a);
                gpu_ops::set_tensor_buffer(encoder, 1, b);
                encoder.set_buffer(2, Some(&output_buffer), 0);
            },
        );

        Ok(Tensor::from_metal_buffer(output_buffer, a.shape().clone(), DType::F16, compute.device().info().id))
    }

    fn gpu_adaln_modulate(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        scale: &Tensor, shift: &Tensor,
        seq_len: usize, hidden: usize, compute: &MetalCompute,
    ) -> Result<Tensor> {
        let device = compute.device().raw();
        let total = seq_len * hidden;
        let output_buffer = device.new_buffer((total * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        compute.dispatch_1d(
            cb, &self.kernels.adaln_modulate, total,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, scale);
                gpu_ops::set_tensor_buffer(encoder, 2, shift);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let h = hidden as u32;
                let c = total as u32;
                encoder.set_bytes(4, 4, &h as *const u32 as *const _);
                // Bug fix: kernel reads `count` from buffer 5 for the bounds check.
                // Previously unset → uninit zero → every thread early-returned →
                // output stayed at the all-zero initial buffer state.
                encoder.set_bytes(5, 4, &c as *const u32 as *const _);
            },
        );

        Ok(Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, compute.device().info().id))
    }

    fn gpu_adaln_gate(
        &self, cb: &metal::CommandBufferRef, residual: &Tensor, update: &Tensor,
        gate: &Tensor, seq_len: usize, hidden: usize, compute: &MetalCompute,
    ) -> Result<Tensor> {
        let device = compute.device().raw();
        let total = seq_len * hidden;
        let output_buffer = device.new_buffer((total * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        compute.dispatch_1d(
            cb, &self.kernels.adaln_gate, total,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, residual);
                gpu_ops::set_tensor_buffer(encoder, 1, update);
                gpu_ops::set_tensor_buffer(encoder, 2, gate);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let h = hidden as u32;
                let c = total as u32;
                encoder.set_bytes(4, 4, &h as *const u32 as *const _);
                // Same bug fix as gpu_adaln_modulate: kernel needs `count` at buffer 5.
                encoder.set_bytes(5, 4, &c as *const u32 as *const _);
            },
        );

        Ok(Tensor::from_metal_buffer(output_buffer, residual.shape().clone(), DType::F16, compute.device().info().id))
    }

    /// Multi-head attention via batched matmul.
    fn gpu_multi_head_attention(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k: &Tensor, v: &Tensor,
        q_seq: usize, kv_seq: usize,
        num_heads: usize, head_dim: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let device = compute.device().raw();
        let device_id = compute.device().info().id;

        // Reshape: [seq, dim] → [seq, heads, head_dim]
        let q = q.reshape([q_seq, num_heads, head_dim])?;
        let k = k.reshape([kv_seq, num_heads, head_dim])?;
        let v = v.reshape([kv_seq, num_heads, head_dim])?;

        // Transpose: [S, H, D] → [H, S, D]
        let q_t = Tensor::empty(Shape::from([num_heads, q_seq, head_dim]), DType::F16, device_id)?;
        let k_t = Tensor::empty(Shape::from([num_heads, kv_seq, head_dim]), DType::F16, device_id)?;
        let v_t = Tensor::empty(Shape::from([num_heads, kv_seq, head_dim]), DType::F16, device_id)?;

        self.transpose_shd_to_hsd(cb, &q, &q_t, q_seq, num_heads, head_dim);
        self.transpose_shd_to_hsd(cb, &k, &k_t, kv_seq, num_heads, head_dim);
        self.transpose_shd_to_hsd(cb, &v, &v_t, kv_seq, num_heads, head_dim);

        // === Mixed-precision attention ===
        // Q@K^T → F32 scores (4 bytes/elem), scale + softmax in F32, write
        // F16 attention weights. Then weights @ V in F16 as before.
        // F16 score storage was overflowing for 21-frame@480p (131k tokens
        // per query, score sums + softmax denominators exceed F16 range).
        // F32 scores buffer fixes this.
        let n_scores = num_heads * q_seq * kv_seq;
        let scores_buffer = device.new_buffer(
            (n_scores * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        {
            let tile: usize = 16;
            compute.dispatch(
                cb, &self.kernels.batched_linear_out_f32,
                ((kv_seq + tile - 1) / tile, (q_seq + tile - 1) / tile, num_heads),
                (tile, tile, 1),
                |encoder| {
                    gpu_ops::set_tensor_buffer(encoder, 0, &q_t);
                    gpu_ops::set_tensor_buffer(encoder, 1, &k_t);
                    encoder.set_buffer(2, Some(&scores_buffer), 0);
                    let m = q_seq as u32;
                    let n = kv_seq as u32;
                    let k_dim = head_dim as u32;
                    encoder.set_bytes(3, 4, &m as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &n as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &k_dim as *const u32 as *const _);
                },
            );
        }

        // Scale + softmax in F32 → F16 attention weights.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let weights_buffer = device.new_buffer(
            (n_scores * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        {
            let total_rows = num_heads * q_seq;
            compute.dispatch_1d(
                cb, &self.kernels.softmax_scale_f32_to_f16, total_rows,
                |encoder| {
                    encoder.set_buffer(0, Some(&scores_buffer), 0);
                    encoder.set_buffer(1, Some(&weights_buffer), 0);
                    let rows = total_rows as u32;
                    let cols = kv_seq as u32;
                    encoder.set_bytes(2, 4, &rows as *const u32 as *const _);
                    encoder.set_bytes(3, 4, &cols as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &scale as *const f32 as *const _);
                },
            );
        }

        // Output = Weights @ V' → [H, q_seq, head_dim]
        let output_t = Tensor::empty(
            Shape::from([num_heads, q_seq, head_dim]), DType::F16, device_id)?;
        {
            let tile: usize = 16;
            compute.dispatch(
                cb, &self.kernels.common.batched_matmul_nn,
                ((head_dim + tile - 1) / tile, (q_seq + tile - 1) / tile, num_heads),
                (tile, tile, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&weights_buffer), 0);
                    gpu_ops::set_tensor_buffer(encoder, 1, &v_t);
                    gpu_ops::set_tensor_buffer(encoder, 2, &output_t);
                    let m = q_seq as u32;
                    let n = head_dim as u32;
                    let k_dim = kv_seq as u32;
                    encoder.set_bytes(3, 4, &m as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &n as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &k_dim as *const u32 as *const _);
                },
            );
        }

        // Transpose back: [H, S, D] → [S, H*D]
        let output = Tensor::empty(
            Shape::from([q_seq, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(cb, &output_t, &output, q_seq, num_heads, head_dim);

        output.reshape([q_seq, num_heads * head_dim])
    }

}

#[cfg(feature = "metal")]
fn set_lazy_buffer(encoder: &metal::ComputeCommandEncoderRef, index: u64, lt: &LazyTensor) {
    encoder.set_buffer(index, Some(lt.buffer()), 0);
}

// ==================== Wan2.1 3D Causal VAE Decoder (CPU) ====================

/// Wan2.1 3D Causal VAE decoder configuration.
#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    /// Latent channels (16).
    pub latent_dim: usize,
    /// Output RGB channels (3).
    pub out_channels: usize,
    /// Spatial downscale factor (8x = 3 spatial upsample stages).
    pub spatial_factor: usize,
    /// Temporal factor (~4x = 2 temporal upsample stages).
    pub temporal_factor: usize,
}

impl Default for WanVaeConfig {
    fn default() -> Self {
        Self {
            latent_dim: 16,
            out_channels: 3,
            spatial_factor: 8,
            temporal_factor: 4,
        }
    }
}

/// Wan2.1 3D Causal VAE decoder — CPU implementation.
///
/// Decodes [16, T, H/8, W/8] latents → [3, T*4, H, W] RGB video frames.
/// All operations on CPU using f32 for precision (weights loaded as f16, converted).
///
/// Architecture:
///   conv2(16→16, 1³) → decoder.conv1(16→384, 3³) → middle(ResBlock+Attn+ResBlock) →
///   15 upsample blocks (spatial 8×, temporal 4×) → head(GroupNorm+SiLU+Conv3d 96→3)
#[cfg(feature = "metal")]
pub struct WanVaeDecoder {
    model: Arc<Model>,
}

#[cfg(feature = "metal")]
impl WanVaeDecoder {
    /// Create a new VAE decoder with pre-loaded model weights.
    pub fn new(model: Arc<Model>) -> Self {
        Self { model }
    }

    /// Decode latents to RGB frames.
    ///
    /// `latents`: [latent_dim, T, H, W] (f16 on GPU) → [3, T_out, H*8, W*8] (f16)
    pub fn decode(&self, latents: &Tensor, device_id: DeviceId) -> Result<Tensor> {
        // Transfer to CPU f32
        let shape = latents.shape();
        let (c, t, h, w) = shape.dims4()
            .ok_or_else(|| crate::core::Error::internal("latents must be [C, T, H, W]"))?;
        let data_f16: Vec<half::f16> = latents.to_vec()?;
        let mut x: Vec<f32> = data_f16.iter().map(|v| v.to_f32()).collect();

        // Wan-spec: DiT predicts in NORMALIZED latent space (mean~0, std~1).
        // VAE was trained on un-normalized latents — we must invert the
        // normalization before running the decoder. Per Wan reference:
        //   z_decoder_input = z_predicted * std + mean   (per-channel)
        // Without this step the decoder sees near-zero, slightly-negative
        // values and outputs low-amplitude greenish soup regardless of prompt.
        // Constants from wan/modules/vae.py WanVAE class (z_dim=16).
        const VAE_LATENT_MEAN: [f32; 16] = [
            -0.7571, -0.7089, -0.9113, 0.1075,
            -0.1745,  0.9653, -0.1517, 1.5508,
             0.4134, -0.0715,  0.5517, -0.3632,
            -0.1922, -0.9497,  0.2503, -0.2921,
        ];
        const VAE_LATENT_STD: [f32; 16] = [
            2.8184, 1.4541, 2.3275, 2.6558,
            1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579,
            1.6382, 1.1253, 2.8251, 1.9160,
        ];
        let plane = t * h * w;
        for ci in 0..c.min(16) {
            let m = VAE_LATENT_MEAN[ci];
            let s = VAE_LATENT_STD[ci];
            let off = ci * plane;
            for j in 0..plane {
                x[off + j] = x[off + j] * s + m;
            }
        }

        // conv2: post-quantization conv, [16, 16, 1, 1, 1]
        x = self.conv3d_cpu(&x, c, t, h, w, "conv2", 16, 16, 1, 1, 1, 0, 0, 0)?;

        // decoder.conv1: [384, 16, 3, 3, 3] with causal temporal padding
        let (c, t, h, w) = (16, t, h, w);
        x = self.conv3d_cpu(&x, c, t, h, w, "decoder.conv1", 384, 16, 3, 3, 3, 2, 1, 1)?;
        let (mut c_cur, mut t_cur, mut h_cur, mut w_cur) = (384, t, h, w);

        // decoder.middle: ResBlock(0) + SelfAttention(1) + ResBlock(2)
        x = self.resblock_cpu(&x, c_cur, t_cur, h_cur, w_cur, "decoder.middle.0", c_cur)?;
        x = self.spatial_attention_cpu(&x, c_cur, t_cur, h_cur, w_cur, "decoder.middle.1")?;
        x = self.resblock_cpu(&x, c_cur, t_cur, h_cur, w_cur, "decoder.middle.2", c_cur)?;

        // decoder.upsamples[0..14]: see architecture table
        let block_configs: Vec<(usize, bool, bool)> = vec![
            // (out_channels, is_spatial_upsample, has_temporal_upsample)
            (384, false, false),  // 0: ResBlock 384→384
            (384, false, false),  // 1: ResBlock 384→384
            (384, false, false),  // 2: ResBlock 384→384
            (192, true, true),    // 3: Spatial+Temporal upsample 384→192
            (384, false, false),  // 4: ResBlock 192→384 (shortcut)
            (384, false, false),  // 5: ResBlock 384→384
            (384, false, false),  // 6: ResBlock 384→384
            (192, true, true),    // 7: Spatial+Temporal upsample 384→192
            (192, false, false),  // 8: ResBlock 192→192
            (192, false, false),  // 9: ResBlock 192→192
            (192, false, false),  // 10: ResBlock 192→192
            (96, true, false),    // 11: Spatial upsample only 192→96
            (96, false, false),   // 12: ResBlock 96→96
            (96, false, false),   // 13: ResBlock 96→96
            (96, false, false),   // 14: ResBlock 96→96
        ];

        for (i, &(out_ch, is_spatial, has_temporal)) in block_configs.iter().enumerate() {
            let prefix = format!("decoder.upsamples.{}", i);
            if is_spatial {
                // Spatial upsample: nearest 2x + Conv2d per frame
                x = self.spatial_upsample_cpu(&x, c_cur, t_cur, h_cur, w_cur, &prefix, out_ch)?;
                h_cur *= 2;
                w_cur *= 2;
                c_cur = out_ch;
                if has_temporal {
                    // Temporal upsample: time_conv [out*2, c_cur, 3, 1, 1] → pixel shuffle 2x in time
                    x = self.temporal_upsample_cpu(&x, c_cur, t_cur, h_cur, w_cur, &prefix)?;
                    t_cur *= 2;
                }
            } else {
                x = self.resblock_cpu(&x, c_cur, t_cur, h_cur, w_cur, &prefix, out_ch)?;
                c_cur = out_ch;
            }
        }

        // decoder.head: GroupNorm → SiLU → Conv3d(96→3, 3³)
        x = self.group_norm_cpu(&x, c_cur, t_cur * h_cur * w_cur, "decoder.head.0")?;
        x = silu_vec(&x);
        x = self.conv3d_cpu(&x, c_cur, t_cur, h_cur, w_cur, "decoder.head.2", 3, c_cur, 3, 3, 3, 2, 1, 1)?;
        c_cur = 3;

        // Convert back to f16 tensor
        let out_f16: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v.clamp(-1.0, 1.0))).collect();
        Tensor::from_slice(&out_f16, Shape::from([c_cur, t_cur, h_cur, w_cur]), DType::F16, device_id)
    }

    // ==================== CPU Primitives ====================

    /// Conv3d on CPU with causal temporal padding.
    ///
    /// Causal: pad_t zeros at front, 0 at back in time dimension.
    /// Spatial: symmetric padding.
    fn conv3d_cpu(
        &self, input: &[f32],
        c_in: usize, t: usize, h: usize, w: usize,
        prefix: &str, c_out: usize, _c_in_check: usize,
        kt: usize, kh: usize, kw: usize,
        pad_t: usize, pad_h: usize, pad_w: usize,
    ) -> Result<Vec<f32>> {
        use crate::tensor::ops::sgemm_transb_cpu;

        let weight = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.weight", prefix))?;
        let bias_name = format!("{}.bias", prefix);
        let bias = gpu_ops::read_weight_vec_f32(&self.model, &bias_name).ok();

        let num_positions = t * h * w;
        let col_dim = c_in * kt * kh * kw;

        // im2col: extract 3D patches → [num_positions, col_dim]
        let mut im2col = vec![0.0f32; num_positions * col_dim];
        for ot in 0..t {
            for oh in 0..h {
                for ow in 0..w {
                    let pos = (ot * h + oh) * w + ow;
                    for ic in 0..c_in {
                        for dt in 0..kt {
                            for dh in 0..kh {
                                for dw in 0..kw {
                                    let it = ot as isize + dt as isize - pad_t as isize;
                                    let ih = oh as isize + dh as isize - pad_h as isize;
                                    let iw = ow as isize + dw as isize - pad_w as isize;
                                    let col = ((ic * kt + dt) * kh + dh) * kw + dw;
                                    im2col[pos * col_dim + col] = if it >= 0 && it < t as isize &&
                                        ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                        input[((ic * t + it as usize) * h + ih as usize) * w + iw as usize]
                                    } else {
                                        0.0
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }

        // Matmul: [num_positions, col_dim] @ [c_out, col_dim]^T → [num_positions, c_out]
        let mut output_nhwc = vec![0.0f32; num_positions * c_out];
        sgemm_transb_cpu(&im2col, &weight, &mut output_nhwc, num_positions, c_out, col_dim);

        // Transpose from [position, c_out] to [c_out, t, h, w] and add bias
        let mut output = vec![0.0f32; c_out * num_positions];
        for oc in 0..c_out {
            let b = bias.as_ref().map(|b| b[oc]).unwrap_or(0.0);
            for pos in 0..num_positions {
                output[oc * num_positions + pos] = output_nhwc[pos * c_out + oc] + b;
            }
        }
        Ok(output)
    }

    /// ResBlock: GroupNorm → SiLU → Conv3d → GroupNorm → SiLU → Conv3d + residual.
    fn resblock_cpu(
        &self, input: &[f32],
        c_in: usize, t: usize, h: usize, w: usize,
        prefix: &str, c_out: usize,
    ) -> Result<Vec<f32>> {
        let spatial = t * h * w;

        // residual.0: GroupNorm
        let mut x = self.group_norm_cpu(input, c_in, spatial, &format!("{}.residual.0", prefix))?;

        // residual.1: SiLU
        x = silu_vec(&x);

        // residual.2: Conv3d (3×3×3, causal pad_t=2, pad_h=1, pad_w=1)
        x = self.conv3d_cpu(&x, c_in, t, h, w, &format!("{}.residual.2", prefix),
            c_out, c_in, 3, 3, 3, 2, 1, 1)?;

        // residual.3: GroupNorm
        x = self.group_norm_cpu(&x, c_out, spatial, &format!("{}.residual.3", prefix))?;

        // residual.4: SiLU
        x = silu_vec(&x);

        // residual.5: Dropout (skip at inference)
        // residual.6: Conv3d (3×3×3)
        x = self.conv3d_cpu(&x, c_out, t, h, w, &format!("{}.residual.6", prefix),
            c_out, c_out, 3, 3, 3, 2, 1, 1)?;

        // Shortcut if channels differ
        let residual = if c_in != c_out {
            self.conv3d_cpu(input, c_in, t, h, w, &format!("{}.shortcut", prefix),
                c_out, c_in, 1, 1, 1, 0, 0, 0)?
        } else {
            input.to_vec()
        };

        // Add residual
        Ok(x.iter().zip(residual.iter()).map(|(a, b)| a + b).collect())
    }

    /// GroupNorm with gamma parameter (no beta/bias).
    /// Groups = 32 (standard for video VAEs).
    fn group_norm_cpu(&self, input: &[f32], channels: usize, spatial: usize, prefix: &str) -> Result<Vec<f32>> {
        let gamma = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.gamma", prefix))?;
        let num_groups = 32;
        let channels_per_group = channels / num_groups;
        let eps = 1e-5f32;

        let mut output = vec![0.0f32; channels * spatial];

        for g in 0..num_groups {
            let ch_start = g * channels_per_group;
            let ch_end = ch_start + channels_per_group;

            // Compute mean and variance for this group across all spatial positions
            let group_size = channels_per_group * spatial;
            let mut mean = 0.0f64;
            for c in ch_start..ch_end {
                for s in 0..spatial {
                    mean += input[c * spatial + s] as f64;
                }
            }
            mean /= group_size as f64;

            let mut var = 0.0f64;
            for c in ch_start..ch_end {
                for s in 0..spatial {
                    let diff = input[c * spatial + s] as f64 - mean;
                    var += diff * diff;
                }
            }
            var /= group_size as f64;

            let inv_std = 1.0 / (var + eps as f64).sqrt();

            for c in ch_start..ch_end {
                let g_val = gamma[c] as f64;
                for s in 0..spatial {
                    let idx = c * spatial + s;
                    output[idx] = ((input[idx] as f64 - mean) * inv_std * g_val) as f32;
                }
            }
        }
        Ok(output)
    }

    /// Spatial upsample: nearest-neighbor 2x → Conv2d(3×3) per frame.
    fn spatial_upsample_cpu(
        &self, input: &[f32],
        c_in: usize, t: usize, h: usize, w: usize,
        prefix: &str, c_out: usize,
    ) -> Result<Vec<f32>> {
        let h2 = h * 2;
        let w2 = w * 2;

        // Nearest-neighbor 2x upsample
        let mut upsampled = vec![0.0f32; c_in * t * h2 * w2];
        for c in 0..c_in {
            for ft in 0..t {
                for fh in 0..h2 {
                    for fw in 0..w2 {
                        let src_h = fh / 2;
                        let src_w = fw / 2;
                        upsampled[((c * t + ft) * h2 + fh) * w2 + fw] =
                            input[((c * t + ft) * h + src_h) * w + src_w];
                    }
                }
            }
        }

        // Conv2d(3×3) per frame via im2col + sgemm (AMX-accelerated)
        let weight = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.resample.1.weight", prefix))?;
        let bias = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.resample.1.bias", prefix))?;
        let hw2 = h2 * w2;
        let col_dim = c_in * 9; // c_in * 3 * 3
        let mut output = vec![0.0f32; c_out * t * hw2];
        let mut im2col = vec![0.0f32; hw2 * col_dim];
        let mut frame_out = vec![0.0f32; hw2 * c_out];

        for ft in 0..t {
            // im2col: extract 3×3 patches for this frame → [h2*w2, c_in*9]
            for oh in 0..h2 {
                for ow in 0..w2 {
                    let pos = oh * w2 + ow;
                    for ic in 0..c_in {
                        for dh in 0..3usize {
                            for dw in 0..3usize {
                                let ih = oh as isize + dh as isize - 1;
                                let iw = ow as isize + dw as isize - 1;
                                let col = ic * 9 + dh * 3 + dw;
                                im2col[pos * col_dim + col] = if ih >= 0 && ih < h2 as isize &&
                                    iw >= 0 && iw < w2 as isize {
                                    upsampled[((ic * t + ft) * h2 + ih as usize) * w2 + iw as usize]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                }
            }

            // sgemm: [h2*w2, col_dim] @ [c_out, col_dim]^T → [h2*w2, c_out]
            crate::tensor::ops::sgemm_transb_cpu(&im2col, &weight, &mut frame_out, hw2, c_out, col_dim);

            // Transpose to [c_out, pos] and add bias
            for oc in 0..c_out {
                let b = bias[oc];
                for pos in 0..hw2 {
                    output[((oc * t + ft) * h2) * w2 + pos] = frame_out[pos * c_out + oc] + b;
                }
            }
        }
        Ok(output)
    }

    /// Temporal upsample via time_conv + pixel shuffle.
    ///
    /// time_conv: Conv3d [c_out*2, c_in, 3, 1, 1] → pixel-shuffle 2x in time.
    fn temporal_upsample_cpu(
        &self, input: &[f32],
        c_in: usize, t: usize, h: usize, w: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let c_out2 = c_in * 2;
        // time_conv: Conv3d with kernel (3, 1, 1), causal pad_t=2
        let conv_out = self.conv3d_cpu(input, c_in, t, h, w,
            &format!("{}.time_conv", prefix),
            c_out2, c_in, 3, 1, 1, 2, 0, 0)?;

        // Pixel shuffle 2x in time: [c_out*2, T, H, W] → [c_out, T*2, H, W]
        let t2 = t * 2;
        let mut output = vec![0.0f32; c_in * t2 * h * w];
        for c in 0..c_in {
            for ft in 0..t {
                for fh in 0..h {
                    for fw in 0..w {
                        // Even time positions from channels [0..c_in]
                        output[((c * t2 + ft * 2) * h + fh) * w + fw] =
                            conv_out[((c * t + ft) * h + fh) * w + fw];
                        // Odd time positions from channels [c_in..c_in*2]
                        output[((c * t2 + ft * 2 + 1) * h + fh) * w + fw] =
                            conv_out[(((c_in + c) * t + ft) * h + fh) * w + fw];
                    }
                }
            }
        }
        Ok(output)
    }

    /// 2D self-attention per frame (middle block only).
    ///
    /// All matmuls accelerated via AMX (cblas_sgemm) instead of triple-nested loops.
    fn spatial_attention_cpu(
        &self, input: &[f32],
        c: usize, t: usize, h: usize, w: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        use crate::tensor::ops::{sgemm_cpu, sgemm_transa_cpu, sgemm_transb_cpu};

        let hw = h * w;
        let gamma = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.norm.gamma", prefix))?;
        let qkv_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.to_qkv.weight", prefix))?;
        let qkv_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.to_qkv.bias", prefix))?;
        let proj_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.proj.weight", prefix))?;
        let proj_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.proj.bias", prefix))?;

        let c3 = c * 3;
        let scale = 1.0 / (c as f32).sqrt();
        let num_groups = 32;
        let cpg = c / num_groups;
        let eps = 1e-5f32;

        let mut output = input.to_vec();

        for ft in 0..t {
            // Extract frame: [c, h, w]
            let mut frame = vec![0.0f32; c * hw];
            for ic in 0..c {
                for s in 0..hw {
                    frame[ic * hw + s] = input[((ic * t + ft) * h) * w + s / w * w + s % w];
                }
            }

            // GroupNorm (2D)
            let mut normed = vec![0.0f32; c * hw];
            for g in 0..num_groups {
                let ch_start = g * cpg;
                let ch_end = ch_start + cpg;
                let group_size = cpg * hw;
                let mut mean = 0.0f64;
                for cc in ch_start..ch_end {
                    for s in 0..hw {
                        mean += frame[cc * hw + s] as f64;
                    }
                }
                mean /= group_size as f64;
                let mut var = 0.0f64;
                for cc in ch_start..ch_end {
                    for s in 0..hw {
                        let d = frame[cc * hw + s] as f64 - mean;
                        var += d * d;
                    }
                }
                var /= group_size as f64;
                let inv_std = 1.0 / (var + eps as f64).sqrt();
                for cc in ch_start..ch_end {
                    let g_val = gamma[cc] as f64;
                    for s in 0..hw {
                        normed[cc * hw + s] = ((frame[cc * hw + s] as f64 - mean) * inv_std * g_val) as f32;
                    }
                }
            }

            // QKV = qkv_w @ normed + bias: [c3, c] @ [c, hw] → [c3, hw]
            let mut qkv = vec![0.0f32; c3 * hw];
            sgemm_cpu(&qkv_w, &normed, &mut qkv, c3, hw, c);
            for oc in 0..c3 {
                let b = qkv_b[oc];
                for s in 0..hw { qkv[oc * hw + s] += b; }
            }

            let q = &qkv[0..c * hw];
            let k = &qkv[c * hw..2 * c * hw];
            let v = &qkv[2 * c * hw..3 * c * hw];

            // Scores = Q^T @ K * scale: [hw, hw]
            // Q is [c, hw], so Q^T @ K = [hw, c] @ [c, hw] = [hw, hw]
            let mut scores = vec![0.0f32; hw * hw];
            sgemm_transa_cpu(q, k, &mut scores, hw, hw, c);
            for s in &mut scores { *s *= scale; }

            // Row-wise softmax
            for qi in 0..hw {
                let row = &mut scores[qi * hw..(qi + 1) * hw];
                let max_s = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in row.iter_mut() { *s = (*s - max_s).exp(); sum_exp += *s; }
                for s in row.iter_mut() { *s /= sum_exp; }
            }

            // Attn output = V @ scores^T: [c, hw] @ [hw, hw]^T → [c, hw]
            let mut attn_out = vec![0.0f32; c * hw];
            sgemm_transb_cpu(v, &scores, &mut attn_out, c, hw, hw);

            // Output projection: proj_w @ attn_out + bias: [c, c] @ [c, hw] → [c, hw]
            let mut projected = vec![0.0f32; c * hw];
            sgemm_cpu(&proj_w, &attn_out, &mut projected, c, hw, c);
            for oc in 0..c {
                let b = proj_b[oc];
                for s in 0..hw { projected[oc * hw + s] += b; }
            }

            // Write back with residual
            for ic in 0..c {
                for fh in 0..h {
                    for fw in 0..w {
                        let out_idx = ((ic * t + ft) * h + fh) * w + fw;
                        let local_idx = ic * hw + fh * w + fw;
                        output[out_idx] = input[out_idx] + projected[local_idx];
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Elementwise SiLU: x * sigmoid(x).
fn silu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * (1.0 / (1.0 + (-v).exp()))).collect()
}
