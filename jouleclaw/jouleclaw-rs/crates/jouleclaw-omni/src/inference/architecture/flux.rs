//! Flux Transformer architecture.
//!
//! Implements the Flux image generation transformer with:
//! - Double-stream blocks (joint image + text attention)
//! - Single-stream blocks (fused QKV + MLP)
//! - AdaLN modulation (adaptive layer normalization)
//! - QK-norm (RMSNorm on Q/K projections)
//! - 2×2 patchification for spatial token reduction

use crate::inference::model::Model;
use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::MetalCompute;
#[cfg(feature = "metal")]
use crate::hal::metal::BorrowedMetalBuffer;
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

/// Flux transformer configuration.
#[derive(Debug, Clone)]
pub struct FluxConfig {
    /// Hidden dimension (3072 for dev, 6144 for full/dev2).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Number of double-stream (joint img+txt) layers.
    pub num_double_layers: usize,
    /// Number of single-stream (fused) layers.
    pub num_single_layers: usize,
    /// Text conditioning dimension (4096 for T5-XXL, 15360 for Mistral 3 projected).
    pub context_dim: usize,
    /// MLP expansion ratio.
    pub mlp_ratio: f32,
    /// RoPE frequency base.
    pub theta: f32,
    /// Whether modulation weights are shared across blocks (Flux 2).
    pub shared_modulation: bool,
    /// Whether linear layers have biases (Flux 1 = true, Flux 2 = false).
    pub use_bias: bool,
}

impl FluxConfig {
    /// Flux 1 Dev configuration (12B params).
    pub fn dev() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 24,
            head_dim: 128,
            num_double_layers: 19,
            num_single_layers: 38,
            context_dim: 4096,
            mlp_ratio: 4.0,
            theta: 10000.0,
            shared_modulation: false,
            use_bias: true,
        }
    }

    /// Flux 2 Dev configuration (larger model with Mistral 3 text encoder).
    ///
    /// Key differences from Flux 1: shared modulation weights, no biases,
    /// 6× MLP ratio, 15360-dim text input (Mistral 3 projected), SigLIP vision.
    pub fn dev2() -> Self {
        Self {
            hidden_size: 6144,
            num_heads: 48,
            head_dim: 128,
            num_double_layers: 8,
            num_single_layers: 48,
            context_dim: 15360,
            mlp_ratio: 6.0,
            theta: 10000.0,
            shared_modulation: true,
            use_bias: false,
        }
    }

    /// Flux 1 Full configuration.
    pub fn full() -> Self {
        Self {
            hidden_size: 6144,
            num_heads: 48,
            head_dim: 128,
            num_double_layers: 8,
            num_single_layers: 48,
            context_dim: 4096,
            mlp_ratio: 4.0,
            theta: 10000.0,
            shared_modulation: false,
            use_bias: true,
        }
    }
}

/// Flux Transformer model wrapper.
///
/// The Flux architecture uses a two-phase transformer:
/// 1. Double-stream blocks: separate QKV for image and text, shared attention
/// 2. Single-stream blocks: concatenated img+txt with fused QKV+MLP
///
/// Both phases use AdaLN modulation from timestep embeddings and QK-norm
/// to prevent attention collapse at scale.
pub struct FluxTransformer {
    model: Arc<Model>,
    config: FluxConfig,
}

impl FluxTransformer {
    /// Create a new Flux transformer wrapper.
    pub fn new(model: Arc<Model>, config: FluxConfig) -> Self {
        Self { model, config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &FluxConfig {
        &self.config
    }
}

// ============================================================================
// AdaLN Modulation (Adaptive Layer Normalization)
// ============================================================================

/// Adaptive Layer Normalization for Flux transformer blocks.
///
/// Computes shift, scale, and gate parameters from a timestep embedding,
/// then applies them to layer-normed activations. This replaces standard
/// LayerNorm with a conditioning-dependent normalization.
pub struct AdaLNModulation;

impl AdaLNModulation {
    /// Compute 6 modulation vectors from a timestep embedding.
    ///
    /// The embedding is passed through SiLU activation, then projected to
    /// 6 × hidden_size and chunked into: [shift1, scale1, gate1, shift2, scale2, gate2].
    /// The first triple modulates attention; the second triple modulates the MLP.
    pub fn compute(emb: &[f32], hidden_size: usize) -> [Vec<f32>; 6] {
        // SiLU activation: x * sigmoid(x)
        let activated: Vec<f32> = emb.iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();

        // Project to 6 * hidden_size
        // In practice this would use learned weights; here we compute the
        // deterministic projection structure for testing.
        let total = 6 * hidden_size;
        let mut projected = vec![0.0f32; total];

        // Simple linear projection approximation using the embedding values
        for i in 0..total {
            let emb_idx = i % activated.len();
            projected[i] = activated[emb_idx];
        }

        // Chunk into 6 equal parts
        let mut result: [Vec<f32>; 6] = Default::default();
        for (chunk_idx, chunk) in projected.chunks(hidden_size).enumerate() {
            if chunk_idx < 6 {
                result[chunk_idx] = chunk.to_vec();
            }
        }

        result
    }

    /// Apply modulation: out = (1 + scale) * x + shift
    pub fn apply(x: &[f32], scale: &[f32], shift: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(scale.iter())
            .zip(shift.iter())
            .map(|((&xi, &si), &sh)| (1.0 + si) * xi + sh)
            .collect()
    }

    /// Apply gated residual: out = x + gate * residual
    pub fn gate(x: &[f32], residual: &[f32], gate: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(residual.iter())
            .zip(gate.iter())
            .map(|((&xi, &ri), &gi)| xi + gi * ri)
            .collect()
    }
}

// ============================================================================
// QK-Norm
// ============================================================================

/// Apply RMSNorm independently to each attention head of Q and K.
///
/// This prevents attention weight explosion at large hidden sizes (Flux uses
/// head_dim=128 with up to 48 heads). Each head is normalized to unit RMS.
pub fn qk_norm(q: &mut [f32], k: &mut [f32], head_dim: usize) {
    for head in q.chunks_mut(head_dim) {
        rms_norm_inplace(head, 1e-6);
    }
    for head in k.chunks_mut(head_dim) {
        rms_norm_inplace(head, 1e-6);
    }
}

/// In-place RMSNorm: normalize vector to unit RMS.
fn rms_norm_inplace(x: &mut [f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for v in x.iter_mut() {
        *v /= rms;
    }
}

// ============================================================================
// 2×2 Patchification
// ============================================================================

/// Convert image features into 2×2 patch tokens.
///
/// Rearranges `[channels, height, width]` → `[height/2 * width/2, channels * 4]`
/// by grouping adjacent 2×2 spatial regions into single tokens with 4× channels.
///
/// Requires: height and width are even.
pub fn patchify(img: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    let ph = height / 2;
    let pw = width / 2;
    let patch_channels = channels * 4;
    let mut patches = vec![0.0f32; ph * pw * patch_channels];

    for py in 0..ph {
        for px in 0..pw {
            let patch_idx = py * pw + px;
            for c in 0..channels {
                // Four pixels in the 2×2 patch
                let y0 = py * 2;
                let y1 = y0 + 1;
                let x0 = px * 2;
                let x1 = x0 + 1;

                // Source: [C, H, W] layout
                let s00 = img[c * height * width + y0 * width + x0];
                let s01 = img[c * height * width + y0 * width + x1];
                let s10 = img[c * height * width + y1 * width + x0];
                let s11 = img[c * height * width + y1 * width + x1];

                // Dest: [ph*pw, C*4] layout
                let base = patch_idx * patch_channels + c * 4;
                patches[base] = s00;
                patches[base + 1] = s01;
                patches[base + 2] = s10;
                patches[base + 3] = s11;
            }
        }
    }

    patches
}

/// Reverse patchification: convert patch tokens back to image features.
///
/// Rearranges `[height/2 * width/2, channels * 4]` → `[channels, height, width]`.
pub fn unpatchify(patches: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    let ph = height / 2;
    let pw = width / 2;
    let patch_channels = channels * 4;
    let mut img = vec![0.0f32; channels * height * width];

    for py in 0..ph {
        for px in 0..pw {
            let patch_idx = py * pw + px;
            for c in 0..channels {
                let base = patch_idx * patch_channels + c * 4;
                let s00 = patches[base];
                let s01 = patches[base + 1];
                let s10 = patches[base + 2];
                let s11 = patches[base + 3];

                let y0 = py * 2;
                let y1 = y0 + 1;
                let x0 = px * 2;
                let x1 = x0 + 1;

                img[c * height * width + y0 * width + x0] = s00;
                img[c * height * width + y0 * width + x1] = s01;
                img[c * height * width + y1 * width + x0] = s10;
                img[c * height * width + y1 * width + x1] = s11;
            }
        }
    }

    img
}

// ============================================================================
// Flux GPU Transformer — Metal Implementation
// ============================================================================

/// Compiled kernel pipelines for Flux operations.
#[cfg(feature = "metal")]
struct FluxKernels {
    linear: Arc<crate::hal::metal::ComputePipeline>,
    silu: Arc<crate::hal::metal::ComputePipeline>,
    add: Arc<crate::hal::metal::ComputePipeline>,
    gelu: Arc<crate::hal::metal::ComputePipeline>,
    layer_norm: Arc<crate::hal::metal::ComputePipeline>,
    rms_norm: Arc<crate::hal::metal::ComputePipeline>,
    adaln_modulate: Arc<crate::hal::metal::ComputePipeline>,
    adaln_gate: Arc<crate::hal::metal::ComputePipeline>,
    patchify: Arc<crate::hal::metal::ComputePipeline>,
    unpatchify: Arc<crate::hal::metal::ComputePipeline>,
    batched_linear: Arc<crate::hal::metal::ComputePipeline>,
    batched_matmul_nn: Arc<crate::hal::metal::ComputePipeline>,
    row_softmax: Arc<crate::hal::metal::ComputePipeline>,
    transpose_shd_hsd: Arc<crate::hal::metal::ComputePipeline>,
    transpose_hsd_shd: Arc<crate::hal::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
impl FluxKernels {
    fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ADALN, "adaln_gate_f16")?,
            patchify: compute.compile_pipeline("patchify", sources::PATCHIFY, "patchify_f16")?,
            unpatchify: compute.compile_pipeline("unpatchify", sources::PATCHIFY, "unpatchify_f16")?,
            batched_linear: compute.compile_pipeline("batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax: compute.compile_pipeline("row_softmax", sources::LINEAR, "row_softmax_scale_f16")?,
            transpose_shd_hsd: compute.compile_pipeline("transpose_shd_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_shd: compute.compile_pipeline("transpose_hsd_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
        })
    }
}

/// Flux GPU transformer — full forward pass on Metal.
#[cfg(feature = "metal")]
pub struct FluxGpuTransformer {
    model: Arc<Model>,
    config: FluxConfig,
    kernels: FluxKernels,
}

#[cfg(feature = "metal")]
impl FluxGpuTransformer {
    /// Create a new Flux GPU transformer.
    pub fn new(model: Arc<Model>, config: FluxConfig, compute: &Arc<MetalCompute>) -> Result<Self> {
        let kernels = FluxKernels::new(compute)?;
        Ok(Self { model, config, kernels })
    }

    /// Full forward pass.
    ///
    /// `latents`: [1, 16, H, W] noisy latent (Flux uses 16-channel VAE)
    /// `context`: [1, txt_seq, 4096] T5-XXL text embeddings
    /// `clip_pooled`: [1, 768] CLIP-L pooled embedding
    /// `timestep`: scalar (0.0 → 1.0)
    /// `guidance`: guidance scale (for guidance-distilled models)
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        clip_pooled: &Tensor,
        timestep: f32,
        guidance: f32,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let (_batch, channels, height, width) = latents.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("latents must be [B, C, H, W]"))?;
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let mlp_dim = (hidden_size as f32 * self.config.mlp_ratio) as usize;
        let device = latents.device();

        // 1. Timestep + guidance + vector embeddings → vec (conditioning vector)
        let temb = self.timestep_embedding(timestep, compute)?;
        let guidance_emb = self.timestep_embedding(guidance * 1000.0, compute)?;
        let cb0 = compute.new_command_buffer();
        let time_proj = self.gpu_mlp(&temb, "time_in", 256, hidden_size, compute, cb0.as_ref())?;
        let guid_proj = self.gpu_mlp(&guidance_emb, "guidance_in", 256, hidden_size, compute, cb0.as_ref())?;
        let vec_proj = self.gpu_mlp(clip_pooled, "vector_in", 768, hidden_size, compute, cb0.as_ref())?;
        cb0.commit();
        cb0.wait_until_completed();

        // vec = time_proj + guid_proj + vec_proj
        let cb1 = compute.new_command_buffer();
        let vec_partial = self.gpu_add(&time_proj, &guid_proj, compute, cb1.as_ref())?;
        let vec = self.gpu_add(&vec_partial, &vec_proj, compute, cb1.as_ref())?;
        cb1.commit();
        cb1.wait_until_completed();

        // 2. Patchify + project image tokens
        let num_patches = (height / 2) * (width / 2);
        let patch_dim = channels * 4; // 16 * 4 = 64

        let cb2 = compute.new_command_buffer();
        let patches = self.gpu_patchify(latents, channels, height, width, compute, cb2.as_ref())?;
        let mut img = self.gpu_linear_auto(&patches, "img_in", patch_dim, hidden_size, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        let img_seq_len = num_patches;

        // 3. Project text tokens
        let txt_seq_len = context.shape().dim(1).unwrap_or(256);
        let ctx_flat = context.reshape(Shape::from([txt_seq_len, self.config.context_dim]))?;
        let cb3 = compute.new_command_buffer();
        let mut txt = self.gpu_linear_auto(&ctx_flat, "txt_in", self.config.context_dim, hidden_size, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        // TODO: 2D RoPE positional encoding (for now positions encoded in attention)

        // 4. Double blocks (19): dual-stream with joint attention
        //    Streaming NVMe: prefetch next block, evict current after use
        self.model.prefetch_prefix("double_blocks.0.");
        for i in 0..self.config.num_double_layers {
            if i + 1 < self.config.num_double_layers {
                self.model.prefetch_prefix(&format!("double_blocks.{}.", i + 1));
            } else {
                self.model.prefetch_prefix("single_blocks.0.");
            }
            let (new_img, new_txt) = self.double_block(
                &img, &txt, &vec, i,
                img_seq_len, txt_seq_len, hidden_size, num_heads, head_dim, mlp_dim,
                compute,
            )?;
            self.model.evict_prefix(&format!("double_blocks.{}.", i));
            img = new_img;
            txt = new_txt;
        }

        // 5. Concatenate img+txt for single stream
        let mut hidden = Tensor::cat(&[txt, img], 0)?;
        let total_seq = txt_seq_len + img_seq_len;

        // 6. Single blocks (38): fused stream with parallel QKV+MLP
        //    Streaming NVMe: prefetch next block, evict current after use
        for i in 0..self.config.num_single_layers {
            if i + 1 < self.config.num_single_layers {
                self.model.prefetch_prefix(&format!("single_blocks.{}.", i + 1));
            } else {
                self.model.prefetch_prefix("final_layer.");
            }
            hidden = self.single_block(
                &hidden, &vec, i,
                total_seq, hidden_size, num_heads, head_dim, mlp_dim,
                compute,
            )?;
            self.model.evict_prefix(&format!("single_blocks.{}.", i));
        }

        // 7. Extract image tokens (last img_seq_len tokens)
        hidden = hidden.slice(0, txt_seq_len, total_seq)?;

        // 8. Final layer: AdaLN modulation → linear → unpatchify
        let (final_scale, final_shift, _final_gate) = self.adaln_3params(
            &vec, "final_layer.adaLN_modulation.1", hidden_size, compute,
        )?;

        let cb_final = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(&hidden, hidden_size, compute, cb_final.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &final_scale, &final_shift, hidden_size, compute, cb_final.as_ref())?;
        let output_patches = self.gpu_linear_auto(&modulated, "final_layer.linear", hidden_size, patch_dim, compute, cb_final.as_ref())?;
        let output = self.gpu_unpatchify(&output_patches, channels, height, width, compute, cb_final.as_ref())?;
        cb_final.commit();
        cb_final.wait_until_completed();

        Ok(output.reshape(Shape::from([1, channels, height, width]))?)
    }

    // ========================================================================
    // Double block (dual-stream)
    // ========================================================================

    fn double_block(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
        block_idx: usize,
        img_seq: usize,
        txt_seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor)> {
        let prefix = format!("double_blocks.{}", block_idx);

        // AdaLN modulation (6 params each for img and txt)
        // Flux 2: shared modulation weights across all double blocks
        let img_mod_name = if self.config.shared_modulation {
            "double_stream_modulation_img.lin".to_string()
        } else {
            format!("{}.img_mod.lin", prefix)
        };
        let txt_mod_name = if self.config.shared_modulation {
            "double_stream_modulation_txt.lin".to_string()
        } else {
            format!("{}.txt_mod.lin", prefix)
        };
        let (img_shift_attn, img_scale_attn, img_gate_attn,
             img_shift_mlp, img_scale_mlp, img_gate_mlp) =
            self.adaln_6params(vec, &img_mod_name, hidden, compute)?;
        let (txt_shift_attn, txt_scale_attn, txt_gate_attn,
             txt_shift_mlp, txt_scale_mlp, txt_gate_mlp) =
            self.adaln_6params(vec, &txt_mod_name, hidden, compute)?;

        // LayerNorm + modulate
        let cb = compute.new_command_buffer();
        let img_normed = self.gpu_layer_norm(img, hidden, compute, cb.as_ref())?;
        let img_mod = self.gpu_adaln_modulate(&img_normed, &img_scale_attn, &img_shift_attn, hidden, compute, cb.as_ref())?;
        let txt_normed = self.gpu_layer_norm(txt, hidden, compute, cb.as_ref())?;
        let txt_mod = self.gpu_adaln_modulate(&txt_normed, &txt_scale_attn, &txt_shift_attn, hidden, compute, cb.as_ref())?;

        // Fused QKV projection: [seq, hidden] → [seq, 3*hidden]
        let img_qkv = self.gpu_linear_auto(&img_mod, &format!("{}.img_attn.qkv", prefix), hidden, 3 * hidden, compute, cb.as_ref())?;
        let txt_qkv = self.gpu_linear_auto(&txt_mod, &format!("{}.txt_attn.qkv", prefix), hidden, 3 * hidden, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Split QKV and apply QK-norm
        let (img_q, img_k, img_v) = self.split_qkv_with_norm(&img_qkv, img_seq, hidden, head_dim, &format!("{}.img_attn.norm", prefix), compute)?;
        let (txt_q, txt_k, txt_v) = self.split_qkv_with_norm(&txt_qkv, txt_seq, hidden, head_dim, &format!("{}.txt_attn.norm", prefix), compute)?;

        // Concatenate K/V for joint attention
        let joint_k = Tensor::cat(&[txt_k, img_k], 0)?;
        let joint_v = Tensor::cat(&[txt_v, img_v], 0)?;
        let total_seq = img_seq + txt_seq;

        // Attention for img queries
        let img_attn = self.batched_attention(&img_q, &joint_k, &joint_v, img_seq, total_seq, num_heads, head_dim, compute)?;
        // Attention for txt queries
        let txt_attn = self.batched_attention(&txt_q, &joint_k, &joint_v, txt_seq, total_seq, num_heads, head_dim, compute)?;

        // Output projection + gated residual
        let cb2 = compute.new_command_buffer();
        let img_proj = self.gpu_linear_auto(&img_attn, &format!("{}.img_attn.proj", prefix), hidden, hidden, compute, cb2.as_ref())?;
        let txt_proj = self.gpu_linear_auto(&txt_attn, &format!("{}.txt_attn.proj", prefix), hidden, hidden, compute, cb2.as_ref())?;
        let img_after_attn = self.gpu_adaln_gate(img, &img_proj, &img_gate_attn, hidden, compute, cb2.as_ref())?;
        let txt_after_attn = self.gpu_adaln_gate(txt, &txt_proj, &txt_gate_attn, hidden, compute, cb2.as_ref())?;

        // MLP with modulation
        let img_mlp_normed = self.gpu_layer_norm(&img_after_attn, hidden, compute, cb2.as_ref())?;
        let img_mlp_mod = self.gpu_adaln_modulate(&img_mlp_normed, &img_scale_mlp, &img_shift_mlp, hidden, compute, cb2.as_ref())?;
        let txt_mlp_normed = self.gpu_layer_norm(&txt_after_attn, hidden, compute, cb2.as_ref())?;
        let txt_mlp_mod = self.gpu_adaln_modulate(&txt_mlp_normed, &txt_scale_mlp, &txt_shift_mlp, hidden, compute, cb2.as_ref())?;

        // MLP: linear → GELU → linear
        let img_mlp_h = self.gpu_linear_auto(&img_mlp_mod, &format!("{}.img_mlp.0", prefix), hidden, mlp_dim, compute, cb2.as_ref())?;
        let img_mlp_act = self.gpu_gelu(&img_mlp_h, compute, cb2.as_ref())?;
        let img_mlp_out = self.gpu_linear_auto(&img_mlp_act, &format!("{}.img_mlp.2", prefix), mlp_dim, hidden, compute, cb2.as_ref())?;
        let txt_mlp_h = self.gpu_linear_auto(&txt_mlp_mod, &format!("{}.txt_mlp.0", prefix), hidden, mlp_dim, compute, cb2.as_ref())?;
        let txt_mlp_act = self.gpu_gelu(&txt_mlp_h, compute, cb2.as_ref())?;
        let txt_mlp_out = self.gpu_linear_auto(&txt_mlp_act, &format!("{}.txt_mlp.2", prefix), mlp_dim, hidden, compute, cb2.as_ref())?;

        // Gated residual for MLP
        let img_out = self.gpu_adaln_gate(&img_after_attn, &img_mlp_out, &img_gate_mlp, hidden, compute, cb2.as_ref())?;
        let txt_out = self.gpu_adaln_gate(&txt_after_attn, &txt_mlp_out, &txt_gate_mlp, hidden, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        Ok((img_out, txt_out))
    }

    // ========================================================================
    // Single block (fused stream)
    // ========================================================================

    fn single_block(
        &self,
        hidden: &Tensor,
        vec: &Tensor,
        block_idx: usize,
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let prefix = format!("single_blocks.{}", block_idx);

        // 3-param modulation: shift, scale, gate
        // Flux 2: shared modulation weights across all single blocks
        let mod_name = if self.config.shared_modulation {
            "single_stream_modulation.lin".to_string()
        } else {
            format!("{}.modulation.lin", prefix)
        };
        let (shift, scale, gate) = self.adaln_3params(
            vec, &mod_name, hidden_size, compute,
        )?;

        // LayerNorm + modulate
        let cb = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(hidden, hidden_size, compute, cb.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &scale, &shift, hidden_size, compute, cb.as_ref())?;

        // Fused linear1: [seq, hidden] → [seq, 3*hidden + mlp_dim]
        // Splits into Q, K, V (each hidden_size) and mlp_up (mlp_dim)
        let fused_dim = 3 * hidden_size + mlp_dim;
        let fused = self.gpu_linear_auto(&modulated, &format!("{}.linear1", prefix), hidden_size, fused_dim, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Split fused output: [Q, K, V, mlp_input]
        let fused_f32 = fused.to_f32_vec()?;
        let q_data: Vec<half::f16> = fused_f32[..seq_len * hidden_size].iter().map(|&v| half::f16::from_f32(v)).collect();
        let k_data: Vec<half::f16> = fused_f32[seq_len * hidden_size..seq_len * 2 * hidden_size].iter().map(|&v| half::f16::from_f32(v)).collect();
        let v_data: Vec<half::f16> = fused_f32[seq_len * 2 * hidden_size..seq_len * 3 * hidden_size].iter().map(|&v| half::f16::from_f32(v)).collect();
        let mlp_data: Vec<half::f16> = fused_f32[seq_len * 3 * hidden_size..].iter().map(|&v| half::f16::from_f32(v)).collect();

        let device = hidden.device();
        let q = Tensor::from_slice(&q_data, Shape::from([seq_len, hidden_size]), DType::F16, device)?;
        let k = Tensor::from_slice(&k_data, Shape::from([seq_len, hidden_size]), DType::F16, device)?;
        let v = Tensor::from_slice(&v_data, Shape::from([seq_len, hidden_size]), DType::F16, device)?;
        let mlp_up = Tensor::from_slice(&mlp_data, Shape::from([seq_len, mlp_dim]), DType::F16, device)?;

        // QK-norm
        let (q_normed, k_normed) = self.apply_qk_norm(&q, &k, seq_len, num_heads, head_dim, &format!("{}.norm", prefix), compute)?;

        // Self-attention
        let attn_out = self.batched_attention(&q_normed, &k_normed, &v, seq_len, seq_len, num_heads, head_dim, compute)?;

        // GELU on mlp_up
        let cb2 = compute.new_command_buffer();
        let mlp_act = self.gpu_gelu(&mlp_up, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // Concat attn + mlp: [seq, hidden + mlp_dim]
        let concat = Tensor::cat(&[attn_out, mlp_act], 1)?;
        let concat_dim = hidden_size + mlp_dim; // 3072 + 12288 = 15360

        // linear2: [seq, concat_dim] → [seq, hidden]
        let cb3 = compute.new_command_buffer();
        let residual = self.gpu_linear_auto(&concat, &format!("{}.linear2", prefix), concat_dim, hidden_size, compute, cb3.as_ref())?;

        // Gated residual
        let output = self.gpu_adaln_gate(hidden, &residual, &gate, hidden_size, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        Ok(output)
    }

    // ========================================================================
    // Helper: QKV split with QK-norm
    // ========================================================================

    fn split_qkv_with_norm(
        &self,
        qkv: &Tensor,
        seq_len: usize,
        hidden: usize,
        head_dim: usize,
        norm_prefix: &str,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv_f32 = qkv.to_f32_vec()?;
        let device = qkv.device();

        let q_f16: Vec<half::f16> = qkv_f32[..seq_len * hidden].iter().map(|&v| half::f16::from_f32(v)).collect();
        let k_f16: Vec<half::f16> = qkv_f32[seq_len * hidden..seq_len * 2 * hidden].iter().map(|&v| half::f16::from_f32(v)).collect();
        let v_f16: Vec<half::f16> = qkv_f32[seq_len * 2 * hidden..].iter().map(|&v| half::f16::from_f32(v)).collect();

        let q = Tensor::from_slice(&q_f16, Shape::from([seq_len, hidden]), DType::F16, device)?;
        let k = Tensor::from_slice(&k_f16, Shape::from([seq_len, hidden]), DType::F16, device)?;
        let v = Tensor::from_slice(&v_f16, Shape::from([seq_len, hidden]), DType::F16, device)?;

        let num_heads = hidden / head_dim;
        let (q_normed, k_normed) = self.apply_qk_norm(&q, &k, seq_len, num_heads, head_dim, norm_prefix, compute)?;

        Ok((q_normed, k_normed, v))
    }

    /// Apply QK-norm: per-head RMSNorm with learned scale.
    fn apply_qk_norm(
        &self,
        q: &Tensor,
        k: &Tensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        norm_prefix: &str,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor)> {
        let q_scale = self.w(&format!("{}.query_norm.scale", norm_prefix))?;
        let k_scale = self.w(&format!("{}.key_norm.scale", norm_prefix))?;

        // Apply per-head RMSNorm with learned scale on CPU (head_dim=128 is small)
        let mut q_f32 = q.to_f32_vec()?;
        let mut k_f32 = k.to_f32_vec()?;
        let q_scale_f32 = q_scale.to_f32_vec()?;
        let k_scale_f32 = k_scale.to_f32_vec()?;

        for token in 0..seq_len {
            for head in 0..num_heads {
                let offset = token * num_heads * head_dim + head * head_dim;
                let slice = &mut q_f32[offset..offset + head_dim];
                rms_norm_inplace(slice, 1e-6);
                for d in 0..head_dim {
                    slice[d] *= q_scale_f32[d];
                }
            }
        }
        for token in 0..seq_len {
            for head in 0..num_heads {
                let offset = token * num_heads * head_dim + head * head_dim;
                let slice = &mut k_f32[offset..offset + head_dim];
                rms_norm_inplace(slice, 1e-6);
                for d in 0..head_dim {
                    slice[d] *= k_scale_f32[d];
                }
            }
        }

        let device = q.device();
        let q_f16: Vec<half::f16> = q_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let k_f16: Vec<half::f16> = k_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let q_out = Tensor::from_slice(&q_f16, q.shape().clone(), DType::F16, device)?;
        let k_out = Tensor::from_slice(&k_f16, k.shape().clone(), DType::F16, device)?;

        Ok((q_out, k_out))
    }

    // ========================================================================
    // Helper: Modulation parameters
    // ========================================================================

    fn adaln_6params(
        &self,
        vec: &Tensor,
        weight_name: &str,
        hidden_size: usize,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let cb = compute.new_command_buffer();
        let activated = self.gpu_silu(vec, compute, cb.as_ref())?;
        let params = self.gpu_linear_auto(&activated, weight_name, hidden_size, hidden_size * 6, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let data = params.to_f32_vec()?;
        let device = vec.device();
        let shape = Shape::from([hidden_size]);
        let to_f16 = |s: usize, e: usize| -> Vec<half::f16> {
            data[s..e].iter().map(|&v| half::f16::from_f32(v)).collect()
        };

        Ok((
            Tensor::from_slice(&to_f16(0, hidden_size), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden_size, hidden_size * 2), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden_size * 2, hidden_size * 3), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden_size * 3, hidden_size * 4), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden_size * 4, hidden_size * 5), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden_size * 5, hidden_size * 6), shape.clone(), DType::F16, device)?,
        ))
    }

    fn adaln_3params(
        &self,
        vec: &Tensor,
        weight_name: &str,
        hidden_size: usize,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cb = compute.new_command_buffer();
        let activated = self.gpu_silu(vec, compute, cb.as_ref())?;
        let params = self.gpu_linear_auto(&activated, weight_name, hidden_size, hidden_size * 3, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let data = params.to_f32_vec()?;
        let device = vec.device();
        let shape = Shape::from([hidden_size]);
        let to_f16 = |s: usize, e: usize| -> Vec<half::f16> {
            data[s..e].iter().map(|&v| half::f16::from_f32(v)).collect()
        };

        Ok((
            Tensor::from_slice(&to_f16(0, hidden_size), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden_size, hidden_size * 2), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden_size * 2, hidden_size * 3), shape.clone(), DType::F16, device)?,
        ))
    }

    // ========================================================================
    // Shared GPU operation helpers
    // ========================================================================

    fn timestep_embedding(&self, timestep: f32, compute: &MetalCompute) -> Result<Tensor> {
        let device = compute.device().info().id;
        crate::inference::architecture::dit::DiTOps::timestep_embedding(timestep, 256, device)
    }

    fn gpu_mlp(
        &self,
        x: &Tensor,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        compute: &MetalCompute,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let h = self.gpu_linear_auto(x, &format!("{}.in_layer", prefix), in_dim, out_dim, compute, cb)?;
        let h = self.gpu_silu_cb(&h, compute, cb)?;
        self.gpu_linear_auto(&h, &format!("{}.out_layer", prefix), out_dim, out_dim, compute, cb)
    }

    fn batched_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_seq_len: usize,
        kv_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_shd = q.reshape(Shape::from([q_seq_len, num_heads, head_dim]))?;
        let k_shd = k.reshape(Shape::from([kv_seq_len, num_heads, head_dim]))?;
        let v_shd = v.reshape(Shape::from([kv_seq_len, num_heads, head_dim]))?;

        let cb = compute.new_command_buffer();
        let q_hsd = self.gpu_transpose_shd_hsd(&q_shd, q_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        let k_hsd = self.gpu_transpose_shd_hsd(&k_shd, kv_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        let v_hsd = self.gpu_transpose_shd_hsd(&v_shd, kv_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Q @ K^T
        let cb2 = compute.new_command_buffer();
        let scores = self.gpu_batched_linear_raw(&q_hsd, &k_hsd, num_heads, q_seq_len, head_dim, kv_seq_len, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // Softmax
        let cb3 = compute.new_command_buffer();
        let attn_weights = self.gpu_row_softmax(&scores, num_heads * q_seq_len, kv_seq_len, scale, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        // Weights @ V
        let cb4 = compute.new_command_buffer();
        let attn_out_hsd = self.gpu_batched_matmul_nn(&attn_weights, &v_hsd, num_heads, q_seq_len, kv_seq_len, head_dim, compute, cb4.as_ref())?;
        let attn_out_shd = self.gpu_transpose_hsd_shd(&attn_out_hsd, q_seq_len, num_heads, head_dim, compute, cb4.as_ref())?;
        cb4.commit();
        cb4.wait_until_completed();

        Ok(attn_out_shd.reshape(Shape::from([q_seq_len, num_heads * head_dim]))?)
    }

    fn w(&self, name: &str) -> Result<&crate::hal::metal::LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("Flux weight not found: {}", name)))
    }

    // ========================================================================
    // Low-level GPU dispatch helpers
    // ========================================================================

    fn gpu_linear_biased(
        &self, x: &Tensor, prefix: &str, in_feat: usize, out_feat: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        self.gpu_linear_biased_cb(x, prefix, in_feat, out_feat, compute, cb)
    }

    fn gpu_linear_biased_cb(
        &self, x: &Tensor, prefix: &str, in_feat: usize, out_feat: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let w = self.w(&format!("{}.weight", prefix))?;
        let b = self.w(&format!("{}.bias", prefix))?;
        let seq_len = x.shape().numel() / in_feat;
        let output = Tensor::empty(Shape::from([seq_len, out_feat]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_feat as u32;
        let k = in_feat as u32;
        let has_bias: u32 = 1;
        // linear_f16: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        compute.dispatch_async(cb, &self.kernels.linear,
            ((out_feat + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w.buffer()), 0);
                enc.set_buffer(2, Some(b.buffer()), 0);
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &m as *const u32 as *const _);
                enc.set_bytes(5, 4, &n as *const u32 as *const _);
                enc.set_bytes(6, 4, &k as *const u32 as *const _);
                enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        Ok(output)
    }

    /// Linear projection without bias: Y = X @ W^T
    fn gpu_linear_cb(
        &self, x: &Tensor, prefix: &str, in_feat: usize, out_feat: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let w = self.w(&format!("{}.weight", prefix))?;
        let seq_len = x.shape().numel() / in_feat;
        let output = Tensor::empty(Shape::from([seq_len, out_feat]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_feat as u32;
        let k = in_feat as u32;
        let has_bias: u32 = 0;
        // linear_f16: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        compute.dispatch_async(cb, &self.kernels.linear,
            ((out_feat + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w.buffer()), 0);
                enc.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias (has_bias=0)
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &m as *const u32 as *const _);
                enc.set_bytes(5, 4, &n as *const u32 as *const _);
                enc.set_bytes(6, 4, &k as *const u32 as *const _);
                enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        Ok(output)
    }

    /// Auto-dispatch linear: uses biased or no-bias based on config.
    fn gpu_linear_auto(
        &self, x: &Tensor, prefix: &str, in_feat: usize, out_feat: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        if self.config.use_bias {
            self.gpu_linear_biased_cb(x, prefix, in_feat, out_feat, compute, cb)
        } else {
            self.gpu_linear_cb(x, prefix, in_feat, out_feat, compute, cb)
        }
    }

    fn gpu_silu(&self, x: &Tensor, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c = count as u32;
        compute.dispatch_async(cb, &self.kernels.silu,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_silu_cb(&self, x: &Tensor, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        self.gpu_silu(x, compute, cb)
    }

    fn gpu_gelu(&self, x: &Tensor, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c = count as u32;
        compute.dispatch_async(cb, &self.kernels.gelu,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_add(&self, a: &Tensor, b: &Tensor, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = a.shape().numel();
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c = count as u32;
        compute.dispatch_async(cb, &self.kernels.add,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(a_buf.as_ref()), 0);
                enc.set_buffer(1, Some(b_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_layer_norm(&self, x: &Tensor, hidden_size: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let seq_len = x.shape().numel() / hidden_size;
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_hidden = hidden_size as u32;
        let eps: f32 = 1e-6;
        compute.dispatch_async(cb, &self.kernels.layer_norm,
            (seq_len, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_hidden as *const u32 as *const _);
                enc.set_bytes(3, 4, &eps as *const f32 as *const _);
            });
        Ok(output)
    }

    fn gpu_adaln_modulate(&self, x: &Tensor, scale: &Tensor, shift: &Tensor, hidden: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let s_buf = borrow_tensor(scale)?;
        let sh_buf = borrow_tensor(shift)?;
        let o_buf = borrow_tensor(&output)?;
        let c_h = hidden as u32;
        let c_n = count as u32;
        compute.dispatch_async(cb, &self.kernels.adaln_modulate,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(s_buf.as_ref()), 0);
                enc.set_buffer(2, Some(sh_buf.as_ref()), 0);
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_n as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_adaln_gate(&self, x: &Tensor, residual: &Tensor, gate: &Tensor, hidden: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let r_buf = borrow_tensor(residual)?;
        let g_buf = borrow_tensor(gate)?;
        let o_buf = borrow_tensor(&output)?;
        let c_h = hidden as u32;
        let c_n = count as u32;
        compute.dispatch_async(cb, &self.kernels.adaln_gate,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(r_buf.as_ref()), 0);
                enc.set_buffer(2, Some(g_buf.as_ref()), 0);
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_n as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_patchify(&self, input: &Tensor, channels: usize, height: usize, width: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let num_patches = (height / 2) * (width / 2);
        let output = Tensor::empty(Shape::from([num_patches, channels * 4]), DType::F16, input.device())?;
        let i_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_ch = channels as u32;
        let c_h = height as u32;
        let c_w = width as u32;
        compute.dispatch_async(cb, &self.kernels.patchify,
            (num_patches, channels, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(i_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_ch as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_w as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_unpatchify(&self, input: &Tensor, channels: usize, height: usize, width: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let num_patches = (height / 2) * (width / 2);
        let output = Tensor::empty(Shape::from([channels, height, width]), DType::F16, input.device())?;
        let i_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_ch = channels as u32;
        let c_h = height as u32;
        let c_w = width as u32;
        compute.dispatch_async(cb, &self.kernels.unpatchify,
            (num_patches, channels, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(i_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_ch as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_w as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_shd_hsd(&self, x: &Tensor, seq: usize, heads: usize, dim: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([heads, seq, dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq as u32; let c_h = heads as u32; let c_d = dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_shd_hsd,
            (heads, seq, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_s as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_hsd_shd(&self, x: &Tensor, seq: usize, heads: usize, dim: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([seq, heads, dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq as u32; let c_h = heads as u32; let c_d = dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_hsd_shd,
            (heads, seq, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_s as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_batched_linear_raw(&self, x: &Tensor, w: &Tensor, batch: usize, m: usize, k: usize, n: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let w_buf = borrow_tensor(w)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32; let c_n = n as u32; let c_k = k as u32; let c_b = batch as u32;
        compute.dispatch_async(cb, &self.kernels.batched_linear,
            ((n + 15) / 16, (m + 15) / 16, batch), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_m as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_n as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_k as *const u32 as *const _);
                enc.set_bytes(6, 4, &c_b as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_batched_matmul_nn(&self, a: &Tensor, b: &Tensor, batch: usize, m: usize, k: usize, n: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32; let c_n = n as u32; let c_k = k as u32; let c_b = batch as u32;
        compute.dispatch_async(cb, &self.kernels.batched_matmul_nn,
            ((n + 15) / 16, (m + 15) / 16, batch), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(a_buf.as_ref()), 0);
                enc.set_buffer(1, Some(b_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_m as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_n as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_k as *const u32 as *const _);
                enc.set_bytes(6, 4, &c_b as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_row_softmax(&self, x: &Tensor, num_rows: usize, row_len: usize, scale: f32, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_cols = row_len as u32;
        compute.dispatch_async(cb, &self.kernels.row_softmax,
            (num_rows, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_cols as *const u32 as *const _);
                enc.set_bytes(3, 4, &scale as *const f32 as *const _);
            });
        Ok(output)
    }
}

#[cfg(feature = "metal")]
fn borrow_tensor(tensor: &Tensor) -> Result<BorrowedMetalBuffer> {
    let ptr = tensor.device_ptr()
        .ok_or_else(|| crate::core::Error::internal("tensor not on device"))?;
    Ok(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
}
