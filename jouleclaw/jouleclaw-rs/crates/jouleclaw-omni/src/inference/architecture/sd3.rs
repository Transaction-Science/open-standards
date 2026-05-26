//! SD 3.5 Large MMDiT (Multi-Modal Diffusion Transformer) architecture.
//!
//! Implements the Stability AI SD 3.5 Large model with:
//! - MMDiT blocks: joint bidirectional attention (image <-> text attend simultaneously)
//! - Triple text encoder: CLIP-L (768d) + CLIP-G (1280d) + T5-XXL (4096d)
//! - AdaLN-Zero modulation from timestep embeddings
//! - QK-norm (RMSNorm per head) to prevent attention collapse
//! - 2x2 patchification for spatial token reduction
//! - Flow matching scheduler
//!
//! Key difference from Flux: MMDiT uses truly simultaneous bidirectional attention
//! where both image and text streams attend to the joint sequence in the same block,
//! rather than Flux's sequential double-stream (separate Q projections, shared K/V).
//! SD3 also uses 3 text encoders (CLIP-L + CLIP-G + T5-XXL) vs Flux's 2 (CLIP-L + T5).

use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::core::Result;
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::MetalCompute;
#[cfg(feature = "metal")]
use crate::hal::metal::BorrowedMetalBuffer;
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

/// SD 3.5 Large MMDiT configuration.
#[derive(Debug, Clone)]
pub struct Sd3Config {
    /// Hidden dimension of the transformer (4096 for SD 3.5 Large).
    pub hidden_size: usize,
    /// Number of attention heads (64).
    pub num_heads: usize,
    /// Dimension per attention head (64).
    pub head_dim: usize,
    /// Number of MMDiT blocks (38).
    pub num_layers: usize,
    /// CLIP-L pooled embedding dimension (768).
    pub clip_l_dim: usize,
    /// CLIP-G pooled embedding dimension (1280).
    pub clip_g_dim: usize,
    /// T5-XXL sequence embedding dimension (4096).
    pub t5_dim: usize,
    /// Context dimension after projection (joint CLIP-L + CLIP-G = 2048, or T5 4096 projected).
    pub context_dim: usize,
    /// MLP expansion ratio (4.0).
    pub mlp_ratio: f32,
    /// Input latent channels (16 for SD3 VAE).
    pub in_channels: usize,
    /// Patch size (2).
    pub patch_size: usize,
}

impl Sd3Config {
    /// SD 3.5 Large configuration (8B params).
    pub fn large() -> Self {
        Self {
            hidden_size: 4096,
            num_heads: 64,
            head_dim: 64,
            num_layers: 38,
            clip_l_dim: 768,
            clip_g_dim: 1280,
            t5_dim: 4096,
            context_dim: 4096,
            mlp_ratio: 4.0,
            in_channels: 16,
            patch_size: 2,
        }
    }

    /// SD 3.5 Medium configuration.
    pub fn medium() -> Self {
        Self {
            hidden_size: 2560,
            num_heads: 40,
            head_dim: 64,
            num_layers: 24,
            clip_l_dim: 768,
            clip_g_dim: 1280,
            t5_dim: 4096,
            context_dim: 4096,
            mlp_ratio: 4.0,
            in_channels: 16,
            patch_size: 2,
        }
    }
}

// ============================================================================
// Compiled Metal kernels
// ============================================================================

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct Sd3Kernels {
    linear: Arc<crate::hal::metal::ComputePipeline>,
    silu: Arc<crate::hal::metal::ComputePipeline>,
    add: Arc<crate::hal::metal::ComputePipeline>,
    gelu: Arc<crate::hal::metal::ComputePipeline>,
    layer_norm: Arc<crate::hal::metal::ComputePipeline>,
    adaln_modulate: Arc<crate::hal::metal::ComputePipeline>,
    adaln_gate: Arc<crate::hal::metal::ComputePipeline>,
    patchify: Arc<crate::hal::metal::ComputePipeline>,
    unpatchify: Arc<crate::hal::metal::ComputePipeline>,
    batched_matmul_nn: Arc<crate::hal::metal::ComputePipeline>,
    row_softmax: Arc<crate::hal::metal::ComputePipeline>,
    transpose_shd_hsd: Arc<crate::hal::metal::ComputePipeline>,
    transpose_hsd_shd: Arc<crate::hal::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
impl Sd3Kernels {
    fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ADALN, "adaln_gate_f16")?,
            patchify: compute.compile_pipeline("patchify", sources::PATCHIFY, "patchify_f16")?,
            unpatchify: compute.compile_pipeline("unpatchify", sources::PATCHIFY, "unpatchify_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax: compute.compile_pipeline("row_softmax", sources::LINEAR, "row_softmax_scale_f16")?,
            transpose_shd_hsd: compute.compile_pipeline("transpose_shd_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_shd: compute.compile_pipeline("transpose_hsd_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
        })
    }
}

// ============================================================================
// SD3 GPU Transformer
// ============================================================================

/// SD 3.5 Large GPU transformer -- full forward pass on Metal.
///
/// The MMDiT architecture processes image and text tokens through joint
/// transformer blocks where both streams attend to the concatenated sequence.
/// Unlike Flux's sequential dual-stream, SD3's blocks are truly bidirectional:
/// both image and text Q/K/V are projected, concatenated, and attend jointly,
/// then split back for per-stream MLP processing.
#[cfg(feature = "metal")]
pub struct Sd3GpuTransformer {
    model: Arc<parking_lot::RwLock<Model>>,
    config: Sd3Config,
    kernels: Sd3Kernels,
}

#[cfg(feature = "metal")]
impl Sd3GpuTransformer {
    /// Create a new SD3 GPU transformer.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: Sd3Config, compute: &Arc<MetalCompute>) -> Result<Self> {
        let kernels = Sd3Kernels::new(compute)?;
        Ok(Self { model, config, kernels })
    }

    /// Full forward pass.
    ///
    /// `latents`: [1, 16, H, W] noisy latent (SD3 uses 16-channel VAE)
    /// `context`: [1, txt_seq, 4096] T5-XXL text embeddings
    /// `pooled_embeds`: [1, 2048] concatenated CLIP-L (768) + CLIP-G (1280) pooled embeddings
    /// `timestep`: scalar (0.0 -> 1.0, flow matching)
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        pooled_embeds: &Tensor,
        timestep: f32,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let (_batch, channels, height, width) = latents.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("latents must be [B, C, H, W]"))?;
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let mlp_dim = (hidden as f32 * self.config.mlp_ratio) as usize;

        // 1. Timestep embedding + pooled text conditioning -> vec
        let temb = self.timestep_embedding(timestep, compute)?;
        let cb0 = compute.new_command_buffer();
        let time_proj = self.gpu_mlp(&temb, "time_text_embed.timestep_embedder", 256, hidden, compute, cb0.as_ref())?;
        // Pooled CLIP embeddings: CLIP-L (768) + CLIP-G (1280) = 2048 -> hidden
        let pooled_proj = self.gpu_linear(pooled_embeds, "time_text_embed.text_embedder.linear_1", 2048, hidden, compute, cb0.as_ref())?;
        let pooled_act = self.gpu_silu(&pooled_proj, compute, cb0.as_ref())?;
        let pooled_out = self.gpu_linear(&pooled_act, "time_text_embed.text_embedder.linear_2", hidden, hidden, compute, cb0.as_ref())?;
        cb0.commit();
        cb0.wait_until_completed();

        // vec = time_proj + pooled_out
        let cb1 = compute.new_command_buffer();
        let vec = self.gpu_add(&time_proj, &pooled_out, compute, cb1.as_ref())?;
        cb1.commit();
        cb1.wait_until_completed();

        // 2. Patchify + project image tokens
        let num_patches = (height / 2) * (width / 2);
        let patch_dim = channels * 4; // 16 * 4 = 64

        let cb2 = compute.new_command_buffer();
        let patches = self.gpu_patchify(latents, channels, height, width, compute, cb2.as_ref())?;
        // pos_embed is a learned embedding, project patches and add position
        let img = self.gpu_linear(&patches, "pos_embed.proj", patch_dim, hidden, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // 3. Project text tokens: T5 4096 -> hidden
        let txt_seq_len = context.shape().dim(1).unwrap_or(256);
        let ctx_flat = context.reshape(Shape::from([txt_seq_len, self.config.t5_dim]))?;
        let cb3 = compute.new_command_buffer();
        let txt = self.gpu_linear(&ctx_flat, "context_embedder", self.config.t5_dim, hidden, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        let img_seq_len = num_patches;
        let total_seq = img_seq_len + txt_seq_len;

        // 4. MMDiT blocks: joint bidirectional attention
        let mut img_hidden = img;
        let mut txt_hidden = txt;

        self.model.read().prefetch_prefix("joint_blocks.0.");
        for i in 0..self.config.num_layers {
            if i + 1 < self.config.num_layers {
                self.model.read().prefetch_prefix(&format!("joint_blocks.{}.", i + 1));
            } else {
                self.model.read().prefetch_prefix("final_layer.");
            }
            let (new_img, new_txt) = self.mmdit_block(
                &img_hidden, &txt_hidden, &vec, i,
                img_seq_len, txt_seq_len, total_seq,
                hidden, num_heads, head_dim, mlp_dim,
                compute,
            )?;
            self.model.read().evict_prefix(&format!("joint_blocks.{}.", i));
            img_hidden = new_img;
            txt_hidden = new_txt;
        }

        // 5. Final layer: AdaLN modulation -> linear -> unpatchify
        let (final_shift, final_scale, _final_gate) = self.adaln_3params(
            &vec, "final_layer.adaLN_modulation.1", hidden, compute,
        )?;

        let cb_final = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(&img_hidden, hidden, compute, cb_final.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &final_scale, &final_shift, hidden, compute, cb_final.as_ref())?;
        let output_patches = self.gpu_linear(&modulated, "final_layer.linear", hidden, patch_dim, compute, cb_final.as_ref())?;
        let output = self.gpu_unpatchify(&output_patches, channels, height, width, compute, cb_final.as_ref())?;
        cb_final.commit();
        cb_final.wait_until_completed();

        Ok(output.reshape(Shape::from([1, channels, height, width]))?)
    }

    // ========================================================================
    // MMDiT Block: joint bidirectional attention
    // ========================================================================

    fn mmdit_block(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
        block_idx: usize,
        img_seq: usize,
        txt_seq: usize,
        total_seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor)> {
        let prefix = format!("joint_blocks.{}", block_idx);

        // AdaLN-Zero modulation: 6 params each for context (text) and x (image)
        let (txt_shift_attn, txt_scale_attn, txt_gate_attn,
             txt_shift_mlp, txt_scale_mlp, txt_gate_mlp) =
            self.adaln_6params(vec, &format!("{}.context_block.adaLN_modulation.1", prefix), hidden, compute)?;
        let (img_shift_attn, img_scale_attn, img_gate_attn,
             img_shift_mlp, img_scale_mlp, img_gate_mlp) =
            self.adaln_6params(vec, &format!("{}.x_block.adaLN_modulation.1", prefix), hidden, compute)?;

        // LayerNorm + modulate
        let cb = compute.new_command_buffer();
        let img_normed = self.gpu_layer_norm(img, hidden, compute, cb.as_ref())?;
        let img_mod = self.gpu_adaln_modulate(&img_normed, &img_scale_attn, &img_shift_attn, hidden, compute, cb.as_ref())?;
        let txt_normed = self.gpu_layer_norm(txt, hidden, compute, cb.as_ref())?;
        let txt_mod = self.gpu_adaln_modulate(&txt_normed, &txt_scale_attn, &txt_shift_attn, hidden, compute, cb.as_ref())?;

        // QKV projections (separate for each stream)
        let img_qkv = self.gpu_linear(&img_mod, &format!("{}.x_block.attn.qkv", prefix), hidden, 3 * hidden, compute, cb.as_ref())?;
        let txt_qkv = self.gpu_linear(&txt_mod, &format!("{}.context_block.attn.qkv", prefix), hidden, 3 * hidden, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Split QKV with QK-norm
        let (img_q, img_k, img_v) = self.split_qkv_with_norm(&img_qkv, img_seq, hidden, head_dim, &format!("{}.x_block.attn.norm", prefix), compute)?;
        let (txt_q, txt_k, txt_v) = self.split_qkv_with_norm(&txt_qkv, txt_seq, hidden, head_dim, &format!("{}.context_block.attn.norm", prefix), compute)?;

        // MMDiT joint attention: concatenate K/V from both streams,
        // both image and text queries attend to the full joint sequence
        let joint_k = Tensor::cat(&[txt_k, img_k], 0)?;
        let joint_v = Tensor::cat(&[txt_v, img_v], 0)?;

        // Image queries attend to joint K/V
        let img_attn = self.batched_attention(&img_q, &joint_k, &joint_v, img_seq, total_seq, num_heads, head_dim, compute)?;
        // Text queries attend to joint K/V
        let txt_attn = self.batched_attention(&txt_q, &joint_k, &joint_v, txt_seq, total_seq, num_heads, head_dim, compute)?;

        // Output projection + gated residual
        let cb2 = compute.new_command_buffer();
        let img_proj = self.gpu_linear(&img_attn, &format!("{}.x_block.attn.proj", prefix), hidden, hidden, compute, cb2.as_ref())?;
        let txt_proj = self.gpu_linear(&txt_attn, &format!("{}.context_block.attn.proj", prefix), hidden, hidden, compute, cb2.as_ref())?;
        let img_after_attn = self.gpu_adaln_gate(img, &img_proj, &img_gate_attn, hidden, compute, cb2.as_ref())?;
        let txt_after_attn = self.gpu_adaln_gate(txt, &txt_proj, &txt_gate_attn, hidden, compute, cb2.as_ref())?;

        // MLP: LayerNorm + modulate + FF (GELU-gated)
        let img_mlp_normed = self.gpu_layer_norm(&img_after_attn, hidden, compute, cb2.as_ref())?;
        let img_mlp_mod = self.gpu_adaln_modulate(&img_mlp_normed, &img_scale_mlp, &img_shift_mlp, hidden, compute, cb2.as_ref())?;
        let txt_mlp_normed = self.gpu_layer_norm(&txt_after_attn, hidden, compute, cb2.as_ref())?;
        let txt_mlp_mod = self.gpu_adaln_modulate(&txt_mlp_normed, &txt_scale_mlp, &txt_shift_mlp, hidden, compute, cb2.as_ref())?;

        // MLP: linear -> GELU -> linear
        let img_mlp_h = self.gpu_linear(&img_mlp_mod, &format!("{}.x_block.mlp.0", prefix), hidden, mlp_dim, compute, cb2.as_ref())?;
        let img_mlp_act = self.gpu_gelu(&img_mlp_h, compute, cb2.as_ref())?;
        let img_mlp_out = self.gpu_linear(&img_mlp_act, &format!("{}.x_block.mlp.2", prefix), mlp_dim, hidden, compute, cb2.as_ref())?;
        let txt_mlp_h = self.gpu_linear(&txt_mlp_mod, &format!("{}.context_block.mlp.0", prefix), hidden, mlp_dim, compute, cb2.as_ref())?;
        let txt_mlp_act = self.gpu_gelu(&txt_mlp_h, compute, cb2.as_ref())?;
        let txt_mlp_out = self.gpu_linear(&txt_mlp_act, &format!("{}.context_block.mlp.2", prefix), mlp_dim, hidden, compute, cb2.as_ref())?;

        // Gated residual for MLP
        let img_out = self.gpu_adaln_gate(&img_after_attn, &img_mlp_out, &img_gate_mlp, hidden, compute, cb2.as_ref())?;
        let txt_out = self.gpu_adaln_gate(&txt_after_attn, &txt_mlp_out, &txt_gate_mlp, hidden, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        Ok((img_out, txt_out))
    }

    // ========================================================================
    // QKV split with QK-norm
    // ========================================================================

    fn split_qkv_with_norm(
        &self,
        qkv: &Tensor,
        seq_len: usize,
        hidden: usize,
        head_dim: usize,
        norm_prefix: &str,
        _compute: &MetalCompute,
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
        let (q_normed, k_normed) = self.apply_qk_norm(&q, &k, seq_len, num_heads, head_dim, norm_prefix)?;

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
    ) -> Result<(Tensor, Tensor)> {
        let q_scale = self.w(&format!("{}.query_norm.scale", norm_prefix))?;
        let k_scale = self.w(&format!("{}.key_norm.scale", norm_prefix))?;

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
    // Modulation helpers
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
        let params = self.gpu_linear(&activated, weight_name, hidden_size, hidden_size * 6, compute, cb.as_ref())?;
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
        let params = self.gpu_linear(&activated, weight_name, hidden_size, hidden_size * 3, compute, cb.as_ref())?;
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
    // Batched attention (flash attention)
    // ========================================================================

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

        let flash_kernel = compute.compile_pipeline(
            "flash_attention_f16",
            crate::hal::metal::shader::sources::PHASE27_OPS,
            "flash_attention_f16",
        )?;
        let device_id = compute.device().info().id;
        let output_hsd = Tensor::empty(
            Shape::from([num_heads, q_seq_len, head_dim]), DType::F16, device_id,
        )?;
        let block: usize = 32;
        let q_blocks = (q_seq_len + block - 1) / block;
        let tg_mem = (3 * block * head_dim * 2) as u64;

        compute.dispatch_async(cb.as_ref(), &flash_kernel,
            (num_heads, q_blocks, 1), (block, 1, 1),
            |encoder| {
                use crate::hal::metal::buffer::BorrowedMetalBuffer;
                fn set_buf(encoder: &metal::ComputeCommandEncoderRef, idx: u64, t: &Tensor) {
                    if let Some(ptr) = t.device_ptr() {
                        let buf = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
                        encoder.set_buffer(idx, Some(buf.as_ref()), 0);
                    }
                }
                set_buf(encoder, 0, &q_hsd);
                set_buf(encoder, 1, &k_hsd);
                set_buf(encoder, 2, &v_hsd);
                set_buf(encoder, 3, &output_hsd);
                let vals: [u32; 3] = [q_seq_len as u32, kv_seq_len as u32, head_dim as u32];
                encoder.set_bytes(4, 4, &vals[0] as *const u32 as *const _);
                encoder.set_bytes(5, 4, &vals[1] as *const u32 as *const _);
                encoder.set_bytes(6, 4, &vals[2] as *const u32 as *const _);
                encoder.set_bytes(7, 4, &scale as *const f32 as *const _);
                encoder.set_threadgroup_memory_length(0, tg_mem);
            },
        );

        let attn_out_shd = self.gpu_transpose_hsd_shd(&output_hsd, q_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        Ok(attn_out_shd.reshape(Shape::from([q_seq_len, num_heads * head_dim]))?)
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
        let h = self.gpu_linear(x, &format!("{}.mlp.0", prefix), in_dim, out_dim, compute, cb)?;
        let h = self.gpu_silu_cb(&h, compute, cb)?;
        self.gpu_linear(&h, &format!("{}.mlp.2", prefix), out_dim, out_dim, compute, cb)
    }

    fn w(&self, name: &str) -> Result<&crate::hal::metal::LazyTensor> {
        self.model.read().get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("SD3 weight not found: {}", name)))
    }

    // ========================================================================
    // Low-level GPU dispatch helpers
    // ========================================================================

    fn gpu_linear(
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
        // Check for bias
        let has_bias_weight = self.model.read().get_weight(&format!("{}.bias", prefix));
        let has_bias: u32 = if has_bias_weight.is_some() { 1 } else { 0 };
        if has_bias == 1 {
            let b = has_bias_weight.unwrap();
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
        } else {
            compute.dispatch_async(cb, &self.kernels.linear,
                ((out_feat + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                    enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                    enc.set_buffer(1, Some(w.buffer()), 0);
                    enc.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias
                    enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                    enc.set_bytes(4, 4, &m as *const u32 as *const _);
                    enc.set_bytes(5, 4, &n as *const u32 as *const _);
                    enc.set_bytes(6, 4, &k as *const u32 as *const _);
                    enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
                });
        }
        Ok(output)
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

    fn gpu_unpatchify(&self, patches: &Tensor, channels: usize, height: usize, width: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([channels, height, width]), DType::F16, patches.device())?;
        let p_buf = borrow_tensor(patches)?;
        let o_buf = borrow_tensor(&output)?;
        let c_ch = channels as u32;
        let c_h = height as u32;
        let c_w = width as u32;
        let num_patches = (height / 2) * (width / 2);
        compute.dispatch_async(cb, &self.kernels.unpatchify,
            (num_patches, channels, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(p_buf.as_ref()), 0);
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
        let s = seq as u32;
        let h = heads as u32;
        let d = dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_shd_hsd,
            (heads, seq, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &s as *const u32 as *const _);
                enc.set_bytes(3, 4, &h as *const u32 as *const _);
                enc.set_bytes(4, 4, &d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_hsd_shd(&self, x: &Tensor, seq: usize, heads: usize, dim: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([seq, heads, dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let s = seq as u32;
        let h = heads as u32;
        let d = dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_hsd_shd,
            (heads, seq, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &s as *const u32 as *const _);
                enc.set_bytes(3, 4, &h as *const u32 as *const _);
                enc.set_bytes(4, 4, &d as *const u32 as *const _);
            });
        Ok(output)
    }
}

// ============================================================================
// Utility
// ============================================================================

fn rms_norm_inplace(x: &mut [f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for v in x.iter_mut() {
        *v /= rms;
    }
}

#[cfg(feature = "metal")]
fn borrow_tensor(t: &Tensor) -> Result<BorrowedMetalBuffer> {
    t.device_ptr()
        .map(|ptr| unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
        .ok_or_else(|| crate::core::Error::internal("tensor has no device pointer"))
}
