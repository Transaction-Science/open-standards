//! AuraFlow v0.3 Diffusion Transformer architecture.
//!
//! Architecture: 4 joint MMDiT blocks (dual-stream text+image attention)
//! + 32 single DiT blocks (image-only attention). Uses:
//! - AdaLN modulation (6-param: shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff)
//! - GEGLU feed-forward networks
//! - 2×2 patchification for spatial token reduction
//! - Register tokens for improved training stability
//!
//! All operations run on Metal GPU.

use crate::core::Result;
use crate::inference::model::Model;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::MetalCompute;
#[cfg(feature = "metal")]
use crate::hal::metal::BorrowedMetalBuffer;
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

/// AuraFlow model configuration.
#[derive(Debug, Clone)]
pub struct AuraFlowConfig {
    /// Hidden dimension (3072).
    pub hidden_size: usize,
    /// Number of attention heads (12).
    pub num_heads: usize,
    /// Dimension per attention head (256).
    pub head_dim: usize,
    /// Number of joint MMDiT blocks (4).
    pub num_joint_blocks: usize,
    /// Number of single DiT blocks (32).
    pub num_single_blocks: usize,
    /// Context embedding dimension from T5 (2048).
    pub context_dim: usize,
    /// Input/output latent channels (4).
    pub in_channels: usize,
    /// Patch size (2).
    pub patch_size: usize,
    /// Max position embedding size.
    pub pos_embed_max_size: usize,
}

impl AuraFlowConfig {
    /// AuraFlow v0.3 default config.
    pub fn v03() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 12,
            head_dim: 256,
            num_joint_blocks: 4,
            num_single_blocks: 32,
            context_dim: 2048,
            in_channels: 4,
            patch_size: 2,
            pos_embed_max_size: 9216,
        }
    }
}

/// AuraFlow transformer forward pass on Metal GPU.
#[cfg(feature = "metal")]
pub struct AuraFlowTransformer {
    model: Arc<Model>,
    config: AuraFlowConfig,
    kernels: AuraFlowKernels,
}

/// Compiled kernel pipelines for AuraFlow operations.
#[cfg(feature = "metal")]
struct AuraFlowKernels {
    linear: Arc<crate::hal::metal::ComputePipeline>,
    silu: Arc<crate::hal::metal::ComputePipeline>,
    add: Arc<crate::hal::metal::ComputePipeline>,
    layer_norm: Arc<crate::hal::metal::ComputePipeline>,
    geglu: Arc<crate::hal::metal::ComputePipeline>,
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
impl AuraFlowKernels {
    fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            geglu: compute.compile_pipeline("geglu", sources::GELU, "geglu_f16")?,
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

#[cfg(feature = "metal")]
impl AuraFlowTransformer {
    /// Create an AuraFlow transformer.
    pub fn new(model: Arc<Model>, config: AuraFlowConfig, compute: &Arc<MetalCompute>) -> Result<Self> {
        let kernels = AuraFlowKernels::new(compute)?;
        Ok(Self { model, config, kernels })
    }

    /// Full forward pass: patchify → embed → joint blocks → single blocks → unpatchify.
    ///
    /// `latents`: [1, 4, H, W] noisy latent image
    /// `context`: [1, seq_len, 2048] T5 text embeddings
    /// `timestep`: scalar timestep value (0.0 → 1.0 for flow matching)
    /// Returns: [1, 4, H, W] predicted velocity
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        timestep: f32,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let (_batch, channels, height, width) = latents.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("latents must be [B, C, H, W]"))?;
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let device = latents.device();

        // 1. Timestep embedding: sinusoidal → SiLU → linear → SiLU → linear
        let temb = self.timestep_embedding(timestep, compute)?;

        // 2. Patchify: [1, C, H, W] → [num_patches, C*patch_size²]
        let num_patches = (height / 2) * (width / 2);
        let patch_dim = channels * 4; // C * 2 * 2

        let cb = compute.new_command_buffer();
        let patches = self.gpu_patchify(latents, channels, height, width, compute, cb.as_ref())?;
        // Project patches to hidden dim: [num_patches, patch_dim] → [num_patches, hidden_size]
        let mut hidden = self.gpu_linear(&patches, "pos_embed.proj", patch_dim, hidden_size, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // 3. Add positional embedding
        hidden = self.add_pos_embed(&hidden, num_patches, compute)?;

        // 4. Prepend register tokens
        hidden = self.prepend_register_tokens(&hidden, compute)?;
        let img_seq_len = hidden.shape().dim(0)
            .ok_or_else(|| crate::core::Error::internal("hidden must have seq dim"))?;

        // 5. Context embedding: project T5 output to hidden_size
        let ctx = context.slice(0, 0, 1)?.reshape(Shape::from([
            context.shape().dim(1).unwrap_or(256),
            self.config.context_dim,
        ]))?;
        let ctx_seq_len = ctx.shape().dim(0).unwrap_or(256);

        let cb2 = compute.new_command_buffer();
        let mut ctx_hidden = self.gpu_linear_nobias(&ctx, "context_embedder", self.config.context_dim, hidden_size, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // 6. Joint transformer blocks (4): dual-stream attention on image + text
        //    Streaming NVMe: prefetch next block, evict current after use
        self.model.prefetch_prefix("joint_transformer_blocks.0.");
        for i in 0..self.config.num_joint_blocks {
            if i + 1 < self.config.num_joint_blocks {
                self.model.prefetch_prefix(&format!("joint_transformer_blocks.{}.", i + 1));
            } else {
                self.model.prefetch_prefix("single_transformer_blocks.0.");
            }
            let (new_hidden, new_ctx) = self.joint_block(
                &hidden, &ctx_hidden, &temb, i,
                img_seq_len, ctx_seq_len, hidden_size, num_heads, head_dim,
                compute,
            )?;
            self.model.evict_prefix(&format!("joint_transformer_blocks.{}.", i));
            hidden = new_hidden;
            ctx_hidden = new_ctx;
        }

        // 7. Single transformer blocks (32): image-only attention
        //    Streaming NVMe: prefetch next block, evict current after use
        for i in 0..self.config.num_single_blocks {
            if i + 1 < self.config.num_single_blocks {
                self.model.prefetch_prefix(&format!("single_transformer_blocks.{}.", i + 1));
            } else {
                self.model.prefetch_prefix("norm_out.");
            }
            hidden = self.single_block(
                &hidden, &temb, i,
                img_seq_len, hidden_size, num_heads, head_dim,
                compute,
            )?;
            self.model.evict_prefix(&format!("single_transformer_blocks.{}.", i));
        }

        // 8. Final norm + projection
        // Remove register tokens (first few tokens before image patches)
        let num_registers = 8; // AuraFlow uses 8 register tokens
        hidden = hidden.slice(0, num_registers, num_registers + num_patches)?;

        // AdaLN final: norm_out.linear produces [shift, scale, gate] but we just need shift+scale
        let cb_final = compute.new_command_buffer();
        let temb_buf = borrow_tensor(&temb)?;
        let norm_out_w = self.w("norm_out.linear.weight")?;

        // SiLU(temb) → linear → chunk into [scale, shift]
        let temb_activated = self.gpu_silu(&temb, compute, cb_final.as_ref())?;
        let norm_params = self.gpu_linear_buf(&temb_activated, norm_out_w, hidden_size, hidden_size * 2, compute, cb_final.as_ref())?;
        cb_final.commit();
        cb_final.wait_until_completed();

        // Split norm_params into scale and shift
        let norm_f32 = norm_params.to_f32_vec()?;
        let scale_f16: Vec<half::f16> = norm_f32[..hidden_size].iter().map(|&v| half::f16::from_f32(v)).collect();
        let shift_f16: Vec<half::f16> = norm_f32[hidden_size..hidden_size * 2].iter().map(|&v| half::f16::from_f32(v)).collect();
        let scale = Tensor::from_slice(&scale_f16, Shape::from([hidden_size]), DType::F16, device)?;
        let shift = Tensor::from_slice(&shift_f16, Shape::from([hidden_size]), DType::F16, device)?;

        // Apply LayerNorm + modulation
        let cb_out = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(&hidden, hidden_size, compute, cb_out.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &scale, &shift, hidden_size, compute, cb_out.as_ref())?;

        // Project to output channels: [num_patches, hidden_size] → [num_patches, patch_dim]
        let output_patches = self.gpu_linear_named(&modulated, "proj_out", hidden_size, patch_dim, compute, cb_out.as_ref())?;
        cb_out.commit();
        cb_out.wait_until_completed();

        // 9. Unpatchify: [num_patches, patch_dim] → [1, C, H, W]
        let cb_unpatch = compute.new_command_buffer();
        let output = self.gpu_unpatchify(&output_patches, channels, height, width, compute, cb_unpatch.as_ref())?;
        cb_unpatch.commit();
        cb_unpatch.wait_until_completed();

        // Reshape to [1, C, H, W]
        Ok(output.reshape(Shape::from([1, channels, height, width]))?)
    }

    // ========================================================================
    // Joint MMDiT Block
    // ========================================================================

    fn joint_block(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        temb: &Tensor,
        block_idx: usize,
        img_seq_len: usize,
        ctx_seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor)> {
        let prefix = format!("joint_transformer_blocks.{}", block_idx);

        // AdaLN modulation: get 6 params for image stream
        let (scale_attn, shift_attn, gate_attn, scale_ff, shift_ff, gate_ff) =
            self.adaln_6params(temb, &format!("{}.norm1.linear", prefix), hidden_size, compute)?;

        // AdaLN modulation: get 6 params for context stream
        let (ctx_scale_attn, ctx_shift_attn, ctx_gate_attn, ctx_scale_ff, ctx_shift_ff, ctx_gate_ff) =
            self.adaln_6params(temb, &format!("{}.norm1_context.linear", prefix), hidden_size, compute)?;

        // LayerNorm + modulate both streams
        let cb = compute.new_command_buffer();
        let img_normed = self.gpu_layer_norm(hidden, hidden_size, compute, cb.as_ref())?;
        let img_mod = self.gpu_adaln_modulate(&img_normed, &scale_attn, &shift_attn, hidden_size, compute, cb.as_ref())?;

        let ctx_normed = self.gpu_layer_norm(context, hidden_size, compute, cb.as_ref())?;
        let ctx_mod = self.gpu_adaln_modulate(&ctx_normed, &ctx_scale_attn, &ctx_shift_attn, hidden_size, compute, cb.as_ref())?;

        // Image QKV projections
        let img_q = self.gpu_linear_named(&img_mod, &format!("{}.attn.to_q", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let img_k = self.gpu_linear_named(&img_mod, &format!("{}.attn.to_k", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let img_v = self.gpu_linear_named(&img_mod, &format!("{}.attn.to_v", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;

        // Context QKV projections
        let ctx_q = self.gpu_linear_named(&ctx_mod, &format!("{}.attn.add_q_proj", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let ctx_k = self.gpu_linear_named(&ctx_mod, &format!("{}.attn.add_k_proj", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let ctx_v = self.gpu_linear_named(&ctx_mod, &format!("{}.attn.add_v_proj", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Concatenate K/V for joint attention: [img_seq + ctx_seq, hidden_size]
        let joint_k = Tensor::cat(&[img_k, ctx_k], 0)?;
        let joint_v = Tensor::cat(&[img_v, ctx_v], 0)?;
        let total_seq = img_seq_len + ctx_seq_len;

        // Joint attention for image queries
        let img_attn_out = self.batched_attention(
            &img_q, &joint_k, &joint_v,
            img_seq_len, total_seq, num_heads, head_dim,
            compute,
        )?;

        // Joint attention for context queries
        let ctx_attn_out = self.batched_attention(
            &ctx_q, &joint_k, &joint_v,
            ctx_seq_len, total_seq, num_heads, head_dim,
            compute,
        )?;

        // Output projections + gated residual
        let cb2 = compute.new_command_buffer();
        let img_proj = self.gpu_linear_named(&img_attn_out, &format!("{}.attn.to_out.0", prefix), hidden_size, hidden_size, compute, cb2.as_ref())?;
        let ctx_proj = self.gpu_linear_named(&ctx_attn_out, &format!("{}.attn.to_add_out", prefix), hidden_size, hidden_size, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // Gated residual for attention
        let cb3 = compute.new_command_buffer();
        let img_after_attn = self.gpu_adaln_gate(hidden, &img_proj, &gate_attn, hidden_size, compute, cb3.as_ref())?;
        let ctx_after_attn = self.gpu_adaln_gate(context, &ctx_proj, &ctx_gate_attn, hidden_size, compute, cb3.as_ref())?;

        // FFN for image stream (GEGLU)
        let img_ff_normed = self.gpu_layer_norm(&img_after_attn, hidden_size, compute, cb3.as_ref())?;
        let img_ff_mod = self.gpu_adaln_modulate(&img_ff_normed, &scale_ff, &shift_ff, hidden_size, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        let img_ff_out = self.geglu_ffn(&img_ff_mod, &format!("{}.ff", prefix), hidden_size, compute)?;

        // FFN for context stream (GEGLU)
        let cb4 = compute.new_command_buffer();
        let ctx_ff_normed = self.gpu_layer_norm(&ctx_after_attn, hidden_size, compute, cb4.as_ref())?;
        let ctx_ff_mod = self.gpu_adaln_modulate(&ctx_ff_normed, &ctx_scale_ff, &ctx_shift_ff, hidden_size, compute, cb4.as_ref())?;
        cb4.commit();
        cb4.wait_until_completed();

        let ctx_ff_out = self.geglu_ffn(&ctx_ff_mod, &format!("{}.ff_context", prefix), hidden_size, compute)?;

        // Gated residual for FFN
        let cb5 = compute.new_command_buffer();
        let img_out = self.gpu_adaln_gate(&img_after_attn, &img_ff_out, &gate_ff, hidden_size, compute, cb5.as_ref())?;
        let ctx_out = self.gpu_adaln_gate(&ctx_after_attn, &ctx_ff_out, &ctx_gate_ff, hidden_size, compute, cb5.as_ref())?;
        cb5.commit();
        cb5.wait_until_completed();

        Ok((img_out, ctx_out))
    }

    // ========================================================================
    // Single DiT Block
    // ========================================================================

    fn single_block(
        &self,
        hidden: &Tensor,
        temb: &Tensor,
        block_idx: usize,
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let prefix = format!("single_transformer_blocks.{}", block_idx);

        // AdaLN modulation: 6 params
        let (scale_attn, shift_attn, gate_attn, scale_ff, shift_ff, gate_ff) =
            self.adaln_6params(temb, &format!("{}.norm1.linear", prefix), hidden_size, compute)?;

        // LayerNorm + modulate
        let cb = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(hidden, hidden_size, compute, cb.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &scale_attn, &shift_attn, hidden_size, compute, cb.as_ref())?;

        // QKV projections
        let q = self.gpu_linear_named(&modulated, &format!("{}.attn.to_q", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let k = self.gpu_linear_named(&modulated, &format!("{}.attn.to_k", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let v = self.gpu_linear_named(&modulated, &format!("{}.attn.to_v", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Self-attention
        let attn_out = self.batched_attention(
            &q, &k, &v,
            seq_len, seq_len, num_heads, head_dim,
            compute,
        )?;

        // Output projection + gated residual
        let cb2 = compute.new_command_buffer();
        let proj = self.gpu_linear_named(&attn_out, &format!("{}.attn.to_out.0", prefix), hidden_size, hidden_size, compute, cb2.as_ref())?;
        let after_attn = self.gpu_adaln_gate(hidden, &proj, &gate_attn, hidden_size, compute, cb2.as_ref())?;

        // FFN (GEGLU)
        let ff_normed = self.gpu_layer_norm(&after_attn, hidden_size, compute, cb2.as_ref())?;
        let ff_mod = self.gpu_adaln_modulate(&ff_normed, &scale_ff, &shift_ff, hidden_size, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        let ff_out = self.geglu_ffn(&ff_mod, &format!("{}.ff", prefix), hidden_size, compute)?;

        // Gated residual
        let cb3 = compute.new_command_buffer();
        let output = self.gpu_adaln_gate(&after_attn, &ff_out, &gate_ff, hidden_size, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        Ok(output)
    }

    // ========================================================================
    // Shared GPU operations
    // ========================================================================

    /// Compute timestep embedding via sinusoidal → MLP.
    fn timestep_embedding(&self, timestep: f32, compute: &MetalCompute) -> Result<Tensor> {
        let device = compute.device().info().id;
        // Sinusoidal embedding (256-dim)
        let emb = crate::inference::architecture::dit::DiTOps::timestep_embedding(timestep, 256, device)?;

        // MLP: linear_1 → SiLU → linear_2
        let cb = compute.new_command_buffer();
        let h = self.gpu_linear_biased(&emb, "time_step_proj.linear_1", 256, self.config.hidden_size, compute, cb.as_ref())?;
        let h = self.gpu_silu(&h, compute, cb.as_ref())?;
        let out = self.gpu_linear_biased(&h, "time_step_proj.linear_2", self.config.hidden_size, self.config.hidden_size, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// Compute 6 AdaLN parameters from timestep embedding.
    fn adaln_6params(
        &self,
        temb: &Tensor,
        weight_name: &str,
        hidden_size: usize,
        compute: &MetalCompute,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let cb = compute.new_command_buffer();
        let activated = self.gpu_silu(temb, compute, cb.as_ref())?;
        let params = self.gpu_linear_named(&activated, weight_name, hidden_size, hidden_size * 6, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Split into 6 chunks of hidden_size
        let data_f32 = params.to_f32_vec()?;
        let device = temb.device();
        let shape = Shape::from([hidden_size]);

        let to_f16_slice = |start: usize, end: usize| -> Vec<half::f16> {
            data_f32[start..end].iter().map(|&v| half::f16::from_f32(v)).collect()
        };

        let shift_attn = Tensor::from_slice(&to_f16_slice(0, hidden_size), shape.clone(), DType::F16, device)?;
        let scale_attn = Tensor::from_slice(&to_f16_slice(hidden_size, hidden_size * 2), shape.clone(), DType::F16, device)?;
        let gate_attn = Tensor::from_slice(&to_f16_slice(hidden_size * 2, hidden_size * 3), shape.clone(), DType::F16, device)?;
        let shift_ff = Tensor::from_slice(&to_f16_slice(hidden_size * 3, hidden_size * 4), shape.clone(), DType::F16, device)?;
        let scale_ff = Tensor::from_slice(&to_f16_slice(hidden_size * 4, hidden_size * 5), shape.clone(), DType::F16, device)?;
        let gate_ff = Tensor::from_slice(&to_f16_slice(hidden_size * 5, hidden_size * 6), shape.clone(), DType::F16, device)?;

        Ok((scale_attn, shift_attn, gate_attn, scale_ff, shift_ff, gate_ff))
    }

    /// GEGLU feed-forward network.
    fn geglu_ffn(
        &self,
        x: &Tensor,
        prefix: &str,
        hidden_size: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        // AuraFlow FFN: linear_1 (gate, up) → GEGLU → out_projection
        // linear_1 projects to 2 * ff_dim, split into gate and up
        // linear_2 seems to be the activation variant; out_projection is the down projection
        let ff_dim = hidden_size * 4; // standard 4x expansion

        let cb = compute.new_command_buffer();
        // linear_1: [seq, hidden_size] → [seq, ff_dim * 2] (GEGLU splits internally)
        let gate_up = self.gpu_linear_named(x, &format!("{}.linear_1", prefix), hidden_size, ff_dim * 2, compute, cb.as_ref())?;

        // Apply GEGLU: split into gate and value, compute gate * GELU(value)
        let seq_len = x.shape().dim(0).unwrap_or(1);
        let activated = self.gpu_geglu(&gate_up, seq_len, ff_dim, compute, cb.as_ref())?;

        // linear_2 (intermediate projection if present, otherwise skip)
        // out_projection: [seq, ff_dim] → [seq, hidden_size]
        let output = self.gpu_linear_named(&activated, &format!("{}.out_projection", prefix), ff_dim, hidden_size, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    /// Batched multi-head attention: Q@K^T → softmax → S@V.
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
        let device = q.device();

        // Reshape Q: [q_seq, num_heads * head_dim] → [num_heads, q_seq, head_dim]
        let q_shd = q.reshape(Shape::from([q_seq_len, num_heads, head_dim]))?;
        let k_shd = k.reshape(Shape::from([kv_seq_len, num_heads, head_dim]))?;
        let v_shd = v.reshape(Shape::from([kv_seq_len, num_heads, head_dim]))?;

        // Transpose SHD → HSD
        let cb = compute.new_command_buffer();
        let q_hsd = self.gpu_transpose_shd_hsd(&q_shd, q_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        let k_hsd = self.gpu_transpose_shd_hsd(&k_shd, kv_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        let v_hsd = self.gpu_transpose_shd_hsd(&v_shd, kv_seq_len, num_heads, head_dim, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Scores = Q @ K^T: [num_heads, q_seq, head_dim] @ [num_heads, head_dim, kv_seq] → [num_heads, q_seq, kv_seq]
        let cb2 = compute.new_command_buffer();
        let k_t = k_hsd; // batched_linear does X@W^T, so K in HSD is already correct
        let scores = self.gpu_batched_linear_raw(&q_hsd, &k_t, num_heads, q_seq_len, head_dim, kv_seq_len, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // Softmax with scale
        let cb3 = compute.new_command_buffer();
        let attn_weights = self.gpu_row_softmax(&scores, num_heads * q_seq_len, kv_seq_len, scale, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        // Output = Weights @ V: [num_heads, q_seq, kv_seq] @ [num_heads, kv_seq, head_dim] → [num_heads, q_seq, head_dim]
        let cb4 = compute.new_command_buffer();
        let attn_out_hsd = self.gpu_batched_matmul_nn(&attn_weights, &v_hsd, num_heads, q_seq_len, kv_seq_len, head_dim, compute, cb4.as_ref())?;

        // Transpose HSD → SHD
        let attn_out_shd = self.gpu_transpose_hsd_shd(&attn_out_hsd, q_seq_len, num_heads, head_dim, compute, cb4.as_ref())?;
        cb4.commit();
        cb4.wait_until_completed();

        // Reshape [q_seq, num_heads, head_dim] → [q_seq, hidden_size]
        Ok(attn_out_shd.reshape(Shape::from([q_seq_len, num_heads * head_dim]))?)
    }

    /// Add positional embedding to hidden states.
    fn add_pos_embed(&self, hidden: &Tensor, num_patches: usize, compute: &MetalCompute) -> Result<Tensor> {
        let pos_embed_weight = self.w("pos_embed.pos_embed")?;
        // pos_embed is [max_size, hidden_size], take first num_patches rows
        let pos_f32 = pos_embed_weight.to_f32_vec()?;
        let pos_f16: Vec<half::f16> = pos_f32[..num_patches * self.config.hidden_size]
            .iter().map(|&v| half::f16::from_f32(v)).collect();
        let pos_tensor = Tensor::from_slice(
            &pos_f16,
            Shape::from([num_patches, self.config.hidden_size]),
            DType::F16,
            hidden.device(),
        )?;

        let cb = compute.new_command_buffer();
        let result = self.gpu_add(hidden, &pos_tensor, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    /// Prepend register tokens to hidden states.
    fn prepend_register_tokens(&self, hidden: &Tensor, compute: &MetalCompute) -> Result<Tensor> {
        let reg_weight = self.w("register_tokens")?;
        // register_tokens: [1, num_registers, hidden_size] — squeeze batch dim
        let reg_f32 = reg_weight.to_f32_vec()?;
        let reg_data: Vec<half::f16> = reg_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let num_registers = reg_data.len() / self.config.hidden_size;
        let reg_tensor = Tensor::from_slice(
            &reg_data,
            Shape::from([num_registers, self.config.hidden_size]),
            DType::F16,
            hidden.device(),
        )?;
        Tensor::cat(&[reg_tensor, hidden.clone()], 0)
    }

    // ========================================================================
    // Low-level GPU kernel dispatch helpers
    // ========================================================================

    fn w(&self, name: &str) -> Result<&crate::hal::metal::LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("AuraFlow weight not found: {}", name)))
    }

    fn gpu_patchify(
        &self, input: &Tensor, channels: usize, height: usize, width: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
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

    fn gpu_unpatchify(
        &self, input: &Tensor, channels: usize, height: usize, width: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
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

    fn gpu_linear(
        &self, x: &Tensor, weight_prefix: &str, in_features: usize, out_features: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let w = self.w(&format!("{}.weight", weight_prefix))?;
        let b = self.model.get_weight(&format!("{}.bias", weight_prefix));
        let seq_len = x.shape().numel() / in_features;
        let output = Tensor::empty(Shape::from([seq_len, out_features]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_features as u32;
        let k = in_features as u32;
        // linear_f16: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        if let Some(bias) = b {
            let has_bias: u32 = 1;
            compute.dispatch_async(cb, &self.kernels.linear,
                ((out_features + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                    enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                    enc.set_buffer(1, Some(w.buffer()), 0);
                    enc.set_buffer(2, Some(bias.buffer()), 0);
                    enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                    enc.set_bytes(4, 4, &m as *const u32 as *const _);
                    enc.set_bytes(5, 4, &n as *const u32 as *const _);
                    enc.set_bytes(6, 4, &k as *const u32 as *const _);
                    enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
                });
        } else {
            let has_bias: u32 = 0;
            compute.dispatch_async(cb, &self.kernels.linear,
                ((out_features + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
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

    fn gpu_linear_nobias(
        &self, x: &Tensor, weight_prefix: &str, in_features: usize, out_features: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let w = self.w(&format!("{}.weight", weight_prefix))?;
        let seq_len = x.shape().numel() / in_features;
        let output = Tensor::empty(Shape::from([seq_len, out_features]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_features as u32;
        let k = in_features as u32;
        let has_bias: u32 = 0;
        // linear_f16: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        compute.dispatch_async(cb, &self.kernels.linear,
            ((out_features + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w.buffer()), 0);
                enc.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &m as *const u32 as *const _);
                enc.set_bytes(5, 4, &n as *const u32 as *const _);
                enc.set_bytes(6, 4, &k as *const u32 as *const _);
                enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_linear_named(
        &self, x: &Tensor, weight_name: &str, in_features: usize, out_features: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let w = self.w(&format!("{}.weight", weight_name))?;
        let seq_len = x.shape().numel() / in_features;
        let output = Tensor::empty(Shape::from([seq_len, out_features]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_features as u32;
        let k = in_features as u32;
        let has_bias: u32 = 0;
        // linear_f16: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        compute.dispatch_async(cb, &self.kernels.linear,
            ((out_features + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w.buffer()), 0);
                enc.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &m as *const u32 as *const _);
                enc.set_bytes(5, 4, &n as *const u32 as *const _);
                enc.set_bytes(6, 4, &k as *const u32 as *const _);
                enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_linear_biased(
        &self, x: &Tensor, weight_prefix: &str, in_features: usize, out_features: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        // Linear Y = X @ W^T + bias
        let w = self.w(&format!("{}.weight", weight_prefix))?;
        let b = self.w(&format!("{}.bias", weight_prefix))?;
        let seq_len = x.shape().numel() / in_features;
        let output = Tensor::empty(Shape::from([seq_len, out_features]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_features as u32;
        let k = in_features as u32;
        let has_bias: u32 = 1;
        // linear_f16: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        compute.dispatch_async(cb, &self.kernels.linear,
            ((out_features + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
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

    fn gpu_linear_buf(
        &self, x: &Tensor, w: &crate::hal::metal::LazyTensor, in_features: usize, out_features: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let seq_len = x.shape().numel() / in_features;
        let output = Tensor::empty(Shape::from([seq_len, out_features]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let m = seq_len as u32;
        let n = out_features as u32;
        let k = in_features as u32;
        let has_bias: u32 = 0;
        // linear_f16: X(0), W(1), bias(2), Y(3), M(4), N(5), K(6), has_bias(7)
        compute.dispatch_async(cb, &self.kernels.linear,
            ((out_features + 15) / 16, (seq_len + 15) / 16, 1), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w.buffer()), 0);
                enc.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &m as *const u32 as *const _);
                enc.set_bytes(5, 4, &n as *const u32 as *const _);
                enc.set_bytes(6, 4, &k as *const u32 as *const _);
                enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_silu(&self, x: &Tensor, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_count = count as u32;
        compute.dispatch_async(cb, &self.kernels.silu,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_count as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_add(&self, a: &Tensor, b: &Tensor, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = a.shape().numel();
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c_count = count as u32;
        compute.dispatch_async(cb, &self.kernels.add,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(a_buf.as_ref()), 0);
                enc.set_buffer(1, Some(b_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_count as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_layer_norm(
        &self, x: &Tensor, hidden_size: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let seq_len = x.shape().numel() / hidden_size;
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_hidden = hidden_size as u32;
        let c_seq = seq_len as u32;
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

    fn gpu_adaln_modulate(
        &self, x: &Tensor, scale: &Tensor, shift: &Tensor, hidden_size: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let s_buf = borrow_tensor(scale)?;
        let sh_buf = borrow_tensor(shift)?;
        let o_buf = borrow_tensor(&output)?;
        let c_hidden = hidden_size as u32;
        let c_count = count as u32;
        compute.dispatch_async(cb, &self.kernels.adaln_modulate,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(s_buf.as_ref()), 0);
                enc.set_buffer(2, Some(sh_buf.as_ref()), 0);
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &c_hidden as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_adaln_gate(
        &self, x: &Tensor, residual: &Tensor, gate: &Tensor, hidden_size: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let r_buf = borrow_tensor(residual)?;
        let g_buf = borrow_tensor(gate)?;
        let o_buf = borrow_tensor(&output)?;
        let c_hidden = hidden_size as u32;
        let c_count = count as u32;
        compute.dispatch_async(cb, &self.kernels.adaln_gate,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(r_buf.as_ref()), 0);
                enc.set_buffer(2, Some(g_buf.as_ref()), 0);
                enc.set_buffer(3, Some(o_buf.as_ref()), 0);
                enc.set_bytes(4, 4, &c_hidden as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_geglu(
        &self, x: &Tensor, seq_len: usize, ff_dim: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([seq_len, ff_dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let total = (seq_len * ff_dim) as u32;
        let c_ff = ff_dim as u32;
        compute.dispatch_async(cb, &self.kernels.geglu,
            ((seq_len * ff_dim + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &total as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_ff as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_shd_hsd(
        &self, x: &Tensor, seq_len: usize, num_heads: usize, head_dim: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq_len as u32;
        let c_h = num_heads as u32;
        let c_d = head_dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_shd_hsd,
            (num_heads, seq_len, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_s as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_hsd_shd(
        &self, x: &Tensor, seq_len: usize, num_heads: usize, head_dim: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq_len as u32;
        let c_h = num_heads as u32;
        let c_d = head_dim as u32;
        compute.dispatch_async(cb, &self.kernels.transpose_hsd_shd,
            (num_heads, seq_len, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_s as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_batched_linear_raw(
        &self, x: &Tensor, w: &Tensor, batch: usize, m: usize, k: usize, n: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let w_buf = borrow_tensor(w)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32;
        let c_n = n as u32;
        let c_k = k as u32;
        let c_batch = batch as u32;
        compute.dispatch_async(cb, &self.kernels.batched_linear,
            ((n + 15) / 16, (m + 15) / 16, batch), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(w_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_m as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_n as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_k as *const u32 as *const _);
                enc.set_bytes(6, 4, &c_batch as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_batched_matmul_nn(
        &self, a: &Tensor, b: &Tensor, batch: usize, m: usize, k: usize, n: usize,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32;
        let c_n = n as u32;
        let c_k = k as u32;
        let c_batch = batch as u32;
        compute.dispatch_async(cb, &self.kernels.batched_matmul_nn,
            ((n + 15) / 16, (m + 15) / 16, batch), (16, 16, 1), |enc| {
                enc.set_buffer(0, Some(a_buf.as_ref()), 0);
                enc.set_buffer(1, Some(b_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c_m as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_n as *const u32 as *const _);
                enc.set_bytes(5, 4, &c_k as *const u32 as *const _);
                enc.set_bytes(6, 4, &c_batch as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_row_softmax(
        &self, x: &Tensor, num_rows: usize, row_len: usize, scale: f32,
        compute: &MetalCompute, cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_rows = num_rows as u32;
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
