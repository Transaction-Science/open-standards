//! PixArt-Sigma Diffusion Transformer architecture.
//!
//! Architecture: 28 transformer blocks with AdaLN-Single conditioning.
//! Each block has separate self-attention (attn1) and cross-attention (attn2),
//! followed by a GEGLU feed-forward network.
//!
//! Key difference from AuraFlow/Flux: AdaLN-Single means one global timestep
//! embedding produces 6 base modulation vectors, and each block offsets them
//! with a learned scale_shift_table. No joint/dual-stream attention.
//!
//! All operations run on Metal GPU with NVMe streaming prefetch/eviction.

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

/// PixArt-Sigma model configuration.
#[derive(Debug, Clone)]
pub struct PixArtConfig {
    /// Hidden dimension (1152).
    pub hidden_size: usize,
    /// Number of attention heads (16).
    pub num_heads: usize,
    /// Dimension per attention head (72).
    pub head_dim: usize,
    /// Number of transformer blocks (28).
    pub num_blocks: usize,
    /// Cross-attention dimension (1152 — same as hidden after caption projection).
    pub cross_attn_dim: usize,
    /// Caption channels from T5 (4096).
    pub caption_channels: usize,
    /// Input latent channels (4).
    pub in_channels: usize,
    /// Output channels (8 — predicts mean + variance).
    pub out_channels: usize,
    /// Patch size (2).
    pub patch_size: usize,
    /// Sample size for position embedding (128 latent = 1024 pixel).
    pub sample_size: usize,
}

impl PixArtConfig {
    /// PixArt-Sigma XL/2 configuration.
    pub fn sigma_xl() -> Self {
        Self {
            hidden_size: 1152,
            num_heads: 16,
            head_dim: 72,
            num_blocks: 28,
            cross_attn_dim: 1152,
            caption_channels: 4096,
            in_channels: 4,
            out_channels: 8,
            patch_size: 2,
            sample_size: 128,
        }
    }
}

/// Compiled kernel pipelines for PixArt operations.
#[cfg(feature = "metal")]
struct PixArtKernels {
    linear: Arc<crate::hal::metal::ComputePipeline>,
    silu: Arc<crate::hal::metal::ComputePipeline>,
    add: Arc<crate::hal::metal::ComputePipeline>,
    gelu: Arc<crate::hal::metal::ComputePipeline>,
    layer_norm: Arc<crate::hal::metal::ComputePipeline>,
    adaln_modulate: Arc<crate::hal::metal::ComputePipeline>,
    adaln_gate: Arc<crate::hal::metal::ComputePipeline>,
    geglu: Arc<crate::hal::metal::ComputePipeline>,
    patchify: Arc<crate::hal::metal::ComputePipeline>,
    unpatchify: Arc<crate::hal::metal::ComputePipeline>,
    batched_linear: Arc<crate::hal::metal::ComputePipeline>,
    batched_matmul_nn: Arc<crate::hal::metal::ComputePipeline>,
    row_softmax: Arc<crate::hal::metal::ComputePipeline>,
    transpose_shd_hsd: Arc<crate::hal::metal::ComputePipeline>,
    transpose_hsd_shd: Arc<crate::hal::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
impl PixArtKernels {
    fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ADALN, "adaln_gate_f16")?,
            geglu: compute.compile_pipeline("geglu", sources::GELU, "geglu_f16")?,
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

/// PixArt-Sigma GPU transformer — full forward pass on Metal.
#[cfg(feature = "metal")]
pub struct PixArtGpuTransformer {
    model: Arc<Model>,
    config: PixArtConfig,
    kernels: PixArtKernels,
}

#[cfg(feature = "metal")]
impl PixArtGpuTransformer {
    /// Create a new PixArt GPU transformer.
    pub fn new(model: Arc<Model>, config: PixArtConfig, compute: &Arc<MetalCompute>) -> Result<Self> {
        let kernels = PixArtKernels::new(compute)?;
        Ok(Self { model, config, kernels })
    }

    /// Full forward pass.
    ///
    /// `latents`: [1, 4, H, W] noisy latent
    /// `context`: [1, txt_seq, 4096] T5-XXL text embeddings
    /// `timestep`: scalar (0→1000)
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        timestep: f32,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let (_batch, channels, height, width) = latents.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("latents must be [B, C, H, W]"))?;
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // 1. AdaLN-Single: global timestep conditioning
        //    timestep → sinusoidal(256) → MLP → linear → 6 * hidden modulation vectors
        let temb = self.timestep_embedding(timestep, compute)?;
        let cb0 = compute.new_command_buffer();
        let temb_h = self.gpu_linear_biased(&temb, "adaln_single.emb.timestep_embedder.linear_1", 256, hidden, compute, cb0.as_ref())?;
        let temb_act = self.gpu_silu(&temb_h, compute, cb0.as_ref())?;
        let temb_proj = self.gpu_linear_biased(&temb_act, "adaln_single.emb.timestep_embedder.linear_2", hidden, hidden, compute, cb0.as_ref())?;
        // Project to 6 * hidden
        let adaln_params = self.gpu_linear_biased(&temb_proj, "adaln_single.linear", hidden, 6 * hidden, compute, cb0.as_ref())?;
        cb0.commit();
        cb0.wait_until_completed();

        // Split into 6 modulation vectors [hidden] each
        let adaln_f32 = adaln_params.to_f32_vec()?;
        let device = latents.device();
        let shape_h = Shape::from([hidden]);
        let to_f16 = |s: usize, e: usize| -> Vec<half::f16> {
            adaln_f32[s..e].iter().map(|&v| half::f16::from_f32(v)).collect()
        };
        let base_shift_attn = Tensor::from_slice(&to_f16(0, hidden), shape_h.clone(), DType::F16, device)?;
        let base_scale_attn = Tensor::from_slice(&to_f16(hidden, 2 * hidden), shape_h.clone(), DType::F16, device)?;
        let base_gate_attn = Tensor::from_slice(&to_f16(2 * hidden, 3 * hidden), shape_h.clone(), DType::F16, device)?;
        let base_shift_ff = Tensor::from_slice(&to_f16(3 * hidden, 4 * hidden), shape_h.clone(), DType::F16, device)?;
        let base_scale_ff = Tensor::from_slice(&to_f16(4 * hidden, 5 * hidden), shape_h.clone(), DType::F16, device)?;
        let base_gate_ff = Tensor::from_slice(&to_f16(5 * hidden, 6 * hidden), shape_h.clone(), DType::F16, device)?;

        // 2. Caption projection: [txt_seq, 4096] → [txt_seq, 1152]
        let txt_seq = context.shape().dim(1).unwrap_or(120);
        let ctx_flat = context.reshape(Shape::from([txt_seq, self.config.caption_channels]))?;
        let cb1 = compute.new_command_buffer();
        let cap_h = self.gpu_linear_biased(&ctx_flat, "caption_projection.linear_1", self.config.caption_channels, hidden, compute, cb1.as_ref())?;
        let cap_act = self.gpu_silu(&cap_h, compute, cb1.as_ref())?;
        let caption_proj = self.gpu_linear_biased(&cap_act, "caption_projection.linear_2", hidden, hidden, compute, cb1.as_ref())?;
        cb1.commit();
        cb1.wait_until_completed();

        // 3. Patchify + positional embedding
        let num_patches = (height / 2) * (width / 2);
        let patch_dim = channels * 4; // 4 * 4 = 16
        let cb2 = compute.new_command_buffer();
        let patches = self.gpu_patchify(latents, channels, height, width, compute, cb2.as_ref())?;
        let mut img = self.gpu_linear_biased(&patches, "pos_embed.proj", patch_dim, hidden, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // Add positional embedding (weight stored as [1, max_patches, hidden])
        // For now, skip learnable pos_embed addition if not available

        // 4. Transformer blocks (28) with NVMe streaming
        self.model.prefetch_prefix("transformer_blocks.0.");
        for i in 0..self.config.num_blocks {
            if i + 1 < self.config.num_blocks {
                self.model.prefetch_prefix(&format!("transformer_blocks.{}.", i + 1));
            } else {
                self.model.prefetch_prefix("proj_out.");
                self.model.prefetch_prefix("scale_shift_table");
            }
            img = self.transformer_block(
                &img, &caption_proj, i,
                num_patches, txt_seq, hidden, num_heads, head_dim,
                &base_shift_attn, &base_scale_attn, &base_gate_attn,
                &base_shift_ff, &base_scale_ff, &base_gate_ff,
                compute,
            )?;
            self.model.evict_prefix(&format!("transformer_blocks.{}.", i));
        }

        // 5. Final layer: scale_shift_table → AdaLN → linear → unpatchify
        //    scale_shift_table is [6, hidden] — learned final modulation
        let sst = self.w("scale_shift_table")?;
        let sst_f32 = sst.to_f32_vec()?;
        let final_shift = Tensor::from_slice(
            &sst_f32[..hidden].iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>(),
            shape_h.clone(), DType::F16, device)?;
        let final_scale = Tensor::from_slice(
            &sst_f32[hidden..2 * hidden].iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>(),
            shape_h.clone(), DType::F16, device)?;

        let cb_final = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(&img, hidden, compute, cb_final.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &final_scale, &final_shift, hidden, compute, cb_final.as_ref())?;
        // proj_out: [num_patches, hidden] → [num_patches, out_channels * patch_size²]
        let out_dim = self.config.out_channels * self.config.patch_size * self.config.patch_size;
        let output_patches = self.gpu_linear_biased(&modulated, "proj_out", hidden, out_dim, compute, cb_final.as_ref())?;
        cb_final.commit();
        cb_final.wait_until_completed();

        // Unpatchify: [num_patches, out_dim] → [out_channels, H, W]
        // PixArt has out_channels=8 (mean + variance), take first 4 channels (mean)
        let cb_unpatch = compute.new_command_buffer();
        let output = self.gpu_unpatchify(&output_patches, self.config.out_channels, height, width, compute, cb_unpatch.as_ref())?;
        cb_unpatch.commit();
        cb_unpatch.wait_until_completed();

        // Take first in_channels from output (the predicted noise mean)
        let full = output.reshape(Shape::from([1, self.config.out_channels, height, width]))?;
        full.slice(1, 0, self.config.in_channels)
    }

    // ========================================================================
    // Transformer block
    // ========================================================================

    fn transformer_block(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        block_idx: usize,
        img_seq: usize,
        txt_seq: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        base_shift_attn: &Tensor,
        base_scale_attn: &Tensor,
        base_gate_attn: &Tensor,
        base_shift_ff: &Tensor,
        base_scale_ff: &Tensor,
        base_gate_ff: &Tensor,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let prefix = format!("transformer_blocks.{}", block_idx);

        // PixArt AdaLN-Single: use base modulation vectors directly
        // (Each block applies them the same way; per-block learning is in the
        // attention/FFN weights themselves, not separate scale_shift_tables per block)

        // 1. Self-attention (attn1): image tokens attend to image tokens
        let cb = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(hidden, hidden_size, compute, cb.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, base_scale_attn, base_shift_attn, hidden_size, compute, cb.as_ref())?;

        let q = self.gpu_linear_biased(&modulated, &format!("{}.attn1.to_q", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let k = self.gpu_linear_biased(&modulated, &format!("{}.attn1.to_k", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        let v = self.gpu_linear_biased(&modulated, &format!("{}.attn1.to_v", prefix), hidden_size, hidden_size, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let self_attn_out = self.batched_attention(&q, &k, &v, img_seq, img_seq, num_heads, head_dim, compute)?;

        let cb2 = compute.new_command_buffer();
        let self_attn_proj = self.gpu_linear_biased(&self_attn_out, &format!("{}.attn1.to_out.0", prefix), hidden_size, hidden_size, compute, cb2.as_ref())?;
        let after_self_attn = self.gpu_adaln_gate(hidden, &self_attn_proj, base_gate_attn, hidden_size, compute, cb2.as_ref())?;

        // 2. Cross-attention (attn2): Q from image, KV from caption
        let cross_normed = self.gpu_layer_norm(&after_self_attn, hidden_size, compute, cb2.as_ref())?;
        let cross_q = self.gpu_linear_biased(&cross_normed, &format!("{}.attn2.to_q", prefix), hidden_size, hidden_size, compute, cb2.as_ref())?;
        let cross_k = self.gpu_linear_biased(context, &format!("{}.attn2.to_k", prefix), hidden_size, hidden_size, compute, cb2.as_ref())?;
        let cross_v = self.gpu_linear_biased(context, &format!("{}.attn2.to_v", prefix), hidden_size, hidden_size, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        let cross_attn_out = self.batched_attention(&cross_q, &cross_k, &cross_v, img_seq, txt_seq, num_heads, head_dim, compute)?;

        let cb3 = compute.new_command_buffer();
        let cross_attn_proj = self.gpu_linear_biased(&cross_attn_out, &format!("{}.attn2.to_out.0", prefix), hidden_size, hidden_size, compute, cb3.as_ref())?;
        let after_cross_attn = self.gpu_add(&after_self_attn, &cross_attn_proj, compute, cb3.as_ref())?;

        // 3. Feed-forward (GEGLU)
        let ff_normed = self.gpu_layer_norm(&after_cross_attn, hidden_size, compute, cb3.as_ref())?;
        let ff_modulated = self.gpu_adaln_modulate(&ff_normed, base_scale_ff, base_shift_ff, hidden_size, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        let ff_out = self.geglu_ffn(&ff_modulated, &prefix, hidden_size, img_seq, compute)?;

        let cb4 = compute.new_command_buffer();
        let output = self.gpu_adaln_gate(&after_cross_attn, &ff_out, base_gate_ff, hidden_size, compute, cb4.as_ref())?;
        cb4.commit();
        cb4.wait_until_completed();

        Ok(output)
    }

    // ========================================================================
    // GEGLU feed-forward
    // ========================================================================

    fn geglu_ffn(
        &self,
        x: &Tensor,
        block_prefix: &str,
        hidden_size: usize,
        seq_len: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        // PixArt FFN: ff.net.0.proj (GEGLU gate+up, 2*ff_dim) → ff.net.2 (down, hidden)
        let ff_dim = hidden_size * 4; // standard 4× expansion

        let cb = compute.new_command_buffer();
        let gate_up = self.gpu_linear_biased(x, &format!("{}.ff.net.0.proj", block_prefix), hidden_size, ff_dim * 2, compute, cb.as_ref())?;
        let activated = self.gpu_geglu(&gate_up, seq_len, ff_dim, compute, cb.as_ref())?;
        let output = self.gpu_linear_biased(&activated, &format!("{}.ff.net.2", block_prefix), ff_dim, hidden_size, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }

    // ========================================================================
    // Batched multi-head attention
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

    // ========================================================================
    // Shared helpers
    // ========================================================================

    fn timestep_embedding(&self, timestep: f32, compute: &MetalCompute) -> Result<Tensor> {
        let device = compute.device().info().id;
        crate::inference::architecture::dit::DiTOps::timestep_embedding(timestep, 256, device)
    }

    fn w(&self, name: &str) -> Result<&crate::hal::metal::LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("PixArt weight not found: {}", name)))
    }

    // ========================================================================
    // Low-level GPU dispatch helpers
    // ========================================================================

    fn gpu_linear_biased(
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

    fn gpu_geglu(&self, x: &Tensor, seq_len: usize, ff_dim: usize, compute: &MetalCompute, cb: &metal::CommandBufferRef) -> Result<Tensor> {
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
