//! FIBO (Bria AI) Diffusion Transformer architecture.
//!
//! Implements the FIBO 8B image generation model with:
//! - DiT backbone with DimFusion conditioning
//! - SmolLM3-3B as text encoder (not T5/CLIP)
//! - Wan 2.2 VAE for latent decode (16 channels, same as Flux/SD3)
//! - JSON-native structured prompting (~1000 words)
//! - Flow matching scheduler
//!
//! DimFusion conditioning:
//! Instead of standard cross-attention or AdaLN-only conditioning, FIBO uses
//! DimFusion which fuses text embeddings into each transformer block by projecting
//! the LLM hidden states into dimension-wise modulation vectors. The text encoder
//! output is projected to per-block shift/scale/gate parameters that modulate
//! both the attention and MLP paths, similar to AdaLN but with richer conditioning
//! from the full LLM sequence rather than just a pooled vector.

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

/// FIBO model configuration.
#[derive(Debug, Clone)]
pub struct FiboConfig {
    /// Hidden dimension of the DiT (4096).
    pub hidden_size: usize,
    /// Number of attention heads (64).
    pub num_heads: usize,
    /// Dimension per attention head (64).
    pub head_dim: usize,
    /// Number of DiT blocks (32).
    pub num_layers: usize,
    /// SmolLM3 text encoder hidden dimension (3072).
    pub text_encoder_dim: usize,
    /// DimFusion conditioning dimension (projected from text_encoder_dim to hidden_size).
    pub cond_dim: usize,
    /// MLP expansion ratio (4.0).
    pub mlp_ratio: f32,
    /// Input latent channels (16, Wan 2.2 VAE).
    pub in_channels: usize,
    /// Patch size (2).
    pub patch_size: usize,
}

impl FiboConfig {
    /// FIBO 8B configuration.
    pub fn v1() -> Self {
        Self {
            hidden_size: 4096,
            num_heads: 64,
            head_dim: 64,
            num_layers: 32,
            text_encoder_dim: 3072,
            cond_dim: 4096,
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
struct FiboKernels {
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
impl FiboKernels {
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
// FIBO GPU Transformer
// ============================================================================

/// FIBO GPU transformer -- full forward pass on Metal.
///
/// FIBO uses a DiT backbone with DimFusion conditioning from SmolLM3-3B.
/// The text encoder produces sequence embeddings that are:
/// 1. Mean-pooled to a single vector
/// 2. Projected to per-block modulation parameters (shift/scale/gate)
/// 3. Applied via DimFusion at each transformer block
///
/// The image tokens go through self-attention (no cross-attention to text),
/// with text conditioning applied purely through DimFusion modulation.
#[cfg(feature = "metal")]
pub struct FiboGpuTransformer {
    model: Arc<parking_lot::RwLock<Model>>,
    config: FiboConfig,
    kernels: FiboKernels,
}

#[cfg(feature = "metal")]
impl FiboGpuTransformer {
    /// Create a new FIBO GPU transformer.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: FiboConfig, compute: &Arc<MetalCompute>) -> Result<Self> {
        let kernels = FiboKernels::new(compute)?;
        Ok(Self { model, config, kernels })
    }

    /// Full forward pass.
    ///
    /// `latents`: [1, 16, H, W] noisy latent (Wan 2.2 VAE, 16 channels)
    /// `text_embeds`: [1, txt_seq, 3072] SmolLM3 text encoder hidden states
    /// `timestep`: scalar (0.0 -> 1.0, flow matching)
    pub fn forward(
        &self,
        latents: &Tensor,
        text_embeds: &Tensor,
        timestep: f32,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let (_batch, channels, height, width) = latents.shape().dims4()
            .ok_or_else(|| crate::core::Error::internal("latents must be [B, C, H, W]"))?;
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let mlp_dim = (hidden as f32 * self.config.mlp_ratio) as usize;

        // 1. Timestep embedding -> conditioning vector
        let temb = self.timestep_embedding(timestep, compute)?;
        let cb0 = compute.new_command_buffer();
        let time_proj = self.gpu_mlp_2layer(&temb, "time_embed", 256, hidden, compute, cb0.as_ref())?;
        cb0.commit();
        cb0.wait_until_completed();

        // 2. DimFusion: project text encoder output to conditioning
        // Mean-pool SmolLM3 sequence -> [1, 3072] -> project to hidden
        let txt_seq_len = text_embeds.shape().dim(1).unwrap_or(256);
        let txt_flat = text_embeds.reshape(Shape::from([txt_seq_len, self.config.text_encoder_dim]))?;

        // Mean pool across sequence dimension
        let txt_pooled = self.mean_pool(&txt_flat, txt_seq_len, self.config.text_encoder_dim)?;

        // Project pooled text to conditioning dimension: 3072 -> hidden
        let cb_cond = compute.new_command_buffer();
        let txt_proj = self.gpu_linear(&txt_pooled, "text_projection.linear_1", self.config.text_encoder_dim, hidden, compute, cb_cond.as_ref())?;
        let txt_act = self.gpu_silu(&txt_proj, compute, cb_cond.as_ref())?;
        let txt_cond = self.gpu_linear(&txt_act, "text_projection.linear_2", hidden, hidden, compute, cb_cond.as_ref())?;
        cb_cond.commit();
        cb_cond.wait_until_completed();

        // Combined conditioning: timestep + text
        let cb_vec = compute.new_command_buffer();
        let vec = self.gpu_add(&time_proj, &txt_cond, compute, cb_vec.as_ref())?;
        cb_vec.commit();
        cb_vec.wait_until_completed();

        // 3. DimFusion also provides cross-attention context from full sequence
        // Project full text sequence for cross-attention: [txt_seq, 3072] -> [txt_seq, hidden]
        let cb_ctx = compute.new_command_buffer();
        let context = self.gpu_linear(&txt_flat, "context_embedder", self.config.text_encoder_dim, hidden, compute, cb_ctx.as_ref())?;
        cb_ctx.commit();
        cb_ctx.wait_until_completed();

        // 4. Patchify + project image tokens
        let num_patches = (height / 2) * (width / 2);
        let patch_dim = channels * 4; // 16 * 4 = 64

        let cb2 = compute.new_command_buffer();
        let patches = self.gpu_patchify(latents, channels, height, width, compute, cb2.as_ref())?;
        let img = self.gpu_linear(&patches, "pos_embed.proj", patch_dim, hidden, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        let img_seq_len = num_patches;

        // 5. DiT blocks with DimFusion conditioning
        let mut hidden_state = img;

        self.model.read().prefetch_prefix("blocks.0.");
        for i in 0..self.config.num_layers {
            if i + 1 < self.config.num_layers {
                self.model.read().prefetch_prefix(&format!("blocks.{}.", i + 1));
            } else {
                self.model.read().prefetch_prefix("final_layer.");
            }
            hidden_state = self.dimfusion_block(
                &hidden_state, &context, &vec, i,
                img_seq_len, txt_seq_len,
                hidden, num_heads, head_dim, mlp_dim,
                compute,
            )?;
            self.model.read().evict_prefix(&format!("blocks.{}.", i));
        }

        // 6. Final layer: AdaLN modulation -> linear -> unpatchify
        let (final_shift, final_scale, _final_gate) = self.adaln_3params(
            &vec, "final_layer.adaLN_modulation.1", hidden, compute,
        )?;

        let cb_final = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(&hidden_state, hidden, compute, cb_final.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &final_scale, &final_shift, hidden, compute, cb_final.as_ref())?;
        let output_patches = self.gpu_linear(&modulated, "final_layer.linear", hidden, patch_dim, compute, cb_final.as_ref())?;
        let output = self.gpu_unpatchify(&output_patches, channels, height, width, compute, cb_final.as_ref())?;
        cb_final.commit();
        cb_final.wait_until_completed();

        Ok(output.reshape(Shape::from([1, channels, height, width]))?)
    }

    // ========================================================================
    // DimFusion Block
    // ========================================================================

    /// DiT block with DimFusion conditioning.
    ///
    /// Each block:
    /// 1. AdaLN modulation from DimFusion vec (timestep + pooled text)
    /// 2. Self-attention on image tokens
    /// 3. Cross-attention from image tokens to projected text sequence
    /// 4. AdaLN-modulated MLP
    fn dimfusion_block(
        &self,
        img: &Tensor,
        context: &Tensor,
        vec: &Tensor,
        block_idx: usize,
        img_seq: usize,
        txt_seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        compute: &MetalCompute,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{}", block_idx);

        // DimFusion modulation: 6 params for self-attn shift/scale/gate + mlp shift/scale/gate
        let (shift_attn, scale_attn, gate_attn,
             shift_mlp, scale_mlp, gate_mlp) =
            self.adaln_6params(vec, &format!("{}.adaLN_modulation.1", prefix), hidden, compute)?;

        // Self-attention with AdaLN
        let cb = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(img, hidden, compute, cb.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &scale_attn, &shift_attn, hidden, compute, cb.as_ref())?;

        // Self-attention QKV
        let qkv = self.gpu_linear(&modulated, &format!("{}.attn.qkv", prefix), hidden, 3 * hidden, compute, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Split QKV
        let qkv_f32 = qkv.to_f32_vec()?;
        let device = qkv.device();
        let q_f16: Vec<half::f16> = qkv_f32[..img_seq * hidden].iter().map(|&v| half::f16::from_f32(v)).collect();
        let k_f16: Vec<half::f16> = qkv_f32[img_seq * hidden..img_seq * 2 * hidden].iter().map(|&v| half::f16::from_f32(v)).collect();
        let v_f16: Vec<half::f16> = qkv_f32[img_seq * 2 * hidden..].iter().map(|&v| half::f16::from_f32(v)).collect();

        let q = Tensor::from_slice(&q_f16, Shape::from([img_seq, hidden]), DType::F16, device)?;
        let k = Tensor::from_slice(&k_f16, Shape::from([img_seq, hidden]), DType::F16, device)?;
        let v = Tensor::from_slice(&v_f16, Shape::from([img_seq, hidden]), DType::F16, device)?;

        // Self-attention
        let self_attn_out = self.batched_attention(&q, &k, &v, img_seq, img_seq, num_heads, head_dim, compute)?;

        // Self-attention output projection + gated residual
        let cb2 = compute.new_command_buffer();
        let self_attn_proj = self.gpu_linear(&self_attn_out, &format!("{}.attn.proj", prefix), hidden, hidden, compute, cb2.as_ref())?;
        let after_self_attn = self.gpu_adaln_gate(img, &self_attn_proj, &gate_attn, hidden, compute, cb2.as_ref())?;
        cb2.commit();
        cb2.wait_until_completed();

        // Cross-attention to text context (DimFusion sequence conditioning)
        let cb3 = compute.new_command_buffer();
        let cross_normed = self.gpu_layer_norm(&after_self_attn, hidden, compute, cb3.as_ref())?;
        let cross_q = self.gpu_linear(&cross_normed, &format!("{}.cross_attn.q", prefix), hidden, hidden, compute, cb3.as_ref())?;
        let cross_k = self.gpu_linear(context, &format!("{}.cross_attn.k", prefix), hidden, hidden, compute, cb3.as_ref())?;
        let cross_v = self.gpu_linear(context, &format!("{}.cross_attn.v", prefix), hidden, hidden, compute, cb3.as_ref())?;
        cb3.commit();
        cb3.wait_until_completed();

        let cross_attn_out = self.batched_attention(&cross_q, &cross_k, &cross_v, img_seq, txt_seq, num_heads, head_dim, compute)?;

        let cb4 = compute.new_command_buffer();
        let cross_proj = self.gpu_linear(&cross_attn_out, &format!("{}.cross_attn.proj", prefix), hidden, hidden, compute, cb4.as_ref())?;
        let after_cross_attn = self.gpu_add(&after_self_attn, &cross_proj, compute, cb4.as_ref())?;
        cb4.commit();
        cb4.wait_until_completed();

        // MLP with AdaLN modulation
        let cb5 = compute.new_command_buffer();
        let mlp_normed = self.gpu_layer_norm(&after_cross_attn, hidden, compute, cb5.as_ref())?;
        let mlp_mod = self.gpu_adaln_modulate(&mlp_normed, &scale_mlp, &shift_mlp, hidden, compute, cb5.as_ref())?;
        let mlp_h = self.gpu_linear(&mlp_mod, &format!("{}.mlp.0", prefix), hidden, mlp_dim, compute, cb5.as_ref())?;
        let mlp_act = self.gpu_gelu(&mlp_h, compute, cb5.as_ref())?;
        let mlp_out = self.gpu_linear(&mlp_act, &format!("{}.mlp.2", prefix), mlp_dim, hidden, compute, cb5.as_ref())?;
        let output = self.gpu_adaln_gate(&after_cross_attn, &mlp_out, &gate_mlp, hidden, compute, cb5.as_ref())?;
        cb5.commit();
        cb5.wait_until_completed();

        Ok(output)
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Mean-pool a tensor along the sequence dimension: [seq, dim] -> [1, dim].
    fn mean_pool(&self, x: &Tensor, seq_len: usize, dim: usize) -> Result<Tensor> {
        let data = x.to_f32_vec()?;
        let mut pooled = vec![0.0f32; dim];
        let inv_seq = 1.0 / seq_len as f32;
        for s in 0..seq_len {
            for d in 0..dim {
                pooled[d] += data[s * dim + d] * inv_seq;
            }
        }
        let pooled_f16: Vec<half::f16> = pooled.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&pooled_f16, Shape::from([1, dim]), DType::F16, x.device())
    }

    fn timestep_embedding(&self, timestep: f32, compute: &MetalCompute) -> Result<Tensor> {
        let device = compute.device().info().id;
        crate::inference::architecture::dit::DiTOps::timestep_embedding(timestep, 256, device)
    }

    fn gpu_mlp_2layer(
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

    fn w(&self, name: &str) -> Result<&crate::hal::metal::LazyTensor> {
        self.model.read().get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("FIBO weight not found: {}", name)))
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

#[cfg(feature = "metal")]
fn borrow_tensor(t: &Tensor) -> Result<BorrowedMetalBuffer> {
    t.device_ptr()
        .map(|ptr| unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
        .ok_or_else(|| crate::core::Error::internal("tensor has no device pointer"))
}
