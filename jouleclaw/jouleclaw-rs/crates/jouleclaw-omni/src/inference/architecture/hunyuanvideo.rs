//! HunyuanVideo: Dual-stream DiT for text-to-video generation.
//!
//! Architecture (HunyuanVideoTransformer3DModel):
//!   Text conditioning: LLaMA-3-8B encoder (4096-dim) + CLIP-L (768-dim)
//!   Video latents: 3D VAE encoder → [B, C=16, T, H, W]
//!
//!   Transformer:
//!     - 20 double-stream blocks (cross-attention between text + video tokens)
//!     - 40 single-stream blocks (joint self-attention)
//!     - 2 token refiner layers (pre-processes text embeddings)
//!     - RoPE 3D: temporal + spatial (rope_axes_dim: [16, 56, 56])
//!     - QK-norm: RMSNorm on Q and K per head
//!     - Patch size: 2×2 spatial, 1 temporal
//!
//!   Weight naming:
//!     Double blocks: transformer_blocks.{i}.attn.to_q/to_k/to_v, .add_q_proj/add_k_proj/add_v_proj
//!     Single blocks: single_transformer_blocks.{i}.attn.to_q/to_k/to_v, .proj_mlp, .proj_out
//!     Embedders: x_embedder.proj, context_embedder.proj_in, time_text_embed.*
//!     Token refiner: context_embedder.token_refiner.refiner_blocks.{i}.*
//!     Output: norm_out.linear, proj_out

use crate::core::{Error, Result};
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

/// HunyuanVideo transformer configuration.
#[derive(Debug, Clone)]
pub struct HunyuanVideoConfig {
    /// Number of input/output latent channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Attention head dimension.
    pub attention_head_dim: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of double-stream (cross-attn) layers.
    pub num_layers: usize,
    /// Number of single-stream (joint) layers.
    pub num_single_layers: usize,
    /// Number of refiner layers.
    pub num_refiner_layers: usize,
    /// MLP ratio (FFN expansion).
    pub mlp_ratio: f32,
    /// Spatial patch size.
    pub patch_size: usize,
    /// Temporal patch size.
    pub patch_size_t: usize,
    /// Pooled projection dimension (from CLIP).
    pub pooled_projection_dim: usize,
    /// Text embedding dimension (from LLaMA).
    pub text_embed_dim: usize,
    /// RoPE axes dimensions [temporal, height, width].
    pub rope_axes_dim: [usize; 3],
    /// RoPE theta.
    pub rope_theta: f32,
    /// Whether to use guidance embeddings.
    pub guidance_embeds: bool,
}

impl Default for HunyuanVideoConfig {
    fn default() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            attention_head_dim: 128,
            num_attention_heads: 24,
            num_layers: 20,
            num_single_layers: 40,
            num_refiner_layers: 2,
            mlp_ratio: 4.0,
            patch_size: 2,
            patch_size_t: 1,
            pooled_projection_dim: 768,
            text_embed_dim: 4096,
            rope_axes_dim: [16, 56, 56],
            rope_theta: 256.0,
            guidance_embeds: true,
        }
    }
}

impl HunyuanVideoConfig {
    /// Parse from transformer config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();
        if let Some(v) = json.get("in_channels").and_then(|v| v.as_u64()) { c.in_channels = v as usize; }
        if let Some(v) = json.get("out_channels").and_then(|v| v.as_u64()) { c.out_channels = v as usize; }
        if let Some(v) = json.get("attention_head_dim").and_then(|v| v.as_u64()) { c.attention_head_dim = v as usize; }
        if let Some(v) = json.get("num_attention_heads").and_then(|v| v.as_u64()) { c.num_attention_heads = v as usize; }
        if let Some(v) = json.get("num_layers").and_then(|v| v.as_u64()) { c.num_layers = v as usize; }
        if let Some(v) = json.get("num_single_layers").and_then(|v| v.as_u64()) { c.num_single_layers = v as usize; }
        if let Some(v) = json.get("num_refiner_layers").and_then(|v| v.as_u64()) { c.num_refiner_layers = v as usize; }
        if let Some(v) = json.get("mlp_ratio").and_then(|v| v.as_f64()) { c.mlp_ratio = v as f32; }
        if let Some(v) = json.get("patch_size").and_then(|v| v.as_u64()) { c.patch_size = v as usize; }
        if let Some(v) = json.get("patch_size_t").and_then(|v| v.as_u64()) { c.patch_size_t = v as usize; }
        if let Some(v) = json.get("pooled_projection_dim").and_then(|v| v.as_u64()) { c.pooled_projection_dim = v as usize; }
        if let Some(v) = json.get("text_embed_dim").and_then(|v| v.as_u64()) { c.text_embed_dim = v as usize; }
        if let Some(v) = json.get("rope_theta").and_then(|v| v.as_f64()) { c.rope_theta = v as f32; }
        if let Some(v) = json.get("guidance_embeds").and_then(|v| v.as_bool()) { c.guidance_embeds = v; }
        if let Some(arr) = json.get("rope_axes_dim").and_then(|v| v.as_array()) {
            if arr.len() == 3 {
                c.rope_axes_dim = [
                    arr[0].as_u64().unwrap_or(16) as usize,
                    arr[1].as_u64().unwrap_or(56) as usize,
                    arr[2].as_u64().unwrap_or(56) as usize,
                ];
            }
        }
        Ok(c)
    }

    /// Hidden dimension = num_heads × head_dim.
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// FFN intermediate dimension.
    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size() as f32 * self.mlp_ratio) as usize
    }
}

// ============================================================================
// GPU Pipeline
// ============================================================================

#[cfg(feature = "metal")]
struct HVKernels {
    linear: Arc<crate::hal::metal::ComputePipeline>,
    silu: Arc<crate::hal::metal::ComputePipeline>,
    add: Arc<crate::hal::metal::ComputePipeline>,
    gelu: Arc<crate::hal::metal::ComputePipeline>,
    layer_norm: Arc<crate::hal::metal::ComputePipeline>,
    adaln_modulate: Arc<crate::hal::metal::ComputePipeline>,
    adaln_gate: Arc<crate::hal::metal::ComputePipeline>,
    batched_linear: Arc<crate::hal::metal::ComputePipeline>,
    batched_matmul_nn: Arc<crate::hal::metal::ComputePipeline>,
    row_softmax: Arc<crate::hal::metal::ComputePipeline>,
    transpose_shd_hsd: Arc<crate::hal::metal::ComputePipeline>,
    transpose_hsd_shd: Arc<crate::hal::metal::ComputePipeline>,
}

#[cfg(feature = "metal")]
impl HVKernels {
    fn new(compute: &Arc<MetalCompute>) -> Result<Self> {
        Ok(Self {
            linear: compute.compile_pipeline("hv_linear", sources::LINEAR, "linear_f16")?,
            silu: compute.compile_pipeline("hv_silu", sources::SILU, "silu_f16")?,
            add: compute.compile_pipeline("hv_add", sources::ELEMENTWISE, "add_f16")?,
            gelu: compute.compile_pipeline("hv_gelu", sources::GELU, "gelu_tanh_f16")?,
            layer_norm: compute.compile_pipeline("hv_ln", sources::LAYER_NORM, "layer_norm_f16")?,
            adaln_modulate: compute.compile_pipeline("hv_adaln_mod", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("hv_adaln_gate", sources::ADALN, "adaln_gate_f16")?,
            batched_linear: compute.compile_pipeline("hv_batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("hv_batched_matmul", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax: compute.compile_pipeline("hv_softmax", sources::LINEAR, "row_softmax_scale_f16")?,
            transpose_shd_hsd: compute.compile_pipeline("hv_t_shd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_shd: compute.compile_pipeline("hv_t_hsd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
        })
    }
}

/// HunyuanVideo transformer pipeline.
#[cfg(feature = "metal")]
pub struct HunyuanVideoPipeline {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: HunyuanVideoConfig,
    kernels: HVKernels,
}

#[cfg(feature = "metal")]
impl HunyuanVideoPipeline {
    /// Create pipeline from loaded transformer model.
    pub fn new(
        transformer_model: Arc<Model>,
        config: HunyuanVideoConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        let kernels = HVKernels::new(&compute)?;
        Ok(Self { model: transformer_model, compute, config, kernels })
    }

    /// Forward pass through the dual-stream + single-stream transformer.
    ///
    /// - `latents`: video latents [C, T, H, W]
    /// - `text_embeds`: LLaMA encoder output [text_seq, 4096]
    /// - `pooled_embeds`: CLIP pooled output [768]
    /// - `timestep`: diffusion timestep (0..1)
    /// - `guidance`: guidance scale
    ///
    /// Returns noise prediction [C, T, H, W].
    pub fn forward(
        &self,
        latents: &Tensor,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        timestep: f32,
        guidance: f32,
    ) -> Result<Tensor> {
        let config = &self.config;
        let hidden = config.hidden_size();       // 3072
        let num_heads = config.num_attention_heads; // 24
        let head_dim = config.attention_head_dim;   // 128
        let mlp_dim = config.intermediate_size();   // 12288
        let compute = &self.compute;
        let device = compute.device().info().id;

        // Parse latent shape: [C, T, H, W]
        let latent_dims = latents.shape().dims();
        let (channels, t_len, h_len, w_len) = if latent_dims.len() == 4 {
            (latent_dims[0], latent_dims[1], latent_dims[2], latent_dims[3])
        } else {
            return Err(Error::internal("latents must be [C, T, H, W]"));
        };
        let ph = h_len / config.patch_size;
        let pw = w_len / config.patch_size;
        let img_seq = t_len * ph * pw;
        let patch_dim = channels * config.patch_size_t * config.patch_size * config.patch_size; // 16*1*2*2=64

        let text_seq = text_embeds.shape().dim(0).unwrap_or(64);

        // ================================================================
        // 1. Timestep + guidance + CLIP text embeddings → temb [1, hidden]
        // ================================================================
        let temb_sinusoidal = self.timestep_embedding(timestep, device)?;
        let cb = compute.new_command_buffer();
        let time_proj = self.gpu_mlp_biased(
            &temb_sinusoidal, "time_text_embed.timestep_embedder", 256, hidden, cb.as_ref(),
        )?;
        let text_proj = self.gpu_mlp_biased(
            pooled_embeds, "time_text_embed.text_embedder",
            config.pooled_projection_dim, hidden, cb.as_ref(),
        )?;
        let temb = self.gpu_add(&time_proj, &text_proj, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let temb = if config.guidance_embeds {
            let gemb = self.timestep_embedding(guidance * 1000.0, device)?;
            let cb = compute.new_command_buffer();
            let guid_proj = self.gpu_mlp_biased(
                &gemb, "time_text_embed.guidance_embedder", 256, hidden, cb.as_ref(),
            )?;
            let result = self.gpu_add(&temb, &guid_proj, cb.as_ref())?;
            cb.commit();
            cb.wait_until_completed();
            result
        } else {
            temb
        };

        // ================================================================
        // 2. Patchify video latents + project
        // ================================================================
        // Conv3D patchify: [C, T, H, W] → [T*H/2*W/2, 64]
        let patches = self.patchify_3d_cpu(latents, channels, t_len, h_len, w_len)?;
        // Linear: [N, 64] → [N, 3072]
        let cb = compute.new_command_buffer();
        let img = self.gpu_linear_biased_prefix(&patches, "x_embedder.proj", patch_dim, hidden, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // ================================================================
        // 3. Project text: [text_seq, 4096] → [text_seq, 3072]
        // ================================================================
        let cb = compute.new_command_buffer();
        let txt = self.gpu_linear_biased_prefix(
            text_embeds, "context_embedder.proj_in", config.text_embed_dim, hidden, cb.as_ref(),
        )?;
        cb.commit();
        cb.wait_until_completed();

        // ================================================================
        // 4. Token refiner (2 blocks) — refines text embeddings
        // ================================================================
        let txt = self.token_refiner(&txt, text_seq, &temb, hidden, num_heads, head_dim, mlp_dim)?;

        // ================================================================
        // 5. Precompute 3D RoPE cos/sin tables for video tokens
        // ================================================================
        let (rope_cos, rope_sin) = self.compute_3d_rope(t_len, ph, pw, head_dim, device)?;

        // ================================================================
        // 6. Double-stream blocks (20 layers)
        // ================================================================
        let mut img = img;
        let mut txt = txt;

        self.model.prefetch_prefix("transformer_blocks.0.");
        for i in 0..config.num_layers {
            if i + 1 < config.num_layers {
                self.model.prefetch_prefix(&format!("transformer_blocks.{}.", i + 1));
            } else {
                self.model.prefetch_prefix("single_transformer_blocks.0.");
            }
            let (new_img, new_txt) = self.double_block(
                &img, &txt, &temb, i,
                img_seq, text_seq, hidden, num_heads, head_dim, mlp_dim,
                &rope_cos, &rope_sin,
            )?;
            self.model.evict_prefix(&format!("transformer_blocks.{}.", i));
            img = new_img;
            txt = new_txt;
        }

        // ================================================================
        // 7. Single-stream blocks (40 layers)
        // ================================================================
        let mut combined = Tensor::cat(&[txt, img], 0)?;
        let total_seq = text_seq + img_seq;

        for i in 0..config.num_single_layers {
            if i + 1 < config.num_single_layers {
                self.model.prefetch_prefix(&format!("single_transformer_blocks.{}.", i + 1));
            }
            combined = self.single_block(
                &combined, &temb, i,
                total_seq, text_seq, img_seq, hidden, num_heads, head_dim, mlp_dim,
                &rope_cos, &rope_sin,
            )?;
            self.model.evict_prefix(&format!("single_transformer_blocks.{}.", i));
        }

        // ================================================================
        // 8. Extract video tokens (after text tokens)
        // ================================================================
        let img_out = combined.slice(0, text_seq, total_seq)?;

        // ================================================================
        // 9. Output: AdaLN modulate → linear → unpatchify
        // ================================================================
        // norm_out.linear: [6144, 3072] → 2 × hidden (shift, scale)
        let (out_shift, out_scale) = self.adaln_2params(&temb, "norm_out.linear", hidden)?;

        let cb = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(&img_out, hidden, cb.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &out_scale, &out_shift, hidden, cb.as_ref())?;
        let output_patches = self.gpu_linear_biased_prefix(&modulated, "proj_out", hidden, patch_dim, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Unpatchify: [N, 64] → [C, T, H, W]
        self.unpatchify_3d_cpu(&output_patches, channels, t_len, h_len, w_len)
    }

    // ========================================================================
    // Token Refiner (2 blocks)
    // ========================================================================

    fn token_refiner(
        &self,
        txt: &Tensor,
        text_seq: usize,
        temb: &Tensor,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
    ) -> Result<Tensor> {
        let compute = &self.compute;

        // Refiner's own time+text embedding (uses same main temb for simplicity)
        // In full impl, refiner has context_embedder.time_text_embed but uses
        // the same timestep and a 4096-dim text pooled embedding.
        // For now, pass the main temb directly.
        let ref_temb = temb;

        let mut x = txt.clone();
        for i in 0..self.config.num_refiner_layers {
            let prefix = format!("context_embedder.token_refiner.refiner_blocks.{}", i);

            // AdaLN modulation: norm_out.linear → [6144, 3072] → 2 × hidden (shift, scale)
            let (shift, scale) = self.adaln_2params(
                ref_temb, &format!("{}.norm_out.linear", prefix), hidden,
            )?;

            // Pre-norm (LayerNorm with learned weight+bias)
            let x_normed = self.layer_norm_affine_cpu(
                &x, &format!("{}.norm1", prefix), hidden,
            )?;

            // Self-attention
            let cb = compute.new_command_buffer();
            let q = self.gpu_linear_biased_prefix(&x_normed, &format!("{}.attn.to_q", prefix), hidden, hidden, cb.as_ref())?;
            let k = self.gpu_linear_biased_prefix(&x_normed, &format!("{}.attn.to_k", prefix), hidden, hidden, cb.as_ref())?;
            let v = self.gpu_linear_biased_prefix(&x_normed, &format!("{}.attn.to_v", prefix), hidden, hidden, cb.as_ref())?;
            cb.commit();
            cb.wait_until_completed();

            let attn_out = self.batched_attention(&q, &k, &v, text_seq, text_seq, num_heads, head_dim)?;

            let cb = compute.new_command_buffer();
            let attn_proj = self.gpu_linear_biased_prefix(&attn_out, &format!("{}.attn.to_out.0", prefix), hidden, hidden, cb.as_ref())?;
            let x_after_attn = self.gpu_add(&x, &attn_proj, cb.as_ref())?;
            cb.commit();
            cb.wait_until_completed();

            // FFN: norm2 → GELU FFN
            let x_normed2 = self.layer_norm_affine_cpu(
                &x_after_attn, &format!("{}.norm2", prefix), hidden,
            )?;

            let cb = compute.new_command_buffer();
            let ff_up = self.gpu_linear_biased_prefix(&x_normed2, &format!("{}.ff.net.0.proj", prefix), hidden, mlp_dim, cb.as_ref())?;
            let ff_act = self.gpu_gelu(&ff_up, cb.as_ref())?;
            let ff_down = self.gpu_linear_biased_prefix(&ff_act, &format!("{}.ff.net.2", prefix), mlp_dim, hidden, cb.as_ref())?;
            let x_after_ff = self.gpu_add(&x_after_attn, &ff_down, cb.as_ref())?;
            cb.commit();
            cb.wait_until_completed();

            // Apply AdaLN modulation (scale + shift)
            let cb = compute.new_command_buffer();
            let normed_out = self.gpu_layer_norm(&x_after_ff, hidden, cb.as_ref())?;
            x = self.gpu_adaln_modulate(&normed_out, &scale, &shift, hidden, cb.as_ref())?;
            cb.commit();
            cb.wait_until_completed();
        }

        Ok(x)
    }

    // ========================================================================
    // Double Block (dual-stream)
    // ========================================================================

    fn double_block(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        block_idx: usize,
        img_seq: usize,
        txt_seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let compute = &self.compute;
        let prefix = format!("transformer_blocks.{}", block_idx);

        // AdaLN: 6 modulation params each for img and txt
        // norm1.linear: [18432, 3072] → 6 × hidden
        let (img_shift_a, img_scale_a, img_gate_a,
             img_shift_f, img_scale_f, img_gate_f) =
            self.adaln_6params(temb, &format!("{}.norm1.linear", prefix), hidden)?;
        let (txt_shift_a, txt_scale_a, txt_gate_a,
             txt_shift_f, txt_scale_f, txt_gate_f) =
            self.adaln_6params(temb, &format!("{}.norm1_context.linear", prefix), hidden)?;

        // LayerNorm + AdaLN modulate
        let cb = compute.new_command_buffer();
        let img_normed = self.gpu_layer_norm(img, hidden, cb.as_ref())?;
        let img_mod = self.gpu_adaln_modulate(&img_normed, &img_scale_a, &img_shift_a, hidden, cb.as_ref())?;
        let txt_normed = self.gpu_layer_norm(txt, hidden, cb.as_ref())?;
        let txt_mod = self.gpu_adaln_modulate(&txt_normed, &txt_scale_a, &txt_shift_a, hidden, cb.as_ref())?;

        // Separate Q/K/V projections
        let img_q = self.gpu_linear_biased_prefix(&img_mod, &format!("{}.attn.to_q", prefix), hidden, hidden, cb.as_ref())?;
        let img_k = self.gpu_linear_biased_prefix(&img_mod, &format!("{}.attn.to_k", prefix), hidden, hidden, cb.as_ref())?;
        let img_v = self.gpu_linear_biased_prefix(&img_mod, &format!("{}.attn.to_v", prefix), hidden, hidden, cb.as_ref())?;
        let txt_q = self.gpu_linear_biased_prefix(&txt_mod, &format!("{}.attn.add_q_proj", prefix), hidden, hidden, cb.as_ref())?;
        let txt_k = self.gpu_linear_biased_prefix(&txt_mod, &format!("{}.attn.add_k_proj", prefix), hidden, hidden, cb.as_ref())?;
        let txt_v = self.gpu_linear_biased_prefix(&txt_mod, &format!("{}.attn.add_v_proj", prefix), hidden, hidden, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // QK-norm (per-head RMSNorm with learned scale)
        let img_q = self.apply_qk_norm_single(&img_q, img_seq, num_heads, head_dim, &format!("{}.attn.norm_q", prefix))?;
        let img_k = self.apply_qk_norm_single(&img_k, img_seq, num_heads, head_dim, &format!("{}.attn.norm_k", prefix))?;
        let txt_q = self.apply_qk_norm_single(&txt_q, txt_seq, num_heads, head_dim, &format!("{}.attn.norm_added_q", prefix))?;
        let txt_k = self.apply_qk_norm_single(&txt_k, txt_seq, num_heads, head_dim, &format!("{}.attn.norm_added_k", prefix))?;

        // Apply 3D RoPE to image Q/K
        let img_q = self.apply_rope_cpu(&img_q, rope_cos, rope_sin, img_seq, num_heads, head_dim)?;
        let img_k = self.apply_rope_cpu(&img_k, rope_cos, rope_sin, img_seq, num_heads, head_dim)?;

        // Concatenate K/V for joint attention: [txt_seq + img_seq, hidden]
        let joint_k = Tensor::cat(&[txt_k, img_k], 0)?;
        let joint_v = Tensor::cat(&[txt_v, img_v], 0)?;
        let total_kv = txt_seq + img_seq;

        // Attention for img queries and txt queries
        let img_attn = self.batched_attention(&img_q, &joint_k, &joint_v, img_seq, total_kv, num_heads, head_dim)?;
        let txt_attn = self.batched_attention(&txt_q, &joint_k, &joint_v, txt_seq, total_kv, num_heads, head_dim)?;

        // Output projection + gated residual
        let cb = compute.new_command_buffer();
        let img_proj = self.gpu_linear_biased_prefix(&img_attn, &format!("{}.attn.to_out.0", prefix), hidden, hidden, cb.as_ref())?;
        let txt_proj = self.gpu_linear_biased_prefix(&txt_attn, &format!("{}.attn.to_add_out", prefix), hidden, hidden, cb.as_ref())?;
        let img_after_attn = self.gpu_adaln_gate(img, &img_proj, &img_gate_a, hidden, cb.as_ref())?;
        let txt_after_attn = self.gpu_adaln_gate(txt, &txt_proj, &txt_gate_a, hidden, cb.as_ref())?;

        // FFN: LayerNorm + AdaLN modulate + GELU FFN + gated residual
        let img_ff_normed = self.gpu_layer_norm(&img_after_attn, hidden, cb.as_ref())?;
        let img_ff_mod = self.gpu_adaln_modulate(&img_ff_normed, &img_scale_f, &img_shift_f, hidden, cb.as_ref())?;
        let txt_ff_normed = self.gpu_layer_norm(&txt_after_attn, hidden, cb.as_ref())?;
        let txt_ff_mod = self.gpu_adaln_modulate(&txt_ff_normed, &txt_scale_f, &txt_shift_f, hidden, cb.as_ref())?;

        // FFN: linear → GELU → linear
        let img_ff_up = self.gpu_linear_biased_prefix(&img_ff_mod, &format!("{}.ff.net.0.proj", prefix), hidden, mlp_dim, cb.as_ref())?;
        let img_ff_act = self.gpu_gelu(&img_ff_up, cb.as_ref())?;
        let img_ff_down = self.gpu_linear_biased_prefix(&img_ff_act, &format!("{}.ff.net.2", prefix), mlp_dim, hidden, cb.as_ref())?;
        let txt_ff_up = self.gpu_linear_biased_prefix(&txt_ff_mod, &format!("{}.ff_context.net.0.proj", prefix), hidden, mlp_dim, cb.as_ref())?;
        let txt_ff_act = self.gpu_gelu(&txt_ff_up, cb.as_ref())?;
        let txt_ff_down = self.gpu_linear_biased_prefix(&txt_ff_act, &format!("{}.ff_context.net.2", prefix), mlp_dim, hidden, cb.as_ref())?;

        // Gated residual for FFN
        let img_out = self.gpu_adaln_gate(&img_after_attn, &img_ff_down, &img_gate_f, hidden, cb.as_ref())?;
        let txt_out = self.gpu_adaln_gate(&txt_after_attn, &txt_ff_down, &txt_gate_f, hidden, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        Ok((img_out, txt_out))
    }

    // ========================================================================
    // Single Block (fused stream)
    // ========================================================================

    fn single_block(
        &self,
        hidden_state: &Tensor,
        temb: &Tensor,
        block_idx: usize,
        total_seq: usize,
        txt_seq: usize,
        img_seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let compute = &self.compute;
        let prefix = format!("single_transformer_blocks.{}", block_idx);

        // norm.linear: [9216, 3072] → 3 × hidden (shift, scale, gate)
        let (shift, scale, gate) = self.adaln_3params(
            temb, &format!("{}.norm.linear", prefix), hidden,
        )?;

        // LayerNorm + AdaLN modulate
        let cb = compute.new_command_buffer();
        let normed = self.gpu_layer_norm(hidden_state, hidden, cb.as_ref())?;
        let modulated = self.gpu_adaln_modulate(&normed, &scale, &shift, hidden, cb.as_ref())?;

        // Separate Q/K/V
        let q = self.gpu_linear_biased_prefix(&modulated, &format!("{}.attn.to_q", prefix), hidden, hidden, cb.as_ref())?;
        let k = self.gpu_linear_biased_prefix(&modulated, &format!("{}.attn.to_k", prefix), hidden, hidden, cb.as_ref())?;
        let v = self.gpu_linear_biased_prefix(&modulated, &format!("{}.attn.to_v", prefix), hidden, hidden, cb.as_ref())?;
        // Parallel MLP: linear → GELU
        let mlp_up = self.gpu_linear_biased_prefix(&modulated, &format!("{}.proj_mlp", prefix), hidden, mlp_dim, cb.as_ref())?;
        let mlp_act = self.gpu_gelu(&mlp_up, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // QK-norm
        let q = self.apply_qk_norm_single(&q, total_seq, num_heads, head_dim, &format!("{}.attn.norm_q", prefix))?;
        let k = self.apply_qk_norm_single(&k, total_seq, num_heads, head_dim, &format!("{}.attn.norm_k", prefix))?;

        // Apply 3D RoPE to image portion of Q/K (last img_seq tokens)
        let q = self.apply_rope_partial_cpu(&q, rope_cos, rope_sin, total_seq, txt_seq, img_seq, num_heads, head_dim)?;
        let k = self.apply_rope_partial_cpu(&k, rope_cos, rope_sin, total_seq, txt_seq, img_seq, num_heads, head_dim)?;

        // Self-attention
        let attn_out = self.batched_attention(&q, &k, &v, total_seq, total_seq, num_heads, head_dim)?;

        // Concat attn output + MLP: [total_seq, hidden + mlp_dim]
        let concat = Tensor::cat(&[attn_out, mlp_act], 1)?;
        let concat_dim = hidden + mlp_dim; // 3072 + 12288 = 15360

        // proj_out: [3072, 15360] → [total_seq, 3072]
        let cb = compute.new_command_buffer();
        let residual = self.gpu_linear_biased_prefix(&concat, &format!("{}.proj_out", prefix), concat_dim, hidden, cb.as_ref())?;
        let output = self.gpu_adaln_gate(hidden_state, &residual, &gate, hidden, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        Ok(output)
    }

    // ========================================================================
    // 3D Patchify / Unpatchify (CPU)
    // ========================================================================

    /// Patchify 3D latent: [C, T, H, W] → [T*H/2*W/2, C*patch_t*patch_h*patch_w]
    fn patchify_3d_cpu(
        &self, latent: &Tensor, c: usize, t: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let ph = h / self.config.patch_size;
        let pw = w / self.config.patch_size;
        let pt = self.config.patch_size_t;
        let ps = self.config.patch_size;
        let patch_dim = c * pt * ps * ps;
        let num_patches = t * ph * pw;

        let data = latent.to_f32_vec()?;
        let mut patches = vec![0.0f32; num_patches * patch_dim];

        for ti in 0..t {
            for hi in 0..ph {
                for wi in 0..pw {
                    let patch_idx = ti * ph * pw + hi * pw + wi;
                    let mut d = 0;
                    for ci in 0..c {
                        for dt in 0..pt {
                            for dh in 0..ps {
                                for dw in 0..ps {
                                    let src_t = ti + dt;
                                    let src_h = hi * ps + dh;
                                    let src_w = wi * ps + dw;
                                    if src_t < t {
                                        let src_idx = ci * t * h * w + src_t * h * w + src_h * w + src_w;
                                        patches[patch_idx * patch_dim + d] = data[src_idx];
                                    }
                                    d += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        let f16_data: Vec<half::f16> = patches.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, Shape::from([num_patches, patch_dim]), DType::F16, latent.device())
    }

    /// Unpatchify: [N, patch_dim] → [C, T, H, W]
    fn unpatchify_3d_cpu(
        &self, patches: &Tensor, c: usize, t: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let ph = h / self.config.patch_size;
        let pw = w / self.config.patch_size;
        let pt = self.config.patch_size_t;
        let ps = self.config.patch_size;
        let patch_dim = c * pt * ps * ps;

        let data = patches.to_f32_vec()?;
        let mut output = vec![0.0f32; c * t * h * w];

        for ti in 0..t {
            for hi in 0..ph {
                for wi in 0..pw {
                    let patch_idx = ti * ph * pw + hi * pw + wi;
                    let mut d = 0;
                    for ci in 0..c {
                        for dt in 0..pt {
                            for dh in 0..ps {
                                for dw in 0..ps {
                                    let dst_t = ti + dt;
                                    let dst_h = hi * ps + dh;
                                    let dst_w = wi * ps + dw;
                                    if dst_t < t {
                                        let dst_idx = ci * t * h * w + dst_t * h * w + dst_h * w + dst_w;
                                        output[dst_idx] = data[patch_idx * patch_dim + d];
                                    }
                                    d += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        let f16_data: Vec<half::f16> = output.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, Shape::from([c, t, h, w]), DType::F16, patches.device())
    }

    // ========================================================================
    // 3D RoPE
    // ========================================================================

    /// Precompute 3D RoPE cos/sin tables for all video token positions.
    /// Returns (cos, sin) each of shape [img_seq, head_dim].
    fn compute_3d_rope(
        &self, t: usize, ph: usize, pw: usize, head_dim: usize, device: crate::hal::DeviceId,
    ) -> Result<(Tensor, Tensor)> {
        let [dim_t, dim_h, dim_w] = self.config.rope_axes_dim;
        let theta = self.config.rope_theta;
        let img_seq = t * ph * pw;

        let mut cos_data = vec![0.0f32; img_seq * head_dim];
        let mut sin_data = vec![0.0f32; img_seq * head_dim];

        for ti in 0..t {
            for hi in 0..ph {
                for wi in 0..pw {
                    let token = ti * ph * pw + hi * pw + wi;
                    let mut d = 0;

                    // Temporal RoPE: dims [0, dim_t)
                    for i in 0..dim_t / 2 {
                        let freq = 1.0 / theta.powf(2.0 * i as f32 / dim_t as f32);
                        let angle = ti as f32 * freq;
                        cos_data[token * head_dim + d] = angle.cos();
                        sin_data[token * head_dim + d] = angle.sin();
                        cos_data[token * head_dim + d + 1] = angle.cos();
                        sin_data[token * head_dim + d + 1] = angle.sin();
                        d += 2;
                    }

                    // Height RoPE: dims [dim_t, dim_t + dim_h)
                    for i in 0..dim_h / 2 {
                        let freq = 1.0 / theta.powf(2.0 * i as f32 / dim_h as f32);
                        let angle = hi as f32 * freq;
                        cos_data[token * head_dim + d] = angle.cos();
                        sin_data[token * head_dim + d] = angle.sin();
                        cos_data[token * head_dim + d + 1] = angle.cos();
                        sin_data[token * head_dim + d + 1] = angle.sin();
                        d += 2;
                    }

                    // Width RoPE: dims [dim_t + dim_h, dim_t + dim_h + dim_w)
                    for i in 0..dim_w / 2 {
                        let freq = 1.0 / theta.powf(2.0 * i as f32 / dim_w as f32);
                        let angle = wi as f32 * freq;
                        cos_data[token * head_dim + d] = angle.cos();
                        sin_data[token * head_dim + d] = angle.sin();
                        cos_data[token * head_dim + d + 1] = angle.cos();
                        sin_data[token * head_dim + d + 1] = angle.sin();
                        d += 2;
                    }
                }
            }
        }

        let shape = Shape::from([img_seq, head_dim]);
        let cos_f16: Vec<half::f16> = cos_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let sin_f16: Vec<half::f16> = sin_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        Ok((
            Tensor::from_slice(&cos_f16, shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&sin_f16, shape, DType::F16, device)?,
        ))
    }

    /// Apply RoPE to all tokens (for double block img Q/K).
    fn apply_rope_cpu(
        &self, x: &Tensor, rope_cos: &Tensor, rope_sin: &Tensor,
        seq_len: usize, num_heads: usize, head_dim: usize,
    ) -> Result<Tensor> {
        let x_f32 = x.to_f32_vec()?;
        let cos_f32 = rope_cos.to_f32_vec()?;
        let sin_f32 = rope_sin.to_f32_vec()?;
        let mut out = x_f32.clone();

        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in (0..head_dim).step_by(2) {
                    let idx = s * num_heads * head_dim + h * head_dim + d;
                    let cos_val = cos_f32[s * head_dim + d];
                    let sin_val = sin_f32[s * head_dim + d];
                    let x0 = x_f32[idx];
                    let x1 = x_f32[idx + 1];
                    out[idx] = x0 * cos_val - x1 * sin_val;
                    out[idx + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }

        let f16: Vec<half::f16> = out.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16, x.shape().clone(), DType::F16, x.device())
    }

    /// Apply RoPE to image portion only (for single block, where text is prepended).
    fn apply_rope_partial_cpu(
        &self, x: &Tensor, rope_cos: &Tensor, rope_sin: &Tensor,
        total_seq: usize, txt_seq: usize, img_seq: usize,
        num_heads: usize, head_dim: usize,
    ) -> Result<Tensor> {
        let x_f32 = x.to_f32_vec()?;
        let cos_f32 = rope_cos.to_f32_vec()?;
        let sin_f32 = rope_sin.to_f32_vec()?;
        let mut out = x_f32.clone();

        // Only apply RoPE to image tokens (positions txt_seq .. total_seq)
        for img_idx in 0..img_seq {
            let s = txt_seq + img_idx;
            for h in 0..num_heads {
                for d in (0..head_dim).step_by(2) {
                    let idx = s * num_heads * head_dim + h * head_dim + d;
                    let cos_val = cos_f32[img_idx * head_dim + d];
                    let sin_val = sin_f32[img_idx * head_dim + d];
                    let x0 = x_f32[idx];
                    let x1 = x_f32[idx + 1];
                    out[idx] = x0 * cos_val - x1 * sin_val;
                    out[idx + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }

        let f16: Vec<half::f16> = out.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16, x.shape().clone(), DType::F16, x.device())
    }

    // ========================================================================
    // QK-Norm
    // ========================================================================

    /// Per-head RMSNorm with learned scale on a single tensor.
    fn apply_qk_norm_single(
        &self, x: &Tensor, seq_len: usize, num_heads: usize, head_dim: usize, weight_name: &str,
    ) -> Result<Tensor> {
        let scale_lt = self.w(&format!("{}.weight", weight_name))?;
        let scale_f32 = scale_lt.to_f32_vec()?;
        let mut x_f32 = x.to_f32_vec()?;

        for token in 0..seq_len {
            for head in 0..num_heads {
                let offset = token * num_heads * head_dim + head * head_dim;
                let slice = &mut x_f32[offset..offset + head_dim];
                // RMSNorm
                let rms = (slice.iter().map(|&v| v * v).sum::<f32>() / head_dim as f32 + 1e-6).sqrt();
                for d in 0..head_dim {
                    slice[d] = slice[d] / rms * scale_f32[d];
                }
            }
        }

        let f16: Vec<half::f16> = x_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16, x.shape().clone(), DType::F16, x.device())
    }

    // ========================================================================
    // LayerNorm with affine (learned weight+bias) on CPU
    // ========================================================================

    fn layer_norm_affine_cpu(&self, x: &Tensor, prefix: &str, hidden: usize) -> Result<Tensor> {
        let w_lt = self.w(&format!("{}.weight", prefix))?;
        let b_lt = self.w(&format!("{}.bias", prefix))?;
        let w_f32 = w_lt.to_f32_vec()?;
        let b_f32 = b_lt.to_f32_vec()?;
        let x_f32 = x.to_f32_vec()?;
        let seq_len = x_f32.len() / hidden;
        let mut out = vec![0.0f32; x_f32.len()];

        for s in 0..seq_len {
            let offset = s * hidden;
            let row = &x_f32[offset..offset + hidden];
            let mean = row.iter().sum::<f32>() / hidden as f32;
            let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let std = (var + 1e-6).sqrt();
            for d in 0..hidden {
                out[offset + d] = (row[d] - mean) / std * w_f32[d] + b_f32[d];
            }
        }

        let f16: Vec<half::f16> = out.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16, x.shape().clone(), DType::F16, x.device())
    }

    // ========================================================================
    // Modulation parameter extraction
    // ========================================================================

    /// Extract 6 modulation parameters from temb via SiLU + linear.
    fn adaln_6params(
        &self, temb: &Tensor, weight_name: &str, hidden: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let compute = &self.compute;
        let cb = compute.new_command_buffer();
        let activated = self.gpu_silu(temb, cb.as_ref())?;
        let params = self.gpu_linear_biased_prefix(&activated, weight_name, hidden, hidden * 6, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let data = params.to_f32_vec()?;
        let device = temb.device();
        let shape = Shape::from([hidden]);
        let to_f16 = |s: usize, e: usize| -> Vec<half::f16> {
            data[s..e].iter().map(|&v| half::f16::from_f32(v)).collect()
        };

        Ok((
            Tensor::from_slice(&to_f16(0, hidden), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden, hidden * 2), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden * 2, hidden * 3), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden * 3, hidden * 4), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden * 4, hidden * 5), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden * 5, hidden * 6), shape.clone(), DType::F16, device)?,
        ))
    }

    /// Extract 3 modulation parameters (shift, scale, gate).
    fn adaln_3params(
        &self, temb: &Tensor, weight_name: &str, hidden: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let compute = &self.compute;
        let cb = compute.new_command_buffer();
        let activated = self.gpu_silu(temb, cb.as_ref())?;
        let params = self.gpu_linear_biased_prefix(&activated, weight_name, hidden, hidden * 3, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let data = params.to_f32_vec()?;
        let device = temb.device();
        let shape = Shape::from([hidden]);
        let to_f16 = |s: usize, e: usize| -> Vec<half::f16> {
            data[s..e].iter().map(|&v| half::f16::from_f32(v)).collect()
        };

        Ok((
            Tensor::from_slice(&to_f16(0, hidden), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden, hidden * 2), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden * 2, hidden * 3), shape.clone(), DType::F16, device)?,
        ))
    }

    /// Extract 2 modulation parameters (shift, scale) for output norm.
    fn adaln_2params(
        &self, temb: &Tensor, weight_name: &str, hidden: usize,
    ) -> Result<(Tensor, Tensor)> {
        let compute = &self.compute;
        let cb = compute.new_command_buffer();
        let activated = self.gpu_silu(temb, cb.as_ref())?;
        let params = self.gpu_linear_biased_prefix(&activated, weight_name, hidden, hidden * 2, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        let data = params.to_f32_vec()?;
        let device = temb.device();
        let shape = Shape::from([hidden]);
        let to_f16 = |s: usize, e: usize| -> Vec<half::f16> {
            data[s..e].iter().map(|&v| half::f16::from_f32(v)).collect()
        };

        Ok((
            Tensor::from_slice(&to_f16(0, hidden), shape.clone(), DType::F16, device)?,
            Tensor::from_slice(&to_f16(hidden, hidden * 2), shape.clone(), DType::F16, device)?,
        ))
    }

    // ========================================================================
    // Batched attention
    // ========================================================================

    fn batched_attention(
        &self, q: &Tensor, k: &Tensor, v: &Tensor,
        q_seq: usize, kv_seq: usize, num_heads: usize, head_dim: usize,
    ) -> Result<Tensor> {
        let compute = &self.compute;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_shd = q.reshape(Shape::from([q_seq, num_heads, head_dim]))?;
        let k_shd = k.reshape(Shape::from([kv_seq, num_heads, head_dim]))?;
        let v_shd = v.reshape(Shape::from([kv_seq, num_heads, head_dim]))?;

        // Transpose SHD → HSD
        let cb = compute.new_command_buffer();
        let q_hsd = self.gpu_transpose_shd_hsd(&q_shd, q_seq, num_heads, head_dim, cb.as_ref())?;
        let k_hsd = self.gpu_transpose_shd_hsd(&k_shd, kv_seq, num_heads, head_dim, cb.as_ref())?;
        let v_hsd = self.gpu_transpose_shd_hsd(&v_shd, kv_seq, num_heads, head_dim, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        // Q @ K^T → [heads, q_seq, kv_seq]
        let cb = compute.new_command_buffer();
        let scores = self.gpu_batched_linear_raw(
            &q_hsd, &k_hsd, num_heads, q_seq, head_dim, kv_seq, cb.as_ref(),
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Scaled softmax
        let cb = compute.new_command_buffer();
        let attn_weights = self.gpu_row_softmax(
            &scores, num_heads * q_seq, kv_seq, scale, cb.as_ref(),
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Weights @ V → [heads, q_seq, head_dim]
        let cb = compute.new_command_buffer();
        let attn_hsd = self.gpu_batched_matmul_nn(
            &attn_weights, &v_hsd, num_heads, q_seq, kv_seq, head_dim, cb.as_ref(),
        )?;
        let attn_shd = self.gpu_transpose_hsd_shd(&attn_hsd, q_seq, num_heads, head_dim, cb.as_ref())?;
        cb.commit();
        cb.wait_until_completed();

        Ok(attn_shd.reshape(Shape::from([q_seq, num_heads * head_dim]))?)
    }

    // ========================================================================
    // Timestep embedding
    // ========================================================================

    fn timestep_embedding(&self, timestep: f32, device: crate::hal::DeviceId) -> Result<Tensor> {
        crate::inference::architecture::dit::DiTOps::timestep_embedding(timestep, 256, device)
    }

    // ========================================================================
    // GPU dispatch helpers
    // ========================================================================

    fn w(&self, name: &str) -> Result<&crate::hal::metal::LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| Error::internal(format!("HunyuanVideo weight not found: {}", name)))
    }

    /// MLP: linear_1 → SiLU → linear_2 (all biased).
    fn gpu_mlp_biased(
        &self, x: &Tensor, prefix: &str, in_dim: usize, out_dim: usize,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let h = self.gpu_linear_biased_prefix(x, &format!("{}.linear_1", prefix), in_dim, out_dim, cb)?;
        let h = self.gpu_silu(&h, cb)?;
        self.gpu_linear_biased_prefix(&h, &format!("{}.linear_2", prefix), out_dim, out_dim, cb)
    }

    fn gpu_linear_biased_prefix(
        &self, x: &Tensor, prefix: &str, in_feat: usize, out_feat: usize,
        cb: &metal::CommandBufferRef,
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
        self.compute.dispatch_async(cb, &self.kernels.linear,
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

    fn gpu_silu(&self, x: &Tensor, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c = count as u32;
        self.compute.dispatch_async(cb, &self.kernels.silu,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_gelu(&self, x: &Tensor, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c = count as u32;
        self.compute.dispatch_async(cb, &self.kernels.gelu,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_add(&self, a: &Tensor, b: &Tensor, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let count = a.shape().numel();
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c = count as u32;
        self.compute.dispatch_async(cb, &self.kernels.add,
            ((count + 255) / 256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(a_buf.as_ref()), 0);
                enc.set_buffer(1, Some(b_buf.as_ref()), 0);
                enc.set_buffer(2, Some(o_buf.as_ref()), 0);
                enc.set_bytes(3, 4, &c as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_layer_norm(&self, x: &Tensor, hidden: usize, cb: &metal::CommandBufferRef) -> Result<Tensor> {
        let seq_len = x.shape().numel() / hidden;
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_hidden = hidden as u32;
        let eps: f32 = 1e-6;
        self.compute.dispatch_async(cb, &self.kernels.layer_norm,
            (seq_len, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_hidden as *const u32 as *const _);
                enc.set_bytes(3, 4, &eps as *const f32 as *const _);
            });
        Ok(output)
    }

    fn gpu_adaln_modulate(
        &self, x: &Tensor, scale: &Tensor, shift: &Tensor, hidden: usize,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let s_buf = borrow_tensor(scale)?;
        let sh_buf = borrow_tensor(shift)?;
        let o_buf = borrow_tensor(&output)?;
        let c_h = hidden as u32;
        let c_n = count as u32;
        self.compute.dispatch_async(cb, &self.kernels.adaln_modulate,
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

    fn gpu_adaln_gate(
        &self, x: &Tensor, residual: &Tensor, gate: &Tensor, hidden: usize,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let count = x.shape().numel();
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let r_buf = borrow_tensor(residual)?;
        let g_buf = borrow_tensor(gate)?;
        let o_buf = borrow_tensor(&output)?;
        let c_h = hidden as u32;
        let c_n = count as u32;
        self.compute.dispatch_async(cb, &self.kernels.adaln_gate,
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

    fn gpu_transpose_shd_hsd(
        &self, x: &Tensor, seq: usize, heads: usize, dim: usize,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([heads, seq, dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq as u32; let c_h = heads as u32; let c_d = dim as u32;
        self.compute.dispatch_async(cb, &self.kernels.transpose_shd_hsd,
            (heads, seq, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf.as_ref()), 0);
                enc.set_buffer(1, Some(o_buf.as_ref()), 0);
                enc.set_bytes(2, 4, &c_s as *const u32 as *const _);
                enc.set_bytes(3, 4, &c_h as *const u32 as *const _);
                enc.set_bytes(4, 4, &c_d as *const u32 as *const _);
            });
        Ok(output)
    }

    fn gpu_transpose_hsd_shd(
        &self, x: &Tensor, seq: usize, heads: usize, dim: usize,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([seq, heads, dim]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_s = seq as u32; let c_h = heads as u32; let c_d = dim as u32;
        self.compute.dispatch_async(cb, &self.kernels.transpose_hsd_shd,
            (heads, seq, 1), (1, 1, 1), |enc| {
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
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let w_buf = borrow_tensor(w)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32; let c_n = n as u32; let c_k = k as u32; let c_b = batch as u32;
        self.compute.dispatch_async(cb, &self.kernels.batched_linear,
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

    fn gpu_batched_matmul_nn(
        &self, a: &Tensor, b: &Tensor, batch: usize, m: usize, k: usize, n: usize,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([batch, m, n]), DType::F16, a.device())?;
        let a_buf = borrow_tensor(a)?;
        let b_buf = borrow_tensor(b)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32; let c_n = n as u32; let c_k = k as u32; let c_b = batch as u32;
        self.compute.dispatch_async(cb, &self.kernels.batched_matmul_nn,
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

    fn gpu_row_softmax(
        &self, x: &Tensor, num_rows: usize, row_len: usize, scale: f32,
        cb: &metal::CommandBufferRef,
    ) -> Result<Tensor> {
        let output = Tensor::empty(x.shape().clone(), DType::F16, x.device())?;
        let x_buf = borrow_tensor(x)?;
        let o_buf = borrow_tensor(&output)?;
        let c_cols = row_len as u32;
        self.compute.dispatch_async(cb, &self.kernels.row_softmax,
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
        .ok_or_else(|| Error::internal("tensor not on device"))?;
    Ok(unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) })
}

// ============================================================================
// VAE Config
// ============================================================================

/// HunyuanVideo 3D VAE configuration.
#[derive(Debug, Clone)]
pub struct HunyuanVideoVaeConfig {
    /// Input/output channels.
    pub in_channels: usize,
    /// Latent channels.
    pub latent_channels: usize,
    /// Block output channels per resolution level.
    pub block_out_channels: Vec<usize>,
    /// Spatial compression ratio.
    pub spatial_compression: usize,
    /// Temporal compression ratio.
    pub temporal_compression: usize,
    /// Layers per block.
    pub layers_per_block: usize,
    /// GroupNorm groups.
    pub norm_num_groups: usize,
}

impl Default for HunyuanVideoVaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            spatial_compression: 8,
            temporal_compression: 4,
            layers_per_block: 2,
            norm_num_groups: 32,
        }
    }
}
