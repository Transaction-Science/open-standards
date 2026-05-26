//! TripoSG: Image-to-3D mesh generation via rectified flow on SDF latents.
//!
//! Architecture:
//!   Image → CLIP-L/14 (global 768) + DINOv2-L/14 (patch features [N, 1024])
//!   → U-shaped Rectified Flow Transformer (encoder→middle→decoder with skip connections)
//!   → SDF VAE decode → marching cubes → watertight triangle mesh
//!
//! Based on VAST-AI/TripoSG (Feb 2025), 1.5B parameters.
//! Uses rectified flow matching with Euler ODE solver (same as Trellis).

use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

// Re-use MeshOutput from InstantMesh
use super::instantmesh::MeshOutput;

/// TripoSG configuration.
#[derive(Debug, Clone)]
pub struct TripoSGConfig {
    /// DINOv2 image input size.
    pub dino_image_size: usize,
    /// DINOv2 hidden dimension (ViT-L/14 = 1024).
    pub dino_hidden: usize,
    /// DINOv2 number of heads.
    pub dino_heads: usize,
    /// DINOv2 number of layers.
    pub dino_layers: usize,
    /// DINOv2 patch size.
    pub dino_patch_size: usize,
    /// CLIP-L/14 hidden dimension.
    pub clip_hidden: usize,
    /// CLIP-L/14 number of layers.
    pub clip_layers: usize,
    /// Flow transformer hidden dimension.
    pub flow_hidden: usize,
    /// Flow transformer number of heads.
    pub flow_num_heads: usize,
    /// Flow transformer head dimension.
    pub flow_head_dim: usize,
    /// Number of U-Net encoder blocks.
    pub flow_encoder_blocks: usize,
    /// Number of U-Net middle blocks.
    pub flow_middle_blocks: usize,
    /// Number of U-Net decoder blocks.
    pub flow_decoder_blocks: usize,
    /// MLP expansion ratio.
    pub mlp_ratio: usize,
    /// Number of latent SDF tokens.
    pub num_latent_tokens: usize,
    /// Latent channel dimension.
    pub latent_channels: usize,
    /// Number of Euler flow steps.
    pub flow_steps: usize,
    /// Classifier-free guidance strength.
    pub cfg_strength: f32,
    /// Minimum sigma for flow matching.
    pub sigma_min: f32,
    /// Marching cubes grid resolution.
    pub grid_resolution: usize,
    /// SDF MLP hidden dimension.
    pub sdf_hidden: usize,
    /// SDF MLP number of layers.
    pub sdf_num_layers: usize,
}

impl Default for TripoSGConfig {
    fn default() -> Self {
        Self {
            dino_image_size: 518,
            dino_hidden: 1024,
            dino_heads: 16,
            dino_layers: 24,
            dino_patch_size: 14,
            clip_hidden: 768,
            clip_layers: 12,
            flow_hidden: 1024,
            flow_num_heads: 16,
            flow_head_dim: 64,
            flow_encoder_blocks: 12,
            flow_middle_blocks: 2,
            flow_decoder_blocks: 12,
            mlp_ratio: 4,
            num_latent_tokens: 2048,
            latent_channels: 64,
            flow_steps: 30,
            cfg_strength: 7.5,
            sigma_min: 1e-5,
            grid_resolution: 256,
            sdf_hidden: 64,
            sdf_num_layers: 4,
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct TripoSGKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    sub: Arc<ComputePipeline>,
    mul: Arc<ComputePipeline>,
    scale: Arc<ComputePipeline>,
    adaln_modulate: Arc<ComputePipeline>,
    adaln_gate: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
}

// ==================== TripoSG Pipeline ====================

/// TripoSG pipeline for image-to-3D mesh generation via rectified flow.
///
/// Forward pipeline:
/// 1. CLIP-L/14: image → [768] global feature (pooled)
/// 2. DINOv2-L/14: image → [N, 1024] patch features
/// 3. U-shaped Flow Transformer: 30 Euler steps on [2048, 64] latent tokens
///    - Encoder blocks (12) with skip connection storage
///    - Middle blocks (2)
///    - Decoder blocks (12) consuming skip connections
/// 4. SDF VAE decode: latent → SDF grid → marching cubes → watertight mesh
#[cfg(feature = "metal")]
pub struct TripoSGPipeline {
    flow_model: Arc<Model>,
    dino_model: Arc<Model>,
    clip_model: Arc<Model>,
    vae_model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: TripoSGConfig,
    kernels: TripoSGKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for TripoSGPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl TripoSGPipeline {
    /// Create a new TripoSG pipeline.
    ///
    /// Requires 4 separate Model objects:
    /// - `flow_model`: U-shaped flow transformer weights
    /// - `dino_model`: DINOv2-ViT-L/14 weights
    /// - `clip_model`: CLIP-ViT-L/14 weights
    /// - `vae_model`: SDF VAE decoder weights
    pub fn new(
        flow_model: Arc<Model>,
        dino_model: Arc<Model>,
        clip_model: Arc<Model>,
        vae_model: Arc<Model>,
        config: TripoSGConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = TripoSGKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            silu: compute.compile_pipeline("silu", sources::GELU, "silu_f16")?,
            sub: compute.compile_pipeline("sub", sources::ELEMENTWISE, "sub_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
            scale: compute.compile_pipeline("scale", sources::ELEMENTWISE, "scale_f16")?,
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ELEMENTWISE, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ELEMENTWISE, "adaln_gate_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
        };

        Ok(Self { flow_model, dino_model, clip_model, vae_model, compute, config, kernels })
    }

    /// Generate 3D mesh from a single image.
    ///
    /// `image_chw`: Image as flat f32 RGB array [3 * 518 * 518] in [C, H, W] format,
    ///              normalized to ImageNet stats for DINOv2, and CLIP normalization for CLIP.
    /// `seed`: Random seed for deterministic noise initialization.
    pub fn generate(&self, image_chw: &[f32], seed: u64) -> Result<MeshOutput> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;

        println!("  [TripoSG] Encoding image with CLIP-L/14...");
        let clip_global = self.clip_encode(image_chw)?;

        println!("  [TripoSG] Encoding image with DINOv2-L/14...");
        let dino_features = self.dino_encode(image_chw)?;
        let num_dino_patches = (config.dino_image_size / config.dino_patch_size).pow(2); // 1369

        // Concatenate CLIP global + DINOv2 patches as conditioning
        let clip_data: Vec<half::f16> = clip_global.to_vec()?;
        let dino_data: Vec<half::f16> = dino_features.to_vec()?;
        // Project CLIP [768] → [1, 1024] to match DINOv2 dim, then concat
        // For simplicity, pad CLIP to match DINOv2 hidden dim
        let mut cond_data = Vec::with_capacity((1 + num_dino_patches) * config.dino_hidden);
        // CLIP token (zero-padded to dino_hidden)
        for i in 0..config.dino_hidden {
            if i < config.clip_hidden {
                cond_data.push(clip_data[i]);
            } else {
                cond_data.push(half::f16::ZERO);
            }
        }
        cond_data.extend_from_slice(&dino_data);
        let num_cond_tokens = 1 + num_dino_patches;
        let cond_features = Tensor::from_slice(
            &cond_data,
            Shape::from([num_cond_tokens, config.dino_hidden]),
            DType::F16, device_id,
        )?;

        println!("  [TripoSG] Running rectified flow ({} steps)...", config.flow_steps);
        let latent = self.flow_matching_loop(
            &cond_features, num_cond_tokens, seed,
        )?;

        println!("  [TripoSG] Decoding SDF + marching cubes...");
        let sdf_grid = self.vae_decode_sdf(&latent)?;
        let mesh = super::instantmesh::marching_cubes(&sdf_grid, config.grid_resolution);

        Ok(mesh)
    }

    // ==================== CLIP-L/14 Encoder ====================

    /// CLIP-ViT-L/14 encode: image → [768] global feature (CLS pooled).
    fn clip_encode(&self, image_chw: &[f32]) -> Result<Tensor> {
        let d_model = self.config.clip_hidden; // 768
        let patch_size = 14;
        let grid = self.config.dino_image_size / patch_size; // 37
        let num_patches = grid * grid; // 1369
        let num_heads = 12;
        let head_dim = d_model / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = num_patches + 1; // 1370 (CLS + patches)
        let device_id = self.compute.device().info().id;

        // Patch embedding
        let patches = self.clip_patch_embed(image_chw, grid, patch_size, d_model)?;

        // Prepend CLS token + position embeddings
        let cls_token = self.weight_f16(&self.clip_model, "embeddings.class_embedding")?;
        let cls_data: Vec<half::f16> = cls_token.to_vec()?;
        let patches_data: Vec<half::f16> = patches.to_vec()?;

        let mut combined = Vec::with_capacity(seq_len * d_model);
        combined.extend_from_slice(&cls_data[..d_model]);
        combined.extend_from_slice(&patches_data);

        let pos_embed = self.weight_f16(&self.clip_model, "embeddings.position_embedding.weight")?;
        let pos_data: Vec<half::f16> = pos_embed.to_vec()?;
        for i in 0..seq_len * d_model {
            combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16, device_id,
        )?;

        // 12 encoder layers
        let ffn_dim = d_model * 4;
        for layer in 0..self.config.clip_layers {
            let prefix = format!("encoder.layers.{}", layer);
            let cb = self.compute.new_command_buffer();

            let normed = self.layer_norm(&cb, &self.clip_model, &hidden,
                &format!("{}.layer_norm1.weight", prefix),
                &format!("{}.layer_norm1.bias", prefix),
                seq_len, d_model, 1e-5)?;

            let q = self.linear_bias(&cb, &self.clip_model, &normed,
                &format!("{}.self_attn.q_proj.weight", prefix),
                &format!("{}.self_attn.q_proj.bias", prefix),
                seq_len, d_model, d_model)?;
            let k = self.linear_bias(&cb, &self.clip_model, &normed,
                &format!("{}.self_attn.k_proj.weight", prefix),
                &format!("{}.self_attn.k_proj.bias", prefix),
                seq_len, d_model, d_model)?;
            let v = self.linear_bias(&cb, &self.clip_model, &normed,
                &format!("{}.self_attn.v_proj.weight", prefix),
                &format!("{}.self_attn.v_proj.bias", prefix),
                seq_len, d_model, d_model)?;

            let attn_out = self.batched_attention(&cb, &q, &k, &v, seq_len, seq_len, num_heads, head_dim, scale)?;
            let proj = self.linear_bias(&cb, &self.clip_model, &attn_out,
                &format!("{}.self_attn.out_proj.weight", prefix),
                &format!("{}.self_attn.out_proj.bias", prefix),
                seq_len, d_model, d_model)?;
            let h = self.add(&cb, &hidden, &proj);

            let normed2 = self.layer_norm(&cb, &self.clip_model, &h,
                &format!("{}.layer_norm2.weight", prefix),
                &format!("{}.layer_norm2.bias", prefix),
                seq_len, d_model, 1e-5)?;
            let ffn_up = self.linear_bias(&cb, &self.clip_model, &normed2,
                &format!("{}.mlp.fc1.weight", prefix),
                &format!("{}.mlp.fc1.bias", prefix),
                seq_len, d_model, ffn_dim)?;
            let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
            let ffn_down = self.linear_bias(&cb, &self.clip_model, &ffn_act,
                &format!("{}.mlp.fc2.weight", prefix),
                &format!("{}.mlp.fc2.bias", prefix),
                seq_len, ffn_dim, d_model)?;
            hidden = self.add(&cb, &h, &ffn_down);

            cb.commit();
            cb.wait_until_completed();
        }

        // Return CLS token [768]
        hidden.slice(0, 0, 1)?.reshape([d_model])
    }

    /// Patch embedding via im2col (CPU memory layout) + GPU matmul (compute).
    fn clip_patch_embed(&self, image_chw: &[f32], grid: usize, patch_size: usize, d_model: usize) -> Result<Tensor> {
        let c_in = 3;
        let num_patches = grid * grid;
        let img_size = self.config.dino_image_size;
        let k_size = c_in * patch_size * patch_size; // 588

        // im2col: extract non-overlapping patches → [num_patches, c_in * patch² ]
        let mut col_data: Vec<half::f16> = vec![half::f16::ZERO; num_patches * k_size];
        for gy in 0..grid {
            for gx in 0..grid {
                let p = gy * grid + gx;
                for in_c in 0..c_in {
                    for ky in 0..patch_size {
                        for kx in 0..patch_size {
                            let iy = gy * patch_size + ky;
                            let ix = gx * patch_size + kx;
                            if iy < img_size && ix < img_size {
                                let val = image_chw[in_c * img_size * img_size + iy * img_size + ix];
                                col_data[p * k_size + in_c * patch_size * patch_size + ky * patch_size + kx] =
                                    half::f16::from_f32(val);
                            }
                        }
                    }
                }
            }
        }

        // GPU matmul: [num_patches, k_size] @ W^T + bias → [num_patches, d_model]
        let col_tensor = Tensor::from_slice(&col_data, Shape::from([num_patches, k_size]), DType::F16, self.compute.device().info().id)?;
        let cb = self.compute.new_command_buffer();
        let result = self.linear_bias(&cb, &self.clip_model, &col_tensor,
            "embeddings.patch_embedding.weight", "embeddings.patch_embedding.bias",
            num_patches, k_size, d_model)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    // ==================== DINOv2-L/14 Encoder ====================

    /// DINOv2-ViT-L/14 encode: image → [N, 1024] patch features.
    fn dino_encode(&self, image_chw: &[f32]) -> Result<Tensor> {
        let d_model = self.config.dino_hidden; // 1024
        let patch_size = self.config.dino_patch_size; // 14
        let grid = self.config.dino_image_size / patch_size; // 37
        let num_patches = grid * grid; // 1369
        let num_heads = self.config.dino_heads; // 16
        let head_dim = d_model / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = num_patches + 1; // 1370 (CLS + patches)
        let device_id = self.compute.device().info().id;

        // Patch embedding
        let patches = self.dino_patch_embed(image_chw, grid, patch_size, d_model)?;

        // CLS + position embeddings
        let cls_token = self.weight_f16(&self.dino_model, "embeddings.cls_token")?;
        let cls_data: Vec<half::f16> = cls_token.to_vec()?;
        let patches_data: Vec<half::f16> = patches.to_vec()?;

        let mut combined = Vec::with_capacity(seq_len * d_model);
        combined.extend_from_slice(&cls_data[..d_model]);
        combined.extend_from_slice(&patches_data);

        let pos_embed = self.weight_f16(&self.dino_model, "embeddings.position_embeddings")?;
        let pos_data: Vec<half::f16> = pos_embed.to_vec()?;
        for i in 0..(seq_len * d_model).min(pos_data.len()) {
            combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16, device_id,
        )?;

        // 24 encoder layers
        let ffn_dim = d_model * 4;
        for layer in 0..self.config.dino_layers {
            let prefix = format!("encoder.layer.{}", layer);
            let cb = self.compute.new_command_buffer();

            let normed = self.layer_norm(&cb, &self.dino_model, &hidden,
                &format!("{}.layernorm_before.weight", prefix),
                &format!("{}.layernorm_before.bias", prefix),
                seq_len, d_model, 1e-5)?;

            let q = self.linear_bias(&cb, &self.dino_model, &normed,
                &format!("{}.attention.attention.query.weight", prefix),
                &format!("{}.attention.attention.query.bias", prefix),
                seq_len, d_model, d_model)?;
            let k = self.linear_bias(&cb, &self.dino_model, &normed,
                &format!("{}.attention.attention.key.weight", prefix),
                &format!("{}.attention.attention.key.bias", prefix),
                seq_len, d_model, d_model)?;
            let v = self.linear_bias(&cb, &self.dino_model, &normed,
                &format!("{}.attention.attention.value.weight", prefix),
                &format!("{}.attention.attention.value.bias", prefix),
                seq_len, d_model, d_model)?;

            let attn_out = self.batched_attention(&cb, &q, &k, &v, seq_len, seq_len, num_heads, head_dim, scale)?;
            let proj = self.linear_bias(&cb, &self.dino_model, &attn_out,
                &format!("{}.attention.output.dense.weight", prefix),
                &format!("{}.attention.output.dense.bias", prefix),
                seq_len, d_model, d_model)?;
            let h = self.add(&cb, &hidden, &proj);

            let normed2 = self.layer_norm(&cb, &self.dino_model, &h,
                &format!("{}.layernorm_after.weight", prefix),
                &format!("{}.layernorm_after.bias", prefix),
                seq_len, d_model, 1e-5)?;
            let ffn_up = self.linear_bias(&cb, &self.dino_model, &normed2,
                &format!("{}.intermediate.dense.weight", prefix),
                &format!("{}.intermediate.dense.bias", prefix),
                seq_len, d_model, ffn_dim)?;
            let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
            let ffn_down = self.linear_bias(&cb, &self.dino_model, &ffn_act,
                &format!("{}.output.dense.weight", prefix),
                &format!("{}.output.dense.bias", prefix),
                seq_len, ffn_dim, d_model)?;
            hidden = self.add(&cb, &h, &ffn_down);

            cb.commit();
            cb.wait_until_completed();
        }

        // Final norm, remove CLS → [num_patches, 1024]
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.dino_model, &hidden,
            "layernorm.weight", "layernorm.bias",
            seq_len, d_model, 1e-5)?;
        cb.commit();
        cb.wait_until_completed();
        normed.slice(0, 1, seq_len)
    }

    /// DINO patch embedding via im2col (CPU) + GPU matmul.
    fn dino_patch_embed(&self, image_chw: &[f32], grid: usize, patch_size: usize, d_model: usize) -> Result<Tensor> {
        let c_in = 3;
        let num_patches = grid * grid;
        let img_size = self.config.dino_image_size;
        let k_size = c_in * patch_size * patch_size;

        // im2col: [num_patches, c_in * patch²]
        let mut col_data: Vec<half::f16> = vec![half::f16::ZERO; num_patches * k_size];
        for gy in 0..grid {
            for gx in 0..grid {
                let p = gy * grid + gx;
                for in_c in 0..c_in {
                    for ky in 0..patch_size {
                        for kx in 0..patch_size {
                            let iy = gy * patch_size + ky;
                            let ix = gx * patch_size + kx;
                            if iy < img_size && ix < img_size {
                                let val = image_chw[in_c * img_size * img_size + iy * img_size + ix];
                                col_data[p * k_size + in_c * patch_size * patch_size + ky * patch_size + kx] =
                                    half::f16::from_f32(val);
                            }
                        }
                    }
                }
            }
        }

        let col_tensor = Tensor::from_slice(&col_data, Shape::from([num_patches, k_size]), DType::F16, self.compute.device().info().id)?;
        let cb = self.compute.new_command_buffer();
        let result = self.linear_bias(&cb, &self.dino_model, &col_tensor,
            "embeddings.patch_embeddings.projection.weight", "embeddings.patch_embeddings.projection.bias",
            num_patches, k_size, d_model)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    // ==================== Rectified Flow Matching ====================

    /// Euler ODE flow matching loop with classifier-free guidance.
    fn flow_matching_loop(
        &self,
        cond_features: &Tensor,
        num_cond_tokens: usize,
        seed: u64,
    ) -> Result<Tensor> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;
        let numel = config.num_latent_tokens * config.latent_channels;

        // Initialize noise
        let sigma_1 = config.sigma_min + (1.0 - config.sigma_min) * 1.0;
        let mut x_data = deterministic_randn(numel, seed);
        for v in &mut x_data {
            *v *= sigma_1;
        }
        let x_f16: Vec<half::f16> = x_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut x = Tensor::from_slice(
            &x_f16,
            Shape::from([config.num_latent_tokens, config.latent_channels]),
            DType::F16, device_id,
        )?;

        // Time schedule: linspace(1, 0, steps+1)
        let t_seq: Vec<f32> = (0..=config.flow_steps)
            .map(|i| 1.0 - i as f32 / config.flow_steps as f32)
            .collect();

        // Pre-compute cross-attention KV for conditional pass
        let cross_kv = self.precompute_flow_cross_kv(
            &cond_features, num_cond_tokens,
        )?;

        // Null features for unconditional pass
        let null_features = Tensor::from_slice(
            &vec![half::f16::ZERO; num_cond_tokens * config.dino_hidden],
            Shape::from([num_cond_tokens, config.dino_hidden]),
            DType::F16, device_id,
        )?;
        let uncond_cross_kv = self.precompute_flow_cross_kv(
            &null_features, num_cond_tokens,
        )?;

        for step in 0..config.flow_steps {
            let t = t_seq[step];
            let dt = t_seq[step + 1] - t;

            println!("    [flow] step {}/{}: t={:.3}", step + 1, config.flow_steps, t);

            // Conditional forward
            let v_cond = self.flow_model_forward(&x, t, &cross_kv, num_cond_tokens)?;

            // Unconditional forward
            let v_uncond = self.flow_model_forward(&x, t, &uncond_cross_kv, num_cond_tokens)?;

            // CFG: v = v_uncond + cfg_strength * (v_cond - v_uncond)
            let cb = self.compute.new_command_buffer();
            let diff = self.elementwise_binary(&cb, &self.kernels.sub, &v_cond, &v_uncond);
            let scaled_diff = self.scale_tensor(&cb, &self.kernels.scale, &diff, config.cfg_strength);
            let v = self.add(&cb, &v_uncond, &scaled_diff);

            // Euler step: x = x + v * dt
            let v_dt = self.scale_tensor(&cb, &self.kernels.scale, &v, dt);
            x = self.add(&cb, &x, &v_dt);
            cb.commit();
            cb.wait_until_completed();
        }

        Ok(x)
    }

    /// Pre-compute cross-attention K/V from condition features for all U-Net blocks.
    fn precompute_flow_cross_kv(
        &self,
        cond_features: &Tensor,
        num_cond_tokens: usize,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let config = &self.config;
        let num_heads = config.flow_num_heads;
        let head_dim = config.flow_head_dim;
        let total_blocks = config.flow_encoder_blocks + config.flow_middle_blocks + config.flow_decoder_blocks;
        let device_id = self.compute.device().info().id;

        let mut cross_kv = Vec::with_capacity(total_blocks);
        for block in 0..total_blocks {
            let prefix = format!("blocks.{}", block);
            let cb = self.compute.new_command_buffer();

            let k = self.linear_on(&cb, &self.flow_model, cond_features,
                &format!("{}.cross_attn.to_k.weight", prefix),
                num_cond_tokens, config.dino_hidden, config.flow_hidden)?;
            let v = self.linear_on(&cb, &self.flow_model, cond_features,
                &format!("{}.cross_attn.to_v.weight", prefix),
                num_cond_tokens, config.dino_hidden, config.flow_hidden)?;

            let k_hsd = Tensor::empty(Shape::from([num_heads, num_cond_tokens, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(Shape::from([num_heads, num_cond_tokens, head_dim]), DType::F16, device_id)?;
            self.transpose_shd_to_hsd(&cb, &k, &k_hsd, num_cond_tokens, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &v, &v_hsd, num_cond_tokens, num_heads, head_dim);

            cb.commit();
            cb.wait_until_completed();
            cross_kv.push((k_hsd, v_hsd));
        }
        Ok(cross_kv)
    }

    /// Single forward pass through the U-shaped flow transformer.
    fn flow_model_forward(
        &self,
        x: &Tensor,
        t: f32,
        cross_kv: &[(Tensor, Tensor)],
        num_cond_tokens: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let num_tokens = config.num_latent_tokens;
        let hidden = config.flow_hidden;
        let num_heads = config.flow_num_heads;
        let head_dim = config.flow_head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mlp_dim = hidden * config.mlp_ratio;
        let device_id = self.compute.device().info().id;

        // Project latent to hidden dim
        let t_f16 = half::f16::from_f32(t);
        let t_tensor = Tensor::from_slice(
            &[t_f16], Shape::from([1, 1]), DType::F16, device_id,
        )?;

        let cb = self.compute.new_command_buffer();
        let mut h = self.linear_on(&cb, &self.flow_model, x,
            "proj_in.weight", num_tokens, config.latent_channels, hidden)?;
        cb.commit();
        cb.wait_until_completed();

        // Encoder blocks with skip connections
        let mut skips = Vec::with_capacity(config.flow_encoder_blocks);
        for block in 0..config.flow_encoder_blocks {
            h = self.flow_block(&h, block, num_tokens, hidden, num_heads, head_dim, mlp_dim, scale, cross_kv, num_cond_tokens)?;
            skips.push(h.clone());
        }

        // Middle blocks
        let enc_blocks = config.flow_encoder_blocks;
        for block in 0..config.flow_middle_blocks {
            h = self.flow_block(&h, enc_blocks + block, num_tokens, hidden, num_heads, head_dim, mlp_dim, scale, cross_kv, num_cond_tokens)?;
        }

        // Decoder blocks consuming skip connections
        let mid_offset = enc_blocks + config.flow_middle_blocks;
        for block in 0..config.flow_decoder_blocks {
            // Add skip connection from encoder
            if let Some(skip) = skips.pop() {
                let cb = self.compute.new_command_buffer();
                h = self.add(&cb, &h, &skip);
                cb.commit();
                cb.wait_until_completed();
            }
            h = self.flow_block(&h, mid_offset + block, num_tokens, hidden, num_heads, head_dim, mlp_dim, scale, cross_kv, num_cond_tokens)?;
        }

        // Project out
        let cb = self.compute.new_command_buffer();
        let out = self.linear_on(&cb, &self.flow_model, &h,
            "proj_out.weight", num_tokens, hidden, config.latent_channels)?;
        cb.commit();
        cb.wait_until_completed();

        Ok(out)
    }

    /// Single flow transformer block: self-attn + cross-attn + FFN.
    fn flow_block(
        &self,
        input: &Tensor,
        block_idx: usize,
        seq_len: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        scale: f32,
        cross_kv: &[(Tensor, Tensor)],
        num_cond_tokens: usize,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{}", block_idx);

        // Self-attention
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.flow_model, input,
            &format!("{}.norm1.weight", prefix),
            &format!("{}.norm1.bias", prefix),
            seq_len, hidden, 1e-5)?;

        let q = self.linear_on(&cb, &self.flow_model, &normed,
            &format!("{}.self_attn.to_q.weight", prefix),
            seq_len, hidden, hidden)?;
        let k = self.linear_on(&cb, &self.flow_model, &normed,
            &format!("{}.self_attn.to_k.weight", prefix),
            seq_len, hidden, hidden)?;
        let v = self.linear_on(&cb, &self.flow_model, &normed,
            &format!("{}.self_attn.to_v.weight", prefix),
            seq_len, hidden, hidden)?;

        let attn_out = self.batched_attention(&cb, &q, &k, &v, seq_len, seq_len, num_heads, head_dim, scale)?;
        let sa_proj = self.linear_on(&cb, &self.flow_model, &attn_out,
            &format!("{}.self_attn.to_out.weight", prefix),
            seq_len, hidden, hidden)?;
        let h = self.add(&cb, input, &sa_proj);
        cb.commit();
        cb.wait_until_completed();

        // Cross-attention
        let cb = self.compute.new_command_buffer();
        let normed2 = self.layer_norm(&cb, &self.flow_model, &h,
            &format!("{}.norm2.weight", prefix),
            &format!("{}.norm2.bias", prefix),
            seq_len, hidden, 1e-5)?;

        let cross_q = self.linear_on(&cb, &self.flow_model, &normed2,
            &format!("{}.cross_attn.to_q.weight", prefix),
            seq_len, hidden, hidden)?;

        let cq_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, self.compute.device().info().id)?;
        self.transpose_shd_to_hsd(&cb, &cross_q, &cq_hsd, seq_len, num_heads, head_dim);

        let (ref ck, ref cv) = cross_kv[block_idx];
        let cross_scores = self.batched_qk(&cb, &cq_hsd, ck, num_heads, seq_len, num_cond_tokens, head_dim);
        self.row_softmax(&cb, &cross_scores, num_heads * seq_len, num_cond_tokens, scale);
        let cross_out_hsd = self.batched_sv(&cb, &cross_scores, cv, num_heads, seq_len, num_cond_tokens, head_dim);

        let cross_shd = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, self.compute.device().info().id)?;
        self.transpose_hsd_to_shd(&cb, &cross_out_hsd, &cross_shd, seq_len, num_heads, head_dim);
        let cross_flat = cross_shd.reshape([seq_len, hidden])?;

        let ca_proj = self.linear_on(&cb, &self.flow_model, &cross_flat,
            &format!("{}.cross_attn.to_out.weight", prefix),
            seq_len, hidden, hidden)?;
        let h = self.add(&cb, &h, &ca_proj);
        cb.commit();
        cb.wait_until_completed();

        // FFN
        let cb = self.compute.new_command_buffer();
        let normed3 = self.layer_norm(&cb, &self.flow_model, &h,
            &format!("{}.norm3.weight", prefix),
            &format!("{}.norm3.bias", prefix),
            seq_len, hidden, 1e-5)?;
        let ffn_up = self.linear_on(&cb, &self.flow_model, &normed3,
            &format!("{}.ffn.fc1.weight", prefix),
            seq_len, hidden, mlp_dim)?;
        let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
        let ffn_down = self.linear_on(&cb, &self.flow_model, &ffn_act,
            &format!("{}.ffn.fc2.weight", prefix),
            seq_len, mlp_dim, hidden)?;
        let result = self.add(&cb, &h, &ffn_down);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    // ==================== SDF VAE Decode ====================

    /// Decode latent tokens to SDF grid via batched GPU MLP.
    /// Batches all res³ grid points into a single GPU matmul per layer.
    fn vae_decode_sdf(&self, latent: &Tensor) -> Result<Vec<f32>> {
        let config = &self.config;
        let res = config.grid_resolution;
        let n_points = res * res * res;
        let latent_data: Vec<half::f16> = latent.to_vec()?;
        let device_id = self.compute.device().info().id;
        let input_dim = config.latent_channels + 3; // features + xyz

        // Build batched input [N, latent_channels + 3] on CPU (just indexing, no FLOPs)
        let mut input_data: Vec<half::f16> = vec![half::f16::ZERO; n_points * input_dim];
        for iz in 0..res {
            for iy in 0..res {
                for ix in 0..res {
                    let idx = iz * res * res + iy * res + ix;
                    let x = (ix as f32 / (res - 1) as f32) * 2.0 - 1.0;
                    let y = (iy as f32 / (res - 1) as f32) * 2.0 - 1.0;
                    let z = (iz as f32 / (res - 1) as f32) * 2.0 - 1.0;

                    let token_idx = ((x + 1.0) * 0.5 * (config.num_latent_tokens - 1) as f32)
                        .clamp(0.0, (config.num_latent_tokens - 1) as f32) as usize;

                    for c in 0..config.latent_channels {
                        input_data[idx * input_dim + c] = latent_data[token_idx * config.latent_channels + c];
                    }
                    input_data[idx * input_dim + config.latent_channels] = half::f16::from_f32(x);
                    input_data[idx * input_dim + config.latent_channels + 1] = half::f16::from_f32(y);
                    input_data[idx * input_dim + config.latent_channels + 2] = half::f16::from_f32(z);
                }
            }
        }

        let mut h = Tensor::from_slice(&input_data, Shape::from([n_points, input_dim]), DType::F16, device_id)?;

        // GPU MLP: linear → ReLU → ... → linear (batched across all grid points)
        let mut in_dim = input_dim;
        for i in 0..config.sdf_num_layers {
            let w_key = format!("decoder.layers.{}.weight", i);
            let b_key = format!("decoder.layers.{}.bias", i);
            let w_f16 = self.weight_f16(&self.vae_model, &w_key)?;
            let out_dim = w_f16.shape().dims()[0];

            let cb = self.compute.new_command_buffer();
            let projected = self.linear_bias(&cb, &self.vae_model, &h, &w_key, &b_key, n_points, in_dim, out_dim)?;
            if i < config.sdf_num_layers - 1 {
                h = self.activation(&cb, &self.kernels.relu, &projected);
            } else {
                h = projected;
            }
            cb.commit();
            cb.wait_until_completed();
            in_dim = out_dim;
        }

        // Read back [N, 1] → Vec<f32>
        let sdf_f16: Vec<half::f16> = h.to_vec()?;
        Ok(sdf_f16.iter().map(|v| v.to_f32()).collect())
    }

    // ==================== GPU Helper Methods ====================

    fn linear_on(
        &self, cb: &metal::CommandBufferRef, model: &Arc<Model>, input: &Tensor,
        weight_name: &str, m: usize, k: usize, n: usize,
    ) -> Result<Tensor> {
        let w_f16 = self.weight_f16(model, weight_name)?;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let tile: usize = 16;
        self.compute.dispatch(
            cb, &self.kernels.common.linear,
            ((n + tile - 1) / tile, (m + tile - 1) / tile, 1), (tile, tile, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &w_f16);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 4] = [m as u32, n as u32, k as u32, 0];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, self.compute.device().info().id))
    }

}

// ==================== Utility ====================

/// Deterministic pseudo-random normal samples (Box-Muller transform).
fn deterministic_randn(n: usize, seed: u64) -> Vec<f32> {
    let mut rng_state = seed;
    let mut output = Vec::with_capacity(n);
    for _ in 0..(n + 1) / 2 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
        let u1 = u1.max(1e-10);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        output.push((r * theta.cos()) as f32);
        output.push((r * theta.sin()) as f32);
    }
    output.truncate(n);
    output
}
