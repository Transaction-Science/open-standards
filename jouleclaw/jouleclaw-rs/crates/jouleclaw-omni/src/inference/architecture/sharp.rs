//! Apple SHARP: Single-image to 3D Gaussian splats in a single feedforward pass.
//!
//! Architecture:
//!   Image (512×512) → Vision encoder (DINO ViT) → spatial features [N, D]
//!   → Depth estimator → depth map
//!   → Gaussian predictor MLP → [500K, 14] (xyz + scale3 + rot4 + rgb3 + opacity1)
//!
//! Single feedforward pass, no iterative diffusion/flow.
//! Based on Apple/SHARP (2025).

use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};

// Re-use GaussianOutput from Trellis
use super::trellis::GaussianOutput;

/// Apple SHARP configuration.
#[derive(Debug, Clone)]
pub struct SharpConfig {
    /// Input image size.
    pub image_size: usize,
    /// Vision encoder hidden dimension.
    pub encoder_hidden: usize,
    /// Vision encoder number of heads.
    pub encoder_heads: usize,
    /// Vision encoder number of layers.
    pub encoder_layers: usize,
    /// Vision encoder patch size.
    pub encoder_patch_size: usize,
    /// Depth estimator hidden dimension.
    pub depth_hidden: usize,
    /// Depth estimator number of layers.
    pub depth_layers: usize,
    /// Gaussian predictor hidden dimension.
    pub predictor_hidden: usize,
    /// Gaussian predictor number of layers.
    pub predictor_layers: usize,
    /// Number of output Gaussians.
    pub num_gaussians: usize,
    /// Parameters per Gaussian (xyz=3 + scale=3 + rot=4 + rgb=3 + opacity=1 = 14).
    pub params_per_gaussian: usize,
}

impl Default for SharpConfig {
    fn default() -> Self {
        Self {
            image_size: 512,
            encoder_hidden: 768,
            encoder_heads: 12,
            encoder_layers: 12,
            encoder_patch_size: 16,
            depth_hidden: 256,
            depth_layers: 4,
            predictor_hidden: 256,
            predictor_layers: 4,
            num_gaussians: 500_000,
            params_per_gaussian: 14,
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct SharpKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
}

// ==================== SHARP Pipeline ====================

/// Apple SHARP pipeline for single-image to 3D Gaussian splats.
///
/// Forward pipeline:
/// 1. Vision encoder (DINO ViT): image → spatial features [N, D]
/// 2. Depth estimator: features → depth map [H, W]
/// 3. Gaussian predictor: features + depth → [num_gaussians, 14]
/// 4. Split: position(3) + scale(3) + rotation(4) + color(3) + opacity(1)
/// 5. Return GaussianOutput
#[cfg(feature = "metal")]
pub struct SharpPipeline {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: SharpConfig,
    kernels: SharpKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for SharpPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl SharpPipeline {
    /// Create a new SHARP pipeline.
    pub fn new(
        model: Arc<Model>,
        config: SharpConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = SharpKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            silu: compute.compile_pipeline("silu", sources::GELU, "silu_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
        };

        Ok(Self { model, compute, config, kernels })
    }

    /// Generate 3D Gaussians from a single image.
    ///
    /// `image_chw`: Image as flat f32 RGB array [3 * image_size * image_size] in [C, H, W],
    ///              normalized to ImageNet stats.
    pub fn generate(&self, image_chw: &[f32]) -> Result<GaussianOutput> {
        let config = &self.config;

        println!("  [SHARP] Encoding image with vision encoder...");
        let features = self.vision_encode(image_chw)?;
        let grid = config.image_size / config.encoder_patch_size; // 32
        let num_patches = grid * grid; // 1024

        println!("  [SHARP] Estimating depth...");
        let depth = self.depth_estimate(&features, num_patches)?;

        println!("  [SHARP] Predicting {} Gaussians...", config.num_gaussians);
        let gaussians = self.predict_gaussians(&features, &depth, num_patches)?;

        // Split raw [N, 14] → position(3) + scale(3) + rotation(4) + color(3) + opacity(1)
        let n = config.num_gaussians;
        let mut positions = Vec::with_capacity(n * 3);
        let mut scales = Vec::with_capacity(n * 3);
        let mut rotations = Vec::with_capacity(n * 4);
        let mut colors = Vec::with_capacity(n * 3);
        let mut opacities = Vec::with_capacity(n);

        for i in 0..n {
            let base = i * config.params_per_gaussian;
            // Position (xyz)
            positions.push(gaussians[base]);
            positions.push(gaussians[base + 1]);
            positions.push(gaussians[base + 2]);
            // Scale (exp activation)
            scales.push(gaussians[base + 3].exp());
            scales.push(gaussians[base + 4].exp());
            scales.push(gaussians[base + 5].exp());
            // Rotation (normalize quaternion)
            let qw = gaussians[base + 6];
            let qx = gaussians[base + 7];
            let qy = gaussians[base + 8];
            let qz = gaussians[base + 9];
            let qnorm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt().max(1e-8);
            rotations.push(qw / qnorm);
            rotations.push(qx / qnorm);
            rotations.push(qy / qnorm);
            rotations.push(qz / qnorm);
            // Color (sigmoid activation)
            colors.push(sigmoid(gaussians[base + 10]));
            colors.push(sigmoid(gaussians[base + 11]));
            colors.push(sigmoid(gaussians[base + 12]));
            // Opacity (sigmoid)
            opacities.push(sigmoid(gaussians[base + 13]));
        }

        Ok(GaussianOutput {
            positions,
            colors,
            scales,
            rotations,
            opacities,
            num_gaussians_per_voxel: 1,
        })
    }

    // ==================== Vision Encoder ====================

    /// DINO ViT encoder: image → [N, D] spatial features.
    fn vision_encode(&self, image_chw: &[f32]) -> Result<Tensor> {
        let d_model = self.config.encoder_hidden; // 768
        let patch_size = self.config.encoder_patch_size; // 16
        let grid = self.config.image_size / patch_size; // 32
        let num_patches = grid * grid; // 1024
        let num_heads = self.config.encoder_heads; // 12
        let head_dim = d_model / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = num_patches + 1; // 1025 (CLS + patches)
        let device_id = self.compute.device().info().id;

        // Patch embedding
        let patches = self.patch_embed(image_chw, grid, patch_size, d_model)?;

        // CLS token + position embeddings
        let cls_token = self.weight_f16(&self.model, "encoder.cls_token")?;
        let cls_data: Vec<half::f16> = cls_token.to_vec()?;
        let patches_data: Vec<half::f16> = patches.to_vec()?;

        let mut combined = Vec::with_capacity(seq_len * d_model);
        combined.extend_from_slice(&cls_data[..d_model]);
        combined.extend_from_slice(&patches_data);

        let pos_embed = self.weight_f16(&self.model, "encoder.position_embeddings")?;
        let pos_data: Vec<half::f16> = pos_embed.to_vec()?;
        for i in 0..(seq_len * d_model).min(pos_data.len()) {
            combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16, device_id,
        )?;

        // Encoder layers
        let ffn_dim = d_model * 4;
        for layer in 0..self.config.encoder_layers {
            let prefix = format!("encoder.layer.{}", layer);
            let cb = self.compute.new_command_buffer();

            let normed = self.layer_norm(&cb, &self.model, &hidden,
                &format!("{}.norm1.weight", prefix),
                &format!("{}.norm1.bias", prefix),
                seq_len, d_model, 1e-5)?;

            let q = self.linear_bias(&cb, &self.model, &normed,
                &format!("{}.attn.q.weight", prefix),
                &format!("{}.attn.q.bias", prefix),
                seq_len, d_model, d_model)?;
            let k = self.linear_bias(&cb, &self.model, &normed,
                &format!("{}.attn.k.weight", prefix),
                &format!("{}.attn.k.bias", prefix),
                seq_len, d_model, d_model)?;
            let v = self.linear_bias(&cb, &self.model, &normed,
                &format!("{}.attn.v.weight", prefix),
                &format!("{}.attn.v.bias", prefix),
                seq_len, d_model, d_model)?;

            let attn_out = self.batched_attention(&cb, &q, &k, &v, seq_len, seq_len, num_heads, head_dim, scale)?;
            let proj = self.linear_bias(&cb, &self.model, &attn_out,
                &format!("{}.attn.proj.weight", prefix),
                &format!("{}.attn.proj.bias", prefix),
                seq_len, d_model, d_model)?;
            let h = self.add(&cb, &hidden, &proj);

            let normed2 = self.layer_norm(&cb, &self.model, &h,
                &format!("{}.norm2.weight", prefix),
                &format!("{}.norm2.bias", prefix),
                seq_len, d_model, 1e-5)?;
            let ffn_up = self.linear_bias(&cb, &self.model, &normed2,
                &format!("{}.mlp.fc1.weight", prefix),
                &format!("{}.mlp.fc1.bias", prefix),
                seq_len, d_model, ffn_dim)?;
            let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
            let ffn_down = self.linear_bias(&cb, &self.model, &ffn_act,
                &format!("{}.mlp.fc2.weight", prefix),
                &format!("{}.mlp.fc2.bias", prefix),
                seq_len, ffn_dim, d_model)?;
            hidden = self.add(&cb, &h, &ffn_down);

            cb.commit();
            cb.wait_until_completed();
        }

        // Remove CLS → [num_patches, D]
        hidden.slice(0, 1, seq_len)
    }

    /// Patch embedding: im2col (CPU memory layout) → GPU matmul.
    fn patch_embed(&self, image_chw: &[f32], grid: usize, patch_size: usize, d_model: usize) -> Result<Tensor> {
        let c_in = 3;
        let num_patches = grid * grid;
        let patch_dim = c_in * patch_size * patch_size;
        let img_size = self.config.image_size;

        // im2col: extract non-overlapping patches → [num_patches, patch_dim]
        let mut im2col_data = vec![half::f16::ZERO; num_patches * patch_dim];
        for gy in 0..grid {
            for gx in 0..grid {
                let patch_idx = gy * grid + gx;
                for ic in 0..c_in {
                    for ky in 0..patch_size {
                        for kx in 0..patch_size {
                            let iy = gy * patch_size + ky;
                            let ix = gx * patch_size + kx;
                            let col = ic * patch_size * patch_size + ky * patch_size + kx;
                            if iy < img_size && ix < img_size {
                                im2col_data[patch_idx * patch_dim + col] =
                                    half::f16::from_f32(image_chw[ic * img_size * img_size + iy * img_size + ix]);
                            }
                        }
                    }
                }
            }
        }

        let device_id = self.compute.device().info().id;
        let input_tensor = Tensor::from_slice(&im2col_data, Shape::from([num_patches, patch_dim]), DType::F16, device_id)?;

        // GPU matmul: [num_patches, patch_dim] @ W^T + bias → [num_patches, d_model]
        let cb = self.compute.new_command_buffer();
        let result = self.linear_bias(&cb, &self.model, &input_tensor,
            "encoder.patch_embed.proj.weight", "encoder.patch_embed.proj.bias",
            num_patches, patch_dim, d_model)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    // ==================== Depth Estimator ====================

    /// Depth estimator: spatial features → depth map (batched GPU MLP).
    fn depth_estimate(&self, features: &Tensor, num_patches: usize) -> Result<Vec<f32>> {
        let config = &self.config;

        // GPU MLP: [num_patches, encoder_hidden] → layers → [num_patches, 1]
        let mut h = features.clone();
        let mut in_dim = config.encoder_hidden;
        for layer in 0..config.depth_layers {
            let w_key = format!("depth.layers.{}.weight", layer);
            let b_key = format!("depth.layers.{}.bias", layer);
            let w_f16 = self.weight_f16(&self.model, &w_key)?;
            let out_dim = w_f16.shape().dims()[0];
            let cb = self.compute.new_command_buffer();
            let projected = self.linear_bias(&cb, &self.model, &h, &w_key, &b_key, num_patches, in_dim, out_dim)?;
            if layer < config.depth_layers - 1 {
                h = self.activation(&cb, &self.kernels.relu, &projected);
            } else {
                h = projected;
            }
            cb.commit();
            cb.wait_until_completed();
            in_dim = out_dim;
        }

        // Read back depth values
        let output_data: Vec<half::f16> = h.to_vec()?;
        let depth: Vec<f32> = (0..num_patches).map(|p| output_data[p].to_f32()).collect();
        Ok(depth)
    }

    // ==================== Gaussian Predictor ====================

    /// Gaussian predictor: features + depth → [N, 14] raw params (batched GPU MLP).
    fn predict_gaussians(
        &self, features: &Tensor, depth: &[f32], num_patches: usize,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let feat_data: Vec<half::f16> = features.to_vec()?;
        let d_model = config.encoder_hidden;
        let n_gauss = config.num_gaussians;
        let n_params = config.params_per_gaussian;
        let gaussians_per_patch = n_gauss / num_patches;
        let input_dim = d_model + 1; // features + depth

        // Build batched input: [num_patches, d_model + 1] on CPU (memory layout only)
        let mut input_data = vec![half::f16::ZERO; num_patches * input_dim];
        for p in 0..num_patches {
            for d in 0..d_model {
                input_data[p * input_dim + d] = feat_data[p * d_model + d];
            }
            input_data[p * input_dim + d_model] = half::f16::from_f32(depth[p]);
        }

        let device_id = self.compute.device().info().id;
        let mut h = Tensor::from_slice(&input_data, Shape::from([num_patches, input_dim]), DType::F16, device_id)?;
        let mut in_dim = input_dim;

        // GPU MLP layers
        for layer in 0..config.predictor_layers {
            let w_key = format!("predictor.layers.{}.weight", layer);
            let b_key = format!("predictor.layers.{}.bias", layer);
            let w_f16 = self.weight_f16(&self.model, &w_key)?;
            let out_dim = w_f16.shape().dims()[0];
            let cb = self.compute.new_command_buffer();
            let projected = self.linear_bias(&cb, &self.model, &h, &w_key, &b_key, num_patches, in_dim, out_dim)?;
            if layer < config.predictor_layers - 1 {
                h = self.activation(&cb, &self.kernels.relu, &projected);
            } else {
                h = projected;
            }
            cb.commit();
            cb.wait_until_completed();
            in_dim = out_dim;
        }

        // Read back and distribute to gaussians
        let output_data: Vec<half::f16> = h.to_vec()?;
        let out_dim = in_dim; // final output dim
        let mut all_params = vec![0.0f32; n_gauss * n_params];
        for p in 0..num_patches {
            let base_gauss = p * gaussians_per_patch;
            for g in 0..gaussians_per_patch {
                let gauss_idx = base_gauss + g;
                if gauss_idx < n_gauss {
                    for param in 0..n_params {
                        let src_idx = g * n_params + param;
                        if src_idx < out_dim {
                            all_params[gauss_idx * n_params + param] = output_data[p * out_dim + src_idx].to_f32();
                        }
                    }
                }
            }
        }

        Ok(all_params)
    }

}

// ==================== Utility ====================

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
