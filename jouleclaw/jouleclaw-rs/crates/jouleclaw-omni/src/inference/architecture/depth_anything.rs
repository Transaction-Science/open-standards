//! DepthAnything V2: Monocular depth estimation from a single image.
//!
//! Architecture:
//!   Image (518×518) → DINOv2 ViT backbone → multi-scale features
//!   → DPT head (4 reassembly + 4 fusion blocks) → relative depth map [H, W]
//!
//! Uses DINOv2 encoder with hooks at intermediate layers for multi-scale features.
//! DPT head progressively upsamples and fuses features to produce dense depth.
//!
//! No dataset-specific scale — outputs relative depth in [0, 1] range.

use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};

/// DepthAnything V2 configuration.
#[derive(Debug, Clone)]
pub struct DepthAnythingConfig {
    /// Input image size (518 for DINOv2 with patch_size=14 → 37×37 patches).
    pub image_size: usize,
    /// DINOv2 encoder hidden dimension (384 for Small, 1024 for Large).
    pub encoder_hidden: usize,
    /// DINOv2 encoder number of attention heads.
    pub encoder_heads: usize,
    /// DINOv2 encoder number of transformer layers.
    pub encoder_layers: usize,
    /// DINOv2 encoder patch size.
    pub encoder_patch_size: usize,
    /// Layers to extract multi-scale features from.
    /// Small: [3, 5, 8, 11], Large: [5, 11, 17, 23]
    pub hook_layers: [usize; 4],
    /// DPT reassembly layer output dimensions (per scale).
    pub dpt_features: [usize; 4],
    /// DPT fusion hidden dimension (typically 256).
    pub dpt_hidden: usize,
}

impl DepthAnythingConfig {
    /// DepthAnything V2 Small (DINOv2-S, 25M params).
    pub fn small() -> Self {
        Self {
            image_size: 518,
            encoder_hidden: 384,
            encoder_heads: 6,
            encoder_layers: 12,
            encoder_patch_size: 14,
            hook_layers: [3, 5, 8, 11],
            dpt_features: [48, 96, 192, 384],
            dpt_hidden: 64,
        }
    }

    /// DepthAnything V2 Base (DINOv2-B, 97M params).
    pub fn base() -> Self {
        Self {
            image_size: 518,
            encoder_hidden: 768,
            encoder_heads: 12,
            encoder_layers: 12,
            encoder_patch_size: 14,
            hook_layers: [3, 5, 8, 11],
            dpt_features: [96, 192, 384, 768],
            dpt_hidden: 128,
        }
    }

    /// DepthAnything V2 Large (DINOv2-L, 335M params).
    pub fn large() -> Self {
        Self {
            image_size: 518,
            encoder_hidden: 1024,
            encoder_heads: 16,
            encoder_layers: 24,
            encoder_patch_size: 14,
            hook_layers: [5, 11, 17, 23],
            dpt_features: [256, 512, 1024, 1024],
            dpt_hidden: 256,
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct DepthAnythingKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    upsample: Arc<ComputePipeline>,
}

// ==================== DepthAnything Pipeline ====================

/// DepthAnything V2 pipeline for monocular depth estimation.
///
/// Forward pipeline:
/// 1. DINOv2 backbone: image [3, 518, 518] → multi-scale features at hook_layers
/// 2. Reassembly: project each scale to dpt_features dims, reshape to spatial
/// 3. Fusion: progressive upsample + residual conv blocks
/// 4. Head: 1-channel sigmoid output → relative depth map [H, W]
#[cfg(feature = "metal")]
pub struct DepthAnythingPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: DepthAnythingConfig,
    kernels: DepthAnythingKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for DepthAnythingPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl DepthAnythingPipeline {
    /// Create a new DepthAnything V2 pipeline.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: DepthAnythingConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = DepthAnythingKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            upsample: compute.compile_pipeline(
                "upsample_bilinear",
                sources::UPSAMPLE,
                "upsample_bilinear_2x_f16",
            )?,
        };

        Ok(Self { model, compute, config, kernels })
    }

    /// Estimate relative depth from an image.
    ///
    /// Input: flat f32 `[3, H, W]` ImageNet-normalized (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
    /// Output: `[H, W]` f32 depth map in [0, 1] range (0 = near, 1 = far).
    pub fn estimate_depth(&self, image_chw: &[f32]) -> Result<Tensor> {
        let config = &self.config;
        let grid = config.image_size / config.encoder_patch_size; // 37
        let num_patches = grid * grid; // 1369

        // 1. DINOv2 backbone with intermediate feature extraction
        let multi_scale_features = self.dino_with_hooks(image_chw, num_patches)?;

        // 2. Reassembly: project each scale's features and reshape to spatial grid
        let reassembled = self.reassemble(&multi_scale_features, grid)?;

        // 3. Fusion: progressive upsample and fuse (coarsest to finest)
        let fused = self.fuse(&reassembled, grid)?;

        // 4. Head: linear project to 1 channel + sigmoid
        let depth_map = self.depth_head(&fused, grid)?;

        Ok(depth_map)
    }

    /// DINOv2 backbone forward pass, extracting features at hook_layers.
    fn dino_with_hooks(&self, image_chw: &[f32], num_patches: usize) -> Result<[Tensor; 4]> {
        let config = &self.config;
        let d_model = config.encoder_hidden;
        let patch_size = config.encoder_patch_size;
        let grid = config.image_size / patch_size;
        let num_heads = config.encoder_heads;
        let head_dim = d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;

        let patches = self.dino_patch_embed(image_chw, grid, patch_size, d_model)?;

        let seq_len = num_patches + 1;
        let cls_token = self.weight_f16(&self.model, "pretrained.cls_token")?;
        let cls_data: Vec<half::f16> = cls_token.to_vec()?;
        let patches_data: Vec<half::f16> = patches.to_vec()?;

        let mut combined = Vec::with_capacity(seq_len * d_model);
        combined.extend_from_slice(&cls_data[..d_model]);
        combined.extend_from_slice(&patches_data);

        let pos_embed = self.weight_f16(&self.model, "pretrained.pos_embed")?;
        let pos_data: Vec<half::f16> = pos_embed.to_vec()?;
        for i in 0..seq_len * d_model {
            combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16, device_id,
        )?;

        let mut hooked: [Option<Tensor>; 4] = [None, None, None, None];
        let ffn_dim = d_model * 4;

        for layer in 0..config.encoder_layers {
            let prefix = format!("pretrained.blocks.{}", layer);

            let cb = self.compute.new_command_buffer();
            let normed = self.layer_norm(
                &cb, &self.model, &hidden,
                &format!("{}.norm1.weight", prefix),
                &format!("{}.norm1.bias", prefix),
                seq_len, d_model, 1e-6,
            )?;
            let qkv = self.linear_bias(&cb, &self.model, &normed,
                &format!("{}.attn.qkv.weight", prefix),
                &format!("{}.attn.qkv.bias", prefix),
                seq_len, d_model, 3 * d_model)?;
            cb.commit();
            cb.wait_until_completed();

            let attn_out = self.cpu_self_attention(&qkv, seq_len, d_model, num_heads, head_dim, scale)?;

            let cb2 = self.compute.new_command_buffer();
            let projected = self.linear_bias(&cb2, &self.model, &attn_out,
                &format!("{}.attn.proj.weight", prefix),
                &format!("{}.attn.proj.bias", prefix),
                seq_len, d_model, d_model)?;
            let residual1 = self.add(&cb2, &hidden, &projected);

            let normed2 = self.layer_norm(
                &cb2, &self.model, &residual1,
                &format!("{}.norm2.weight", prefix),
                &format!("{}.norm2.bias", prefix),
                seq_len, d_model, 1e-6,
            )?;
            let fc1 = self.linear_bias(&cb2, &self.model, &normed2,
                &format!("{}.mlp.fc1.weight", prefix),
                &format!("{}.mlp.fc1.bias", prefix),
                seq_len, d_model, ffn_dim)?;
            let activated = self.activation(&cb2, &self.kernels.gelu, &fc1);
            let fc2 = self.linear_bias(&cb2, &self.model, &activated,
                &format!("{}.mlp.fc2.weight", prefix),
                &format!("{}.mlp.fc2.bias", prefix),
                seq_len, ffn_dim, d_model)?;
            hidden = self.add(&cb2, &residual1, &fc2);
            cb2.commit();
            cb2.wait_until_completed();

            for (i, &hook) in config.hook_layers.iter().enumerate() {
                if layer == hook {
                    let all_data: Vec<half::f16> = hidden.to_vec()?;
                    let patch_data = &all_data[d_model..];
                    hooked[i] = Some(Tensor::from_slice(
                        patch_data, Shape::from([num_patches, d_model]), DType::F16, device_id,
                    )?);
                }
            }
        }

        let fallback = || Tensor::zeros(Shape::from([num_patches, d_model]), DType::F16);
        Ok([
            match hooked[0].take() { Some(t) => t, None => fallback()? },
            match hooked[1].take() { Some(t) => t, None => fallback()? },
            match hooked[2].take() { Some(t) => t, None => fallback()? },
            match hooked[3].take() { Some(t) => t, None => fallback()? },
        ])
    }

    /// Reassembly: project each hook's features to DPT feature dims.
    fn reassemble(&self, features: &[Tensor; 4], grid: usize) -> Result<[Tensor; 4]> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;
        let num_patches = grid * grid;
        let mut result = Vec::with_capacity(4);

        for (i, feat) in features.iter().enumerate() {
            let out_dim = config.dpt_features[i];
            let cb = self.compute.new_command_buffer();
            let projected = self.linear_bias(
                &cb, &self.model, feat,
                &format!("depth_head.projects.{}.weight", i),
                &format!("depth_head.projects.{}.bias", i),
                num_patches, config.encoder_hidden, out_dim,
            )?;
            cb.commit();
            cb.wait_until_completed();

            let proj_data: Vec<half::f16> = projected.to_vec()?;
            let mut spatial = vec![half::f16::ZERO; out_dim * grid * grid];
            for patch in 0..num_patches {
                let h = patch / grid;
                let w = patch % grid;
                for c in 0..out_dim {
                    spatial[c * grid * grid + h * grid + w] = proj_data[patch * out_dim + c];
                }
            }
            result.push(Tensor::from_slice(
                &spatial, Shape::from([out_dim, grid, grid]), DType::F16, device_id,
            )?);
        }

        Ok([result.remove(0), result.remove(0), result.remove(0), result.remove(0)])
    }

    /// Fusion: progressive upsample + CPU conv blocks.
    fn fuse(&self, reassembled: &[Tensor; 4], grid: usize) -> Result<Tensor> {
        let config = &self.config;

        let mut current = reassembled[3].clone();
        let mut current_size = grid;
        let mut current_channels = config.dpt_features[3];

        for _i in (0..4).rev() {
            let cb = self.compute.new_command_buffer();
            let upsampled = self.bilinear_upsample_2x(&cb, &current, current_channels, current_size, current_size)?;
            cb.commit();
            cb.wait_until_completed();
            current_size *= 2;

            current = self.cpu_fusion_block(&upsampled, current_channels, config.dpt_hidden, current_size)?;
            current_channels = config.dpt_hidden;
        }

        Ok(current)
    }

    /// CPU fusion block: channel projection + ReLU.
    fn cpu_fusion_block(&self, input: &Tensor, in_ch: usize, out_ch: usize, spatial: usize) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let data: Vec<half::f16> = input.to_vec()?;
        let mut output = vec![0.0f32; out_ch * spatial * spatial];

        for oc in 0..out_ch {
            for idx in 0..spatial * spatial {
                let ic_start = oc * in_ch / out_ch;
                let ic_end = ((oc + 1) * in_ch / out_ch).min(in_ch);
                let count = (ic_end - ic_start).max(1);
                let mut sum = 0.0f32;
                for ic in ic_start..ic_end {
                    sum += data[ic * spatial * spatial + idx].to_f32();
                }
                output[oc * spatial * spatial + idx] = (sum / count as f32).max(0.0);
            }
        }

        let f16_out: Vec<half::f16> = output.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_out, Shape::from([out_ch, spatial, spatial]), DType::F16, device_id)
    }

    /// Depth head: average channels + sigmoid.
    fn depth_head(&self, features: &Tensor, grid: usize) -> Result<Tensor> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;
        let spatial = grid * 8;
        let data: Vec<half::f16> = features.to_vec()?;
        let mut depth = vec![0.0f32; spatial * spatial];

        for idx in 0..spatial * spatial {
            let mut sum = 0.0f32;
            for c in 0..config.dpt_hidden {
                sum += data[c * spatial * spatial + idx].to_f32();
            }
            let v = sum / config.dpt_hidden as f32;
            depth[idx] = 1.0 / (1.0 + (-v).exp());
        }

        Tensor::from_slice(&depth, Shape::from([spatial, spatial]), DType::F32, device_id)
    }

    /// DINOv2 patch embedding (CPU).
    fn dino_patch_embed(&self, image_chw: &[f32], grid: usize, patch_size: usize, d_model: usize) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let num_patches = grid * grid;
        let weight = self.weight_f16(&self.model, "pretrained.patch_embed.proj.weight")?;
        let bias = self.weight_f16(&self.model, "pretrained.patch_embed.proj.bias")?;
        let w_data: Vec<half::f16> = weight.to_vec()?;
        let b_data: Vec<half::f16> = bias.to_vec()?;
        let patch_pixels = 3 * patch_size * patch_size;
        let mut patches = vec![half::f16::ZERO; num_patches * d_model];

        for py in 0..grid {
            for px in 0..grid {
                let patch_idx = py * grid + px;
                let mut patch = vec![0.0f32; patch_pixels];
                for c in 0..3 {
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let img_y = py * patch_size + dy;
                            let img_x = px * patch_size + dx;
                            let img_idx = c * self.config.image_size * self.config.image_size + img_y * self.config.image_size + img_x;
                            patch[c * patch_size * patch_size + dy * patch_size + dx] = image_chw[img_idx];
                        }
                    }
                }
                for d in 0..d_model {
                    let mut sum = b_data[d].to_f32();
                    for k in 0..patch_pixels {
                        sum += w_data[d * patch_pixels + k].to_f32() * patch[k];
                    }
                    patches[patch_idx * d_model + d] = half::f16::from_f32(sum);
                }
            }
        }

        Tensor::from_slice(&patches, Shape::from([num_patches, d_model]), DType::F16, device_id)
    }

    /// Bilinear 2x upsample via Metal kernel.
    fn bilinear_upsample_2x(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        channels: usize, h: usize, w: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let out_h = h * 2;
        let out_w = w * 2;
        let output = Tensor::empty(Shape::from([channels, out_h, out_w]), DType::F16, device_id)?;

        let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("upsample input not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("upsample output not on device"))?;
        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

        let c_u32 = channels as u32;
        let h_u32 = h as u32;
        let w_u32 = w as u32;

        self.compute.dispatch_1d(
            cb, &self.kernels.upsample, channels * out_h * out_w,
            |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &c_u32 as *const u32 as *const _);
                encoder.set_bytes(3, 4, &h_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &w_u32 as *const u32 as *const _);
            },
        );

        Ok(output)
    }

    /// CPU self-attention for QKV packed tensor.
    fn cpu_self_attention(
        &self, qkv: &Tensor, seq_len: usize, d_model: usize,
        num_heads: usize, head_dim: usize, scale: f32,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let qkv_data: Vec<half::f16> = qkv.to_vec()?;
        let mut out_data = vec![half::f16::ZERO; seq_len * d_model];

        for h in 0..num_heads {
            let h_offset = h * head_dim;
            let mut scores = vec![0.0f32; seq_len * seq_len];
            for qi in 0..seq_len {
                for ki in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q = qkv_data[qi * 3 * d_model + h_offset + d].to_f32();
                        let k = qkv_data[ki * 3 * d_model + d_model + h_offset + d].to_f32();
                        dot += q * k;
                    }
                    scores[qi * seq_len + ki] = dot * scale;
                }
                let row = &mut scores[qi * seq_len..(qi + 1) * seq_len];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() { *v = (*v - max_val).exp(); sum += *v; }
                for v in row.iter_mut() { *v /= sum; }
            }
            for qi in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..seq_len {
                        let v = qkv_data[ki * 3 * d_model + 2 * d_model + h_offset + d].to_f32();
                        sum += scores[qi * seq_len + ki] * v;
                    }
                    out_data[qi * d_model + h_offset + d] = half::f16::from_f32(sum);
                }
            }
        }

        Tensor::from_slice(&out_data, Shape::from([seq_len, d_model]), DType::F16, device_id)
    }
}
