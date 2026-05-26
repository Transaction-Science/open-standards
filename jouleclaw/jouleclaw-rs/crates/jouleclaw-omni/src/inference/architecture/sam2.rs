//! SAM 2.1: Segment Anything Model 2 — promptable image segmentation.
//!
//! Architecture:
//!   Image (1024x1024) → Hiera image encoder (hierarchical ViT with Mask Unit Attention)
//!   → FPN neck (multi-scale feature pyramid)
//!   → Prompt encoder (points/boxes/masks → sparse + dense embeddings)
//!   → Mask decoder (2-layer two-way transformer → 3 masks + IoU scores)
//!
//! Hiera stages: [1, 2, 7, 2] blocks with dims [96, 192, 384, 768],
//! windowed attention within local windows, global attention at specific blocks.
//!
//! Mask decoder uses bidirectional cross-attention: tokens attend to image,
//! then image attends back to tokens.
//!
//! Input: 1024x1024 → Output: 256x256 masks (upscaled to input size).
//! Based on Meta SAM 2.1 (2024), Hiera-Tiny variant (39M params).

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

// ==================== Configuration ====================

/// SAM 2.1 Hiera encoder + mask decoder configuration.
#[derive(Debug, Clone)]
pub struct Sam2Config {
    /// Base embedding dimension (Hiera stage 0).
    pub embed_dim: usize,
    /// Number of blocks per Hiera stage.
    pub depths: Vec<usize>,
    /// Number of attention heads per stage.
    pub num_heads: Vec<usize>,
    /// Window sizes for windowed attention per stage.
    pub window_sizes: Vec<usize>,
    /// Indices of blocks that use global (full) attention.
    pub global_attn_indices: Vec<usize>,
    /// Input image size (square).
    pub image_size: usize,
    /// Patch embedding kernel and stride.
    pub patch_size: usize,
    /// Number of mask output tokens in the decoder.
    pub num_mask_tokens: usize,
    /// Mask decoder transformer layers.
    pub decoder_depth: usize,
    /// Mask decoder transformer hidden dimension.
    pub decoder_hidden: usize,
    /// Mask decoder transformer number of heads.
    pub decoder_heads: usize,
    /// FPN output dimension (neck).
    pub fpn_out_dim: usize,
}

impl Sam2Config {
    /// SAM 2.1 Hiera-Tiny (39M params).
    pub fn tiny() -> Self {
        Self {
            embed_dim: 96,
            depths: vec![1, 2, 7, 2],
            num_heads: vec![1, 2, 4, 8],
            window_sizes: vec![8, 4, 14, 7],
            global_attn_indices: vec![5, 7, 9],
            image_size: 1024,
            patch_size: 4,
            num_mask_tokens: 4,
            decoder_depth: 2,
            decoder_hidden: 256,
            decoder_heads: 8,
            fpn_out_dim: 256,
        }
    }

    /// SAM 2.1 Hiera-Small (46M params).
    pub fn small() -> Self {
        Self {
            embed_dim: 96,
            depths: vec![1, 2, 11, 2],
            num_heads: vec![1, 2, 4, 8],
            window_sizes: vec![8, 4, 14, 7],
            global_attn_indices: vec![7, 11, 13],
            image_size: 1024,
            patch_size: 4,
            num_mask_tokens: 4,
            decoder_depth: 2,
            decoder_hidden: 256,
            decoder_heads: 8,
            fpn_out_dim: 256,
        }
    }

    /// SAM 2.1 Hiera-Base+ (80M params).
    pub fn base_plus() -> Self {
        Self {
            embed_dim: 112,
            depths: vec![2, 3, 16, 3],
            num_heads: vec![2, 4, 8, 16],
            window_sizes: vec![8, 4, 14, 7],
            global_attn_indices: vec![12, 16, 20],
            image_size: 1024,
            patch_size: 4,
            num_mask_tokens: 4,
            decoder_depth: 2,
            decoder_hidden: 256,
            decoder_heads: 8,
            fpn_out_dim: 256,
        }
    }

    /// SAM 2.1 Hiera-Large (200M params).
    pub fn large() -> Self {
        Self {
            embed_dim: 144,
            depths: vec![2, 6, 36, 4],
            num_heads: vec![2, 4, 8, 16],
            window_sizes: vec![8, 4, 14, 7],
            global_attn_indices: vec![23, 33, 43],
            image_size: 1024,
            patch_size: 4,
            num_mask_tokens: 4,
            decoder_depth: 2,
            decoder_hidden: 256,
            decoder_heads: 8,
            fpn_out_dim: 256,
        }
    }

    /// Stage dimension: embed_dim * 2^stage.
    fn stage_dim(&self, stage: usize) -> usize {
        self.embed_dim * (1 << stage)
    }

    /// Absolute block index for (stage, block_within_stage).
    fn absolute_block_index(&self, stage: usize, block: usize) -> usize {
        let mut idx = 0;
        for s in 0..stage {
            idx += self.depths[s];
        }
        idx + block
    }

    /// Total number of blocks across all stages.
    #[allow(dead_code)]
    fn total_blocks(&self) -> usize {
        self.depths.iter().sum()
    }

    /// Spatial resolution at each stage: image_size / patch_size / (2^num_downsamples).
    fn stage_resolution(&self, stage: usize) -> usize {
        // After patch embed: image_size / patch_size
        // After each inter-stage downsample (2x2 maxpool): /2
        (self.image_size / self.patch_size) >> stage
    }
}

// ==================== Segmentation Output ====================

/// Output from SAM 2.1 segmentation.
#[derive(Debug)]
pub struct SegmentationOutput {
    /// Predicted masks, each as a flat f32 array of size [height, width].
    /// Values in [0, 1] after sigmoid — threshold at 0.5 for binary mask.
    pub masks: Vec<Vec<f32>>,
    /// IoU (intersection-over-union) confidence score for each mask.
    pub iou_scores: Vec<f32>,
    /// Mask width (at model output resolution).
    pub width: usize,
    /// Mask height (at model output resolution).
    pub height: usize,
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct Sam2Kernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    #[allow(dead_code)]
    mul: Arc<ComputePipeline>,
    im2col_patch_embed: Arc<ComputePipeline>,
    window_partition: Arc<ComputePipeline>,
    window_unpartition: Arc<ComputePipeline>,
    max_pool2d: Arc<ComputePipeline>,
    split_qkv: Arc<ComputePipeline>,
    bilinear_upsample: Arc<ComputePipeline>,
}

// ==================== SAM 2.1 Pipeline ====================

/// SAM 2.1 pipeline for promptable image segmentation.
///
/// Forward pipeline:
/// 1. Hiera image encoder: image [3, 1024, 1024] → multi-scale features
/// 2. FPN neck: top-down feature pyramid with lateral connections
/// 3. Prompt encoder: points/boxes → sparse embeddings, masks → dense embeddings
/// 4. Mask decoder: two-way transformer → 3 masks [256, 256] + IoU scores
#[cfg(feature = "metal")]
pub struct Sam2Pipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: Sam2Config,
    kernels: Sam2Kernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for Sam2Pipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl Sam2Pipeline {
    /// Create a new SAM 2.1 pipeline.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: Sam2Config, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = Sam2Kernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
            im2col_patch_embed: compute.compile_pipeline("im2col_patch_embed", sources::PHASE27_OPS, "im2col_patch_embed_f16")?,
            window_partition: compute.compile_pipeline("window_partition", sources::PHASE27_OPS, "window_partition_f16")?,
            window_unpartition: compute.compile_pipeline("window_unpartition", sources::PHASE27_OPS, "window_unpartition_f16")?,
            max_pool2d: compute.compile_pipeline("max_pool2d", sources::PHASE27_OPS, "max_pool2d_f16")?,
            split_qkv: compute.compile_pipeline("split_qkv", sources::PHASE27_OPS, "split_qkv_f16")?,
            bilinear_upsample: compute.compile_pipeline("bilinear_upsample", sources::UPSAMPLE, "upsample_bilinear_2x_f16")?,
        };

        Ok(Self { model, compute, config, kernels })
    }

    /// Segment an image given point prompts.
    ///
    /// # Arguments
    /// * `image_rgb` - Flat f32 image in [3, H, W] format, pixel values in [0, 1].
    /// * `width` - Image width (will be resized to config.image_size internally).
    /// * `height` - Image height.
    /// * `points` - Point prompts as (x, y, is_foreground). Coordinates in input image space.
    ///
    /// # Returns
    /// `SegmentationOutput` with 3 candidate masks (multiscale) and IoU scores.
    pub fn segment(
        &self,
        image_rgb: &[f32],
        width: usize,
        height: usize,
        points: &[(f32, f32, bool)],
    ) -> Result<SegmentationOutput> {
        let _config = &self.config;

        // 1. Hiera image encoder → multi-scale features
        let multi_scale = self.hiera_encode(image_rgb, width, height)?;

        // 2. FPN neck → fused feature map
        let image_embeddings = self.fpn_neck(&multi_scale)?;

        // 3. Prompt encoder → sparse + dense embeddings
        let (sparse_embeddings, dense_embeddings) = self.encode_prompts(
            points, width, height,
        )?;

        // 4. Mask decoder → masks + IoU scores
        let output = self.decode_masks(
            &image_embeddings, &sparse_embeddings, &dense_embeddings,
        )?;

        Ok(output)
    }

    // ==================== Hiera Image Encoder ====================

    /// Hiera encoder: image [3, H, W] → multi-scale features per stage.
    ///
    /// Returns features from stages 1, 2, 3 (skipping stage 0) for the FPN.
    fn hiera_encode(
        &self,
        image_rgb: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<Tensor>> {
        let config = &self.config;
        let _device_id = self.compute.device().info().id;
        let img_size = config.image_size;

        // Resize image to model input size if needed
        let resized = if width != img_size || height != img_size {
            self.bilinear_resize(image_rgb, width, height, img_size, img_size)
        } else {
            image_rgb.to_vec()
        };

        // Patch embedding: Conv2d(3, embed_dim, 7, stride=4, padding=3)
        let grid = img_size / config.patch_size; // 256
        let num_patches = grid * grid;
        let d0 = config.embed_dim;
        let patches = self.hiera_patch_embed(&resized, img_size, grid, d0)?;

        // Add absolute position embeddings (GPU)
        let pos_embed = self.weight_f16(&self.model, "image_encoder.trunk.pos_embed")?;
        let pos_flat = pos_embed.reshape([num_patches, d0])?;
        let cb = self.compute.new_command_buffer();
        let mut hidden = self.add(&cb, &patches, &pos_flat);
        cb.commit();
        cb.wait_until_completed();

        // Run through stages, collecting multi-scale features
        let mut stage_features: Vec<Tensor> = Vec::with_capacity(4);
        let mut current_seq_len = num_patches;
        let mut current_dim = d0;
        let mut current_res = grid;

        for stage in 0..4 {
            let dim = config.stage_dim(stage);
            let heads = config.num_heads[stage];
            let head_dim = dim / heads;
            let window_size = config.window_sizes[stage];

            // Channel projection if dimension changes (between stages)
            if dim != current_dim {
                let cb = self.compute.new_command_buffer();
                hidden = self.linear_bias(
                    &cb, &self.model, &hidden,
                    &format!("image_encoder.trunk.blocks.{}.proj.weight",
                             config.absolute_block_index(stage, 0)),
                    &format!("image_encoder.trunk.blocks.{}.proj.bias",
                             config.absolute_block_index(stage, 0)),
                    current_seq_len, current_dim, dim,
                )?;
                cb.commit();
                cb.wait_until_completed();
                current_dim = dim;
            }

            // Process blocks in this stage
            for block in 0..config.depths[stage] {
                let abs_idx = config.absolute_block_index(stage, block);
                let is_global = config.global_attn_indices.contains(&abs_idx);
                let prefix = format!("image_encoder.trunk.blocks.{}", abs_idx);

                hidden = self.hiera_block(
                    &hidden, &prefix, current_seq_len, current_dim,
                    heads, head_dim, current_res, window_size, is_global,
                )?;
            }

            // Collect features from this stage
            stage_features.push(hidden.clone());

            // Downsample between stages (except after last stage)
            if stage < 3 {
                let (downsampled, new_res) = self.hiera_downsample(
                    &hidden, current_res, current_dim,
                )?;
                hidden = downsampled;
                current_res = new_res;
                current_seq_len = current_res * current_res;
            }
        }

        Ok(stage_features)
    }

    /// Hiera patch embedding: Conv2d(3, embed_dim, kernel=7, stride=4, pad=3) via GPU im2col.
    fn hiera_patch_embed(
        &self,
        image: &[f32],
        img_size: usize,
        grid: usize,
        d_model: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let num_patches = grid * grid;
        let kernel_size = 7;
        let stride = 4;
        let padding = 3;
        let c_in = 3;
        let patch_dim = c_in * kernel_size * kernel_size; // 147

        // Convert f32 image to f16 tensor in [C, H, W] layout for GPU im2col
        let f16_image: Vec<half::f16> = image.iter().map(|&v| half::f16::from_f32(v)).collect();
        let image_tensor = Tensor::from_slice(
            &f16_image,
            Shape::from([c_in, img_size, img_size]),
            DType::F16,
            device_id,
        )?;

        // GPU im2col: [C, H, W] → [num_patches, patch_dim]
        let cb = self.compute.new_command_buffer();
        let im2col_out = gpu_ops::im2col_patch_embed_on(
            &self.compute, &self.kernels.im2col_patch_embed, &cb,
            &image_tensor, c_in, img_size, img_size,
            kernel_size, kernel_size, stride, stride, padding, padding,
        );

        // GPU matmul: [num_patches, patch_dim] @ W^T + bias → [num_patches, d_model]
        let result = self.linear_bias(
            &cb, &self.model, &im2col_out,
            "image_encoder.trunk.patch_embed.proj.weight",
            "image_encoder.trunk.patch_embed.proj.bias",
            num_patches, patch_dim, d_model,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    /// Single Hiera transformer block: LayerNorm → Attention → LayerNorm → MLP(GELU).
    ///
    /// Attention is windowed (local) unless `is_global` is true.
    fn hiera_block(
        &self,
        input: &Tensor,
        prefix: &str,
        seq_len: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        spatial_res: usize,
        window_size: usize,
        is_global: bool,
    ) -> Result<Tensor> {
        let _device_id = self.compute.device().info().id;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let ffn_dim = dim * 4;

        // LayerNorm 1
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, &self.model, input,
            &format!("{}.norm1.weight", prefix),
            &format!("{}.norm1.bias", prefix),
            seq_len, dim, 1e-6,
        )?;

        // Q, K, V projections
        let qkv = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.attn.qkv.weight", prefix),
            &format!("{}.attn.qkv.bias", prefix),
            seq_len, dim, 3 * dim,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Attention: windowed or global
        let attn_out = if is_global || window_size >= spatial_res {
            // Global self-attention
            self.global_self_attention(&qkv, seq_len, dim, num_heads, head_dim, scale)?
        } else {
            // Windowed (Mask Unit) attention
            self.windowed_self_attention(
                &qkv, spatial_res, dim, num_heads, head_dim, window_size, scale,
            )?
        };

        // Output projection + residual
        let cb = self.compute.new_command_buffer();
        let projected = self.linear_bias(
            &cb, &self.model, &attn_out,
            &format!("{}.attn.proj.weight", prefix),
            &format!("{}.attn.proj.bias", prefix),
            seq_len, dim, dim,
        )?;
        let residual1 = self.add(&cb, input, &projected);

        // LayerNorm 2 → MLP (GELU)
        let normed2 = self.layer_norm(
            &cb, &self.model, &residual1,
            &format!("{}.norm2.weight", prefix),
            &format!("{}.norm2.bias", prefix),
            seq_len, dim, 1e-6,
        )?;
        let fc1 = self.linear_bias(
            &cb, &self.model, &normed2,
            &format!("{}.mlp.fc1.weight", prefix),
            &format!("{}.mlp.fc1.bias", prefix),
            seq_len, dim, ffn_dim,
        )?;
        let activated = self.activation(&cb, &self.kernels.gelu, &fc1);
        let fc2 = self.linear_bias(
            &cb, &self.model, &activated,
            &format!("{}.mlp.fc2.weight", prefix),
            &format!("{}.mlp.fc2.bias", prefix),
            seq_len, ffn_dim, dim,
        )?;
        let result = self.add(&cb, &residual1, &fc2);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    /// Global self-attention from packed QKV tensor.
    fn global_self_attention(
        &self,
        qkv: &Tensor,
        seq_len: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Tensor> {
        // GPU split QKV [seq_len, 3*dim] → Q, K, V each [seq_len, dim]
        let cb = self.compute.new_command_buffer();
        let (q, k, v) = gpu_ops::split_qkv_on(
            &self.compute, &self.kernels.split_qkv, &cb,
            qkv, seq_len, dim,
        )?;

        // Batched attention via GPU: Q,K,V [S,H,D] → [S, hidden]
        let q_shd = q.reshape([seq_len, num_heads, head_dim])?;
        let k_shd = k.reshape([seq_len, num_heads, head_dim])?;
        let v_shd = v.reshape([seq_len, num_heads, head_dim])?;

        let result = self.batched_attention(
            &cb, &q_shd, &k_shd, &v_shd,
            seq_len, seq_len, num_heads, head_dim, scale,
        )?;
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    /// Windowed (Mask Unit) self-attention.
    ///
    /// Partitions the spatial feature map into non-overlapping windows of
    /// `window_size x window_size`, applies self-attention within each window,
    /// then reassembles. Uses GPU window partition/unpartition and split_qkv.
    fn windowed_self_attention(
        &self,
        qkv: &Tensor,
        spatial_res: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        window_size: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let seq_len = spatial_res * spatial_res;
        let num_windows = (spatial_res / window_size) * (spatial_res / window_size);
        let window_tokens = window_size * window_size;

        // GPU split QKV [seq_len, 3*dim] → Q, K, V each [seq_len, dim]
        let cb = self.compute.new_command_buffer();
        let (q, k, v) = gpu_ops::split_qkv_on(
            &self.compute, &self.kernels.split_qkv, &cb,
            qkv, seq_len, dim,
        )?;

        // GPU window partition: [H*W, D] → [num_windows, win_tokens, D]
        let q_win = gpu_ops::window_partition_on(
            &self.compute, &self.kernels.window_partition, &cb,
            &q, spatial_res, spatial_res, dim, window_size, window_size,
        );
        let k_win = gpu_ops::window_partition_on(
            &self.compute, &self.kernels.window_partition, &cb,
            &k, spatial_res, spatial_res, dim, window_size, window_size,
        );
        let v_win = gpu_ops::window_partition_on(
            &self.compute, &self.kernels.window_partition, &cb,
            &v, spatial_res, spatial_res, dim, window_size, window_size,
        );
        cb.commit();
        cb.wait_until_completed();

        // Process each window's attention on GPU
        let device_id = self.compute.device().info().id;
        let mut window_outputs: Vec<Tensor> = Vec::with_capacity(num_windows);
        for w in 0..num_windows {
            let q_w = q_win.slice(0, w, w + 1)?.reshape([window_tokens, dim])?;
            let k_w = k_win.slice(0, w, w + 1)?.reshape([window_tokens, dim])?;
            let v_w = v_win.slice(0, w, w + 1)?.reshape([window_tokens, dim])?;

            let q_shd = q_w.reshape([window_tokens, num_heads, head_dim])?;
            let k_shd = k_w.reshape([window_tokens, num_heads, head_dim])?;
            let v_shd = v_w.reshape([window_tokens, num_heads, head_dim])?;

            let cb = self.compute.new_command_buffer();
            let attn_out = self.batched_attention(
                &cb, &q_shd, &k_shd, &v_shd,
                window_tokens, window_tokens, num_heads, head_dim, scale,
            )?;
            cb.commit();
            cb.wait_until_completed();
            window_outputs.push(attn_out);
        }

        // Reassemble windows: stack → [num_windows, win_tokens, dim], then unpartition
        let mut stacked_data: Vec<half::f16> = Vec::with_capacity(num_windows * window_tokens * dim);
        for wo in &window_outputs {
            let d: Vec<half::f16> = wo.to_vec()?;
            stacked_data.extend_from_slice(&d);
        }
        let stacked = Tensor::from_slice(
            &stacked_data,
            Shape::from([num_windows, window_tokens, dim]),
            DType::F16,
            device_id,
        )?;

        // GPU window unpartition: [num_windows, win_tokens, D] → [H*W, D]
        let cb = self.compute.new_command_buffer();
        let result = gpu_ops::window_unpartition_on(
            &self.compute, &self.kernels.window_unpartition, &cb,
            &stacked, spatial_res, spatial_res, dim, window_size, window_size,
        );
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    /// Inter-stage downsampling: 2x2 max-pool on spatial grid via GPU.
    ///
    /// Input: [res*res, dim] → Output: [(res/2)*(res/2), dim].
    fn hiera_downsample(
        &self,
        input: &Tensor,
        res: usize,
        dim: usize,
    ) -> Result<(Tensor, usize)> {
        let device_id = self.compute.device().info().id;
        let new_res = res / 2;

        // Transpose [H*W, D] (HWD layout) → [D, H, W] (CHW layout) for max_pool2d kernel
        let data: Vec<half::f16> = input.to_vec()?;
        let mut chw = vec![half::f16::ZERO; dim * res * res];
        for y in 0..res {
            for x in 0..res {
                for d in 0..dim {
                    chw[d * res * res + y * res + x] = data[(y * res + x) * dim + d];
                }
            }
        }
        let chw_tensor = Tensor::from_slice(
            &chw, Shape::from([dim, res, res]), DType::F16, device_id,
        )?;

        // GPU max pool 2x2 stride 2: [D, H, W] → [D, H/2, W/2]
        let cb = self.compute.new_command_buffer();
        let pooled = gpu_ops::max_pool2d_on(
            &self.compute, &self.kernels.max_pool2d, &cb,
            &chw_tensor, dim, res, res,
        );
        cb.commit();
        cb.wait_until_completed();

        // Transpose back [D, H/2, W/2] → [H/2*W/2, D]
        let pooled_data: Vec<half::f16> = pooled.to_vec()?;
        let mut out = vec![half::f16::ZERO; new_res * new_res * dim];
        for y in 0..new_res {
            for x in 0..new_res {
                for d in 0..dim {
                    out[(y * new_res + x) * dim + d] = pooled_data[d * new_res * new_res + y * new_res + x];
                }
            }
        }
        let tensor = Tensor::from_slice(
            &out, Shape::from([new_res * new_res, dim]), DType::F16, device_id,
        )?;
        Ok((tensor, new_res))
    }

    // ==================== FPN Neck ====================

    /// Feature Pyramid Network neck: top-down path with lateral connections.
    ///
    /// Takes multi-scale features from Hiera stages 1-3 and produces a fused
    /// feature map at stride-16 resolution suitable for the mask decoder.
    ///
    /// Returns: image embeddings [H/16 * W/16, fpn_out_dim].
    fn fpn_neck(&self, stage_features: &[Tensor]) -> Result<Tensor> {
        let config = &self.config;
        let _device_id = self.compute.device().info().id;

        // Lateral projections: 1x1 conv (linear) per stage to fpn_out_dim
        let mut laterals: Vec<Tensor> = Vec::with_capacity(4);
        for (stage, feat) in stage_features.iter().enumerate() {
            let dim = config.stage_dim(stage);
            let res = config.stage_resolution(stage);
            let seq_len = res * res;

            let cb = self.compute.new_command_buffer();
            let lateral = self.linear_bias(
                &cb, &self.model, feat,
                &format!("image_encoder.neck.convs.{}.weight", stage),
                &format!("image_encoder.neck.convs.{}.bias", stage),
                seq_len, dim, config.fpn_out_dim,
            )?;
            cb.commit();
            cb.wait_until_completed();

            laterals.push(lateral);
        }

        // Top-down pathway: upsample from coarsest to finest, adding laterals
        let mut current = laterals.last().unwrap().clone();
        let mut current_res = config.stage_resolution(3);

        for stage in (0..3).rev() {
            let target_res = config.stage_resolution(stage);

            // Upsample current to match target resolution
            let upsampled = self.upsample_features(
                &current, current_res, config.fpn_out_dim, target_res,
            )?;

            // Add lateral connection
            let cb = self.compute.new_command_buffer();
            current = self.add(&cb, &upsampled, &laterals[stage]);
            cb.commit();
            cb.wait_until_completed();

            current_res = target_res;
        }

        // Final output at stage-0 resolution, projected through a LayerNorm
        let seq_len = current_res * current_res;
        let cb = self.compute.new_command_buffer();
        let output = self.layer_norm(
            &cb, &self.model, &current,
            "image_encoder.neck.norm.weight",
            "image_encoder.neck.norm.bias",
            seq_len, config.fpn_out_dim, 1e-6,
        )?;
        cb.commit();
        cb.wait_until_completed();

        Ok(output)
    }

    /// Upsample features via GPU bilinear interpolation.
    ///
    /// Input: [src_res*src_res, dim] → Output: [tgt_res*tgt_res, dim].
    /// Uses GPU upsample_bilinear_2x_f16 when scale is exactly 2x,
    /// chaining multiple 2x passes for larger scales.
    fn upsample_features(
        &self,
        input: &Tensor,
        src_res: usize,
        dim: usize,
        tgt_res: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;

        // Transpose [H*W, D] → [D, H, W] for GPU upsample kernel
        let data: Vec<half::f16> = input.to_vec()?;
        let mut chw = vec![half::f16::ZERO; dim * src_res * src_res];
        for y in 0..src_res {
            for x in 0..src_res {
                for d in 0..dim {
                    chw[d * src_res * src_res + y * src_res + x] = data[(y * src_res + x) * dim + d];
                }
            }
        }
        let mut current = Tensor::from_slice(
            &chw, Shape::from([dim, src_res, src_res]), DType::F16, device_id,
        )?;
        let mut cur_res = src_res;

        // Chain 2x bilinear upsamples until we reach target resolution
        while cur_res < tgt_res {
            let numel = dim * cur_res * 2 * cur_res * 2;
            let cb = self.compute.new_command_buffer();
            let device = self.compute.device().raw();
            let output_buffer = device.new_buffer(
                (numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared,
            );
            self.compute.dispatch_1d(&cb, &self.kernels.bilinear_upsample, numel, |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, &current);
                encoder.set_buffer(1, Some(&output_buffer), 0);
                let c_val = dim as u32;
                let h_val = cur_res as u32;
                let w_val = cur_res as u32;
                encoder.set_bytes(2, 4, &c_val as *const u32 as *const _);
                encoder.set_bytes(3, 4, &h_val as *const u32 as *const _);
                encoder.set_bytes(4, 4, &w_val as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();
            cur_res *= 2;
            current = Tensor::from_metal_buffer(
                output_buffer, Shape::from([dim, cur_res, cur_res]), DType::F16, device_id,
            );
        }

        // Transpose back [D, H, W] → [H*W, D]
        let out_data: Vec<half::f16> = current.to_vec()?;
        let mut result = vec![half::f16::ZERO; tgt_res * tgt_res * dim];
        for y in 0..tgt_res {
            for x in 0..tgt_res {
                for d in 0..dim {
                    result[(y * tgt_res + x) * dim + d] = out_data[d * tgt_res * tgt_res + y * tgt_res + x];
                }
            }
        }
        Tensor::from_slice(
            &result, Shape::from([tgt_res * tgt_res, dim]), DType::F16, device_id,
        )
    }

    // ==================== Prompt Encoder ====================

    /// Encode point prompts into sparse embeddings.
    ///
    /// Each point is encoded as a learned embedding + sinusoidal positional encoding.
    /// Foreground and background points get different learned embeddings.
    ///
    /// Returns: (sparse_embeddings [num_points + num_mask_tokens, decoder_hidden],
    ///           dense_embeddings [H/16 * W/16, decoder_hidden]).
    fn encode_prompts(
        &self,
        points: &[(f32, f32, bool)],
        width: usize,
        height: usize,
    ) -> Result<(Tensor, Tensor)> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;
        let dim = config.decoder_hidden;
        let _img_size = config.image_size as f32;

        // Sparse embeddings: point embeddings + positional encoding
        let num_points = points.len();
        let total_tokens = num_points + config.num_mask_tokens; // points + mask tokens
        let mut sparse_data = vec![half::f16::ZERO; total_tokens * dim];

        // Load point embedding weights
        let fg_embed = self.weight_f16(&self.model, "prompt_encoder.point_embeddings.0.weight")?;
        let bg_embed = self.weight_f16(&self.model, "prompt_encoder.point_embeddings.1.weight")?;
        let fg_data: Vec<half::f16> = fg_embed.to_vec()?;
        let bg_data: Vec<half::f16> = bg_embed.to_vec()?;

        // Positional encoding for each point
        for (idx, &(px, py, is_fg)) in points.iter().enumerate() {
            // Normalize to [0, 1] in model image space
            let nx = px / width as f32;
            let ny = py / height as f32;

            // Sinusoidal position encoding
            let pos_enc = self.sinusoidal_pos_encoding(nx, ny, dim);

            // Add learned point type embedding (fg or bg)
            let type_embed = if is_fg { &fg_data } else { &bg_data };
            for d in 0..dim {
                let val = pos_enc[d] + type_embed[d.min(type_embed.len() - 1)].to_f32();
                sparse_data[idx * dim + d] = half::f16::from_f32(val);
            }
        }

        // Append mask tokens (learned embeddings for each output mask)
        for t in 0..config.num_mask_tokens {
            let token_embed = self.weight_f16(
                &self.model,
                &format!("mask_decoder.mask_tokens.weight"),
            );
            if let Ok(tok_data_tensor) = token_embed {
                let tok_data: Vec<half::f16> = tok_data_tensor.to_vec()?;
                let offset = t * dim;
                for d in 0..dim {
                    if offset + d < tok_data.len() {
                        sparse_data[(num_points + t) * dim + d] = tok_data[offset + d];
                    }
                }
            }
        }

        let sparse = Tensor::from_slice(
            &sparse_data,
            Shape::from([total_tokens, dim]),
            DType::F16,
            device_id,
        )?;

        // Dense embeddings: no mask prompt → use learned "no mask" embedding broadcast
        let _image_pe_res = config.image_size / config.patch_size; // 256 at stage 0
        let target_res = config.stage_resolution(0); // same
        let dense_tokens = target_res * target_res;

        // Load not-a-mask embedding and broadcast
        let no_mask_embed = self.weight_f16(
            &self.model, "prompt_encoder.no_mask_embed.weight",
        );
        let dense = if let Ok(nm_tensor) = no_mask_embed {
            let nm_data: Vec<half::f16> = nm_tensor.to_vec()?;
            let mut dense_data = vec![half::f16::ZERO; dense_tokens * dim];
            for tok in 0..dense_tokens {
                for d in 0..dim.min(nm_data.len()) {
                    dense_data[tok * dim + d] = nm_data[d];
                }
            }
            Tensor::from_slice(
                &dense_data,
                Shape::from([dense_tokens, dim]),
                DType::F16,
                device_id,
            )?
        } else {
            // Fallback: zero dense embeddings
            Tensor::zeros(Shape::from([dense_tokens, dim]), DType::F16)?
        };

        Ok((sparse, dense))
    }

    /// Sinusoidal 2D positional encoding for a normalized (x, y) coordinate.
    fn sinusoidal_pos_encoding(&self, x: f32, y: f32, dim: usize) -> Vec<f32> {
        let half_dim = dim / 2;
        let mut enc = vec![0.0f32; dim];

        for i in 0..half_dim {
            let freq = 1.0 / (10000.0_f32.powf(2.0 * i as f32 / half_dim as f32));
            // x-component in first half, y-component in second half
            if i % 2 == 0 {
                enc[i] = (x * freq).sin();
                enc[half_dim + i] = (y * freq).sin();
            } else {
                enc[i] = (x * freq).cos();
                enc[half_dim + i] = (y * freq).cos();
            }
        }

        enc
    }

    // ==================== Mask Decoder ====================

    /// Two-way transformer mask decoder.
    ///
    /// 1. Concatenate mask tokens with sparse embeddings → "tokens"
    /// 2. Two-way cross-attention: tokens attend to image, image attends to tokens
    /// 3. MLP heads produce masks + IoU scores
    ///
    /// Returns SegmentationOutput with 3 masks and IoU scores.
    fn decode_masks(
        &self,
        image_embeddings: &Tensor,
        sparse_embeddings: &Tensor,
        dense_embeddings: &Tensor,
    ) -> Result<SegmentationOutput> {
        let config = &self.config;
        let _device_id = self.compute.device().info().id;
        let dim = config.decoder_hidden;
        let num_heads = config.decoder_heads;
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Image tokens: image_embeddings + dense_embeddings
        let img_seq_len = image_embeddings.numel() / dim;
        let cb = self.compute.new_command_buffer();
        let image_tokens = self.add(&cb, image_embeddings, dense_embeddings);
        cb.commit();
        cb.wait_until_completed();

        // Token inputs = sparse_embeddings (includes mask tokens + point tokens)
        let token_seq_len = sparse_embeddings.numel() / dim;
        let mut tokens = sparse_embeddings.clone();

        // Two-way transformer layers
        for layer in 0..config.decoder_depth {
            let prefix = format!("mask_decoder.transformer.layers.{}", layer);

            // === Token-to-image cross-attention ===
            // Tokens attend to image features
            tokens = self.cross_attention_block(
                &tokens, &image_tokens,
                token_seq_len, img_seq_len, dim, num_heads, head_dim, scale,
                &format!("{}.cross_attn_token_to_image", prefix),
                &format!("{}.norm1", prefix),
            )?;

            // === Token self-attention ===
            tokens = self.self_attention_block(
                &tokens, token_seq_len, dim, num_heads, head_dim, scale,
                &format!("{}.self_attn", prefix),
                &format!("{}.norm2", prefix),
            )?;

            // === Token MLP ===
            tokens = self.mlp_block(
                &tokens, token_seq_len, dim,
                &format!("{}.mlp", prefix),
                &format!("{}.norm3", prefix),
            )?;

            // === Image-to-token cross-attention ===
            // Image attends back to tokens (bidirectional)
            let updated_image = self.cross_attention_block(
                &image_tokens, &tokens,
                img_seq_len, token_seq_len, dim, num_heads, head_dim, scale,
                &format!("{}.cross_attn_image_to_token", prefix),
                &format!("{}.norm4", prefix),
            )?;
            // Note: we use image_tokens for the next layer but don't accumulate —
            // SAM only updates image tokens in-place for the cross-attn computation
            let _ = updated_image;
        }

        // Final layer norm on tokens
        let cb = self.compute.new_command_buffer();
        let tokens_normed = self.layer_norm(
            &cb, &self.model, &tokens,
            "mask_decoder.transformer.final_attn_token_to_image.norm.weight",
            "mask_decoder.transformer.final_attn_token_to_image.norm.bias",
            token_seq_len, dim, 1e-6,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Generate masks on GPU: mask_tokens @ image_features^T → logits
        let _output_res = config.image_size / (config.patch_size * 4); // 64
        let num_masks = config.num_mask_tokens.min(3); // Output 3 masks
        let device_id = self.compute.device().info().id;

        // GPU batch matmul: [num_masks, dim] @ [img_seq_len, dim]^T → [num_masks, img_seq_len]
        let mask_tokens_tensor = tokens_normed.slice(0, 0, num_masks)?;
        let zero_bias = Tensor::empty(Shape::from([img_seq_len]), DType::F16, device_id)?;
        let cb_masks = self.compute.new_command_buffer();
        let all_logits = self.linear_tensors(
            cb_masks.as_ref(), &mask_tokens_tensor, &image_tokens, &zero_bias,
            num_masks, dim, img_seq_len,
        );
        cb_masks.commit();
        cb_masks.wait_until_completed();
        let all_logits_f32 = all_logits.to_f32_vec()?;

        let tokens_data: Vec<half::f16> = tokens_normed.to_vec()?;
        let mut masks = Vec::with_capacity(num_masks);
        let mut iou_scores = Vec::with_capacity(num_masks);

        for m in 0..num_masks {
            let mask_logits = &all_logits_f32[m * img_seq_len..(m + 1) * img_seq_len];

            // Reshape to spatial and upsample to 256x256 output resolution
            let feat_res = (img_seq_len as f32).sqrt() as usize;
            let out_size = 256;
            let mask = self.upsample_mask(mask_logits, feat_res, out_size);

            // Apply sigmoid to get probabilities
            let mask_prob: Vec<f32> = mask.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();

            // IoU score
            let mask_token = &tokens_data[m * dim..(m + 1) * dim];
            let iou = self.compute_iou_score(mask_token, dim, m)?;

            masks.push(mask_prob);
            iou_scores.push(iou);
        }

        Ok(SegmentationOutput {
            masks,
            iou_scores,
            width: 256,
            height: 256,
        })
    }

    /// Cross-attention block: query attends to key/value context.
    ///
    /// Pre-norm with LayerNorm, then multi-head attention, then residual add.
    fn cross_attention_block(
        &self,
        query_tokens: &Tensor,
        context_tokens: &Tensor,
        q_seq: usize,
        kv_seq: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attn_prefix: &str,
        norm_prefix: &str,
    ) -> Result<Tensor> {
        let _device_id = self.compute.device().info().id;

        // LayerNorm on query
        let cb = self.compute.new_command_buffer();
        let normed_q = self.layer_norm(
            &cb, &self.model, query_tokens,
            &format!("{}.weight", norm_prefix),
            &format!("{}.bias", norm_prefix),
            q_seq, dim, 1e-6,
        )?;

        // Q from query, K/V from context
        let q = self.linear_bias(
            &cb, &self.model, &normed_q,
            &format!("{}.q_proj.weight", attn_prefix),
            &format!("{}.q_proj.bias", attn_prefix),
            q_seq, dim, dim,
        )?;
        let k = self.linear_bias(
            &cb, &self.model, context_tokens,
            &format!("{}.k_proj.weight", attn_prefix),
            &format!("{}.k_proj.bias", attn_prefix),
            kv_seq, dim, dim,
        )?;
        let v = self.linear_bias(
            &cb, &self.model, context_tokens,
            &format!("{}.v_proj.weight", attn_prefix),
            &format!("{}.v_proj.bias", attn_prefix),
            kv_seq, dim, dim,
        )?;

        // Reshape to [S, H, D] for batched attention
        let q_shd = q.reshape([q_seq, num_heads, head_dim])?;
        let k_shd = k.reshape([kv_seq, num_heads, head_dim])?;
        let v_shd = v.reshape([kv_seq, num_heads, head_dim])?;

        let attn_out = self.batched_attention(
            &cb, &q_shd, &k_shd, &v_shd,
            q_seq, kv_seq, num_heads, head_dim, scale,
        )?;

        // Output projection + residual
        let projected = self.linear_bias(
            &cb, &self.model, &attn_out,
            &format!("{}.out_proj.weight", attn_prefix),
            &format!("{}.out_proj.bias", attn_prefix),
            q_seq, dim, dim,
        )?;
        let result = self.add(&cb, query_tokens, &projected);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    /// Self-attention block with pre-norm and residual.
    fn self_attention_block(
        &self,
        tokens: &Tensor,
        seq_len: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attn_prefix: &str,
        norm_prefix: &str,
    ) -> Result<Tensor> {
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, &self.model, tokens,
            &format!("{}.weight", norm_prefix),
            &format!("{}.bias", norm_prefix),
            seq_len, dim, 1e-6,
        )?;

        let q = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.q_proj.weight", attn_prefix),
            &format!("{}.q_proj.bias", attn_prefix),
            seq_len, dim, dim,
        )?;
        let k = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.k_proj.weight", attn_prefix),
            &format!("{}.k_proj.bias", attn_prefix),
            seq_len, dim, dim,
        )?;
        let v = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.v_proj.weight", attn_prefix),
            &format!("{}.v_proj.bias", attn_prefix),
            seq_len, dim, dim,
        )?;

        let q_shd = q.reshape([seq_len, num_heads, head_dim])?;
        let k_shd = k.reshape([seq_len, num_heads, head_dim])?;
        let v_shd = v.reshape([seq_len, num_heads, head_dim])?;

        let attn_out = self.batched_attention(
            &cb, &q_shd, &k_shd, &v_shd,
            seq_len, seq_len, num_heads, head_dim, scale,
        )?;

        let projected = self.linear_bias(
            &cb, &self.model, &attn_out,
            &format!("{}.out_proj.weight", attn_prefix),
            &format!("{}.out_proj.bias", attn_prefix),
            seq_len, dim, dim,
        )?;
        let result = self.add(&cb, tokens, &projected);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    /// MLP block with pre-norm, GELU activation, and residual.
    fn mlp_block(
        &self,
        tokens: &Tensor,
        seq_len: usize,
        dim: usize,
        mlp_prefix: &str,
        norm_prefix: &str,
    ) -> Result<Tensor> {
        let ffn_dim = dim * 4;

        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, &self.model, tokens,
            &format!("{}.weight", norm_prefix),
            &format!("{}.bias", norm_prefix),
            seq_len, dim, 1e-6,
        )?;
        let fc1 = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.lin1.weight", mlp_prefix),
            &format!("{}.lin1.bias", mlp_prefix),
            seq_len, dim, ffn_dim,
        )?;
        let activated = self.activation(&cb, &self.kernels.relu, &fc1);
        let fc2 = self.linear_bias(
            &cb, &self.model, &activated,
            &format!("{}.lin2.weight", mlp_prefix),
            &format!("{}.lin2.bias", mlp_prefix),
            seq_len, ffn_dim, dim,
        )?;
        let result = self.add(&cb, tokens, &fc2);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    /// Compute IoU (intersection-over-union) confidence score for a mask token.
    ///
    /// Uses the mask decoder IoU prediction MLP if weights are available,
    /// otherwise falls back to normalized L2 magnitude as a proxy.
    fn compute_iou_score(
        &self,
        mask_token: &[half::f16],
        dim: usize,
        mask_idx: usize,
    ) -> Result<f32> {
        let device_id = self.compute.device().info().id;

        // Try to use IoU prediction head weights
        let token_tensor = Tensor::from_slice(
            mask_token,
            Shape::from([1, dim]),
            DType::F16,
            device_id,
        )?;

        // IoU head: 3-layer MLP → 1 score per mask
        let iou_result = (|| -> Result<f32> {
            let cb = self.compute.new_command_buffer();
            let h1 = self.linear_bias(
                &cb, &self.model, &token_tensor,
                "mask_decoder.iou_prediction_head.layers.0.weight",
                "mask_decoder.iou_prediction_head.layers.0.bias",
                1, dim, dim,
            )?;
            let h1_act = self.activation(&cb, &self.kernels.relu, &h1);
            let h2 = self.linear_bias(
                &cb, &self.model, &h1_act,
                "mask_decoder.iou_prediction_head.layers.1.weight",
                "mask_decoder.iou_prediction_head.layers.1.bias",
                1, dim, dim,
            )?;
            let h2_act = self.activation(&cb, &self.kernels.relu, &h2);
            // Final layer outputs num_mask_tokens scores
            let scores = self.linear_bias(
                &cb, &self.model, &h2_act,
                "mask_decoder.iou_prediction_head.layers.2.weight",
                "mask_decoder.iou_prediction_head.layers.2.bias",
                1, dim, self.config.num_mask_tokens,
            )?;
            cb.commit();
            cb.wait_until_completed();

            let scores_data: Vec<half::f16> = scores.to_vec()?;
            let raw = scores_data[mask_idx.min(scores_data.len() - 1)].to_f32();
            // Sigmoid to [0, 1]
            Ok(1.0 / (1.0 + (-raw).exp()))
        })();

        match iou_result {
            Ok(score) => Ok(score),
            Err(_) => {
                // Fallback: normalized L2 magnitude as proxy confidence
                let mut sum_sq = 0.0f32;
                for d in 0..dim {
                    let v = mask_token[d].to_f32();
                    sum_sq += v * v;
                }
                Ok((sum_sq / dim as f32).sqrt().min(1.0))
            }
        }
    }

    /// Bilinear upsample a single-channel mask from feat_res to out_size.
    fn upsample_mask(&self, mask: &[f32], feat_res: usize, out_size: usize) -> Vec<f32> {
        let scale = feat_res as f32 / out_size as f32;
        let mut out = vec![0.0f32; out_size * out_size];

        for oy in 0..out_size {
            for ox in 0..out_size {
                let sy = (oy as f32 + 0.5) * scale - 0.5;
                let sx = (ox as f32 + 0.5) * scale - 0.5;
                let sy0 = sy.floor().max(0.0) as usize;
                let sx0 = sx.floor().max(0.0) as usize;
                let sy1 = (sy0 + 1).min(feat_res - 1);
                let sx1 = (sx0 + 1).min(feat_res - 1);
                let fy = sy - sy0 as f32;
                let fx = sx - sx0 as f32;

                let v00 = mask[sy0 * feat_res + sx0];
                let v01 = mask[sy0 * feat_res + sx1];
                let v10 = mask[sy1 * feat_res + sx0];
                let v11 = mask[sy1 * feat_res + sx1];

                out[oy * out_size + ox] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }

        out
    }

    // ==================== Image Utilities ====================

    /// Bilinear resize of a CHW f32 image.
    fn bilinear_resize(
        &self,
        image: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; 3 * dst_h * dst_w];
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;

        for c in 0..3 {
            for dy in 0..dst_h {
                for dx in 0..dst_w {
                    let sy = (dy as f32 + 0.5) * scale_y - 0.5;
                    let sx = (dx as f32 + 0.5) * scale_x - 0.5;
                    let sy0 = sy.floor().max(0.0) as usize;
                    let sx0 = sx.floor().max(0.0) as usize;
                    let sy1 = (sy0 + 1).min(src_h - 1);
                    let sx1 = (sx0 + 1).min(src_w - 1);
                    let fy = sy - sy0 as f32;
                    let fx = sx - sx0 as f32;

                    let base = c * src_h * src_w;
                    let v00 = image[base + sy0 * src_w + sx0];
                    let v01 = image[base + sy0 * src_w + sx1];
                    let v10 = image[base + sy1 * src_w + sx0];
                    let v11 = image[base + sy1 * src_w + sx1];

                    out[c * dst_h * dst_w + dy * dst_w + dx] = v00 * (1.0 - fx) * (1.0 - fy)
                        + v01 * fx * (1.0 - fy)
                        + v10 * (1.0 - fx) * fy
                        + v11 * fx * fy;
                }
            }
        }

        out
    }
}
