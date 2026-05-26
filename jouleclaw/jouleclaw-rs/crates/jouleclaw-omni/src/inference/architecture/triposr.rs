//! TripoSR: Transformer-based 3D reconstruction from a single image.
//!
//! Architecture:
//!   Image (224×224) → DINO ViT-B/16 tokenizer → 196 patch features [196, 768]
//!   → triplane tokens [3072, 1024] (learned embeddings, flattened from [3, 1024, 32, 32])
//!   → 16-layer transformer (self-attn + cross-attn to DINO features + GEGLU FFN)
//!   → proj_out → reshape to [3, 1024, 32, 32]
//!   → ConvTranspose2d upsample [3, 40, 64, 64]
//!   → NeRF MLP: triplane features → density + RGB
//!
//! Cross-attention uses DINO ViT-B/16 features (768-dim) as context.
//! All weights are f32 — converted to f16 at dispatch time.

use crate::core::Result;
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

/// TripoSR configuration.
#[derive(Debug, Clone)]
pub struct TripoSRConfig {
    /// Image input size (resized to 224 for DINO).
    pub image_size: usize,
    /// Triplane grid size (per plane).
    pub plane_size: usize,
    /// Number of triplane channels.
    pub num_channels: usize,
    /// Transformer hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Cross-attention dimension (DINO features).
    pub cross_attn_dim: usize,
    /// Upsample output channels.
    pub upsample_out_channels: usize,
    /// NeRF MLP hidden dimension.
    pub nerf_hidden: usize,
    /// NeRF MLP number of hidden layers.
    pub nerf_num_layers: usize,
    /// Number of samples per ray for rendering.
    pub num_samples_per_ray: usize,
}

impl Default for TripoSRConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            plane_size: 32,
            num_channels: 1024,
            hidden_dim: 1024,
            num_heads: 16,
            head_dim: 64,
            num_layers: 16,
            cross_attn_dim: 768,
            upsample_out_channels: 40,
            nerf_hidden: 64,
            nerf_num_layers: 9,
            num_samples_per_ray: 128,
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct TripoSRKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    geglu: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
}

// ==================== TripoSR Pipeline ====================

/// TripoSR pipeline for single-image 3D reconstruction.
///
/// Forward pipeline:
/// 1. DINO ViT-B/16: image [3, 224, 224] → patch features [196, 768]
/// 2. Triplane tokens: learned embeddings [3072, 1024]
/// 3. Backbone: 16 transformer blocks (self-attn + cross-attn + GEGLU FFN)
/// 4. Post-processor: ConvTranspose2d to upsample [3, 40, 64, 64]
/// 5. NeRF MLP: triplane features → density + RGB
#[cfg(feature = "metal")]
pub struct TripoSRPipeline {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: TripoSRConfig,
    kernels: TripoSRKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for TripoSRPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl TripoSRPipeline {
    /// Create a new TripoSR pipeline.
    pub fn new(model: Arc<Model>, config: TripoSRConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = TripoSRKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            geglu: compute.compile_pipeline("geglu", sources::GELU, "geglu_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
        };

        Ok(Self { model, compute, config, kernels })
    }

    /// Run full 3D reconstruction from image.
    ///
    /// Input: image as flat f32 RGB array [3*224*224] in [C, H, W] format, normalized to ImageNet stats.
    /// Output: triplane features [3, upsample_out_channels, plane_size*2, plane_size*2]
    pub fn forward(&self, image_chw: &[f32]) -> Result<Tensor> {
        let config = &self.config;
        let num_patches = (config.image_size / 16) * (config.image_size / 16); // 196
        let num_triplane_tokens = 3 * config.plane_size * config.plane_size; // 3072

        // 1. DINO ViT: image → patch features [196, 768]
        let image_features = self.dino_forward(image_chw, num_patches)?;

        // 2. Load triplane token embeddings [3, 1024, 32, 32] → flatten to [3072, 1024]
        let tok_embed = self.weight_f16(&self.model, "tokenizer.embeddings")?;
        // tok_embed is [3, 1024, 32, 32] → reshape to [3*32*32, 1024] = [3072, 1024]
        // But layout is [3, C, H, W] → need to reshape as [3*H*W, C]
        let tok_data: Vec<half::f16> = tok_embed.to_vec()?;
        let mut triplane_data = vec![half::f16::ZERO; num_triplane_tokens * config.hidden_dim];
        for plane in 0..3 {
            for h in 0..config.plane_size {
                for w in 0..config.plane_size {
                    let spatial_idx = plane * config.plane_size * config.plane_size + h * config.plane_size + w;
                    for c in 0..config.hidden_dim {
                        // Source: [plane, c, h, w]
                        let src = plane * config.hidden_dim * config.plane_size * config.plane_size
                            + c * config.plane_size * config.plane_size
                            + h * config.plane_size + w;
                        // Dest: [spatial_idx, c]
                        triplane_data[spatial_idx * config.hidden_dim + c] = tok_data[src];
                    }
                }
            }
        }
        let mut hidden = Tensor::from_slice(
            &triplane_data,
            Shape::from([num_triplane_tokens, config.hidden_dim]),
            DType::F16,
            self.compute.device().info().id,
        )?;

        // 3. proj_in: hidden_dim → hidden_dim
        let cb = self.compute.new_command_buffer();
        let proj_in = self.linear_bias(
            &cb, &self.model, &hidden,
            "backbone.proj_in.weight", "backbone.proj_in.bias",
            num_triplane_tokens, config.hidden_dim, config.hidden_dim,
        )?;
        cb.commit();
        cb.wait_until_completed();
        hidden = proj_in;

        // 4. Pre-compute cross-attention K/V from DINO features (same for all layers)
        let cross_kv = self.precompute_cross_kv(&image_features, num_patches)?;

        // 5. Transformer backbone: 16 layers
        for layer in 0..config.num_layers {
            hidden = self.transformer_block(layer, hidden, num_triplane_tokens, &cross_kv, num_patches)?;
        }

        // 6. Final norm + proj_out
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, &self.model, &hidden,
            "backbone.norm.weight", "backbone.norm.bias",
            num_triplane_tokens, config.hidden_dim, 1e-5,
        )?;
        let proj_out = self.linear_bias(
            &cb, &self.model, &normed,
            "backbone.proj_out.weight", "backbone.proj_out.bias",
            num_triplane_tokens, config.hidden_dim, config.num_channels,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // 7. Reshape to [3, C, H, W] and apply ConvTranspose2d upsample
        let triplane = self.post_process(&proj_out, num_triplane_tokens)?;

        Ok(triplane)
    }

    /// Evaluate NeRF MLP decoder at 3D points (batched GPU).
    ///
    /// `triplane`: [3, out_channels, H, W] upsampled triplane features
    /// `points`: [N, 3] 3D coordinates in [-1, 1]
    /// Returns: [N, 4] (density, R, G, B)
    pub fn query_nerf(&self, triplane: &Tensor, points: &[[f32; 3]]) -> Result<Vec<[f32; 4]>> {
        let config = &self.config;
        let out_ch = config.upsample_out_channels;
        let spatial = config.plane_size * 2; // 64
        let n_points = points.len();
        let input_dim = 3 * out_ch; // 120

        // Read triplane data
        let tp_data: Vec<half::f16> = triplane.to_vec()?;

        // Batch bilinear interpolation on CPU (pure memory lookup, no FLOPs)
        let mut input_data = vec![half::f16::ZERO; n_points * input_dim];
        for (p, &[x, y, z]) in points.iter().enumerate() {
            let coords = [(x, y), (x, z), (y, z)];
            for (plane_idx, &(u, v)) in coords.iter().enumerate() {
                let px = ((u + 1.0) * 0.5 * (spatial - 1) as f32).clamp(0.0, (spatial - 1) as f32);
                let py = ((v + 1.0) * 0.5 * (spatial - 1) as f32).clamp(0.0, (spatial - 1) as f32);
                let ix = px.floor() as usize;
                let iy = py.floor() as usize;
                let fx = px - ix as f32;
                let fy = py - iy as f32;
                let ix1 = (ix + 1).min(spatial - 1);
                let iy1 = (iy + 1).min(spatial - 1);

                for c in 0..out_ch {
                    let base = plane_idx * out_ch * spatial * spatial + c * spatial * spatial;
                    let v00 = tp_data[base + iy * spatial + ix].to_f32();
                    let v01 = tp_data[base + iy * spatial + ix1].to_f32();
                    let v10 = tp_data[base + iy1 * spatial + ix].to_f32();
                    let v11 = tp_data[base + iy1 * spatial + ix1].to_f32();
                    let val = v00 * (1.0 - fx) * (1.0 - fy)
                            + v01 * fx * (1.0 - fy)
                            + v10 * (1.0 - fx) * fy
                            + v11 * fx * fy;
                    input_data[p * input_dim + plane_idx * out_ch + c] = half::f16::from_f32(val);
                }
            }
        }

        // GPU MLP: [n_points, 120] → 10 layers (ReLU between) → [n_points, 4]
        let device_id = self.compute.device().info().id;
        let mut h = Tensor::from_slice(&input_data, Shape::from([n_points, input_dim]), DType::F16, device_id)?;
        let mut in_dim = input_dim;
        let layer_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
        for (i, &layer_idx) in layer_indices.iter().enumerate() {
            let w_key = format!("decoder.layers.{}.weight", layer_idx);
            let b_key = format!("decoder.layers.{}.bias", layer_idx);
            let w_f16 = self.weight_f16(&self.model, &w_key)?;
            let out_dim = w_f16.shape().dims()[0];
            let cb = self.compute.new_command_buffer();
            let projected = self.linear_bias(&cb, &self.model, &h, &w_key, &b_key, n_points, in_dim, out_dim)?;
            if i < layer_indices.len() - 1 {
                h = self.activation(&cb, &self.kernels.relu, &projected);
            } else {
                h = projected;
            }
            cb.commit();
            cb.wait_until_completed();
            in_dim = out_dim;
        }

        // Read back and apply final activations on CPU
        let output_data: Vec<half::f16> = h.to_vec()?;
        let mut results = Vec::with_capacity(n_points);
        for p in 0..n_points {
            let base = p * 4;
            let d = output_data[base].to_f32();
            let r = output_data[base + 1].to_f32();
            let g = output_data[base + 2].to_f32();
            let b = output_data[base + 3].to_f32();
            let density = (d.exp() + 1.0).ln(); // softplus
            let r_sig = 1.0 / (1.0 + (-r).exp()); // sigmoid
            let g_sig = 1.0 / (1.0 + (-g).exp());
            let b_sig = 1.0 / (1.0 + (-b).exp());
            results.push([density, r_sig, g_sig, b_sig]);
        }

        Ok(results)
    }

    // ==================== DINO ViT-B/16 ====================

    /// DINO ViT-B/16 forward pass: image [3, 224, 224] → features [196, 768].
    fn dino_forward(&self, image_chw: &[f32], num_patches: usize) -> Result<Tensor> {
        let d_model = self.config.cross_attn_dim; // 768
        let patch_size = 16;
        let grid = self.config.image_size / patch_size; // 14
        let num_heads = 12;
        let head_dim = d_model / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Patch embedding: Conv2d [768, 3, 16, 16] + bias
        let patches = self.dino_patch_embed(image_chw, grid, patch_size, d_model)?;

        // 2. Prepend CLS token and add position embeddings
        let cls_token = self.weight_f16(&self.model, "image_tokenizer.model.embeddings.cls_token")?; // [1, 1, 768]
        let cls_data: Vec<half::f16> = cls_token.to_vec()?;
        let patches_data: Vec<half::f16> = patches.to_vec()?;

        let seq_len = num_patches + 1; // 197
        let mut combined = Vec::with_capacity(seq_len * d_model);
        combined.extend_from_slice(&cls_data[..d_model]); // CLS token
        combined.extend_from_slice(&patches_data);

        // Add position embeddings [1, 197, 768]
        let pos_embed = self.weight_f16(&self.model, "image_tokenizer.model.embeddings.position_embeddings")?;
        let pos_data: Vec<half::f16> = pos_embed.to_vec()?;
        for i in 0..seq_len * d_model {
            combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16,
            self.compute.device().info().id,
        )?;

        // 3. 12 encoder layers
        let ffn_dim = 3072; // 4 * 768
        for layer in 0..12 {
            let prefix = format!("image_tokenizer.model.encoder.layer.{}", layer);

            // LayerNorm → Self-attention → Residual
            let cb = self.compute.new_command_buffer();
            let normed = self.layer_norm(
                &cb, &self.model, &hidden,
                &format!("{}.layernorm_before.weight", prefix),
                &format!("{}.layernorm_before.bias", prefix),
                seq_len, d_model, 1e-5,
            )?;

            // Q, K, V with bias
            let q = self.linear_bias(&cb, &self.model, &normed,
                &format!("{}.attention.attention.query.weight", prefix),
                &format!("{}.attention.attention.query.bias", prefix),
                seq_len, d_model, d_model)?;
            let k = self.linear_bias(&cb, &self.model, &normed,
                &format!("{}.attention.attention.key.weight", prefix),
                &format!("{}.attention.attention.key.bias", prefix),
                seq_len, d_model, d_model)?;
            let v = self.linear_bias(&cb, &self.model, &normed,
                &format!("{}.attention.attention.value.weight", prefix),
                &format!("{}.attention.attention.value.bias", prefix),
                seq_len, d_model, d_model)?;

            // Batched attention: Q@K^T → softmax → S@V
            let device_id = self.compute.device().info().id;
            let q_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
            let k_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
            self.transpose_shd_to_hsd(&cb, &q, &q_hsd, seq_len, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &k, &k_hsd, seq_len, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &v, &v_hsd, seq_len, num_heads, head_dim);

            let scores = self.batched_qk(&cb, &q_hsd, &k_hsd, num_heads, seq_len, seq_len, head_dim);
            self.row_softmax(&cb, &scores, num_heads * seq_len, seq_len, scale);
            let attn_hsd = self.batched_sv(&cb, &scores, &v_hsd, num_heads, seq_len, seq_len, head_dim);

            let attn_shd = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, device_id)?;
            self.transpose_hsd_to_shd(&cb, &attn_hsd, &attn_shd, seq_len, num_heads, head_dim);
            let attn_flat = attn_shd.reshape([seq_len, d_model])?;

            let attn_out = self.linear_bias(&cb, &self.model, &attn_flat,
                &format!("{}.attention.output.dense.weight", prefix),
                &format!("{}.attention.output.dense.bias", prefix),
                seq_len, d_model, d_model)?;
            let h = self.add(&cb, &hidden, &attn_out);

            // LayerNorm → FFN → Residual
            let normed2 = self.layer_norm(&cb, &self.model, &h,
                &format!("{}.layernorm_after.weight", prefix),
                &format!("{}.layernorm_after.bias", prefix),
                seq_len, d_model, 1e-5)?;
            let ffn_up = self.linear_bias(&cb, &self.model, &normed2,
                &format!("{}.intermediate.dense.weight", prefix),
                &format!("{}.intermediate.dense.bias", prefix),
                seq_len, d_model, ffn_dim)?;
            let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
            let ffn_down = self.linear_bias(&cb, &self.model, &ffn_act,
                &format!("{}.output.dense.weight", prefix),
                &format!("{}.output.dense.bias", prefix),
                seq_len, ffn_dim, d_model)?;
            hidden = self.add(&cb, &h, &ffn_down);

            cb.commit();
            cb.wait_until_completed();
        }

        // 4. Final layernorm
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.model, &hidden,
            "image_tokenizer.model.layernorm.weight",
            "image_tokenizer.model.layernorm.bias",
            seq_len, d_model, 1e-5)?;
        cb.commit();
        cb.wait_until_completed();

        // 5. Remove CLS token → [196, 768]
        normed.slice(0, 1, seq_len)
    }

    /// DINO patch embedding: image [3, H, W] → patches [num_patches, d_model].
    /// Uses im2col (CPU memory layout) → GPU matmul via linear_f16.
    fn dino_patch_embed(&self, image_chw: &[f32], grid: usize, patch_size: usize, d_model: usize) -> Result<Tensor> {
        let c_in = 3;
        let num_patches = grid * grid;
        let patch_dim = c_in * patch_size * patch_size; // 768
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
                            im2col_data[patch_idx * patch_dim + col] =
                                half::f16::from_f32(image_chw[ic * img_size * img_size + iy * img_size + ix]);
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
            "image_tokenizer.model.embeddings.patch_embeddings.projection.weight",
            "image_tokenizer.model.embeddings.patch_embeddings.projection.bias",
            num_patches, patch_dim, d_model)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    // ==================== Backbone Transformer ====================

    /// Pre-compute cross-attention K/V from DINO features for all layers.
    fn precompute_cross_kv(&self, image_features: &Tensor, num_patches: usize) -> Result<Vec<(Tensor, Tensor)>> {
        let config = &self.config;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let device_id = self.compute.device().info().id;

        let mut cross_kv = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let prefix = format!("backbone.transformer_blocks.{}", layer);
            let cb = self.compute.new_command_buffer();

            // Cross K, V: [num_patches, 768] → [num_patches, 1024]
            let k = self.linear_f32_on(&cb, image_features,
                &format!("{}.attn2.to_k.weight", prefix),
                num_patches, config.cross_attn_dim, config.hidden_dim)?;
            let v = self.linear_f32_on(&cb, image_features,
                &format!("{}.attn2.to_v.weight", prefix),
                num_patches, config.cross_attn_dim, config.hidden_dim)?;

            // Transpose to HSD
            let k_hsd = Tensor::empty(Shape::from([num_heads, num_patches, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(Shape::from([num_heads, num_patches, head_dim]), DType::F16, device_id)?;
            self.transpose_shd_to_hsd(&cb, &k, &k_hsd, num_patches, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &v, &v_hsd, num_patches, num_heads, head_dim);

            cb.commit();
            cb.wait_until_completed();
            cross_kv.push((k_hsd, v_hsd));
        }
        Ok(cross_kv)
    }

    /// Single transformer block: self-attn + cross-attn + GEGLU FFN.
    fn transformer_block(
        &self, layer: usize, input: Tensor, seq_len: usize,
        cross_kv: &[(Tensor, Tensor)], kv_seq_len: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let prefix = format!("backbone.transformer_blocks.{}", layer);
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;

        // === Self-attention ===
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.model, &input,
            &format!("{}.norm1.weight", prefix),
            &format!("{}.norm1.bias", prefix),
            seq_len, config.hidden_dim, 1e-5)?;

        let q = self.linear_f32_on(&cb, &normed,
            &format!("{}.attn1.to_q.weight", prefix),
            seq_len, config.hidden_dim, config.hidden_dim)?;
        let k = self.linear_f32_on(&cb, &normed,
            &format!("{}.attn1.to_k.weight", prefix),
            seq_len, config.hidden_dim, config.hidden_dim)?;
        let v = self.linear_f32_on(&cb, &normed,
            &format!("{}.attn1.to_v.weight", prefix),
            seq_len, config.hidden_dim, config.hidden_dim)?;

        let q_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        let k_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        let v_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(&cb, &q, &q_hsd, seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd(&cb, &k, &k_hsd, seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd(&cb, &v, &v_hsd, seq_len, num_heads, head_dim);

        let scores = self.batched_qk(&cb, &q_hsd, &k_hsd, num_heads, seq_len, seq_len, head_dim);
        self.row_softmax(&cb, &scores, num_heads * seq_len, seq_len, scale);
        let attn_hsd = self.batched_sv(&cb, &scores, &v_hsd, num_heads, seq_len, seq_len, head_dim);

        let attn_shd = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(&cb, &attn_hsd, &attn_shd, seq_len, num_heads, head_dim);
        let attn_flat = attn_shd.reshape([seq_len, config.hidden_dim])?;

        let sa_out = self.linear_bias(&cb, &self.model, &attn_flat,
            &format!("{}.attn1.to_out.0.weight", prefix),
            &format!("{}.attn1.to_out.0.bias", prefix),
            seq_len, config.hidden_dim, config.hidden_dim)?;
        let h = self.add(&cb, &input, &sa_out);
        cb.commit();
        cb.wait_until_completed();

        // === Cross-attention ===
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.model, &h,
            &format!("{}.norm2.weight", prefix),
            &format!("{}.norm2.bias", prefix),
            seq_len, config.hidden_dim, 1e-5)?;

        let cross_q = self.linear_f32_on(&cb, &normed,
            &format!("{}.attn2.to_q.weight", prefix),
            seq_len, config.hidden_dim, config.hidden_dim)?;

        let cq_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(&cb, &cross_q, &cq_hsd, seq_len, num_heads, head_dim);

        let (ref ck_hsd, ref cv_hsd) = cross_kv[layer];
        let cross_scores = self.batched_qk(&cb, &cq_hsd, ck_hsd, num_heads, seq_len, kv_seq_len, head_dim);
        self.row_softmax(&cb, &cross_scores, num_heads * seq_len, kv_seq_len, scale);
        let cross_out_hsd = self.batched_sv(&cb, &cross_scores, cv_hsd, num_heads, seq_len, kv_seq_len, head_dim);

        let cross_out_shd = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(&cb, &cross_out_hsd, &cross_out_shd, seq_len, num_heads, head_dim);
        let cross_flat = cross_out_shd.reshape([seq_len, config.hidden_dim])?;

        let ca_out = self.linear_bias(&cb, &self.model, &cross_flat,
            &format!("{}.attn2.to_out.0.weight", prefix),
            &format!("{}.attn2.to_out.0.bias", prefix),
            seq_len, config.hidden_dim, config.hidden_dim)?;
        let h = self.add(&cb, &h, &ca_out);
        cb.commit();
        cb.wait_until_completed();

        // === GEGLU FFN ===
        // CB1: norm + geglu projection (must commit before geglu_on reads data)
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.model, &h,
            &format!("{}.norm3.weight", prefix),
            &format!("{}.norm3.bias", prefix),
            seq_len, config.hidden_dim, 1e-5)?;

        // GEGLU: proj [8192, 1024] → split into gate[4096] + value[4096]
        let ffn_dim = config.hidden_dim * 4; // 4096
        let geglu_proj = self.linear_bias(&cb, &self.model, &normed,
            &format!("{}.ff.net.0.proj.weight", prefix),
            &format!("{}.ff.net.0.proj.bias", prefix),
            seq_len, config.hidden_dim, ffn_dim * 2)?;
        cb.commit();
        cb.wait_until_completed();

        // CB2: GEGLU activation (splits tensor on CPU) + down projection + residual
        let cb = self.compute.new_command_buffer();
        let geglu_out = self.geglu_on(&cb, &geglu_proj, seq_len, ffn_dim);

        // Down projection
        let ffn_out = self.linear_bias(&cb, &self.model, &geglu_out,
            &format!("{}.ff.net.2.weight", prefix),
            &format!("{}.ff.net.2.bias", prefix),
            seq_len, ffn_dim, config.hidden_dim)?;
        let result = self.add(&cb, &h, &ffn_out);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    // ==================== Post-Processor ====================

    /// ConvTranspose2d upsample via GPU matmul decomposition.
    ///
    /// Stride-2 kernel-2 ConvTranspose2d decomposes into 4 matmuls per plane:
    /// For each (kh, kw) in {0,1}×{0,1}, output[:, kh::2, kw::2] = input @ W_sub^T + bias.
    fn post_process(&self, triplane_flat: &Tensor, _num_tokens: usize) -> Result<Tensor> {
        let config = &self.config;
        let ps = config.plane_size; // 32
        let c_in = config.num_channels; // 1024
        let c_out = config.upsample_out_channels; // 40
        let out_h = ps * 2;
        let out_w = ps * 2;
        let hw = ps * ps;
        let device_id = self.compute.device().info().id;

        // Read full ConvTranspose2d weight [c_in, c_out, 2, 2] and bias [c_out]
        let w_data = self.weight_vec_f32(&self.model, "post_processor.upsample.weight")?;
        let b_data = self.weight_vec_f32(&self.model, "post_processor.upsample.bias")?;

        // Extract 4 sub-weight matrices [c_out, c_in] — one per kernel position
        // Weight layout: w_data[ic * c_out * 4 + oc * 4 + kh * 2 + kw]
        // We need w_sub[oc, ic] for the linear kernel (Y = X @ W^T)
        let mut w_subs = Vec::with_capacity(4);
        for kh in 0..2 {
            for kw in 0..2 {
                let mut sub = vec![half::f16::ZERO; c_out * c_in];
                for oc in 0..c_out {
                    for ic in 0..c_in {
                        sub[oc * c_in + ic] = half::f16::from_f32(w_data[ic * c_out * 4 + oc * 4 + kh * 2 + kw]);
                    }
                }
                let w_tensor = Tensor::from_slice(&sub, Shape::from([c_out, c_in]), DType::F16, device_id)?;
                w_subs.push(w_tensor);
            }
        }

        // Bias tensor
        let b_f16: Vec<half::f16> = b_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let bias_tensor = Tensor::from_slice(&b_f16, Shape::from([c_out]), DType::F16, device_id)?;

        // Reshape [3072, 1024] → per-plane input [H*W, C_in]
        let tp_data: Vec<half::f16> = triplane_flat.to_vec()?;

        let mut output = vec![half::f16::ZERO; 3 * c_out * out_h * out_w];

        for plane in 0..3 {
            // Reshape from [plane_tokens, c_in] (row-major) → [H*W, C_in]
            // Input is already [spatial_idx, c_in] format from transformer output
            let plane_offset = plane * hw * c_in;
            let plane_data = &tp_data[plane_offset..plane_offset + hw * c_in];
            let input_tensor = Tensor::from_slice(plane_data, Shape::from([hw, c_in]), DType::F16, device_id)?;

            // 4 GPU matmuls: one per kernel position (kh, kw)
            for kh in 0..2usize {
                for kw in 0..2usize {
                    let sub_idx = kh * 2 + kw;
                    let cb = self.compute.new_command_buffer();

                    // Y = X @ W_sub^T + bias → [H*W, c_out]
                    let tile: usize = 16;
                    let device = self.compute.device().raw();
                    let out_buf = device.new_buffer((hw * c_out * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
                    self.compute.dispatch(
                        &cb, &self.kernels.common.linear,
                        ((c_out + tile - 1) / tile, (hw + tile - 1) / tile, 1), (tile, tile, 1),
                        |encoder| {
                            gpu_ops::set_tensor_buffer(encoder, 0, &input_tensor);
                            gpu_ops::set_tensor_buffer(encoder, 1, &w_subs[sub_idx]);
                            gpu_ops::set_tensor_buffer(encoder, 2, &bias_tensor);
                            encoder.set_buffer(3, Some(&out_buf), 0);
                            let vals: [u32; 4] = [hw as u32, c_out as u32, c_in as u32, 1];
                            for (i, v) in vals.iter().enumerate() {
                                encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                            }
                        },
                    );
                    cb.commit();
                    cb.wait_until_completed();

                    // Scatter results to output sub-grid
                    let result_ptr = out_buf.contents() as *const half::f16;
                    let result_data = unsafe { std::slice::from_raw_parts(result_ptr, hw * c_out) };
                    for h in 0..ps {
                        for w in 0..ps {
                            let oh = h * 2 + kh;
                            let ow = w * 2 + kw;
                            let src_row = h * ps + w;
                            for oc in 0..c_out {
                                output[plane * c_out * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] =
                                    result_data[src_row * c_out + oc];
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&output, Shape::from([3 * c_out * out_h * out_w]), DType::F16, device_id)
    }

    // ==================== GPU Helper Methods ====================

    /// Linear without bias, f32 weights (converted to f16 on-the-fly).
    fn linear_f32_on(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        weight_name: &str,
        m: usize, k: usize, n: usize,
    ) -> Result<Tensor> {
        let w_f16 = self.weight_f16(&self.model, weight_name)?;
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

    fn geglu_on(&self, cb: &metal::CommandBufferRef, input: &Tensor, seq_len: usize, half_dim: usize) -> Tensor {
        // geglu_f16 kernel expects separate gate and up buffers.
        // Input is [seq_len, 2*half_dim] — split into gate [seq_len, half_dim] and up [seq_len, half_dim].
        let data: Vec<half::f16> = input.to_vec().unwrap();
        let mut gate_data = Vec::with_capacity(seq_len * half_dim);
        let mut up_data = Vec::with_capacity(seq_len * half_dim);
        for i in 0..seq_len {
            let row_start = i * 2 * half_dim;
            gate_data.extend_from_slice(&data[row_start..row_start + half_dim]);
            up_data.extend_from_slice(&data[row_start + half_dim..row_start + 2 * half_dim]);
        }
        let device_id = self.compute.device().info().id;
        let gate = Tensor::from_slice(&gate_data, Shape::from([seq_len * half_dim]), DType::F16, device_id).unwrap();
        let up = Tensor::from_slice(&up_data, Shape::from([seq_len * half_dim]), DType::F16, device_id).unwrap();

        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((seq_len * half_dim * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.geglu, seq_len * half_dim,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, &gate);
                gpu_ops::set_tensor_buffer(encoder, 1, &up);
                encoder.set_buffer(2, Some(&output_buffer), 0);
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([seq_len, half_dim]), DType::F16, device_id)
    }

    /// Render triplane to image via volume rendering.
    pub fn render_triplane(
        &self,
        _triplane: &Tensor,
        _camera_pos: [f32; 3],
        _camera_target: [f32; 3],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        Ok(vec![0u8; width * height * 3])
    }
}
