//! InstantMesh: Multi-view image-to-3D mesh generation.
//!
//! Architecture:
//!   6 views (320×320) → resize to 224×224 → DINO ViT-B/16 with AdaLN camera modulation
//!   → concatenate [1176, 768] → triplane tokens [12288, 1024] (learned embeddings)
//!   → 16-layer transformer (self-attn + cross-attn + GEGLU FFN)
//!   → proj_out → reshape to [3, 1024, 64, 64]
//!   → ConvTranspose2d upsample [3, 80, 128, 128]
//!   → SDF MLP: triplane features → signed distance field
//!   → Marching cubes → triangle mesh
//!
//! Based on TencentARC/InstantMesh (April 2024).
//! Shares DINO ViT-B/16 + triplane transformer backbone with TripoSR.

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

/// InstantMesh configuration.
#[derive(Debug, Clone)]
pub struct InstantMeshConfig {
    /// Per-view input image size.
    pub image_size: usize,
    /// DINO input size (views are resized to this).
    pub dino_image_size: usize,
    /// Number of input views.
    pub num_views: usize,
    /// Triplane grid size (per plane).
    pub plane_size: usize,
    /// Triplane channels before upsample.
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
    /// Camera embedding dimension (Plucker ray params).
    pub camera_embed_dim: usize,
    /// Marching cubes grid resolution.
    pub grid_resolution: usize,
    /// SDF MLP hidden dimension.
    pub sdf_hidden: usize,
    /// SDF MLP number of hidden layers.
    pub sdf_num_layers: usize,
}

impl Default for InstantMeshConfig {
    fn default() -> Self {
        Self {
            image_size: 320,
            dino_image_size: 224,
            num_views: 6,
            plane_size: 64,
            num_channels: 1024,
            hidden_dim: 1024,
            num_heads: 16,
            head_dim: 64,
            num_layers: 16,
            cross_attn_dim: 768,
            upsample_out_channels: 80,
            camera_embed_dim: 16,
            grid_resolution: 128,
            sdf_hidden: 64,
            sdf_num_layers: 4,
        }
    }
}

/// Mesh output from InstantMesh.
#[derive(Debug)]
pub struct MeshOutput {
    /// Vertex positions [N, 3].
    pub vertices: Vec<[f32; 3]>,
    /// Triangle face indices [M, 3].
    pub faces: Vec<[u32; 3]>,
}

impl MeshOutput {
    /// Export mesh to OBJ format string.
    pub fn to_obj(&self) -> String {
        let mut obj = String::new();
        for v in &self.vertices {
            obj.push_str(&format!("v {} {} {}\n", v[0], v[1], v[2]));
        }
        for f in &self.faces {
            // OBJ uses 1-based indices
            obj.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
        obj
    }
}


// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct InstantMeshKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    geglu: Arc<ComputePipeline>,
    mul: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
}

// ==================== InstantMesh Pipeline ====================

/// InstantMesh pipeline for multi-view image-to-3D mesh generation.
///
/// Forward pipeline:
/// 1. DINO ViT-B/16 × 6 views: each [3, 224, 224] → [196, 768], concat → [1176, 768]
/// 2. AdaLN camera modulation per DINO layer
/// 3. Triplane tokens: learned embeddings [12288, 1024]
/// 4. Backbone: 16 transformer blocks (self-attn + cross-attn + GEGLU FFN)
/// 5. Post-processor: ConvTranspose2d → [3, 80, 128, 128]
/// 6. SDF MLP + Marching cubes → triangle mesh
#[cfg(feature = "metal")]
pub struct InstantMeshPipeline {
    model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: InstantMeshConfig,
    kernels: InstantMeshKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for InstantMeshPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl InstantMeshPipeline {
    /// Create a new InstantMesh pipeline.
    pub fn new(model: Arc<Model>, config: InstantMeshConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = InstantMeshKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            geglu: compute.compile_pipeline("geglu", sources::GELU, "geglu_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
            silu: compute.compile_pipeline("silu", sources::GELU, "silu_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
        };

        Ok(Self { model, compute, config, kernels })
    }

    /// Generate 3D mesh from 6 multi-view images.
    ///
    /// `views`: 6 images as flat f32 RGB arrays, each [3 * dino_image_size * dino_image_size] in [C, H, W] format,
    ///          pre-resized to 224×224, normalized to ImageNet stats.
    /// `cameras`: 6 camera parameter vectors, each [camera_embed_dim] (16-dim Plucker coordinates).
    pub fn generate(&self, views: &[&[f32]], cameras: &[[f32; 16]]) -> Result<MeshOutput> {
        let config = &self.config;
        assert_eq!(views.len(), config.num_views);
        assert_eq!(cameras.len(), config.num_views);

        let num_patches = (config.dino_image_size / 16) * (config.dino_image_size / 16); // 196
        let total_patches = num_patches * config.num_views; // 1176
        let num_triplane_tokens = 3 * config.plane_size * config.plane_size; // 12288

        // 1. DINO encode each view with AdaLN camera modulation, then concatenate
        let image_features = self.encode_views(views, cameras, num_patches)?;

        // 2. Load triplane token embeddings [3, 1024, 64, 64] → flatten to [12288, 1024]
        let tok_embed = self.weight_f16(&self.model, "tokenizer.embeddings")?;
        let tok_data: Vec<half::f16> = tok_embed.to_vec()?;
        let mut triplane_data = vec![half::f16::ZERO; num_triplane_tokens * config.hidden_dim];
        for plane in 0..3 {
            for h in 0..config.plane_size {
                for w in 0..config.plane_size {
                    let spatial_idx = plane * config.plane_size * config.plane_size + h * config.plane_size + w;
                    for c in 0..config.hidden_dim {
                        let src = plane * config.hidden_dim * config.plane_size * config.plane_size
                            + c * config.plane_size * config.plane_size
                            + h * config.plane_size + w;
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
        let proj_in = self.linear_bias(&cb, &self.model, &hidden,
            "backbone.proj_in.weight", "backbone.proj_in.bias",
            num_triplane_tokens, config.hidden_dim, config.hidden_dim,
        )?;
        cb.commit();
        cb.wait_until_completed();
        hidden = proj_in;

        // 4. Pre-compute cross-attention K/V from DINO features (same for all layers)
        let cross_kv = self.precompute_cross_kv(&image_features, total_patches)?;

        // 5. Transformer backbone: 16 layers
        for layer in 0..config.num_layers {
            hidden = self.transformer_block(layer, hidden, num_triplane_tokens, &cross_kv, total_patches)?;
        }

        // 6. Final norm + proj_out
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.model, &hidden,
            "backbone.norm.weight", "backbone.norm.bias",
            num_triplane_tokens, config.hidden_dim, 1e-5,
        )?;
        let proj_out = self.linear_bias(&cb, &self.model, &normed,
            "backbone.proj_out.weight", "backbone.proj_out.bias",
            num_triplane_tokens, config.hidden_dim, config.num_channels,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // 7. Reshape to [3, C, H, W] and apply ConvTranspose2d upsample
        let triplane = self.post_process(&proj_out, num_triplane_tokens)?;

        // 8. SDF evaluation at grid points + marching cubes
        let sdf_grid = self.evaluate_sdf_grid(&triplane)?;
        let mesh = marching_cubes(&sdf_grid, config.grid_resolution);

        Ok(mesh)
    }

    // ==================== Multi-View DINO Encoder ====================

    /// Encode 6 views through DINO ViT-B/16 with AdaLN camera modulation.
    /// Returns concatenated features [total_patches, 768].
    fn encode_views(&self, views: &[&[f32]], cameras: &[[f32; 16]], num_patches: usize) -> Result<Tensor> {
        let config = &self.config;
        let d_model = config.cross_attn_dim; // 768
        let device_id = self.compute.device().info().id;

        let mut all_features: Vec<half::f16> = Vec::with_capacity(
            config.num_views * num_patches * d_model,
        );

        for view_idx in 0..config.num_views {
            let view_features = self.dino_forward_adaln(
                views[view_idx], &cameras[view_idx], num_patches,
            )?;
            let data: Vec<half::f16> = view_features.to_vec()?;
            all_features.extend_from_slice(&data);
        }

        let total_patches = config.num_views * num_patches;
        Tensor::from_slice(
            &all_features,
            Shape::from([total_patches, d_model]),
            DType::F16, device_id,
        )
    }

    /// DINO ViT-B/16 forward pass with AdaLN camera modulation.
    /// image [3, 224, 224] → features [196, 768].
    fn dino_forward_adaln(&self, image_chw: &[f32], camera: &[f32; 16], num_patches: usize) -> Result<Tensor> {
        let d_model = self.config.cross_attn_dim; // 768
        let patch_size = 16;
        let grid = self.config.dino_image_size / patch_size; // 14
        let num_heads = 12;
        let head_dim = d_model / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Patch embedding
        let patches = self.dino_patch_embed(image_chw, grid, patch_size, d_model)?;

        // 2. Prepend CLS token and add position embeddings
        let cls_token = self.weight_f16(&self.model, "encoder.model.embeddings.cls_token")?;
        let cls_data: Vec<half::f16> = cls_token.to_vec()?;
        let patches_data: Vec<half::f16> = patches.to_vec()?;

        let seq_len = num_patches + 1; // 197
        let mut combined = Vec::with_capacity(seq_len * d_model);
        combined.extend_from_slice(&cls_data[..d_model]);
        combined.extend_from_slice(&patches_data);

        let pos_embed = self.weight_f16(&self.model, "encoder.model.embeddings.position_embeddings")?;
        let pos_data: Vec<half::f16> = pos_embed.to_vec()?;
        for i in 0..seq_len * d_model {
            combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16,
            self.compute.device().info().id,
        )?;

        // 3. 12 encoder layers with AdaLN camera modulation
        let ffn_dim = 3072; // 4 * 768
        for layer in 0..12 {
            let prefix = format!("encoder.model.encoder.layer.{}", layer);

            // Compute AdaLN modulation from camera params
            let (scale_vec, shift_vec) = self.compute_adaln_modulation(camera, layer, d_model)?;

            // LayerNorm → AdaLN modulate → Self-attention → Residual
            let cb = self.compute.new_command_buffer();
            let normed = self.layer_norm(&cb, &self.model, &hidden,
                &format!("{}.layernorm_before.weight", prefix),
                &format!("{}.layernorm_before.bias", prefix),
                seq_len, d_model, 1e-5,
            )?;

            // Apply AdaLN: x = normed * (1 + scale) + shift
            let modulated = self.apply_adaln_on(&cb, &normed, &scale_vec, &shift_vec, seq_len, d_model)?;

            // Q, K, V
            let q = self.linear_bias(&cb, &self.model, &modulated,
                &format!("{}.attention.attention.query.weight", prefix),
                &format!("{}.attention.attention.query.bias", prefix),
                seq_len, d_model, d_model)?;
            let k = self.linear_bias(&cb, &self.model, &modulated,
                &format!("{}.attention.attention.key.weight", prefix),
                &format!("{}.attention.attention.key.bias", prefix),
                seq_len, d_model, d_model)?;
            let v = self.linear_bias(&cb, &self.model, &modulated,
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
            "encoder.model.layernorm.weight",
            "encoder.model.layernorm.bias",
            seq_len, d_model, 1e-5)?;
        cb.commit();
        cb.wait_until_completed();

        // 5. Remove CLS token → [196, 768]
        normed.slice(0, 1, seq_len)
    }

    /// Compute AdaLN modulation vectors from camera parameters.
    /// camera [16] → MLP → (scale [d_model], shift [d_model]).
    fn compute_adaln_modulation(&self, camera: &[f32; 16], layer: usize, d_model: usize) -> Result<(Tensor, Tensor)> {
        let device_id = self.compute.device().info().id;
        let prefix = format!("encoder.camera_embed.{}", layer);

        // MLP: camera [16] → linear1 [16, d_model*2] → SiLU → linear2 [d_model*2, d_model*2]
        let cam_tensor = Tensor::from_slice(
            &camera.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>(),
            Shape::from([1, 16]),
            DType::F16, device_id,
        )?;

        let cb = self.compute.new_command_buffer();
        let h = self.linear_bias(&cb, &self.model, &cam_tensor,
            &format!("{}.0.weight", prefix),
            &format!("{}.0.bias", prefix),
            1, 16, d_model * 2)?;
        cb.commit();
        cb.wait_until_completed();

        // SiLU activation
        let cb = self.compute.new_command_buffer();
        let h_act = self.activation(&cb, &self.kernels.silu, &h);
        cb.commit();
        cb.wait_until_completed();

        let cb = self.compute.new_command_buffer();
        let out = self.linear_bias(&cb, &self.model, &h_act,
            &format!("{}.2.weight", prefix),
            &format!("{}.2.bias", prefix),
            1, d_model * 2, d_model * 2)?;
        cb.commit();
        cb.wait_until_completed();

        // Split into scale and shift
        let out_data: Vec<half::f16> = out.to_vec()?;
        let scale_data: Vec<half::f16> = out_data[..d_model].to_vec();
        let shift_data: Vec<half::f16> = out_data[d_model..].to_vec();

        let scale = Tensor::from_slice(&scale_data, Shape::from([d_model]), DType::F16, device_id)?;
        let shift = Tensor::from_slice(&shift_data, Shape::from([d_model]), DType::F16, device_id)?;

        Ok((scale, shift))
    }

    /// Apply AdaLN: output = input * (1 + scale) + shift, broadcasting over seq_len.
    fn apply_adaln_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, scale: &Tensor, shift: &Tensor,
        seq_len: usize, d_model: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;

        // Broadcast scale [d_model] → [seq_len, d_model] and add 1.0
        let scale_data: Vec<half::f16> = scale.to_vec()?;
        let shift_data: Vec<half::f16> = shift.to_vec()?;

        let mut scale_broad = Vec::with_capacity(seq_len * d_model);
        let mut shift_broad = Vec::with_capacity(seq_len * d_model);
        for _ in 0..seq_len {
            for j in 0..d_model {
                scale_broad.push(half::f16::from_f32(1.0 + scale_data[j].to_f32()));
                shift_broad.push(shift_data[j]);
            }
        }

        let scale_t = Tensor::from_slice(&scale_broad, Shape::from([seq_len, d_model]), DType::F16, device_id)?;
        let shift_t = Tensor::from_slice(&shift_broad, Shape::from([seq_len, d_model]), DType::F16, device_id)?;

        // output = input * (1 + scale)
        let scaled = self.elementwise_binary(cb, &self.kernels.mul, input, &scale_t);
        // output = scaled + shift
        Ok(self.add(cb, &scaled, &shift_t))
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
                            let val = image_chw[in_c * img_size * img_size + iy * img_size + ix];
                            col_data[p * k_size + in_c * patch_size * patch_size + ky * patch_size + kx] =
                                half::f16::from_f32(val);
                        }
                    }
                }
            }
        }

        let col_tensor = Tensor::from_slice(&col_data, Shape::from([num_patches, k_size]), DType::F16, self.compute.device().info().id)?;
        let cb = self.compute.new_command_buffer();
        let result = self.linear_bias(&cb, &self.model, &col_tensor,
            "encoder.model.embeddings.patch_embeddings.projection.weight",
            "encoder.model.embeddings.patch_embeddings.projection.bias",
            num_patches, k_size, d_model)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    // ==================== Backbone Transformer ====================

    /// Pre-compute cross-attention K/V from DINO features for all layers.
    fn precompute_cross_kv(&self, image_features: &Tensor, total_patches: usize) -> Result<Vec<(Tensor, Tensor)>> {
        let config = &self.config;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let device_id = self.compute.device().info().id;

        let mut cross_kv = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let prefix = format!("backbone.transformer_blocks.{}", layer);
            let cb = self.compute.new_command_buffer();

            let k = self.linear_f32_on(&cb, image_features,
                &format!("{}.attn2.to_k.weight", prefix),
                total_patches, config.cross_attn_dim, config.hidden_dim)?;
            let v = self.linear_f32_on(&cb, image_features,
                &format!("{}.attn2.to_v.weight", prefix),
                total_patches, config.cross_attn_dim, config.hidden_dim)?;

            let k_hsd = Tensor::empty(Shape::from([num_heads, total_patches, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(Shape::from([num_heads, total_patches, head_dim]), DType::F16, device_id)?;
            self.transpose_shd_to_hsd(&cb, &k, &k_hsd, total_patches, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &v, &v_hsd, total_patches, num_heads, head_dim);

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
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.model, &h,
            &format!("{}.norm3.weight", prefix),
            &format!("{}.norm3.bias", prefix),
            seq_len, config.hidden_dim, 1e-5)?;

        let ffn_dim = config.hidden_dim * 4;
        let geglu_proj = self.linear_bias(&cb, &self.model, &normed,
            &format!("{}.ff.net.0.proj.weight", prefix),
            &format!("{}.ff.net.0.proj.bias", prefix),
            seq_len, config.hidden_dim, ffn_dim * 2)?;
        cb.commit();
        cb.wait_until_completed();

        let cb = self.compute.new_command_buffer();
        let geglu_out = self.geglu_on(&cb, &geglu_proj, seq_len, ffn_dim);

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
        let ps = config.plane_size; // 64
        let c_in = config.num_channels; // 1024
        let c_out = config.upsample_out_channels; // 80
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

        // Reshape triplane from [3*H*W, C_in] (token-major) to per-plane [H*W, C_in]
        let tp_data: Vec<half::f16> = triplane_flat.to_vec()?;

        let mut output = vec![half::f16::ZERO; 3 * c_out * out_h * out_w];

        for plane in 0..3 {
            // Input is [spatial_idx, c_in] format from transformer output
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

    // ==================== SDF Evaluation ====================

    /// Evaluate SDF at grid points: batch triplane sampling (CPU) + GPU MLP.
    fn evaluate_sdf_grid(&self, triplane: &Tensor) -> Result<Vec<f32>> {
        let config = &self.config;
        let res = config.grid_resolution; // 128
        let out_ch = config.upsample_out_channels; // 80
        let spatial = config.plane_size * 2; // 128
        let n_points = res * res * res;
        let feat_dim = 3 * out_ch; // 240
        let device_id = self.compute.device().info().id;

        let tp_data: Vec<half::f16> = triplane.to_vec()?;

        // Batch triplane sampling on CPU (bilinear interp, no FLOPs — just memory lookup)
        let mut input_data: Vec<half::f16> = vec![half::f16::ZERO; n_points * feat_dim];
        for iz in 0..res {
            for iy in 0..res {
                for ix in 0..res {
                    let idx = iz * res * res + iy * res + ix;
                    let x = (ix as f32 / (res - 1) as f32) * 2.0 - 1.0;
                    let y = (iy as f32 / (res - 1) as f32) * 2.0 - 1.0;
                    let z = (iz as f32 / (res - 1) as f32) * 2.0 - 1.0;

                    let coords = [(x, y), (x, z), (y, z)];
                    for (plane_idx, &(u, v)) in coords.iter().enumerate() {
                        let px = ((u + 1.0) * 0.5 * (spatial - 1) as f32).clamp(0.0, (spatial - 1) as f32);
                        let py = ((v + 1.0) * 0.5 * (spatial - 1) as f32).clamp(0.0, (spatial - 1) as f32);
                        let ix0 = px.floor() as usize;
                        let iy0 = py.floor() as usize;
                        let fx = px - ix0 as f32;
                        let fy = py - iy0 as f32;
                        let ix1 = (ix0 + 1).min(spatial - 1);
                        let iy1 = (iy0 + 1).min(spatial - 1);

                        for c in 0..out_ch {
                            let base = plane_idx * out_ch * spatial * spatial + c * spatial * spatial;
                            let v00 = tp_data[base + iy0 * spatial + ix0].to_f32();
                            let v01 = tp_data[base + iy0 * spatial + ix1].to_f32();
                            let v10 = tp_data[base + iy1 * spatial + ix0].to_f32();
                            let v11 = tp_data[base + iy1 * spatial + ix1].to_f32();
                            let val = v00 * (1.0 - fx) * (1.0 - fy)
                                    + v01 * fx * (1.0 - fy)
                                    + v10 * (1.0 - fx) * fy
                                    + v11 * fx * fy;
                            input_data[idx * feat_dim + plane_idx * out_ch + c] = half::f16::from_f32(val);
                        }
                    }
                }
            }
        }

        // GPU MLP: [N, 240] → linear+ReLU × (num_layers-1) → linear → [N, 1]
        let mut h = Tensor::from_slice(&input_data, Shape::from([n_points, feat_dim]), DType::F16, device_id)?;
        let mut in_dim = feat_dim;
        for i in 0..config.sdf_num_layers {
            let w_key = format!("decoder.layers.{}.weight", i * 2);
            let b_key = format!("decoder.layers.{}.bias", i * 2);
            let w_f16 = self.weight_f16(&self.model, &w_key)?;
            let out_dim = w_f16.shape().dims()[0];

            let cb = self.compute.new_command_buffer();
            let projected = self.linear_bias(&cb, &self.model, &h, &w_key, &b_key, n_points, in_dim, out_dim)?;
            if i < config.sdf_num_layers - 1 {
                h = self.activation(&cb, &self.kernels.relu, &projected);
            } else {
                h = projected;
            }
            cb.commit();
            cb.wait_until_completed();
            in_dim = out_dim;
        }

        let sdf_f16: Vec<half::f16> = h.to_vec()?;
        Ok(sdf_f16.iter().map(|v| v.to_f32()).collect())
    }

    // ==================== GPU Helper Methods ====================

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

}

// ==================== Marching Cubes ====================

/// Standard marching cubes algorithm for extracting a triangle mesh from a signed distance field.
pub fn marching_cubes(sdf: &[f32], resolution: usize) -> MeshOutput {
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    let step = 2.0 / (resolution - 1) as f32;

    for iz in 0..resolution - 1 {
        for iy in 0..resolution - 1 {
            for ix in 0..resolution - 1 {
                // 8 corners of the cube
                let corners = [
                    (ix, iy, iz),
                    (ix + 1, iy, iz),
                    (ix + 1, iy + 1, iz),
                    (ix, iy + 1, iz),
                    (ix, iy, iz + 1),
                    (ix + 1, iy, iz + 1),
                    (ix + 1, iy + 1, iz + 1),
                    (ix, iy + 1, iz + 1),
                ];

                let values: [f32; 8] = [
                    sdf[corners[0].2 * resolution * resolution + corners[0].1 * resolution + corners[0].0],
                    sdf[corners[1].2 * resolution * resolution + corners[1].1 * resolution + corners[1].0],
                    sdf[corners[2].2 * resolution * resolution + corners[2].1 * resolution + corners[2].0],
                    sdf[corners[3].2 * resolution * resolution + corners[3].1 * resolution + corners[3].0],
                    sdf[corners[4].2 * resolution * resolution + corners[4].1 * resolution + corners[4].0],
                    sdf[corners[5].2 * resolution * resolution + corners[5].1 * resolution + corners[5].0],
                    sdf[corners[6].2 * resolution * resolution + corners[6].1 * resolution + corners[6].0],
                    sdf[corners[7].2 * resolution * resolution + corners[7].1 * resolution + corners[7].0],
                ];

                // Compute cube index (which corners are inside the surface)
                let mut cube_index: u8 = 0;
                for i in 0..8 {
                    if values[i] < 0.0 {
                        cube_index |= 1 << i;
                    }
                }

                let edges = EDGE_TABLE[cube_index as usize];
                if edges == 0 {
                    continue;
                }

                // Compute vertex positions on edges via linear interpolation
                let positions: [[f32; 3]; 8] = std::array::from_fn(|i| {
                    let (cx, cy, cz) = corners[i];
                    [
                        -1.0 + cx as f32 * step,
                        -1.0 + cy as f32 * step,
                        -1.0 + cz as f32 * step,
                    ]
                });

                let mut edge_verts = [[0.0f32; 3]; 12];
                let edge_pairs: [(usize, usize); 12] = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ];

                for (ei, &(a, b)) in edge_pairs.iter().enumerate() {
                    if edges & (1 << ei) != 0 {
                        let t = if (values[a] - values[b]).abs() > 1e-10 {
                            -values[a] / (values[b] - values[a])
                        } else {
                            0.5
                        };
                        edge_verts[ei] = [
                            positions[a][0] + t * (positions[b][0] - positions[a][0]),
                            positions[a][1] + t * (positions[b][1] - positions[a][1]),
                            positions[a][2] + t * (positions[b][2] - positions[a][2]),
                        ];
                    }
                }

                // Emit triangles
                let tri_row = &TRI_TABLE[cube_index as usize];
                let mut i = 0;
                while i < 16 && tri_row[i] != -1 {
                    let base = vertices.len() as u32;
                    vertices.push(edge_verts[tri_row[i] as usize]);
                    vertices.push(edge_verts[tri_row[i + 1] as usize]);
                    vertices.push(edge_verts[tri_row[i + 2] as usize]);
                    faces.push([base, base + 1, base + 2]);
                    i += 3;
                }
            }
        }
    }

    MeshOutput { vertices, faces }
}

// Marching cubes lookup tables (256 entries each).
// Edge table: which edges are intersected for each cube configuration.
#[rustfmt::skip]
static EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

// Triangle table: which edge vertices form triangles for each cube configuration.
// -1 marks the end of the triangle list.
#[rustfmt::skip]
static TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1],
    [3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1],
    [3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1],
    [9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1],
    [8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1],
    [3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1],
    [4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1],
    [9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1],
    [10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1],
    [5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1],
    [0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1],
    [1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1],
    [2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1],
    [7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1],
    [11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1],
    [9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1],
    [2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1],
    [6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1],
    [6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1],
    [6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1],
    [10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1],
    [10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1],
    [1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1],
    [0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1],
    [10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1],
    [6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1],
    [10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1],
    [7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1],
    [7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1],
    [7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1],
    [2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1],
    [10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1],
    [7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1],
    [6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1],
    [6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1],
    [0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1],
    [1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1],
    [10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1],
    [9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1],
    [6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1],
    [6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1],
    [9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1],
    [0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1],
    [5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1],
    [11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1],
    [9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1],
    [9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1],
    [9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1],
    [5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1],
    [5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1],
    [8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1],
    [9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1],
    [11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1],
    [4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1],
    [0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1],
    [9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1],
    [1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];
