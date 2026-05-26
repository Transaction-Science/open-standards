//! Hunyuan3D 2.0: Image-to-3D shape generation via flow-matching DiT.
//!
//! Architecture:
//!   Image (518×518) → DINOv2-Giant/14 (1536-dim, 40 layers)
//!   → FLUX-style DiT: 10 dual-stream + 11 single-stream blocks
//!   → ShapeVAE decode: 3072 latent tokens → SDF grid → marching cubes → mesh
//!
//! Flow matching with Euler ODE solver, classifier-free guidance.
//! Based on Tencent/Hunyuan3D-2 (Jan 2025).

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

use super::instantmesh::MeshOutput;

/// Hunyuan3D 2.0 configuration.
#[derive(Debug, Clone)]
pub struct Hunyuan3DConfig {
    /// DINOv2-Giant image input size.
    pub dino_image_size: usize,
    /// DINOv2-Giant hidden dimension.
    pub dino_hidden: usize,
    /// DINOv2-Giant number of heads.
    pub dino_heads: usize,
    /// DINOv2-Giant number of layers.
    pub dino_layers: usize,
    /// DINOv2-Giant patch size.
    pub dino_patch_size: usize,
    /// DiT hidden dimension.
    pub dit_hidden: usize,
    /// DiT attention head dimension.
    pub dit_head_dim: usize,
    /// DiT number of attention heads.
    pub dit_num_heads: usize,
    /// Number of dual-stream (cross-attn) blocks.
    pub dit_double_blocks: usize,
    /// Number of single-stream (joint) blocks.
    pub dit_single_blocks: usize,
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
    /// Marching cubes grid resolution.
    pub grid_resolution: usize,
    /// ShapeVAE MLP hidden dimension.
    pub vae_hidden: usize,
    /// ShapeVAE MLP layers.
    pub vae_num_layers: usize,
}

impl Default for Hunyuan3DConfig {
    fn default() -> Self {
        Self {
            dino_image_size: 518,
            dino_hidden: 1536,
            dino_heads: 24,
            dino_layers: 40,
            dino_patch_size: 14,
            dit_hidden: 1024,
            dit_head_dim: 64,
            dit_num_heads: 16,
            dit_double_blocks: 16,
            dit_single_blocks: 32,
            mlp_ratio: 4,
            num_latent_tokens: 3072,
            latent_channels: 64,
            flow_steps: 5,
            cfg_strength: 7.5,
            grid_resolution: 128,
            vae_hidden: 256,
            vae_num_layers: 4,
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct Hunyuan3DKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    sub: Arc<ComputePipeline>,
    scale: Arc<ComputePipeline>,
    adaln_modulate: Arc<ComputePipeline>,
    adaln_gate: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    swiglu_split: Arc<ComputePipeline>,
    rms_norm: Arc<ComputePipeline>,
    gelu_exact: Arc<ComputePipeline>,
}

// ==================== Hunyuan3D Pipeline ====================

/// Hunyuan3D 2.0 pipeline for image-to-3D shape generation.
///
/// Forward pipeline:
/// 1. DINOv2-Giant/14: image → [1369, 1536] features
/// 2. FLUX-style DiT: 30-step Euler ODE on [3072, 64] latent tokens
///    - 10 dual-stream blocks: separate img/latent streams with joint attention
///    - 11 single-stream blocks: concatenated joint processing
/// 3. ShapeVAE decode: latent → SDF grid → marching cubes → mesh
#[cfg(feature = "metal")]
pub struct Hunyuan3DPipeline {
    dit_model: Arc<Model>,
    dino_model: Arc<Model>,
    vae_model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: Hunyuan3DConfig,
    kernels: Hunyuan3DKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for Hunyuan3DPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl Hunyuan3DPipeline {
    /// Create a new Hunyuan3D pipeline.
    pub fn new(
        dit_model: Arc<Model>,
        dino_model: Arc<Model>,
        vae_model: Arc<Model>,
        config: Hunyuan3DConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = Hunyuan3DKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            sub: compute.compile_pipeline("sub", sources::ELEMENTWISE, "sub_f16")?,
            scale: compute.compile_pipeline("scale", sources::ELEMENTWISE, "scale_f16")?,
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ADALN, "adaln_gate_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            swiglu_split: compute.compile_pipeline("swiglu_split", sources::SWIGLU, "swiglu_split_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            gelu_exact: compute.compile_pipeline("gelu_exact", sources::GELU, "gelu_exact_f16")?,
        };

        Ok(Self { dit_model, dino_model, vae_model, compute, config, kernels })
    }

    /// Generate 3D mesh from a single image.
    ///
    /// `image_chw`: Image as flat f32 RGB array [3 * 518 * 518] in [C, H, W] format,
    ///              normalized to ImageNet stats.
    /// `seed`: Random seed for deterministic noise initialization.
    pub fn generate(&self, image_chw: &[f32], seed: u64) -> Result<MeshOutput> {
        let config = &self.config;

        println!("  [Hunyuan3D] Encoding image with DINOv2-Giant/14...");
        // dino_encode returns patches only — we need cls + patches for cond_in.
        // The verified `01_dino_final_ln` is the conditioner output (1370 tokens
        // including CLS). dino_encode currently slices off CLS at the end; for
        // DiT we want the full sequence. Re-run without the slice.
        let cond_features = self.dino_encode_full(image_chw)?;
        let num_cond_tokens = cond_features.shape().dim(0).unwrap();

        println!("  [Hunyuan3D] Running flow-matching DiT ({} steps)...", config.flow_steps);
        let latent = self.flow_matching_loop_v2(&cond_features, num_cond_tokens, seed)?;

        println!("  [Hunyuan3D] Decoding latent → mesh (volume_decode + marching_cubes, res={})...",
            config.grid_resolution);
        self.vae_latent_to_mesh(&latent, config.grid_resolution)
    }

    // ==================== DINOv2-Giant/14 Encoder ====================

    /// Public verifier hook: run only the DINO encode pass. Used by
    /// examples/hunyuan3d_dino_verify.rs to dump intermediate tensors
    /// (gated by HY3D_DUMP_DIR) and compare to the HF reference.
    pub fn encode_image(&self, image_chw: &[f32]) -> Result<Tensor> {
        self.dino_encode(image_chw)
    }

    /// Like `dino_encode` but returns the full [1370, 1536] sequence (CLS +
    /// patches), matching `01_conditioner_out__main` / `01_dino_final_ln`. This
    /// is what the DiT `cond_in` expects.
    fn dino_encode_full(&self, image_chw: &[f32]) -> Result<Tensor> {
        self.dino_encode_with_cls(image_chw)
    }

    /// Full-precision variant of dino_encode that does NOT strip CLS at the end.
    fn dino_encode_with_cls(&self, image_chw: &[f32]) -> Result<Tensor> {
        let d_model = self.config.dino_hidden;
        let patch_size = self.config.dino_patch_size;
        let grid = self.config.dino_image_size / patch_size;
        let num_patches = grid * grid;
        let num_heads = self.config.dino_heads;
        let head_dim = d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = num_patches + 1;
        let device_id = self.compute.device().info().id;

        let patches = self.dino_patch_embed(image_chw, grid, patch_size, d_model)?;

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
        let mut hidden = Tensor::from_slice(&combined,
            Shape::from([seq_len, d_model]), DType::F16, device_id)?;

        let swiglu_hidden = {
            let raw = (d_model * self.config.mlp_ratio) * 2 / 3;
            ((raw + 7) / 8) * 8
        };
        let swiglu_in_dim = 2 * swiglu_hidden;

        for layer in 0..self.config.dino_layers {
            let prefix = format!("encoder.layer.{}", layer);
            let cb = self.compute.new_command_buffer();

            let normed = self.layer_norm(&cb, &self.dino_model, &hidden,
                &format!("{}.norm1.weight", prefix),
                &format!("{}.norm1.bias", prefix),
                seq_len, d_model, 1e-6)?;
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
            let lambda1 = self.weight_f16(&self.dino_model, &format!("{}.layer_scale1.lambda1", prefix))?;
            let h = self.layer_scale_on(&cb, &hidden, &proj, &lambda1, seq_len, d_model);
            let normed2 = self.layer_norm(&cb, &self.dino_model, &h,
                &format!("{}.norm2.weight", prefix),
                &format!("{}.norm2.bias", prefix),
                seq_len, d_model, 1e-6)?;
            let mlp_in = self.linear_bias(&cb, &self.dino_model, &normed2,
                &format!("{}.mlp.weights_in.weight", prefix),
                &format!("{}.mlp.weights_in.bias", prefix),
                seq_len, d_model, swiglu_in_dim)?;
            let mlp_act = self.swiglu_split_on(&cb, &mlp_in, seq_len, swiglu_hidden);
            let mlp_out = self.linear_bias(&cb, &self.dino_model, &mlp_act,
                &format!("{}.mlp.weights_out.weight", prefix),
                &format!("{}.mlp.weights_out.bias", prefix),
                seq_len, swiglu_hidden, d_model)?;
            let lambda2 = self.weight_f16(&self.dino_model, &format!("{}.layer_scale2.lambda1", prefix))?;
            hidden = self.layer_scale_on(&cb, &h, &mlp_out, &lambda2, seq_len, d_model);
            cb.commit();
            cb.wait_until_completed();
        }

        let cb = self.compute.new_command_buffer();
        let final_ln = self.layer_norm(&cb, &self.dino_model, &hidden,
            "layernorm.weight", "layernorm.bias",
            seq_len, d_model, 1e-6)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(final_ln)
    }

    /// DINOv2-ViT-Giant/14: image → [N, 1536] patch features.
    fn dino_encode(&self, image_chw: &[f32]) -> Result<Tensor> {
        let d_model = self.config.dino_hidden; // 1536
        let patch_size = self.config.dino_patch_size; // 14
        let grid = self.config.dino_image_size / patch_size; // 37
        let num_patches = grid * grid; // 1369
        let num_heads = self.config.dino_heads; // 24
        let head_dim = d_model / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = num_patches + 1; // 1370 (CLS + patches)
        let device_id = self.compute.device().info().id;

        // HY3D_DUMP_DIR=/path : intermediate tensors are written to disk
        // for element-wise verification against the HF/MPS reference.
        // HY3D_PIXEL_INPUT=/path/to/00_pixel_values.f32 : override the host
        // preprocessing by injecting the exact [1,3,518,518] tensor the HF
        // pipeline produced. Lets us isolate model arithmetic from preprocessing.
        let dump_dir = std::env::var("HY3D_DUMP_DIR").ok();
        let pixel_override: Option<Vec<f32>> = std::env::var("HY3D_PIXEL_INPUT").ok().and_then(|p| {
            std::fs::read(&p).ok().map(|bytes| {
                let mut v = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                v
            })
        });
        let image_input: &[f32] = pixel_override.as_deref().unwrap_or(image_chw);

        // Patch embedding (CPU conv)
        let patches = self.dino_patch_embed(image_input, grid, patch_size, d_model)?;

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

        if let Some(ref dir) = dump_dir {
            let mut out = Vec::with_capacity(combined.len() * 4);
            for v in &combined { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
            std::fs::write(format!("{}/01_dino_embeddings.f32", dir), &out).ok();
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16, device_id,
        )?;

        // 40 encoder layers (pre-norm + LayerScale, SwiGLU MLP)
        //   x = x + ls1.lambda1 * attn(norm1(x))
        //   x = x + ls2.lambda1 * weights_out( swiglu_split( weights_in( norm2(x) ) ) )
        //
        // HF SwiGLUFFN hidden_features = ((hidden * mlp_ratio) * 2/3 + 7) // 8 * 8
        // For hidden=1536, mlp_ratio=4 → 4096. weights_in produces 2*4096=8192.
        let swiglu_hidden = {
            let raw = (d_model * self.config.mlp_ratio) * 2 / 3;
            ((raw + 7) / 8) * 8
        };
        let swiglu_in_dim = 2 * swiglu_hidden;

        for layer in 0..self.config.dino_layers {
            let prefix = format!("encoder.layer.{}", layer);
            let cb = self.compute.new_command_buffer();

            let normed = self.layer_norm(&cb, &self.dino_model, &hidden,
                &format!("{}.norm1.weight", prefix),
                &format!("{}.norm1.bias", prefix),
                seq_len, d_model, 1e-6)?;

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

            // LayerScale1: h = hidden + lambda1 * proj
            let lambda1 = self.weight_f16(&self.dino_model, &format!("{}.layer_scale1.lambda1", prefix))?;
            let h = self.layer_scale_on(&cb, &hidden, &proj, &lambda1, seq_len, d_model);

            let normed2 = self.layer_norm(&cb, &self.dino_model, &h,
                &format!("{}.norm2.weight", prefix),
                &format!("{}.norm2.bias", prefix),
                seq_len, d_model, 1e-6)?;

            // SwiGLU MLP: weights_in → split(silu * value) → weights_out
            let mlp_in = self.linear_bias(&cb, &self.dino_model, &normed2,
                &format!("{}.mlp.weights_in.weight", prefix),
                &format!("{}.mlp.weights_in.bias", prefix),
                seq_len, d_model, swiglu_in_dim)?;
            let mlp_act = self.swiglu_split_on(&cb, &mlp_in, seq_len, swiglu_hidden);
            let mlp_out = self.linear_bias(&cb, &self.dino_model, &mlp_act,
                &format!("{}.mlp.weights_out.weight", prefix),
                &format!("{}.mlp.weights_out.bias", prefix),
                seq_len, swiglu_hidden, d_model)?;

            // LayerScale2: hidden = h + lambda2 * mlp_out  (note: key is layer_scale2.lambda1)
            let lambda2 = self.weight_f16(&self.dino_model, &format!("{}.layer_scale2.lambda1", prefix))?;
            hidden = self.layer_scale_on(&cb, &h, &mlp_out, &lambda2, seq_len, d_model);

            cb.commit();
            cb.wait_until_completed();

            if let Some(ref dir) = dump_dir {
                let data: Vec<half::f16> = hidden.to_vec()?;
                let path = format!("{}/01_dino_layer{:02}.f32", dir, layer);
                let mut out = Vec::with_capacity(data.len() * 4);
                for v in &data { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
                std::fs::write(&path, &out).ok();
            }
        }

        // Final norm, remove CLS → [num_patches, 1536]
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(&cb, &self.dino_model, &hidden,
            "layernorm.weight", "layernorm.bias",
            seq_len, d_model, 1e-6)?;
        cb.commit();
        cb.wait_until_completed();

        if let Some(ref dir) = dump_dir {
            let data: Vec<half::f16> = normed.to_vec()?;
            let mut out = Vec::with_capacity(data.len() * 4);
            for v in &data { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
            std::fs::write(format!("{}/01_dino_final_ln.f32", dir), &out).ok();
        }

        normed.slice(0, 1, seq_len)
    }

    /// DINOv2-Giant patch embedding via im2col (CPU) + GPU matmul.
    fn dino_patch_embed(&self, image_chw: &[f32], grid: usize, patch_size: usize, d_model: usize) -> Result<Tensor> {
        let c_in = 3;
        let num_patches = grid * grid;
        let img_size = self.config.dino_image_size;
        let k_size = c_in * patch_size * patch_size;

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

    // ==================== Flow Matching Loop ====================

    /// Euler ODE flow matching with classifier-free guidance.
    fn flow_matching_loop(
        &self,
        image_features: &Tensor,
        num_img_tokens: usize,
        seed: u64,
    ) -> Result<Tensor> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;
        let numel = config.num_latent_tokens * config.latent_channels;

        // Initialize noise
        let x_data = deterministic_randn(numel, seed);
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

        // Null features for unconditional pass (zeros)
        let null_features = Tensor::from_slice(
            &vec![half::f16::ZERO; num_img_tokens * config.dino_hidden],
            Shape::from([num_img_tokens, config.dino_hidden]),
            DType::F16, device_id,
        )?;

        for step in 0..config.flow_steps {
            let t = t_seq[step];
            let dt = t_seq[step + 1] - t;

            println!("    [flow] step {}/{}: t={:.3}", step + 1, config.flow_steps, t);

            // Conditional forward
            let v_cond = self.dit_forward(&x, t, image_features, num_img_tokens)?;

            // Unconditional forward
            let v_uncond = self.dit_forward(&x, t, &null_features, num_img_tokens)?;

            // CFG: v = v_uncond + cfg * (v_cond - v_uncond)
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

    // ==================== FLUX-style DiT Forward ====================

    /// Single forward pass through the FLUX-style DiT.
    /// Dual-stream blocks: image tokens + latent tokens interact via joint attention.
    /// Single-stream blocks: concatenated joint processing.
    fn dit_forward(
        &self,
        latent: &Tensor,
        t: f32,
        img_features: &Tensor,
        num_img_tokens: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let hidden = config.dit_hidden;
        let num_heads = config.dit_num_heads;
        let head_dim = config.dit_head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mlp_dim = hidden * config.mlp_ratio;
        let num_latent = config.num_latent_tokens;
        let device_id = self.compute.device().info().id;

        // Timestep embedding: sinusoidal → MLP → [1, hidden]
        let temb = self.timestep_embedding(t)?;

        // Project image features to DiT hidden dim [N_img, hidden]
        let cb = self.compute.new_command_buffer();
        let img = self.linear_on(&cb, &self.dit_model, img_features,
            "img_proj.weight", num_img_tokens, config.dino_hidden, hidden)?;
        // Project latent to hidden dim [N_lat, hidden]
        let lat = self.linear_on(&cb, &self.dit_model, latent,
            "latent_proj.weight", num_latent, config.latent_channels, hidden)?;
        cb.commit();
        cb.wait_until_completed();

        let mut img = img;
        let mut lat = lat;

        // Dual-stream blocks (10)
        for i in 0..config.dit_double_blocks {
            let (new_img, new_lat) = self.double_block(
                &img, &lat, &temb, i,
                num_img_tokens, num_latent, hidden, num_heads, head_dim, mlp_dim, scale,
            )?;
            img = new_img;
            lat = new_lat;
        }

        // Single-stream blocks (11): concatenate img + lat
        let mut combined = Tensor::cat(&[img, lat], 0)?;
        let total_seq = num_img_tokens + num_latent;

        for i in 0..config.dit_single_blocks {
            combined = self.single_block(
                &combined, &temb, i,
                total_seq, hidden, num_heads, head_dim, mlp_dim, scale,
            )?;
        }

        // Extract latent tokens (after image tokens)
        let lat_out = combined.slice(0, num_img_tokens, total_seq)?;

        // Output projection
        let cb = self.compute.new_command_buffer();
        let out = self.linear_on(&cb, &self.dit_model, &lat_out,
            "proj_out.weight", num_latent, hidden, config.latent_channels)?;
        cb.commit();
        cb.wait_until_completed();

        Ok(out)
    }

    /// Timestep embedding: scalar t → sinusoidal → MLP → [1, hidden].
    fn timestep_embedding(&self, t: f32) -> Result<Tensor> {
        let hidden = self.config.dit_hidden;
        let device_id = self.compute.device().info().id;
        let half_dim = hidden / 2;

        // Sinusoidal embedding
        let mut emb = vec![0.0f32; hidden];
        for i in 0..half_dim {
            let freq = (-(i as f32) / half_dim as f32 * (10000.0f32).ln()).exp();
            emb[i] = (t * freq).sin();
            emb[i + half_dim] = (t * freq).cos();
        }

        // MLP: linear → SiLU → linear (on CPU for single vector)
        let w1 = self.weight_vec_f32(&self.dit_model, "time_embed.0.weight")?;
        let b1 = self.weight_vec_f32(&self.dit_model, "time_embed.0.bias")?;
        let w2 = self.weight_vec_f32(&self.dit_model, "time_embed.2.weight")?;
        let b2 = self.weight_vec_f32(&self.dit_model, "time_embed.2.bias")?;
        let mlp_dim = b1.len();

        // First linear (AMX-accelerated)
        let mut h1 = vec![0.0f32; mlp_dim];
        crate::tensor::ops::linear_amx(&emb, &w1, &b1, &mut h1, 1, hidden, mlp_dim);
        // SiLU
        for v in &mut h1 { *v = *v * (1.0 / (1.0 + (-*v).exp())); }
        // Second linear (AMX-accelerated)
        let mut h2 = vec![0.0f32; hidden];
        crate::tensor::ops::linear_amx(&h1, &w2, &b2, &mut h2, 1, mlp_dim, hidden);

        let h2_f16: Vec<half::f16> = h2.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&h2_f16, Shape::from([1, hidden]), DType::F16, device_id)
    }

    // ==================== Dual-Stream Block ====================

    /// FLUX-style dual-stream block: separate streams with joint attention.
    fn double_block(
        &self,
        img: &Tensor,
        lat: &Tensor,
        temb: &Tensor,
        block_idx: usize,
        img_seq: usize,
        lat_seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        scale: f32,
    ) -> Result<(Tensor, Tensor)> {
        let prefix = format!("double_blocks.{}", block_idx);

        // AdaLN modulation: 6 params each for img and latent
        let (img_s_a, img_sc_a, img_g_a, img_s_f, img_sc_f, img_g_f) =
            self.adaln_6params(temb, &format!("{}.img_norm.linear", prefix), hidden)?;
        let (lat_s_a, lat_sc_a, lat_g_a, lat_s_f, lat_sc_f, lat_g_f) =
            self.adaln_6params(temb, &format!("{}.lat_norm.linear", prefix), hidden)?;

        // LayerNorm + AdaLN modulate
        let cb = self.compute.new_command_buffer();
        let img_normed = self.layer_norm_bare_on(&cb, img, img_seq, hidden)?;
        let img_mod = self.adaln_modulate_on(&cb, &img_normed, &img_sc_a, &img_s_a, img_seq, hidden);
        let lat_normed = self.layer_norm_bare_on(&cb, lat, lat_seq, hidden)?;
        let lat_mod = self.adaln_modulate_on(&cb, &lat_normed, &lat_sc_a, &lat_s_a, lat_seq, hidden);

        // Q/K/V projections for both streams
        let img_q = self.linear_on(&cb, &self.dit_model, &img_mod,
            &format!("{}.img_attn.to_q.weight", prefix), img_seq, hidden, hidden)?;
        let img_k = self.linear_on(&cb, &self.dit_model, &img_mod,
            &format!("{}.img_attn.to_k.weight", prefix), img_seq, hidden, hidden)?;
        let img_v = self.linear_on(&cb, &self.dit_model, &img_mod,
            &format!("{}.img_attn.to_v.weight", prefix), img_seq, hidden, hidden)?;
        let lat_q = self.linear_on(&cb, &self.dit_model, &lat_mod,
            &format!("{}.lat_attn.to_q.weight", prefix), lat_seq, hidden, hidden)?;
        let lat_k = self.linear_on(&cb, &self.dit_model, &lat_mod,
            &format!("{}.lat_attn.to_k.weight", prefix), lat_seq, hidden, hidden)?;
        let lat_v = self.linear_on(&cb, &self.dit_model, &lat_mod,
            &format!("{}.lat_attn.to_v.weight", prefix), lat_seq, hidden, hidden)?;
        cb.commit();
        cb.wait_until_completed();

        // Joint attention: concatenate K/V from both streams
        let joint_k = Tensor::cat(&[img_k, lat_k], 0)?;
        let joint_v = Tensor::cat(&[img_v, lat_v], 0)?;
        let total_kv = img_seq + lat_seq;

        // Attention for both streams against joint K/V
        let cb = self.compute.new_command_buffer();
        let img_attn = self.batched_attention(&cb, &img_q, &joint_k, &joint_v, img_seq, total_kv, num_heads, head_dim, scale)?;
        let lat_attn = self.batched_attention(&cb, &lat_q, &joint_k, &joint_v, lat_seq, total_kv, num_heads, head_dim, scale)?;

        // Output projection + gated residual
        let img_proj = self.linear_on(&cb, &self.dit_model, &img_attn,
            &format!("{}.img_attn.to_out.weight", prefix), img_seq, hidden, hidden)?;
        let lat_proj = self.linear_on(&cb, &self.dit_model, &lat_attn,
            &format!("{}.lat_attn.to_out.weight", prefix), lat_seq, hidden, hidden)?;
        let img_after = self.adaln_gate_on(&cb, img, &img_proj, &img_g_a, img_seq, hidden);
        let lat_after = self.adaln_gate_on(&cb, lat, &lat_proj, &lat_g_a, lat_seq, hidden);

        // FFN: LayerNorm + AdaLN + GELU FFN + gated residual
        let img_ff_n = self.layer_norm_bare_on(&cb, &img_after, img_seq, hidden)?;
        let img_ff_m = self.adaln_modulate_on(&cb, &img_ff_n, &img_sc_f, &img_s_f, img_seq, hidden);
        let lat_ff_n = self.layer_norm_bare_on(&cb, &lat_after, lat_seq, hidden)?;
        let lat_ff_m = self.adaln_modulate_on(&cb, &lat_ff_n, &lat_sc_f, &lat_s_f, lat_seq, hidden);

        let img_up = self.linear_on(&cb, &self.dit_model, &img_ff_m,
            &format!("{}.img_ff.0.weight", prefix), img_seq, hidden, mlp_dim)?;
        let img_act = self.activation(&cb, &self.kernels.gelu, &img_up);
        let img_down = self.linear_on(&cb, &self.dit_model, &img_act,
            &format!("{}.img_ff.2.weight", prefix), img_seq, mlp_dim, hidden)?;
        let lat_up = self.linear_on(&cb, &self.dit_model, &lat_ff_m,
            &format!("{}.lat_ff.0.weight", prefix), lat_seq, hidden, mlp_dim)?;
        let lat_act = self.activation(&cb, &self.kernels.gelu, &lat_up);
        let lat_down = self.linear_on(&cb, &self.dit_model, &lat_act,
            &format!("{}.lat_ff.2.weight", prefix), lat_seq, mlp_dim, hidden)?;

        let img_out = self.adaln_gate_on(&cb, &img_after, &img_down, &img_g_f, img_seq, hidden);
        let lat_out = self.adaln_gate_on(&cb, &lat_after, &lat_down, &lat_g_f, lat_seq, hidden);
        cb.commit();
        cb.wait_until_completed();

        Ok((img_out, lat_out))
    }

    // ==================== Single-Stream Block ====================

    /// FLUX-style single-stream block: joint processing on concatenated tokens.
    fn single_block(
        &self,
        hidden_state: &Tensor,
        temb: &Tensor,
        block_idx: usize,
        seq_len: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let prefix = format!("single_blocks.{}", block_idx);

        // AdaLN: 3 params (shift, scale, gate)
        let (shift, scale_param, gate) = self.adaln_3params(
            temb, &format!("{}.norm.linear", prefix), hidden,
        )?;

        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm_bare_on(&cb, hidden_state, seq_len, hidden)?;
        let modulated = self.adaln_modulate_on(&cb, &normed, &scale_param, &shift, seq_len, hidden);

        // Q/K/V + parallel MLP
        let q = self.linear_on(&cb, &self.dit_model, &modulated,
            &format!("{}.attn.to_q.weight", prefix), seq_len, hidden, hidden)?;
        let k = self.linear_on(&cb, &self.dit_model, &modulated,
            &format!("{}.attn.to_k.weight", prefix), seq_len, hidden, hidden)?;
        let v = self.linear_on(&cb, &self.dit_model, &modulated,
            &format!("{}.attn.to_v.weight", prefix), seq_len, hidden, hidden)?;
        let mlp_up = self.linear_on(&cb, &self.dit_model, &modulated,
            &format!("{}.proj_mlp.weight", prefix), seq_len, hidden, mlp_dim)?;
        let mlp_act = self.activation(&cb, &self.kernels.gelu, &mlp_up);
        cb.commit();
        cb.wait_until_completed();

        // Attention
        let cb = self.compute.new_command_buffer();
        let attn_out = self.batched_attention(&cb, &q, &k, &v, seq_len, seq_len, num_heads, head_dim, scale)?;

        // Output: project attn + concat MLP, then gated residual
        let attn_proj = self.linear_on(&cb, &self.dit_model, &attn_out,
            &format!("{}.attn.to_out.weight", prefix), seq_len, hidden, hidden)?;
        let mlp_down = self.linear_on(&cb, &self.dit_model, &mlp_act,
            &format!("{}.proj_out.weight", prefix), seq_len, mlp_dim, hidden)?;
        let combined_out = self.add(&cb, &attn_proj, &mlp_down);
        let result = self.adaln_gate_on(&cb, hidden_state, &combined_out, &gate, seq_len, hidden);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    // ==================== AdaLN Helpers ====================

    /// Extract 6 AdaLN modulation parameters from temb → linear → chunk(6).
    fn adaln_6params(
        &self, temb: &Tensor, weight_name: &str, hidden: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let cb = self.compute.new_command_buffer();
        let proj = self.linear_on(&cb, &self.dit_model, temb, weight_name, 1, hidden, hidden * 6)?;
        let silu_out = self.activation(&cb, &self.kernels.silu, &proj);
        cb.commit();
        cb.wait_until_completed();

        let data: Vec<half::f16> = silu_out.to_vec()?;
        let device_id = self.compute.device().info().id;
        let mk = |start: usize| -> Result<Tensor> {
            Tensor::from_slice(&data[start..start + hidden], Shape::from([1, hidden]), DType::F16, device_id)
        };
        Ok((mk(0)?, mk(hidden)?, mk(hidden * 2)?, mk(hidden * 3)?, mk(hidden * 4)?, mk(hidden * 5)?))
    }

    /// Extract 3 AdaLN modulation parameters from temb → linear → chunk(3).
    fn adaln_3params(
        &self, temb: &Tensor, weight_name: &str, hidden: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cb = self.compute.new_command_buffer();
        let proj = self.linear_on(&cb, &self.dit_model, temb, weight_name, 1, hidden, hidden * 3)?;
        let silu_out = self.activation(&cb, &self.kernels.silu, &proj);
        cb.commit();
        cb.wait_until_completed();

        let data: Vec<half::f16> = silu_out.to_vec()?;
        let device_id = self.compute.device().info().id;
        let mk = |start: usize| -> Result<Tensor> {
            Tensor::from_slice(&data[start..start + hidden], Shape::from([1, hidden]), DType::F16, device_id)
        };
        Ok((mk(0)?, mk(hidden)?, mk(hidden * 2)?))
    }

    // ==================== ShapeVAE Decode ====================

    /// Decode latent tokens to SDF grid via batched GPU MLP.
    fn vae_decode_sdf(&self, latent: &Tensor) -> Result<Vec<f32>> {
        let config = &self.config;
        let res = config.grid_resolution;
        let n_points = res * res * res;
        let latent_data: Vec<half::f16> = latent.to_vec()?;
        let device_id = self.compute.device().info().id;
        let input_dim = config.latent_channels + 3;

        // Build batched input [N, latent_channels + 3] on CPU (indexing only)
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

        // GPU MLP: linear → ReLU → ... → linear
        let mut in_dim = input_dim;
        for i in 0..config.vae_num_layers {
            let w_key = format!("decoder.layers.{}.weight", i);
            let b_key = format!("decoder.layers.{}.bias", i);
            let w_f16 = self.weight_f16(&self.vae_model, &w_key)?;
            let out_dim = w_f16.shape().dims()[0];

            let cb = self.compute.new_command_buffer();
            let projected = self.linear_bias(&cb, &self.vae_model, &h, &w_key, &b_key, n_points, in_dim, out_dim)?;
            if i < config.vae_num_layers - 1 {
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

    /// Layer norm without affine params (bare normalization for AdaLN).
    fn layer_norm_bare_on(
        &self, cb: &metal::CommandBufferRef, input: &Tensor, n: usize, d: usize,
    ) -> Result<Tensor> {
        // Use ones/zeros as dummy weight/bias
        let device_id = self.compute.device().info().id;
        let ones = Tensor::from_slice(
            &vec![half::f16::ONE; d], Shape::from([d]), DType::F16, device_id,
        )?;
        let zeros = Tensor::from_slice(
            &vec![half::f16::ZERO; d], Shape::from([d]), DType::F16, device_id,
        )?;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.common.layer_norm, n,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &ones);
                gpu_ops::set_tensor_buffer(encoder, 2, &zeros);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let n_u32 = n as u32; let d_u32 = d as u32; let eps: f32 = 1e-6;
                encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
            },
        );
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([n, d]), DType::F16, self.compute.device().info().id))
    }

    fn adaln_modulate_on(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        scale: &Tensor, shift: &Tensor, n: usize, d: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.adaln_modulate, n * d,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, scale);
                gpu_ops::set_tensor_buffer(encoder, 2, shift);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let d_u32 = d as u32;
                let count_u32 = (n * d) as u32;
                encoder.set_bytes(4, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &count_u32 as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([n, d]), DType::F16, self.compute.device().info().id)
    }

    fn adaln_gate_on(
        &self, cb: &metal::CommandBufferRef,
        residual: &Tensor, projected: &Tensor, gate: &Tensor,
        n: usize, d: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.adaln_gate, n * d,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, residual);
                gpu_ops::set_tensor_buffer(encoder, 1, projected);
                gpu_ops::set_tensor_buffer(encoder, 2, gate);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let d_u32 = d as u32;
                let count_u32 = (n * d) as u32;
                encoder.set_bytes(4, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &count_u32 as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([n, d]), DType::F16, self.compute.device().info().id)
    }

    /// SwiGLU split: input [n, 2*half_dim] -> output [n, half_dim]
    /// output[n, h] = silu(input[n, h]) * input[n, half_dim + h]
    fn swiglu_split_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, n: usize, half_dim: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let count = n * half_dim;
        let output_buffer = device.new_buffer((count * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.swiglu_split, count,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(&output_buffer), 0);
                let half_u32 = half_dim as u32;
                let count_u32 = count as u32;
                encoder.set_bytes(2, 4, &half_u32 as *const u32 as *const _);
                encoder.set_bytes(3, 4, &count_u32 as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([n, half_dim]), DType::F16, self.compute.device().info().id)
    }

    /// Per-residual LayerScale: residual + lambda[h] * branch[i].
    /// Reuses adaln_gate kernel (semantics match exactly).
    fn layer_scale_on(
        &self, cb: &metal::CommandBufferRef,
        residual: &Tensor, branch: &Tensor, lambda: &Tensor,
        n: usize, d: usize,
    ) -> Tensor {
        self.adaln_gate_on(cb, residual, branch, lambda, n, d)
    }

    // ==================== FLUX-style DiT helpers ====================
    //
    // Implements the Hunyuan3D-2 DiT (FLUX-style) per hy3dgen's
    // `hunyuan3ddit.py`. Uses real `model.*` keys from dit_model.safetensors.

    /// Per-head RMSNorm with a [head_dim] scale parameter.
    /// Input is [seq, hidden] flat; the kernel treats it as [seq*num_heads, head_dim]
    /// so each (seq, head) row is normalised independently and scaled by `scale_w`.
    fn rms_norm_per_head_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, scale_w: &Tensor,
        seq: usize, num_heads: usize, head_dim: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let total = seq * num_heads;
        let output_buffer = device.new_buffer((total * head_dim * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.rms_norm, total,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, scale_w);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                let n_u32 = total as u32;
                let d_u32 = head_dim as u32;
                let eps: f32 = 1e-6;
                encoder.set_bytes(3, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([seq, num_heads * head_dim]), DType::F16, self.compute.device().info().id)
    }

    /// FLUX-style modulation: vec → silu → linear[hidden, k*hidden] → chunk(k).
    /// Returns k tensors each of shape [1, hidden].
    fn dit_modulation(
        &self, cb: &metal::CommandBufferRef,
        vec: &Tensor, prefix: &str, hidden: usize, num_chunks: usize,
    ) -> Result<Vec<Tensor>> {
        // SiLU on vec [1, hidden]
        let silu_vec = self.activation(cb, &self.kernels.silu, vec);
        // Linear to [1, num_chunks * hidden]
        let proj = self.linear_bias(cb, &self.dit_model, &silu_vec,
            &format!("{}.weight", prefix), &format!("{}.bias", prefix),
            1, hidden, num_chunks * hidden)?;
        // Chunk along last dim → num_chunks tensors each [1, hidden]
        cb.commit();
        cb.wait_until_completed();
        let flat: Vec<half::f16> = proj.to_vec()?;
        let device_id = self.compute.device().info().id;
        let mut chunks = Vec::with_capacity(num_chunks);
        for c in 0..num_chunks {
            let start = c * hidden;
            let slice = flat[start..start + hidden].to_vec();
            chunks.push(Tensor::from_slice(&slice, Shape::from([hidden]), DType::F16, device_id)?);
        }
        Ok(chunks)
    }

    /// FLUX-style time embedding: sinusoidal(t, 256) → in_layer → silu → out_layer.
    /// `t_sinusoid_override`, when Some, replaces the host-computed sinusoidal
    /// with the captured-from-HF tensor (for cross-impl element-wise verification).
    fn dit_time_embed(&self, t: f32, t_sinusoid_override: Option<&[f32]>) -> Result<Tensor> {
        let hidden = self.config.dit_hidden;
        let device_id = self.compute.device().info().id;
        let dim = 256_usize;

        // sinusoidal positional embedding per FLUX timestep_embedding (max_period=10000,
        // time_factor=1000 applied OUTSIDE in hunyuan3ddit; we receive raw t already scaled
        // OR an override).
        let sinusoid: Vec<half::f16> = if let Some(over) = t_sinusoid_override {
            over.iter().map(|&v| half::f16::from_f32(v)).collect()
        } else {
            let half_dim = dim / 2;
            let mut emb = vec![0.0f32; dim];
            let t_scaled = t * 1000.0;
            for i in 0..half_dim {
                let freq = (-(i as f32) * (10000.0f32).ln() / (half_dim as f32 - 1.0)).exp();
                emb[i] = (t_scaled * freq).cos();
                emb[half_dim + i] = (t_scaled * freq).sin();
            }
            emb.iter().map(|&v| half::f16::from_f32(v)).collect()
        };
        let sin_tensor = Tensor::from_slice(&sinusoid, Shape::from([1, dim]), DType::F16, device_id)?;

        let cb = self.compute.new_command_buffer();
        let h1 = self.linear_bias(&cb, &self.dit_model, &sin_tensor,
            "model.time_in.in_layer.weight", "model.time_in.in_layer.bias",
            1, dim, hidden)?;
        let h1_silu = self.activation(&cb, &self.kernels.silu, &h1);
        let vec = self.linear_bias(&cb, &self.dit_model, &h1_silu,
            "model.time_in.out_layer.weight", "model.time_in.out_layer.bias",
            1, hidden, hidden)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(vec)
    }

    /// One FLUX-style DoubleStreamBlock (img + txt with joint attention).
    /// Mirrors hy3dgen's `DoubleStreamBlock.forward()` exactly.
    /// `img` = latent stream [img_seq, hidden]; `txt` = cond stream [txt_seq, hidden]; `vec` = [1, hidden].
    fn dit_double_block_flux(
        &self,
        img: &Tensor, txt: &Tensor, vec: &Tensor,
        block_idx: usize, hidden: usize, num_heads: usize, head_dim: usize,
        img_seq: usize, txt_seq: usize,
    ) -> Result<(Tensor, Tensor)> {
        let prefix = format!("model.double_blocks.{}", block_idx);
        let mlp_dim = hidden * self.config.mlp_ratio;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Modulation: img_mod → 6 chunks; txt_mod → 6 chunks. Order: (shift1, scale1, gate1, shift2, scale2, gate2).
        let mod_cb = self.compute.new_command_buffer();
        let img_mod = self.dit_modulation(&mod_cb, vec, &format!("{}.img_mod.lin", prefix), hidden, 6)?;
        let mod_cb2 = self.compute.new_command_buffer();
        let txt_mod = self.dit_modulation(&mod_cb2, vec, &format!("{}.txt_mod.lin", prefix), hidden, 6)?;
        let (img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2) =
            (&img_mod[0], &img_mod[1], &img_mod[2], &img_mod[3], &img_mod[4], &img_mod[5]);
        let (txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2) =
            (&txt_mod[0], &txt_mod[1], &txt_mod[2], &txt_mod[3], &txt_mod[4], &txt_mod[5]);

        let cb = self.compute.new_command_buffer();

        // ---- img stream: norm1 + modulate + qkv + QKNorm ----
        let img_n1 = self.layer_norm_bare_on(&cb, img, img_seq, hidden)?;
        let img_modulated = self.adaln_modulate_on(&cb, &img_n1, img_scale1, img_shift1, img_seq, hidden);
        let img_qkv = self.linear_bias(&cb, &self.dit_model, &img_modulated,
            &format!("{}.img_attn.qkv.weight", prefix),
            &format!("{}.img_attn.qkv.bias", prefix),
            img_seq, hidden, 3 * hidden)?;
        cb.commit();
        cb.wait_until_completed();

        // Slice img_qkv [img_seq, 3*hidden] into Q, K, V along the last dim.
        // To keep elementwise ops happy (slice updates strides but kernels need contig),
        // do the split host-side once and rebuild contiguous Q/K/V buffers.
        let img_qkv_data: Vec<half::f16> = img_qkv.to_vec()?;
        let mut img_q_v = Vec::with_capacity(img_seq * hidden);
        let mut img_k_v = Vec::with_capacity(img_seq * hidden);
        let mut img_v_v = Vec::with_capacity(img_seq * hidden);
        for s in 0..img_seq {
            let base = s * 3 * hidden;
            img_q_v.extend_from_slice(&img_qkv_data[base..base + hidden]);
            img_k_v.extend_from_slice(&img_qkv_data[base + hidden..base + 2 * hidden]);
            img_v_v.extend_from_slice(&img_qkv_data[base + 2 * hidden..base + 3 * hidden]);
        }
        let device_id = self.compute.device().info().id;
        let img_q = Tensor::from_slice(&img_q_v, Shape::from([img_seq, hidden]), DType::F16, device_id)?;
        let img_k = Tensor::from_slice(&img_k_v, Shape::from([img_seq, hidden]), DType::F16, device_id)?;
        let img_v = Tensor::from_slice(&img_v_v, Shape::from([img_seq, hidden]), DType::F16, device_id)?;

        // ---- txt stream: same ----
        let cb = self.compute.new_command_buffer();
        let txt_n1 = self.layer_norm_bare_on(&cb, txt, txt_seq, hidden)?;
        let txt_modulated = self.adaln_modulate_on(&cb, &txt_n1, txt_scale1, txt_shift1, txt_seq, hidden);
        let txt_qkv = self.linear_bias(&cb, &self.dit_model, &txt_modulated,
            &format!("{}.txt_attn.qkv.weight", prefix),
            &format!("{}.txt_attn.qkv.bias", prefix),
            txt_seq, hidden, 3 * hidden)?;
        cb.commit();
        cb.wait_until_completed();
        let txt_qkv_data: Vec<half::f16> = txt_qkv.to_vec()?;
        let mut txt_q_v = Vec::with_capacity(txt_seq * hidden);
        let mut txt_k_v = Vec::with_capacity(txt_seq * hidden);
        let mut txt_v_v = Vec::with_capacity(txt_seq * hidden);
        for s in 0..txt_seq {
            let base = s * 3 * hidden;
            txt_q_v.extend_from_slice(&txt_qkv_data[base..base + hidden]);
            txt_k_v.extend_from_slice(&txt_qkv_data[base + hidden..base + 2 * hidden]);
            txt_v_v.extend_from_slice(&txt_qkv_data[base + 2 * hidden..base + 3 * hidden]);
        }
        let txt_q = Tensor::from_slice(&txt_q_v, Shape::from([txt_seq, hidden]), DType::F16, device_id)?;
        let txt_k = Tensor::from_slice(&txt_k_v, Shape::from([txt_seq, hidden]), DType::F16, device_id)?;
        let txt_v = Tensor::from_slice(&txt_v_v, Shape::from([txt_seq, hidden]), DType::F16, device_id)?;

        // QKNorm: per-head RMSNorm on Q and K (V untouched). scale param shape [head_dim].
        let cb = self.compute.new_command_buffer();
        let img_qn_scale = self.weight_f16(&self.dit_model, &format!("{}.img_attn.norm.query_norm.scale", prefix))?;
        let img_kn_scale = self.weight_f16(&self.dit_model, &format!("{}.img_attn.norm.key_norm.scale", prefix))?;
        let img_qn = self.rms_norm_per_head_on(&cb, &img_q, &img_qn_scale, img_seq, num_heads, head_dim);
        let img_kn = self.rms_norm_per_head_on(&cb, &img_k, &img_kn_scale, img_seq, num_heads, head_dim);
        let txt_qn_scale = self.weight_f16(&self.dit_model, &format!("{}.txt_attn.norm.query_norm.scale", prefix))?;
        let txt_kn_scale = self.weight_f16(&self.dit_model, &format!("{}.txt_attn.norm.key_norm.scale", prefix))?;
        let txt_qn = self.rms_norm_per_head_on(&cb, &txt_q, &txt_qn_scale, txt_seq, num_heads, head_dim);
        let txt_kn = self.rms_norm_per_head_on(&cb, &txt_k, &txt_kn_scale, txt_seq, num_heads, head_dim);
        cb.commit();
        cb.wait_until_completed();

        // Joint attention: concat [txt, img] along seq dim, then split back.
        let joint_q = Tensor::cat(&[txt_qn, img_qn], 0)?;
        let joint_k = Tensor::cat(&[txt_kn, img_kn], 0)?;
        let joint_v = Tensor::cat(&[txt_v, img_v], 0)?;
        let total = txt_seq + img_seq;

        let cb = self.compute.new_command_buffer();
        let attn = self.batched_attention(&cb, &joint_q, &joint_k, &joint_v, total, total, num_heads, head_dim, scale)?;
        cb.commit();
        cb.wait_until_completed();
        let attn_data: Vec<half::f16> = attn.to_vec()?;
        let txt_attn_data = attn_data[..txt_seq * hidden].to_vec();
        let img_attn_data = attn_data[txt_seq * hidden..].to_vec();
        let txt_attn = Tensor::from_slice(&txt_attn_data, Shape::from([txt_seq, hidden]), DType::F16, device_id)?;
        let img_attn = Tensor::from_slice(&img_attn_data, Shape::from([img_seq, hidden]), DType::F16, device_id)?;

        // proj + gated residual then MLP + gated residual, for each stream.
        let cb = self.compute.new_command_buffer();
        let img_proj = self.linear_bias(&cb, &self.dit_model, &img_attn,
            &format!("{}.img_attn.proj.weight", prefix),
            &format!("{}.img_attn.proj.bias", prefix),
            img_seq, hidden, hidden)?;
        let img_after_attn = self.adaln_gate_on(&cb, img, &img_proj, img_gate1, img_seq, hidden);
        let img_n2 = self.layer_norm_bare_on(&cb, &img_after_attn, img_seq, hidden)?;
        let img_mod2_in = self.adaln_modulate_on(&cb, &img_n2, img_scale2, img_shift2, img_seq, hidden);
        let img_mlp1 = self.linear_bias(&cb, &self.dit_model, &img_mod2_in,
            &format!("{}.img_mlp.0.weight", prefix),
            &format!("{}.img_mlp.0.bias", prefix),
            img_seq, hidden, mlp_dim)?;
        let img_mlp_act = self.activation(&cb, &self.kernels.gelu, &img_mlp1);
        let img_mlp2 = self.linear_bias(&cb, &self.dit_model, &img_mlp_act,
            &format!("{}.img_mlp.2.weight", prefix),
            &format!("{}.img_mlp.2.bias", prefix),
            img_seq, mlp_dim, hidden)?;
        let img_out = self.adaln_gate_on(&cb, &img_after_attn, &img_mlp2, img_gate2, img_seq, hidden);

        let txt_proj = self.linear_bias(&cb, &self.dit_model, &txt_attn,
            &format!("{}.txt_attn.proj.weight", prefix),
            &format!("{}.txt_attn.proj.bias", prefix),
            txt_seq, hidden, hidden)?;
        let txt_after_attn = self.adaln_gate_on(&cb, txt, &txt_proj, txt_gate1, txt_seq, hidden);
        let txt_n2 = self.layer_norm_bare_on(&cb, &txt_after_attn, txt_seq, hidden)?;
        let txt_mod2_in = self.adaln_modulate_on(&cb, &txt_n2, txt_scale2, txt_shift2, txt_seq, hidden);
        let txt_mlp1 = self.linear_bias(&cb, &self.dit_model, &txt_mod2_in,
            &format!("{}.txt_mlp.0.weight", prefix),
            &format!("{}.txt_mlp.0.bias", prefix),
            txt_seq, hidden, mlp_dim)?;
        let txt_mlp_act = self.activation(&cb, &self.kernels.gelu, &txt_mlp1);
        let txt_mlp2 = self.linear_bias(&cb, &self.dit_model, &txt_mlp_act,
            &format!("{}.txt_mlp.2.weight", prefix),
            &format!("{}.txt_mlp.2.bias", prefix),
            txt_seq, mlp_dim, hidden)?;
        let txt_out = self.adaln_gate_on(&cb, &txt_after_attn, &txt_mlp2, txt_gate2, txt_seq, hidden);
        cb.commit();
        cb.wait_until_completed();

        Ok((img_out, txt_out))
    }

    /// One FLUX-style SingleStreamBlock (parallel attn + MLP, joint stream).
    /// Mirrors hy3dgen's `SingleStreamBlock.forward()` exactly.
    /// `x` is the joint stream [seq, hidden] (txt || img concatenated).
    fn dit_single_block_flux(
        &self,
        x: &Tensor, vec: &Tensor,
        block_idx: usize, hidden: usize, num_heads: usize, head_dim: usize, seq: usize,
    ) -> Result<Tensor> {
        let prefix = format!("model.single_blocks.{}", block_idx);
        let mlp_dim = hidden * self.config.mlp_ratio;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Modulation: 3 chunks (shift, scale, gate).
        let mod_cb = self.compute.new_command_buffer();
        let m = self.dit_modulation(&mod_cb, vec, &format!("{}.modulation.lin", prefix), hidden, 3)?;
        let (shift, scale_p, gate) = (&m[0], &m[1], &m[2]);

        let cb = self.compute.new_command_buffer();
        let pre = self.layer_norm_bare_on(&cb, x, seq, hidden)?;
        let x_mod = self.adaln_modulate_on(&cb, &pre, scale_p, shift, seq, hidden);

        // linear1: [seq, hidden] → [seq, 3*hidden + mlp_dim]
        let out_dim1 = 3 * hidden + mlp_dim;
        let l1 = self.linear_bias(&cb, &self.dit_model, &x_mod,
            &format!("{}.linear1.weight", prefix),
            &format!("{}.linear1.bias", prefix),
            seq, hidden, out_dim1)?;
        cb.commit();
        cb.wait_until_completed();

        // Split linear1 output into QKV (first 3*hidden) and MLP-in (next mlp_dim).
        let l1_data: Vec<half::f16> = l1.to_vec()?;
        let mut q_v = Vec::with_capacity(seq * hidden);
        let mut k_v = Vec::with_capacity(seq * hidden);
        let mut v_v = Vec::with_capacity(seq * hidden);
        let mut mlp_v = Vec::with_capacity(seq * mlp_dim);
        for s in 0..seq {
            let base = s * out_dim1;
            q_v.extend_from_slice(&l1_data[base..base + hidden]);
            k_v.extend_from_slice(&l1_data[base + hidden..base + 2 * hidden]);
            v_v.extend_from_slice(&l1_data[base + 2 * hidden..base + 3 * hidden]);
            mlp_v.extend_from_slice(&l1_data[base + 3 * hidden..base + 3 * hidden + mlp_dim]);
        }
        let device_id = self.compute.device().info().id;
        let q = Tensor::from_slice(&q_v, Shape::from([seq, hidden]), DType::F16, device_id)?;
        let k = Tensor::from_slice(&k_v, Shape::from([seq, hidden]), DType::F16, device_id)?;
        let v_t = Tensor::from_slice(&v_v, Shape::from([seq, hidden]), DType::F16, device_id)?;
        let mlp_in = Tensor::from_slice(&mlp_v, Shape::from([seq, mlp_dim]), DType::F16, device_id)?;

        // QKNorm.
        let cb = self.compute.new_command_buffer();
        let q_scale = self.weight_f16(&self.dit_model, &format!("{}.norm.query_norm.scale", prefix))?;
        let k_scale = self.weight_f16(&self.dit_model, &format!("{}.norm.key_norm.scale", prefix))?;
        let qn = self.rms_norm_per_head_on(&cb, &q, &q_scale, seq, num_heads, head_dim);
        let kn = self.rms_norm_per_head_on(&cb, &k, &k_scale, seq, num_heads, head_dim);

        // Attention (joint sequence — no cat needed).
        let attn = self.batched_attention(&cb, &qn, &kn, &v_t, seq, seq, num_heads, head_dim, scale)?;
        // MLP path: gelu_tanh(mlp_in)
        let mlp_act = self.activation(&cb, &self.kernels.gelu, &mlp_in);
        cb.commit();
        cb.wait_until_completed();

        // Concatenate (attn, mlp_act) along last dim into [seq, hidden + mlp_dim], host-side.
        let attn_data: Vec<half::f16> = attn.to_vec()?;
        let mlp_act_data: Vec<half::f16> = mlp_act.to_vec()?;
        let mut cat_data = Vec::with_capacity(seq * (hidden + mlp_dim));
        for s in 0..seq {
            cat_data.extend_from_slice(&attn_data[s * hidden..(s + 1) * hidden]);
            cat_data.extend_from_slice(&mlp_act_data[s * mlp_dim..(s + 1) * mlp_dim]);
        }
        let cat_tensor = Tensor::from_slice(&cat_data,
            Shape::from([seq, hidden + mlp_dim]), DType::F16, device_id)?;

        // linear2: [seq, hidden + mlp_dim] → [seq, hidden]
        let cb = self.compute.new_command_buffer();
        let l2 = self.linear_bias(&cb, &self.dit_model, &cat_tensor,
            &format!("{}.linear2.weight", prefix),
            &format!("{}.linear2.bias", prefix),
            seq, hidden + mlp_dim, hidden)?;
        // Gated residual: x + gate * l2
        let out = self.adaln_gate_on(&cb, x, &l2, gate, seq, hidden);
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// FLUX-style LastLayer: adaLN_modulation (silu→linear→chunk2 = shift, scale)
    /// + (1 + scale) * LN_bare(x) + shift → final linear to out_channels.
    fn dit_final_layer_flux(
        &self,
        x: &Tensor, vec: &Tensor,
        hidden: usize, out_ch: usize, seq: usize,
    ) -> Result<Tensor> {
        // adaLN_modulation = Sequential(SiLU, Linear(hidden, 2*hidden))
        // weight key: model.final_layer.adaLN_modulation.1.{weight,bias}
        let mod_cb = self.compute.new_command_buffer();
        let chunks = self.dit_modulation(
            &mod_cb, vec,
            "model.final_layer.adaLN_modulation.1",
            hidden, 2)?;
        let (shift, scale_p) = (&chunks[0], &chunks[1]);
        // Note: hy3dgen's LastLayer.forward does `shift, scale = ...chunk(2, dim=1)`
        // — shift FIRST. dit_modulation returns chunks in the order they appear,
        // so chunks[0] = shift, chunks[1] = scale. (Different from Modulation which
        // returned shift, scale, gate.)

        let cb = self.compute.new_command_buffer();
        let pre = self.layer_norm_bare_on(&cb, x, seq, hidden)?;
        let x_mod = self.adaln_modulate_on(&cb, &pre, scale_p, shift, seq, hidden);
        let out = self.linear_bias(&cb, &self.dit_model, &x_mod,
            "model.final_layer.linear.weight",
            "model.final_layer.linear.bias",
            seq, hidden, out_ch)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// Per-head LayerNorm with affine weight/bias on [head_dim].
    /// Used by ShapeVAE QK-norm (different from DiT's RMSNorm path).
    fn layer_norm_per_head_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, weight: &Tensor, bias: &Tensor,
        seq: usize, num_heads: usize, head_dim: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let total = seq * num_heads;
        let output_buffer = device.new_buffer((total * head_dim * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.common.layer_norm, total,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, weight);
                gpu_ops::set_tensor_buffer(encoder, 2, bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let n_u32 = total as u32;
                let d_u32 = head_dim as u32;
                let eps: f32 = 1e-6;
                encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([seq, num_heads * head_dim]), DType::F16, self.compute.device().info().id)
    }

    /// One VAE transformer.resblocks block: pre-LN self-attention + pre-LN MLP.
    /// Mirrors hy3dgen's ResidualAttentionBlock.forward exactly.
    /// Input: [seq, hidden]. hidden=1024, num_heads=16, head_dim=64.
    fn vae_resblock(
        &self,
        x: &Tensor, block_idx: usize,
        hidden: usize, num_heads: usize, head_dim: usize, seq: usize,
    ) -> Result<Tensor> {
        let prefix = format!("transformer.resblocks.{}", block_idx);
        let mlp_dim = hidden * 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;

        let cb = self.compute.new_command_buffer();
        let ln1 = self.layer_norm(&cb, &self.vae_model, x,
            &format!("{}.ln_1.weight", prefix),
            &format!("{}.ln_1.bias", prefix),
            seq, hidden, 1e-6)?;
        // Fused QKV: c_qkv [hidden → 3*hidden]. ShapeVAE has qkv_bias=False — no bias here.
        let qkv = self.linear_on(&cb, &self.vae_model, &ln1,
            &format!("{}.attn.c_qkv.weight", prefix),
            seq, hidden, 3 * hidden)?;
        cb.commit();
        cb.wait_until_completed();

        // Host-side split — VAE uses per-head INTERLEAVED Q|K|V layout
        // (hy3dgen's QKVMultiheadAttention: `qkv.view(bs,n_ctx,heads,-1).split(attn_ch,-1)`).
        // qkv[s, :3*hidden] is reshaped to [s, heads, 3*head_dim] where each head's
        // 3*head_dim channels are laid out [Q(head_dim) | K(head_dim) | V(head_dim)].
        let qkv_data: Vec<half::f16> = qkv.to_vec()?;
        let mut q_v = vec![half::f16::ZERO; seq * hidden];
        let mut k_v = vec![half::f16::ZERO; seq * hidden];
        let mut v_v = vec![half::f16::ZERO; seq * hidden];
        for s in 0..seq {
            let base = s * 3 * hidden;
            for h in 0..num_heads {
                let head_base = base + h * 3 * head_dim;
                let dst_base = s * hidden + h * head_dim;
                for d in 0..head_dim {
                    q_v[dst_base + d] = qkv_data[head_base + d];
                    k_v[dst_base + d] = qkv_data[head_base + head_dim + d];
                    v_v[dst_base + d] = qkv_data[head_base + 2 * head_dim + d];
                }
            }
        }
        let q = Tensor::from_slice(&q_v, Shape::from([seq, hidden]), DType::F16, device_id)?;
        let k = Tensor::from_slice(&k_v, Shape::from([seq, hidden]), DType::F16, device_id)?;
        let v_t = Tensor::from_slice(&v_v, Shape::from([seq, hidden]), DType::F16, device_id)?;

        // Per-head LayerNorm (affine) on Q and K.
        let cb = self.compute.new_command_buffer();
        let q_w = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.q_norm.weight", prefix))?;
        let q_b = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.q_norm.bias", prefix))?;
        let k_w = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.k_norm.weight", prefix))?;
        let k_b = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.k_norm.bias", prefix))?;
        let qn = self.layer_norm_per_head_on(&cb, &q, &q_w, &q_b, seq, num_heads, head_dim);
        let kn = self.layer_norm_per_head_on(&cb, &k, &k_w, &k_b, seq, num_heads, head_dim);

        let attn = self.batched_attention(&cb, &qn, &kn, &v_t, seq, seq, num_heads, head_dim, scale)?;
        let attn_proj = self.linear_bias(&cb, &self.vae_model, &attn,
            &format!("{}.attn.c_proj.weight", prefix),
            &format!("{}.attn.c_proj.bias", prefix),
            seq, hidden, hidden)?;
        let post_attn = self.add(&cb, x, &attn_proj);

        let ln2 = self.layer_norm(&cb, &self.vae_model, &post_attn,
            &format!("{}.ln_2.weight", prefix),
            &format!("{}.ln_2.bias", prefix),
            seq, hidden, 1e-6)?;
        let fc = self.linear_bias(&cb, &self.vae_model, &ln2,
            &format!("{}.mlp.c_fc.weight", prefix),
            &format!("{}.mlp.c_fc.bias", prefix),
            seq, hidden, mlp_dim)?;
        // VAE MLP uses standard nn.GELU() — erf-based, NOT tanh.
        let fc_act = self.activation(&cb, &self.kernels.gelu_exact, &fc);
        let proj = self.linear_bias(&cb, &self.vae_model, &fc_act,
            &format!("{}.mlp.c_proj.weight", prefix),
            &format!("{}.mlp.c_proj.bias", prefix),
            seq, mlp_dim, hidden)?;
        let out = self.add(&cb, &post_attn, &proj);
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// Fourier positional embedder (matches hy3dgen's `FourierEmbedder`).
    /// Input: queries [n_q, 3]. Output: [n_q, 3 + 8*3*2] = [n_q, 51].
    /// Frequencies = 2^[0..8] (logspace=True, include_pi=False per VAE config).
    fn fourier_embed(&self, queries: &[f32], n_q: usize) -> Result<Tensor> {
        let num_freqs = 8;
        let input_dim = 3;
        let out_dim = input_dim * (num_freqs * 2 + 1); // 51
        let mut emb_f16: Vec<half::f16> = Vec::with_capacity(n_q * out_dim);
        let freqs: Vec<f32> = (0..num_freqs)
            .map(|i| 2.0_f32.powi(i as i32))
            .collect();
        for q in 0..n_q {
            // Identity x, y, z
            for d in 0..input_dim { emb_f16.push(half::f16::from_f32(queries[q * input_dim + d])); }
            // sin(f * x), iterating freqs slowest? Per hy3dgen:
            //   embed = (x[..., None] * frequencies).view(..., -1)   # [..., dim*num_freqs]
            //   The flatten goes dim slowest, freqs fastest:
            //   embed[d * num_freqs + f] = x[d] * freqs[f]
            // Then cat([x, sin(embed), cos(embed)], -1).
            let mut embed_raw = Vec::with_capacity(input_dim * num_freqs);
            for d in 0..input_dim {
                for f in 0..num_freqs {
                    embed_raw.push(queries[q * input_dim + d] * freqs[f]);
                }
            }
            for v in &embed_raw { emb_f16.push(half::f16::from_f32(v.sin())); }
            for v in &embed_raw { emb_f16.push(half::f16::from_f32(v.cos())); }
        }
        Tensor::from_slice(&emb_f16, Shape::from([n_q, out_dim]), DType::F16,
            self.compute.device().info().id)
    }

    /// One ResidualCrossAttentionBlock for the geo_decoder.
    /// `x` = query embeddings [n_q, hidden]; `data` = transformer output [n_kv, hidden].
    fn vae_cross_attn_block(
        &self,
        x: &Tensor, data: &Tensor,
        hidden: usize, num_heads: usize, head_dim: usize, n_q: usize, n_kv: usize,
    ) -> Result<Tensor> {
        let prefix = "geo_decoder.cross_attn_decoder";
        let mlp_dim = hidden * 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;

        let cb = self.compute.new_command_buffer();
        let q_pre = self.layer_norm(&cb, &self.vae_model, x,
            &format!("{}.ln_1.weight", prefix), &format!("{}.ln_1.bias", prefix),
            n_q, hidden, 1e-6)?;
        let kv_pre = self.layer_norm(&cb, &self.vae_model, data,
            &format!("{}.ln_2.weight", prefix), &format!("{}.ln_2.bias", prefix),
            n_kv, hidden, 1e-6)?;
        // c_q: Linear(hidden, hidden, bias=False); c_kv: Linear(hidden, 2*hidden, bias=False)
        let q_proj = self.linear_on(&cb, &self.vae_model, &q_pre,
            &format!("{}.attn.c_q.weight", prefix),
            n_q, hidden, hidden)?;
        let kv_proj = self.linear_on(&cb, &self.vae_model, &kv_pre,
            &format!("{}.attn.c_kv.weight", prefix),
            n_kv, hidden, 2 * hidden)?;
        cb.commit();
        cb.wait_until_completed();

        // Host-side split of kv [n_kv, 2*hidden] into K, V using per-head INTERLEAVED layout
        // (same as transformer.resblocks): kv.view(bs, n_kv, heads, 2*head_dim) → split(head_dim, -1).
        let kv_data: Vec<half::f16> = kv_proj.to_vec()?;
        let mut k_v = vec![half::f16::ZERO; n_kv * hidden];
        let mut v_v = vec![half::f16::ZERO; n_kv * hidden];
        for s in 0..n_kv {
            let base = s * 2 * hidden;
            for h in 0..num_heads {
                let head_base = base + h * 2 * head_dim;
                let dst_base = s * hidden + h * head_dim;
                for d in 0..head_dim {
                    k_v[dst_base + d] = kv_data[head_base + d];
                    v_v[dst_base + d] = kv_data[head_base + head_dim + d];
                }
            }
        }
        let k = Tensor::from_slice(&k_v, Shape::from([n_kv, hidden]), DType::F16, device_id)?;
        let v_t = Tensor::from_slice(&v_v, Shape::from([n_kv, hidden]), DType::F16, device_id)?;

        // Per-head LayerNorm on Q and K.
        let cb = self.compute.new_command_buffer();
        let q_w = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.q_norm.weight", prefix))?;
        let q_b = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.q_norm.bias", prefix))?;
        let k_w = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.k_norm.weight", prefix))?;
        let k_b = self.weight_f16(&self.vae_model, &format!("{}.attn.attention.k_norm.bias", prefix))?;
        let qn = self.layer_norm_per_head_on(&cb, &q_proj, &q_w, &q_b, n_q, num_heads, head_dim);
        let kn = self.layer_norm_per_head_on(&cb, &k, &k_w, &k_b, n_kv, num_heads, head_dim);

        // Cross attention.
        let attn = self.batched_attention(&cb, &qn, &kn, &v_t,
            n_q, n_kv, num_heads, head_dim, scale)?;
        let attn_proj = self.linear_bias(&cb, &self.vae_model, &attn,
            &format!("{}.attn.c_proj.weight", prefix),
            &format!("{}.attn.c_proj.bias", prefix),
            n_q, hidden, hidden)?;
        let post_attn = self.add(&cb, x, &attn_proj);

        let ln3 = self.layer_norm(&cb, &self.vae_model, &post_attn,
            &format!("{}.ln_3.weight", prefix),
            &format!("{}.ln_3.bias", prefix),
            n_q, hidden, 1e-6)?;
        let fc = self.linear_bias(&cb, &self.vae_model, &ln3,
            &format!("{}.mlp.c_fc.weight", prefix),
            &format!("{}.mlp.c_fc.bias", prefix),
            n_q, hidden, mlp_dim)?;
        let fc_act = self.activation(&cb, &self.kernels.gelu_exact, &fc);
        let proj = self.linear_bias(&cb, &self.vae_model, &fc_act,
            &format!("{}.mlp.c_proj.weight", prefix),
            &format!("{}.mlp.c_proj.bias", prefix),
            n_q, mlp_dim, hidden)?;
        let out = self.add(&cb, &post_attn, &proj);
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// Run the full geo_decoder: fourier embed → query_proj → cross_attn_decoder
    /// → ln_post → output_proj. Returns SDF scalar per query [n_q, 1].
    fn vae_geo_decode(
        &self,
        queries: &[f32], n_q: usize,
        latents: &Tensor, n_kv: usize, hidden: usize,
        num_heads: usize, head_dim: usize,
    ) -> Result<Tensor> {
        let fe = self.fourier_embed(queries, n_q)?;
        if let Ok(dir) = std::env::var("HY3D_DUMP_DIR") {
            let data: Vec<half::f16> = fe.to_vec()?;
            let mut out = Vec::with_capacity(data.len() * 4);
            for v in &data { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
            std::fs::write(format!("{}/04_geo_fourier.f32", dir), &out).ok();
        }
        let cb = self.compute.new_command_buffer();
        let qe = self.linear_bias(&cb, &self.vae_model, &fe,
            "geo_decoder.query_proj.weight", "geo_decoder.query_proj.bias",
            n_q, 3 * (8 * 2 + 1), hidden)?;
        cb.commit();
        cb.wait_until_completed();

        if let Ok(dir) = std::env::var("HY3D_DUMP_DIR") {
            let data: Vec<half::f16> = qe.to_vec()?;
            let mut out = Vec::with_capacity(data.len() * 4);
            for v in &data { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
            std::fs::write(format!("{}/04_geo_query_emb.f32", dir), &out).ok();
        }

        let attn_out = self.vae_cross_attn_block(&qe, latents, hidden, num_heads, head_dim, n_q, n_kv)?;

        let cb = self.compute.new_command_buffer();
        let post = self.layer_norm(&cb, &self.vae_model, &attn_out,
            "geo_decoder.ln_post.weight", "geo_decoder.ln_post.bias",
            n_q, hidden, 1e-5)?;
        let sdf = self.linear_bias(&cb, &self.vae_model, &post,
            "geo_decoder.output_proj.weight",
            "geo_decoder.output_proj.bias",
            n_q, hidden, 1)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(sdf)
    }

    /// Volume decode: sample SDF on a `resolution³` grid in [-bbox, bbox]³, return
    /// flat z-y-x ordered SDF values for direct consumption by `marching_cubes`.
    fn vae_volume_decode(
        &self,
        latents: &Tensor, n_kv: usize, hidden: usize,
        num_heads: usize, head_dim: usize,
        resolution: usize, bbox: f32, chunk: usize,
    ) -> Result<Vec<f32>> {
        let total = resolution * resolution * resolution;
        let mut sdf = vec![0.0f32; total];
        let step = 2.0 * bbox / (resolution as f32 - 1.0);

        let mut q_flat: Vec<f32> = Vec::with_capacity(chunk * 3);
        let mut produced = 0_usize;

        let mut idx = 0_usize;
        for iz in 0..resolution {
            let z = -bbox + iz as f32 * step;
            for iy in 0..resolution {
                let y = -bbox + iy as f32 * step;
                for ix in 0..resolution {
                    let x = -bbox + ix as f32 * step;
                    q_flat.extend_from_slice(&[x, y, z]);
                    idx += 1;
                    if q_flat.len() / 3 == chunk || idx == total {
                        let n_q = q_flat.len() / 3;
                        let s = self.vae_geo_decode(&q_flat, n_q,
                            latents, n_kv, hidden, num_heads, head_dim)?;
                        let sd: Vec<half::f16> = s.to_vec()?;
                        for (i, v) in sd.iter().enumerate() {
                            sdf[produced + i] = v.to_f32();
                        }
                        produced += n_q;
                        q_flat.clear();
                    }
                }
            }
        }
        Ok(sdf)
    }

    /// End-to-end VAE decode → mesh: runs the full VAE (post_kl + 16 transformer
    /// resblocks + volume_decode + marching_cubes) on a latent and returns
    /// a triangle mesh. Public entry for the full pipeline once flow_matching is wired.
    pub fn vae_latent_to_mesh(
        &self,
        latent: &Tensor,
        resolution: usize,
    ) -> Result<super::instantmesh::MeshOutput> {
        let hidden = 1024_usize;
        let num_heads = 16_usize;
        let head_dim = 64_usize;
        let seq = 3072_usize;
        let device_id = self.compute.device().info().id;
        // Allow caller-provided latent in either [seq, latent_ch] (already correct)
        // or [latent_ch, seq] orientation — normalise to [seq, 64] in fp16.
        let l_f16: Vec<half::f16> = latent.to_vec()?;
        let latent_in = Tensor::from_slice(&l_f16,
            Shape::from([seq, 64]), DType::F16, device_id)?;

        let cb = self.compute.new_command_buffer();
        let pk = self.linear_bias(&cb, &self.vae_model, &latent_in,
            "post_kl.weight", "post_kl.bias",
            seq, 64, hidden)?;
        cb.commit();
        cb.wait_until_completed();

        let mut x = pk;
        for i in 0..16 {
            x = self.vae_resblock(&x, i, hidden, num_heads, head_dim, seq)?;
        }

        let sdf = self.vae_volume_decode(
            &x, seq, hidden, num_heads, head_dim,
            resolution, 1.01, 16384)?;
        let mn = sdf.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = sdf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pos = sdf.iter().filter(|&&v| v > 0.0).count();
        let mean = sdf.iter().sum::<f32>() / sdf.len() as f32;
        println!("  SDF stats: min={:.4} max={:.4} mean={:.4} positive={}/{}",
            mn, mx, mean, pos, sdf.len());
        // Hunyuan3D outputs occupancy logits (positive = inside, negative = outside,
        // mc_level ≈ -1/512). The existing marching_cubes interprets `value < 0`
        // as inside — invert sign so the iso-surface lines up.
        let inv: Vec<f32> = sdf.iter().map(|v| -v).collect();
        Ok(super::instantmesh::marching_cubes(&inv, resolution))
    }

    /// Verify-only entrypoint: reads `HY3D_VAE_INPUT` (= `03_vae_input.f32`),
    /// runs `post_kl` + 16 transformer.resblocks, dumps each output.
    pub fn vae_verify_transformer(&self) -> Result<()> {
        let hidden = 1024_usize;
        let num_heads = 16_usize;
        let head_dim = 64_usize;
        let seq = 3072_usize;
        let latent_ch = 64_usize;
        let device_id = self.compute.device().info().id;

        let path = std::env::var("HY3D_VAE_INPUT")
            .map_err(|_| crate::core::Error::internal("HY3D_VAE_INPUT not set"))?;
        let bytes = std::fs::read(&path)
            .map_err(|e| crate::core::Error::internal(&format!("read {path}: {e}")))?;
        let mut latent = Vec::with_capacity(bytes.len() / 4);
        for c in bytes.chunks_exact(4) {
            latent.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        // The dump is shape [1, 3072, 64] = 196,608 f32 elements.
        assert_eq!(latent.len(), seq * latent_ch, "expected 1*3072*64 latent");
        let latent_f16: Vec<half::f16> = latent.iter().map(|&v| half::f16::from_f32(v)).collect();
        let latent_tensor = Tensor::from_slice(&latent_f16,
            Shape::from([seq, latent_ch]), DType::F16, device_id)?;

        let dump_dir = std::env::var("HY3D_DUMP_DIR").ok();
        let dump_t = |t: &Tensor, name: &str| -> Result<()> {
            if let Some(ref dir) = dump_dir {
                let data: Vec<half::f16> = t.to_vec()?;
                let mut out = Vec::with_capacity(data.len() * 4);
                for v in &data { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
                std::fs::write(format!("{}/{}.f32", dir, name), &out).ok();
            }
            Ok(())
        };

        // post_kl: Linear(64 → 1024)
        let cb = self.compute.new_command_buffer();
        let pk = self.linear_bias(&cb, &self.vae_model, &latent_tensor,
            "post_kl.weight", "post_kl.bias",
            seq, latent_ch, hidden)?;
        cb.commit();
        cb.wait_until_completed();
        dump_t(&pk, "03_vae_post_kl")?;

        let mut x = pk;
        for i in 0..16 {
            x = self.vae_resblock(&x, i, hidden, num_heads, head_dim, seq)?;
            dump_t(&x, &format!("03_vae_resblock{:02}", i))?;
        }

        // Geo decoder: read fixed query set saved by the Python oracle.
        let q_path = std::env::var("HY3D_GEO_QUERIES")
            .map_err(|_| crate::core::Error::internal("HY3D_GEO_QUERIES not set"))?;
        let qbytes = std::fs::read(&q_path)
            .map_err(|e| crate::core::Error::internal(&format!("read {q_path}: {e}")))?;
        let mut queries: Vec<f32> = Vec::with_capacity(qbytes.len() / 4);
        for c in qbytes.chunks_exact(4) {
            queries.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        let n_q = queries.len() / 3;
        let sdf = self.vae_geo_decode(&queries, n_q, &x, seq, hidden, num_heads, head_dim)?;
        dump_t(&sdf, "04_geo_sdf")?;
        Ok(())
    }

    /// One DiT step using the verified flux blocks: vec from time, cond_in/latent_in
    /// projections, 16 doubles + 32 singles + final_layer. Returns noise prediction
    /// [num_latent, latent_channels].
    fn dit_step_v2(
        &self,
        latent: &Tensor,
        cond_features: &Tensor, num_cond_tokens: usize,
        t: f32,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let hidden = cfg.dit_hidden;
        let num_heads = cfg.dit_num_heads;
        let head_dim = cfg.dit_head_dim;
        let img_seq = cfg.num_latent_tokens;
        let cond_in_dim = cfg.dino_hidden;
        let device_id = self.compute.device().info().id;

        let vec_tensor = self.dit_time_embed(t, None)?;

        let cb = self.compute.new_command_buffer();
        let img = self.linear_bias(&cb, &self.dit_model, latent,
            "model.latent_in.weight", "model.latent_in.bias",
            img_seq, cfg.latent_channels, hidden)?;
        let txt = self.linear_bias(&cb, &self.dit_model, cond_features,
            "model.cond_in.weight", "model.cond_in.bias",
            num_cond_tokens, cond_in_dim, hidden)?;
        cb.commit();
        cb.wait_until_completed();

        let mut img = img;
        let mut txt = txt;
        for i in 0..cfg.dit_double_blocks {
            let (img_after, txt_after) = self.dit_double_block_flux(
                &img, &txt, &vec_tensor, i, hidden, num_heads, head_dim, img_seq, num_cond_tokens)?;
            img = img_after;
            txt = txt_after;
        }

        let mut joint = Tensor::cat(&[txt, img], 0)?;
        let total_seq = num_cond_tokens + img_seq;
        for i in 0..cfg.dit_single_blocks {
            joint = self.dit_single_block_flux(
                &joint, &vec_tensor, i, hidden, num_heads, head_dim, total_seq)?;
        }

        let joint_data: Vec<half::f16> = joint.to_vec()?;
        let latent_only: Vec<half::f16> = joint_data[num_cond_tokens * hidden..].to_vec();
        let latent_tensor = Tensor::from_slice(&latent_only,
            Shape::from([img_seq, hidden]), DType::F16, device_id)?;

        self.dit_final_layer_flux(&latent_tensor, &vec_tensor,
            hidden, cfg.latent_channels, img_seq)
    }

    /// Flow-matching Euler ODE with CFG, built on the verified DiT blocks.
    /// Replaces the legacy fictional `flow_matching_loop`.
    fn flow_matching_loop_v2(
        &self,
        cond_features: &Tensor, num_cond_tokens: usize,
        seed: u64,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let device_id = self.compute.device().info().id;
        let numel = cfg.num_latent_tokens * cfg.latent_channels;

        // Init noise.
        let x_data = deterministic_randn(numel, seed);
        let x_f16: Vec<half::f16> = x_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut x = Tensor::from_slice(&x_f16,
            Shape::from([cfg.num_latent_tokens, cfg.latent_channels]),
            DType::F16, device_id)?;

        // Time schedule: linspace(1, 0, steps+1) Euler.
        let t_seq: Vec<f32> = (0..=cfg.flow_steps)
            .map(|i| 1.0 - i as f32 / cfg.flow_steps as f32)
            .collect();

        // Uncond features = zeros, same shape as cond_features.
        let zeros: Vec<half::f16> = vec![half::f16::ZERO; num_cond_tokens * cfg.dino_hidden];
        let null_features = Tensor::from_slice(&zeros,
            Shape::from([num_cond_tokens, cfg.dino_hidden]), DType::F16, device_id)?;

        for step in 0..cfg.flow_steps {
            let t = t_seq[step];
            let dt = t_seq[step + 1] - t;
            println!("    [flow] step {}/{}: t={:.3}", step + 1, cfg.flow_steps, t);

            let v_cond = self.dit_step_v2(&x, cond_features, num_cond_tokens, t)?;
            let v_uncond = self.dit_step_v2(&x, &null_features, num_cond_tokens, t)?;

            // CFG: v = v_uncond + cfg_strength * (v_cond - v_uncond)
            let cb = self.compute.new_command_buffer();
            let diff = self.elementwise_binary(&cb, &self.kernels.sub, &v_cond, &v_uncond);
            let scaled_diff = self.scale_tensor(&cb, &self.kernels.scale, &diff, cfg.cfg_strength);
            let v = self.add(&cb, &v_uncond, &scaled_diff);
            // Euler step: x = x + v * dt
            let v_dt = self.scale_tensor(&cb, &self.kernels.scale, &v, dt);
            x = self.add(&cb, &x, &v_dt);
            cb.commit();
            cb.wait_until_completed();
        }
        Ok(x)
    }

    /// Verify-only entrypoint: read injected `02_dit_t_sinusoid.f32` and run
    /// it through `time_in.in_layer → silu → time_in.out_layer`. Dumps the
    /// resulting vec for comparison against captured `02_dit_vec.f32`.
    pub fn dit_verify_time_embed(&self) -> Result<()> {
        let hidden = self.config.dit_hidden;
        let path = std::env::var("HY3D_DIT_T_SINUSOID")
            .map_err(|_| crate::core::Error::internal("HY3D_DIT_T_SINUSOID not set"))?;
        let bytes = std::fs::read(&path)
            .map_err(|e| crate::core::Error::internal(&format!("read {path}: {e}")))?;
        let mut t_full = Vec::with_capacity(bytes.len() / 4);
        for c in bytes.chunks_exact(4) {
            t_full.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        // batch 0 of [2, 256]
        let t_b0: Vec<f32> = t_full[..256].to_vec();
        let vec = self.dit_time_embed(0.0, Some(&t_b0))?;
        if let Ok(dir) = std::env::var("HY3D_DUMP_DIR") {
            let data: Vec<half::f16> = vec.to_vec()?;
            let mut out = Vec::with_capacity(data.len() * 4);
            for v in &data { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
            std::fs::write(format!("{}/02_dit_vec_rust.f32", dir), &out).ok();
            println!("  wrote 02_dit_vec_rust.f32 ({} f32 elements, expected {})",
                data.len(), hidden);
        }
        Ok(())
    }

    /// Verify-only entrypoint for the DiT path.
    /// Reads boundary inputs from disk (HF-captured tensors) and runs through
    /// the first double_block, dumping `02_dit_double00` for cosine comparison.
    /// Boundary file paths come from env vars:
    ///   HY3D_DIT_LATENT_INPUT, HY3D_DIT_T_SINUSOID, HY3D_DIT_COND_INPUT, HY3D_DIT_VEC
    ///   HY3D_DUMP_DIR for output
    pub fn dit_verify_double_block_step1(&self) -> Result<()> {
        let cfg = &self.config;
        let hidden = cfg.dit_hidden;
        let num_heads = cfg.dit_num_heads;
        let head_dim = cfg.dit_head_dim;
        let device_id = self.compute.device().info().id;
        let img_seq = cfg.num_latent_tokens; // 3072
        let cond_in_dim = cfg.dino_hidden;  // 1536

        let read_f32 = |env: &str| -> Result<Vec<f32>> {
            let path = std::env::var(env).map_err(|_| {
                crate::core::Error::internal(&format!("{env} not set"))
            })?;
            let bytes = std::fs::read(&path).map_err(|e| {
                crate::core::Error::internal(&format!("read {path}: {e}"))
            })?;
            let mut v = Vec::with_capacity(bytes.len() / 4);
            for c in bytes.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(v)
        };

        // HF tensors are batched [2, ...] (CFG cond + uncond). Use batch 0 (cond).
        let latent_in_full = read_f32("HY3D_DIT_LATENT_INPUT")?; // [2, 3072, 64]
        let cond_in_full   = read_f32("HY3D_DIT_COND_INPUT")?;   // [2, 1370, 1536]
        let vec_full       = read_f32("HY3D_DIT_VEC")?;          // [2, 1024]
        let t_sin_full     = read_f32("HY3D_DIT_T_SINUSOID")?;   // [2, 256] (optional sanity)
        let _ = t_sin_full;

        let txt_seq = cond_in_full.len() / 2 / cond_in_dim; // expect 1370

        // HY3D_DIT_BATCH=0 → cond pass; HY3D_DIT_BATCH=1 → uncond pass.
        let batch: usize = std::env::var("HY3D_DIT_BATCH")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let lat_off = batch * img_seq * cfg.latent_channels;
        let cond_off = batch * txt_seq * cond_in_dim;
        let vec_off = batch * hidden;

        let latent_b0: Vec<half::f16> = latent_in_full[lat_off..lat_off + img_seq * cfg.latent_channels]
            .iter().map(|&v| half::f16::from_f32(v)).collect();
        let cond_b0: Vec<half::f16> = cond_in_full[cond_off..cond_off + txt_seq * cond_in_dim]
            .iter().map(|&v| half::f16::from_f32(v)).collect();
        let vec_b0: Vec<half::f16> = vec_full[vec_off..vec_off + hidden]
            .iter().map(|&v| half::f16::from_f32(v)).collect();

        let latent_in_tensor = Tensor::from_slice(&latent_b0,
            Shape::from([img_seq, cfg.latent_channels]), DType::F16, device_id)?;
        let cond_in_tensor = Tensor::from_slice(&cond_b0,
            Shape::from([txt_seq, cond_in_dim]), DType::F16, device_id)?;
        let vec_tensor = Tensor::from_slice(&vec_b0,
            Shape::from([1, hidden]), DType::F16, device_id)?;

        // Project: latent_in (64 → 1024), cond_in (1536 → 1024). NO time MLP — vec is injected.
        let cb = self.compute.new_command_buffer();
        let img = self.linear_bias(&cb, &self.dit_model, &latent_in_tensor,
            "model.latent_in.weight", "model.latent_in.bias",
            img_seq, cfg.latent_channels, hidden)?;
        let txt = self.linear_bias(&cb, &self.dit_model, &cond_in_tensor,
            "model.cond_in.weight", "model.cond_in.bias",
            txt_seq, cond_in_dim, hidden)?;
        cb.commit();
        cb.wait_until_completed();

        let dump_dir = std::env::var("HY3D_DUMP_DIR").ok();
        let dump_tensor = |t: &Tensor, name: &str| -> Result<()> {
            if let Some(ref dir) = dump_dir {
                let data: Vec<half::f16> = t.to_vec()?;
                let mut out = Vec::with_capacity(data.len() * 4);
                for v in &data { out.extend_from_slice(&v.to_f32().to_le_bytes()); }
                std::fs::write(format!("{}/{}.f32", dir, name), &out).ok();
            }
            Ok(())
        };

        // 16 double_blocks.
        let mut img = img;
        let mut txt = txt;
        for i in 0..cfg.dit_double_blocks {
            let (img_after, txt_after) = self.dit_double_block_flux(
                &img, &txt, &vec_tensor, i, hidden, num_heads, head_dim, img_seq, txt_seq)?;
            img = img_after;
            txt = txt_after;
            dump_tensor(&img, &format!("02_dit_double{:02}", i))?;
        }

        // Concat (cond, latent) — txt first per hy3dgen `latent = torch.cat((cond, latent), 1)`.
        let mut joint = Tensor::cat(&[txt, img], 0)?;
        let total_seq = txt_seq + img_seq;

        // 32 single_blocks.
        for i in 0..cfg.dit_single_blocks {
            joint = self.dit_single_block_flux(
                &joint, &vec_tensor, i, hidden, num_heads, head_dim, total_seq)?;
            dump_tensor(&joint, &format!("02_dit_single{:02}", i))?;
        }

        // Slice the latent-only portion (after the cond tokens).
        let joint_data: Vec<half::f16> = joint.to_vec()?;
        let latent_only: Vec<half::f16> = joint_data[txt_seq * hidden..].to_vec();
        let latent_tensor = Tensor::from_slice(&latent_only,
            Shape::from([img_seq, hidden]), DType::F16, device_id)?;

        let final_out = self.dit_final_layer_flux(
            &latent_tensor, &vec_tensor,
            hidden, cfg.latent_channels, img_seq)?;
        dump_tensor(&final_out, "02_dit_final")?;
        Ok(())
    }

}

// ==================== Utility ====================

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
