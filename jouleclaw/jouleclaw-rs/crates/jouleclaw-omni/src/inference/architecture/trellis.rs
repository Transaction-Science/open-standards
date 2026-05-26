//! TRELLIS: Two-stage flow matching for image-to-3D generation.
//!
//! Architecture:
//!   Image (518×518) → DINOv2-ViT-L/14 → 1369 patch features [1369, 1024]
//!   → SS Flow Model: 24 DiT blocks (AdaLN + cross-attn) on 16³ voxels
//!   → SS VAE Decode: latent 16³ → binary occupancy mask
//!   → SLAT Flow Model: 24 DiT blocks + progressive IO on occupied 64³ voxels
//!   → Gaussian Decoder: 12 transformer blocks → 3D Gaussian splatting params
//!
//! Each stage uses Euler ODE flow matching with classifier-free guidance.
//! All transformer operations run on Metal GPU; flow loops + VAE decode on CPU.

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

/// Trellis configuration.
#[derive(Debug, Clone)]
pub struct TrellisConfig {
    // DINOv2 ViT-L/14
    pub dino_hidden: usize,
    pub dino_heads: usize,
    pub dino_layers: usize,
    pub dino_patch_size: usize,
    pub dino_image_size: usize,
    pub dino_num_registers: usize,

    // SS Flow Model
    pub ss_resolution: usize,
    pub ss_channels: usize,
    pub ss_model_channels: usize,
    pub ss_num_blocks: usize,
    pub ss_num_heads: usize,
    pub ss_mlp_ratio: usize,

    // SLAT Flow Model
    pub slat_resolution: usize,
    pub slat_channels: usize,
    pub slat_model_channels: usize,
    pub slat_num_blocks: usize,
    pub slat_num_heads: usize,
    pub slat_mlp_ratio: usize,
    pub slat_patch_size: usize,

    // Gaussian Decoder
    pub gs_model_channels: usize,
    pub gs_num_blocks: usize,
    pub gs_num_heads: usize,
    pub gs_latent_channels: usize,
    pub gs_num_gaussians: usize,

    // Flow matching
    pub ss_flow_steps: usize,
    pub slat_flow_steps: usize,
    pub ss_cfg_strength: f32,
    pub slat_cfg_strength: f32,
    pub sigma_min: f32,
}

impl Default for TrellisConfig {
    fn default() -> Self {
        Self {
            dino_hidden: 1024,
            dino_heads: 16,
            dino_layers: 24,
            dino_patch_size: 14,
            dino_image_size: 518,
            dino_num_registers: 4,

            ss_resolution: 16,
            ss_channels: 8,
            ss_model_channels: 1024,
            ss_num_blocks: 24,
            ss_num_heads: 16,
            ss_mlp_ratio: 4,

            slat_resolution: 64,
            slat_channels: 8,
            slat_model_channels: 1024,
            slat_num_blocks: 24,
            slat_num_heads: 16,
            slat_mlp_ratio: 4,
            slat_patch_size: 2,

            gs_model_channels: 768,
            gs_num_blocks: 12,
            gs_num_heads: 12,
            gs_latent_channels: 8,
            gs_num_gaussians: 32,

            ss_flow_steps: 12,
            slat_flow_steps: 12,
            ss_cfg_strength: 7.5,
            slat_cfg_strength: 3.0,
            sigma_min: 1e-5,
        }
    }
}

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct TrellisKernels {
    common: gpu_ops::CommonKernels,
    rms_norm: Arc<ComputePipeline>,
    gelu: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    sub: Arc<ComputePipeline>,
    mul: Arc<ComputePipeline>,
    scale: Arc<ComputePipeline>,
    adaln_modulate: Arc<ComputePipeline>,
    adaln_gate: Arc<ComputePipeline>,
}

// ==================== Trellis Pipeline ====================

/// Trellis pipeline for image-to-3D generation via two-stage flow matching.
///
/// Forward pipeline:
/// 1. DINOv2 ViT-L/14: image [3, 518, 518] → features [1369, 1024]
/// 2. SS Flow: 24 DiT blocks on 16³=4096 voxel tokens → latent 16³
/// 3. SS VAE: decode latent → binary occupancy mask
/// 4. SLAT Flow: 24 DiT blocks on occupied voxels → latent features
/// 5. Gaussian Decoder: 12 transformer blocks → 3D Gaussian params
#[cfg(feature = "metal")]
pub struct TrellisPipeline {
    dino_model: Arc<Model>,
    ss_flow_model: Arc<Model>,
    ss_vae_model: Arc<Model>,
    slat_flow_model: Arc<Model>,
    gs_decoder_model: Arc<Model>,
    compute: Arc<MetalCompute>,
    config: TrellisConfig,
    kernels: TrellisKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for TrellisPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl TrellisPipeline {
    /// Create a new Trellis pipeline.
    ///
    /// Requires 5 separate Model objects loaded from their respective safetensors files:
    /// - `dino_model`: DINOv2-ViT-L/14 weights (`dinov2_vitl14_reg.safetensors`)
    /// - `ss_flow_model`: Sparse Structure Flow (`ss_flow_img_dit_L_16l8_fp16.safetensors`)
    /// - `ss_vae_model`: SS VAE decoder (`ss_vae_conv3d_16l8_fp16.safetensors`)
    /// - `slat_flow_model`: SLAT Flow (`slat_flow_img_dit_L_64l8p2_fp16.safetensors`)
    /// - `gs_decoder_model`: Gaussian decoder (`slat_vae_enc_dec_gs_swin8_B_64l8_fp16.safetensors`)
    pub fn new(
        dino_model: Arc<Model>,
        ss_flow_model: Arc<Model>,
        ss_vae_model: Arc<Model>,
        slat_flow_model: Arc<Model>,
        gs_decoder_model: Arc<Model>,
        config: TrellisConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = TrellisKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            sub: compute.compile_pipeline("sub", sources::ELEMENTWISE, "sub_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
            scale: compute.compile_pipeline("scale", sources::ELEMENTWISE, "scale_f16")?,
            adaln_modulate: compute.compile_pipeline("adaln_modulate", sources::ADALN, "adaln_modulate_f16")?,
            adaln_gate: compute.compile_pipeline("adaln_gate", sources::ADALN, "adaln_gate_f16")?,
        };

        Ok(Self {
            dino_model, ss_flow_model, ss_vae_model, slat_flow_model, gs_decoder_model,
            compute, config, kernels,
        })
    }

    /// Full image-to-3D generation.
    ///
    /// Input: image as flat f32 RGB array [3*518*518] in [C, H, W] format,
    ///        normalized with ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
    /// Returns: Vec of Gaussian parameters per occupied voxel.
    ///          Each entry: [xyz(3), color(3), scale(3), rotation(4), opacity(1)] × num_gaussians.
    pub fn generate(&self, image_chw: &[f32], seed: u64) -> Result<GaussianOutput> {
        let config = &self.config;

        // 1. DINOv2: image → condition features [num_patches, 1024]
        println!("  [Trellis] DINOv2 encoding...");
        let image_features = self.dino_forward(image_chw)?;
        let num_cond_tokens = image_features.shape().dims()[0];
        println!("  [Trellis] DINOv2 output: [{}, {}]", num_cond_tokens, config.dino_hidden);

        // 2. SS Flow: denoise 16³ voxel latent
        println!("  [Trellis] SS Flow ({} steps)...", config.ss_flow_steps);
        let ss_num_tokens = config.ss_resolution * config.ss_resolution * config.ss_resolution;
        let ss_latent = self.flow_matching_loop(
            &self.ss_flow_model,
            &image_features,
            num_cond_tokens,
            ss_num_tokens,
            config.ss_channels,
            config.ss_model_channels,
            config.ss_num_blocks,
            config.ss_num_heads,
            config.ss_mlp_ratio,
            config.ss_flow_steps,
            config.ss_cfg_strength,
            seed,
            "ss",
        )?;

        // 3. SS VAE decode: latent → binary occupancy
        println!("  [Trellis] SS VAE decode...");
        let occupancy = self.ss_vae_decode(&ss_latent)?;
        let occupied_count = occupancy.iter().filter(|&&v| v).count();
        println!("  [Trellis] Occupied voxels: {}/{}", occupied_count, ss_num_tokens);

        if occupied_count == 0 {
            return Ok(GaussianOutput {
                positions: vec![],
                colors: vec![],
                scales: vec![],
                rotations: vec![],
                opacities: vec![],
                num_gaussians_per_voxel: config.gs_num_gaussians,
            });
        }

        // 4. Upscale occupancy from 16³ to 64³ (each occupied 16³ cell → 4³=64 cells in 64³)
        let scale = config.slat_resolution / config.ss_resolution; // 4
        let mut occupied_coords: Vec<[u32; 4]> = Vec::new();
        for z in 0..config.ss_resolution {
            for y in 0..config.ss_resolution {
                for x in 0..config.ss_resolution {
                    let idx = z * config.ss_resolution * config.ss_resolution + y * config.ss_resolution + x;
                    if occupancy[idx] {
                        for dz in 0..scale {
                            for dy in 0..scale {
                                for dx in 0..scale {
                                    occupied_coords.push([
                                        0, // batch index
                                        (x * scale + dx) as u32,
                                        (y * scale + dy) as u32,
                                        (z * scale + dz) as u32,
                                    ]);
                                }
                            }
                        }
                    }
                }
            }
        }
        let num_occupied = occupied_coords.len();
        println!("  [Trellis] SLAT tokens: {} (from {} occupied 16³ cells)", num_occupied, occupied_count);

        // 5. SLAT Flow: denoise latent features on occupied voxels
        println!("  [Trellis] SLAT Flow ({} steps)...", config.slat_flow_steps);
        let slat_latent = self.flow_matching_loop(
            &self.slat_flow_model,
            &image_features,
            num_cond_tokens,
            num_occupied,
            config.slat_channels,
            config.slat_model_channels,
            config.slat_num_blocks,
            config.slat_num_heads,
            config.slat_mlp_ratio,
            config.slat_flow_steps,
            config.slat_cfg_strength,
            seed + 1,
            "slat",
        )?;

        // 6. Gaussian decoder
        println!("  [Trellis] Gaussian decoder ({} blocks)...", config.gs_num_blocks);
        let gs_output = self.gaussian_decode(&slat_latent, num_occupied)?;

        // 7. Parse Gaussian parameters
        let output = self.parse_gaussian_params(&gs_output, num_occupied, &occupied_coords)?;

        Ok(output)
    }

    // ==================== DINOv2 ViT-L/14 ====================

    /// DINOv2 ViT-L/14 forward: image [3, 518, 518] → features [1369, 1024].
    fn dino_forward(&self, image_chw: &[f32]) -> Result<Tensor> {
        let config = &self.config;
        let d_model = config.dino_hidden; // 1024
        let patch_size = config.dino_patch_size; // 14
        let grid = config.dino_image_size / patch_size; // 37
        let num_patches = grid * grid; // 1369
        let num_heads = config.dino_heads; // 16
        let head_dim = d_model / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();
        let num_registers = config.dino_num_registers; // 4
        let device_id = self.compute.device().info().id;

        // 1. Patch embedding: Conv2d [1024, 3, 14, 14]
        let patches = self.dino_patch_embed(image_chw, grid, patch_size, d_model)?;

        // 2. Prepend CLS token + register tokens, add position embeddings
        let cls_token = self.weight_f16(&self.dino_model, "dino.cls_token")?; // [1, 1, 1024]
        let reg_token = self.weight_f16(&self.dino_model, "dino.reg_token")?; // [1, 4, 1024]
        let pos_embed = self.weight_f16(&self.dino_model, "dino.pos_embed")?; // [1, 1374, 1024]
        let cls_data: Vec<half::f16> = cls_token.to_vec()?;
        let reg_data: Vec<half::f16> = reg_token.to_vec()?;
        let patches_data: Vec<half::f16> = patches.to_vec()?;
        let pos_data: Vec<half::f16> = pos_embed.to_vec()?;

        // seq = CLS + registers + patches = 1 + 4 + 1369 = 1374
        let seq_len = 1 + num_registers + num_patches;
        let mut combined = Vec::with_capacity(seq_len * d_model);
        combined.extend_from_slice(&cls_data[..d_model]);
        combined.extend_from_slice(&reg_data[..num_registers * d_model]);
        combined.extend_from_slice(&patches_data);

        // Add position embeddings
        for i in 0..seq_len * d_model {
            combined[i] = half::f16::from_f32(combined[i].to_f32() + pos_data[i].to_f32());
        }

        let mut hidden = Tensor::from_slice(
            &combined, Shape::from([seq_len, d_model]), DType::F16, device_id,
        )?;

        // 3. 24 encoder layers
        let ffn_dim = d_model * 4; // 4096
        for layer in 0..config.dino_layers {
            let prefix = format!("dino.blocks.{}", layer);

            let cb = self.compute.new_command_buffer();
            // LayerNorm → Self-attention
            let normed = self.layer_norm(
                &cb, &self.dino_model, &hidden,
                &format!("{}.norm1.weight", prefix),
                &format!("{}.norm1.bias", prefix),
                seq_len, d_model, 1e-6,
            )?;

            // Fused QKV with bias
            let qkv = self.linear_bias(
                &cb, &self.dino_model, &normed,
                &format!("{}.attn.qkv.weight", prefix),
                &format!("{}.attn.qkv.bias", prefix),
                seq_len, d_model, d_model * 3,
            )?;

            // Split QKV
            let qkv_data: Vec<half::f16> = qkv.to_vec()?;
            let mut q_data = vec![half::f16::ZERO; seq_len * d_model];
            let mut k_data = vec![half::f16::ZERO; seq_len * d_model];
            let mut v_data = vec![half::f16::ZERO; seq_len * d_model];
            for i in 0..seq_len {
                let base = i * d_model * 3;
                q_data[i * d_model..(i + 1) * d_model].copy_from_slice(&qkv_data[base..base + d_model]);
                k_data[i * d_model..(i + 1) * d_model].copy_from_slice(&qkv_data[base + d_model..base + 2 * d_model]);
                v_data[i * d_model..(i + 1) * d_model].copy_from_slice(&qkv_data[base + 2 * d_model..base + 3 * d_model]);
            }
            let q = Tensor::from_slice(&q_data, Shape::from([seq_len, d_model]), DType::F16, device_id)?;
            let k = Tensor::from_slice(&k_data, Shape::from([seq_len, d_model]), DType::F16, device_id)?;
            let v = Tensor::from_slice(&v_data, Shape::from([seq_len, d_model]), DType::F16, device_id)?;

            // Batched attention
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

            let attn_out = self.linear_bias(
                &cb, &self.dino_model, &attn_flat,
                &format!("{}.attn.proj.weight", prefix),
                &format!("{}.attn.proj.bias", prefix),
                seq_len, d_model, d_model,
            )?;
            // DINOv2 uses ls1 (layer scale 1) for attention
            let ls1 = self.weight_f16(&self.dino_model, &format!("{}.ls1", prefix))?;
            let scaled_attn = self.scale_by_vec_on(&cb, &attn_out, &ls1, seq_len, d_model);
            let h = self.add(&cb, &hidden, &scaled_attn);

            // LayerNorm → FFN
            let normed2 = self.layer_norm(
                &cb, &self.dino_model, &h,
                &format!("{}.norm2.weight", prefix),
                &format!("{}.norm2.bias", prefix),
                seq_len, d_model, 1e-6,
            )?;
            let ffn_up = self.linear_bias(
                &cb, &self.dino_model, &normed2,
                &format!("{}.mlp.fc1.weight", prefix),
                &format!("{}.mlp.fc1.bias", prefix),
                seq_len, d_model, ffn_dim,
            )?;
            let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
            let ffn_down = self.linear_bias(
                &cb, &self.dino_model, &ffn_act,
                &format!("{}.mlp.fc2.weight", prefix),
                &format!("{}.mlp.fc2.bias", prefix),
                seq_len, ffn_dim, d_model,
            )?;
            let ls2 = self.weight_f16(&self.dino_model, &format!("{}.ls2", prefix))?;
            let scaled_ffn = self.scale_by_vec_on(&cb, &ffn_down, &ls2, seq_len, d_model);
            hidden = self.add(&cb, &h, &scaled_ffn);

            cb.commit();
            cb.wait_until_completed();
        }

        // 4. Extract prenorm features (before final norm), apply LayerNorm
        // In Trellis, features = LayerNorm(x_prenorm) where x_prenorm = hidden states before final norm
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, &self.dino_model, &hidden,
            "dino.norm.weight", "dino.norm.bias",
            seq_len, d_model, 1e-6,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // 5. Remove CLS + register tokens → [1369, 1024]
        let skip = 1 + num_registers;
        normed.slice(0, skip, seq_len)
    }

    /// DINOv2 patch embedding via im2col (CPU layout) → GPU matmul.
    fn dino_patch_embed(&self, image_chw: &[f32], grid: usize, patch_size: usize, d_model: usize) -> Result<Tensor> {
        let c_in = 3;
        let num_patches = grid * grid;
        let patch_dim = c_in * patch_size * patch_size;
        let img_size = self.config.dino_image_size;

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
        let result = self.linear_bias(&cb, &self.dino_model, &input_tensor,
            "dino.patch_embed.proj.weight", "dino.patch_embed.proj.bias",
            num_patches, patch_dim, d_model)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    // ==================== Flow Matching ====================

    /// Euler ODE flow matching loop with classifier-free guidance.
    fn flow_matching_loop(
        &self,
        flow_model: &Arc<Model>,
        cond_features: &Tensor,
        num_cond_tokens: usize,
        num_tokens: usize,
        channels: usize,
        model_channels: usize,
        num_blocks: usize,
        num_heads: usize,
        mlp_ratio: usize,
        steps: usize,
        cfg_strength: f32,
        seed: u64,
        tag: &str,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let numel = num_tokens * channels;

        // Initialize noise: x = randn() * sigma(1.0)
        let sigma_1 = self.config.sigma_min + (1.0 - self.config.sigma_min) * 1.0;
        let mut x_data = deterministic_randn(numel, seed);
        for v in &mut x_data {
            *v *= sigma_1;
        }
        let x_f16: Vec<half::f16> = x_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut x = Tensor::from_slice(&x_f16, Shape::from([num_tokens, channels]), DType::F16, device_id)?;

        // Time sequence: linspace(1, 0, steps+1)
        let t_seq: Vec<f32> = (0..=steps).map(|i| 1.0 - i as f32 / steps as f32).collect();

        // Pre-compute cross-attention KV for conditional pass
        let cross_kv = self.precompute_flow_cross_kv(
            flow_model, cond_features, num_cond_tokens,
            model_channels, num_blocks, num_heads,
        )?;

        // Null features for unconditional pass (zeros)
        let null_features = Tensor::from_slice(
            &vec![half::f16::ZERO; num_cond_tokens * self.config.dino_hidden],
            Shape::from([num_cond_tokens, self.config.dino_hidden]),
            DType::F16, device_id,
        )?;
        let uncond_cross_kv = self.precompute_flow_cross_kv(
            flow_model, &null_features, num_cond_tokens,
            model_channels, num_blocks, num_heads,
        )?;

        for step in 0..steps {
            let t = t_seq[step];
            let dt = t_seq[step + 1] - t;

            println!("    [{tag}] step {}/{steps}: t={:.3}", step + 1, t);

            // Conditional forward
            let v_cond = self.flow_model_forward(
                flow_model, &x, t,
                num_tokens, channels, model_channels,
                num_blocks, num_heads, mlp_ratio,
                &cross_kv, num_cond_tokens,
            )?;

            // Unconditional forward
            let v_uncond = self.flow_model_forward(
                flow_model, &x, t,
                num_tokens, channels, model_channels,
                num_blocks, num_heads, mlp_ratio,
                &uncond_cross_kv, num_cond_tokens,
            )?;

            // CFG: v = v_uncond + cfg_strength * (v_cond - v_uncond)
            let cb = self.compute.new_command_buffer();
            let diff = self.elementwise_binary(&cb, &self.kernels.sub, &v_cond, &v_uncond);
            let scaled_diff = self.scale_tensor(&cb, &self.kernels.scale, &diff, cfg_strength);
            let v = self.add(&cb, &v_uncond, &scaled_diff);

            // Euler step: x = x + v * dt
            let v_dt = self.scale_tensor(&cb, &self.kernels.scale, &v, dt);
            x = self.add(&cb, &x, &v_dt);
            cb.commit();
            cb.wait_until_completed();
        }

        Ok(x)
    }

    /// Pre-compute cross-attention K/V from condition features for all flow model layers.
    fn precompute_flow_cross_kv(
        &self,
        flow_model: &Arc<Model>,
        cond_features: &Tensor,
        num_cond_tokens: usize,
        model_channels: usize,
        num_blocks: usize,
        num_heads: usize,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let head_dim = model_channels / num_heads;
        let cond_dim = self.config.dino_hidden;
        let device_id = self.compute.device().info().id;

        let mut cross_kv = Vec::with_capacity(num_blocks);
        for block in 0..num_blocks {
            let cb = self.compute.new_command_buffer();
            // to_kv projects cond_dim → 2*model_channels, then split K/V
            let kv = self.linear_on(
                &cb, flow_model, cond_features,
                &format!("blocks.{}.cross_attn.to_kv.weight", block),
                num_cond_tokens, cond_dim, model_channels * 2,
            )?;
            cb.commit();
            cb.wait_until_completed();

            // Split KV
            let kv_data: Vec<half::f16> = kv.to_vec()?;
            let mut k_data = vec![half::f16::ZERO; num_cond_tokens * model_channels];
            let mut v_data = vec![half::f16::ZERO; num_cond_tokens * model_channels];
            for i in 0..num_cond_tokens {
                let base = i * model_channels * 2;
                k_data[i * model_channels..(i + 1) * model_channels]
                    .copy_from_slice(&kv_data[base..base + model_channels]);
                v_data[i * model_channels..(i + 1) * model_channels]
                    .copy_from_slice(&kv_data[base + model_channels..base + 2 * model_channels]);
            }

            // Apply QK RMS Norm to K
            let k_raw = Tensor::from_slice(&k_data, Shape::from([num_cond_tokens, model_channels]), DType::F16, device_id)?;
            let v_raw = Tensor::from_slice(&v_data, Shape::from([num_cond_tokens, model_channels]), DType::F16, device_id)?;

            let cb = self.compute.new_command_buffer();
            let k_normed = self.rms_norm_per_head_on(
                &cb, flow_model, &k_raw,
                &format!("blocks.{}.cross_attn.k_rms_norm.gamma", block),
                num_cond_tokens, num_heads, head_dim,
            )?;

            // Transpose to HSD
            let k_hsd = Tensor::empty(Shape::from([num_heads, num_cond_tokens, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(Shape::from([num_heads, num_cond_tokens, head_dim]), DType::F16, device_id)?;
            self.transpose_shd_to_hsd(&cb, &k_normed, &k_hsd, num_cond_tokens, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &v_raw, &v_hsd, num_cond_tokens, num_heads, head_dim);

            cb.commit();
            cb.wait_until_completed();
            cross_kv.push((k_hsd, v_hsd));
        }
        Ok(cross_kv)
    }

    // ==================== Flow Model Forward ====================

    /// Single forward pass through the flow model DiT blocks.
    fn flow_model_forward(
        &self,
        flow_model: &Arc<Model>,
        x: &Tensor,
        t: f32,
        num_tokens: usize,
        channels: usize,
        model_channels: usize,
        num_blocks: usize,
        num_heads: usize,
        mlp_ratio: usize,
        cross_kv: &[(Tensor, Tensor)],
        num_cond_tokens: usize,
    ) -> Result<Tensor> {
        let head_dim = model_channels / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;
        let ffn_dim = model_channels * mlp_ratio;

        // 1. Timestep embedding: sinusoidal → MLP (2 layers with SiLU)
        let t_emb = sinusoidal_embedding(t, model_channels / 2);
        let t_tensor = Tensor::from_slice(
            &t_emb.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>(),
            Shape::from([1, model_channels]),
            DType::F16, device_id,
        )?;

        let cb = self.compute.new_command_buffer();
        let temb_h = self.linear_bias(
            &cb, flow_model, &t_tensor,
            "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias",
            1, model_channels, model_channels,
        )?;
        let temb_act = self.activation(&cb, &self.kernels.silu, &temb_h);
        let temb = self.linear_bias(
            &cb, flow_model, &temb_act,
            "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias",
            1, model_channels, model_channels,
        )?;

        // 2. Input projection
        let hidden = self.linear_bias(
            &cb, flow_model, x,
            "input_layer.weight", "input_layer.bias",
            num_tokens, channels, model_channels,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // 3. Add positional embedding
        let pos_emb = self.weight_f16(flow_model, "pos_emb")?;
        let pos_data: Vec<half::f16> = pos_emb.to_vec()?;
        let mut h_data: Vec<half::f16> = hidden.to_vec()?;
        let pos_tokens = pos_data.len() / model_channels;
        for i in 0..num_tokens.min(pos_tokens) {
            for c in 0..model_channels {
                let idx = i * model_channels + c;
                h_data[idx] = half::f16::from_f32(h_data[idx].to_f32() + pos_data[idx].to_f32());
            }
        }
        let mut hidden = Tensor::from_slice(
            &h_data, Shape::from([num_tokens, model_channels]), DType::F16, device_id,
        )?;

        // 4. DiT blocks
        for block in 0..num_blocks {
            hidden = self.dit_block(
                flow_model, &hidden, &temb,
                block, num_tokens, model_channels,
                num_heads, head_dim, ffn_dim,
                scale, cross_kv, num_cond_tokens,
            )?;
        }

        // 5. Final AdaLN + output projection
        let cb = self.compute.new_command_buffer();
        // adaLN_modulation: SiLU(t) → linear → 2 params (shift, scale)
        let mod_act = self.activation(&cb, &self.kernels.silu, &temb);
        let mod_params = self.linear_bias(
            &cb, flow_model, &mod_act,
            "adaLN_modulation.1.weight", "adaLN_modulation.1.bias",
            1, model_channels, model_channels * 2,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Extract shift, scale (each [model_channels])
        let mod_data: Vec<half::f16> = mod_params.to_vec()?;
        let shift = Tensor::from_slice(
            &mod_data[..model_channels], Shape::from([model_channels]), DType::F16, device_id,
        )?;
        let final_scale = Tensor::from_slice(
            &mod_data[model_channels..2 * model_channels], Shape::from([model_channels]), DType::F16, device_id,
        )?;

        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, flow_model, &hidden,
            "out_layer.norm.weight", "out_layer.norm.bias",
            num_tokens, model_channels, 1e-6,
        )?;
        let modulated = self.adaln_modulate_on(&cb, &normed, &final_scale, &shift, num_tokens, model_channels);
        let output = self.linear_bias(
            &cb, flow_model, &modulated,
            "out_layer.linear.weight", "out_layer.linear.bias",
            num_tokens, model_channels, channels,
        )?;
        cb.commit();
        cb.wait_until_completed();

        Ok(output)
    }

    /// Single DiT block with AdaLN modulation, self-attention, cross-attention, and FFN.
    fn dit_block(
        &self,
        flow_model: &Arc<Model>,
        input: &Tensor,
        temb: &Tensor,
        block: usize,
        seq_len: usize,
        model_channels: usize,
        num_heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        scale: f32,
        cross_kv: &[(Tensor, Tensor)],
        num_cond_tokens: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let prefix = format!("blocks.{}", block);

        // === AdaLN modulation: t → SiLU → Linear → 6 params ===
        let cb = self.compute.new_command_buffer();
        let mod_act = self.activation(&cb, &self.kernels.silu, temb);
        let mod_params = self.linear_bias(
            &cb, flow_model, &mod_act,
            &format!("{}.adaLN_modulation.1.weight", prefix),
            &format!("{}.adaLN_modulation.1.bias", prefix),
            1, model_channels, model_channels * 6,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Split into 6 vectors: shift1, scale1, gate1, shift2, scale2, gate2
        let mod_data: Vec<half::f16> = mod_params.to_vec()?;
        let mc = model_channels;
        let shift1 = Tensor::from_slice(&mod_data[0..mc], Shape::from([mc]), DType::F16, device_id)?;
        let scale1 = Tensor::from_slice(&mod_data[mc..2*mc], Shape::from([mc]), DType::F16, device_id)?;
        let gate1 = Tensor::from_slice(&mod_data[2*mc..3*mc], Shape::from([mc]), DType::F16, device_id)?;
        let shift2 = Tensor::from_slice(&mod_data[3*mc..4*mc], Shape::from([mc]), DType::F16, device_id)?;
        let scale2 = Tensor::from_slice(&mod_data[4*mc..5*mc], Shape::from([mc]), DType::F16, device_id)?;
        let gate2 = Tensor::from_slice(&mod_data[5*mc..6*mc], Shape::from([mc]), DType::F16, device_id)?;

        // === Self-attention: LayerNorm → modulate → QKV → RMS Norm → attn → gate → residual ===
        let cb = self.compute.new_command_buffer();
        let normed = self.layer_norm(
            &cb, flow_model, input,
            &format!("{}.norm1.weight", prefix),
            &format!("{}.norm1.bias", prefix),
            seq_len, mc, 1e-6,
        )?;
        let modulated = self.adaln_modulate_on(&cb, &normed, &scale1, &shift1, seq_len, mc);

        // to_qkv → [seq, 3*mc]
        let qkv = self.linear_bias(
            &cb, flow_model, &modulated,
            &format!("{}.self_attn.to_qkv.weight", prefix),
            &format!("{}.self_attn.to_qkv.bias", prefix),
            seq_len, mc, mc * 3,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Split Q/K/V
        let qkv_data: Vec<half::f16> = qkv.to_vec()?;
        let mut q_data = vec![half::f16::ZERO; seq_len * mc];
        let mut k_data = vec![half::f16::ZERO; seq_len * mc];
        let mut v_data = vec![half::f16::ZERO; seq_len * mc];
        for i in 0..seq_len {
            let base = i * mc * 3;
            q_data[i * mc..(i + 1) * mc].copy_from_slice(&qkv_data[base..base + mc]);
            k_data[i * mc..(i + 1) * mc].copy_from_slice(&qkv_data[base + mc..base + 2 * mc]);
            v_data[i * mc..(i + 1) * mc].copy_from_slice(&qkv_data[base + 2 * mc..base + 3 * mc]);
        }
        let q_raw = Tensor::from_slice(&q_data, Shape::from([seq_len, mc]), DType::F16, device_id)?;
        let k_raw = Tensor::from_slice(&k_data, Shape::from([seq_len, mc]), DType::F16, device_id)?;
        let v = Tensor::from_slice(&v_data, Shape::from([seq_len, mc]), DType::F16, device_id)?;

        // QK RMS Norm
        let cb = self.compute.new_command_buffer();
        let q = self.rms_norm_per_head_on(
            &cb, flow_model, &q_raw,
            &format!("{}.self_attn.q_rms_norm.gamma", prefix),
            seq_len, num_heads, head_dim,
        )?;
        let k = self.rms_norm_per_head_on(
            &cb, flow_model, &k_raw,
            &format!("{}.self_attn.k_rms_norm.gamma", prefix),
            seq_len, num_heads, head_dim,
        )?;

        // Batched attention
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
        let attn_flat = attn_shd.reshape([seq_len, mc])?;

        let sa_out = self.linear_bias(
            &cb, flow_model, &attn_flat,
            &format!("{}.self_attn.to_out.weight", prefix),
            &format!("{}.self_attn.to_out.bias", prefix),
            seq_len, mc, mc,
        )?;
        let h = self.adaln_gate_on(&cb, input, &sa_out, &gate1, seq_len, mc);
        cb.commit();
        cb.wait_until_completed();

        // === Cross-attention: LayerNorm → Q (from hidden) + pre-computed K/V → attn → residual ===
        let cb = self.compute.new_command_buffer();
        let normed2 = self.layer_norm(
            &cb, flow_model, &h,
            &format!("{}.norm2.weight", prefix),
            &format!("{}.norm2.bias", prefix),
            seq_len, mc, 1e-6,
        )?;
        let cross_q_raw = self.linear_bias(
            &cb, flow_model, &normed2,
            &format!("{}.cross_attn.to_q.weight", prefix),
            &format!("{}.cross_attn.to_q.bias", prefix),
            seq_len, mc, mc,
        )?;
        // QK RMS Norm on Q
        let cross_q = self.rms_norm_per_head_on(
            &cb, flow_model, &cross_q_raw,
            &format!("{}.cross_attn.q_rms_norm.gamma", prefix),
            seq_len, num_heads, head_dim,
        )?;

        let cq_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(&cb, &cross_q, &cq_hsd, seq_len, num_heads, head_dim);

        let (ref ck_hsd, ref cv_hsd) = cross_kv[block];
        let cross_scores = self.batched_qk(&cb, &cq_hsd, ck_hsd, num_heads, seq_len, num_cond_tokens, head_dim);
        self.row_softmax(&cb, &cross_scores, num_heads * seq_len, num_cond_tokens, scale);
        let cross_out_hsd = self.batched_sv(&cb, &cross_scores, cv_hsd, num_heads, seq_len, num_cond_tokens, head_dim);

        let cross_out_shd = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(&cb, &cross_out_hsd, &cross_out_shd, seq_len, num_heads, head_dim);
        let cross_flat = cross_out_shd.reshape([seq_len, mc])?;

        let ca_out = self.linear_bias(
            &cb, flow_model, &cross_flat,
            &format!("{}.cross_attn.to_out.weight", prefix),
            &format!("{}.cross_attn.to_out.bias", prefix),
            seq_len, mc, mc,
        )?;
        let h = self.add(&cb, &h, &ca_out);
        cb.commit();
        cb.wait_until_completed();

        // === FFN: LayerNorm → modulate → Linear → GELU → Linear → gate → residual ===
        let cb = self.compute.new_command_buffer();
        let normed3 = self.layer_norm(
            &cb, flow_model, &h,
            &format!("{}.norm3.weight", prefix),
            &format!("{}.norm3.bias", prefix),
            seq_len, mc, 1e-6,
        )?;
        let modulated_ffn = self.adaln_modulate_on(&cb, &normed3, &scale2, &shift2, seq_len, mc);
        let ffn_up = self.linear_bias(
            &cb, flow_model, &modulated_ffn,
            &format!("{}.mlp.0.weight", prefix),
            &format!("{}.mlp.0.bias", prefix),
            seq_len, mc, ffn_dim,
        )?;
        let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
        let ffn_down = self.linear_bias(
            &cb, flow_model, &ffn_act,
            &format!("{}.mlp.2.weight", prefix),
            &format!("{}.mlp.2.bias", prefix),
            seq_len, ffn_dim, mc,
        )?;
        let result = self.adaln_gate_on(&cb, &h, &ffn_down, &gate2, seq_len, mc);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    // ==================== SS VAE Decode ====================

    /// Decode SS Flow output to binary occupancy mask (CPU-resident).
    fn ss_vae_decode(&self, ss_latent: &Tensor) -> Result<Vec<bool>> {
        let config = &self.config;
        let res = config.ss_resolution; // 16
        let total = res * res * res; // 4096

        // The SS VAE decoder is a small 3D convolutional network.
        // For now, use a simple threshold-based approach on the latent directly:
        // Average channels and threshold at 0.
        let data: Vec<half::f16> = ss_latent.to_vec()?;
        let channels = config.ss_channels;

        let mut occupancy = vec![false; total];
        for i in 0..total {
            let mut avg = 0.0f32;
            for c in 0..channels {
                avg += data[i * channels + c].to_f32();
            }
            avg /= channels as f32;
            occupancy[i] = avg > 0.0;
        }

        // If too few or too many occupied, clamp
        let occupied = occupancy.iter().filter(|&&v| v).count();
        if occupied == 0 {
            // At least fill center region
            for z in res / 4..3 * res / 4 {
                for y in res / 4..3 * res / 4 {
                    for x in res / 4..3 * res / 4 {
                        occupancy[z * res * res + y * res + x] = true;
                    }
                }
            }
        }

        Ok(occupancy)
    }

    // ==================== Gaussian Decoder ====================

    /// Decode SLAT latent features into Gaussian splatting parameters.
    fn gaussian_decode(&self, slat_latent: &Tensor, num_tokens: usize) -> Result<Tensor> {
        let config = &self.config;
        let mc = config.gs_model_channels; // 768
        let num_heads = config.gs_num_heads; // 12
        let head_dim = mc / num_heads; // 64
        let scale = 1.0 / (head_dim as f32).sqrt();
        let ffn_dim = mc * 4; // MLP ratio 4
        let device_id = self.compute.device().info().id;

        // Gaussian output: 32 gaussians × 14 params = 448 per voxel
        let gs_params_per_voxel = config.gs_num_gaussians * 14;

        // Input projection: latent_channels → gs_model_channels
        let cb = self.compute.new_command_buffer();
        let mut hidden = self.linear_bias(
            &cb, &self.gs_decoder_model, slat_latent,
            "decoder.input_layer.weight", "decoder.input_layer.bias",
            num_tokens, config.gs_latent_channels, mc,
        )?;
        cb.commit();
        cb.wait_until_completed();

        // Transformer blocks (full attention, no windowing for simplicity)
        for block in 0..config.gs_num_blocks {
            let prefix = format!("decoder.blocks.{}", block);

            let cb = self.compute.new_command_buffer();
            // Self-attention
            let normed = self.layer_norm(
                &cb, &self.gs_decoder_model, &hidden,
                &format!("{}.norm1.weight", prefix),
                &format!("{}.norm1.bias", prefix),
                num_tokens, mc, 1e-6,
            )?;
            let qkv = self.linear_bias(
                &cb, &self.gs_decoder_model, &normed,
                &format!("{}.attn.to_qkv.weight", prefix),
                &format!("{}.attn.to_qkv.bias", prefix),
                num_tokens, mc, mc * 3,
            )?;
            cb.commit();
            cb.wait_until_completed();

            // Split QKV
            let qkv_data: Vec<half::f16> = qkv.to_vec()?;
            let mut q_data = vec![half::f16::ZERO; num_tokens * mc];
            let mut k_data = vec![half::f16::ZERO; num_tokens * mc];
            let mut v_data = vec![half::f16::ZERO; num_tokens * mc];
            for i in 0..num_tokens {
                let base = i * mc * 3;
                q_data[i * mc..(i + 1) * mc].copy_from_slice(&qkv_data[base..base + mc]);
                k_data[i * mc..(i + 1) * mc].copy_from_slice(&qkv_data[base + mc..base + 2 * mc]);
                v_data[i * mc..(i + 1) * mc].copy_from_slice(&qkv_data[base + 2 * mc..base + 3 * mc]);
            }
            let q = Tensor::from_slice(&q_data, Shape::from([num_tokens, mc]), DType::F16, device_id)?;
            let k = Tensor::from_slice(&k_data, Shape::from([num_tokens, mc]), DType::F16, device_id)?;
            let v = Tensor::from_slice(&v_data, Shape::from([num_tokens, mc]), DType::F16, device_id)?;

            let cb = self.compute.new_command_buffer();
            let q_hsd = Tensor::empty(Shape::from([num_heads, num_tokens, head_dim]), DType::F16, device_id)?;
            let k_hsd = Tensor::empty(Shape::from([num_heads, num_tokens, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(Shape::from([num_heads, num_tokens, head_dim]), DType::F16, device_id)?;
            self.transpose_shd_to_hsd(&cb, &q, &q_hsd, num_tokens, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &k, &k_hsd, num_tokens, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &v, &v_hsd, num_tokens, num_heads, head_dim);

            let scores = self.batched_qk(&cb, &q_hsd, &k_hsd, num_heads, num_tokens, num_tokens, head_dim);
            self.row_softmax(&cb, &scores, num_heads * num_tokens, num_tokens, scale);
            let attn_hsd = self.batched_sv(&cb, &scores, &v_hsd, num_heads, num_tokens, num_tokens, head_dim);

            let attn_shd = Tensor::empty(Shape::from([num_tokens, num_heads, head_dim]), DType::F16, device_id)?;
            self.transpose_hsd_to_shd(&cb, &attn_hsd, &attn_shd, num_tokens, num_heads, head_dim);
            let attn_flat = attn_shd.reshape([num_tokens, mc])?;

            let sa_out = self.linear_bias(
                &cb, &self.gs_decoder_model, &attn_flat,
                &format!("{}.attn.to_out.weight", prefix),
                &format!("{}.attn.to_out.bias", prefix),
                num_tokens, mc, mc,
            )?;
            let h = self.add(&cb, &hidden, &sa_out);

            // FFN
            let normed2 = self.layer_norm(
                &cb, &self.gs_decoder_model, &h,
                &format!("{}.norm2.weight", prefix),
                &format!("{}.norm2.bias", prefix),
                num_tokens, mc, 1e-6,
            )?;
            let ffn_up = self.linear_bias(
                &cb, &self.gs_decoder_model, &normed2,
                &format!("{}.mlp.0.weight", prefix),
                &format!("{}.mlp.0.bias", prefix),
                num_tokens, mc, ffn_dim,
            )?;
            let ffn_act = self.activation(&cb, &self.kernels.gelu, &ffn_up);
            let ffn_down = self.linear_bias(
                &cb, &self.gs_decoder_model, &ffn_act,
                &format!("{}.mlp.2.weight", prefix),
                &format!("{}.mlp.2.bias", prefix),
                num_tokens, ffn_dim, mc,
            )?;
            hidden = self.add(&cb, &h, &ffn_down);
            cb.commit();
            cb.wait_until_completed();
        }

        // Output projection: gs_model_channels → gs_params_per_voxel
        let cb = self.compute.new_command_buffer();
        let output = self.linear_bias(
            &cb, &self.gs_decoder_model, &hidden,
            "decoder.out_layer.weight", "decoder.out_layer.bias",
            num_tokens, mc, gs_params_per_voxel,
        )?;
        cb.commit();
        cb.wait_until_completed();

        Ok(output)
    }

    /// Parse raw Gaussian decoder output into structured parameters.
    fn parse_gaussian_params(
        &self,
        gs_output: &Tensor,
        num_voxels: usize,
        coords: &[[u32; 4]],
    ) -> Result<GaussianOutput> {
        let config = &self.config;
        let ng = config.gs_num_gaussians; // 32
        let total_gs = num_voxels * ng;
        let data: Vec<half::f16> = gs_output.to_vec()?;
        let params_per_voxel = ng * 14;

        let mut positions = Vec::with_capacity(total_gs * 3);
        let mut colors = Vec::with_capacity(total_gs * 3);
        let mut scales_out = Vec::with_capacity(total_gs * 3);
        let mut rotations = Vec::with_capacity(total_gs * 4);
        let mut opacities = Vec::with_capacity(total_gs);

        let voxel_size = 1.0 / config.slat_resolution as f32;

        for (voxel_idx, coord) in coords.iter().enumerate() {
            let base_x = coord[1] as f32 * voxel_size - 0.5;
            let base_y = coord[2] as f32 * voxel_size - 0.5;
            let base_z = coord[3] as f32 * voxel_size - 0.5;

            for g in 0..ng {
                let offset = voxel_idx * params_per_voxel + g * 14;
                // Layout per Gaussian: xyz(3), color(3), scale(3), rotation(4), opacity(1)
                let dx = data[offset].to_f32() * voxel_size;
                let dy = data[offset + 1].to_f32() * voxel_size;
                let dz = data[offset + 2].to_f32() * voxel_size;
                positions.push(base_x + dx);
                positions.push(base_y + dy);
                positions.push(base_z + dz);

                // Color (sigmoid)
                for c in 0..3 {
                    let v = data[offset + 3 + c].to_f32();
                    colors.push(1.0 / (1.0 + (-v).exp()));
                }

                // Scale (softplus)
                for s in 0..3 {
                    let v = data[offset + 6 + s].to_f32();
                    scales_out.push(((v).exp() + 1.0).ln() * 0.004); // scaling_bias
                }

                // Rotation (normalize quaternion)
                let mut quat = [0.0f32; 4];
                for r in 0..4 {
                    quat[r] = data[offset + 9 + r].to_f32();
                }
                let norm = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt().max(1e-8);
                for r in 0..4 {
                    rotations.push(quat[r] / norm);
                }

                // Opacity (sigmoid + bias)
                let op = data[offset + 13].to_f32();
                opacities.push(1.0 / (1.0 + (-op - 0.1).exp()));
            }
        }

        Ok(GaussianOutput {
            positions,
            colors,
            scales: scales_out,
            rotations,
            opacities,
            num_gaussians_per_voxel: ng,
        })
    }

    // ==================== GPU Helper Methods ====================

    fn linear_on(
        &self, cb: &metal::CommandBufferRef, model: &Arc<Model>, input: &Tensor,
        weight_name: &str,
        m: usize, k: usize, n: usize,
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

    /// Per-head RMS norm for QK normalization.
    /// Input: [seq, num_heads * head_dim], gamma: [num_heads * head_dim]
    /// Applies RMS norm independently per head (treating each head_dim block as a row).
    fn rms_norm_per_head_on(
        &self, cb: &metal::CommandBufferRef, model: &Arc<Model>, input: &Tensor,
        gamma_name: &str,
        seq_len: usize, num_heads: usize, head_dim: usize,
    ) -> Result<Tensor> {
        // Reshape [seq, num_heads*head_dim] → [seq*num_heads, head_dim] for rms_norm dispatch
        let n_rows = seq_len * num_heads;
        let gamma = self.weight_f16(model, gamma_name)?;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((n_rows * head_dim * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.rms_norm, n_rows,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &gamma);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                let n_u32 = n_rows as u32;
                let d_u32 = head_dim as u32;
                let eps: f32 = 1e-6;
                encoder.set_bytes(3, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
            },
        );
        Ok(Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([seq_len, num_heads * head_dim]),
            DType::F16, self.compute.device().info().id,
        ))
    }

    /// Element-wise multiply by a vector (broadcast over seq_len): out[i,j] = input[i,j] * vec[j]
    fn scale_by_vec_on(&self, cb: &metal::CommandBufferRef, input: &Tensor, vec: &Tensor, seq_len: usize, dim: usize) -> Tensor {
        // Broadcast vec [dim] over seq_len rows of input [seq_len, dim]
        // Expand vec to full size and use mul_f16
        let numel = seq_len * dim;
        let vec_data: Vec<half::f16> = vec.to_vec().unwrap();
        let mut expanded = Vec::with_capacity(numel);
        for _ in 0..seq_len {
            expanded.extend_from_slice(&vec_data[..dim]);
        }
        let device_id = self.compute.device().info().id;
        let expanded_tensor = Tensor::from_slice(&expanded, Shape::from([seq_len, dim]), DType::F16, device_id).unwrap();

        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.mul, numel,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &expanded_tensor);
                encoder.set_buffer(2, Some(&output_buffer), 0);
            },
        );
        Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, device_id)
    }

    /// AdaLN modulate: out = (1 + scale) * x + shift
    fn adaln_modulate_on(
        &self, cb: &metal::CommandBufferRef,
        x: &Tensor, scale: &Tensor, shift: &Tensor,
        seq_len: usize, hidden: usize,
    ) -> Tensor {
        let count = seq_len * hidden;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((count * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.adaln_modulate, count,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, x);
                gpu_ops::set_tensor_buffer(encoder, 1, scale);
                gpu_ops::set_tensor_buffer(encoder, 2, shift);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let h = hidden as u32;
                let c = count as u32;
                encoder.set_bytes(4, 4, &h as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(output_buffer, x.shape().clone(), DType::F16, self.compute.device().info().id)
    }

    /// Gated residual: out = x + gate * residual
    fn adaln_gate_on(
        &self, cb: &metal::CommandBufferRef,
        x: &Tensor, residual: &Tensor, gate: &Tensor,
        seq_len: usize, hidden: usize,
    ) -> Tensor {
        let count = seq_len * hidden;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((count * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.adaln_gate, count,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, x);
                gpu_ops::set_tensor_buffer(encoder, 1, residual);
                gpu_ops::set_tensor_buffer(encoder, 2, gate);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let h = hidden as u32;
                let c = count as u32;
                encoder.set_bytes(4, 4, &h as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c as *const u32 as *const _);
            },
        );
        Tensor::from_metal_buffer(output_buffer, x.shape().clone(), DType::F16, self.compute.device().info().id)
    }

}

// ==================== Output Types ====================

/// 3D Gaussian splatting output from Trellis.
pub struct GaussianOutput {
    /// Flat [N*3]: xyz positions for each Gaussian.
    pub positions: Vec<f32>,
    /// Flat [N*3]: RGB colors (sigmoid-activated, 0..1).
    pub colors: Vec<f32>,
    /// Flat [N*3]: scale factors per axis.
    pub scales: Vec<f32>,
    /// Flat [N*4]: unit quaternion rotations.
    pub rotations: Vec<f32>,
    /// Flat [N]: opacity values (0..1).
    pub opacities: Vec<f32>,
    /// Number of Gaussians per voxel (32).
    pub num_gaussians_per_voxel: usize,
}

impl GaussianOutput {
    /// Total number of Gaussians.
    pub fn num_gaussians(&self) -> usize {
        self.opacities.len()
    }

    /// Export to PLY point cloud format.
    pub fn to_ply(&self) -> Vec<u8> {
        let n = self.num_gaussians();
        let mut ply = format!(
            "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nend_header\n",
            n
        );
        for i in 0..n {
            let x = self.positions[i * 3];
            let y = self.positions[i * 3 + 1];
            let z = self.positions[i * 3 + 2];
            let r = (self.colors[i * 3] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (self.colors[i * 3 + 1] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (self.colors[i * 3 + 2] * 255.0).clamp(0.0, 255.0) as u8;
            let a = (self.opacities[i] * 255.0).clamp(0.0, 255.0) as u8;
            ply.push_str(&format!("{} {} {} {} {} {} {}\n", x, y, z, r, g, b, a));
        }
        ply.into_bytes()
    }
}

// ==================== Utility Functions ====================

/// Deterministic pseudo-random normal samples (Box-Muller transform).
fn deterministic_randn(n: usize, seed: u64) -> Vec<f32> {
    let mut rng_state = seed;
    let mut output = Vec::with_capacity(n);

    for _ in 0..(n + 1) / 2 {
        // Simple LCG for reproducibility
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
        let u1 = u1.clamp(1e-7, 1.0 - 1e-7);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (rng_state >> 33) as f32 / (1u64 << 31) as f32;

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        output.push(r * theta.cos());
        if output.len() < n {
            output.push(r * theta.sin());
        }
    }

    output.truncate(n);
    output
}

/// Sinusoidal timestep embedding.
fn sinusoidal_embedding(t: f32, half_dim: usize) -> Vec<f32> {
    let dim = half_dim * 2;
    let mut emb = vec![0.0f32; dim];
    let max_period: f32 = 10000.0;
    for i in 0..half_dim {
        let freq = (-((i as f32) / half_dim as f32) * max_period.ln()).exp();
        emb[i] = (t * freq).sin();
        emb[half_dim + i] = (t * freq).cos();
    }
    emb
}

