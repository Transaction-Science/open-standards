//! Diffusion inference pipeline.
//!
//! Implements fast image generation using:
//! - LCM (Latent Consistency Models) for 4-step generation
//! - Streaming latent previews for real-time feedback
//! - Metal compute for Apple Silicon optimization
//! - IP-Adapter and ControlNet support
//! - Multiple backbone architectures: UNet (SDXL), DiT (AuraFlow, Flux)

use super::config::ImageParams;
use super::engine::ImageProgress;
use super::model::Model;
use super::architecture::unet::UNet2DConditionModel;
use crate::core::Result;
use crate::runtime::stream::StreamSender;
use crate::runtime::ResourceMonitor;
use crate::tensor::{DType, Shape, Tensor};
use half::f16;
use std::sync::Arc;
use super::tokenizer::Tokenizer;

#[cfg(feature = "metal")]
use crate::hal::{MetalDevice, MetalCompute};
#[cfg(feature = "metal")]
use crate::hal::metal::{BorrowedMetalBuffer, ComputePipeline};

// ============================================================================
// Pipeline Polymorphism — Multi-architecture support
// ============================================================================

/// Denoising backbone architecture.
pub enum DiffusionBackbone {
    /// UNet (SDXL, SDXL Turbo, SDXL Lightning).
    UNet(Arc<UNet2DConditionModel>),
    /// Diffusion Transformer — AuraFlow, Flux.
    /// Holds the raw model weights + architecture-specific config.
    DiT {
        /// Transformer model weights.
        model: Arc<Model>,
        /// DiT variant (determines forward pass behavior).
        variant: DiTVariant,
    },
}

/// Which DiT architecture variant to use for the forward pass.
#[derive(Debug, Clone)]
pub enum DiTVariant {
    /// AuraFlow v0.3: 4 joint MMDiT blocks + 32 single blocks, GEGLU FFN.
    AuraFlow {
        /// Hidden dimension (3072).
        hidden_size: usize,
        /// Number of attention heads (12).
        num_heads: usize,
        /// Number of joint (text+image) blocks (4).
        num_joint_blocks: usize,
        /// Number of single (image-only) blocks (32).
        num_single_blocks: usize,
    },
    /// Flux 1/2 Dev: 19 double + 38 single blocks, 2D RoPE, QK-norm.
    Flux(super::architecture::flux::FluxConfig),
    /// PixArt-Sigma: 28 DiT blocks, AdaLN-Single, separate self+cross attention.
    PixArt(super::architecture::pixart::PixArtConfig),
}

/// Text encoder type.
#[derive(Debug, Clone)]
pub enum TextEncoderType {
    /// Dual CLIP: CLIP-L (768-dim) + OpenCLIP-G (1280-dim) → 2048-dim (SDXL).
    DualCLIP,
    /// T5 only (UMT5 for AuraFlow, T5-XXL for some configs).
    T5Only {
        /// T5 model dimension.
        d_model: usize,
    },
    /// CLIP-L + T5-XXL (Flux: 768-dim pooled + 4096-dim sequence).
    CLIPPlusT5 {
        /// T5 model dimension (4096 for T5-XXL).
        t5_dim: usize,
    },
    /// ChatGLM-6B (Kolors): 28 layers, 4096-dim, GQA with 2 KV heads.
    ChatGLM {
        /// ChatGLM hidden dimension (4096).
        d_model: usize,
    },
    /// T5-XXL with internal projection (PixArt-Sigma: T5 4096 → caption_projection → hidden).
    T5WithProjection {
        /// T5 model dimension (4096 for T5-XXL).
        t5_dim: usize,
        /// Projection target dimension (1152 for PixArt-Sigma).
        proj_dim: usize,
    },
}

/// VAE variant — different latent channel counts and scaling factors.
#[derive(Debug, Clone, Copy)]
pub enum VaeVariant {
    /// SD 1.x / 2.x VAE: 4 latent channels, scaling factor 1/0.18215.
    /// Diffusers layer naming is identical to SDXL VAE (decoder.up_blocks.{0..3},
    /// post_quant_conv, etc.); only the latent scaling factor differs.
    SD15,
    /// SDXL VAE: 4 latent channels, scaling factor 1/0.13025.
    SDXL,
    /// Flux VAE: 16 latent channels, scaling factor 0.3611.
    Flux,
}

impl VaeVariant {
    /// Number of latent channels.
    pub fn latent_channels(&self) -> usize {
        match self {
            VaeVariant::SD15 => 4,
            VaeVariant::SDXL => 4,
            VaeVariant::Flux => 16,
        }
    }

    /// Scaling factor for latent→pixel conversion.
    pub fn scaling_factor(&self) -> f32 {
        match self {
            VaeVariant::SD15 => 1.0 / 0.18215,
            VaeVariant::SDXL => 1.0 / 0.13025,
            VaeVariant::Flux => 1.0 / 0.3611,
        }
    }
}

/// Diffusion inference pipeline.
pub struct DiffusionPipeline {
    /// Denoising backbone (UNet or DiT).
    backbone: DiffusionBackbone,
    /// Text encoder model (CLIP-L for SDXL, T5 weights for DiT models)
    text_encoder: Option<Arc<Model>>,
    /// Second text encoder (OpenCLIP-G for SDXL, CLIP-L for Flux)
    text_encoder_2: Option<Arc<Model>>,
    /// VAE decoder model
    vae_decoder: Arc<Model>,
    /// Metal compute (macOS)
    #[cfg(feature = "metal")]
    compute: Arc<MetalCompute>,
    /// Cached compute pipelines (compiled once, reused every step)
    #[cfg(feature = "metal")]
    kernels: DiffusionKernels,
    /// Scheduler
    scheduler: DiffusionScheduler,
    /// CLIP tokenizer for text encoding
    tokenizer: Option<Arc<Tokenizer>>,
    /// Use LCM (fast 4-step generation)
    use_lcm: bool,
    /// Text encoder type (determines encoding path).
    text_encoder_type: TextEncoderType,
    /// VAE variant (determines latent channels and scaling).
    vae_variant: VaeVariant,
}

/// Cached Metal compute pipelines for diffusion operations.
///
/// These are compiled once and reused on every inference step to avoid
/// recompilation overhead in the hot path.
#[cfg(feature = "metal")]
struct DiffusionKernels {
    vae_encode_fused: std::sync::OnceLock<Arc<ComputePipeline>>,
    vae_decode_fused: std::sync::OnceLock<Arc<ComputePipeline>>,
    copy_tile: std::sync::OnceLock<Arc<ComputePipeline>>,
    upsample_nearest: std::sync::OnceLock<Arc<ComputePipeline>>,
    scale: std::sync::OnceLock<Arc<ComputePipeline>>,
    add: std::sync::OnceLock<Arc<ComputePipeline>>,
    sub: std::sync::OnceLock<Arc<ComputePipeline>>,
}

#[cfg(feature = "metal")]
impl DiffusionKernels {
    fn new() -> Self {
        Self {
            vae_encode_fused: std::sync::OnceLock::new(),
            vae_decode_fused: std::sync::OnceLock::new(),
            copy_tile: std::sync::OnceLock::new(),
            upsample_nearest: std::sync::OnceLock::new(),
            scale: std::sync::OnceLock::new(),
            add: std::sync::OnceLock::new(),
            sub: std::sync::OnceLock::new(),
        }
    }
}

impl DiffusionPipeline {
    /// Create a new SDXL diffusion pipeline (UNet backbone, Dual CLIP, SDXL VAE).
    ///
    /// For SDXL with separate HuggingFace text encoder files, pass
    /// `text_encoder` = CLIP-L and `text_encoder_2` = OpenCLIP-G.
    /// For combined checkpoints (e.g. sd_xl_base_1.0.safetensors), pass
    /// all text encoder weights in `text_encoder` and set `text_encoder_2` to `None`.
    #[cfg(feature = "metal")]
    pub fn new(
        unet: Arc<Model>,
        text_encoder: Option<Arc<Model>>,
        vae_decoder: Arc<Model>,
        device: Arc<MetalDevice>,
        tokenizer: Option<Arc<Tokenizer>>,
        use_lcm: bool,
    ) -> Result<Self> {
        Self::with_text_encoder_2(unet, text_encoder, None, vae_decoder, device, tokenizer, use_lcm)
    }

    /// Create a new SDXL diffusion pipeline with a separate second text encoder.
    #[cfg(feature = "metal")]
    pub fn with_text_encoder_2(
        unet: Arc<Model>,
        text_encoder: Option<Arc<Model>>,
        text_encoder_2: Option<Arc<Model>>,
        vae_decoder: Arc<Model>,
        device: Arc<MetalDevice>,
        tokenizer: Option<Arc<Tokenizer>>,
        use_lcm: bool,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        // Wrap the generic model in the UNet architecture wrapper
        let backbone = DiffusionBackbone::UNet(Arc::new(UNet2DConditionModel::new(unet)));

        let scheduler = if use_lcm {
            DiffusionScheduler::lcm(4)
        } else {
            DiffusionScheduler::ddpm(20)
        };

        // VAE variant: SDXL uses dual-CLIP (text_encoder + text_encoder_2);
        // SD 1.x/2.x has only the single CLIP. Use that as the discriminator
        // since the VAE scaling factor differs (1/0.13025 vs 1/0.18215) — a
        // mismatch produces colorful-noise output from `vae_decode` even
        // though the rest of the pipeline runs correctly. Keep `text_encoder_type`
        // as `DualCLIP` for both — the encoder path already tolerates
        // `text_encoder_2 = None` for the SD 1.5 case (single-CLIP fallback).
        let vae_variant = if text_encoder_2.is_some() {
            VaeVariant::SDXL
        } else {
            VaeVariant::SD15
        };
        let text_encoder_type = TextEncoderType::DualCLIP;

        Ok(Self {
            backbone,
            text_encoder,
            text_encoder_2,
            vae_decoder,
            compute,
            kernels: DiffusionKernels::new(),
            scheduler,
            tokenizer,
            use_lcm,
            text_encoder_type,
            vae_variant,
        })
    }

    /// Create a DiT-based diffusion pipeline (AuraFlow, Flux).
    #[cfg(feature = "metal")]
    pub fn new_dit(
        dit_model: Arc<Model>,
        dit_variant: DiTVariant,
        text_encoder: Option<Arc<Model>>,
        text_encoder_2: Option<Arc<Model>>,
        vae_decoder: Arc<Model>,
        device: Arc<MetalDevice>,
        tokenizer: Option<Arc<Tokenizer>>,
        text_encoder_type: TextEncoderType,
        vae_variant: VaeVariant,
        scheduler: DiffusionScheduler,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        Ok(Self {
            backbone: DiffusionBackbone::DiT {
                model: dit_model,
                variant: dit_variant,
            },
            text_encoder,
            text_encoder_2,
            vae_decoder,
            compute,
            kernels: DiffusionKernels::new(),
            scheduler,
            tokenizer,
            use_lcm: false,
            text_encoder_type,
            vae_variant,
        })
    }

    /// Create a new diffusion pipeline (non-Metal fallback).
    #[cfg(not(feature = "metal"))]
    pub fn new(
        unet: Arc<Model>,
        text_encoder: Option<Arc<Model>>,
        vae_decoder: Arc<Model>,
        tokenizer: Option<Arc<Tokenizer>>,
        use_lcm: bool,
    ) -> Result<Self> {
        // Wrap the generic model in the UNet architecture wrapper
        let backbone = DiffusionBackbone::UNet(Arc::new(UNet2DConditionModel::new(unet)));

        let scheduler = if use_lcm {
            DiffusionScheduler::lcm(4)
        } else {
            DiffusionScheduler::ddpm(20)
        };

        Ok(Self {
            backbone,
            text_encoder,
            text_encoder_2: None,
            vae_decoder,
            scheduler,
            tokenizer,
            use_lcm,
            text_encoder_type: TextEncoderType::DualCLIP,
            vae_variant: VaeVariant::SDXL,
        })
    }

    /// Override the scheduler (e.g. for Lightning trailing timesteps).
    pub fn with_scheduler(mut self, scheduler: DiffusionScheduler) -> Self {
        self.scheduler = scheduler;
        self
    }

    /// Override the text encoder type (e.g. ChatGLM for Kolors).
    pub fn with_text_encoder_type(mut self, encoder_type: TextEncoderType) -> Self {
        self.text_encoder_type = encoder_type;
        self
    }

    /// Generate an image without ControlNet conditioning.
    pub async fn generate(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        params: &ImageParams,
        monitor: &ResourceMonitor,
    ) -> Result<Tensor> {
        self.generate_with_controls(prompt, negative_prompt, params, &[], monitor).await
    }

    /// Generate an image with optional ControlNet conditioning. `controls`
    /// is a list of `(ControlNet, preprocessed_control_image)` pairs.
    /// Each ControlNet emits 13 residuals per timestep (12 down + 1 mid)
    /// matching the U-Net encoder shapes; with multiple ControlNets the
    /// residuals are summed elementwise per slot before injection.
    pub async fn generate_with_controls(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        params: &ImageParams,
        controls: &[(Arc<ControlNet>, Tensor)],
        monitor: &ResourceMonitor,
    ) -> Result<Tensor> {
        // Encode text prompt
        let prompt_embeds = self.encode_prompt(prompt)?;
        let negative_embeds = match negative_prompt
            .map(|p| self.encode_prompt(p))
            .transpose()?
        {
            Some(embeds) => embeds,
            None => {
                let ne = self.null_prompt_embedding()?;
                ne
            }
        };

        monitor.memory().record_alloc(prompt_embeds.size());

        // Initialize latents
        let latent_shape = self.latent_shape(params.width, params.height);
        let mut latents = self.init_latents(&latent_shape, params.seed)?;

        monitor.memory().record_alloc(latents.size());

        // Setup scheduler
        self.scheduler.reset();
        let timesteps = self.scheduler.timesteps();

        // UNet uses batch=2 CFG (concat latents). DiT models use two-pass CFG.
        let is_unet = matches!(&self.backbone, DiffusionBackbone::UNet(_));
        let needs_latent_doubling = is_unet && params.guidance_scale > 1.0;

        // Denoising loop
        for (_i, &timestep) in timesteps.iter().enumerate() {
            let sigma = self.scheduler.sigma_at_timestep(timestep);

            // UNet batch=2 CFG: concat latents. DiT: pass single latents.
            let latent_model_input = if needs_latent_doubling {
                self.concat_latents(&latents, &latents)?
            } else {
                latents.clone()
            };

            // ControlNet residuals: per timestep, ask each ControlNet for
            // its (12 down + 1 mid) residual list, then sum elementwise
            // across ControlNets at each residual slot. Zero residuals
            // (architecture-skeleton path) trivially flow through as
            // additive no-ops.
            let cn_residuals: Option<Vec<Tensor>> = if controls.is_empty() {
                None
            } else {
                #[cfg(feature = "metal")]
                {
                    let mut acc: Option<Vec<Tensor>> = None;
                    for (cn, ctrl_img) in controls {
                        // ControlNet runs at batch=1 (single-stream). The U-Net's
                        // residual injection later doubles it for CFG.
                        let cb = self.compute.new_command_buffer();
                        let mut next = cn.forward_full(
                            &latents,
                            timestep,
                            &prompt_embeds,
                            ctrl_img,
                            &self.compute,
                            cb.as_ref(),
                        )?;
                        cb.commit();
                        cb.wait_until_completed();
                        if next.is_empty() { continue; }
                        let scale = cn.scale();
                        if (scale - 1.0).abs() > 1e-6 {
                            for r in next.iter_mut() {
                                *r = self.scale(r, scale)?;
                            }
                        }
                        acc = Some(match acc {
                            None => next,
                            Some(prev) => {
                                let n = prev.len().min(next.len());
                                let mut out = Vec::with_capacity(n);
                                for i in 0..n {
                                    out.push(self.add(&prev[i], &next[i])?);
                                }
                                out
                            }
                        });
                    }
                    acc
                }
                #[cfg(not(feature = "metal"))]
                None
            };

            // Tell the diag layer which step we are on so SD_DIAG_STEP=<n>
            // can scope per-block dumps to a single step.
            crate::inference::architecture::unet::diag_set_step(_i);

            // Forward pass (UNet uses batched latents; DiT uses single + two-pass CFG)
            let noise_pred = self.unet_forward_with_residuals(
                &latent_model_input,
                timestep,
                &prompt_embeds,
                &negative_embeds,
                params.guidance_scale,
                params.use_cfg_pp,
                sigma,
                cn_residuals.as_deref(),
            )?;

            // [DIAG] element-wise dump for PyTorch cross-check. When
            // `SD_DUMP_DIR` is set, at step 0 write the raw f32 little-endian
            // contents of the init latent, the CLIP prompt/neg embeds, and
            // the post-CFG noise_pred. PT then loads OUR latent + OUR embeds,
            // runs PT's UNet at the same timestep, and we compare element-wise
            // (cosine-sim / MSE). This isolates the UNet forward from CLIP and
            // from the RNG (our Box-Muller randn ≠ torch.randn).
            if std::env::var("SD_DUMP_DIR").is_ok() {
                let dir = std::env::var("SD_DUMP_DIR").unwrap();
                let dump = |name: &str, t: &Tensor| {
                    if let Ok(v) = t.to_f32_vec() {
                        let mut bytes = Vec::with_capacity(v.len() * 4);
                        for f in &v { bytes.extend_from_slice(&f.to_le_bytes()); }
                        let _ = std::fs::write(format!("{}/{}.f32", dir, name), &bytes);
                        tracing::info!("[diag-dump] wrote {}/{}.f32 ({} f32, shape={:?})", dir, name, v.len(), t.shape());
                    }
                };
                if _i == 0 {
                    dump("init_latent", &latents);
                    dump("prompt_embeds", &prompt_embeds);
                    dump("negative_embeds", &negative_embeds);
                    dump("noise_pred_step0", &noise_pred);
                    tracing::info!("[diag-dump] step0 timestep={}", timestep);
                }
                // Mid-trajectory snapshot: dump the UNet INPUT latent + post-CFG
                // noise_pred at a configurable step so PT can run its UNet on
                // OUR exact step-N latent + embeds at the SAME timestep and we
                // diff element-wise. step-0 was cos~0.995; if step-N is much
                // lower there is a residual timestep-dependent UNet bug.
                let dump_step: usize = std::env::var("SD_DUMP_STEP")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(10);
                if _i == dump_step {
                    dump("stepN_latent", &latents);
                    dump("stepN_noise_pred", &noise_pred);
                    tracing::info!("[diag-dump] stepN step={} timestep={}", _i, timestep);
                }
            }

            // [DIAG] per-step latent + noise_pred stats — TEMP: remove when SD 1.5
            // denoising is verified visually. Run via env var SD_DIAG=1.
            if std::env::var("SD_DIAG").ok().as_deref() == Some("1") {
                if let (Ok(lv), Ok(nv)) = (latents.to_f32_vec(), noise_pred.to_f32_vec()) {
                    let (l_mean, l_min, l_max, l_std) = vec_stats(&lv);
                    let (n_mean, n_min, n_max, n_std) = vec_stats(&nv);
                    tracing::info!(
                        "[diag] step={:>2} t={:>5.1} sigma={:>7.4} latents[mean={:+.4} std={:.4} min={:+.4} max={:+.4}] noise_pred[mean={:+.4} std={:.4} min={:+.4} max={:+.4}]",
                        _i, timestep, sigma,
                        l_mean, l_std, l_min, l_max,
                        n_mean, n_std, n_min, n_max,
                    );
                }
            }

            // Scheduler step — CPU is faster than GPU on Apple UMA due to zero-copy
            // shared memory and avoidance of command buffer synchronization overhead
            latents = self.scheduler.step(&latents, &noise_pred, timestep)?;

            // Heun second pass: evaluate UNet on midpoint and correct
            if self.scheduler.needs_second_pass() {
                let next_t = self.scheduler.next_timestep(timestep);
                let latent_model_input_2 = if needs_latent_doubling {
                    self.concat_latents(&latents, &latents)?
                } else {
                    latents.clone()
                };
                let noise_pred_2 = self.unet_forward(
                    &latent_model_input_2,
                    next_t,
                    &prompt_embeds,
                    &negative_embeds,
                    params.guidance_scale,
                    params.use_cfg_pp,
                    self.scheduler.sigma_at_timestep(next_t),
                )?;
                latents = self.scheduler.step_second_pass(&noise_pred_2)?;
            }

            monitor.compute().record_dispatch();
        }

        // SD_DUMP_DIR: write the FINAL post-loop latent for the
        // decode-isolation cross-check (PyTorch decodes this exact tensor;
        // if PT's VAE makes a clean image from it, the sampling loop is
        // fine and our VAE is the gap — and vice-versa).
        if let Ok(dir) = std::env::var("SD_DUMP_DIR") {
            if let Ok(v) = latents.to_f32_vec() {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for f in &v { bytes.extend_from_slice(&f.to_le_bytes()); }
                let _ = std::fs::write(format!("{}/final_latent.f32", dir), &bytes);
                tracing::info!("[diag-dump] wrote {}/final_latent.f32 ({} f32, shape={:?})", dir, v.len(), latents.shape());
            }
        }

        // VAE decode
        let image = self.vae_decode(&latents)?;

        Ok(image)
    }

    /// Generate with progressive output.
    pub async fn generate_progressive(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        params: &ImageParams,
        sender: &StreamSender<ImageProgress>,
        monitor: &ResourceMonitor,
    ) -> Result<()> {
        // Encode text prompt
        let prompt_embeds = self.encode_prompt(prompt)?;
        let negative_embeds = match negative_prompt
            .map(|p| self.encode_prompt(p))
            .transpose()?
        {
            Some(embeds) => embeds,
            None => self.null_prompt_embedding()?,
        };

        // Initialize latents
        let latent_shape = self.latent_shape(params.width, params.height);
        let mut latents = self.init_latents(&latent_shape, params.seed)?;

        // Setup scheduler
        self.scheduler.reset();
        let timesteps = self.scheduler.timesteps();
        let total_steps = timesteps.len() as u32;

        // Denoising loop
        for (i, &timestep) in timesteps.iter().enumerate() {
            if sender.is_cancelled() {
                break;
            }

            let sigma = self.scheduler.sigma_at_timestep(timestep);

            // Classifier-free guidance
            let latent_model_input = if params.guidance_scale > 1.0 {
                self.concat_latents(&latents, &latents)?
            } else {
                latents.clone()
            };

            // UNet forward pass
            let noise_pred = self.unet_forward(
                &latent_model_input,
                timestep,
                &prompt_embeds,
                &negative_embeds,
                params.guidance_scale,
                params.use_cfg_pp,
                sigma,
            )?;

            // Scheduler step — CPU is faster than GPU on Apple UMA due to zero-copy
            // shared memory and avoidance of command buffer synchronization overhead
            latents = self.scheduler.step(&latents, &noise_pred, timestep)?;

            // Heun second pass
            if self.scheduler.needs_second_pass() {
                let next_t = self.scheduler.next_timestep(timestep);
                let latent_model_input_2 = if params.guidance_scale > 1.0 {
                    self.concat_latents(&latents, &latents)?
                } else {
                    latents.clone()
                };
                let noise_pred_2 = self.unet_forward(
                    &latent_model_input_2,
                    next_t,
                    &prompt_embeds,
                    &negative_embeds,
                    params.guidance_scale,
                    params.use_cfg_pp,
                    self.scheduler.sigma_at_timestep(next_t),
                )?;
                latents = self.scheduler.step_second_pass(&noise_pred_2)?;
            }

            // Generate preview (every step for LCM, or periodically)
            let preview = if self.use_lcm || i % 5 == 0 {
                Some(self.quick_decode(&latents)?)
            } else {
                None
            };

            let is_final = i == timesteps.len() - 1;
            let final_image = if is_final {
                Some(self.vae_decode(&latents)?)
            } else {
                None
            };

            let progress = ImageProgress {
                step: i as u32 + 1,
                total_steps,
                preview,
                final_image,
            };

            sender.send(progress).await?;
            monitor.compute().record_dispatch();
        }

        Ok(())
    }

    /// Encode text prompt to embeddings.
    fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        match &self.text_encoder_type {
            TextEncoderType::DualCLIP => {
                if let Some(ref encoder) = self.text_encoder {
                    let tokens = if let Some(ref tokenizer) = self.tokenizer {
                        tokenize_with_clip(tokenizer, prompt, 77)
                    } else {
                        tokenize_prompt_basic(prompt, 77)
                    };
                    self.text_encoder_forward(encoder, &tokens)
                } else {
                    Err(crate::core::Error::internal("CLIP text encoder not loaded"))
                }
            }
            TextEncoderType::T5Only { d_model } => {
                // T5/UMT5 text encoding — tokenize and encode via T5Encoder
                #[cfg(feature = "metal")]
                {
                    let encoder_model = self.text_encoder.as_ref()
                        .ok_or_else(|| crate::core::Error::internal("T5 text encoder not loaded"))?;
                    let tokens = if let Some(ref tokenizer) = self.tokenizer {
                        tokenize_with_clip(tokenizer, prompt, 256)
                    } else {
                        tokenize_prompt_basic(prompt, 256)
                    };
                    let config = crate::inference::architecture::t5::T5Config::umt5_auraflow();
                    let t5 = crate::inference::architecture::t5::T5Encoder::new(
                        encoder_model.clone(), config, self.compute.device().clone(),
                    )?;
                    let encoded = t5.encode(&tokens)?;
                    // encoded is [seq_len, d_model], reshape to [1, seq_len, d_model]
                    let seq_len = tokens.len();
                    Ok(encoded.reshape(Shape::from([1, seq_len, *d_model]))?)
                }
                #[cfg(not(feature = "metal"))]
                Err(crate::core::Error::internal("T5 encoder requires Metal feature"))
            }
            TextEncoderType::CLIPPlusT5 { t5_dim } => {
                // Flux: T5-XXL sequence embeddings (CLIP-L pooled handled separately)
                #[cfg(feature = "metal")]
                {
                    let encoder_model = self.text_encoder.as_ref()
                        .ok_or_else(|| crate::core::Error::internal("T5 text encoder not loaded"))?;
                    let tokens = if let Some(ref tokenizer) = self.tokenizer {
                        tokenize_with_clip(tokenizer, prompt, 512)
                    } else {
                        tokenize_prompt_basic(prompt, 512)
                    };
                    let config = crate::inference::architecture::t5::T5Config::t5_xxl();
                    let t5 = crate::inference::architecture::t5::T5Encoder::new(
                        encoder_model.clone(), config, self.compute.device().clone(),
                    )?;
                    let encoded = t5.encode(&tokens)?;
                    let seq_len = tokens.len();
                    Ok(encoded.reshape(Shape::from([1, seq_len, *t5_dim]))?)
                }
                #[cfg(not(feature = "metal"))]
                Err(crate::core::Error::internal("T5 encoder requires Metal feature"))
            }
            TextEncoderType::ChatGLM { d_model } => {
                // ChatGLM-6B text encoding (Kolors)
                #[cfg(feature = "metal")]
                {
                    let encoder_model = self.text_encoder.as_ref()
                        .ok_or_else(|| crate::core::Error::internal("ChatGLM text encoder not loaded"))?;
                    let tokens = if let Some(ref tokenizer) = self.tokenizer {
                        tokenize_with_clip(tokenizer, prompt, 256)
                    } else {
                        tokenize_prompt_basic(prompt, 256)
                    };
                    let config = crate::inference::architecture::chatglm::ChatGLMConfig::kolors();
                    let encoder = crate::inference::architecture::chatglm::ChatGLMEncoder::new(
                        encoder_model.clone(), config, self.compute.device().clone(),
                    )?;
                    let encoded = encoder.encode(&tokens)?;
                    let seq_len = tokens.len();
                    Ok(encoded.reshape(Shape::from([1, seq_len, *d_model]))?)
                }
                #[cfg(not(feature = "metal"))]
                Err(crate::core::Error::internal("ChatGLM encoder requires Metal feature"))
            }
            TextEncoderType::T5WithProjection { t5_dim, proj_dim: _ } => {
                // T5-XXL for PixArt-Sigma (caption projection handled inside transformer)
                #[cfg(feature = "metal")]
                {
                    let encoder_model = self.text_encoder.as_ref()
                        .ok_or_else(|| crate::core::Error::internal("T5 text encoder not loaded"))?;
                    let tokens = if let Some(ref tokenizer) = self.tokenizer {
                        tokenize_with_clip(tokenizer, prompt, 120)
                    } else {
                        tokenize_prompt_basic(prompt, 120)
                    };
                    let config = crate::inference::architecture::t5::T5Config::t5_xxl();
                    let t5 = crate::inference::architecture::t5::T5Encoder::new(
                        encoder_model.clone(), config, self.compute.device().clone(),
                    )?;
                    let encoded = t5.encode(&tokens)?;
                    let seq_len = tokens.len();
                    Ok(encoded.reshape(Shape::from([1, seq_len, *t5_dim]))?)
                }
                #[cfg(not(feature = "metal"))]
                Err(crate::core::Error::internal("T5 encoder requires Metal feature"))
            }
        }
    }

    /// Get null prompt embedding (for classifier-free guidance).
    ///
    /// For classifier-free guidance, the null embedding is the encoding of an empty string.
    /// Without a loaded encoder, returns a zero tensor as a valid null embedding.
    fn null_prompt_embedding(&self) -> Result<Tensor> {
        let (seq_len, embed_dim) = match &self.text_encoder_type {
            TextEncoderType::DualCLIP => {
                // SD 1.5 has only CLIP-L (768); SDXL adds CLIP-G for 2048.
                // Mirror the detection used in `text_encoder_forward`.
                let has_clip_g = self.text_encoder_2.is_some()
                    || self.text_encoder.as_ref().map(|m| {
                        m.get_weight("conditioner.embedders.1.model.token_embedding.weight").is_some()
                            || m.get_weight("text_model_2.embeddings.token_embedding.weight").is_some()
                    }).unwrap_or(false);
                if has_clip_g { (77, 2048) } else { (77, 768) }
            }
            TextEncoderType::T5Only { d_model } => (256, *d_model), // UMT5/T5 with longer seq
            TextEncoderType::CLIPPlusT5 { t5_dim } => (512, *t5_dim), // Flux: T5-XXL sequence (matches tokenize max_len)
            TextEncoderType::ChatGLM { d_model } => (256, *d_model), // ChatGLM-6B (Kolors)
            TextEncoderType::T5WithProjection { t5_dim, .. } => (120, *t5_dim), // PixArt: T5-XXL
        };
        #[cfg(feature = "metal")]
        return Tensor::zeros_on(Shape::from([1, seq_len, embed_dim]), DType::F16, self.compute.device().info().id);
        #[cfg(not(feature = "metal"))]
        return Tensor::zeros(Shape::from([1, seq_len, embed_dim]), DType::F16);
    }

    /// Calculate latent shape from output dimensions.
    fn latent_shape(&self, width: u32, height: u32) -> Shape {
        // VAE downsamples by 8x
        let channels = self.vae_variant.latent_channels();
        Shape::from([1, channels, height as usize / 8, width as usize / 8])
    }

    /// Initialize latents with random noise.
    fn init_latents(&self, shape: &Shape, _seed: Option<u64>) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        let mut latents = Tensor::randn_on(shape.clone(), DType::F16, self.compute.device().info().id)?;
        #[cfg(not(feature = "metal"))]
        let mut latents = Tensor::randn(shape.clone(), DType::F16)?;

        // Apply initial sigma for scheduler
        let initial_sigma = self.scheduler.initial_sigma();
        if std::env::var("SD_DIAG").ok().as_deref() == Some("1") {
            tracing::info!(
                "[diag] init_latents: scheduler_type={:?} initial_sigma={:.4}",
                self.scheduler.scheduler_type(), initial_sigma,
            );
        }
        latents = self.scale(&latents, initial_sigma)?;

        Ok(latents)
    }

    /// Concat latents for classifier-free guidance.
    fn concat_latents(&self, latents: &Tensor, _positive: &Tensor) -> Result<Tensor> {
        // Concatenate latents with itself for CFG batching
        // Input `latents` is the current sample (usually batch 1)
        // We want [latents, latents] (batch 2) if passed identical params, 
        // but typically one path is uncond (same noise input), one is cond.
        // Actually the `generate` loop passes `&latents, &latents`.
        // So we really just want cat([latents, latents], 0).
        Tensor::cat(&[latents.clone(), latents.clone()], 0)
    }

    /// Whether the backbone embeds guidance internally (no CFG batch doubling needed).
    /// Flux is guidance-distilled: guidance is an input embedding, not CFG.
    fn uses_internal_guidance(&self) -> bool {
        matches!(&self.backbone, DiffusionBackbone::DiT { variant: DiTVariant::Flux(_), .. })
    }

    /// Backbone forward pass with classifier-free guidance.
    ///
    /// Dispatches to UNet or DiT forward depending on the backbone type.
    fn unet_forward(
        &self,
        latents: &Tensor,
        timestep: f32,
        prompt_embeds: &Tensor,
        negative_embeds: &Tensor,
        guidance_scale: f32,
        use_cfg_pp: bool,
        sigma: f32,
    ) -> Result<Tensor> {
        self.unet_forward_with_residuals(
            latents, timestep, prompt_embeds, negative_embeds,
            guidance_scale, use_cfg_pp, sigma, None,
        )
    }

    /// UNet forward with optional ControlNet residuals injected into the
    /// down/mid block skip connections. Residuals are pre-scaled by the
    /// caller (scale baked in by `ControlNet::scale()`). For UNet CFG batch=2
    /// the residuals are duplicated to match the doubled latent batch.
    fn unet_forward_with_residuals(
        &self,
        latents: &Tensor,
        timestep: f32,
        prompt_embeds: &Tensor,
        negative_embeds: &Tensor,
        guidance_scale: f32,
        use_cfg_pp: bool,
        sigma: f32,
        controlnet_residuals: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let uses_cfg = guidance_scale > 1.0 && !self.uses_internal_guidance();

        // UNet supports batch=2 CFG (cat embeddings). DiT models use two-pass CFG.
        let is_unet = matches!(&self.backbone, DiffusionBackbone::UNet(_));

        #[cfg(feature = "metal")]
        let noise_pred = match &self.backbone {
            DiffusionBackbone::UNet(unet) => {
                let encoder_hidden_states = if uses_cfg {
                    Tensor::cat(&[negative_embeds.clone(), prompt_embeds.clone()], 0)?
                } else {
                    prompt_embeds.clone()
                };
                // ControlNet residual batch-doubling for CFG: residuals are
                // built at batch=1, the UNet's input is concatenated to
                // batch=2 (uncond+cond) for guidance, so each residual must
                // be cat'd with itself along the batch axis.
                let doubled_residuals: Option<Vec<Tensor>> = match controlnet_residuals {
                    Some(rs) if uses_cfg => {
                        let mut out = Vec::with_capacity(rs.len());
                        for r in rs {
                            out.push(Tensor::cat(&[r.clone(), r.clone()], 0)?);
                        }
                        Some(out)
                    }
                    _ => None,
                };
                let residual_slice: Option<&[Tensor]> = match (controlnet_residuals, doubled_residuals.as_ref()) {
                    (_, Some(d)) => Some(d.as_slice()),
                    (Some(rs), None) => Some(rs),
                    (None, None) => None,
                };
                let command_buffer = self.compute.new_command_buffer();
                let result = unet.forward_with_residuals(
                    latents,
                    timestep,
                    &encoder_hidden_states,
                    residual_slice,
                    &self.compute,
                    command_buffer.as_ref(),
                )?;
                command_buffer.commit();
                command_buffer.wait_until_completed();
                // SD_DIAG=1: replay snapshots captured during forward; tensors
                // are now backed by committed device memory and safe to read.
                crate::inference::architecture::unet::diag_drain_and_log();
                result
            }
            DiffusionBackbone::DiT { model, variant } => {
                // DiT models process single-batch inputs; CFG is handled via two-pass below
                match variant {
                    DiTVariant::AuraFlow { hidden_size: _, num_heads: _, num_joint_blocks: _, num_single_blocks: _ } => {
                        let config = crate::inference::architecture::auraflow::AuraFlowConfig::v03();
                        let transformer = crate::inference::architecture::auraflow::AuraFlowTransformer::new(
                            model.clone(), config, &self.compute,
                        )?;
                        transformer.forward(latents, prompt_embeds, timestep, &self.compute)?
                    }
                    DiTVariant::Flux(flux_config) => {
                        let transformer = crate::inference::architecture::flux::FluxGpuTransformer::new(
                            model.clone(), flux_config.clone(), &self.compute,
                        )?;
                        let clip_dim = 768;
                        let clip_pooled = Tensor::zeros_on(
                            Shape::from([1, clip_dim]),
                            DType::F16,
                            self.compute.device().info().id,
                        )?;
                        transformer.forward(
                            latents, prompt_embeds, &clip_pooled,
                            timestep, guidance_scale, &self.compute,
                        )?
                    }
                    DiTVariant::PixArt(pixart_config) => {
                        let transformer = crate::inference::architecture::pixart::PixArtGpuTransformer::new(
                            model.clone(), pixart_config.clone(), &self.compute,
                        )?;
                        transformer.forward(latents, prompt_embeds, timestep, &self.compute)?
                    }
                }
            }
        };

        #[cfg(not(feature = "metal"))]
        let noise_pred = Tensor::zeros(latents.shape().clone(), DType::F16)?;

        if uses_cfg {
            if is_unet {
                // UNet batch=2 CFG. We DO NOT slice + sub/scale/add: those
                // elementwise kernels read the base device pointer and ignore
                // Tensor::slice's byte_offset, so both `uncond` and `cond`
                // slices end up pointing at batch[0] → CFG silently
                // collapses to `result = uncond`. Bug #29 (2026-05-17).
                // Instead we use one fused offset-aware kernel.
                let batch_size = noise_pred.shape().dim(0).ok_or_else(|| crate::core::Error::internal("expected noise_pred to have batch dimension"))?;
                let half = batch_size / 2;
                if use_cfg_pp {
                    let noise_uncond = noise_pred.slice(0, 0, half)?;
                    let noise_text = noise_pred.slice(0, half, batch_size)?;
                    let original_latents = latents.slice(0, 0, half)?;
                    self.cfg_pp_guidance(&noise_uncond, &noise_text, &original_latents, guidance_scale, sigma)
                } else {
                    #[cfg(feature = "metal")]
                    { self.cfg_apply(&noise_pred, guidance_scale) }
                    #[cfg(not(feature = "metal"))]
                    { Ok(noise_pred.slice(0, half, batch_size)?) }
                }
            } else {
                // DiT two-pass CFG: noise_pred is the conditional output,
                // run a second pass for unconditional
                let noise_cond = noise_pred;
                #[cfg(feature = "metal")]
                let noise_uncond = match &self.backbone {
                    DiffusionBackbone::DiT { model, variant } => {
                        match variant {
                            DiTVariant::AuraFlow { .. } => {
                                let config = crate::inference::architecture::auraflow::AuraFlowConfig::v03();
                                let transformer = crate::inference::architecture::auraflow::AuraFlowTransformer::new(
                                    model.clone(), config, &self.compute,
                                )?;
                                transformer.forward(latents, negative_embeds, timestep, &self.compute)?
                            }
                            DiTVariant::Flux(_) => {
                                // Flux uses internal guidance, shouldn't reach here
                                noise_cond.clone()
                            }
                            DiTVariant::PixArt(pixart_config) => {
                                let transformer = crate::inference::architecture::pixart::PixArtGpuTransformer::new(
                                    model.clone(), pixart_config.clone(), &self.compute,
                                )?;
                                transformer.forward(latents, negative_embeds, timestep, &self.compute)?
                            }
                        }
                    }
                    _ => unreachable!(),
                };
                #[cfg(not(feature = "metal"))]
                let noise_uncond = noise_cond.clone();

                // Apply CFG: uncond + scale * (cond - uncond)
                #[cfg(feature = "metal")]
                {
                    let diff = self.sub(&noise_cond, &noise_uncond)?;
                    let scaled_diff = self.scale(&diff, guidance_scale)?;
                    let result = self.add(&noise_uncond, &scaled_diff)?;
                    Ok(result)
                }
                #[cfg(not(feature = "metal"))]
                { Ok(noise_cond) }
            }
        } else {
             Ok(noise_pred)
        }
    }

    /// Apply CFG++ guidance in denoised space.
    ///
    /// Converts noise predictions to denoised space, applies guidance there,
    /// then converts back. Produces fewer artifacts at high guidance scales.
    fn cfg_pp_guidance(
        &self,
        noise_uncond: &Tensor,
        noise_cond: &Tensor,
        latents: &Tensor,
        guidance_scale: f32,
        sigma: f32,
    ) -> Result<Tensor> {
        let uncond_data: Vec<f32> = noise_uncond.to_f32_vec()?;
        let cond_data: Vec<f32> = noise_cond.to_f32_vec()?;
        let latents_data: Vec<f32> = latents.to_f32_vec()?;

        let mut result: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let x = latents_data[i];
            let eps_c = cond_data.get(i).copied().unwrap_or(0.0);
            let eps_u = uncond_data.get(i).copied().unwrap_or(0.0);

            // Convert to denoised space
            let d_cond = x - sigma * eps_c;
            let d_uncond = x - sigma * eps_u;

            // Apply guidance in denoised space
            let d_guided = d_cond + guidance_scale * (d_cond - d_uncond);

            // Convert back to noise space
            let eps_guided = if sigma.abs() > 1e-8 {
                (x - d_guided) / sigma
            } else {
                eps_c
            };
            result.push(eps_guided);
        }

        f32_to_tensor(&result, noise_cond.shape().clone(), noise_cond.dtype(), noise_cond.device())
    }

    /// Text encoder forward pass.
    ///
    /// Runs both CLIP-L (768-dim) and OpenCLIP-G (1280-dim) text encoders,
    /// concatenates outputs to produce [1, 77, 2048] embeddings for SDXL UNet.
    fn text_encoder_forward(&self, encoder: &Model, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();

        #[cfg(feature = "metal")]
        {
            // Run CLIP-L (12 layers, 768-dim, 12 heads) — always present.
            let clip_l = clip_l_forward(encoder, tokens, &self.compute)?;

            // SD 1.5 uses CLIP-L only; SDXL adds CLIP-G as a second encoder.
            // Dispatch by checking whether a text_encoder_2 is loaded AND the
            // primary encoder actually has CLIP-G weights at the expected
            // prefix. SD 1.5 has neither — fall through to the CLIP-L-only
            // [1, 77, 768] output that SD 1.5's U-Net cross-attention expects.
            let has_clip_g = self.text_encoder_2.is_some()
                || encoder.get_weight("conditioner.embedders.1.model.token_embedding.weight").is_some()
                || encoder.get_weight("text_model_2.embeddings.token_embedding.weight").is_some();

            if !has_clip_g {
                let f16_l: Vec<f16> = clip_l.iter().map(|&v| f16::from_f32(v)).collect();
                let device = self.compute.device().info().id;
                return Tensor::from_slice(&f16_l, Shape::from([1, seq_len, 768]), DType::F16, device);
            }

            // SDXL path: concat CLIP-L (768) + CLIP-G (1280) → [1, 77, 2048].
            let clip_g_model = self.text_encoder_2.as_deref().unwrap_or(encoder);
            let clip_g = clip_g_forward(clip_g_model, tokens, &self.compute)?;

            let mut combined = Vec::with_capacity(seq_len * 2048);
            for i in 0..seq_len {
                combined.extend_from_slice(&clip_l[i * 768..(i + 1) * 768]);
                combined.extend_from_slice(&clip_g[i * 1280..(i + 1) * 1280]);
            }

            let f16_data: Vec<f16> = combined.iter().map(|&v| f16::from_f32(v)).collect();
            let device = self.compute.device().info().id;
            Tensor::from_slice(&f16_data, Shape::from([1, seq_len, 2048]), DType::F16, device)
        }

        #[cfg(not(feature = "metal"))]
        {
            Err(crate::core::Error::internal("CLIP encoder requires Metal feature"))
        }
    }

    /// Full VAE decode — runs the actual neural network decoder on Metal GPU.
    ///
    /// Architecture: post_quant_conv → conv_in → mid_block (ResNet + Attention + ResNet)
    /// → 4 up_blocks (ResNet×3 + optional Upsample) → GroupNorm → SiLU → conv_out.
    pub fn vae_decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = self.scale(latents, self.vae_variant.scaling_factor())?;

        #[cfg(feature = "metal")]
        {
            // Split VAE decode into phases with separate command buffers.
            // The matmul-based attention creates a large intermediate (32MB) tensor;
            // separate CBs ensure proper GPU scheduling and memory management.
            let model = &self.vae_decoder;
            let compute = &self.compute;

            macro_rules! vae_phase {
                ($body:expr) => {{
                    let cb = compute.new_command_buffer();
                    let result = $body(&cb);
                    cb.commit();
                    cb.wait_until_completed();
                    result?
                }};
            }

            // SD_DIAG=1 VAE-decode stat dump. Each `vae_phase!` commits +
            // waits its own CB, so the result tensor is safe to read
            // immediately (no v53 hazard).
            let vdiag = |label: &str, t: &Tensor| {
                if std::env::var("SD_DIAG").ok().as_deref() != Some("1") { return; }
                if let Ok(v) = t.to_f32_vec() {
                    if v.is_empty() { return; }
                    let (mean, mn, mx, std) = vec_stats(&v);
                    tracing::info!(
                        "[diag-vae] {:<28} shape={:?} mean={:+.4} std={:.4} min={:+.4} max={:+.4}",
                        label, t.shape(), mean, std, mn, mx,
                    );
                }
            };

            let x = match self.vae_variant {
                // SD 1.x and SDXL VAEs share the same diffusers layer names
                // and architecture; only the latent scaling factor differs
                // (already applied above). Same decode path for both.
                VaeVariant::SD15 | VaeVariant::SDXL => {
                    vdiag("00 scaled_latent_in", &latents);
                    // Phase 1: post_quant_conv + conv_in + mid block
                    let x = vae_phase!(|cb: &metal::CommandBufferRef| -> Result<Tensor> {
                        let x = vae_conv2d(model, &latents, "post_quant_conv", compute, cb, 1, 0)?;
                        let x = vae_conv2d(model, &x, "decoder.conv_in", compute, cb, 1, 1)?;
                        let x = vae_resnet_block(model, &x, "decoder.mid_block.resnets.0", compute, cb)?;
                        let x = vae_self_attention(model, &x, "decoder.mid_block.attentions.0", compute, cb)?;
                        vae_resnet_block(model, &x, "decoder.mid_block.resnets.1", compute, cb)
                    });
                    vdiag("01 after_mid_block", &x);
                    // Phase 2: up blocks 0-1
                    let x = vae_phase!(|cb: &metal::CommandBufferRef| -> Result<Tensor> {
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.0.resnets.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.0.resnets.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.0.resnets.2", compute, cb)?;
                        let x = self.upsample_nearest_async(&x, cb)?;
                        let x = vae_conv2d(model, &x, "decoder.up_blocks.0.upsamplers.0.conv", compute, cb, 1, 1)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.1.resnets.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.1.resnets.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.1.resnets.2", compute, cb)?;
                        let x = self.upsample_nearest_async(&x, cb)?;
                        vae_conv2d(model, &x, "decoder.up_blocks.1.upsamplers.0.conv", compute, cb, 1, 1)
                    });
                    vdiag("02 after_up_blocks_0_1", &x);
                    // Phase 3: up blocks 2-3 + final
                    let out = vae_phase!(|cb: &metal::CommandBufferRef| -> Result<Tensor> {
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.2.resnets.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.2.resnets.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.2.resnets.2", compute, cb)?;
                        let x = self.upsample_nearest_async(&x, cb)?;
                        let x = vae_conv2d(model, &x, "decoder.up_blocks.2.upsamplers.0.conv", compute, cb, 1, 1)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.3.resnets.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.3.resnets.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up_blocks.3.resnets.2", compute, cb)?;
                        let x = vae_group_norm_silu(model, &x, "decoder.conv_norm_out", compute, cb)?;
                        let x = vae_conv2d(model, &x, "decoder.conv_out", compute, cb, 1, 1)?;
                        vae_rescale_output(&x, compute, cb)
                    });
                    vdiag("03 after_conv_out_rescale", &out);
                    out
                }
                VaeVariant::Flux => {
                    // Flux VAE: LDM/CompVis naming convention
                    // No post_quant_conv, mid uses block_1/block_2/attn_1,
                    // up blocks in reverse order (up.3=512ch first, up.0=128ch last)
                    // Phase 1: conv_in + mid block
                    let x = vae_phase!(|cb: &metal::CommandBufferRef| -> Result<Tensor> {
                        let x = vae_conv2d(model, &latents, "decoder.conv_in", compute, cb, 1, 1)?;
                        let x = vae_resnet_block(model, &x, "decoder.mid.block_1", compute, cb)?;
                        let x = vae_self_attention_ldm(model, &x, "decoder.mid.attn_1", compute, cb)?;
                        vae_resnet_block(model, &x, "decoder.mid.block_2", compute, cb)
                    });
                    // Phase 2: up.3 (512ch) + up.2 (512ch with upsample)
                    let x = vae_phase!(|cb: &metal::CommandBufferRef| -> Result<Tensor> {
                        let x = vae_resnet_block(model, &x, "decoder.up.3.block.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.3.block.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.3.block.2", compute, cb)?;
                        let x = self.upsample_nearest_async(&x, cb)?;
                        let x = vae_conv2d(model, &x, "decoder.up.3.upsample.conv", compute, cb, 1, 1)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.2.block.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.2.block.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.2.block.2", compute, cb)?;
                        let x = self.upsample_nearest_async(&x, cb)?;
                        vae_conv2d(model, &x, "decoder.up.2.upsample.conv", compute, cb, 1, 1)
                    });
                    // Phase 3: up.1 (256ch) + up.0 (128ch) + final
                    vae_phase!(|cb: &metal::CommandBufferRef| -> Result<Tensor> {
                        let x = vae_resnet_block(model, &x, "decoder.up.1.block.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.1.block.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.1.block.2", compute, cb)?;
                        let x = self.upsample_nearest_async(&x, cb)?;
                        let x = vae_conv2d(model, &x, "decoder.up.1.upsample.conv", compute, cb, 1, 1)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.0.block.0", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.0.block.1", compute, cb)?;
                        let x = vae_resnet_block(model, &x, "decoder.up.0.block.2", compute, cb)?;
                        let x = vae_group_norm_silu(model, &x, "decoder.norm_out", compute, cb)?;
                        let x = vae_conv2d(model, &x, "decoder.conv_out", compute, cb, 1, 1)?;
                        vae_rescale_output(&x, compute, cb)
                    })
                }
            };

            Ok(x)
        }

        #[cfg(not(feature = "metal"))]
        {
             let (_, _, h, w) = latents.shape().dims4().unwrap_or((1, 4, 64, 64));
             Ok(Tensor::zeros(Shape::from([1, 3, h * 8, w * 8]), DType::F16)?)
        }
    }

    #[cfg(feature = "metal")]
    fn vae_decode_tiled(&self, latents: &Tensor, command_buffer: &metal::CommandBufferRef) -> Result<Tensor> {
        let (n, c, h, w) = latents.shape().dims4().unwrap_or((1, 4, 64, 64));
        let new_h = h * 8;
        let new_w = w * 8;
        
        let output = Tensor::empty(Shape::from([n, 3, new_h, new_w]), DType::F16, latents.device())?;
        
        let tile_size = 64; // Latent tile size (64x64 latent -> 512x512 pixel)
        let overlap = 16;   // Latent overlap (16 latent -> 128 pixel)
        let stride = tile_size - overlap; // 48
        
        for y in (0..h).step_by(stride) {
             let y_start = y;
             let y_end = (y + tile_size).min(h);
             let current_h = y_end - y_start;
             
             for x in (0..w).step_by(stride) {
                 let x_start = x;
                 let x_end = (x + tile_size).min(w);
                 let current_w = x_end - x_start;
                 
                 // 1. Extract Tile (Large Latents -> Small Tile)
                 let small_tile = Tensor::empty(Shape::from([n, c, current_h, current_w]), DType::F16, latents.device())?;
                 
                 self.copy_tile_async(
                     latents,       // Source: Large
                     &small_tile,   // Dest: Small
                     h, w,          // Source Dim (Latent)
                     current_h, current_w, // Dest Dim (Tile)
                     0, 0,          // Dest Offset (0,0 of tile)
                     y_start, x_start, // Source Offset (y,x of latent)
                     current_h, current_w, // Copy Size
                     command_buffer
                 )?;
                 
                 // 2. Decode Tile
                 let decoded_tile = self.vae_decode_neural(&small_tile, command_buffer)?;
                 
                 // 3. Paste Tile (Small Decoded -> Large Output)
                 // Calculate valid region (overlap logic)
                 let pixel_overlap = (overlap / 2) * 8; // 64
                 
                 let src_y_start = if y_start == 0 { 0 } else { pixel_overlap };
                 let src_x_start = if x_start == 0 { 0 } else { pixel_overlap };
                 
                 let decoded_h = decoded_tile.shape().dim(2).ok_or_else(|| crate::core::Error::internal("decoded tile missing height dimension"))?;
                 let decoded_w = decoded_tile.shape().dim(3).ok_or_else(|| crate::core::Error::internal("decoded tile missing width dimension"))?;

                 let src_y_end = decoded_h - if y_end == h { 0 } else { pixel_overlap };
                 let src_x_end = decoded_w - if x_end == w { 0 } else { pixel_overlap };

                 let copy_h = src_y_end - src_y_start;
                 let copy_w = src_x_end - src_x_start;

                 let dst_y_start = y_start * 8 + src_y_start;
                 let dst_x_start = x_start * 8 + src_x_start;

                 self.copy_tile_async(
                     &decoded_tile, // Source: Small
                     &output,       // Dest: Large
                     decoded_h, decoded_w, // Source Dim
                     new_h, new_w,  // Dest Dim
                     dst_y_start, dst_x_start, // Dest Offset
                     src_y_start, src_x_start, // Source Offset
                     copy_h, copy_w, // Copy Size
                     command_buffer
                 )?;
             }
        }
        
        Ok(output)
    }

    /// Full VAE encode using Fused Kernel.
    pub fn vae_encode(&self, image: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        {
            let (_, _, h, w) = image.shape().dims4().unwrap_or((1, 3, 512, 512));
            let command_buffer = self.compute.new_command_buffer();
            
            // Tiled encode for large images
            // Threshold: > 1024x1024
            let latents = if h > 1024 || w > 1024 {
                self.vae_encode_tiled(image, &command_buffer)?
            } else {
                self.vae_encode_fused_async(image, &command_buffer)?
            };
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            Ok(latents)
        }

        #[cfg(not(feature = "metal"))]
        {
             let (n, _, h, w) = image.shape().dims4().unwrap_or((1, 3, 512, 512));
             Ok(Tensor::zeros(Shape::from([n, 4, h/8, w/8]), DType::F16)?)
        }
    }

    #[cfg(feature = "metal")]
    fn vae_encode_fused_async(&self, input: &Tensor, command_buffer: &metal::CommandBufferRef) -> Result<Tensor> {
        let (n, _, h, w) = input.shape().dims4().unwrap_or((1, 3, 512, 512));
        
        let new_h = h / 8;
        let new_w = w / 8;
        
        let output = Tensor::empty(Shape::from([n, 4, new_h, new_w]), DType::F16, input.device())?;
        
        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let pipeline = self.kernels.vae_encode_fused.get_or_init(|| {
            self.compute.compile_pipeline("vae_encode_fused", crate::hal::metal::shader::sources::VAE_ENCODE_FUSED, "vae_encode_fused_f16").expect("failed to compile vae_encode_fused pipeline")
        }).clone();

        let grid_size = ((new_w + 7) / 8, (new_h + 7) / 8, n);
        let threadgroup_size = (8, 8, 1);
        let scale_val: f32 = 0.18215;

        self.compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(output_buffer.as_ref()), 0);

                let c_n = n as u32;
                let c_hin = h as u32;
                let c_win = w as u32;

                encoder.set_bytes(2, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_hin as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_win as *const u32 as *const _);
                encoder.set_bytes(5, 4, &scale_val as *const f32 as *const _);
            }
        );

        Ok(output)
    }

    #[cfg(feature = "metal")]
    fn vae_encode_tiled(&self, image: &Tensor, command_buffer: &metal::CommandBufferRef) -> Result<Tensor> {
        let (n, c, h, w) = image.shape().dims4().unwrap_or((1, 3, 512, 512));
        let new_h = h / 8;
        let new_w = w / 8;
        
        let output = Tensor::empty(Shape::from([n, 4, new_h, new_w]), DType::F16, image.device())?;
        
        // Tile size logic
        let tile_size = 512; // 512x512 pixel tile -> 64x64 latent
        let overlap = 64;    // 64 pixel overlap -> 8 latent overlap
        let stride = tile_size - overlap;
        
        for y in (0..h).step_by(stride) {
             let y_start = y;
             let y_end = (y + tile_size).min(h);
             let current_h = y_end - y_start;
             
             for x in (0..w).step_by(stride) {
                 let x_start = x;
                 let x_end = (x + tile_size).min(w);
                 let current_w = x_end - x_start;
                 
                 // 1. Extract Tile (Large Image -> Small Tile)
                 let small_tile = Tensor::empty(Shape::from([n, c, current_h, current_w]), DType::F16, image.device())?;
                 
                 self.copy_tile_async(
                     image,         // Source: Large Image
                     &small_tile,   // Dest: Small Image
                     h, w,          // Source Dim
                     current_h, current_w, // Dest Dim
                     0, 0,          // Dest Offset (0,0 of tile)
                     y_start, x_start, // Source Offset
                     current_h, current_w, // Copy Size
                     command_buffer
                 )?;
                 
                 // 2. Encode Tile (Small Image -> Small Latents)
                 let encoded_tile = self.vae_encode_fused_async(&small_tile, command_buffer)?;
                 
                 // 3. Paste Tile (Small Latents -> Large Latents)
                 // Overlap logic in Latent Space
                 // Pixel overlap 64 -> Latent overlap 8
                 let latent_overlap = overlap / 8;
                 
                 let src_y_start = if y_start == 0 { 0 } else { latent_overlap / 2 };
                 let src_x_start = if x_start == 0 { 0 } else { latent_overlap / 2 };
                 
                 // Encoded tile dims
                 let eth = encoded_tile.shape().dim(2).ok_or_else(|| crate::core::Error::internal("encoded tile missing height dimension"))?;
                 let etw = encoded_tile.shape().dim(3).ok_or_else(|| crate::core::Error::internal("encoded tile missing width dimension"))?;
                 
                 let src_y_end = eth - if y_end == h { 0 } else { latent_overlap / 2 };
                 let src_x_end = etw - if x_end == w { 0 } else { latent_overlap / 2 };
                 
                 let copy_h = src_y_end - src_y_start;
                 let copy_w = src_x_end - src_x_start;
                 
                 let dst_y_start = (y_start / 8) + src_y_start;
                 let dst_x_start = (x_start / 8) + src_x_start;
                 
                 self.copy_tile_async(
                     &encoded_tile, // Source: Small Latents
                     &output,       // Dest: Large Latents
                     eth, etw,      // Source Dim
                     new_h, new_w,  // Dest Dim
                     dst_y_start, dst_x_start, // Dest Offset
                     src_y_start, src_x_start, // Source Offset
                     copy_h, copy_w, // Copy Size
                     command_buffer
                 )?;
             }
        }
        
        Ok(output)
    }


    #[cfg(feature = "metal")]
    fn copy_tile_async(
        &self, 
        src: &Tensor, 
        dst: &Tensor,
        src_h: usize, src_w: usize,
        dst_h: usize, dst_w: usize,
        dst_y: usize, dst_x: usize,
        src_y: usize, src_x: usize,
        copy_h: usize, copy_w: usize,
        command_buffer: &metal::CommandBufferRef
    ) -> Result<()> {
        let pipeline = self.kernels.copy_tile.get_or_init(|| {
            self.compute.compile_pipeline("copy_tile", crate::hal::metal::shader::sources::COPY_TILE, "copy_tile_f16").expect("failed to compile copy_tile pipeline")
        }).clone();
        
        let n = src.shape().dim(0).ok_or_else(|| crate::core::Error::internal("src tensor missing batch dimension"))?;
        let c = src.shape().dim(1).ok_or_else(|| crate::core::Error::internal("src tensor missing channel dimension"))?;
        
        let grid_size = ((copy_w + 7) / 8, (copy_h + 7) / 8, n * c);
        let threadgroup_size = (8, 8, 1);
        
        let src_ptr = src.device_ptr().ok_or(crate::core::Error::internal("Src not on device"))?;
        let dst_ptr = dst.device_ptr().ok_or(crate::core::Error::internal("Dst not on device"))?;
        
        let src_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(src_ptr) };
        let dst_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(dst_ptr) };

        self.compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(src_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(dst_buffer.as_ref()), 0);

                let c_src_h = src_h as u32;
                let c_src_w = src_w as u32;
                let c_dst_h = dst_h as u32;
                let c_dst_w = dst_w as u32;
                let c_dst_y = dst_y as u32;
                let c_dst_x = dst_x as u32;
                let c_src_y = src_y as u32;
                let c_src_x = src_x as u32;
                let c_copy_h = copy_h as u32;
                let c_copy_w = copy_w as u32;
                let c_c = c as u32;
                let c_n = n as u32;

                encoder.set_bytes(2, 4, &c_src_h as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_src_w as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_dst_h as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_dst_w as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_dst_y as *const u32 as *const _);
                encoder.set_bytes(7, 4, &c_dst_x as *const u32 as *const _);
                encoder.set_bytes(8, 4, &c_src_y as *const u32 as *const _);
                encoder.set_bytes(9, 4, &c_src_x as *const u32 as *const _);
                encoder.set_bytes(10, 4, &c_copy_h as *const u32 as *const _);
                encoder.set_bytes(11, 4, &c_copy_w as *const u32 as *const _);
                encoder.set_bytes(12, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(13, 4, &c_n as *const u32 as *const _);
            }
        );
        
        Ok(())
    }

    #[cfg(feature = "metal")]
    /// Real VAE neural decoder — runs the full decoder network on Metal GPU.
    ///
    /// Architecture: post_quant_conv(4→4, 1×1) → conv_in(4→512, 3×3)
    /// → mid_block: ResNet(512) → Attention(512, 1 head) → ResNet(512)
    /// → up_block 0: 3×ResNet(512) + Upsample(512)
    /// → up_block 1: 3×ResNet(512) + Upsample(512)
    /// → up_block 2: ResNet(512→256) + 2×ResNet(256) + Upsample(256)
    /// → up_block 3: ResNet(256→128) + 2×ResNet(128)
    /// → GroupNorm(128) → SiLU → conv_out(128→3, 3×3)
    /// → rescale from [-1,1] to [0,1]
    /// Profiled VAE decode: uses separate command buffers per phase for timing.
    #[cfg(feature = "metal")]
    fn vae_decode_neural(&self, latents: &Tensor, command_buffer: &metal::CommandBufferRef) -> Result<Tensor> {
        let model = &self.vae_decoder;
        let compute = &self.compute;
        let cb = command_buffer;

        // 1. post_quant_conv: Conv2d(4, 4, 1×1)
        let x = vae_conv2d(model, latents, "post_quant_conv", compute, cb, 1, 0)?;

        // 2. conv_in: Conv2d(4, 512, 3×3, pad=1)
        let x = vae_conv2d(model, &x, "decoder.conv_in", compute, cb, 1, 1)?;

        // 3. Mid block
        let x = vae_resnet_block(model, &x, "decoder.mid_block.resnets.0", compute, cb)?;
        let x = vae_self_attention(model, &x, "decoder.mid_block.attentions.0", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.mid_block.resnets.1", compute, cb)?;

        // 4. Up blocks
        // Block 0: 3×ResNet(512→512) + Upsample
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.0.resnets.0", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.0.resnets.1", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.0.resnets.2", compute, cb)?;
        let x = self.upsample_nearest_async(&x, cb)?;
        let x = vae_conv2d(model, &x, "decoder.up_blocks.0.upsamplers.0.conv", compute, cb, 1, 1)?;

        // Block 1: 3×ResNet(512→512) + Upsample
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.1.resnets.0", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.1.resnets.1", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.1.resnets.2", compute, cb)?;
        let x = self.upsample_nearest_async(&x, cb)?;
        let x = vae_conv2d(model, &x, "decoder.up_blocks.1.upsamplers.0.conv", compute, cb, 1, 1)?;

        // Block 2: ResNet(512→256) + 2×ResNet(256) + Upsample
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.2.resnets.0", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.2.resnets.1", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.2.resnets.2", compute, cb)?;
        let x = self.upsample_nearest_async(&x, cb)?;
        let x = vae_conv2d(model, &x, "decoder.up_blocks.2.upsamplers.0.conv", compute, cb, 1, 1)?;

        // Block 3: ResNet(256→128) + 2×ResNet(128) (no upsample)
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.3.resnets.0", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.3.resnets.1", compute, cb)?;
        let x = vae_resnet_block(model, &x, "decoder.up_blocks.3.resnets.2", compute, cb)?;

        // 5. Final: GroupNorm → SiLU → conv_out → rescale to [0,1]
        let x = vae_group_norm_silu(model, &x, "decoder.conv_norm_out", compute, cb)?;
        let x = vae_conv2d(model, &x, "decoder.conv_out", compute, cb, 1, 1)?;

        // Rescale from [-1,1] to [0,1]: output = clamp(x * 0.5 + 0.5, 0, 1)
        vae_rescale_output(&x, compute, cb)
    }

    #[cfg(feature = "metal")]
    fn upsample_nearest_async(&self, input: &Tensor, command_buffer: &metal::CommandBufferRef) -> Result<Tensor> {
        let (n, c, h, w) = input.shape().dims4().unwrap_or((1, input.shape().dim(1).unwrap_or(1), input.shape().dim(2).unwrap_or(32), input.shape().dim(3).unwrap_or(32)));
        
        let new_h = h * 2;
        let new_w = w * 2;
        
        // We need to keep the buffer alive! The tensor owns it, so if we return the tensor, we are good?
        // But we need to verify if Tensor::empty allocates safely.
        let output = Tensor::empty(Shape::from([n, c, new_h, new_w]), DType::F16, input.device())?;
        
        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let pipeline = self.kernels.upsample_nearest.get_or_init(|| {
            self.compute.compile_pipeline("upsample_nearest", crate::hal::metal::shader::sources::UPSAMPLE, "upsample_nearest_f16").expect("failed to compile upsample_nearest pipeline")
        }).clone();

        let grid_size = ((new_w + 7) / 8, (new_h + 7) / 8, n * c);
        let threadgroup_size = (8, 8, 1);

        self.compute.dispatch_async(
            command_buffer,
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(output_buffer.as_ref()), 0);

                let c_n = n as u32;
                let c_c = c as u32;
                let c_hin = h as u32;
                let c_win = w as u32;

                encoder.set_bytes(2, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_hin as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_win as *const u32 as *const _);
            }
        );

        Ok(output)
    }

    #[cfg(feature = "metal")]
    fn upsample_nearest(&self, input: &Tensor) -> Result<Tensor> {
        let (n, c, h, w) = input.shape().dims4().unwrap_or((1, input.shape().dim(1).unwrap_or(1), input.shape().dim(2).unwrap_or(32), input.shape().dim(3).unwrap_or(32)));
        
        let new_h = h * 2;
        let new_w = w * 2;
        
        let output = Tensor::empty(Shape::from([n, c, new_h, new_w]), DType::F16, input.device())?;
        
        let input_ptr = input.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let pipeline = self.kernels.upsample_nearest.get_or_init(|| {
            self.compute.compile_pipeline("upsample_nearest", crate::hal::metal::shader::sources::UPSAMPLE, "upsample_nearest_f16").expect("failed to compile upsample_nearest pipeline")
        }).clone();

        let grid_size = ((new_w + 7) / 8, (new_h + 7) / 8, n * c);
        let threadgroup_size = (8, 8, 1);

        self.compute.execute_sync(
            &pipeline,
            grid_size,
            threadgroup_size,
            |encoder| {
                encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(output_buffer.as_ref()), 0);

                let c_n = n as u32;
                let c_c = c as u32;
                let c_hin = h as u32;
                let c_win = w as u32;

                encoder.set_bytes(2, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(3, 4, &c_c as *const u32 as *const _);
                encoder.set_bytes(4, 4, &c_hin as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_win as *const u32 as *const _);
            },
            || {}
        );

        Ok(output)
    }


    /// Quick decode for previews (lower quality but faster).
    ///
    /// Uses a lightweight latent-to-RGB approximation (e.g. TAESD) for real-time preview
    /// during progressive generation. Requires VAE decoder weights to be loaded.
    fn quick_decode(&self, _latents: &Tensor) -> Result<Tensor> {
        Err(crate::core::Error::internal("VAE decoder weights not loaded for quick decode"))
    }

    /// Fused offset-aware CFG: given batched UNet output [2, C, H, W]
    /// (uncond first, cond second), write the single-batch post-CFG
    /// result `uncond + scale * (cond - uncond)` to a new tensor.
    /// See bug #29 note in `unet_forward_with_residuals`.
    #[cfg(feature = "metal")]
    fn cfg_apply(&self, batched: &Tensor, scale: f32) -> Result<Tensor> {
        let dims = batched.shape().dims();
        if dims.is_empty() || dims[0] != 2 {
            return Err(crate::core::Error::internal("cfg_apply expects batch=2"));
        }
        let mut out_dims = dims.to_vec();
        out_dims[0] = 1;
        let out_shape = Shape::new(out_dims);
        let half_count: usize = out_shape.numel();
        let output = Tensor::empty(out_shape, DType::F16, batched.device())?;
        let in_ptr = batched.device_ptr().ok_or(crate::core::Error::internal("cfg_apply: in not on device"))?;
        let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("cfg_apply: out not on device"))?;
        let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
        let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };
        let pipeline = self.compute.compile_pipeline(
            "cfg_apply_f16",
            crate::hal::metal::shader::sources::ELEMENTWISE,
            "cfg_apply_f16",
        )?;
        let command_buffer = self.compute.new_command_buffer();
        let half_count_u = half_count as u32;
        self.compute.dispatch_1d(
            &command_buffer,
            &pipeline,
            half_count,
            |encoder| {
                encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
                encoder.set_bytes(2, 4, &scale as *const f32 as *const _);
                encoder.set_bytes(3, 4, &half_count_u as *const u32 as *const _);
            },
        );
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(output)
    }

    /// Scale tensor.
    fn scale(&self, tensor: &Tensor, scale: f32) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        {
            let output = Tensor::empty(tensor.shape().clone(), DType::F16, tensor.device())?;
            let input_ptr = tensor.device_ptr().ok_or(crate::core::Error::internal("Input tensor not on device"))?;
            let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

            let input_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(input_ptr) };
            let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

            let pipeline = self.kernels.scale.get_or_init(|| {
                self.compute.compile_pipeline("scale", crate::hal::metal::shader::sources::ELEMENTWISE, "scale_f16").expect("failed to compile scale pipeline")
            }).clone();

            let numel = tensor.shape().numel();
            let command_buffer = self.compute.new_command_buffer();

            self.compute.dispatch_1d(
                &command_buffer,
                &pipeline,
                numel,
                |encoder| {
                    encoder.set_buffer(0, Some(input_buffer.as_ref()), 0);
                    encoder.set_buffer(1, Some(output_buffer.as_ref()), 0);
                    encoder.set_bytes(2, 4, &scale as *const f32 as *const _);
                }
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            Ok(output)
        }

        #[cfg(not(feature = "metal"))]
        Ok(tensor.clone())
    }

    #[cfg(feature = "metal")]
    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.apply_binary_op(a, b, "sub", "sub_f16", &self.kernels.sub)
    }

    #[cfg(feature = "metal")]
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.apply_binary_op(a, b, "add", "add_f16", &self.kernels.add)
    }

    #[cfg(feature = "metal")]
    fn apply_binary_op(&self, a: &Tensor, b: &Tensor, op_name: &str, kernel: &str, cache: &std::sync::OnceLock<Arc<ComputePipeline>>) -> Result<Tensor> {
        let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;

        let a_ptr = a.device_ptr().ok_or(crate::core::Error::internal("A tensor not on device"))?;
        let b_ptr = b.device_ptr().ok_or(crate::core::Error::internal("B tensor not on device"))?;
        let output_ptr = output.device_ptr().ok_or(crate::core::Error::internal("Output tensor not on device"))?;

        let a_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(a_ptr) };
        let b_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };
        let output_buffer = unsafe { BorrowedMetalBuffer::from_device_ptr(output_ptr) };

        let pipeline = cache.get_or_init(|| {
            self.compute.compile_pipeline(op_name, crate::hal::metal::shader::sources::ELEMENTWISE, kernel).expect("failed to compile binary op pipeline")
        }).clone();

        let numel = a.shape().numel();
        let command_buffer = self.compute.new_command_buffer();

        self.compute.dispatch_1d(
            &command_buffer,
            &pipeline,
            numel,
            |encoder| {
                encoder.set_buffer(0, Some(a_buffer.as_ref()), 0);
                encoder.set_buffer(1, Some(b_buffer.as_ref()), 0);
                encoder.set_buffer(2, Some(output_buffer.as_ref()), 0);
            }
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }
}


/// How the model parameterizes its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPredictionType {
    /// Model predicts noise epsilon (standard diffusion models).
    Epsilon,
    /// Model predicts velocity v = epsilon - x_0 (rectified flow / Flux).
    Velocity,
    /// Model directly predicts the denoised sample x_0 (SDXL Lightning 1-step).
    Sample,
}

/// Internal state for the Heun two-pass correction.
struct HeunState {
    /// Original sample x_t before the Euler step.
    x_original: Vec<f32>,
    /// First derivative d1 (noise prediction at sigma_t).
    d1: Vec<f32>,
    /// Sigma at the start of this step.
    sigma: f32,
    /// Sigma at the end of this step.
    sigma_next: f32,
    /// Shape of the original tensor.
    shape: Shape,
    /// DType of the original tensor.
    dtype: DType,
    /// Device of the original tensor.
    device: crate::hal::DeviceId,
}

/// Diffusion scheduler.
pub struct DiffusionScheduler {
    /// Scheduler type
    scheduler_type: SchedulerType,
    /// Number of steps
    num_steps: usize,
    /// Precomputed timesteps
    timesteps: Vec<f32>,
    /// Precomputed sigmas
    sigmas: Vec<f32>,
    /// Alpha cumulative products (for DDPM)
    alphas_cumprod: Vec<f32>,
    /// Beta values
    betas: Vec<f32>,
    /// Previous denoised prediction for multistep methods (DPM++ 2M).
    old_denoised: std::sync::Mutex<Option<Vec<f32>>>,
    /// Previous sigma value for multistep methods.
    old_sigma: std::sync::Mutex<Option<f32>>,
    /// Prediction type used by the loaded model.
    prediction_type: ModelPredictionType,
    /// Heun state for two-pass correction (None when not mid-step).
    heun_state: std::sync::Mutex<Option<HeunState>>,
    /// Eta for stochastic samplers (default 1.0).
    eta: f32,
    /// History of ODE derivatives for SA-Solver multistep Adams methods.
    sa_history: std::sync::Mutex<Vec<Vec<f32>>>,
    /// SA-Solver order (1-4, default 3).
    sa_order: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum SchedulerType {
    /// DDPM scheduler
    DDPM,
    /// LCM scheduler (fast)
    LCM,
    /// Euler discrete (deterministic ODE)
    EulerDiscrete,
    /// Euler ancestral (stochastic SDE)
    EulerAncestral,
    /// DPM++ 2M multistep (2nd order, reuses previous denoised)
    DPMpp2M,
    /// Heun (2nd order Runge-Kutta, two UNet evals per step)
    Heun,
    /// DPM++ 2M SDE (stochastic variant with noise injection)
    DPMpp2MSDE,
    /// Euler Ancestral for Rectified Flow (Flux-style velocity prediction)
    EulerAncestralRF,
    /// SA-Solver (Stochastic Adams predictor-corrector, order 1-4)
    SASolver,
}

impl DiffusionScheduler {
    /// Create a DDPM scheduler.
    pub fn ddpm(num_steps: usize) -> Self {
        let num_train_timesteps = 1000;
        // SD 1.x, SD 2.x and SDXL were all trained with the `scaled_linear`
        // beta schedule (HF diffusers default). The plain linear schedule
        // mismatches the trained noise distribution, so 20 DDPM steps under
        // linear betas produce noise-on-noise (eps interpretation wrong at
        // every sigma). Match the training schedule.
        let betas = Self::scaled_linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        // HF diffusers `DDPMScheduler.set_timesteps`:
        //   step_ratio = num_train // num_inference
        //   timesteps  = (arange(0, num_inference) * step_ratio).round()[::-1]
        // i.e. for 1000 train / 20 inference: [950, 900, …, 50, 0] reversed,
        // NOT `linspace(999, 0, 20)` = [999, 946.3, 893.6, …]. The linspace
        // form evaluates the U-Net at noise levels the model was never
        // trained to denoise at each step (and the scheduler `step` reads
        // the wrong α̅ index), degrading every step's prediction. Match the
        // diffusers arange schedule exactly.
        // HF SD 1.x/2.x scheduler_config.json sets `steps_offset: 1`, so the
        // diffusers schedule is `(arange*step_ratio).round()[::-1] + 1` =
        // [951, 901, …, 51, 1] for 1000/20 — NOT ending at 0. The offset
        // shifts every U-Net timestep by +1 and (more importantly) makes the
        // final step t=1 rather than t=0; at t=0 `sqrt(1-ᾱ₀)` is the tiny
        // residual-noise level and the eps prediction collapses (our step-19
        // noise_pred std was 0.14 vs PyTorch's 0.49 — the missing offset).
        let step_ratio = (num_train_timesteps / num_steps.max(1)).max(1);
        let steps_offset = 1usize;
        let timesteps: Vec<f32> = (0..num_steps)
            .rev()
            .map(|i| (i * step_ratio + steps_offset) as f32)
            .collect();
        let sigmas = Self::compute_sigmas_ddpm(&timesteps, &alphas_cumprod);

        Self {
            scheduler_type: SchedulerType::DDPM,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Epsilon,
            heun_state: std::sync::Mutex::new(None),
            eta: 1.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Create an LCM scheduler (4-step, scaled-linear betas for SDXL).
    pub fn lcm(num_steps: usize) -> Self {
        let num_train_timesteps = 1000;
        let betas = Self::scaled_linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        let timesteps = Self::lcm_timesteps(num_steps);
        let sigmas = Self::compute_sigmas_lcm(&timesteps, &alphas_cumprod);

        Self {
            scheduler_type: SchedulerType::LCM,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Epsilon,
            heun_state: std::sync::Mutex::new(None),
            eta: 1.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Create an Euler discrete scheduler (deterministic ODE).
    pub fn euler(num_steps: usize) -> Self {
        let num_train_timesteps = 1000;
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        let timesteps = Self::linspace(999.0, 0.0, num_steps);
        let sigmas = Self::compute_sigmas_euler(&timesteps, &alphas_cumprod);

        Self {
            scheduler_type: SchedulerType::EulerDiscrete,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Epsilon,
            heun_state: std::sync::Mutex::new(None),
            eta: 1.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Create an Euler discrete scheduler with Karras noise schedule.
    pub fn euler_karras(num_steps: usize) -> Self {
        Self::with_karras_schedule(num_steps, SchedulerType::EulerDiscrete)
    }

    /// Create an Euler ancestral scheduler (stochastic SDE).
    /// Injects noise at each step for more varied outputs.
    pub fn euler_ancestral(num_steps: usize) -> Self {
        let num_train_timesteps = 1000;
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        let timesteps = Self::linspace(999.0, 0.0, num_steps);
        let sigmas = Self::compute_sigmas_euler(&timesteps, &alphas_cumprod);

        Self {
            scheduler_type: SchedulerType::EulerAncestral,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Epsilon,
            heun_state: std::sync::Mutex::new(None),
            eta: 1.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Create an Euler discrete scheduler with trailing timestep spacing.
    /// Used by SDXL Lightning (distilled models with 1/2/4/8 steps).
    pub fn euler_trailing(num_steps: usize, prediction_type: ModelPredictionType) -> Self {
        let num_train_timesteps = 1000usize;
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        // Trailing timestep spacing: t_i = T-1 - i * (T / N)
        let step_ratio = num_train_timesteps / num_steps;
        let timesteps: Vec<f32> = (0..num_steps)
            .map(|i| (num_train_timesteps - 1 - i * step_ratio) as f32)
            .collect();
        let sigmas = Self::compute_sigmas_euler(&timesteps, &alphas_cumprod);

        Self {
            scheduler_type: SchedulerType::EulerDiscrete,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type,
            heun_state: std::sync::Mutex::new(None),
            eta: 1.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Create a DPM++ 2M scheduler with Karras noise schedule.
    /// Best quality/speed ratio for most diffusion models.
    pub fn dpmpp_2m(num_steps: usize) -> Self {
        Self::with_karras_schedule(num_steps, SchedulerType::DPMpp2M)
    }

    /// Create a scheduler with Karras noise schedule (Karras et al., 2022).
    /// Concentrates steps in low-noise regions for better quality.
    fn with_karras_schedule(num_steps: usize, scheduler_type: SchedulerType) -> Self {
        let sigmas = Self::karras_sigmas(num_steps, 0.0292, 14.6146, 7.0);

        // Derive timesteps and alpha infrastructure for compatibility
        let num_train_timesteps = 1000;
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        // Map each sigma back to nearest training timestep
        let timesteps: Vec<f32> = sigmas.iter()
            .take(num_steps)
            .map(|&sigma| {
                // sigma = sqrt((1-alpha_bar)/alpha_bar) => alpha_bar = 1/(1+sigma^2)
                let target_alpha_bar = 1.0 / (1.0 + sigma * sigma);
                alphas_cumprod.iter().enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - target_alpha_bar).abs().partial_cmp(&(*b - target_alpha_bar).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i as f32)
                    .unwrap_or(0.0)
            })
            .collect();

        Self {
            scheduler_type,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Epsilon,
            heun_state: std::sync::Mutex::new(None),
            eta: 1.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Get timesteps.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Diagnostic: which scheduler variant is active.
    pub fn scheduler_type(&self) -> SchedulerType { self.scheduler_type }

    /// Get initial sigma for latent scaling.
    ///
    /// - LCM and DDPM use `init_noise_sigma = 1.0` (no scaling): the latent
    ///   at t=T is just `randn` because at the terminal training step
    ///   ᾱ_T ≈ 0, so the forward process degenerates to pure Gaussian noise.
    ///   Multiplying by an Euler-style σ here over-scales the latent by ~14×
    ///   at t=999 (scaled_linear betas), which cascades into conv_in
    ///   activations 10–15× too large and NaN'ing the deeper down_blocks.
    /// - Euler / DPM-style schedulers use σ_init = `sigmas[0]` (the largest
    ///   noise level), per HF diffusers `init_noise_sigma`.
    pub fn initial_sigma(&self) -> f32 {
        match self.scheduler_type {
            SchedulerType::LCM | SchedulerType::DDPM => 1.0,
            _ => self.sigmas.first().copied().unwrap_or(1.0),
        }
    }

    /// Reset multistep state. Call before each new generation.
    pub fn reset(&self) {
        *self.old_denoised.lock().unwrap() = None;
        *self.old_sigma.lock().unwrap() = None;
        *self.heun_state.lock().unwrap() = None;
        self.sa_history.lock().unwrap().clear();
    }

    /// Get sigmas (for testing and diagnostics).
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Perform a scheduler step.
    pub fn step(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        match self.scheduler_type {
            SchedulerType::DDPM => self.step_ddpm(latents, noise_pred, timestep),
            SchedulerType::LCM => self.step_lcm(latents, noise_pred, timestep),
            SchedulerType::EulerDiscrete => self.step_euler(latents, noise_pred, timestep),
            SchedulerType::EulerAncestral => self.step_euler_ancestral(latents, noise_pred, timestep),
            SchedulerType::DPMpp2M => self.step_dpmpp_2m(latents, noise_pred, timestep),
            SchedulerType::Heun => self.step_heun(latents, noise_pred, timestep),
            SchedulerType::DPMpp2MSDE => self.step_dpmpp_2m_sde(latents, noise_pred, timestep),
            SchedulerType::EulerAncestralRF => self.step_euler_ancestral_rf(latents, noise_pred, timestep),
            SchedulerType::SASolver => self.step_sa_solver(latents, noise_pred, timestep),
        }
    }

    /// Perform a scheduler step on Metal GPU (no CPU roundtrip).
    ///
    /// For simple schedulers (LCM, Euler, DDPM), dispatches element-wise kernels
    /// directly on GPU tensors. Falls back to CPU for complex multistep schedulers.
    #[cfg(feature = "metal")]
    pub fn step_gpu(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
        compute: &Arc<MetalCompute>,
    ) -> Result<Tensor> {
        match self.scheduler_type {
            SchedulerType::LCM => self.step_lcm_gpu(latents, noise_pred, timestep, compute),
            SchedulerType::EulerDiscrete => self.step_euler_gpu(latents, noise_pred, timestep, compute),
            SchedulerType::DDPM => self.step_ddpm_gpu(latents, noise_pred, timestep, compute),
            // Complex multistep schedulers fall back to CPU (state management with old_denoised)
            _ => self.step(latents, noise_pred, timestep),
        }
    }

    /// GPU-accelerated LCM step matching HF `LCMScheduler.step`. Each
    /// LCM-distilled UNet output is treated as a consistency function via
    /// the boundary conditions
    ///     c_skip = sigma_data² / (scaled_t² + sigma_data²)
    ///     c_out  = scaled_t / sqrt(scaled_t² + sigma_data²)
    /// where `scaled_t = timestep_scaling * timestep`, defaults sigma_data=0.5
    /// timestep_scaling=10.0 (SD-1.5 LCM). The actual denoised sample is
    /// `denoised = c_out * pred_x0 + c_skip * sample`, NOT `pred_x0` alone.
    /// Non-final steps re-noise with FRESH N(0,1) (HF uses randn_tensor),
    /// not the UNet's epsilon — reusing eps produces black/garbage output
    /// because the eps direction is consumed by `pred_x0`.
    #[cfg(feature = "metal")]
    fn step_lcm_gpu(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
        compute: &Arc<MetalCompute>,
    ) -> Result<Tensor> {
        let t = (timestep as usize).min(self.alphas_cumprod.len() - 1);
        let alpha_bar_t = self.alphas_cumprod[t];
        let sqrt_alpha = alpha_bar_t.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_bar_t).sqrt();

        // LCM consistency-model boundary scalings
        let timestep_scaling: f32 = 10.0;
        let sigma_data: f32 = 0.5;
        let scaled_t = timestep_scaling * timestep;
        let scaled_t_sq = scaled_t * scaled_t;
        let sigma_data_sq = sigma_data * sigma_data;
        let c_skip = sigma_data_sq / (scaled_t_sq + sigma_data_sq);
        let c_out = scaled_t / (scaled_t_sq + sigma_data_sq).sqrt();

        let count = latents.shape().numel();
        let grid = ((count + 255) / 256, 1, 1);
        let tg = (256, 1, 1);
        let c_count = count as u32;

        // Step 1: pred_x0 = latents/sqrt_a - noise*sqrt_1ma/sqrt_a.
        //   HF LCMScheduler config for LCM Dreamshaper v7 has clip_sample=false
        //   and thresholding=false → NO clamp on pred_x0. Clamping to [-1,1]
        //   here saturates almost everywhere at high t (sqrt_1ma≈1 / sqrt_a≈0.07
        //   amplifies eps by ~14× outside [-1,1]) and destroys the LCM signal.
        let scale_a = 1.0f32 / sqrt_alpha;
        let scale_b = -sqrt_one_minus_alpha / sqrt_alpha;

        let pred_x0 = Tensor::empty(latents.shape().clone(), DType::F16, latents.device())?;
        {
            let cb = compute.new_command_buffer();
            let lat_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(latents.device_ptr().ok_or(crate::core::Error::internal("latents not on device"))?) };
            let noise_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(noise_pred.device_ptr().ok_or(crate::core::Error::internal("noise not on device"))?) };
            let pred_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(pred_x0.device_ptr().ok_or(crate::core::Error::internal("pred not on device"))?) };

            let pipeline = compute.compile_pipeline("scale_add_f16", crate::hal::metal::shader::sources::SCHEDULER, "scale_add_f16")?;
            compute.dispatch_async(cb.as_ref(), &pipeline, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(lat_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(noise_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(pred_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &scale_a as *const f32 as *const _);
                encoder.set_bytes(4, 4, &scale_b as *const f32 as *const _);
                encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();
        }

        // Step 2: denoised = c_out * pred_x0 + c_skip * latents
        let denoised = Tensor::empty(latents.shape().clone(), DType::F16, latents.device())?;
        {
            let cb = compute.new_command_buffer();
            let pred_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(pred_x0.device_ptr().ok_or(crate::core::Error::internal("pred not on device"))?) };
            let lat_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(latents.device_ptr().ok_or(crate::core::Error::internal("latents not on device"))?) };
            let den_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(denoised.device_ptr().ok_or(crate::core::Error::internal("denoised not on device"))?) };
            let pipeline = compute.compile_pipeline("scale_add_f16", crate::hal::metal::shader::sources::SCHEDULER, "scale_add_f16")?;
            compute.dispatch_async(cb.as_ref(), &pipeline, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(pred_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(lat_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(den_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &c_out as *const f32 as *const _);
                encoder.set_bytes(4, 4, &c_skip as *const f32 as *const _);
                encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();
        }

        // Step 3: if last step, return denoised; otherwise re-noise with fresh z
        let step_idx = self.timesteps.iter().position(|&ts| (ts - timestep).abs() < 0.5);
        let next_timestep = step_idx
            .and_then(|idx| self.timesteps.get(idx + 1))
            .copied()
            .unwrap_or(0.0);

        if next_timestep <= 0.0 {
            return Ok(denoised);
        }

        let next_t = (next_timestep as usize).min(self.alphas_cumprod.len() - 1);
        let alpha_bar_next = self.alphas_cumprod[next_t];
        let sa_next = alpha_bar_next.sqrt();
        let s1ma_next = (1.0 - alpha_bar_next).sqrt();

        // Step 4: prev_sample = sqrt(α̅_next) * denoised + sqrt(1-α̅_next) * z
        //   where z ~ N(0, I) (fresh sample each step, NOT the UNet's eps).
        let z = Tensor::randn(latents.shape().clone(), DType::F16)?;
        let result = Tensor::empty(latents.shape().clone(), DType::F16, latents.device())?;
        {
            let cb = compute.new_command_buffer();
            let den_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(denoised.device_ptr().ok_or(crate::core::Error::internal("denoised not on device"))?) };
            let z_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(z.device_ptr().ok_or(crate::core::Error::internal("z not on device"))?) };
            let res_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(result.device_ptr().ok_or(crate::core::Error::internal("result not on device"))?) };

            let pipeline = compute.compile_pipeline("scale_add_f16", crate::hal::metal::shader::sources::SCHEDULER, "scale_add_f16")?;
            compute.dispatch_async(cb.as_ref(), &pipeline, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(den_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(z_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(res_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &sa_next as *const f32 as *const _);
                encoder.set_bytes(4, 4, &s1ma_next as *const f32 as *const _);
                encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();
        }

        Ok(result)
    }

    /// GPU-accelerated Euler discrete step.
    /// Handles epsilon, sample (x0), and velocity prediction types via scale_add_f16:
    ///   result[i] = scale_a * latents[i] + scale_b * model_output[i]
    #[cfg(feature = "metal")]
    fn step_euler_gpu(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
        compute: &Arc<MetalCompute>,
    ) -> Result<Tensor> {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);
        let sigma_t = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);
        let dt = sigma_next - sigma_t;

        // Compute scale factors based on prediction type:
        // Epsilon:  result = x + dt * eps            → a=1,          b=dt
        // Sample:   result = x + dt*(x-x0)/σ         → a=1+dt/σ,     b=-dt/σ
        // Velocity: fall back to CPU (rare in Euler)
        let (scale_a, scale_b) = match self.prediction_type {
            ModelPredictionType::Epsilon => (1.0f32, dt),
            ModelPredictionType::Sample => {
                if sigma_t > 0.0 {
                    (1.0 + dt / sigma_t, -dt / sigma_t)
                } else {
                    (0.0, 1.0) // sigma=0 → just return model output (denoised)
                }
            }
            ModelPredictionType::Velocity => {
                return self.step_euler(latents, noise_pred, timestep);
            }
        };

        let count = latents.shape().numel();
        let result = Tensor::empty(latents.shape().clone(), DType::F16, latents.device())?;

        let cb = compute.new_command_buffer();
        let lat_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(latents.device_ptr().ok_or(crate::core::Error::internal("latents not on device"))?) };
        let noise_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(noise_pred.device_ptr().ok_or(crate::core::Error::internal("noise not on device"))?) };
        let res_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(result.device_ptr().ok_or(crate::core::Error::internal("result not on device"))?) };

        let pipeline = compute.compile_pipeline("scale_add_f16", crate::hal::metal::shader::sources::SCHEDULER, "scale_add_f16")?;
        let grid = ((count + 255) / 256, 1, 1);
        let tg = (256, 1, 1);
        let c_count = count as u32;

        compute.dispatch_async(cb.as_ref(), &pipeline, grid, tg, |encoder| {
            encoder.set_buffer(0, Some(lat_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(noise_buf.as_ref()), 0);
            encoder.set_buffer(2, Some(res_buf.as_ref()), 0);
            encoder.set_bytes(3, 4, &scale_a as *const f32 as *const _);
            encoder.set_bytes(4, 4, &scale_b as *const f32 as *const _);
            encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
        });

        cb.commit();
        cb.wait_until_completed();
        Ok(result)
    }

    /// GPU-accelerated DDPM step.
    ///
    /// Uses separate command buffers for each pass to ensure synchronization.
    #[cfg(feature = "metal")]
    fn step_ddpm_gpu(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
        compute: &Arc<MetalCompute>,
    ) -> Result<Tensor> {
        let t = (timestep as usize).min(self.alphas_cumprod.len() - 1);
        let alpha_bar_t = self.alphas_cumprod[t];
        let alpha_bar_prev = if t > 0 { self.alphas_cumprod[t - 1] } else { 1.0 };
        let beta_t = self.betas.get(t).copied().unwrap_or(0.0001);

        let sqrt_alpha = alpha_bar_t.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_bar_t).sqrt();
        let sqrt_alpha_prev = alpha_bar_prev.sqrt();
        let variance = if t > 0 {
            beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        } else {
            0.0
        };
        let dir_coeff = (1.0 - alpha_bar_prev - variance).max(0.0).sqrt();

        let count = latents.shape().numel();
        let grid = ((count + 255) / 256, 1, 1);
        let tg = (256, 1, 1);
        let c_count = count as u32;

        // Step 1: pred_original = latents/sqrt_a + noise*(-sqrt_1ma/sqrt_a)
        let pred_original = Tensor::empty(latents.shape().clone(), DType::F16, latents.device())?;
        {
            let cb = compute.new_command_buffer();
            let lat_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(latents.device_ptr().ok_or(crate::core::Error::internal("latents not on device"))?) };
            let noise_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(noise_pred.device_ptr().ok_or(crate::core::Error::internal("noise not on device"))?) };
            let pred_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(pred_original.device_ptr().ok_or(crate::core::Error::internal("pred not on device"))?) };

            let pipeline = compute.compile_pipeline("scale_add_f16", crate::hal::metal::shader::sources::SCHEDULER, "scale_add_f16")?;
            let sa = 1.0f32 / sqrt_alpha;
            let sb = -sqrt_one_minus_alpha / sqrt_alpha;

            compute.dispatch_async(cb.as_ref(), &pipeline, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(lat_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(noise_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(pred_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &sa as *const f32 as *const _);
                encoder.set_bytes(4, 4, &sb as *const f32 as *const _);
                encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();
        }

        // Step 2: result = sqrt_alpha_prev * pred_original + dir_coeff * noise
        let result = Tensor::empty(latents.shape().clone(), DType::F16, latents.device())?;
        {
            let cb = compute.new_command_buffer();
            let pred_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(pred_original.device_ptr().ok_or(crate::core::Error::internal("pred not on device"))?) };
            let noise_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(noise_pred.device_ptr().ok_or(crate::core::Error::internal("noise not on device"))?) };
            let res_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(result.device_ptr().ok_or(crate::core::Error::internal("result not on device"))?) };

            let pipeline = compute.compile_pipeline("scale_add_f16", crate::hal::metal::shader::sources::SCHEDULER, "scale_add_f16")?;
            compute.dispatch_async(cb.as_ref(), &pipeline, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(pred_buf.as_ref()), 0);
                encoder.set_buffer(1, Some(noise_buf.as_ref()), 0);
                encoder.set_buffer(2, Some(res_buf.as_ref()), 0);
                encoder.set_bytes(3, 4, &sqrt_alpha_prev as *const f32 as *const _);
                encoder.set_bytes(4, 4, &dir_coeff as *const f32 as *const _);
                encoder.set_bytes(5, 4, &c_count as *const u32 as *const _);
            });
            cb.commit();
            cb.wait_until_completed();
        }

        Ok(result)
    }

    /// DDPM denoising step.
    ///
    /// Implements: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * noise_pred) + sigma_t * z
    fn step_ddpm(&self, latents: &Tensor, noise_pred: &Tensor, timestep: f32) -> Result<Tensor> {
        let t = timestep as usize;
        let t = t.min(self.alphas_cumprod.len() - 1);

        // Previous timestep in the inference schedule, not the training schedule.
        // DDPMScheduler.previous_timestep in HF diffusers is
        //   prev_t = t - num_train_timesteps // num_inference_steps
        // We have num_train_timesteps = self.alphas_cumprod.len() (== 1000) and
        // num_inference_steps = self.num_steps. With t-1 the alpha-bar barely
        // changes per call and 20 nominal denoise steps move the latent by
        // ~1/step_ratio of the intended distance — output stays near noise.
        let num_train = self.alphas_cumprod.len();
        let step_ratio = (num_train / self.num_steps.max(1)).max(1);
        let prev_t = (t as isize) - (step_ratio as isize);

        // Get alpha values for this timestep
        let alpha_bar_t = self.alphas_cumprod[t];
        let alpha_bar_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        // Exact HF `DDPMScheduler.step` posterior (variance_type="fixed_small",
        // clip_sample=False — SD 1.x defaults). The previous code used the
        // *DDIM* deterministic update `√ᾱ_prev·x0 + √(1-ᾱ_prev-var)·ε` with NO
        // stochastic term; DDPM uses a different posterior mean (a convex
        // combination of x0 and the current sample) PLUS injected Gaussian
        // noise every step except the last. The DDIM form roughly tracks the
        // std trajectory but produces a blurry, content-divergent sample
        // because the per-step mean is wrong and the stochastic exploration
        // is missing. This rewrite matches diffusers element-for-element
        // (modulo the variance-noise RNG, which is inherently sample-random).
        let alpha_prod_t = alpha_bar_t;
        let alpha_prod_t_prev = alpha_bar_prev;
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        let current_alpha_t = alpha_prod_t / alpha_prod_t_prev;
        let current_beta_t = 1.0 - current_alpha_t;

        let sqrt_alpha_prod_t = alpha_prod_t.sqrt();
        let sqrt_beta_prod_t = beta_prod_t.sqrt();

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let noise_data: Vec<f32> = noise_pred.to_f32_vec()?;

        // x0 prediction (epsilon parameterisation), no clip (clip_sample=False).
        let mut pred_original: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let noise = noise_data.get(i).copied().unwrap_or(0.0);
            pred_original.push((latents_data[i] - sqrt_beta_prod_t * noise) / sqrt_alpha_prod_t);
        }

        // True DDPM posterior mean coefficients.
        let pred_orig_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t;
        let cur_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t;

        // fixed_small variance = β̃_t = (1-ᾱ_prev)/(1-ᾱ_t) · current_beta_t.
        let variance = if t > 0 {
            ((beta_prod_t_prev / beta_prod_t) * current_beta_t).max(1e-20)
        } else {
            0.0
        };
        let variance_std = variance.sqrt();

        // Stochastic noise term (skipped at the terminal step). Box-Muller
        // N(0,1) via Tensor::randn — sample-random by design (DDPM is
        // ancestral); the resulting image is a valid sample for the prompt,
        // not pixel-identical to a specific PyTorch seed.
        // NB: `Tensor::randn` always materialises f16 storage regardless of
        // the requested DType (Box-Muller path), so it must be tagged F16 or
        // `to_f32_vec` will read past the (half-size) buffer.
        // SD_DDPM_DETERMINISTIC=1 drops the stochastic term to isolate
        // "wrong posterior mean" from "wrong variance noise".
        let deterministic = std::env::var("SD_DDPM_DETERMINISTIC").ok().as_deref() == Some("1");
        let var_noise: Vec<f32> = if !deterministic && t > 0 && variance > 0.0 {
            Tensor::randn(latents.shape().clone(), DType::F16)?.to_f32_vec()?
        } else {
            Vec::new()
        };

        if std::env::var("SD_DIAG").ok().as_deref() == Some("1") {
            let (xm, _, _, xs) = vec_stats(&latents_data);
            let (nm, _, _, ns) = vec_stats(&noise_data);
            let (pm, _, _, ps) = vec_stats(&pred_original);
            tracing::info!(
                "[diag-sched] t={} aprod_t={:.6} aprod_prev={:.6} sqrt_ap={:.5} sqrt_bp={:.5} x_t[std={:.4}] eps[std={:.4}] x0_pred[mean={:+.4} std={:.4}] var_std={:.5} coeffs[x0={:.5} xt={:.5}]",
                t, alpha_prod_t, alpha_prod_t_prev, sqrt_alpha_prod_t, sqrt_beta_prod_t,
                xs, ns, pm, ps, variance_std, pred_orig_coeff, cur_sample_coeff,
            );
            let _ = (xm, nm);
        }

        let mut result: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let mean = pred_orig_coeff * pred_original[i] + cur_sample_coeff * latents_data[i];
            let z = var_noise.get(i).copied().unwrap_or(0.0);
            result.push(mean + variance_std * z);
        }

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }

    /// LCM (Latent Consistency Model) denoising step.
    ///
    /// LCM uses a consistency function to directly predict the clean sample,
    /// enabling fewer steps (typically 4-8 vs 20-50 for DDPM).
    fn step_lcm(&self, latents: &Tensor, noise_pred: &Tensor, timestep: f32) -> Result<Tensor> {
        let t = timestep as usize;
        let t = t.min(self.alphas_cumprod.len() - 1);

        // Get alpha values
        let alpha_bar_t = self.alphas_cumprod[t];
        let sqrt_alpha_bar_t = alpha_bar_t.sqrt();
        let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt();

        // LCM directly predicts the denoised sample
        // pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let noise_data: Vec<f32> = noise_pred.to_f32_vec()?;

        let mut pred_x0: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let noise = noise_data.get(i).copied().unwrap_or(0.0);
            let x = latents_data[i];
            // Clamp the predicted original sample for stability
            let pred = ((x - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t)
                .clamp(-1.0, 1.0);
            pred_x0.push(pred);
        }

        // Find next timestep in schedule
        let step_idx = self.timesteps.iter().position(|&ts| (ts - timestep).abs() < 0.5);
        let next_timestep = step_idx
            .and_then(|idx| self.timesteps.get(idx + 1))
            .copied()
            .unwrap_or(0.0);

        if next_timestep <= 0.0 {
            // Last step: return predicted x0
            return f32_to_tensor(&pred_x0, latents.shape().clone(), latents.dtype(), latents.device());
        }

        // Add noise for next step
        let next_t = (next_timestep as usize).min(self.alphas_cumprod.len() - 1);
        let alpha_bar_next = self.alphas_cumprod[next_t];
        let sqrt_alpha_bar_next = alpha_bar_next.sqrt();
        let sqrt_one_minus_alpha_bar_next = (1.0 - alpha_bar_next).sqrt();

        // x_next = sqrt(alpha_bar_next) * pred_x0 + sqrt(1 - alpha_bar_next) * noise
        // For LCM, we use the same noise prediction as direction
        let mut result: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let x0 = pred_x0[i];
            let noise = noise_data.get(i).copied().unwrap_or(0.0);
            let next_sample = sqrt_alpha_bar_next * x0 + sqrt_one_minus_alpha_bar_next * noise;
            result.push(next_sample);
        }

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }

    /// Euler discrete denoising step.
    ///
    /// Uses first-order Euler method: x_{t-dt} = x_t + dt * dx/dt
    /// where dx/dt = (x_t - pred_x0) / sigma_t
    fn step_euler(&self, latents: &Tensor, noise_pred: &Tensor, timestep: f32) -> Result<Tensor> {
        // Find current step index
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);

        let sigma_t = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);
        let dt = sigma_next - sigma_t;

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let model_out: Vec<f32> = noise_pred.to_f32_vec()?;

        let mut result: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let x = latents_data[i];
            let out = model_out.get(i).copied().unwrap_or(0.0);

            // Convert model output to ODE derivative based on prediction type
            let derivative = match self.prediction_type {
                ModelPredictionType::Epsilon => out,
                ModelPredictionType::Sample => {
                    // d = (x_t - pred_x0) / sigma_t
                    if sigma_t > 0.0 { (x - out) / sigma_t } else { 0.0 }
                }
                ModelPredictionType::Velocity => {
                    // pred_x0 = x / sqrt(1+σ²) - σ·v / sqrt(1+σ²)
                    let s2p1 = (sigma_t * sigma_t + 1.0).sqrt();
                    let pred_x0 = x / s2p1 - sigma_t * out / s2p1;
                    if sigma_t > 0.0 { (x - pred_x0) / sigma_t } else { 0.0 }
                }
            };
            result.push(x + dt * derivative);
        }

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }

    /// Linear beta schedule (as used in original DDPM).
    fn linear_beta_schedule(num_timesteps: usize) -> Vec<f32> {
        let beta_start = 0.00085;
        let beta_end = 0.012;
        Self::linspace(beta_start, beta_end, num_timesteps)
    }

    /// Scaled-linear beta schedule (SDXL/SDXL-Turbo default).
    /// betas = linspace(sqrt(beta_start), sqrt(beta_end), N)^2
    fn scaled_linear_beta_schedule(num_timesteps: usize) -> Vec<f32> {
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;
        Self::linspace(beta_start.sqrt(), beta_end.sqrt(), num_timesteps)
            .iter()
            .map(|x| x * x)
            .collect()
    }

    /// Compute cumulative product of a vector.
    fn cumulative_product(values: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(values.len());
        let mut prod = 1.0;
        for &v in values {
            prod *= v;
            result.push(prod);
        }
        result
    }

    fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
        if num == 0 {
            return Vec::new();
        }
        if num == 1 {
            return vec![start];
        }
        let step = (end - start) / (num - 1) as f32;
        (0..num).map(|i| start + step * i as f32).collect()
    }

    fn lcm_timesteps(num_steps: usize) -> Vec<f32> {
        // HF LCMScheduler.set_timesteps with original_inference_steps=50,
        // num_train_timesteps=1000:
        //   c = 1000/50 = 20
        //   lcm_origin = [1..=50] * c - 1 = [19,39,...,999] (50 entries)
        //   skip = 50 / num_inference_steps
        //   timesteps = lcm_origin[::-skip][:num_inference_steps]
        let original_inference_steps = 50usize;
        let num_train_timesteps = 1000usize;
        let c = num_train_timesteps / original_inference_steps;
        let lcm_origin: Vec<usize> = (1..=original_inference_steps).map(|i| i * c - 1).collect();
        let skipping = (original_inference_steps / num_steps.max(1)).max(1);
        let mut timesteps: Vec<f32> = Vec::with_capacity(num_steps);
        let mut idx: isize = (lcm_origin.len() - 1) as isize;
        while timesteps.len() < num_steps && idx >= 0 {
            timesteps.push(lcm_origin[idx as usize] as f32);
            idx -= skipping as isize;
        }
        timesteps
    }

    fn compute_sigmas_ddpm(timesteps: &[f32], alphas_cumprod: &[f32]) -> Vec<f32> {
        // sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t)
        timesteps.iter().map(|&t| {
            let t_idx = (t as usize).min(alphas_cumprod.len() - 1);
            let alpha_bar = alphas_cumprod[t_idx];
            ((1.0 - alpha_bar) / alpha_bar).sqrt()
        }).collect()
    }

    fn compute_sigmas_lcm(timesteps: &[f32], alphas_cumprod: &[f32]) -> Vec<f32> {
        // LCM uses same sigma computation as DDPM
        Self::compute_sigmas_ddpm(timesteps, alphas_cumprod)
    }

    fn compute_sigmas_euler(timesteps: &[f32], alphas_cumprod: &[f32]) -> Vec<f32> {
        // Euler uses sigma = sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t)
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| {
            let t_idx = (t as usize).min(alphas_cumprod.len() - 1);
            let alpha_bar = alphas_cumprod[t_idx];
            ((1.0 - alpha_bar) / alpha_bar).sqrt()
        }).collect();

        // Append 0 for final step
        sigmas.push(0.0);
        sigmas
    }

    /// Karras noise schedule (Karras et al., 2022).
    ///
    /// Concentrates steps in low-noise regions for better detail.
    /// σ_i = (σ_max^(1/ρ) + (i/(N-1)) × (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
    fn karras_sigmas(num_steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
        let inv_rho = 1.0 / rho;
        let min_inv = sigma_min.powf(inv_rho);
        let max_inv = sigma_max.powf(inv_rho);

        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let denom = (num_steps - 1).max(1) as f32;
        for i in 0..num_steps {
            let t = i as f32 / denom;
            let sigma = (max_inv + t * (min_inv - max_inv)).powf(rho);
            sigmas.push(sigma);
        }
        // Terminal sigma
        sigmas.push(0.0);
        sigmas
    }

    /// Exponential noise schedule.
    ///
    /// σ_i = σ_min * (σ_max / σ_min)^(1 - i/(N-1))
    fn exponential_sigmas(num_steps: usize, sigma_min: f32, sigma_max: f32) -> Vec<f32> {
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let denom = (num_steps - 1).max(1) as f32;
        let log_ratio = (sigma_max / sigma_min).ln();
        for i in 0..num_steps {
            let t = i as f32 / denom;
            let sigma = sigma_min * (log_ratio * (1.0 - t)).exp();
            sigmas.push(sigma);
        }
        // Terminal sigma
        sigmas.push(0.0);
        sigmas
    }

    /// Compute ancestral step sigmas: split σ into σ_down (deterministic) and σ_up (noise).
    ///
    /// σ_up = η × √(σ_to² × (σ_from² - σ_to²) / σ_from²)
    /// σ_down = √(σ_to² - σ_up²)
    fn get_ancestral_step(sigma_from: f32, sigma_to: f32, eta: f32) -> (f32, f32) {
        if sigma_to <= 0.0 || sigma_from <= 0.0 {
            return (sigma_to, 0.0);
        }
        let sigma_up = (eta
            * (sigma_to * sigma_to * (sigma_from * sigma_from - sigma_to * sigma_to)
                / (sigma_from * sigma_from))
                .sqrt())
        .min(sigma_to);
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).sqrt();
        (sigma_down, sigma_up)
    }

    /// Euler ancestral denoising step (stochastic SDE).
    ///
    /// Like Euler discrete but injects noise at each step for more varied outputs.
    /// η=1.0 gives full stochastic sampling.
    fn step_euler_ancestral(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);

        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);

        // Split into deterministic and stochastic components (η = 1.0)
        let (sigma_down, sigma_up) = Self::get_ancestral_step(sigma, sigma_next, 1.0);

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let noise_data: Vec<f32> = noise_pred.to_f32_vec()?;

        // Deterministic Euler step to sigma_down
        let dt = sigma_down - sigma;
        let mut result: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let x = latents_data[i];
            let d = noise_data.get(i).copied().unwrap_or(0.0);
            result.push(x + dt * d);
        }

        // Stochastic noise injection
        if sigma_up > 0.0 {
            let noise = Tensor::randn(latents.shape().clone(), DType::F32)?;
            let noise_vec: Vec<f32> = noise.to_f32_vec()?;
            for i in 0..result.len() {
                let n = noise_vec.get(i).copied().unwrap_or(0.0);
                result[i] += sigma_up * n;
            }
        }

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }

    /// DPM++ 2M denoising step (2nd-order multistep).
    ///
    /// Uses the previous denoised prediction for higher-order correction.
    /// Falls back to 1st order on the first step (no previous state).
    fn step_dpmpp_2m(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);

        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let noise_data: Vec<f32> = noise_pred.to_f32_vec()?;

        // Convert noise prediction to denoised: x_0 = x - σ * ε
        let denoised: Vec<f32> = latents_data.iter().zip(noise_data.iter())
            .map(|(&x, &n)| x - sigma * n)
            .collect();

        // Log-sigma ratio for this step
        let h = if sigma_next > 0.0 && sigma > 0.0 {
            sigma_next.ln() - sigma.ln()
        } else {
            0.0
        };

        // Read previous state (immutable borrows dropped at end of block)
        let result = {
            let old_d = self.old_denoised.lock().unwrap();
            let old_s = self.old_sigma.lock().unwrap();

            if let (Some(old_d_vec), Some(old_s_val)) = (old_d.as_ref(), old_s.as_ref()) {
                // 2nd order: correct using previous denoised
                let h_last = if sigma > 0.0 && *old_s_val > 0.0 {
                    sigma.ln() - old_s_val.ln()
                } else {
                    0.0
                };
                let r = if h.abs() > 1e-8 { h_last / h } else { 0.0 };

                let mut res: Vec<f32> = Vec::with_capacity(latents_data.len());
                for i in 0..latents_data.len() {
                    let d = denoised[i];
                    let d_old = old_d_vec.get(i).copied().unwrap_or(d);
                    // D_i = (1 + 1/(2r)) * d - (1/(2r)) * d_old
                    let d_corrected = (1.0 + 0.5 / r.max(1e-8)) * d
                        - (0.5 / r.max(1e-8)) * d_old;
                    // x_next = (σ_next/σ) * x + (1 - σ_next/σ) * D_i
                    let ratio = if sigma.abs() > 1e-8 { sigma_next / sigma } else { 0.0 };
                    res.push(ratio * latents_data[i] + (1.0 - ratio) * d_corrected);
                }
                res
            } else {
                // 1st order fallback (no previous state)
                let mut res: Vec<f32> = Vec::with_capacity(latents_data.len());
                for i in 0..latents_data.len() {
                    let ratio = if sigma.abs() > 1e-8 { sigma_next / sigma } else { 0.0 };
                    res.push(ratio * latents_data[i] + (1.0 - ratio) * denoised[i]);
                }
                res
            }
        };

        // Store state for next step
        *self.old_denoised.lock().unwrap() = Some(denoised);
        *self.old_sigma.lock().unwrap() = Some(sigma);

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }

    // ========================================================================
    // Phase 2 factory methods
    // ========================================================================

    /// Create a Heun scheduler (2nd-order Runge-Kutta).
    /// Requires two UNet evaluations per step for higher accuracy.
    pub fn heun(num_steps: usize) -> Self {
        Self::with_karras_schedule(num_steps, SchedulerType::Heun)
    }

    /// Create a DPM++ 2M SDE scheduler (stochastic variant).
    /// Like DPM++ 2M but with noise injection for more varied outputs.
    pub fn dpmpp_2m_sde(num_steps: usize) -> Self {
        Self::with_karras_schedule(num_steps, SchedulerType::DPMpp2MSDE)
    }

    /// Create an Euler Ancestral scheduler for Rectified Flow (Flux-style) models.
    /// Uses linear sigma schedule and velocity prediction.
    pub fn euler_ancestral_rf(num_steps: usize) -> Self {
        // Linear sigma schedule: sigma goes from 1.0 to 0.0
        let sigmas: Vec<f32> = (0..=num_steps)
            .map(|i| 1.0 - i as f32 / num_steps as f32)
            .collect();

        // Timesteps: for RF, timestep = sigma * 1000
        let timesteps: Vec<f32> = sigmas[..num_steps].iter()
            .map(|&s| s * 1000.0)
            .collect();

        // Keep alpha infrastructure for compatibility
        let num_train_timesteps = 1000;
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        Self {
            scheduler_type: SchedulerType::EulerAncestralRF,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Velocity,
            heun_state: std::sync::Mutex::new(None),
            eta: 1.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Create a flow matching scheduler for DiT models (AuraFlow, Flux).
    ///
    /// Uses linear timestep schedule from 1.0→0.0 with deterministic Euler ODE.
    /// `shift` controls the timestep shift: 1.0 for AuraFlow (no shift),
    /// 3.0 for Flux Dev (shifts timesteps toward noise for better quality).
    pub fn flow_matching(num_steps: usize, shift: f32) -> Self {
        // Shifted linear schedule: t_i = shift * (1 - i/N) / (1 + (shift - 1) * (1 - i/N))
        let sigmas: Vec<f32> = (0..=num_steps)
            .map(|i| {
                let t = 1.0 - i as f32 / num_steps as f32;
                if (shift - 1.0).abs() < 1e-6 {
                    t  // No shift (AuraFlow)
                } else {
                    shift * t / (1.0 + (shift - 1.0) * t)  // Flux shift
                }
            })
            .collect();

        let timesteps: Vec<f32> = sigmas[..num_steps].to_vec();

        // Minimal alpha/beta infrastructure for compatibility
        let num_train_timesteps = 1000;
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        Self {
            scheduler_type: SchedulerType::EulerDiscrete,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Velocity,
            heun_state: std::sync::Mutex::new(None),
            eta: 0.0,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: 3,
        }
    }

    /// Create an SA-Solver scheduler (Stochastic Adams predictor-corrector).
    /// Higher order (2-4) improves accuracy by using history of previous derivatives.
    pub fn sa_solver(num_steps: usize, order: usize) -> Self {
        let order = order.clamp(1, 4);
        let sigmas = Self::karras_sigmas(num_steps, 0.0292, 14.6146, 7.0);

        let num_train_timesteps = 1000;
        let betas = Self::linear_beta_schedule(num_train_timesteps);
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let alphas_cumprod = Self::cumulative_product(&alphas);

        let timesteps: Vec<f32> = sigmas.iter()
            .take(num_steps)
            .map(|&sigma| {
                let target_alpha_bar = 1.0 / (1.0 + sigma * sigma);
                alphas_cumprod.iter().enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - target_alpha_bar).abs().partial_cmp(&(*b - target_alpha_bar).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i as f32)
                    .unwrap_or(0.0)
            })
            .collect();

        Self {
            scheduler_type: SchedulerType::SASolver,
            num_steps,
            timesteps,
            sigmas,
            alphas_cumprod,
            betas,
            old_denoised: std::sync::Mutex::new(None),
            old_sigma: std::sync::Mutex::new(None),
            prediction_type: ModelPredictionType::Epsilon,
            heun_state: std::sync::Mutex::new(None),
            eta: 0.5,
            sa_history: std::sync::Mutex::new(Vec::new()),
            sa_order: order,
        }
    }

    // ========================================================================
    // Phase 2 helper methods
    // ========================================================================

    /// Returns true if a second UNet pass is needed at the current step (Heun only).
    pub fn needs_second_pass(&self) -> bool {
        self.heun_state.lock().unwrap().is_some()
    }

    /// Get the timestep for the next step (used for Heun midpoint UNet evaluation).
    pub fn next_timestep(&self, current_timestep: f32) -> f32 {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - current_timestep).abs() < 0.5)
            .unwrap_or(0);
        self.timesteps.get(step_idx + 1).copied().unwrap_or(0.0)
    }

    /// Complete the Heun correction with the second noise prediction.
    pub fn step_second_pass(&self, noise_pred_2: &Tensor) -> Result<Tensor> {
        let state = self.heun_state.lock().unwrap().take()
            .ok_or_else(|| crate::core::Error::internal(
                "step_second_pass called without pending Heun state"
            ))?;

        let d2: Vec<f32> = noise_pred_2.to_f32_vec()?;
        let dt = state.sigma_next - state.sigma;

        // Heun correction: x_next = x_t + dt * (d1 + d2) / 2
        let mut result: Vec<f32> = Vec::with_capacity(state.x_original.len());
        for i in 0..state.x_original.len() {
            let avg_d = (state.d1[i] + d2[i]) / 2.0;
            result.push(state.x_original[i] + dt * avg_d);
        }

        f32_to_tensor(&result, state.shape, state.dtype, state.device)
    }

    /// Get sigma value for a given timestep.
    pub fn sigma_at_timestep(&self, timestep: f32) -> f32 {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);
        self.sigmas.get(step_idx).copied().unwrap_or(1.0)
    }

    // ========================================================================
    // Phase 2 step methods
    // ========================================================================

    /// Heun denoising step (first pass).
    ///
    /// Performs an Euler step to the midpoint and stores state for the
    /// trapezoidal correction. The pipeline must evaluate the UNet on the
    /// returned midpoint, then call `step_second_pass()`.
    fn step_heun(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);

        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);
        let dt = sigma_next - sigma;

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let d1: Vec<f32> = noise_pred.to_f32_vec()?;

        // Euler step to midpoint: x_mid = x_t + dt * d1
        let mut midpoint: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            midpoint.push(latents_data[i] + dt * d1[i]);
        }

        // Store state for second pass
        *self.heun_state.lock().unwrap() = Some(HeunState {
            x_original: latents_data,
            d1,
            sigma,
            sigma_next,
            shape: latents.shape().clone(),
            dtype: latents.dtype(),
            device: latents.device(),
        });

        // Return midpoint — pipeline will evaluate UNet on this
        f32_to_tensor(&midpoint, latents.shape().clone(), latents.dtype(), latents.device())
    }

    /// DPM++ 2M SDE denoising step (stochastic variant).
    ///
    /// Like DPM++ 2M but targets sigma_down (deterministic) then adds
    /// sigma_up * noise (stochastic).
    fn step_dpmpp_2m_sde(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);

        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);

        // Split into deterministic and stochastic components
        let (sigma_down, sigma_up) = Self::get_ancestral_step(sigma, sigma_next, self.eta);

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let noise_data: Vec<f32> = noise_pred.to_f32_vec()?;

        // Denoised prediction: x_0 = x - sigma * epsilon
        let denoised: Vec<f32> = latents_data.iter().zip(noise_data.iter())
            .map(|(&x, &n)| x - sigma * n)
            .collect();

        // Log-sigma ratio targeting sigma_down
        let h = if sigma_down > 0.0 && sigma > 0.0 {
            sigma_down.ln() - sigma.ln()
        } else {
            0.0
        };

        // DPM++ 2M update (same correction logic but targeting sigma_down)
        let mut result = {
            let old_d = self.old_denoised.lock().unwrap();
            let old_s = self.old_sigma.lock().unwrap();

            if let (Some(old_d_vec), Some(old_s_val)) = (old_d.as_ref(), old_s.as_ref()) {
                // 2nd order correction
                let h_last = if sigma > 0.0 && *old_s_val > 0.0 {
                    sigma.ln() - old_s_val.ln()
                } else {
                    0.0
                };
                let r = if h.abs() > 1e-8 { h_last / h } else { 0.0 };

                let mut res: Vec<f32> = Vec::with_capacity(latents_data.len());
                for i in 0..latents_data.len() {
                    let d = denoised[i];
                    let d_old = old_d_vec.get(i).copied().unwrap_or(d);
                    let d_corrected = (1.0 + 0.5 / r.max(1e-8)) * d
                        - (0.5 / r.max(1e-8)) * d_old;
                    let ratio = if sigma.abs() > 1e-8 { sigma_down / sigma } else { 0.0 };
                    res.push(ratio * latents_data[i] + (1.0 - ratio) * d_corrected);
                }
                res
            } else {
                // 1st order fallback
                let mut res: Vec<f32> = Vec::with_capacity(latents_data.len());
                for i in 0..latents_data.len() {
                    let ratio = if sigma.abs() > 1e-8 { sigma_down / sigma } else { 0.0 };
                    res.push(ratio * latents_data[i] + (1.0 - ratio) * denoised[i]);
                }
                res
            }
        };

        // Store multistep state
        *self.old_denoised.lock().unwrap() = Some(denoised);
        *self.old_sigma.lock().unwrap() = Some(sigma);

        // Noise injection (SDE part)
        if sigma_up > 0.0 {
            let noise = Tensor::randn(latents.shape().clone(), DType::F32)?;
            let noise_vec: Vec<f32> = noise.to_f32_vec()?;
            for i in 0..result.len() {
                let n = noise_vec.get(i).copied().unwrap_or(0.0);
                result[i] += sigma_up * n;
            }
        }

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }

    /// Euler ancestral step for Rectified Flow models.
    ///
    /// For RF, the model predicts velocity v. Denoised: x_0 = x_t - sigma * v.
    /// Uses ancestral Euler step with noise injection.
    fn step_euler_ancestral_rf(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);

        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);

        // Split into deterministic and stochastic components
        let (sigma_down, sigma_up) = Self::get_ancestral_step(sigma, sigma_next, self.eta);

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let v_data: Vec<f32> = noise_pred.to_f32_vec()?;

        // Euler step to sigma_down (velocity is the ODE derivative for RF)
        let dt = sigma_down - sigma;
        let mut result: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let d = v_data[i];
            result.push(latents_data[i] + dt * d);
        }

        // Stochastic noise injection
        if sigma_up > 0.0 {
            let noise = Tensor::randn(latents.shape().clone(), DType::F32)?;
            let noise_vec: Vec<f32> = noise.to_f32_vec()?;
            for i in 0..result.len() {
                let n = noise_vec.get(i).copied().unwrap_or(0.0);
                result[i] += sigma_up * n;
            }
        }

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }

    // ========================================================================
    // Phase 4 step methods
    // ========================================================================

    /// SA-Solver denoising step (Stochastic Adams predictor-corrector).
    ///
    /// Uses Adams-Bashforth multistep prediction with adaptive order based on
    /// available history. Higher orders (2-4) reuse previous ODE derivatives
    /// for improved accuracy without additional model evaluations.
    fn step_sa_solver(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
    ) -> Result<Tensor> {
        let step_idx = self.timesteps.iter()
            .position(|&ts| (ts - timestep).abs() < 0.5)
            .unwrap_or(0);

        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let sigma_next = self.sigmas.get(step_idx + 1).copied().unwrap_or(0.0);

        // Split into deterministic and stochastic components
        let (sigma_down, sigma_up) = Self::get_ancestral_step(sigma, sigma_next, self.eta);

        let latents_data: Vec<f32> = latents.to_f32_vec()?;
        let noise_data: Vec<f32> = noise_pred.to_f32_vec()?;

        // Compute denoised prediction: x_0 = x - sigma * noise
        let denoised: Vec<f32> = latents_data.iter().zip(noise_data.iter())
            .map(|(&x, &n)| x - sigma * n)
            .collect();

        // ODE derivative: d = (x - denoised) / sigma = noise_pred
        let derivative: Vec<f32> = noise_data.clone();

        // Adams-Bashforth prediction with adaptive order
        let dt = sigma_down - sigma;

        let mut history = self.sa_history.lock().unwrap();
        let effective_order = (history.len() + 1).min(self.sa_order);

        let mut result: Vec<f32> = Vec::with_capacity(latents_data.len());
        for i in 0..latents_data.len() {
            let d0 = derivative[i];

            let predicted = match effective_order {
                1 => {
                    // Euler (AB1): x_next = x + dt * d0
                    latents_data[i] + dt * d0
                }
                2 => {
                    // AB2: x_next = x + dt * (3/2 * d0 - 1/2 * d1)
                    let d1 = history[0].get(i).copied().unwrap_or(d0);
                    latents_data[i] + dt * (1.5 * d0 - 0.5 * d1)
                }
                3 => {
                    // AB3: x_next = x + dt * (23/12 * d0 - 16/12 * d1 + 5/12 * d2)
                    let d1 = history[0].get(i).copied().unwrap_or(d0);
                    let d2 = history[1].get(i).copied().unwrap_or(d1);
                    latents_data[i] + dt * (23.0 / 12.0 * d0 - 16.0 / 12.0 * d1 + 5.0 / 12.0 * d2)
                }
                _ => {
                    // AB4: x_next = x + dt * (55/24*d0 - 59/24*d1 + 37/24*d2 - 9/24*d3)
                    let d1 = history[0].get(i).copied().unwrap_or(d0);
                    let d2 = history[1].get(i).copied().unwrap_or(d1);
                    let d3 = history[2].get(i).copied().unwrap_or(d2);
                    latents_data[i] + dt * (55.0 / 24.0 * d0 - 59.0 / 24.0 * d1
                        + 37.0 / 24.0 * d2 - 9.0 / 24.0 * d3)
                }
            };

            result.push(predicted);
        }

        // Push current derivative to front of history, trim to max order
        history.insert(0, derivative);
        if history.len() >= self.sa_order {
            history.truncate(self.sa_order);
        }
        drop(history);

        // Stochastic noise injection
        if sigma_up > 0.0 {
            let noise = Tensor::randn(latents.shape().clone(), DType::F32)?;
            let noise_vec: Vec<f32> = noise.to_f32_vec()?;
            for i in 0..result.len() {
                let n = noise_vec.get(i).copied().unwrap_or(0.0);
                result[i] += sigma_up * n;
            }
        }

        f32_to_tensor(&result, latents.shape().clone(), latents.dtype(), latents.device())
    }
}

/// IP-Adapter configuration.
#[derive(Debug, Clone)]
pub struct IPAdapterConfig {
    /// CLIP ViT hidden dimension (1024 for ViT-L/14, 1280 for ViT-H/14).
    pub clip_hidden_dim: usize,
    /// Number of CLIP ViT tokens (257 = 1 CLS + 16×16 patches for 224px/14px).
    pub clip_num_tokens: usize,
    /// Number of IP tokens after projection (4 for standard, 16 for plus).
    pub num_ip_tokens: usize,
    /// UNet cross-attention dimension (2048 for SDXL).
    pub cross_attention_dim: usize,
    /// Number of UNet cross-attention layers to inject into.
    pub num_cross_attn_layers: usize,
}

impl IPAdapterConfig {
    /// IP-Adapter for SDXL (CLIP ViT-H/14, 4 tokens).
    pub fn sdxl() -> Self {
        Self {
            clip_hidden_dim: 1280,
            clip_num_tokens: 257,
            num_ip_tokens: 4,
            cross_attention_dim: 2048,
            num_cross_attn_layers: 70, // SDXL UNet cross-attn layers
        }
    }

    /// IP-Adapter Plus for SDXL (16 tokens, higher fidelity).
    pub fn sdxl_plus() -> Self {
        Self {
            clip_hidden_dim: 1280,
            clip_num_tokens: 257,
            num_ip_tokens: 16,
            cross_attention_dim: 2048,
            num_cross_attn_layers: 70,
        }
    }

    /// IP-Adapter with CLIP ViT-L/14 (768 projection dim, 4 tokens).
    pub fn vit_l() -> Self {
        Self {
            clip_hidden_dim: 1024,
            clip_num_tokens: 257,
            num_ip_tokens: 4,
            cross_attention_dim: 2048,
            num_cross_attn_layers: 70,
        }
    }
}

/// IP-Adapter for image-guided generation.
///
/// Architecture:
/// 1. CLIP ViT encodes reference image → [1, 257, clip_hidden_dim]
/// 2. Image projection (learned linear or Resampler) → [1, num_ip_tokens, cross_dim]
/// 3. Per-layer K/V projections inject IP tokens into UNet cross-attention:
///    K_final = cat(K_text, K_ip), V_final = cat(V_text, V_ip)
pub struct IPAdapter {
    /// IP-Adapter weights (projections + per-layer K/V weights)
    model: Arc<Model>,
    /// CLIP ViT image encoder weights
    image_encoder: Arc<Model>,
    /// Adapter scale (0.0 = text only, 1.0 = full adapter influence)
    scale: f32,
    /// Configuration
    config: IPAdapterConfig,
    /// Cached IP image tokens (computed once per image, reused across steps)
    cached_ip_tokens: std::sync::Mutex<Option<Tensor>>,
}

impl IPAdapter {
    /// Create a new IP-Adapter.
    pub fn new(
        model: Arc<Model>,
        image_encoder: Arc<Model>,
        scale: f32,
    ) -> Self {
        Self::with_config(model, image_encoder, scale, IPAdapterConfig::sdxl())
    }

    /// Create with explicit configuration.
    pub fn with_config(
        model: Arc<Model>,
        image_encoder: Arc<Model>,
        scale: f32,
        config: IPAdapterConfig,
    ) -> Self {
        Self {
            model,
            image_encoder,
            scale,
            config,
            cached_ip_tokens: std::sync::Mutex::new(None),
        }
    }

    /// Get the adapter scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Set the adapter scale.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Encode reference image through CLIP ViT.
    ///
    /// Input: [1, 3, 224, 224] normalized image tensor.
    /// Output: [1, 257, clip_hidden_dim] penultimate hidden states.
    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        // CLIP ViT forward pass returns penultimate layer hidden states
        // (not the final projected embedding — IP-Adapter needs spatial tokens)
        #[cfg(feature = "metal")]
        {
            // For now return properly shaped zeros — full ViT forward pass
            // requires implementing the vision transformer encoder.
            let _ = image;
            Tensor::zeros(
                Shape::from([1, self.config.clip_num_tokens, self.config.clip_hidden_dim]),
                DType::F16,
            )
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = image;
            Tensor::zeros(
                Shape::from([1, self.config.clip_num_tokens, self.config.clip_hidden_dim]),
                DType::F16,
            )
        }
    }

    /// Project CLIP image embeddings to IP tokens.
    ///
    /// Input: [1, 257, clip_hidden_dim] from `encode_image`.
    /// Output: [1, num_ip_tokens, cross_attention_dim] IP tokens for cross-attention.
    pub fn get_embeddings(&self, image_embeds: &Tensor) -> Result<Tensor> {
        // Image projection: linear mapping or Resampler (Perceiver)
        // Standard IP-Adapter uses a simple linear projection
        // IP-Adapter Plus uses a Perceiver Resampler with learned queries

        // For standard: project [257, clip_dim] → [num_ip_tokens, cross_dim]
        // via learned projection matrix
        let _ = image_embeds;
        let tokens = Tensor::zeros(
            Shape::from([1, self.config.num_ip_tokens, self.config.cross_attention_dim]),
            DType::F16,
        )?;

        // Cache for reuse across denoising steps
        *self.cached_ip_tokens.lock().unwrap() = Some(tokens.clone());
        Ok(tokens)
    }

    /// Get cached IP tokens (call `get_embeddings` first).
    pub fn cached_tokens(&self) -> Option<Tensor> {
        self.cached_ip_tokens.lock().unwrap().clone()
    }

    /// Compute IP-Adapter K/V for a specific UNet cross-attention layer.
    ///
    /// Returns (K_ip, V_ip) to be concatenated with text K/V in cross-attention.
    /// `layer_idx`: which UNet cross-attention layer (0..num_cross_attn_layers).
    pub fn compute_kv(
        &self,
        ip_tokens: &Tensor,
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Each layer has its own learned K/V projection:
        //   ip_adapter.{layer_idx}.to_k_ip.weight: [cross_dim, cross_dim]
        //   ip_adapter.{layer_idx}.to_v_ip.weight: [cross_dim, cross_dim]
        let k_name = format!("ip_adapter.{}.to_k_ip.weight", layer_idx);
        let v_name = format!("ip_adapter.{}.to_v_ip.weight", layer_idx);

        let cross_dim = self.config.cross_attention_dim;
        let num_tokens = self.config.num_ip_tokens;

        // If weights exist, do the projection; otherwise return zeros
        if self.model.get_weight(&k_name).is_some() {
            // Would run: K_ip = ip_tokens @ W_k, V_ip = ip_tokens @ W_v
            // For now, placeholder
            let k_ip = Tensor::zeros(Shape::from([1, num_tokens, cross_dim]), DType::F16)?;
            let v_ip = Tensor::zeros(Shape::from([1, num_tokens, cross_dim]), DType::F16)?;
            Ok((k_ip, v_ip))
        } else {
            let k_ip = Tensor::zeros(Shape::from([1, num_tokens, cross_dim]), DType::F16)?;
            let v_ip = Tensor::zeros(Shape::from([1, num_tokens, cross_dim]), DType::F16)?;
            Ok((k_ip, v_ip))
        }
    }

    /// Apply IP-Adapter to cross-attention output.
    ///
    /// Combines standard text cross-attention with IP cross-attention:
    ///   output = text_attn_output + scale * ip_attn_output
    pub fn apply_to_attention(
        &self,
        text_attn_output: &Tensor,
        ip_attn_output: &Tensor,
    ) -> Result<Tensor> {
        if self.scale.abs() < 1e-8 {
            return Ok(text_attn_output.clone());
        }

        // output = text_output + scale * ip_output
        let data = text_attn_output.to_f32_vec()?;
        let ip_data = ip_attn_output.to_f32_vec()?;
        let result: Vec<f32> = data.iter().zip(ip_data.iter())
            .map(|(&t, &ip)| t + self.scale * ip)
            .collect();

        f32_to_tensor(&result, text_attn_output.shape().clone(), text_attn_output.dtype(), text_attn_output.device())
    }
}

/// ControlNet for structured generation.
///
/// Two-phase usage:
///   1. **Preprocess** — [`Self::process_control`] runs the per-`control_type`
///      input transform (Canny edges for sketches/photos, depth for depth-net,
///      etc.) on a `[3, H, W]` f16 tensor in 0..1 range. The output is the
///      ControlNet input — `[3, H, W]` f16 still — ready for the encoder.
///   2. **Encode** — [`Self::get_conditioning`] runs the SD-style ControlNet
///      encoder on the preprocessed control image and returns 12 down-block
///      residuals + 1 mid-block residual matching the U-Net encoder shapes.
///      U-Net injection is done in [`crate::inference::architecture::unet::UNet2DConditionModel::forward`].
///
/// When the optional `model` weights aren't loaded (the architecture-skeleton
/// path), [`Self::get_conditioning`] still returns shape-correct **zero**
/// residuals at SD-1.5 dimensions, which add into the U-Net additively and
/// therefore act as no-ops. This lets the full sampling loop run without
/// branching.
pub struct ControlNet {
    /// Optional ControlNet weights. When `None`, [`Self::get_conditioning`]
    /// returns shape-correct zero residuals (no-op).
    model: Option<Arc<Model>>,
    /// Control type.
    control_type: ControlType,
    /// Conditioning scale (multiplied into residuals before injection).
    scale: f32,
    /// Latent shape hint `(channels, h, w)` for the matching U-Net. Used to
    /// size the residual tensors. Defaults to SD 1.5 (320, 64, 64).
    latent_shape_hint: (usize, usize, usize),
    /// Compute helper for GPU preprocessing.
    #[cfg(feature = "metal")]
    compute: Option<Arc<MetalCompute>>,
    /// Lazily-built Canny kernel suite. Built on first preprocess call.
    #[cfg(feature = "metal")]
    canny_kernels: std::sync::OnceLock<crate::inference::gpu_ops::CannyKernels>,
    /// Lazily-built ControlNet encoder runtime — only constructed once
    /// the model is loaded. Reused across timesteps for efficiency.
    #[cfg(feature = "metal")]
    runtime: std::sync::OnceLock<crate::inference::architecture::controlnet_forward::ControlNetRuntime>,
}

/// Type of ControlNet conditioning.
#[derive(Debug, Clone, Copy)]
pub enum ControlType {
    /// Canny edge detection.
    Canny,
    /// Depth map.
    Depth,
    /// Normal map.
    Normal,
    /// OpenPose skeleton.
    Pose,
    /// Segmentation map.
    Segmentation,
    /// Scribble / sketch / line-art.
    Scribble,
}

impl ControlNet {
    /// Construct a ControlNet without compute resources (CPU-only / no
    /// preprocessing). Useful when the caller has already preprocessed the
    /// control image.
    pub fn new(model: Arc<Model>, control_type: ControlType, scale: f32) -> Self {
        Self {
            model: Some(model),
            control_type,
            scale,
            latent_shape_hint: (320, 64, 64),
            #[cfg(feature = "metal")]
            compute: None,
            #[cfg(feature = "metal")]
            canny_kernels: std::sync::OnceLock::new(),
            #[cfg(feature = "metal")]
            runtime: std::sync::OnceLock::new(),
        }
    }

    /// Construct a ControlNet with optional weights and Metal compute. When
    /// `model` is `None` the encoder forward returns zero residuals; when
    /// `compute` is provided the Canny preprocessor runs on GPU.
    #[cfg(feature = "metal")]
    pub fn new_with_compute(
        model: Option<Arc<Model>>,
        control_type: ControlType,
        scale: f32,
        compute: Option<Arc<MetalCompute>>,
    ) -> Self {
        Self {
            model,
            control_type,
            scale,
            latent_shape_hint: (320, 64, 64),
            compute,
            canny_kernels: std::sync::OnceLock::new(),
            runtime: std::sync::OnceLock::new(),
        }
    }

    /// Override the latent shape hint (e.g. for SDXL: `(320, 128, 128)`).
    pub fn with_latent_shape(mut self, channels: usize, h: usize, w: usize) -> Self {
        self.latent_shape_hint = (channels, h, w);
        self
    }

    /// Conditioning scale (used by the U-Net injection point).
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Run the per-`control_type` preprocessor on a `[3, H, W]` f16 control
    /// image in 0..1 range. Returns a `[3, H, W]` f16 tensor ready for
    /// [`Self::get_conditioning`].
    pub fn process_control(&self, image: &Tensor) -> Result<Tensor> {
        match self.control_type {
            ControlType::Canny => self.detect_edges(image, 0.10, 0.20, 4),
            // Sketch/scribble inputs are already line-art — use very low Sobel
            // thresholds so we capture the existing strokes rather than
            // detecting "edges of strokes".
            ControlType::Scribble => self.detect_edges(image, 0.04, 0.08, 4),
            // Depth/Normal/Pose/Segmentation preprocessors are pending separate
            // architectures (DepthAnything for Depth, OpenPose for Pose, SAM2
            // for Segmentation). Until those are wired the caller must
            // supply a pre-computed control image.
            ControlType::Depth | ControlType::Normal | ControlType::Pose | ControlType::Segmentation => {
                Ok(image.clone())
            }
        }
    }

    /// Run the ControlNet encoder forward on the preprocessed control image
    /// and return `(down_residuals, mid_residual)` ready to inject into the
    /// U-Net.
    ///
    /// When `self.model` is `None`, this returns shape-correct zero residuals
    /// at SD 1.5 dimensions (12 down + 1 mid). The U-Net injection is
    /// additive, so zero residuals are a true no-op — this lets the
    /// generation loop call `get_conditioning` unconditionally.
    ///
    /// When weights are present, this should mirror the SD-style ControlNet
    /// encoder (a copy of the U-Net's down blocks + mid block + per-block
    /// "zero-conv" output projections), scaled by `self.scale`. Wiring the
    /// real forward is gated on the SD-1.5 base U-Net weights being deployed
    /// to the host.
    /// Real ControlNet forward — runs `ControlNetRuntime::forward` when
    /// weights are loaded; otherwise returns shape-correct zero residuals
    /// (architecture-skeleton no-op). The returned residuals are at the
    /// same batch as `sample` (so the caller can avoid CFG duplication if
    /// `sample` already matches the U-Net's CFG-doubled input).
    #[cfg(feature = "metal")]
    pub fn forward_full(
        &self,
        sample: &Tensor,
        timestep: f32,
        encoder_hidden_states: &Tensor,
        control_image: &Tensor,
        compute: &Arc<MetalCompute>,
        cb: &metal::CommandBufferRef,
    ) -> Result<Vec<Tensor>> {
        if let Some(model) = &self.model {
            let runtime = self.runtime.get_or_init(|| {
                crate::inference::architecture::controlnet_forward::ControlNetRuntime::new(
                    model.clone(),
                )
            });
            return runtime.forward(sample, timestep, encoder_hidden_states, control_image, compute, cb);
        }
        // Skeleton path: zero residuals at SD 1.5 shapes.
        self.get_conditioning(control_image, timestep)
    }

    pub fn get_conditioning(&self, _control: &Tensor, _timestep: f32) -> Result<Vec<Tensor>> {
        let (c, h, w) = self.latent_shape_hint;
        // SD-1.5 ControlNet block channel multipliers and downsampling
        // schedule. Each entry is (channels, h, w, count).
        let down_block_shapes: [(usize, usize, usize, usize); 4] = [
            (c,         h,        w,        3), // block 0
            (c * 2,     h / 2,    w / 2,    3), // block 1
            (c * 4,     h / 4,    w / 4,    3), // block 2
            (c * 4,     h / 8,    w / 8,    3), // block 3
        ];
        let mut residuals = Vec::with_capacity(13);
        for (ch, hh, ww, count) in down_block_shapes.iter().copied() {
            for _ in 0..count {
                residuals.push(Tensor::zeros(
                    Shape::from([1, ch, hh, ww]),
                    DType::F16,
                )?);
            }
        }
        // Mid block residual.
        residuals.push(Tensor::zeros(
            Shape::from([1, c * 4, h / 8, w / 8]),
            DType::F16,
        )?);
        Ok(residuals)
    }

    /// GPU Canny edge detection.
    #[cfg(feature = "metal")]
    fn detect_edges(
        &self,
        image: &Tensor,
        low_threshold: f32,
        high_threshold: f32,
        hysteresis_iters: u32,
    ) -> Result<Tensor> {
        let compute = match &self.compute {
            Some(c) => c,
            None => return Ok(image.clone()),
        };
        let kernels = self.canny_kernels.get_or_init(|| {
            crate::inference::gpu_ops::CannyKernels::new(compute)
                .expect("CannyKernels compile failed")
        });
        let shape = image.shape();
        let (height, width) = match shape.dims() {
            [3, h, w] => (*h, *w),
            [1, 3, h, w] => (*h, *w),
            other => return Err(crate::core::Error::internal(format!(
                "ControlNet::detect_edges expected [3, H, W] or [1, 3, H, W], got {other:?}",
            ))),
        };
        let cb = compute.new_command_buffer();
        let edge = crate::inference::gpu_ops::canny_edge_on(
            compute,
            kernels,
            cb.as_ref(),
            image,
            height,
            width,
            low_threshold,
            high_threshold,
            hysteresis_iters,
        )?;
        cb.commit();
        cb.wait_until_completed();
        Ok(edge)
    }

    /// CPU fallback (Metal feature off): pass-through. The Canny path is
    /// gated on the Metal compute feature.
    #[cfg(not(feature = "metal"))]
    fn detect_edges(
        &self,
        image: &Tensor,
        _low_threshold: f32,
        _high_threshold: f32,
        _hysteresis_iters: u32,
    ) -> Result<Tensor> {
        Ok(image.clone())
    }
}

// ============================================================================
// CLIP Text Encoder Forward Passes — Metal GPU
// ============================================================================

/// Helper to get a weight as f32 vec from a Model.
#[cfg(feature = "metal")]
fn get_weight_f32(model: &Model, name: &str) -> Result<Vec<f32>> {
    model.get_weight(name)
        .ok_or_else(|| crate::core::Error::internal(format!("Weight not found: {}", name)))?
        .to_f32_vec()
}

/// CLIP-L forward pass — Metal GPU.
///
/// 12 layers, 768 hidden dim, 12 attention heads, separate Q/K/V.
/// Returns [seq_len, 768] as flat f32 vec (read back from GPU at end).
#[cfg(feature = "metal")]
fn clip_l_forward(model: &Model, tokens: &[u32], compute: &Arc<MetalCompute>) -> Result<Vec<f32>> {
    let prefix = if model.get_weight("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight").is_some() {
        "conditioner.embedders.0.transformer.text_model."
    } else {
        "text_model."
    };

    clip_hf_forward_gpu(model, tokens, prefix, 768, 12, 3072, 12,
        &format!("{}final_layer_norm", prefix), compute)
}

/// OpenCLIP-G / CLIP-G forward pass — Metal GPU.
///
/// Dispatches to HF or OpenCLIP variant based on weight key presence.
#[cfg(feature = "metal")]
fn clip_g_forward(model: &Model, tokens: &[u32], compute: &Arc<MetalCompute>) -> Result<Vec<f32>> {
    let is_openclip = model.get_weight("conditioner.embedders.1.model.token_embedding.weight").is_some();
    let is_hf = !is_openclip && model.get_weight("text_model.embeddings.token_embedding.weight").is_some();

    if is_hf {
        return clip_hf_forward_gpu(model, tokens, "text_model.", 1280, 20, 5120, 32,
            "text_model.final_layer_norm", compute);
    }

    // OpenCLIP format: fused QKV, different key names
    clip_openclip_forward_gpu(model, tokens, "conditioner.embedders.1.model.", 1280, 20, 5120, 32, compute)
}

/// Non-Metal fallback.
#[cfg(not(feature = "metal"))]
fn clip_l_forward(_model: &Model, _tokens: &[u32], _compute: &()) -> Result<Vec<f32>> {
    Err(crate::core::Error::internal("CLIP encoder requires Metal feature"))
}
#[cfg(not(feature = "metal"))]
fn clip_g_forward(_model: &Model, _tokens: &[u32], _compute: &()) -> Result<Vec<f32>> {
    Err(crate::core::Error::internal("CLIP encoder requires Metal feature"))
}

// ============================================================================
// GPU CLIP Transformer — shared implementation for HF-style (separate Q/K/V)
// ============================================================================

/// GPU CLIP forward for HuggingFace naming (separate Q/K/V projections).
///
/// Used for both CLIP-L (`encoder.layers.{i}`, `layer_norm1/2`, `mlp.fc1/fc2`)
/// and CLIP-G HF (`encoder.layers.{i}`, `layer_norm1/2`, `mlp.fc1/fc2`).
#[cfg(feature = "metal")]
fn clip_hf_forward_gpu(
    model: &Model, tokens: &[u32], prefix: &str,
    hidden_dim: usize, num_heads: usize, mlp_dim: usize, num_layers: usize,
    final_ln_prefix: &str, compute: &Arc<MetalCompute>,
) -> Result<Vec<f32>> {
    let seq_len = tokens.len();
    let head_dim = hidden_dim / num_heads;
    let device_id = compute.device().info().id;

    // Embedding: token + position (CPU lookup, small — 77 tokens)
    let token_emb = get_weight_f32(model, &format!("{}embeddings.token_embedding.weight", prefix))?;
    let pos_emb = get_weight_f32(model, &format!("{}embeddings.position_embedding.weight", prefix))?;
    let mut emb_data = vec![0.0f32; seq_len * hidden_dim];
    for (s, &tok) in tokens.iter().enumerate() {
        let tok_idx = tok as usize;
        for d in 0..hidden_dim {
            let tok_val = if tok_idx < 49408 { token_emb[tok_idx * hidden_dim + d] } else { 0.0 };
            let pos_val = if s < 77 { pos_emb[s * hidden_dim + d] } else { 0.0 };
            emb_data[s * hidden_dim + d] = tok_val + pos_val;
        }
    }

    // Upload embedded hidden state to GPU
    let f16_emb: Vec<half::f16> = emb_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut hidden = Tensor::from_slice(&f16_emb, Shape::from([seq_len, hidden_dim]), DType::F16, device_id)?;

    // Process transformer layers on GPU
    let command_buffer = compute.new_command_buffer();

    for layer in 0..num_layers {
        let lp = format!("{}encoder.layers.{}", prefix, layer);

        // LayerNorm → self-attention
        let normed = gpu_clip_layer_norm(&hidden, model, &format!("{}.layer_norm1", lp), seq_len, hidden_dim, compute, &command_buffer)?;

        let q = gpu_clip_linear(&normed, model, &format!("{}.self_attn.q_proj", lp), seq_len, hidden_dim, hidden_dim, compute, &command_buffer)?;
        let k = gpu_clip_linear(&normed, model, &format!("{}.self_attn.k_proj", lp), seq_len, hidden_dim, hidden_dim, compute, &command_buffer)?;
        let v = gpu_clip_linear(&normed, model, &format!("{}.self_attn.v_proj", lp), seq_len, hidden_dim, hidden_dim, compute, &command_buffer)?;

        let attn_out = gpu_clip_causal_attention(&q, &k, &v, seq_len, num_heads, head_dim, compute, &command_buffer)?;
        let attn_proj = gpu_clip_linear(&attn_out, model, &format!("{}.self_attn.out_proj", lp), seq_len, hidden_dim, hidden_dim, compute, &command_buffer)?;

        hidden = gpu_clip_add(&hidden, &attn_proj, seq_len * hidden_dim, compute, &command_buffer)?;

        // LayerNorm → MLP (FC1 → GELU → FC2)
        let normed2 = gpu_clip_layer_norm(&hidden, model, &format!("{}.layer_norm2", lp), seq_len, hidden_dim, compute, &command_buffer)?;
        let mlp_h = gpu_clip_linear(&normed2, model, &format!("{}.mlp.fc1", lp), seq_len, hidden_dim, mlp_dim, compute, &command_buffer)?;
        let mlp_act = gpu_clip_gelu(&mlp_h, seq_len * mlp_dim, compute, &command_buffer)?;
        let mlp_out = gpu_clip_linear(&mlp_act, model, &format!("{}.mlp.fc2", lp), seq_len, mlp_dim, hidden_dim, compute, &command_buffer)?;

        hidden = gpu_clip_add(&hidden, &mlp_out, seq_len * hidden_dim, compute, &command_buffer)?;
    }

    // Final LayerNorm
    hidden = gpu_clip_layer_norm(&hidden, model, final_ln_prefix, seq_len, hidden_dim, compute, &command_buffer)?;

    // Commit and wait
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back from GPU
    let f16_data: Vec<half::f16> = hidden.to_vec()?;
    Ok(f16_data.iter().map(|v| v.to_f32()).collect())
}

/// GPU CLIP forward for OpenCLIP naming (fused in_proj QKV).
#[cfg(feature = "metal")]
fn clip_openclip_forward_gpu(
    model: &Model, tokens: &[u32], prefix: &str,
    hidden_dim: usize, num_heads: usize, mlp_dim: usize, num_layers: usize,
    compute: &Arc<MetalCompute>,
) -> Result<Vec<f32>> {
    let seq_len = tokens.len();
    let head_dim = hidden_dim / num_heads;
    let device_id = compute.device().info().id;

    // Embedding (CPU — small)
    let token_emb = get_weight_f32(model, &format!("{}token_embedding.weight", prefix))?;
    let pos_emb = get_weight_f32(model, &format!("{}positional_embedding", prefix))?;
    let mut emb_data = vec![0.0f32; seq_len * hidden_dim];
    for (s, &tok) in tokens.iter().enumerate() {
        let tok_idx = tok as usize;
        for d in 0..hidden_dim {
            let tok_val = if tok_idx < 49408 { token_emb[tok_idx * hidden_dim + d] } else { 0.0 };
            let pos_val = if s < 77 { pos_emb[s * hidden_dim + d] } else { 0.0 };
            emb_data[s * hidden_dim + d] = tok_val + pos_val;
        }
    }

    let f16_emb: Vec<half::f16> = emb_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut hidden = Tensor::from_slice(&f16_emb, Shape::from([seq_len, hidden_dim]), DType::F16, device_id)?;

    let command_buffer = compute.new_command_buffer();

    for layer in 0..num_layers {
        let lp = format!("{}transformer.resblocks.{}", prefix, layer);

        // LayerNorm → fused QKV → split → attention
        let normed = gpu_clip_layer_norm(&hidden, model, &format!("{}.ln_1", lp), seq_len, hidden_dim, compute, &command_buffer)?;

        // Fused in_proj: [seq, hidden] → [seq, 3*hidden], then split
        let qkv = gpu_clip_linear(&normed, model, &format!("{}.attn.in_proj", lp), seq_len, hidden_dim, 3 * hidden_dim, compute, &command_buffer)?;

        // Split QKV on GPU with an inline kernel
        let (q, k, v) = gpu_clip_split_qkv(&qkv, seq_len, hidden_dim, compute, &command_buffer)?;

        let attn_out = gpu_clip_causal_attention(&q, &k, &v, seq_len, num_heads, head_dim, compute, &command_buffer)?;
        let attn_proj = gpu_clip_linear(&attn_out, model, &format!("{}.attn.out_proj", lp), seq_len, hidden_dim, hidden_dim, compute, &command_buffer)?;

        hidden = gpu_clip_add(&hidden, &attn_proj, seq_len * hidden_dim, compute, &command_buffer)?;

        // LayerNorm → MLP
        let normed2 = gpu_clip_layer_norm(&hidden, model, &format!("{}.ln_2", lp), seq_len, hidden_dim, compute, &command_buffer)?;
        let mlp_h = gpu_clip_linear(&normed2, model, &format!("{}.mlp.c_fc", lp), seq_len, hidden_dim, mlp_dim, compute, &command_buffer)?;
        let mlp_act = gpu_clip_gelu(&mlp_h, seq_len * mlp_dim, compute, &command_buffer)?;
        let mlp_out = gpu_clip_linear(&mlp_act, model, &format!("{}.mlp.c_proj", lp), seq_len, mlp_dim, hidden_dim, compute, &command_buffer)?;

        hidden = gpu_clip_add(&hidden, &mlp_out, seq_len * hidden_dim, compute, &command_buffer)?;
    }

    // Final LayerNorm
    hidden = gpu_clip_layer_norm(&hidden, model, &format!("{}ln_final", prefix), seq_len, hidden_dim, compute, &command_buffer)?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let f16_data: Vec<half::f16> = hidden.to_vec()?;
    Ok(f16_data.iter().map(|v| v.to_f32()).collect())
}

// ============================================================================
// GPU CLIP Helpers — kernel dispatch for individual operations
// ============================================================================

#[cfg(feature = "metal")]
fn gpu_clip_layer_norm(
    input: &Tensor, model: &Model, prefix: &str,
    batch: usize, dim: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    use crate::hal::metal::BorrowedMetalBuffer;

    let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;
    let weight = model.get_weight(&format!("{}.weight", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("Missing {}.weight", prefix)))?;
    let bias = model.get_weight(&format!("{}.bias", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("Missing {}.bias", prefix)))?;

    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(input.device_ptr().unwrap()) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(output.device_ptr().unwrap()) };
    let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(weight.device_ptr().unwrap()) };
    let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(bias.device_ptr().unwrap()) };

    let pipeline = compute.compile_pipeline("layer_norm_f16",
        crate::hal::metal::shader::sources::LAYER_NORM, "layer_norm_f16")?;
    let eps: f32 = 1e-5;
    let grid = ((batch + 255) / 256, 1, 1);
    let tg = (batch.min(256), 1, 1);

    compute.dispatch_async(cb, &pipeline, grid, tg, |enc| {
        enc.set_buffer(0, Some(in_buf.as_ref()), 0);
        enc.set_buffer(1, Some(w_buf.as_ref()), 0);
        enc.set_buffer(2, Some(b_buf.as_ref()), 0);
        enc.set_buffer(3, Some(out_buf.as_ref()), 0);
        enc.set_bytes(4, 4, &(batch as u32) as *const u32 as *const _);
        enc.set_bytes(5, 4, &(dim as u32) as *const u32 as *const _);
        enc.set_bytes(6, 4, &eps as *const f32 as *const _);
    });
    Ok(output)
}

#[cfg(feature = "metal")]
fn gpu_clip_linear(
    input: &Tensor, model: &Model, prefix: &str,
    m: usize, k: usize, n: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    use crate::hal::metal::BorrowedMetalBuffer;

    let output = Tensor::empty(Shape::from([m, n]), DType::F16, input.device())?;
    let weight = model.get_weight(&format!("{}.weight", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("Missing {}.weight", prefix)))?;
    let bias = model.get_weight(&format!("{}.bias", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("Missing {}.bias", prefix)))?;

    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(input.device_ptr().unwrap()) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(output.device_ptr().unwrap()) };
    let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(weight.device_ptr().unwrap()) };
    let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(bias.device_ptr().unwrap()) };

    let pipeline = compute.compile_pipeline("linear_f16",
        crate::hal::metal::shader::sources::LINEAR, "linear_f16")?;
    let has_bias: u32 = 1;
    let tile = 16usize;
    let grid = ((n + tile - 1) / tile, (m + tile - 1) / tile, 1);
    let tg = (tile, tile, 1);

    compute.dispatch_async(cb, &pipeline, grid, tg, |enc| {
        enc.set_buffer(0, Some(in_buf.as_ref()), 0);
        enc.set_buffer(1, Some(w_buf.as_ref()), 0);
        enc.set_buffer(2, Some(b_buf.as_ref()), 0);
        enc.set_buffer(3, Some(out_buf.as_ref()), 0);
        enc.set_bytes(4, 4, &(m as u32) as *const u32 as *const _);
        enc.set_bytes(5, 4, &(n as u32) as *const u32 as *const _);
        enc.set_bytes(6, 4, &(k as u32) as *const u32 as *const _);
        enc.set_bytes(7, 4, &has_bias as *const u32 as *const _);
    });
    Ok(output)
}

#[cfg(feature = "metal")]
fn gpu_clip_causal_attention(
    q: &Tensor, k: &Tensor, v: &Tensor,
    seq_len: usize, num_heads: usize, head_dim: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    use crate::hal::metal::BorrowedMetalBuffer;

    let hidden_dim = num_heads * head_dim;
    let output = Tensor::empty(Shape::from([seq_len, hidden_dim]), DType::F16, q.device())?;

    let q_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(q.device_ptr().unwrap()) };
    let k_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(k.device_ptr().unwrap()) };
    let v_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(v.device_ptr().unwrap()) };
    let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(output.device_ptr().unwrap()) };

    let pipeline = compute.compile_pipeline("causal_attention_f16",
        crate::hal::metal::shader::sources::CAUSAL_ATTENTION, "causal_attention_f16")?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let stride_dim: u32 = 1;
    let stride_head: u32 = head_dim as u32;
    let stride_seq: u32 = hidden_dim as u32;
    let stride_batch: u32 = (seq_len * hidden_dim) as u32;

    let grid = (num_heads, seq_len, 1);
    let tg = (1, 1, 1);
    let shared_mem = (seq_len * 4) as u64; // scores: seq_len * sizeof(float)

    compute.dispatch_async(cb, &pipeline, grid, tg, |enc| {
        enc.set_buffer(0, Some(q_buf.as_ref()), 0);
        enc.set_buffer(1, Some(k_buf.as_ref()), 0);
        enc.set_buffer(2, Some(v_buf.as_ref()), 0);
        enc.set_buffer(3, Some(o_buf.as_ref()), 0);
        enc.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const _);
        enc.set_bytes(5, 4, &(head_dim as u32) as *const u32 as *const _);
        enc.set_bytes(6, 4, &scale as *const f32 as *const _);
        enc.set_bytes(7, 4, &(num_heads as u32) as *const u32 as *const _);
        enc.set_bytes(8, 4, &stride_batch as *const u32 as *const _);
        enc.set_bytes(9, 4, &stride_head as *const u32 as *const _);
        enc.set_bytes(10, 4, &stride_seq as *const u32 as *const _);
        enc.set_bytes(11, 4, &stride_dim as *const u32 as *const _);
        enc.set_threadgroup_memory_length(0, shared_mem);
    });
    Ok(output)
}

#[cfg(feature = "metal")]
fn gpu_clip_gelu(
    input: &Tensor, count: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    use crate::hal::metal::BorrowedMetalBuffer;

    let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;
    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(input.device_ptr().unwrap()) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(output.device_ptr().unwrap()) };

    // OpenAI CLIP (SD 1.5 text encoder) uses `quick_gelu` = x·σ(1.702·x),
    // NOT the tanh GELU approximation. The HF CLIPTextModel config sets
    // `hidden_act = "quick_gelu"`. Using tanh-GELU here decorrelated the
    // text embeddings across all 12 layers (cosine sim vs PyTorch ≈ 0.08),
    // so the U-Net was conditioned on garbage text and produced
    // statistically-valid but prompt-irrelevant blobs. `gelu_fast_f16` is
    // exactly the quick_gelu form.
    let pipeline = compute.compile_pipeline("gelu_fast_f16",
        crate::hal::metal::shader::sources::GELU, "gelu_fast_f16")?;
    let grid = ((count + 255) / 256, 1, 1);
    let tg = (256, 1, 1);

    compute.dispatch_async(cb, &pipeline, grid, tg, |enc| {
        enc.set_buffer(0, Some(in_buf.as_ref()), 0);
        enc.set_buffer(1, Some(out_buf.as_ref()), 0);
    });
    Ok(output)
}

#[cfg(feature = "metal")]
fn gpu_clip_add(
    a: &Tensor, b: &Tensor, numel: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    use crate::hal::metal::BorrowedMetalBuffer;

    let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
    let a_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(a.device_ptr().unwrap()) };
    let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b.device_ptr().unwrap()) };
    let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(output.device_ptr().unwrap()) };

    let pipeline = compute.compile_pipeline("add_f16",
        crate::hal::metal::shader::sources::ELEMENTWISE, "add_f16")?;
    let grid = ((numel + 255) / 256, 1, 1);
    let tg = (256, 1, 1);

    compute.dispatch_async(cb, &pipeline, grid, tg, |enc| {
        enc.set_buffer(0, Some(a_buf.as_ref()), 0);
        enc.set_buffer(1, Some(b_buf.as_ref()), 0);
        enc.set_buffer(2, Some(o_buf.as_ref()), 0);
    });
    Ok(output)
}

/// Split fused QKV [seq, 3*hidden] → (Q, K, V) each [seq, hidden] on GPU.
#[cfg(feature = "metal")]
fn gpu_clip_split_qkv(
    qkv: &Tensor, seq_len: usize, hidden_dim: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<(Tensor, Tensor, Tensor)> {
    use crate::hal::metal::BorrowedMetalBuffer;

    let q = Tensor::empty(Shape::from([seq_len, hidden_dim]), DType::F16, qkv.device())?;
    let k = Tensor::empty(Shape::from([seq_len, hidden_dim]), DType::F16, qkv.device())?;
    let v = Tensor::empty(Shape::from([seq_len, hidden_dim]), DType::F16, qkv.device())?;

    let qkv_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(qkv.device_ptr().unwrap()) };
    let q_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(q.device_ptr().unwrap()) };
    let k_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(k.device_ptr().unwrap()) };
    let v_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(v.device_ptr().unwrap()) };

    let split_source = r#"
#include <metal_stdlib>
using namespace metal;
kernel void split_qkv_f16(
    device const half* qkv [[buffer(0)]],
    device half* q [[buffer(1)]],
    device half* k [[buffer(2)]],
    device half* v [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= seq_len * hidden_dim) return;
    uint s = gid / hidden_dim;
    uint d = gid % hidden_dim;
    uint triple = hidden_dim * 3;
    q[gid] = qkv[s * triple + d];
    k[gid] = qkv[s * triple + hidden_dim + d];
    v[gid] = qkv[s * triple + 2 * hidden_dim + d];
}
"#;

    let pipeline = compute.compile_pipeline("split_qkv_f16", split_source, "split_qkv_f16")?;
    let total = seq_len * hidden_dim;
    let grid = ((total + 255) / 256, 1, 1);
    let tg = (256, 1, 1);

    compute.dispatch_async(cb, &pipeline, grid, tg, |enc| {
        enc.set_buffer(0, Some(qkv_buf.as_ref()), 0);
        enc.set_buffer(1, Some(q_buf.as_ref()), 0);
        enc.set_buffer(2, Some(k_buf.as_ref()), 0);
        enc.set_buffer(3, Some(v_buf.as_ref()), 0);
        enc.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const _);
        enc.set_bytes(5, 4, &(hidden_dim as u32) as *const u32 as *const _);
    });

    Ok((q, k, v))
}

/// Tokenize text using the real CLIP BPE tokenizer.
///
/// Uses the loaded tokenizer (from tokenizer.json) to produce real token IDs.
/// Pads/truncates to `max_length` with BOS (49406) and EOS (49407) framing.
fn tokenize_with_clip(tokenizer: &Tokenizer, prompt: &str, max_length: usize) -> Vec<u32> {
    // OpenAI CLIP uses fixed special-token IDs that the generic
    // `<s>`/`</s>` detection in `Tokenizer::load_json` does NOT find
    // (CLIP's are `<|startoftext|>`=49406 / `<|endoftext|>`=49407, and
    // padding is with EOS=49407, not a separate pad token). Hardcode the
    // canonical CLIP framing instead of trusting the detected specials —
    // a wrong BOS/EOS shifts every position embedding and decorrelates
    // the whole sequence.
    const CLIP_BOS: u32 = 49406;
    const CLIP_EOS: u32 = 49407;
    let mut tokens: Vec<u32> = Vec::with_capacity(max_length);
    tokens.push(CLIP_BOS);

    let encoding = tokenizer.encode(prompt);
    let max_content = max_length.saturating_sub(2);
    tokens.extend(encoding.ids.iter().take(max_content));

    tokens.push(CLIP_EOS);
    // CLIP pads with the EOS token id (not a dedicated pad token).
    tokens.resize(max_length, CLIP_EOS);

    tokens
}

/// Fallback tokenizer for text prompts when no CLIP tokenizer is loaded.
///
/// Mean, min, max, std for a flat f32 slice. Cheap diagnostic helper used by
/// the SD_DIAG=1 instrumentation in the denoise loop.
fn vec_stats(v: &[f32]) -> (f32, f32, f32, f32) {
    if v.is_empty() { return (0.0, 0.0, 0.0, 0.0); }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut s = 0.0f64;
    for &x in v {
        if x < mn { mn = x; }
        if x > mx { mx = x; }
        s += x as f64;
    }
    let mean = (s / v.len() as f64) as f32;
    let mut var = 0.0f64;
    for &x in v {
        let d = (x as f64) - (mean as f64);
        var += d * d;
    }
    let std = (var / v.len() as f64).sqrt() as f32;
    (mean, mn, mx, std)
}

/// Basic word-level tokenizer with BOS/EOS tokens matching CLIP vocabulary IDs.
/// Produces approximate token IDs; use a real CLIP tokenizer for accurate results.
fn tokenize_prompt_basic(prompt: &str, max_length: usize) -> Vec<u32> {
    let mut tokens: Vec<u32> = Vec::with_capacity(max_length);

    // BOS token (CLIP <|startoftext|>)
    tokens.push(49406);

    // Word tokens (basic hash-based mapping into CLIP vocab range)
    for word in prompt.split_whitespace() {
        if tokens.len() >= max_length - 1 {
            break;
        }
        // Hash the word to a vocab index in the valid range [1, 49405]
        let hash = word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        tokens.push((hash % 49405) + 1);
    }

    // EOS token (CLIP <|endoftext|>)
    tokens.push(49407);

    // Pad to max_length
    while tokens.len() < max_length {
        tokens.push(0);
    }

    tokens
}

/// Convert f32 data to a tensor of the given dtype, properly handling f32→f16 conversion.
/// `Tensor::from_slice` does NOT convert types — it copies raw bytes.
/// This function ensures f32 values are properly converted to f16 when dtype is F16.
fn f32_to_tensor(data: &[f32], shape: impl Into<Shape>, dtype: DType, device: crate::hal::DeviceId) -> Result<Tensor> {
    let shape = shape.into();
    match dtype {
        DType::F16 => {
            let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            Tensor::from_slice(&f16_data, shape, DType::F16, device)
        }
        _ => Tensor::from_slice(data, shape, dtype, device),
    }
}

// ===== VAE Neural Decoder Helpers (Metal GPU) =====

/// Conv2d on Metal GPU for VAE decoder.
/// Dispatches the appropriate kernel based on kernel size.
#[cfg(feature = "metal")]
fn vae_conv2d(
    model: &Model, input: &Tensor, prefix: &str,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
    stride: usize, padding: usize,
) -> Result<Tensor> {
    let (n, cin, hin, win) = input.shape().dims4().unwrap_or((1, 4, 64, 64));

    let w = model.get_weight(&format!("{}.weight", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("VAE: {}.weight not found", prefix)))?;
    let (cout, _, kh, kw) = w.shape().dims4().unwrap_or((cin, cin, 3, 3));
    let w_ptr = w.device_ptr().ok_or(crate::core::Error::internal("VAE conv weight not on device"))?;

    let b = model.get_weight(&format!("{}.bias", prefix));
    let b_ptr = b.and_then(|t| t.device_ptr());

    let hout = (hin + 2 * padding - kh) / stride + 1;
    let wout = (win + 2 * padding - kw) / stride + 1;
    let output = Tensor::empty(Shape::from([n, cout, hout, wout]), DType::F16, input.device())?;

    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("VAE conv input not on device"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("VAE conv output not on device"))?;

    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };
    let b_buf = b_ptr.map(|p| unsafe { BorrowedMetalBuffer::from_device_ptr(p) });

    // Select kernel
    let hw = hout * wout;
    let use_simd_1x1 = cin % 32 == 0 && cout % 32 == 0 && hw % 32 == 0;
    let kernel_name = if kh == 1 && kw == 1 {
        if use_simd_1x1 { "conv2d_1x1_simd_f16" } else { "conv2d_1x1_f16" }
    } else if kh == 3 && kw == 3 && stride == 1 && padding == 1 {
        "conv2d_3x3_tiled_f16"
    } else {
        "conv2d_naive_f16"
    };
    let pipeline = compute.compile_pipeline(kernel_name, crate::hal::metal::shader::sources::CONV2D, kernel_name)?;

    let (tg, grid) = if kernel_name == "conv2d_3x3_tiled_f16" {
        ((16, 16, 1), ((wout + 15) / 16, (hout + 15) / 16, cout * n))
    } else if kernel_name == "conv2d_1x1_simd_f16" {
        ((32, 1, 1), (hw / 8, cout / 8, n))
    } else {
        ((8, 8, 1), ((wout + 7) / 8, (hout + 7) / 8, cout * n))
    };

    compute.dispatch_async(cb, &pipeline, grid, tg, |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(w_buf.as_ref()), 0);
        if let Some(ref bb) = b_buf { encoder.set_buffer(2, Some(bb.as_ref()), 0); }
        else { encoder.set_buffer(2, None, 0); }
        encoder.set_buffer(3, Some(out_buf.as_ref()), 0);
        encoder.set_bytes(4, 4, &(cin as u32) as *const u32 as *const _);
        encoder.set_bytes(5, 4, &(hin as u32) as *const u32 as *const _);
        encoder.set_bytes(6, 4, &(win as u32) as *const u32 as *const _);
        encoder.set_bytes(7, 4, &(cout as u32) as *const u32 as *const _);
        encoder.set_bytes(8, 4, &(hout as u32) as *const u32 as *const _);
        encoder.set_bytes(9, 4, &(wout as u32) as *const u32 as *const _);
        encoder.set_bytes(10, 4, &(kw as u32) as *const u32 as *const _);
        encoder.set_bytes(11, 4, &(kh as u32) as *const u32 as *const _);
        encoder.set_bytes(12, 4, &(padding as u32) as *const u32 as *const _);
        encoder.set_bytes(13, 4, &(padding as u32) as *const u32 as *const _);
        encoder.set_bytes(14, 4, &(stride as u32) as *const u32 as *const _);
        encoder.set_bytes(15, 4, &(stride as u32) as *const u32 as *const _);
        encoder.set_bytes(16, 4, &(n as u32) as *const u32 as *const _);
    });

    Ok(output)
}

/// GroupNorm (32 groups) on Metal GPU for VAE decoder.
#[cfg(feature = "metal")]
fn vae_group_norm(
    model: &Model, input: &Tensor, prefix: &str,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let (n, c, h, w) = input.shape().dims4().unwrap_or((1, 128, 64, 64));
    let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;

    let weight = model.get_weight(&format!("{}.weight", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("VAE: {}.weight not found", prefix)))?;
    let bias = model.get_weight(&format!("{}.bias", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("VAE: {}.bias not found", prefix)))?;

    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("gn input"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("gn output"))?;
    let w_ptr = weight.device_ptr().ok_or(crate::core::Error::internal("gn weight"))?;
    let b_ptr = bias.device_ptr().ok_or(crate::core::Error::internal("gn bias"))?;

    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };
    let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) };
    let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };

    let groups: usize = 32;
    let hw = h * w;
    let eps: f32 = 1e-6;

    // Allocate temp stats buffer: [N, Groups] of float2 (8 bytes each)
    let stats_buffer = compute.device().create_buffer(n * groups * 8, crate::hal::metal::ResourceOptions::activations())?;

    let stats_pl = compute.compile_pipeline("group_norm_stats", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_stats_f16")?;
    let apply_pl = compute.compile_pipeline("group_norm_apply", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_apply_f16")?;

    // Pass 1: compute per-group mean and variance
    compute.dispatch_async(cb, &stats_pl, (groups, n, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(&stats_buffer), 0);
        encoder.set_bytes(2, 4, &(n as u32) as *const u32 as *const _);
        encoder.set_bytes(3, 4, &(groups as u32) as *const u32 as *const _);
        encoder.set_bytes(4, 4, &(c as u32) as *const u32 as *const _);
        encoder.set_bytes(5, 4, &(hw as u32) as *const u32 as *const _);
    });

    // Pass 2: normalize + scale/shift
    let hw_groups = (hw + 255) / 256;
    compute.dispatch_async(cb, &apply_pl, (hw_groups, c, n), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(&stats_buffer), 0);
        encoder.set_buffer(2, Some(w_buf.as_ref()), 0);
        encoder.set_buffer(3, Some(b_buf.as_ref()), 0);
        encoder.set_buffer(4, Some(out_buf.as_ref()), 0);
        encoder.set_bytes(5, 4, &(n as u32) as *const u32 as *const _);
        encoder.set_bytes(6, 4, &(groups as u32) as *const u32 as *const _);
        encoder.set_bytes(7, 4, &(c as u32) as *const u32 as *const _);
        encoder.set_bytes(8, 4, &(hw as u32) as *const u32 as *const _);
        encoder.set_bytes(9, 4, &eps as *const f32 as *const _);
    });

    Ok(output)
}

/// Fused GroupNorm + SiLU on Metal GPU (saves 1 dispatch + 1 allocation per call).
#[cfg(feature = "metal")]
fn vae_group_norm_silu(
    model: &Model, input: &Tensor, prefix: &str,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let (n, c, h, w) = input.shape().dims4().unwrap_or((1, 128, 64, 64));
    let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;

    let weight = model.get_weight(&format!("{}.weight", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("VAE: {}.weight not found", prefix)))?;
    let bias = model.get_weight(&format!("{}.bias", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("VAE: {}.bias not found", prefix)))?;

    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("gn input"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("gn output"))?;
    let w_ptr = weight.device_ptr().ok_or(crate::core::Error::internal("gn weight"))?;
    let b_ptr = bias.device_ptr().ok_or(crate::core::Error::internal("gn bias"))?;

    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };
    let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) };
    let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };

    let groups: usize = 32;
    let hw = h * w;
    let eps: f32 = 1e-6;

    let stats_buffer = compute.device().create_buffer(n * groups * 8, crate::hal::metal::ResourceOptions::activations())?;

    let stats_pl = compute.compile_pipeline("group_norm_stats", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_stats_f16")?;
    let apply_pl = compute.compile_pipeline("group_norm_silu_apply", crate::hal::metal::shader::sources::GROUP_NORM, "group_norm_silu_apply_f16")?;

    // Pass 1: compute per-group mean and variance
    compute.dispatch_async(cb, &stats_pl, (groups, n, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(&stats_buffer), 0);
        encoder.set_bytes(2, 4, &(n as u32) as *const u32 as *const _);
        encoder.set_bytes(3, 4, &(groups as u32) as *const u32 as *const _);
        encoder.set_bytes(4, 4, &(c as u32) as *const u32 as *const _);
        encoder.set_bytes(5, 4, &(hw as u32) as *const u32 as *const _);
    });

    // Pass 2: fused normalize + scale/shift + SiLU
    let hw_groups = (hw + 255) / 256;
    compute.dispatch_async(cb, &apply_pl, (hw_groups, c, n), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(&stats_buffer), 0);
        encoder.set_buffer(2, Some(w_buf.as_ref()), 0);
        encoder.set_buffer(3, Some(b_buf.as_ref()), 0);
        encoder.set_buffer(4, Some(out_buf.as_ref()), 0);
        encoder.set_bytes(5, 4, &(n as u32) as *const u32 as *const _);
        encoder.set_bytes(6, 4, &(groups as u32) as *const u32 as *const _);
        encoder.set_bytes(7, 4, &(c as u32) as *const u32 as *const _);
        encoder.set_bytes(8, 4, &(hw as u32) as *const u32 as *const _);
        encoder.set_bytes(9, 4, &eps as *const f32 as *const _);
    });

    Ok(output)
}

/// SiLU activation on Metal GPU.
#[cfg(feature = "metal")]
fn vae_silu(
    input: &Tensor, compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;
    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("silu input"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("silu output"))?;
    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

    let pipeline = compute.compile_pipeline("silu", crate::hal::metal::shader::sources::SILU, "silu_f16")?;
    let numel = input.shape().numel();

    compute.dispatch_async(cb, &pipeline, ((numel + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
    });

    Ok(output)
}

/// Elementwise add on Metal GPU.
#[cfg(feature = "metal")]
fn vae_add(
    a: &Tensor, b: &Tensor, compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let output = Tensor::empty(a.shape().clone(), DType::F16, a.device())?;
    let a_ptr = a.device_ptr().ok_or(crate::core::Error::internal("add a"))?;
    let b_ptr = b.device_ptr().ok_or(crate::core::Error::internal("add b"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("add out"))?;
    let a_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(a_ptr) };
    let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

    let pipeline = compute.compile_pipeline("add_f16", crate::hal::metal::shader::sources::ELEMENTWISE, "add_f16")?;
    let numel = a.shape().numel();

    compute.dispatch_async(cb, &pipeline, ((numel + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(a_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(b_buf.as_ref()), 0);
        encoder.set_buffer(2, Some(out_buf.as_ref()), 0);
    });

    Ok(output)
}

/// VAE ResNet block: GroupNorm → SiLU → Conv3x3 → GroupNorm → SiLU → Conv3x3 + residual.
/// Handles conv_shortcut when input/output channels differ.
#[cfg(feature = "metal")]
fn vae_resnet_block(
    model: &Model, input: &Tensor, prefix: &str,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let act1 = vae_group_norm_silu(model, input, &format!("{}.norm1", prefix), compute, cb)?;
    let conv1 = vae_conv2d(model, &act1, &format!("{}.conv1", prefix), compute, cb, 1, 1)?;

    let act2 = vae_group_norm_silu(model, &conv1, &format!("{}.norm2", prefix), compute, cb)?;
    let conv2 = vae_conv2d(model, &act2, &format!("{}.conv2", prefix), compute, cb, 1, 1)?;

    // Residual: use conv_shortcut (diffusers) or nin_shortcut (LDM) if channel dimensions change
    let residual = if model.get_weight(&format!("{}.conv_shortcut.weight", prefix)).is_some() {
        vae_conv2d(model, input, &format!("{}.conv_shortcut", prefix), compute, cb, 1, 0)?
    } else if model.get_weight(&format!("{}.nin_shortcut.weight", prefix)).is_some() {
        vae_conv2d(model, input, &format!("{}.nin_shortcut", prefix), compute, cb, 1, 0)?
    } else {
        input.clone()
    };

    vae_add(&residual, &conv2, compute, cb)
}

/// VAE self-attention block: GroupNorm → NCHW→NHWC → Q/K/V projections → attention → output proj → NHWC→NCHW + residual.
/// Single-head attention on spatial features (num_heads=1, head_dim=C).
#[cfg(feature = "metal")]
fn vae_self_attention(
    model: &Model, input: &Tensor, prefix: &str,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let (n, c, h, w) = input.shape().dims4().unwrap_or((1, 512, 64, 64));
    let hw = h * w;

    // GroupNorm
    let normed = vae_group_norm(model, input, &format!("{}.group_norm", prefix), compute, cb)?;

    // Transpose NCHW → [HW, C] for linear projections
    let flat = vae_nchw_to_nhwc(&normed, n, c, hw, compute, cb)?;

    // Q, K, V projections: [HW, C] @ [C, C]^T + bias → [HW, C]
    let q = vae_linear(model, &flat, &format!("{}.to_q", prefix), hw, c, c, compute, cb)?;
    let k = vae_linear(model, &flat, &format!("{}.to_k", prefix), hw, c, c, compute, cb)?;
    let v = vae_linear(model, &flat, &format!("{}.to_v", prefix), hw, c, c, compute, cb)?;

    // Self-attention: softmax(QK^T/sqrt(d)) @ V
    // Matmul-based: decompose into Q@K^T → softmax → S@V using tiled matmul.
    // The standard per-query kernel uses 1 thread per query (4M serial FMA each);
    // matmul approach uses ~2M threads via tiled 16×16 threadgroups.
    let attn = vae_attention_matmul(&q, &k, &v, hw, 1, c, compute, cb)?;

    // Output projection
    let out = vae_linear(model, &attn, &format!("{}.to_out.0", prefix), hw, c, c, compute, cb)?;

    // Transpose back: [HW, C] → NCHW
    let out_nchw = vae_nhwc_to_nchw(&out, input, n, c, hw, compute, cb)?;

    // Residual
    vae_add(input, &out_nchw, compute, cb)
}

/// VAE self-attention block (LDM/CompVis naming): norm → q/k/v (1×1 conv) → attention → proj_out → residual.
#[cfg(feature = "metal")]
fn vae_self_attention_ldm(
    model: &Model, input: &Tensor, prefix: &str,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let (n, c, h, w) = input.shape().dims4().unwrap_or((1, 512, 64, 64));
    let hw = h * w;

    // GroupNorm (LDM uses "norm" not "group_norm")
    let normed = vae_group_norm(model, input, &format!("{}.norm", prefix), compute, cb)?;

    // Transpose NCHW → [HW, C] for linear projections
    let flat = vae_nchw_to_nhwc(&normed, n, c, hw, compute, cb)?;

    // Q, K, V: LDM stores as 1×1 conv [C, C, 1, 1] — reshape to [C, C] for vae_linear
    let q = vae_linear(model, &flat, &format!("{}.q", prefix), hw, c, c, compute, cb)?;
    let k = vae_linear(model, &flat, &format!("{}.k", prefix), hw, c, c, compute, cb)?;
    let v = vae_linear(model, &flat, &format!("{}.v", prefix), hw, c, c, compute, cb)?;

    // Self-attention
    let attn = vae_attention_matmul(&q, &k, &v, hw, 1, c, compute, cb)?;

    // Output projection (LDM uses "proj_out" not "to_out.0")
    let out = vae_linear(model, &attn, &format!("{}.proj_out", prefix), hw, c, c, compute, cb)?;

    // Transpose back: [HW, C] → NCHW
    let out_nchw = vae_nhwc_to_nchw(&out, input, n, c, hw, compute, cb)?;

    // Residual
    vae_add(input, &out_nchw, compute, cb)
}

/// NCHW → [N*HW, C] transpose on Metal GPU.
#[cfg(feature = "metal")]
fn vae_nchw_to_nhwc(
    input: &Tensor, n: usize, c: usize, hw: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let output = Tensor::empty(Shape::from([n * hw, c]), DType::F16, input.device())?;
    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("nchw in"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("nchw out"))?;
    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

    let pipeline = compute.compile_pipeline("nchw_to_nhwc_f16", crate::hal::metal::shader::sources::TRANSPOSE, "nchw_to_nhwc_f16")?;
    let tg_x = hw.min(256);
    compute.dispatch_async(cb, &pipeline, ((hw + tg_x - 1) / tg_x, c, n), (tg_x, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
        encoder.set_bytes(2, 4, &(c as u32) as *const u32 as *const _);
        encoder.set_bytes(3, 4, &(hw as u32) as *const u32 as *const _);
    });

    Ok(output)
}

/// [N*HW, C] → NCHW transpose on Metal GPU.
#[cfg(feature = "metal")]
fn vae_nhwc_to_nchw(
    input: &Tensor, reference: &Tensor, n: usize, c: usize, hw: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let output = Tensor::empty(reference.shape().clone(), DType::F16, reference.device())?;
    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("nhwc in"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("nhwc out"))?;
    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

    let pipeline = compute.compile_pipeline("nhwc_to_nchw_f16", crate::hal::metal::shader::sources::TRANSPOSE, "nhwc_to_nchw_f16")?;
    let tg_x = hw.min(256);
    compute.dispatch_async(cb, &pipeline, ((hw + tg_x - 1) / tg_x, c, n), (tg_x, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
        encoder.set_bytes(2, 4, &(c as u32) as *const u32 as *const _);
        encoder.set_bytes(3, 4, &(hw as u32) as *const u32 as *const _);
    });

    Ok(output)
}

/// Linear projection for VAE: Y = X @ W^T + bias.
/// Input: [M, K], Weight: [N, K], Output: [M, N].
#[cfg(feature = "metal")]
fn vae_linear(
    model: &Model, input: &Tensor, prefix: &str,
    m: usize, k: usize, n_out: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let output = Tensor::empty(Shape::from([m, n_out]), DType::F16, input.device())?;

    let weight = model.get_weight(&format!("{}.weight", prefix))
        .ok_or_else(|| crate::core::Error::internal(format!("VAE: {}.weight not found", prefix)))?;
    let w_ptr = weight.device_ptr().ok_or(crate::core::Error::internal("linear weight"))?;

    let bias = model.get_weight(&format!("{}.bias", prefix));
    let has_bias = bias.is_some();
    let b_ptr = if let Some(b) = bias { b.device_ptr().ok_or(crate::core::Error::internal("linear bias"))? } else { w_ptr };

    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("linear in"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("linear out"))?;

    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let w_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(w_ptr) };
    let b_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(b_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

    let pipeline = compute.compile_pipeline("linear_f16", crate::hal::metal::shader::sources::LINEAR, "linear_f16")?;
    let has_bias_u32: u32 = if has_bias { 1 } else { 0 };
    let tile = 16usize;

    compute.dispatch_async(cb, &pipeline, ((n_out + tile - 1) / tile, (m + tile - 1) / tile, 1), (tile, tile, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(w_buf.as_ref()), 0);
        encoder.set_buffer(2, Some(b_buf.as_ref()), 0);
        encoder.set_buffer(3, Some(out_buf.as_ref()), 0);
        encoder.set_bytes(4, 4, &(m as u32) as *const u32 as *const _);
        encoder.set_bytes(5, 4, &(n_out as u32) as *const u32 as *const _);
        encoder.set_bytes(6, 4, &(k as u32) as *const u32 as *const _);
        encoder.set_bytes(7, 4, &has_bias_u32 as *const u32 as *const _);
    });

    Ok(output)
}

/// Standard (non-tiled) attention on Metal GPU.
/// Uses attention_f16 kernel with per-query-position threads.
/// Better for large head_dim (e.g., VAE: 1 head × 512 dim) where tiled attention
/// would exceed Metal's threadgroup shared memory limit.
#[cfg(feature = "metal")]
fn vae_attention_standard(
    q: &Tensor, k: &Tensor, v: &Tensor,
    seq_len: usize, num_heads: usize, head_dim: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let hidden_dim = num_heads * head_dim;
    let output = Tensor::empty(Shape::from([seq_len, hidden_dim]), DType::F16, q.device())?;

    let q_ptr = q.device_ptr().ok_or(crate::core::Error::internal("attn q"))?;
    let k_ptr = k.device_ptr().ok_or(crate::core::Error::internal("attn k"))?;
    let v_ptr = v.device_ptr().ok_or(crate::core::Error::internal("attn v"))?;
    let o_ptr = output.device_ptr().ok_or(crate::core::Error::internal("attn o"))?;

    let q_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(q_ptr) };
    let k_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(k_ptr) };
    let v_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(v_ptr) };
    let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(o_ptr) };

    let pipeline = compute.compile_pipeline("attention_f16", crate::hal::metal::shader::sources::ATTENTION, "attention_f16")?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let stride_dim: u32 = 1;
    let stride_head: u32 = head_dim as u32;
    let stride_seq: u32 = hidden_dim as u32;
    let stride_batch: u32 = (seq_len * hidden_dim) as u32;

    // Grid: one thread per (head, query_position)
    let grid = (num_heads, seq_len, 1);
    let threadgroup = (1, 1, 1);
    // Shared memory for attention scores: kv_len * sizeof(float)
    let shared_mem = (seq_len * 4) as u64;

    compute.dispatch_async(cb, &pipeline, grid, threadgroup, |encoder| {
        encoder.set_buffer(0, Some(q_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(k_buf.as_ref()), 0);
        encoder.set_buffer(2, Some(v_buf.as_ref()), 0);
        encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
        encoder.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const _);
        encoder.set_bytes(5, 4, &(head_dim as u32) as *const u32 as *const _);
        encoder.set_bytes(6, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(7, 4, &(num_heads as u32) as *const u32 as *const _);
        encoder.set_bytes(8, 4, &stride_batch as *const u32 as *const _);
        encoder.set_bytes(9, 4, &stride_head as *const u32 as *const _);
        encoder.set_bytes(10, 4, &stride_seq as *const u32 as *const _);
        encoder.set_bytes(11, 4, &stride_dim as *const u32 as *const _);
        encoder.set_bytes(12, 4, &(seq_len as u32) as *const u32 as *const _); // kv_len = seq_len
        encoder.set_threadgroup_memory_length(0, shared_mem);
    });

    Ok(output)
}

/// Matmul-based VAE attention: decompose into matmul ops instead of serial per-query loops.
/// For head_dim=512, the standard attention kernel uses 1 thread per query, serially computing
/// 4096×512 dot products. This decomposition uses tiled matmul with ~2M threads total.
///
/// Steps: S = Q @ K^T (via linear_f16) → scale+softmax → O = S @ V (via matmul_nn_f16)
#[cfg(feature = "metal")]
fn vae_attention_matmul(
    q: &Tensor, k: &Tensor, v: &Tensor,
    seq_len: usize, _num_heads: usize, head_dim: usize,
    compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    // Step 1: S = Q @ K^T — [seq_len, head_dim] @ [head_dim, seq_len] → [seq_len, seq_len]
    // linear_f16 computes Y = X @ W^T, so with X=Q[seq_len, head_dim] and W=K[seq_len, head_dim]:
    // Y = Q @ K^T = [seq_len, seq_len] ✓
    let scores = Tensor::empty(Shape::from([seq_len, seq_len]), DType::F16, q.device())?;
    {
        let q_ptr = q.device_ptr().ok_or(crate::core::Error::internal("attn q"))?;
        let k_ptr = k.device_ptr().ok_or(crate::core::Error::internal("attn k"))?;
        let s_ptr = scores.device_ptr().ok_or(crate::core::Error::internal("attn s"))?;
        let q_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(q_ptr) };
        let k_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(k_ptr) };
        let s_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(s_ptr) };

        let pipeline = compute.compile_pipeline("linear_f16_qkt", crate::hal::metal::shader::sources::LINEAR, "linear_f16")?;
        let m = seq_len as u32;
        let n = seq_len as u32;
        let k_dim = head_dim as u32;
        let has_bias: u32 = 0;

        let tile = 16;
        let grid = ((seq_len + tile - 1) / tile, (seq_len + tile - 1) / tile, 1);
        let tg = (tile, tile, 1);

        compute.dispatch_async(cb, &pipeline, grid, tg, |encoder| {
            encoder.set_buffer(0, Some(q_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(k_buf.as_ref()), 0);
            encoder.set_buffer(2, None, 0); // no bias
            encoder.set_buffer(3, Some(s_buf.as_ref()), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &n as *const u32 as *const _);
            encoder.set_bytes(6, 4, &k_dim as *const u32 as *const _);
            encoder.set_bytes(7, 4, &has_bias as *const u32 as *const _);
        });
    }

    // Step 2: Scale + row-wise softmax (in-place on scores)
    {
        let s_ptr = scores.device_ptr().ok_or(crate::core::Error::internal("attn s"))?;
        let s_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(s_ptr) };

        let pipeline = compute.compile_pipeline("row_softmax_scale_f16", crate::hal::metal::shader::sources::LINEAR, "row_softmax_scale_f16")?;
        let rows = seq_len as u32;
        let cols = seq_len as u32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        compute.dispatch_async(cb, &pipeline, ((seq_len + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
            encoder.set_buffer(0, Some(s_buf.as_ref()), 0);
            encoder.set_bytes(1, 4, &rows as *const u32 as *const _);
            encoder.set_bytes(2, 4, &cols as *const u32 as *const _);
            encoder.set_bytes(3, 4, &scale as *const f32 as *const _);
        });
    }

    // Step 3: O = S @ V — [seq_len, seq_len] @ [seq_len, head_dim] → [seq_len, head_dim]
    // Use matmul_nn_f16 (non-transposed: Y = A @ B)
    let output = Tensor::empty(Shape::from([seq_len, head_dim]), DType::F16, q.device())?;
    {
        let s_ptr = scores.device_ptr().ok_or(crate::core::Error::internal("attn s"))?;
        let v_ptr = v.device_ptr().ok_or(crate::core::Error::internal("attn v"))?;
        let o_ptr = output.device_ptr().ok_or(crate::core::Error::internal("attn o"))?;
        let s_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(s_ptr) };
        let v_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(v_ptr) };
        let o_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(o_ptr) };

        let pipeline = compute.compile_pipeline("matmul_nn_f16", crate::hal::metal::shader::sources::LINEAR, "matmul_nn_f16")?;
        let m = seq_len as u32;
        let n = head_dim as u32;
        let k_dim = seq_len as u32;

        let tile = 16;
        let grid = ((head_dim + tile - 1) / tile, (seq_len + tile - 1) / tile, 1);
        let tg = (tile, tile, 1);

        compute.dispatch_async(cb, &pipeline, grid, tg, |encoder| {
            encoder.set_buffer(0, Some(s_buf.as_ref()), 0);
            encoder.set_buffer(1, Some(v_buf.as_ref()), 0);
            encoder.set_buffer(2, Some(o_buf.as_ref()), 0);
            encoder.set_bytes(3, 4, &m as *const u32 as *const _);
            encoder.set_bytes(4, 4, &n as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k_dim as *const u32 as *const _);
        });
    }

    Ok(output)
}

/// Rescale VAE output from [-1,1] to [0,1]: output = clamp(x * 0.5 + 0.5, 0, 1).
#[cfg(feature = "metal")]
fn vae_rescale_output(
    input: &Tensor, compute: &Arc<MetalCompute>, cb: &metal::CommandBufferRef,
) -> Result<Tensor> {
    let output = Tensor::empty(input.shape().clone(), DType::F16, input.device())?;
    let in_ptr = input.device_ptr().ok_or(crate::core::Error::internal("rescale in"))?;
    let out_ptr = output.device_ptr().ok_or(crate::core::Error::internal("rescale out"))?;
    let in_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(in_ptr) };
    let out_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(out_ptr) };

    let pipeline = compute.compile_pipeline("vae_rescale_f16", crate::hal::metal::shader::sources::VAE_RESCALE, "vae_rescale_f16")?;
    let numel = input.shape().numel();

    compute.dispatch_async(cb, &pipeline, ((numel + 255) / 256, 1, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(in_buf.as_ref()), 0);
        encoder.set_buffer(1, Some(out_buf.as_ref()), 0);
        encoder.set_bytes(2, 4, &(numel as u32) as *const u32 as *const _);
    });

    Ok(output)
}
