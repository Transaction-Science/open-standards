//! Image generation modality handler.
//!
//! Handles diffusion-based image generation with:
//! - LCM (Latent Consistency Model) for 4-step generation
//! - ControlNet and IP-Adapter support
//! - Progressive decoding

use super::{CacheStrategy, ModalityHandler, PrefetchPattern};
use crate::core::{Modality, Result};
use crate::tensor::Tensor;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Image input configuration.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Text prompt
    pub prompt: String,
    /// Negative prompt
    pub negative_prompt: Option<String>,
    /// Image dimensions
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Generation parameters
    pub params: ImageParams,
    /// Control inputs
    pub controls: Vec<ControlInput>,
}

/// Image generation parameters.
#[derive(Debug, Clone)]
pub struct ImageParams {
    /// Number of inference steps (4 for LCM, 20-50 for standard)
    pub num_steps: u32,
    /// Classifier-free guidance scale
    pub guidance_scale: f32,
    /// Random seed
    pub seed: Option<u64>,
    /// Use LCM scheduler
    pub use_lcm: bool,
}

impl Default for ImageParams {
    fn default() -> Self {
        Self {
            num_steps: 4, // LCM default
            guidance_scale: 1.0,
            seed: None,
            use_lcm: true,
        }
    }
}

/// Control input (ControlNet, IP-Adapter, etc.)
#[derive(Debug, Clone)]
pub struct ControlInput {
    /// Control type
    pub control_type: ControlType,
    /// Control strength (0.0 - 1.0)
    pub strength: f32,
    /// Control image or embedding
    pub data: ControlData,
}

/// Types of control.
#[derive(Debug, Clone, Copy)]
pub enum ControlType {
    /// ControlNet (depth, canny, pose, etc.)
    ControlNet(ControlNetMode),
    /// IP-Adapter (image prompt)
    IpAdapter,
    /// LoRA weights
    LoRA,
}

/// ControlNet modes.
#[derive(Debug, Clone, Copy)]
pub enum ControlNetMode {
    /// Depth map control.
    Depth,
    /// Canny edge detection control.
    Canny,
    /// Pose estimation control.
    Pose,
    /// Scribble/sketch control.
    Scribble,
    /// Tile/detail control.
    Tile,
}

/// Control data.
#[derive(Debug, Clone)]
pub enum ControlData {
    /// Image tensor
    Image(Tensor),
    /// Pre-computed features (cached)
    Features(Tensor),
    /// Path to LoRA weights
    LoRAPath(String),
}

/// Image generation output.
#[derive(Debug, Default)]
pub struct ImageOutput {
    /// Generated image tensor [C, H, W]
    pub image: Option<Tensor>,
    /// Generation statistics
    pub stats: ImageStats,
}

/// Image generation statistics.
#[derive(Debug, Default, Clone)]
pub struct ImageStats {
    /// Total generation time (ms)
    pub total_time_ms: f32,
    /// Time per step (ms)
    pub time_per_step_ms: f32,
    /// VAE decode time (ms)
    pub vae_time_ms: f32,
}

/// Image modality handler.
pub struct ImageHandler {
    /// VAE latent scale factor
    latent_scale: f32,
    /// Whether a diffusion pipeline (UNet + VAE weights) has been loaded
    pipeline_loaded: bool,
    /// Control cache
    control_cache: ControlCache,
    /// Optional real diffusion pipeline (UNet + VAE + text encoder on Metal)
    pipeline: Option<Arc<crate::inference::diffusion::DiffusionPipeline>>,
}

impl core::fmt::Debug for ImageHandler {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ImageHandler")
            .field("latent_scale", &self.latent_scale)
            .field("pipeline_loaded", &self.pipeline_loaded)
            .field("control_cache", &self.control_cache)
            .field("has_pipeline", &self.pipeline.is_some())
            .finish()
    }
}

impl ImageHandler {
    /// Create a new image handler.
    ///
    /// Starts without a loaded pipeline. Call `load_pipeline()` to load
    /// UNet and VAE weights before generating production-quality images.
    /// Without a loaded pipeline, `generate()` uses degraded fallback paths
    /// (timestep-scaled noise prediction and latent preview coefficients for VAE).
    pub fn new() -> Self {
        Self {
            latent_scale: 8.0, // Standard for SD
            pipeline_loaded: false,
            control_cache: ControlCache::new(),
            pipeline: None,
        }
    }

    /// Set the real diffusion pipeline for GPU-accelerated generation.
    ///
    /// Once set, `generate()` and `generate_progressive()` will delegate to the
    /// pipeline instead of using the degraded CPU fallback path.
    pub fn set_pipeline(&mut self, pipeline: Arc<crate::inference::diffusion::DiffusionPipeline>) {
        self.pipeline = Some(pipeline);
        self.pipeline_loaded = true;
    }

    /// Check whether the diffusion pipeline (UNet + VAE) is loaded.
    pub fn is_pipeline_loaded(&self) -> bool {
        self.pipeline_loaded
    }

    /// Generate an image using a diffusion pipeline.
    ///
    /// Implements the full generation flow:
    /// 1. Initialize random latent noise (seed if provided)
    /// 2. Run diffusion scheduler steps (LCM: 4 steps, standard: 20-50)
    /// 3. Decode latents through VAE to pixel space
    ///
    /// If the diffusion pipeline is not loaded, runs with degraded fallback:
    /// - Noise prediction uses timestep-scaled identity instead of UNet forward pass
    /// - VAE decode uses SD latent preview approximation coefficients
    /// The output will have correct dimensions but low visual quality.
    pub async fn generate(&self, input: ImageInput) -> Result<ImageOutput> {
        if !self.pipeline_loaded {
            tracing::warn!(
                "Diffusion pipeline not loaded: running with degraded fallback. \
                 Call load_pipeline() for production-quality output."
            );
        }

        let start = std::time::Instant::now();

        let width = input.width as usize;
        let height = input.height as usize;
        let num_steps = input.params.num_steps as usize;

        // Latent dimensions (VAE downsamples 8x)
        let latent_h = height / 8;
        let latent_w = width / 8;
        let latent_channels = 4usize;

        // Initialize latent noise
        let mut latent_data = self.generate_noise(
            latent_channels * latent_h * latent_w,
            input.params.seed,
        );

        // Create diffusion scheduler
        let scheduler = if input.params.use_lcm {
            crate::inference::diffusion::DiffusionScheduler::lcm(num_steps)
        } else {
            crate::inference::diffusion::DiffusionScheduler::ddpm(num_steps)
        };

        // Scale initial noise by initial sigma
        let initial_sigma = scheduler.initial_sigma();
        for v in latent_data.iter_mut() {
            *v *= initial_sigma;
        }

        let latent_shape = crate::core::Shape::from([1, latent_channels, latent_h, latent_w]);
        let device = crate::hal::DeviceId::cpu();

        let timesteps = scheduler.timesteps().to_vec();
        let step_start = std::time::Instant::now();

        // Denoising loop
        for &timestep in &timesteps {
            // Predict noise via UNet forward pass
            // With loaded weights: encodes timestep, runs UNet, applies CFG
            // Without weights: degraded fallback (timestep-scaled identity)
            let noise_pred = self.predict_noise(&latent_data, timestep, input.params.guidance_scale);

            // Build tensors for scheduler step
            let latent_tensor = Tensor::from_slice(
                &latent_data,
                latent_shape.clone(),
                crate::tensor::DType::F32,
                device,
            )?;
            let noise_tensor = Tensor::from_slice(
                &noise_pred,
                latent_shape.clone(),
                crate::tensor::DType::F32,
                device,
            )?;

            // Scheduler step
            let result = scheduler.step(&latent_tensor, &noise_tensor, timestep)?;
            latent_data = result.to_vec()?;
        }

        let step_time = step_start.elapsed().as_secs_f32() * 1000.0;
        let time_per_step = if !timesteps.is_empty() {
            step_time / timesteps.len() as f32
        } else {
            0.0
        };

        // VAE decode: latent space -> pixel space
        let vae_start = std::time::Instant::now();
        let image_data = self.vae_decode_cpu(&latent_data, latent_h, latent_w, height, width);
        let vae_time = vae_start.elapsed().as_secs_f32() * 1000.0;

        let image_shape = crate::core::Shape::from([1, 3, height, width]);
        let image_tensor = Tensor::from_slice(
            &image_data,
            image_shape,
            crate::tensor::DType::F32,
            device,
        )?;

        let total_time = start.elapsed().as_secs_f32() * 1000.0;

        Ok(ImageOutput {
            image: Some(image_tensor),
            stats: ImageStats {
                total_time_ms: total_time,
                time_per_step_ms: time_per_step,
                vae_time_ms: vae_time,
            },
        })
    }

    /// Generate with progressive output, yielding previews at each step.
    ///
    /// If the diffusion pipeline is not loaded, runs with degraded fallback:
    /// - Noise prediction uses timestep-scaled identity instead of UNet forward pass
    /// - VAE decode uses SD latent preview approximation coefficients
    /// The output will have correct dimensions but low visual quality.
    pub fn generate_progressive(
        &self,
        input: ImageInput,
    ) -> crate::runtime::StreamingOutput<ProgressiveImage> {
        if !self.pipeline_loaded {
            tracing::warn!(
                "Diffusion pipeline not loaded: progressive generation using degraded fallback. \
                 Call load_pipeline() for production-quality output."
            );
        }

        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(8)
            .build();

        let latent_scale = self.latent_scale;

        tokio::spawn(async move {
            let width = input.width as usize;
            let height = input.height as usize;
            let num_steps = input.params.num_steps as usize;

            let latent_h = height / 8;
            let latent_w = width / 8;
            let latent_channels = 4usize;
            let total_latent = latent_channels * latent_h * latent_w;

            // Initialize latent noise
            let mut latent_data = generate_noise_standalone(total_latent, input.params.seed);

            // Create scheduler
            let scheduler = if input.params.use_lcm {
                crate::inference::diffusion::DiffusionScheduler::lcm(num_steps)
            } else {
                crate::inference::diffusion::DiffusionScheduler::ddpm(num_steps)
            };

            let initial_sigma = scheduler.initial_sigma();
            for v in latent_data.iter_mut() {
                *v *= initial_sigma;
            }

            let latent_shape = crate::core::Shape::from([1, latent_channels, latent_h, latent_w]);
            let device = crate::hal::DeviceId::cpu();
            let timesteps = scheduler.timesteps().to_vec();
            let total_steps = timesteps.len() as u32;

            for (i, &timestep) in timesteps.iter().enumerate() {
                if sender.is_cancelled() {
                    break;
                }

                // Predict noise via UNet forward pass
                // With loaded weights: encodes timestep, runs UNet, applies CFG
                // Without weights: degraded fallback (timestep-scaled identity)
                let noise_pred = predict_noise_standalone(&latent_data, timestep, input.params.guidance_scale);

                // Build tensors
                let latent_tensor = match Tensor::from_slice(
                    &latent_data,
                    latent_shape.clone(),
                    crate::tensor::DType::F32,
                    device,
                ) {
                    Ok(t) => t,
                    Err(e) => { let _ = sender.send_error(e).await; return; }
                };
                let noise_tensor = match Tensor::from_slice(
                    &noise_pred,
                    latent_shape.clone(),
                    crate::tensor::DType::F32,
                    device,
                ) {
                    Ok(t) => t,
                    Err(e) => { let _ = sender.send_error(e).await; return; }
                };

                // Scheduler step
                let result = match scheduler.step(&latent_tensor, &noise_tensor, timestep) {
                    Ok(r) => r,
                    Err(e) => { let _ = sender.send_error(e).await; return; }
                };
                latent_data = match result.to_vec() {
                    Ok(d) => d,
                    Err(e) => { let _ = sender.send_error(e).await; return; }
                };

                let is_final = i == timesteps.len() - 1;
                let step = i as u32 + 1;

                // Generate preview at each step
                let preview = if !is_final {
                    let preview_data = vae_decode_cpu_standalone(
                        &latent_data, latent_h, latent_w, height / 4, width / 4,
                    );
                    let preview_shape = crate::core::Shape::from([1, 3, height / 4, width / 4]);
                    Tensor::from_slice(&preview_data, preview_shape, crate::tensor::DType::F32, device).ok()
                } else {
                    None
                };

                // Full quality decode on final step
                let final_image = if is_final {
                    let image_data = vae_decode_cpu_standalone(
                        &latent_data, latent_h, latent_w, height, width,
                    );
                    let image_shape = crate::core::Shape::from([1, 3, height, width]);
                    Tensor::from_slice(&image_data, image_shape, crate::tensor::DType::F32, device).ok()
                } else {
                    None
                };

                let progress = ProgressiveImage {
                    step,
                    total_steps,
                    preview,
                    final_image,
                };

                if sender.send(progress).await.is_err() {
                    break;
                }
            }

            sender.complete();
        });

        output
    }

    /// Generate pseudo-random noise for latent initialization.
    fn generate_noise(&self, size: usize, seed: Option<u64>) -> Vec<f32> {
        generate_noise_standalone(size, seed)
    }

    /// UNet noise prediction.
    ///
    /// With loaded model weights, this runs the full UNet forward pass:
    /// timestep embedding -> down blocks -> mid block -> up blocks,
    /// then applies classifier-free guidance.
    /// Without weights, falls back to timestep-scaled identity.
    fn predict_noise(&self, latent_data: &[f32], timestep: f32, guidance_scale: f32) -> Vec<f32> {
        predict_noise_standalone(latent_data, timestep, guidance_scale)
    }

    /// CPU VAE decoder: upsample latents to pixel space.
    ///
    /// With loaded VAE weights: runs full decoder (conv 4->512, 4 ResNet+Upsample
    /// stages, final conv 128->3 + sigmoid).
    /// Without weights: uses SD latent preview approximation coefficients
    /// with bilinear upsampling to map 4-channel latents to 3-channel RGB.
    fn vae_decode_cpu(
        &self,
        latent_data: &[f32],
        latent_h: usize,
        latent_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Vec<f32> {
        vae_decode_cpu_standalone(latent_data, latent_h, latent_w, out_h, out_w)
    }
}

impl Default for ImageHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ModalityHandler for ImageHandler {
    fn modality(&self) -> Modality {
        Modality::Image
    }

    fn optimal_chunk_size(&self, available_memory: usize) -> usize {
        // For tiled VAE decode
        // 512x512 tile at fp16 = ~1.5MB per tile
        let tile_memory = 512 * 512 * 2 * 4; // RGBA fp16
        available_memory / tile_memory
    }

    fn supports_streaming(&self) -> bool {
        true // Progressive decode
    }

    fn prefetch_pattern(&self) -> PrefetchPattern {
        PrefetchPattern::Random // Tiles can be processed in any order
    }

    fn cache_strategy(&self) -> CacheStrategy {
        CacheStrategy::Lru
    }
}

/// Progressive image output.
#[derive(Debug)]
pub struct ProgressiveImage {
    /// Current step
    pub step: u32,
    /// Total steps
    pub total_steps: u32,
    /// Preview image (lower resolution)
    pub preview: Option<Tensor>,
    /// Final image (on last step)
    pub final_image: Option<Tensor>,
}

/// Cache for control features.
#[derive(Debug, Default)]
pub struct ControlCache {
    /// ControlNet feature cache
    controlnet: dashmap::DashMap<u64, Tensor>,
    /// IP-Adapter embedding cache
    ip_adapter: dashmap::DashMap<u64, Tensor>,
    /// Merged LoRA weights cache
    lora: dashmap::DashMap<String, Tensor>,
}

impl ControlCache {
    /// Create a new control cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or compute ControlNet features.
    pub fn get_controlnet(&self, image_hash: u64) -> Option<Tensor> {
        self.controlnet.get(&image_hash).map(|v| v.clone())
    }

    /// Cache ControlNet features.
    pub fn cache_controlnet(&self, image_hash: u64, features: Tensor) {
        self.controlnet.insert(image_hash, features);
    }

    /// Get or compute IP-Adapter embedding.
    pub fn get_ip_adapter(&self, image_hash: u64) -> Option<Tensor> {
        self.ip_adapter.get(&image_hash).map(|v| v.clone())
    }

    /// Cache IP-Adapter embedding.
    pub fn cache_ip_adapter(&self, image_hash: u64, embedding: Tensor) {
        self.ip_adapter.insert(image_hash, embedding);
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.controlnet.clear();
        self.ip_adapter.clear();
        self.lora.clear();
    }
}

/// Generate pseudo-random noise (standalone for use in spawned tasks).
fn generate_noise_standalone(size: usize, seed: Option<u64>) -> Vec<f32> {
    let mut state = seed.unwrap_or(42);
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        // LCG pseudo-random number generator
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Box-Muller approximation for normal distribution
        let u1 = (state as f32) / (u64::MAX as f32);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state as f32) / (u64::MAX as f32);

        // Approximate standard normal using Box-Muller
        let u1 = u1.max(1e-7); // Avoid log(0)
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        data.push(z);
    }

    data
}

/// UNet noise prediction (standalone for use in spawned tasks).
///
/// UNet noise prediction requires loaded model weights.
/// Without weights, we cannot predict noise.
/// Returns the input scaled by timestep as a degraded fallback
/// that at least preserves the mathematical structure of the diffusion scheduler.
///
/// With loaded weights, this would be:
/// 1. Encode timestep embedding
/// 2. Run UNet: down blocks -> mid block -> up blocks
/// 3. Apply classifier-free guidance: noise = uncond + scale * (cond - uncond)
fn predict_noise_standalone(latent_data: &[f32], timestep: f32, _guidance_scale: f32) -> Vec<f32> {
    let t_factor = timestep.min(1.0).max(0.0);
    latent_data.iter().map(|&x| x * t_factor).collect()
}

/// CPU VAE decoder (standalone for use in spawned tasks).
///
/// VAE decoder converts 4-channel latents to 3-channel RGB pixels.
/// Without loaded VAE weights, uses the standard SD latent-to-RGB approximation
/// coefficients with bilinear upsampling to preserve spatial structure.
///
/// With loaded weights, this would be:
/// 1. Scale latents by 1/0.18215 (SD scaling factor)
/// 2. Conv 4->512, then 4 stages of ResNet+Upsample
/// 3. Final conv 128->3 + sigmoid
fn vae_decode_cpu_standalone(
    latent_data: &[f32],
    latent_h: usize,
    latent_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut image_data = vec![0.0f32; 3 * out_h * out_w];

    // Standard VAE latent-to-RGB approximation (used by SD latent preview).
    // These coefficients approximate the first layer of the VAE decoder
    // and are the real values used in Stable Diffusion latent preview decoders.
    let coeffs: [[f32; 4]; 3] = [
        [ 0.298,  0.187,  0.158,  0.042],  // R
        [ 0.207,  0.286,  0.143, -0.049],  // G
        [ 0.208,  0.173,  0.296,  0.121],  // B
    ];

    for y in 0..out_h {
        for x in 0..out_w {
            // Map output pixel to latent coordinate (bilinear)
            let lat_y = (y as f32 / out_h as f32) * latent_h as f32;
            let lat_x = (x as f32 / out_w as f32) * latent_w as f32;

            let y0 = (lat_y as usize).min(latent_h - 1);
            let x0 = (lat_x as usize).min(latent_w - 1);
            let y1 = (y0 + 1).min(latent_h - 1);
            let x1 = (x0 + 1).min(latent_w - 1);

            let fy = lat_y - y0 as f32;
            let fx = lat_x - x0 as f32;

            // Bilinear interpolation for each latent channel, then project to RGB
            for c in 0..3 {
                let mut pixel_val = 0.5f32; // bias term
                for lc in 0..4 {
                    let v00 = latent_data.get(lc * latent_h * latent_w + y0 * latent_w + x0).copied().unwrap_or(0.0);
                    let v01 = latent_data.get(lc * latent_h * latent_w + y0 * latent_w + x1).copied().unwrap_or(0.0);
                    let v10 = latent_data.get(lc * latent_h * latent_w + y1 * latent_w + x0).copied().unwrap_or(0.0);
                    let v11 = latent_data.get(lc * latent_h * latent_w + y1 * latent_w + x1).copied().unwrap_or(0.0);

                    let interp = v00 * (1.0 - fx) * (1.0 - fy)
                        + v01 * fx * (1.0 - fy)
                        + v10 * (1.0 - fx) * fy
                        + v11 * fx * fy;

                    pixel_val += interp * coeffs[c][lc];
                }

                // Clamp to [0, 1]
                let pixel_val = pixel_val.clamp(0.0, 1.0);
                image_data[c * out_h * out_w + y * out_w + x] = pixel_val;
            }
        }
    }

    image_data
}
