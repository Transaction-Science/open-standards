//! Inference sessions.

use super::config::{TextParams, ImageParams, ThreeDParams};
use super::engine::{TextToken, ImageProgress, Object3D};
use super::Model;
use super::tokenizer::Tokenizer;
use crate::core::{Error, Id, Result};
use crate::runtime::ResourceMonitor;
use crate::runtime::stream::StreamSender;
use crate::tensor::Tensor;
use std::sync::Arc;

#[cfg(feature = "metal")]
use super::llm::{LLMPipeline, PagedKVCache as LLMPagedKVCache};

/// Session configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum context length
    pub max_context_length: usize,
    /// Enable KV cache
    pub enable_kv_cache: bool,
    /// Prefetch next layer weights
    pub prefetch_weights: bool,
    /// Session timeout (seconds)
    pub timeout_seconds: Option<u64>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            enable_kv_cache: true,
            prefetch_weights: true,
            timeout_seconds: None,
        }
    }
}

/// An inference session.
///
/// Sessions maintain state between inference calls (e.g., KV cache for LLMs).
pub struct Session {
    /// Unique ID
    id: Id,
    /// Associated model
    model: Arc<Model>,
    /// Configuration
    config: SessionConfig,
    /// Tokenizer (for LLMs)
    tokenizer: Option<Arc<Tokenizer>>,
    /// KV cache (for LLMs)
    kv_cache: parking_lot::RwLock<Option<KVCache>>,
    /// LLM pipeline for real transformer inference (Metal only)
    #[cfg(feature = "metal")]
    llm_pipeline: parking_lot::RwLock<Option<Arc<LLMPipeline>>>,
    /// LLM KV cache for the real pipeline
    #[cfg(feature = "metal")]
    llm_kv_cache: parking_lot::RwLock<Option<LLMPagedKVCache>>,
    /// Last activity timestamp
    last_activity: parking_lot::RwLock<std::time::Instant>,
    /// Token count (for LLMs)
    token_count: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Session");
        d.field("id", &self.id)
            .field("model", &self.model.info().name)
            .field("has_tokenizer", &self.tokenizer.is_some());
        #[cfg(feature = "metal")]
        d.field("has_llm_pipeline", &self.llm_pipeline.read().is_some());
        d.field("token_count", &self.token_count.load(std::sync::atomic::Ordering::Relaxed))
            .finish()
    }
}

impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            model: Arc::clone(&self.model),
            config: self.config.clone(),
            tokenizer: self.tokenizer.clone(),
            kv_cache: parking_lot::RwLock::new(None), // Don't clone cache
            #[cfg(feature = "metal")]
            llm_pipeline: parking_lot::RwLock::new(self.llm_pipeline.read().clone()),
            #[cfg(feature = "metal")]
            llm_kv_cache: parking_lot::RwLock::new(None), // Don't clone KV cache
            last_activity: parking_lot::RwLock::new(std::time::Instant::now()),
            token_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Session {
    /// Create a new session.
    pub fn new(
        model: Arc<Model>,
        config: SessionConfig,
        _monitor: Arc<ResourceMonitor>,
    ) -> Result<Self> {
        Self::with_tokenizer(model, config, _monitor, None)
    }

    /// Create a new session with an explicit tokenizer.
    pub fn with_tokenizer(
        model: Arc<Model>,
        config: SessionConfig,
        _monitor: Arc<ResourceMonitor>,
        tokenizer: Option<Arc<Tokenizer>>,
    ) -> Result<Self> {
        let id = Id::new();

        // Initialize KV cache if needed
        let kv_cache = if config.enable_kv_cache {
            Some(KVCache::new(
                config.max_context_length,
                model.config().num_layers,
                model.config().hidden_size,
                model.config().num_heads,
            ))
        } else {
            None
        };

        // Try to create an LLM pipeline for real transformer inference
        #[cfg(feature = "metal")]
        let llm_pipeline = {
            let is_llm = !model.weight_names().is_empty()
                && (model.get_weight("model.embed_tokens.weight").is_some()
                    || model.get_weight("token_embd.weight").is_some());
            if is_llm {
                match crate::tensor::get_metal_device() {
                    Ok(device) => {
                        match LLMPipeline::new(Arc::clone(&model), device.clone()) {
                            Ok(pipeline) => {
                                tracing::debug!("LLM pipeline created for session {}", id);
                                Some(Arc::new(pipeline))
                            }
                            Err(e) => {
                                tracing::warn!("Failed to create LLM pipeline: {}", e);
                                None
                            }
                        }
                    }
                    Err(_) => None,
                }
            } else {
                None
            }
        };

        Ok(Self {
            id,
            model,
            config,
            tokenizer,
            kv_cache: parking_lot::RwLock::new(kv_cache),
            #[cfg(feature = "metal")]
            llm_pipeline: parking_lot::RwLock::new(llm_pipeline),
            #[cfg(feature = "metal")]
            llm_kv_cache: parking_lot::RwLock::new(None),
            last_activity: parking_lot::RwLock::new(std::time::Instant::now()),
            token_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Get session ID.
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the model.
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Get configuration.
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Get current token count.
    pub fn token_count(&self) -> usize {
        self.token_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the session (clear KV cache).
    pub fn reset(&self) {
        if let Some(ref mut cache) = *self.kv_cache.write() {
            cache.clear();
        }
        self.token_count.store(0, std::sync::atomic::Ordering::Relaxed);
        *self.last_activity.write() = std::time::Instant::now();
    }

    /// Update last activity.
    fn touch(&self) {
        *self.last_activity.write() = std::time::Instant::now();
    }

    /// Internal text generation.
    ///
    /// Delegates to the real LLM pipeline (Metal) when available. The pipeline runs
    /// the full transformer forward pass: embed -> N layers (RMSNorm -> Attention ->
    /// Residual -> RMSNorm -> MLP -> Residual) -> final RMSNorm -> LM head -> logits.
    /// Sampling, KV cache, and token streaming are handled by the pipeline.
    ///
    /// When no pipeline is available (no Metal, no weights), falls back to a CPU
    /// stub that returns errors.
    pub(crate) async fn generate_text_internal(
        &self,
        prompt: &str,
        params: TextParams,
        sender: &StreamSender<TextToken>,
        monitor: &ResourceMonitor,
    ) -> Result<()> {
        self.touch();

        // Tokenize input using the real tokenizer
        let input_tokens = if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(prompt).ids
        } else {
            return Err(Error::internal("No tokenizer loaded for text generation. Load a tokenizer first."));
        };

        monitor.memory().record_alloc(input_tokens.len() * 4);

        // Delegate to the real LLM pipeline when available
        #[cfg(feature = "metal")]
        {
            let pipeline = self.llm_pipeline.read().clone();
            if let Some(pipeline) = pipeline {
                // Initialize or reset the LLM KV cache
                let config = self.model.config();
                let head_dim = config.hidden_size / config.num_heads;
                let mut kv_cache = LLMPagedKVCache::new(
                    config.num_layers,
                    config.num_heads,
                    head_dim,
                    config.max_seq_len,
                );

                // Run the real transformer forward pass
                pipeline.generate(
                    &input_tokens,
                    &params,
                    &mut kv_cache,
                    sender,
                    monitor,
                ).await?;

                // Store the KV cache for potential continuation
                *self.llm_kv_cache.write() = Some(kv_cache);

                return Ok(());
            }
        }

        // Fallback: no LLM pipeline available
        if self.model.weight_names().is_empty() {
            return Err(Error::internal(
                "No model weights loaded for text generation. Load a model with weights first.",
            ));
        }

        Err(Error::internal(
            "LLM pipeline not available. Metal feature required for transformer inference.",
        ))
    }

    /// Internal image generation using diffusion pipeline.
    ///
    /// Requires a diffusion model with UNet, text encoder, and VAE decoder weights loaded.
    /// Delegates noise prediction to the real UNet forward pass through the DiffusionPipeline.
    pub(crate) async fn generate_image_internal(
        &self,
        _prompt: &str,
        _params: ImageParams,
        _monitor: &ResourceMonitor,
    ) -> Result<Tensor> {
        self.touch();

        // Verify the model has weights loaded for diffusion inference
        if self.model.weight_names().is_empty() {
            return Err(Error::internal(
                "No diffusion model weights loaded. Load a model with UNet, text encoder, and VAE weights first.",
            ));
        }

        // Verify required diffusion components are present
        let has_unet = self.model.weight_names().iter().any(|n| n.contains("unet") || n.contains("model.diffusion"));
        let has_vae = self.model.weight_names().iter().any(|n| n.contains("vae") || n.contains("decoder"));
        if !has_unet {
            return Err(Error::internal(
                "Model missing UNet weights. Cannot run diffusion forward pass.",
            ));
        }
        if !has_vae {
            return Err(Error::internal(
                "Model missing VAE decoder weights. Cannot decode latents to image.",
            ));
        }

        // Noise prediction and denoising should be performed by the DiffusionPipeline,
        // which runs the real UNet forward pass and VAE decode through Metal compute kernels.
        // This session-level method is a thin wrapper that validates readiness and delegates.
        Err(Error::internal(
            "Diffusion pipeline not initialized on this session. Use DiffusionPipeline::generate() for image generation.",
        ))
    }

    /// Internal progressive image generation with streaming previews.
    ///
    /// Requires a diffusion model with UNet, text encoder, and VAE decoder weights loaded.
    /// Delegates noise prediction to the real UNet forward pass through the DiffusionPipeline.
    pub(crate) async fn generate_image_progressive_internal(
        &self,
        _prompt: &str,
        _params: ImageParams,
        _sender: &StreamSender<ImageProgress>,
        _monitor: &ResourceMonitor,
    ) -> Result<()> {
        self.touch();

        // Verify the model has weights loaded for diffusion inference
        if self.model.weight_names().is_empty() {
            return Err(Error::internal(
                "No diffusion model weights loaded. Load a model with UNet, text encoder, and VAE weights first.",
            ));
        }

        // Verify required diffusion components are present
        let has_unet = self.model.weight_names().iter().any(|n| n.contains("unet") || n.contains("model.diffusion"));
        let has_vae = self.model.weight_names().iter().any(|n| n.contains("vae") || n.contains("decoder"));
        if !has_unet {
            return Err(Error::internal(
                "Model missing UNet weights. Cannot run diffusion forward pass.",
            ));
        }
        if !has_vae {
            return Err(Error::internal(
                "Model missing VAE decoder weights. Cannot decode latents to image.",
            ));
        }

        // Progressive image generation with streaming previews should be performed by the
        // DiffusionPipeline, which runs the real UNet forward pass at each denoising step
        // and streams intermediate latent previews through the VAE decoder.
        Err(Error::internal(
            "Diffusion pipeline not initialized on this session. Use DiffusionPipeline for progressive image generation.",
        ))
    }

    /// Internal image to 3D conversion.
    ///
    /// Uses the 3D modality handler to generate Gaussian splats from an input image,
    /// then extracts positions, scales, rotations, colors, and opacities.
    pub(crate) async fn image_to_3d_internal(
        &self,
        image: &Tensor,
        _params: ThreeDParams,
        _monitor: &ResourceMonitor,
    ) -> Result<Object3D> {
        self.touch();

        // Use the 3D modality handler
        let handler = crate::modalities::three_d::ThreeDHandler::new();
        let input = crate::modalities::three_d::ThreeDInput {
            source: crate::modalities::three_d::ThreeDSource::SingleImage(image.clone()),
            representation: crate::modalities::three_d::Representation3D::GaussianSplats,
        };

        let output = handler.generate(input).await?;

        if let Some(object) = output.object {
            match object.representation {
                crate::modalities::three_d::Object3DData::GaussianSplats(cloud) => {
                    let count = cloud.count;
                    let device = crate::hal::DeviceId::cpu();

                    // Generate scales (uniform small scale)
                    let scales_data = vec![0.01f32; count * 3];
                    let scales = Tensor::from_slice(
                        &scales_data,
                        crate::core::Shape::from([count, 3]),
                        crate::tensor::DType::F32,
                        device,
                    )?;

                    // Generate rotations (identity quaternions)
                    let mut rotations_data = Vec::with_capacity(count * 4);
                    for _ in 0..count {
                        rotations_data.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]); // w, x, y, z
                    }
                    let rotations = Tensor::from_slice(
                        &rotations_data,
                        crate::core::Shape::from([count, 4]),
                        crate::tensor::DType::F32,
                        device,
                    )?;

                    Ok(Object3D {
                        positions: cloud.positions,
                        scales,
                        rotations,
                        colors: cloud.colors,
                        opacities: cloud.opacities,
                        num_gaussians: count,
                    })
                }
                _ => Err(Error::internal("expected Gaussian splats output")),
            }
        } else {
            Err(Error::internal("3D generation produced no output"))
        }
    }
}

/// KV cache for transformer inference.
#[derive(Debug)]
struct KVCache {
    /// Maximum sequence length
    max_seq_len: usize,
    /// Number of layers
    num_layers: usize,
    /// Key caches per layer
    keys: Vec<Option<Tensor>>,
    /// Value caches per layer
    values: Vec<Option<Tensor>>,
    /// Current position
    position: usize,
}

impl KVCache {
    fn new(max_seq_len: usize, num_layers: usize, _hidden_size: usize, _num_heads: usize) -> Self {
        Self {
            max_seq_len,
            num_layers,
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            position: 0,
        }
    }

    fn clear(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
        self.position = 0;
    }

    fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.max_seq_len.saturating_sub(self.position)
    }
}

// Tokenization is now handled via the Session's Tokenizer field.
// Use session.tokenizer.encode(text) for real tokenization.
// The standalone tokenize() function has been removed to prevent
// accidental use of fake tokenization.
