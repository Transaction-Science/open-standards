//! REST API server for efficient-genai.
//!
//! Provides endpoints for:
//! - Text generation (LLM inference)
//! - Image generation (diffusion models)
//! - Model management
//! - Health checks and metrics

pub mod model_fetch;
// `model_library` and `worker` are dev-only scaffolding; they reference
// AppState fields and crates not present in this build. Re-enable when the
// branch stabilises.
// pub mod model_library;
// pub mod worker;

use axum::{
    routing::{get, post, delete},
    Router,
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use crate::core::Result;
use half::f16;
use tokio::fs;

#[cfg(feature = "metal")]
use crate::inference::Model;
#[cfg(feature = "metal")]
use crate::inference::architecture::florence2::{Florence2Pipeline, Florence2Config};
#[cfg(feature = "metal")]
use crate::inference::architecture::hunyuan3d::{Hunyuan3DPipeline, Hunyuan3DConfig};
#[cfg(feature = "metal")]
use crate::inference::architecture::sana_wm::{SanaWmPipeline, SanaWmConfig};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, LazyLoader};

// ============================================================================
// Server State
// ============================================================================

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    /// Diffusion pipeline for image generation
    pipeline: Option<Arc<crate::inference::DiffusionPipeline>>,
    /// Florence-2 vision-language pipeline for grounding / captioning / OCR.
    /// Loaded at startup when `FLORENCE2_MODEL_PATH` points at a single
    /// safetensors checkpoint.
    #[cfg(feature = "metal")]
    florence2: Option<Arc<Florence2Pipeline>>,
    /// Hunyuan3D 2.0 image-to-3D pipeline. Loaded at startup when
    /// `HUNYUAN3D_MODEL_DIR` is set (expects `dit_model.safetensors`,
    /// `dino_model.safetensors`, `vae_model.safetensors` in that dir).
    #[cfg(feature = "metal")]
    hunyuan3d: Option<Arc<Hunyuan3DPipeline>>,
    /// SANA-WM image+action→video pipeline. Loaded at startup when
    /// `SANA_WM_MODEL_DIR` is set (expects `dit/`, `vae/`,
    /// `text_encoder/` subdirectories each containing a single
    /// `*.safetensors`).
    #[cfg(feature = "metal")]
    sana_wm: Option<Arc<SanaWmPipeline>>,
    /// ControlNet (Canny preprocessor variant). Lazily constructed on first
    /// use; weights loaded from `CONTROLNET_CANNY_PATH` env. Falls back to
    /// architecture-skeleton (zero residuals) when env not set.
    #[cfg(feature = "metal")]
    controlnet_canny: Arc<std::sync::OnceLock<Option<Arc<crate::inference::diffusion::ControlNet>>>>,
    /// ControlNet (Scribble preprocessor variant). Lazily constructed
    /// from `CONTROLNET_SCRIBBLE_PATH`.
    #[cfg(feature = "metal")]
    controlnet_scribble: Arc<std::sync::OnceLock<Option<Arc<crate::inference::diffusion::ControlNet>>>>,
    /// HY-WorldMirror 2.0 reconstruction pipeline. Lazily constructed on
    /// first use; weights from `HY_WORLDMIRROR_PATH`.
    #[cfg(feature = "metal")]
    hy_worldmirror: Arc<std::sync::OnceLock<Option<Arc<crate::inference::architecture::hyworld::HYWorldPipeline>>>>,
    /// Text handler for LLM text generation
    text_handler: Arc<RwLock<crate::modalities::text::TextHandler>>,
    /// Loaded models registry
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    /// Server metrics
    metrics: Arc<RwLock<ServerMetrics>>,
    /// Server configuration
    config: ServerConfig,
    /// Server start time for uptime tracking
    start_time: Instant,
}

/// Information about a loaded model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model type
    pub model_type: String,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// When the model was loaded
    pub loaded_at: String,
    /// Model status
    pub status: String,
}

/// Server metrics.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ServerMetrics {
    /// Total requests served
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Total generation time (ms)
    pub total_generation_time_ms: u64,
    /// Average tokens per second (for text)
    pub avg_tokens_per_second: f32,
    /// Average images per second
    pub avg_images_per_second: f32,
    /// Current memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Default generation timeout (seconds)
    pub timeout_secs: u64,
    /// Output directory
    pub output_dir: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            timeout_secs: 300,
            output_dir: "outputs".to_string(),
        }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Florence-2 vision-language request.
///
/// `task` is one of `"caption"`, `"detailed_caption"`, `"ocr"`,
/// `"object_detection"`, `"grounding"`. The image is a base64-encoded
/// PNG/JPEG byte stream. `prompt` is optional and used by tasks that
/// support text conditioning (e.g. open-vocabulary grounding); ignored
/// for now since `florence2.rs::analyze` takes only the task string.
#[derive(Debug, Deserialize)]
pub struct VisionGroundingRequest {
    pub image_base64: String,
    pub task: String,
    #[serde(default)]
    pub prompt: Option<String>,
}

/// Florence-2 response.
#[derive(Debug, Serialize)]
pub struct VisionGroundingResponse {
    pub status: String,
    /// Raw text Florence-2 emits for the requested task. For grounding /
    /// object_detection this is a structured Florence-2 string the client
    /// can post-process; we return it as-is so callers can apply their own
    /// schema constraints (xgrammar / outlines / llguidance).
    pub text: Option<String>,
    /// Optional parsed regions, one per detected entity. Present when the
    /// server can decode Florence-2's location tokens; otherwise empty and
    /// the caller should parse `text`.
    pub regions: Vec<GroundedRegion>,
    /// Original image dimensions echoed back so clients can map normalised
    /// box coordinates without re-decoding the image.
    pub image_width: u32,
    pub image_height: u32,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

/// One labelled region.
#[derive(Debug, Serialize)]
pub struct GroundedRegion {
    pub label: String,
    /// Normalised 0..=1 bounding box `[x0, y0, x1, y1]`.
    pub bbox: [f32; 4],
    pub score: f32,
}

/// Hunyuan3D 2.0 image-to-3D request.
#[derive(Debug, Deserialize)]
pub struct Generate3dRequest {
    /// Base64-encoded PNG/JPEG. Resized to 518×518 for DINOv2-Giant input.
    pub image_base64: String,
    /// Random seed for the flow-matching DiT (default: 42).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Output mesh format. `"obj"` (default) or `"glb"` (planned).
    #[serde(default)]
    pub format: Option<String>,
}

/// Hunyuan3D 2.0 response.
#[derive(Debug, Serialize)]
pub struct Generate3dResponse {
    pub status: String,
    /// Mesh contents. For OBJ this is the literal `.obj` text. For GLB
    /// this is a base64-encoded blob (planned).
    pub mesh: Option<String>,
    /// Mesh format actually returned (`"obj"` for now).
    pub format: String,
    pub vertex_count: u32,
    pub face_count: u32,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

/// SANA-WM video generation request.
#[derive(Debug, Deserialize)]
pub struct GenerateVideoSanaRequest {
    /// Base64-encoded PNG/JPEG. Resized to the pipeline's `image_size`
    /// before VAE encode.
    pub image_base64: String,
    /// Text guidance (truncated to `caption_max_tokens` after tokenization).
    pub prompt: String,
    /// Camera action DSL (e.g. `"w-80,jw-40,w-40,lw-60,w-100"`).
    pub action: String,
    /// Total frames to produce. Default per `SanaWmConfig`.
    #[serde(default)]
    pub num_frames: Option<usize>,
    /// Per-step translation magnitude.
    #[serde(default)]
    pub translation_speed: Option<f32>,
    /// Per-step rotation magnitude (deg).
    #[serde(default)]
    pub rotation_speed_deg: Option<f32>,
    /// RNG seed for the flow-matching DiT.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// SANA-WM video generation response.
#[derive(Debug, Serialize)]
pub struct GenerateVideoSanaResponse {
    pub status: String,
    /// Public URL the encoded mp4 is served at (e.g. `/static/video/<uuid>.mp4`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_url: Option<String>,
    pub frames: u32,
    pub resolution: String,
    pub duration_s: f32,
    pub execution_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Image generation request.
#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    /// Text prompt
    pub prompt: String,
    /// Negative prompt
    pub negative_prompt: Option<String>,
    /// Image height
    pub height: Option<u32>,
    /// Image width
    pub width: Option<u32>,
    /// Number of inference steps
    pub steps: Option<u32>,
    /// Guidance scale
    pub guidance_scale: Option<f32>,
    /// Random seed
    pub seed: Option<u64>,
    /// Use LCM scheduler
    pub use_lcm: Option<bool>,
    /// Optional ControlNet conditioning. Each entry pairs a control image
    /// (already preprocessed, or raw — server runs the matching preprocessor
    /// based on `control_type`) with a strength multiplier.
    #[serde(default)]
    pub controls: Option<Vec<ControlInputRequest>>,
}

/// Single ControlNet input on the wire. Mirrors
/// [`crate::modalities::image::ControlInput`] but keeps the JSON shape
/// flat (no nested enum). Server converts to the in-engine type before
/// calling the diffusion pipeline.
#[derive(Debug, Deserialize)]
pub struct ControlInputRequest {
    /// One of: `canny`, `scribble`, `depth`, `normal`, `pose`, `segmentation`.
    pub control_type: String,
    /// Base64-encoded PNG/JPEG bytes of the control image. The server
    /// decodes to RGB CHW, normalises to 0..1, and runs the matching
    /// preprocessor (Canny / Scribble already implemented on Metal;
    /// Depth/Normal/Pose/Segmentation pass through pending separate
    /// architectures landing).
    pub image_base64: String,
    /// Conditioning strength in 0..1 (0.0 = no influence, 1.0 = full).
    /// Defaults to 1.0 if absent.
    #[serde(default = "default_control_strength")]
    pub strength: f32,
}

fn default_control_strength() -> f32 { 1.0 }

/// HY-World 2.0 reconstruction request. Takes a stack of multi-view images
/// (the deployable WorldMirror-2.0 path; full text/image → world generation
/// waits on Tencent's HY-Pano 2.0 + WorldStereo 2.0 release).
#[derive(Debug, Deserialize)]
pub struct WorldReconstructRequest {
    /// Multi-view RGB images, each base64-encoded PNG/JPEG. 1+ views;
    /// WorldMirror 2.0 trained on 2–8 view stacks.
    pub images_base64: Vec<String>,
    /// Render quality preset: `draft`, `standard`, `cinematic`. Default
    /// `standard`. Controls splat budget + (eventually) trajectory density.
    #[serde(default)]
    pub quality: Option<String>,
    /// RNG seed for deterministic placeholder trajectories until the real
    /// camera head ships.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Opt-in: compute and return depth maps via WorldMirror's `depth_head`.
    /// Adds ~30s CPU compute per view on the current path. Default false.
    #[serde(default)]
    pub compute_depth: Option<bool>,
    /// Opt-in: compute surface normals via `norm_head`. Default false.
    #[serde(default)]
    pub compute_normals: Option<bool>,
    /// Opt-in: compute 3D point cloud via `pts_head`. Default false.
    #[serde(default)]
    pub compute_points: Option<bool>,
    /// Optional pow3r conditioning: per-view depth values (base64 f32 LE,
    /// length 1369 × 196 = patches × 14×14). When supplied, the trained
    /// `depth_embed` conditioning path is exercised — adds a 1024-dim
    /// embedding to each patch token before the transformer.
    #[serde(default)]
    pub depth_hints_base64: Option<Vec<String>>,
    /// Optional pow3r conditioning: per-view camera pose [tx, ty, tz, qw, qx, qy, qz]
    /// (7-dim). Adds a broadcast 1024-dim embedding to all patch tokens.
    #[serde(default)]
    pub pose_hints: Option<Vec<[f32; 7]>>,
    /// Optional pow3r conditioning: per-view ray direction (4-dim). Adds a
    /// broadcast 1024-dim embedding to all patch tokens.
    #[serde(default)]
    pub ray_hints: Option<Vec<[f32; 4]>>,
}

/// HY-World 2.0 reconstruction response.
#[derive(Debug, Serialize)]
pub struct WorldReconstructResponse {
    /// Status: `success` or `error`.
    pub status: String,
    /// Base64-encoded `.splat` archive (32 bytes per Gaussian, web-viewer
    /// compatible: gsplat.js / Niantic / etc.). Present on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub splat_archive_base64: Option<String>,
    /// Splat count in the returned archive.
    pub splat_count: u32,
    /// Camera trajectory: array of `[px, py, pz, fx, fy, fz, ux, uy, uz, fov_deg]`.
    pub trajectory: Vec<Vec<f32>>,
    /// Inference time in milliseconds.
    pub execution_time_ms: u64,
    /// Optional depth maps per view (one base64 f32 LE blob per view).
    /// Present when the request set `compute_depth=true`. Each blob layout:
    /// flat row-major, length = depth_grid_h × depth_grid_w f32 values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_maps_base64: Option<Vec<String>>,
    /// Optional surface normals per view (base64 f32 LE blob, layout
    /// CHW = 3 × normal_grid_h × normal_grid_w). Opt-in via
    /// `compute_normals=true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_maps_base64: Option<Vec<String>>,
    /// Optional 3D point clouds per view (base64 f32 LE blob, layout CHW
    /// = 3 × points_grid_h × points_grid_w; xyz). Opt-in via
    /// `compute_points=true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_clouds_base64: Option<Vec<String>>,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Image generation response.
#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    /// Status of the request
    pub status: String,
    /// Path to the generated image
    pub image_path: Option<String>,
    /// Base64 encoded image data (if requested)
    pub image_base64: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of steps actually run
    pub steps: u32,
    /// Error message if failed
    pub error: Option<String>,
}

/// Text generation request.
#[derive(Debug, Deserialize)]
pub struct TextGenerationRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    /// Temperature for sampling
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling
    pub top_p: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<usize>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Whether to stream the response
    pub stream: Option<bool>,
}

/// Text generation response.
#[derive(Debug, Serialize)]
pub struct TextGenerationResponse {
    /// Status of the request
    pub status: String,
    /// Generated text
    pub text: Option<String>,
    /// Token IDs
    pub tokens: Option<Vec<u32>>,
    /// Number of tokens generated
    pub num_tokens: usize,
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Time to first token (ms)
    pub time_to_first_token_ms: f32,
    /// Total execution time (ms)
    pub execution_time_ms: u64,
    /// Finish reason
    pub finish_reason: String,
    /// Error message if failed
    pub error: Option<String>,
}

/// Model load request.
#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    /// Path to model weights
    pub model_path: String,
    /// Model type (llm, diffusion, etc.)
    pub model_type: String,
    /// Model ID (optional, defaults to filename)
    pub model_id: Option<String>,
}

/// Model load response.
#[derive(Debug, Serialize)]
pub struct ModelLoadResponse {
    /// Status
    pub status: String,
    /// Model ID
    pub model_id: Option<String>,
    /// Memory usage
    pub memory_bytes: Option<usize>,
    /// Error message
    pub error: Option<String>,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Status
    pub status: String,
    /// Version
    pub version: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Number of loaded models
    pub models_loaded: usize,
    /// Memory usage
    pub memory_usage_bytes: usize,
}

// ============================================================================
// Server Implementation
// ============================================================================

/// Start the REST API server.
pub async fn start_server(port: u16) -> Result<()> {
    tracing_subscriber::fmt::init();

    tracing::info!("Initializing Efficient GenAI Server...");

    // Initialize pipeline
    #[cfg(feature = "metal")]
    let pipeline = {
        use crate::inference::{DiffusionPipeline, Model, ModelType};
        use crate::hal::{MetalDevice, LazyLoader};

        let device = Arc::new(MetalDevice::new()?);
        let loader = Arc::new(LazyLoader::new(device.clone()));

        // Check for SDXL model path via environment variable
        let model_path = std::env::var("SDXL_MODEL_PATH").ok();

        if let Some(ref path_str) = model_path {
            let path = std::path::Path::new(path_str);
            if path.exists() {
                tracing::info!("Loading SDXL model from: {}", path.display());
                match Model::load("sdxl", path, loader.clone()) {
                    Ok(model) => {
                        let model = Arc::new(model);
                        // For SDXL, the same checkpoint contains UNet, text encoder, and VAE
                        // The DiffusionPipeline will use weight prefixes to access each component
                        let unet = model.clone();
                        let text_encoder = Some(model.clone());
                        let vae = model.clone();

                        // Try to load the CLIP tokenizer. Diffusers SD 1.x/2.x
                        // layouts put it at `{model}/tokenizer/` (legacy CLIP
                        // vocab.json + merges.txt, OR a fast tokenizer.json),
                        // NOT a top-level `tokenizer.json`. Search, in order:
                        //   1. $CLIP_TOKENIZER_PATH (explicit override)
                        //   2. {model}/tokenizer.json
                        //   3. {model}/tokenizer/tokenizer.json
                        //   4. {model}/tokenizer/vocab.json
                        // Without a real tokenizer the CLIP encoder silently
                        // falls back to `tokenize_prompt_basic` (a fake
                        // word-level splitter) → wrong token IDs → text
                        // embeddings ~orthogonal to reference (cos≈0.06) →
                        // prompt-irrelevant images. This was the dominant
                        // remaining bug after the U-Net/VAE/scheduler were
                        // all verified correct.
                        let candidates: Vec<std::path::PathBuf> = {
                            let mut v = Vec::new();
                            if let Ok(p) = std::env::var("CLIP_TOKENIZER_PATH") {
                                v.push(std::path::PathBuf::from(p));
                            }
                            v.push(path.join("tokenizer.json"));
                            v.push(path.join("tokenizer").join("tokenizer.json"));
                            v.push(path.join("tokenizer").join("vocab.json"));
                            v
                        };
                        let clip_tokenizer = candidates.iter().find(|p| p.exists()).and_then(|tk| {
                            match crate::inference::tokenizer::Tokenizer::load(tk) {
                                Ok(t) => {
                                    tracing::info!("Loaded CLIP tokenizer from {}", tk.display());
                                    Some(Arc::new(t))
                                }
                                Err(e) => {
                                    tracing::error!("Failed to load CLIP tokenizer from {}: {} — text conditioning WILL be wrong", tk.display(), e);
                                    None
                                }
                            }
                        });

                        // use_lcm only applies to LCM-distilled checkpoints (4-step
                        // x0-prediction). SD 1.x / SDXL / Flux base weights need
                        // their default eps-prediction scheduler (DDPM in `new()`),
                        // so default to `false`. `LCM_MODEL=1` opt-in for LCM weights.
                        let use_lcm = std::env::var("LCM_MODEL").ok().as_deref() == Some("1");
                        match DiffusionPipeline::new(unet, text_encoder, vae, device.clone(), clip_tokenizer, use_lcm) {
                            Ok(p) => {
                                tracing::info!("Diffusion pipeline initialized with SDXL weights ({} parameters)",
                                    model.info().num_parameters);
                                Some(Arc::new(p))
                            }
                            Err(e) => {
                                tracing::error!("Failed to init pipeline with SDXL: {}", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to load SDXL model: {}", e);
                        None
                    }
                }
            } else {
                tracing::warn!("SDXL_MODEL_PATH set but file not found: {}", path.display());
                None
            }
        } else {
            // No model path specified, start with dummy models
            tracing::info!("No SDXL_MODEL_PATH set. Starting without model. Use POST /api/v1/models to load one.");
            let unet = Arc::new(Model::dummy(ModelType::TextToImage));
            let text_encoder = Some(Arc::new(Model::dummy(ModelType::LLM)));
            let vae = Arc::new(Model::dummy(ModelType::ImageToImage));

            match DiffusionPipeline::new(unet, text_encoder, vae, device, None, false) {
                Ok(p) => {
                    tracing::info!("Diffusion pipeline initialized (no weights loaded)");
                    Some(Arc::new(p))
                }
                Err(e) => {
                    tracing::error!("Failed to init pipeline: {}", e);
                    None
                }
            }
        }
    };

    #[cfg(not(feature = "metal"))]
    let pipeline = None;

    // Florence-2 vision-language pipeline (Apple-Silicon, optional).
    // Loaded when `FLORENCE2_MODEL_PATH` points at a single safetensors
    // checkpoint. Used for `<GROUNDING>`, `<CAPTION>`, `<OCR>`, etc.
    #[cfg(feature = "metal")]
    let florence2 = {
        match std::env::var("FLORENCE2_MODEL_PATH").ok() {
            Some(path_str) => {
                let path = std::path::Path::new(&path_str);
                if !path.exists() {
                    tracing::warn!(
                        "FLORENCE2_MODEL_PATH set but file not found: {}",
                        path.display()
                    );
                    None
                } else {
                    tracing::info!("Loading Florence-2 model from: {}", path.display());
                    let device = Arc::new(MetalDevice::new()?);
                    let loader = Arc::new(LazyLoader::new(device.clone()));
                    match Model::load("florence2", path, loader) {
                        Ok(model) => {
                            let cfg = Florence2Config::base();
                            match Florence2Pipeline::new(Arc::new(model), cfg, device) {
                                Ok(p) => {
                                    tracing::info!("Florence-2 pipeline initialized");
                                    Some(Arc::new(p))
                                }
                                Err(e) => {
                                    tracing::error!("Florence-2 pipeline init failed: {}", e);
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to load Florence-2 weights: {}", e);
                            None
                        }
                    }
                }
            }
            None => {
                tracing::info!(
                    "FLORENCE2_MODEL_PATH not set — vision/grounding endpoint will return 503"
                );
                None
            }
        }
    };

    // Hunyuan3D 2.0 image-to-3D pipeline (Apple-Silicon, optional).
    // Loaded when `HUNYUAN3D_MODEL_DIR` points at a directory containing
    // `dit_model.safetensors`, `dino_model.safetensors`, and
    // `vae_model.safetensors`. See examples/hunyuan3d_generate.rs.
    #[cfg(feature = "metal")]
    let hunyuan3d = {
        match std::env::var("HUNYUAN3D_MODEL_DIR").ok() {
            Some(dir_str) => {
                let dir = std::path::Path::new(&dir_str);
                let dit_path = dir.join("dit_model.safetensors");
                let dino_path = dir.join("dino_model.safetensors");
                let vae_path = dir.join("vae_model.safetensors");
                if !dit_path.exists() || !dino_path.exists() || !vae_path.exists() {
                    tracing::warn!(
                        "HUNYUAN3D_MODEL_DIR set but missing one of dit/dino/vae .safetensors: {}",
                        dir.display()
                    );
                    None
                } else {
                    tracing::info!("Loading Hunyuan3D 2.0 from: {}", dir.display());
                    let device = Arc::new(MetalDevice::new()?);
                    let loader = Arc::new(LazyLoader::new(device.clone()));
                    let load_one = |name: &str, p: &std::path::Path| {
                        Model::load(name, p, loader.clone()).map(Arc::new)
                    };
                    let dit = load_one("hunyuan3d_dit", &dit_path);
                    let dino = load_one("hunyuan3d_dino", &dino_path);
                    let vae = load_one("hunyuan3d_vae", &vae_path);
                    match (dit, dino, vae) {
                        (Ok(dit), Ok(dino), Ok(vae)) => {
                            let cfg = Hunyuan3DConfig::default();
                            match Hunyuan3DPipeline::new(dit, dino, vae, cfg, device) {
                                Ok(p) => {
                                    tracing::info!("Hunyuan3D 2.0 pipeline initialized");
                                    Some(Arc::new(p))
                                }
                                Err(e) => {
                                    tracing::error!("Hunyuan3D pipeline init failed: {}", e);
                                    None
                                }
                            }
                        }
                        (a, b, c) => {
                            for (label, r) in
                                [("dit", a), ("dino", b), ("vae", c)]
                            {
                                if let Err(e) = r {
                                    tracing::error!("Failed to load Hunyuan3D {label}: {e}");
                                }
                            }
                            None
                        }
                    }
                }
            }
            None => {
                tracing::info!(
                    "HUNYUAN3D_MODEL_DIR not set — 3d/generate endpoint will return 503"
                );
                None
            }
        }
    };
    #[cfg(not(feature = "metal"))]
    let _hunyuan3d_unused: Option<()> = None;

    // SANA-WM image+action→video pipeline (Apple-Silicon, optional).
    // Loaded when `SANA_WM_MODEL_DIR` points at a directory containing
    // `dit/*.safetensors`, `vae/*.safetensors`, `text_encoder/*.safetensors`.
    #[cfg(feature = "metal")]
    let sana_wm = {
        match std::env::var("SANA_WM_MODEL_DIR").ok() {
            Some(dir_str) => {
                let dir = std::path::Path::new(&dir_str);
                let find_safetensors = |subdir: &str| -> Option<std::path::PathBuf> {
                    let sd = dir.join(subdir);
                    std::fs::read_dir(&sd).ok().and_then(|rd| {
                        rd.filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .find(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
                    })
                };
                let dit_path = find_safetensors("dit");
                let vae_path = find_safetensors("vae");
                let text_path = find_safetensors("text_encoder");
                match (dit_path, vae_path, text_path) {
                    (Some(dit_p), Some(vae_p), Some(text_p)) => {
                        tracing::info!("Loading SANA-WM from: {}", dir.display());
                        let device = Arc::new(MetalDevice::new()?);
                        let loader = Arc::new(LazyLoader::new(device.clone()));
                        let load_one = |name: &str, p: &std::path::Path| {
                            Model::load(name, p, loader.clone()).map(Arc::new)
                        };
                        let dit = load_one("sana_wm_dit", &dit_p);
                        let vae = load_one("sana_wm_vae", &vae_p);
                        let text = load_one("sana_wm_text", &text_p);
                        match (dit, vae, text) {
                            (Ok(dit), Ok(vae), Ok(text)) => {
                                let cfg = SanaWmConfig::default();
                                match SanaWmPipeline::new(dit, vae, text, cfg, device) {
                                    Ok(p) => {
                                        tracing::info!("SANA-WM pipeline initialized");
                                        Some(Arc::new(p))
                                    }
                                    Err(e) => {
                                        tracing::error!("SANA-WM pipeline init failed: {}", e);
                                        None
                                    }
                                }
                            }
                            (a, b, c) => {
                                for (label, r) in
                                    [("dit", a), ("vae", b), ("text_encoder", c)]
                                {
                                    if let Err(e) = r {
                                        tracing::error!("Failed to load SANA-WM {label}: {e}");
                                    }
                                }
                                None
                            }
                        }
                    }
                    _ => {
                        tracing::warn!(
                            "SANA_WM_MODEL_DIR set but missing one of dit/vae/text_encoder/*.safetensors: {}",
                            dir.display()
                        );
                        None
                    }
                }
            }
            None => {
                tracing::info!(
                    "SANA_WM_MODEL_DIR not set — video/sana endpoint will return 503"
                );
                None
            }
        }
    };
    #[cfg(not(feature = "metal"))]
    let _sana_wm_unused: Option<()> = None;

    let config = ServerConfig::default();

    // Create output directory
    fs::create_dir_all(&config.output_dir).await
        .map_err(|e: std::io::Error| crate::core::Error::io("output_dir", e.to_string()))?;

    // Static-asset directory for video outputs (mp4 produced by
    // /api/v1/video/sana). Mounted at `/static/...` below via `ServeDir`.
    fs::create_dir_all("static/video").await
        .map_err(|e: std::io::Error| crate::core::Error::io("static/video", e.to_string()))?;

    // Initialize text handler - will load model later via POST /api/v1/models
    let mut text_handler = crate::modalities::text::TextHandler::new();

    // Try to load LLM model from environment variable
    if let Ok(llm_path) = std::env::var("LLM_MODEL_PATH") {
        let llm_dir = std::path::Path::new(&llm_path);
        if llm_dir.exists() {
            match text_handler.load_model(llm_dir) {
                Ok(()) => tracing::info!("Loaded LLM tokenizer from: {}", llm_path),
                Err(e) => tracing::warn!("Failed to load LLM model: {}", e),
            }
        }
    }

    let state = AppState {
        pipeline,
        #[cfg(feature = "metal")]
        florence2,
        #[cfg(feature = "metal")]
        hunyuan3d,
        #[cfg(feature = "metal")]
        sana_wm,
        #[cfg(feature = "metal")]
        controlnet_canny: Arc::new(std::sync::OnceLock::new()),
        #[cfg(feature = "metal")]
        controlnet_scribble: Arc::new(std::sync::OnceLock::new()),
        #[cfg(feature = "metal")]
        hy_worldmirror: Arc::new(std::sync::OnceLock::new()),
        text_handler: Arc::new(RwLock::new(text_handler)),
        models: Arc::new(RwLock::new(HashMap::new())),
        metrics: Arc::new(RwLock::new(ServerMetrics::default())),
        config,
        start_time: Instant::now(),
    };

    // Pre-warm the WorldMirror pipeline in a background task. First user
    // request would otherwise pay ~762s for safetensors mmap demand-paging
    // + initial weight cache population. With this prewarm running in
    // parallel, the first user request typically arrives after the cache is
    // warm and returns in ~13s.
    #[cfg(feature = "metal")]
    {
        let state_clone = state.clone();
        tokio::spawn(async move {
            tokio::task::spawn_blocking(move || {
                if let Some(pipeline) = worldmirror_pipeline(&state_clone) {
                    tracing::info!("HY-WorldMirror: pre-warming runtime...");
                    let start = std::time::Instant::now();
                    match pipeline.prewarm() {
                        Ok(()) => tracing::info!(
                            "HY-WorldMirror: pre-warm complete in {:.1}s",
                            start.elapsed().as_secs_f32(),
                        ),
                        Err(e) => tracing::warn!("HY-WorldMirror: pre-warm failed: {e}"),
                    }
                }
            }).await.ok();
        });
    }

    // Build router with all endpoints
    let app = Router::new()
        // Health and metrics
        .route("/health", get(health_check))
        .route("/metrics", get(get_metrics))

        // Image generation
        .route("/api/v1/images/generate", post(generate_image))
        .route("/generate", post(generate_image)) // Legacy endpoint

        // Text generation
        .route("/api/v1/text/generate", post(generate_text))
        .route("/api/v1/text/completions", post(generate_text))

        // Vision-language (Florence-2)
        .route("/api/v1/vision/grounding", post(vision_grounding))

        // 3D generation (Hunyuan3D 2.0). Replaces the previous stub; the
        // handler returns 503 when the pipeline isn't loaded.
        .route("/api/v1/3d/generate", post(generate_3d))

        // 3D world reconstruction (HY-World 2.0 / WorldMirror 2.0). Takes
        // a multi-view image stack, returns a Gaussian-splat archive.
        .route("/api/v1/world/reconstruct", post(world_reconstruct))

        // Video generation (SANA-WM). Image + camera action → mp4.
        .route("/api/v1/video/sana", post(generate_video_sana))

        // Unimplemented modality stubs (return 501)
        .route("/api/v1/audio/generate", post(generate_audio_stub))
        .route("/api/v1/video/generate", post(generate_video_stub))

        // Model management
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/models", post(load_model))
        .route("/api/v1/models/:model_id", get(get_model_info))
        .route("/api/v1/models/:model_id", delete(unload_model))

        // Static assets (video outputs etc.). Serves files from ./static
        // under /static/<...>. Created above as ./static/video for mp4
        // outputs from /api/v1/video/sana.
        .nest_service("/static", tower_http::services::ServeDir::new("static"))

        // Middleware
        .layer(CorsLayer::new()
            .allow_origin(tower_http::cors::Any) // TODO: configure per-environment
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
            .allow_headers([axum::http::header::CONTENT_TYPE])
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("API documentation available at http://{}/ endpoints:", addr);
    tracing::info!("  GET  /health              - Health check");
    tracing::info!("  GET  /metrics             - Server metrics");
    tracing::info!("  POST /api/v1/images/generate - Generate images");
    tracing::info!("  POST /api/v1/text/generate   - Generate text");
    tracing::info!("  GET  /api/v1/models          - List loaded models");
    tracing::info!("  POST /api/v1/models          - Load a model");

    let listener = TcpListener::bind(addr).await
        .map_err(|e| crate::core::Error::internal(format!("Failed to bind to port {}: {}", port, e)))?;

    axum::serve(listener, app).await
        .map_err(|e| crate::core::Error::internal(format!("Server error: {}", e)))?;

    Ok(())
}

// ============================================================================
// Endpoint Handlers
// ============================================================================

/// Health check endpoint.
async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let models = state.models.read().await;
    let metrics = state.metrics.read().await;

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        models_loaded: models.len(),
        memory_usage_bytes: metrics.memory_usage_bytes,
    })
}

/// Get server metrics.
async fn get_metrics(State(state): State<AppState>) -> Json<ServerMetrics> {
    let metrics = state.metrics.read().await;
    Json(metrics.clone())
}

/// Generate image endpoint.
async fn generate_image(
    State(state): State<AppState>,
    Json(payload): Json<ImageGenerationRequest>,
) -> Json<ImageGenerationResponse> {
    tracing::info!("Image generation request: {:?}", payload.prompt);
    let start = Instant::now();

    // Update metrics
    {
        let mut metrics = state.metrics.write().await;
        metrics.total_requests += 1;
    }

    let height = payload.height.unwrap_or(512).min(2048).max(64);
    let width = payload.width.unwrap_or(512).min(2048).max(64);
    let steps = payload.steps.unwrap_or(4);

    if let Some(pipeline) = &state.pipeline {
        use crate::inference::ImageParams;
        use crate::runtime::ResourceMonitor;

        let params = ImageParams {
            height,
            width,
            num_steps: steps,
            seed: payload.seed,
            guidance_scale: payload.guidance_scale.unwrap_or(7.5),
            negative_prompt: payload.negative_prompt.clone(),
            use_lcm: payload.use_lcm.unwrap_or(true),
            use_cfg_pp: false,
            scheduler: None,
        };

        let monitor = ResourceMonitor::new();

        // Resolve any ControlNet inputs the request supplied. Each maps
        // to a (ControlNet, preprocessed control image) pair the pipeline
        // consumes per timestep. Unknown control_type strings or missing
        // CONTROLNET_*_PATH env vars are silently dropped — the diffusion
        // path falls back to vanilla generation in that case.
        #[cfg(feature = "metal")]
        let controls: Vec<(Arc<crate::inference::diffusion::ControlNet>, crate::tensor::Tensor)> = {
            let mut collected = Vec::new();
            if let Some(reqs) = payload.controls.as_ref() {
                for c in reqs {
                    let Some(cn) = controlnet_for(&state, &c.control_type) else {
                        tracing::warn!(
                            "ControlNet '{}' unavailable (env var unset or load failed) — skipping",
                            c.control_type
                        );
                        continue;
                    };
                    match preprocess_control_image(&cn, &c.image_base64, 512) {
                        Ok(t) => collected.push((cn, t)),
                        Err(e) => tracing::warn!(
                            "ControlNet preprocess failed for {}: {e} — skipping",
                            c.control_type
                        ),
                    }
                }
            }
            collected
        };
        #[cfg(not(feature = "metal"))]
        let controls: Vec<(Arc<crate::inference::diffusion::ControlNet>, crate::tensor::Tensor)> = Vec::new();

        match pipeline.generate_with_controls(
            &payload.prompt,
            payload.negative_prompt.as_deref(),
            &params,
            &controls,
            &monitor,
        ).await {
            Ok(output_tensor) => {
                let filename = format!("{}/{}.png", state.config.output_dir, uuid::Uuid::new_v4());

                // Validate output path to prevent path traversal
                let canonical_output_dir = match std::fs::canonicalize(&state.config.output_dir) {
                    Ok(p) => p,
                    Err(e) => {
                        let mut metrics = state.metrics.write().await;
                        metrics.failed_requests += 1;
                        return Json(ImageGenerationResponse {
                            status: "error".to_string(),
                            image_path: None,
                            image_base64: None,
                            execution_time_ms: start.elapsed().as_millis() as u64,
                            steps,
                            error: Some(format!("Invalid output directory: {}", e)),
                        });
                    }
                };
                let resolved_path = canonical_output_dir.join(format!("{}.png", uuid::Uuid::new_v4()));
                if !resolved_path.starts_with(&canonical_output_dir) {
                    let mut metrics = state.metrics.write().await;
                    metrics.failed_requests += 1;
                    return Json(ImageGenerationResponse {
                        status: "error".to_string(),
                        image_path: None,
                        image_base64: None,
                        execution_time_ms: start.elapsed().as_millis() as u64,
                        steps,
                        error: Some("Path traversal detected in output path".to_string()),
                    });
                }
                let filename = resolved_path.to_string_lossy().to_string();

                // Convert tensor to image
                let data: Vec<f16> = output_tensor.to_vec().unwrap_or_else(|_| vec![]);

                if !data.is_empty() {
                    // Extract actual dimensions and layout from tensor shape
                    let tensor_shape = output_tensor.shape();
                    let (h, w, is_planar) = if let (Some(d0), Some(d1), Some(d2)) = (tensor_shape.dim(0), tensor_shape.dim(1), tensor_shape.dim(2)) {
                        if d0 == 3 {
                            // [C, H, W] planar layout
                            (d1 as u32, d2 as u32, true)
                        } else if d2 == 3 {
                            // [H, W, C] interleaved layout
                            (d0 as u32, d1 as u32, false)
                        } else {
                            (height, width, true)
                        }
                    } else {
                        (height, width, true)
                    };
                    let area = (h * w) as usize;

                    let img = image::ImageBuffer::from_fn(w, h, |x, y| {
                        let pixel = (y as usize) * (w as usize) + (x as usize);
                        if pixel >= area { return image::Rgb([0, 0, 0]); }

                        let (r_val, g_val, b_val) = if is_planar {
                            // [C, H, W]: channels are contiguous planes
                            let r = data.get(pixel).map(|v| v.to_f32()).unwrap_or(0.0);
                            let g = data.get(area + pixel).map(|v| v.to_f32()).unwrap_or(0.0);
                            let b = data.get(2 * area + pixel).map(|v| v.to_f32()).unwrap_or(0.0);
                            (r, g, b)
                        } else {
                            // [H, W, C]: channels are interleaved per pixel
                            let base = pixel * 3;
                            let r = data.get(base).map(|v| v.to_f32()).unwrap_or(0.0);
                            let g = data.get(base + 1).map(|v| v.to_f32()).unwrap_or(0.0);
                            let b = data.get(base + 2).map(|v| v.to_f32()).unwrap_or(0.0);
                            (r, g, b)
                        };

                        // VAE decode already runs vae_rescale_output (x*0.5+0.5),
                        // so the tensor is in [0,1]. Don't add another +0.5.
                        let r = (r_val.clamp(0.0, 1.0) * 255.0) as u8;
                        let g = (g_val.clamp(0.0, 1.0) * 255.0) as u8;
                        let b = (b_val.clamp(0.0, 1.0) * 255.0) as u8;

                        image::Rgb([r, g, b])
                    });

                    if let Err(e) = img.save(&filename) {
                        let mut metrics = state.metrics.write().await;
                        metrics.failed_requests += 1;

                        return Json(ImageGenerationResponse {
                            status: "error".to_string(),
                            image_path: None,
                            image_base64: None,
                            execution_time_ms: start.elapsed().as_millis() as u64,
                            steps,
                            error: Some(format!("Failed to save image: {}", e)),
                        });
                    }
                } else {
                    // Fallback: save blank image when tensor data is empty
                    let img = image::RgbImage::new(width, height);
                    let _ = img.save(&filename);
                }

                let execution_time_ms = start.elapsed().as_millis() as u64;

                // Update metrics
                {
                    let mut metrics = state.metrics.write().await;
                    metrics.successful_requests += 1;
                    metrics.total_generation_time_ms += execution_time_ms;
                }

                tracing::info!("Generated image: {} in {}ms", filename, execution_time_ms);

                Json(ImageGenerationResponse {
                    status: "completed".to_string(),
                    image_path: Some(filename),
                    image_base64: None,
                    execution_time_ms,
                    steps,
                    error: None,
                })
            }
            Err(e) => {
                let mut metrics = state.metrics.write().await;
                metrics.failed_requests += 1;

                tracing::error!("Generation failed: {}", e);
                Json(ImageGenerationResponse {
                    status: "error".to_string(),
                    image_path: None,
                    image_base64: None,
                    execution_time_ms: start.elapsed().as_millis() as u64,
                    steps,
                    error: Some(e.to_string()),
                })
            }
        }
    } else {
        let mut metrics = state.metrics.write().await;
        metrics.failed_requests += 1;

        Json(ImageGenerationResponse {
            status: "error".to_string(),
            image_path: None,
            image_base64: None,
            execution_time_ms: start.elapsed().as_millis() as u64,
            steps,
            error: Some("No pipeline available. Load a diffusion model first via POST /api/v1/models".to_string()),
        })
    }
}

/// Generate text endpoint.
async fn generate_text(
    State(state): State<AppState>,
    Json(payload): Json<TextGenerationRequest>,
) -> Json<TextGenerationResponse> {
    tracing::info!("Text generation request: {:?}", payload.prompt);
    let start = Instant::now();

    // Update metrics
    {
        let mut metrics = state.metrics.write().await;
        metrics.total_requests += 1;
    }

    let max_tokens = payload.max_tokens.unwrap_or(256).min(8192).max(1);
    let temperature = {
        let t = payload.temperature.unwrap_or(0.7);
        if t.is_nan() { 1.0 } else { t.clamp(0.0, 2.0) }
    };

    // Use the shared TextHandler for text generation
    let handler = state.text_handler.read().await;

    if !handler.is_model_loaded() {
        let mut metrics = state.metrics.write().await;
        metrics.failed_requests += 1;

        return Json(TextGenerationResponse {
            status: "error".to_string(),
            text: None,
            tokens: None,
            num_tokens: 0,
            tokens_per_second: 0.0,
            time_to_first_token_ms: 0.0,
            execution_time_ms: start.elapsed().as_millis() as u64,
            finish_reason: "error".to_string(),
            error: Some("No LLM model loaded. Set LLM_MODEL_PATH or use POST /api/v1/models to load one.".to_string()),
        });
    }

    let text_input = crate::modalities::text::TextInput {
        content: crate::modalities::text::TextContent::Text(payload.prompt.clone()),
        params: crate::modalities::text::GenerationParams {
            max_tokens,
            temperature,
            top_p: payload.top_p.unwrap_or(0.95),
            top_k: payload.top_k.unwrap_or(50),
            repetition_penalty: 1.0,
            stop_sequences: payload.stop.unwrap_or_default(),
        },
    };

    match handler.generate(text_input).await {
        Ok(output) => {
            let execution_time_ms = start.elapsed().as_millis() as u64;
            let num_tokens = output.stats.generated_tokens;

            // Update metrics
            {
                let mut metrics = state.metrics.write().await;
                metrics.successful_requests += 1;
                metrics.total_generation_time_ms += execution_time_ms;
            }

            Json(TextGenerationResponse {
                status: "completed".to_string(),
                text: Some(output.text),
                tokens: None,
                num_tokens,
                tokens_per_second: output.stats.tokens_per_second,
                time_to_first_token_ms: output.stats.time_to_first_token_ms,
                execution_time_ms,
                finish_reason: "stop".to_string(),
                error: None,
            })
        }
        Err(e) => {
            let mut metrics = state.metrics.write().await;
            metrics.failed_requests += 1;

            Json(TextGenerationResponse {
                status: "error".to_string(),
                text: None,
                tokens: None,
                num_tokens: 0,
                tokens_per_second: 0.0,
                time_to_first_token_ms: 0.0,
                execution_time_ms: start.elapsed().as_millis() as u64,
                finish_reason: "error".to_string(),
                error: Some(format!("Text generation not available: {}. Load a model first via POST /api/v1/models.", e)),
            })
        }
    }
}

/// List loaded models.
async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfo>> {
    let models = state.models.read().await;
    Json(models.values().cloned().collect())
}

/// Load a model.
async fn load_model(
    State(state): State<AppState>,
    Json(payload): Json<ModelLoadRequest>,
) -> impl IntoResponse {
    tracing::info!("Model load request: {} ({})", payload.model_path, payload.model_type);

    let model_id = payload.model_id.unwrap_or_else(|| {
        std::path::Path::new(&payload.model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string())
    });

    // Check if model already loaded
    {
        let models = state.models.read().await;
        if models.contains_key(&model_id) {
            return (
                StatusCode::CONFLICT,
                Json(ModelLoadResponse {
                    status: "error".to_string(),
                    model_id: Some(model_id.clone()),
                    memory_bytes: None,
                    error: Some(format!("Model '{}' already loaded", model_id)),
                }),
            );
        }
    }

    // Calculate actual model size from the weight files on disk
    let memory_bytes = match fs::metadata(&payload.model_path).await {
        Ok(metadata) => metadata.len() as usize,
        Err(_) => {
            // Try as directory: sum all file sizes within it
            let mut total: usize = 0;
            if let Ok(mut entries) = tokio::fs::read_dir(&payload.model_path).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    if let Ok(meta) = entry.metadata().await {
                        if meta.is_file() {
                            total += meta.len() as usize;
                        }
                    }
                }
            }
            total
        }
    };

    // Validate and canonicalize model path
    let model_path = std::path::Path::new(&payload.model_path);
    if !model_path.exists() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ModelLoadResponse {
                status: "error".to_string(),
                model_id: Some(model_id),
                memory_bytes: None,
                error: Some(format!("Model path does not exist: {}", payload.model_path)),
            }),
        );
    }
    let model_path = match model_path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ModelLoadResponse {
                    status: "error".to_string(),
                    model_id: Some(model_id),
                    memory_bytes: None,
                    error: Some(format!("Invalid model path: {}", e)),
                }),
            );
        }
    };

    // Restrict to MODEL_BASE_DIR if set (prevents path traversal)
    if let Ok(base_dir) = std::env::var("MODEL_BASE_DIR") {
        if let Ok(base) = std::path::Path::new(&base_dir).canonicalize() {
            if !model_path.starts_with(&base) {
                return (
                    StatusCode::FORBIDDEN,
                    Json(ModelLoadResponse {
                        status: "error".to_string(),
                        model_id: Some(model_id),
                        memory_bytes: None,
                        error: Some(format!("Model path is outside allowed directory: {}", base.display())),
                    }),
                );
            }
        }
    }

    // Actually load the model into the appropriate inference pipeline
    match payload.model_type.as_str() {
        "llm" | "text" => {
            let mut handler = state.text_handler.write().await;
            if let Err(e) = handler.load_model(&model_path) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ModelLoadResponse {
                        status: "error".to_string(),
                        model_id: Some(model_id),
                        memory_bytes: None,
                        error: Some(format!("Failed to load LLM model: {}", e)),
                    }),
                );
            }
            tracing::info!("LLM model loaded into text handler");
        }
        "diffusion" | "image" => {
            // Diffusion models are loaded at startup via SDXL_MODEL_PATH.
            // Dynamic loading would require rebuilding the DiffusionPipeline,
            // which is not yet supported. Record metadata for now.
            tracing::info!("Diffusion model registered (use SDXL_MODEL_PATH env var for pipeline loading)");
        }
        _ => {
            tracing::info!("Model type '{}' registered (metadata only)", payload.model_type);
        }
    }

    let model_info = ModelInfo {
        id: model_id.clone(),
        model_type: payload.model_type.clone(),
        memory_bytes,
        loaded_at: format!("{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()),
        status: "loaded".to_string(),
    };

    // Update memory metrics
    {
        let mut metrics = state.metrics.write().await;
        metrics.memory_usage_bytes += memory_bytes;
        if metrics.memory_usage_bytes > metrics.peak_memory_bytes {
            metrics.peak_memory_bytes = metrics.memory_usage_bytes;
        }
    }

    {
        let mut models = state.models.write().await;
        models.insert(model_id.clone(), model_info.clone());
    }

    tracing::info!("Model '{}' loaded successfully ({} bytes)", model_id, memory_bytes);

    (
        StatusCode::CREATED,
        Json(ModelLoadResponse {
            status: "success".to_string(),
            model_id: Some(model_id),
            memory_bytes: Some(model_info.memory_bytes),
            error: None,
        }),
    )
}

/// Get model info.
async fn get_model_info(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
    let models = state.models.read().await;

    if let Some(info) = models.get(&model_id) {
        (StatusCode::OK, Json(serde_json::json!(info)))
    } else {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": format!("Model '{}' not found", model_id)
        })))
    }
}

/// Unload a model.
async fn unload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
    tracing::info!("Unloading model: {}", model_id);

    let mut models = state.models.write().await;

    if let Some(info) = models.remove(&model_id) {
        // Update memory metrics
        {
            let mut metrics = state.metrics.write().await;
            metrics.memory_usage_bytes = metrics.memory_usage_bytes.saturating_sub(info.memory_bytes);
        }

        tracing::info!("Model '{}' unloaded successfully", model_id);
        (StatusCode::OK, Json(serde_json::json!({
            "status": "success",
            "message": format!("Model '{}' unloaded", model_id)
        })))
    } else {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": format!("Model '{}' not found", model_id)
        })))
    }
}

/// Audio generation stub (not yet implemented).
async fn generate_audio_stub() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "status": "error",
            "error": "Audio generation is not yet implemented. This endpoint is reserved for future use."
        })),
    )
}

/// Video generation stub (not yet implemented).
async fn generate_video_stub() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "status": "error",
            "error": "Video generation is not yet implemented. This endpoint is reserved for future use."
        })),
    )
}

// ============================================================================
// Vision (Florence-2) — POST /api/v1/vision/grounding
// ============================================================================

/// Florence-2 vision-language endpoint. Runs the requested task
/// (`grounding`, `caption`, `detailed_caption`, `ocr`, `object_detection`)
/// against the provided image. The image is base64-encoded PNG/JPEG.
#[cfg(feature = "metal")]
async fn vision_grounding(
    State(state): State<AppState>,
    Json(payload): Json<VisionGroundingRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    {
        let mut metrics = state.metrics.write().await;
        metrics.total_requests += 1;
    }

    let pipeline = match &state.florence2 {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "error",
                    "error": "Florence-2 not loaded — set FLORENCE2_MODEL_PATH",
                })),
            )
                .into_response()
        }
    };

    let (image_chw, w, h) = match decode_image_to_rgb_chw(&payload.image_base64, 768) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("image decode: {e}"),
                })),
            )
                .into_response()
        }
    };

    // Florence2Pipeline holds Metal handles that aren't Send, so call
    // directly rather than push the closure across a worker thread.
    let result = pipeline.analyze(&image_chw, w, h, &payload.task);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok(text) => {
            let regions = parse_florence2_grounding(&text);
            {
                let mut metrics = state.metrics.write().await;
                metrics.successful_requests += 1;
                metrics.total_generation_time_ms += elapsed_ms;
            }
            (
                StatusCode::OK,
                Json(serde_json::json!(VisionGroundingResponse {
                    status: "success".into(),
                    text: Some(text),
                    regions,
                    image_width: w as u32,
                    image_height: h as u32,
                    execution_time_ms: elapsed_ms,
                    error: None,
                })),
            )
                .into_response()
        }
        Err(e) => {
            let mut metrics = state.metrics.write().await;
            metrics.failed_requests += 1;
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("Florence-2 inference failed: {e}"),
                })),
            )
                .into_response()
        }
    }
}

#[cfg(not(feature = "metal"))]
async fn vision_grounding() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "status": "error",
            "error": "vision/grounding requires metal feature"
        })),
    )
}

// ============================================================================
// 3D (Hunyuan3D 2.0) — POST /api/v1/3d/generate
// ============================================================================

#[cfg(feature = "metal")]
async fn generate_3d(
    State(state): State<AppState>,
    Json(payload): Json<Generate3dRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    {
        let mut metrics = state.metrics.write().await;
        metrics.total_requests += 1;
    }

    let pipeline = match &state.hunyuan3d {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "error",
                    "error": "Hunyuan3D not loaded — set HUNYUAN3D_MODEL_DIR",
                })),
            )
                .into_response()
        }
    };

    let format = payload.format.unwrap_or_else(|| "obj".into());
    if format != "obj" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "status": "error",
                "error": "only `obj` format is supported in this build (glb planned)",
            })),
        )
            .into_response();
    }

    // Hunyuan3D wants 518×518 ImageNet-normalised CHW.
    let image_chw = match decode_image_imagenet_chw(&payload.image_base64, 518) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("image decode: {e}"),
                })),
            )
                .into_response()
        }
    };

    let seed = payload.seed.unwrap_or(42);
    // Same Send-bound caveat as Florence-2 above; call directly.
    let result = pipeline.generate(&image_chw, seed);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok(mesh) => {
            let vertex_count = mesh.vertices.len() as u32;
            let face_count = mesh.faces.len() as u32;
            let obj_text = mesh.to_obj();
            {
                let mut metrics = state.metrics.write().await;
                metrics.successful_requests += 1;
                metrics.total_generation_time_ms += elapsed_ms;
            }
            (
                StatusCode::OK,
                Json(serde_json::json!(Generate3dResponse {
                    status: "success".into(),
                    mesh: Some(obj_text),
                    format: "obj".into(),
                    vertex_count,
                    face_count,
                    execution_time_ms: elapsed_ms,
                    error: None,
                })),
            )
                .into_response()
        }
        Err(e) => {
            let mut metrics = state.metrics.write().await;
            metrics.failed_requests += 1;
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("Hunyuan3D inference failed: {e}"),
                })),
            )
                .into_response()
        }
    }
}

/// Lazily construct a ControlNet for a given control type. Reads the
/// matching `CONTROLNET_*_PATH` env var, loads the safetensors as a
/// `Model`, and wraps it in `ControlNet`. Returns `None` if env not set
/// or load fails (the diffusion path falls back to no-residuals).
#[cfg(feature = "metal")]
fn controlnet_for(
    state: &AppState,
    control_type_str: &str,
) -> Option<Arc<crate::inference::diffusion::ControlNet>> {
    use crate::hal::{LazyLoader, MetalDevice};
    use crate::hal::metal::MetalCompute;
    use crate::inference::diffusion::{ControlNet, ControlType};
    use crate::inference::Model;

    let (slot, env_var, ctype) = match control_type_str.to_ascii_lowercase().as_str() {
        "canny" => (&state.controlnet_canny, "CONTROLNET_CANNY_PATH", ControlType::Canny),
        "scribble" => (&state.controlnet_scribble, "CONTROLNET_SCRIBBLE_PATH", ControlType::Scribble),
        // Other variants pending (Depth/Normal/Pose/Segmentation preprocessors not built).
        _ => return None,
    };

    slot.get_or_init(|| {
        let path_str = std::env::var(env_var).ok()?;
        let path = std::path::Path::new(&path_str);
        if !path.exists() {
            tracing::warn!("{} set but file not found: {}", env_var, path.display());
            return None;
        }
        tracing::info!("Loading ControlNet ({}) from: {}", control_type_str, path.display());
        let device = match MetalDevice::new() {
            Ok(d) => Arc::new(d),
            Err(e) => {
                tracing::error!("ControlNet ({}): MetalDevice failed: {e}", control_type_str);
                return None;
            }
        };
        let loader = Arc::new(LazyLoader::new(device.clone()));
        let model = match Model::load("controlnet", path, loader) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                tracing::error!("ControlNet ({}): model load failed: {e}", control_type_str);
                return None;
            }
        };
        let compute = Arc::new(MetalCompute::new(device));
        let cn = ControlNet::new_with_compute(Some(model), ctype, 1.0, Some(compute));
        Some(Arc::new(cn))
    }).clone()
}

/// Decode a base64-encoded PNG/JPEG control image into a `[3, H, W]` f16
/// tensor (0..1 range), then run the matching ControlNet preprocessor
/// (e.g. Canny edges) on it. Returns the preprocessed tensor ready for
/// `ControlNet::get_conditioning`.
#[cfg(feature = "metal")]
fn preprocess_control_image(
    cn: &crate::inference::diffusion::ControlNet,
    image_b64: &str,
    target_size: usize,
) -> crate::core::Result<crate::tensor::Tensor> {
    use crate::hal::{DeviceId, DeviceType, MetalDevice};
    use crate::tensor::{DType, Shape, Tensor};

    let (raw_chw, width, height) = decode_image_to_rgb_chw(image_b64, target_size as u32)
        .map_err(crate::core::Error::internal)?;
    let device_id = MetalDevice::new()
        .map(|d| d.info().id)
        .unwrap_or(DeviceId::new(DeviceType::Metal, 0));
    let f16: Vec<half::f16> = raw_chw.iter().map(|&v| half::f16::from_f32(v)).collect();
    let tensor = Tensor::from_slice(
        &f16,
        Shape::from([3usize, height, width]),
        DType::F16,
        device_id,
    )?;
    cn.process_control(&tensor)
}

/// Lazily construct the WorldMirror 2.0 pipeline from `HY_WORLDMIRROR_PATH`.
/// Returns `None` if the env var isn't set or the file is missing.
#[cfg(feature = "metal")]
fn worldmirror_pipeline(
    state: &AppState,
) -> Option<Arc<crate::inference::architecture::hyworld::HYWorldPipeline>> {
    state
        .hy_worldmirror
        .get_or_init(|| {
            use crate::inference::architecture::hyworld::{HYWorldConfig, HYWorldPipeline};
            use crate::hal::{LazyLoader, MetalDevice};
            use crate::inference::Model;

            let path_str = std::env::var("HY_WORLDMIRROR_PATH").ok()?;
            let path = std::path::Path::new(&path_str);
            if !path.exists() {
                tracing::warn!("HY_WORLDMIRROR_PATH set but file not found: {}", path.display());
                return None;
            }

            tracing::info!("Loading HY-WorldMirror 2.0 from: {}", path.display());
            let device = match MetalDevice::new() {
                Ok(d) => Arc::new(d),
                Err(e) => {
                    tracing::error!("HY-WorldMirror: MetalDevice init failed: {e}");
                    return None;
                }
            };
            let loader = Arc::new(LazyLoader::new(device.clone()));
            let mirror_model = match Model::load("hy-worldmirror-2.0", path, loader) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    tracing::error!("HY-WorldMirror: model load failed: {e}");
                    return None;
                }
            };
            let config = HYWorldConfig::default_release();
            match HYWorldPipeline::new(
                None, // pano: pending upstream
                None, // nav: pending upstream
                None, // stereo: pending upstream
                Some(mirror_model),
                config,
                device,
            ) {
                Ok(p) => {
                    tracing::info!("HY-WorldMirror 2.0 loaded ({} reconstructor weights)", "ok");
                    Some(Arc::new(p))
                }
                Err(e) => {
                    tracing::error!("HY-WorldMirror: pipeline init failed: {e}");
                    None
                }
            }
        })
        .clone()
}

#[cfg(feature = "metal")]
async fn world_reconstruct(
    State(state): State<AppState>,
    Json(payload): Json<WorldReconstructRequest>,
) -> impl IntoResponse {
    use crate::inference::architecture::hyworld::{encode_splat_archive, WorldQuality};
    use base64::Engine as _;

    let start = Instant::now();
    {
        let mut metrics = state.metrics.write().await;
        metrics.total_requests += 1;
    }

    if payload.images_base64.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "status": "error",
                "error": "images_base64 must contain at least one view",
            })),
        )
            .into_response();
    }

    let pipeline = match worldmirror_pipeline(&state) {
        Some(p) => p,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "error",
                    "error": "HY-WorldMirror 2.0 not loaded — set HY_WORLDMIRROR_PATH",
                })),
            )
                .into_response()
        }
    };

    // Decode each view to ImageNet-normalised CHW. WorldMirror's DINOv2-L/14
    // patch-embed expects 518×518 inputs; smaller multiples-of-14 work but
    // 518 is the trained resolution.
    let mut view_chws = Vec::with_capacity(payload.images_base64.len());
    for (i, b64) in payload.images_base64.iter().enumerate() {
        match decode_image_imagenet_chw(b64, 518) {
            Ok(chw) => view_chws.push(chw),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "status": "error",
                        "error": format!("view {i} decode: {e}"),
                    })),
                )
                    .into_response();
            }
        }
    }

    // Stack views into [N, 3, 518, 518] and pass to the reconstruction
    // entry point. Until the real WorldMirror forward lands, this returns
    // a synthetic spiral splat cloud sized to the configured quality
    // preset — but the wire shape and code path are real.
    let n_views = view_chws.len();
    let mut stacked_f16: Vec<half::f16> = Vec::with_capacity(n_views * 3 * 518 * 518);
    for v in &view_chws {
        for &x in v {
            stacked_f16.push(half::f16::from_f32(x));
        }
    }
    use crate::hal::{DeviceId, DeviceType};
    let device_id = match crate::hal::MetalDevice::new() {
        Ok(d) => d.info().id,
        Err(_) => DeviceId::new(DeviceType::Metal, 0),
    };
    let stacked_tensor = match crate::tensor::Tensor::from_slice(
        &stacked_f16,
        crate::tensor::Shape::from([n_views, 3, 518, 518]),
        crate::tensor::DType::F16,
        device_id,
    ) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("view stack tensor: {e}"),
                })),
            )
                .into_response();
        }
    };

    let _quality = match payload.quality.as_deref() {
        Some("draft") => WorldQuality::Draft,
        Some("cinematic") => WorldQuality::Cinematic,
        _ => WorldQuality::Standard,
    };
    let seed = payload.seed.unwrap_or(42);

    let compute_depth = payload.compute_depth.unwrap_or(false);
    let compute_normals = payload.compute_normals.unwrap_or(false);
    let compute_points = payload.compute_points.unwrap_or(false);

    // v8 pow3r: decode optional conditioning hints. Each hint is per-view;
    // when supplied, the corresponding trained embedder contributes a
    // 1024-dim embedding added to patch tokens before frame_blocks.
    let mut hints = crate::inference::architecture::hyworld::Pow3rHints::default();
    if let Some(d_b64) = &payload.depth_hints_base64 {
        let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(d_b64.len());
        for s in d_b64 {
            match base64::engine::general_purpose::STANDARD.decode(s) {
                Ok(raw) => {
                    if raw.len() % 4 == 0 {
                        let n = raw.len() / 4;
                        let mut v = Vec::with_capacity(n);
                        for i in 0..n {
                            let b = [raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]];
                            v.push(f32::from_le_bytes(b));
                        }
                        decoded.push(v);
                    }
                }
                Err(_) => {}
            }
        }
        if !decoded.is_empty() { hints.depth_per_view = Some(decoded); }
    }
    if let Some(p) = &payload.pose_hints {
        hints.pose_per_view = Some(p.clone());
    }
    if let Some(r) = &payload.ray_hints {
        hints.ray_per_view = Some(r.clone());
    }

    match pipeline.reconstruct_from_views_with_hints(
        &stacked_tensor, seed, compute_depth, compute_normals, compute_points,
        &hints,
    ) {
        Ok(world) => {
            let elapsed = start.elapsed().as_millis() as u64;
            let archive = encode_splat_archive(&world.splats);
            let archive_b64 = base64::engine::general_purpose::STANDARD.encode(&archive);
            let trajectory: Vec<Vec<f32>> = world
                .trajectory
                .iter()
                .map(|p| {
                    vec![
                        p.position[0], p.position[1], p.position[2],
                        p.forward[0],  p.forward[1],  p.forward[2],
                        p.up[0],       p.up[1],       p.up[2],
                        p.fov_deg,
                    ]
                })
                .collect();
            {
                let mut metrics = state.metrics.write().await;
                metrics.successful_requests += 1;
                metrics.total_generation_time_ms += elapsed;
            }
            // Optional secondary head outputs (depth/normal/points). Each
            // is a Vec<Vec<f32>> over views; encode each view's f32 buf as
            // little-endian base64 binary so callers can decode efficiently.
            let encode_views_f32 = |views: &Option<Vec<Vec<f32>>>| -> Option<Vec<String>> {
                views.as_ref().map(|vs| {
                    vs.iter().map(|v| {
                        let mut bytes = Vec::with_capacity(v.len() * 4);
                        for &f in v { bytes.extend_from_slice(&f.to_le_bytes()); }
                        base64::engine::general_purpose::STANDARD.encode(&bytes)
                    }).collect()
                })
            };
            let depth_maps_base64 = encode_views_f32(&world.depth_maps);
            let normal_maps_base64 = encode_views_f32(&world.normal_maps);
            let point_clouds_base64 = encode_views_f32(&world.point_clouds);

            (
                StatusCode::OK,
                Json(serde_json::json!(WorldReconstructResponse {
                    status: "success".into(),
                    splat_archive_base64: Some(archive_b64),
                    splat_count: world.splats.len() as u32,
                    trajectory,
                    execution_time_ms: elapsed,
                    depth_maps_base64,
                    normal_maps_base64,
                    point_clouds_base64,
                    error: None,
                })),
            )
                .into_response()
        }
        Err(e) => {
            let mut metrics = state.metrics.write().await;
            metrics.failed_requests += 1;
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("WorldMirror reconstruction failed: {e}"),
                })),
            )
                .into_response()
        }
    }
}

#[cfg(not(feature = "metal"))]
async fn world_reconstruct() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "status": "error",
            "error": "world reconstruction requires the metal feature",
        })),
    )
        .into_response()
}

#[cfg(not(feature = "metal"))]
async fn generate_3d() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "status": "error",
            "error": "3d/generate requires metal feature"
        })),
    )
}

// ============================================================================
// Video (SANA-WM) — POST /api/v1/video/sana
// ============================================================================

#[cfg(feature = "metal")]
async fn generate_video_sana(
    State(state): State<AppState>,
    Json(payload): Json<GenerateVideoSanaRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    {
        let mut metrics = state.metrics.write().await;
        metrics.total_requests += 1;
    }

    let pipeline = match &state.sana_wm {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "error",
                    "error": "SANA-WM not loaded — set SANA_WM_MODEL_DIR",
                })),
            )
                .into_response()
        }
    };

    // SANA-WM expects [3, image_size, image_size] f32 in [0, 1]. The
    // pipeline's image_size is 720 (matches SanaWmConfig::default()).
    // Use the EXACT (force-square) decoder — aspect-preserve would leave
    // arbitrary input shapes that crash vae_encode downstream.
    let image_size: u32 = 720;
    let image_chw = match decode_image_to_rgb_chw_exact(&payload.image_base64, image_size) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("image decode: {e}"),
                })),
            )
                .into_response()
        }
    };

    let seed = payload.seed.unwrap_or(42);
    let prompt = payload.prompt.clone();
    let action = payload.action.clone();

    let result = pipeline.generate(&image_chw, &prompt, &action, seed);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    match result {
        Ok(video) => {
            let uuid = uuid::Uuid::new_v4();
            let mp4_path = std::path::PathBuf::from(format!("static/video/{}.mp4", uuid));
            if let Err(e) = encode_rgb_frames_to_mp4(
                &video.frames,
                video.num_frames,
                video.width,
                video.height,
                video.fps,
                &mp4_path,
            ) {
                let mut metrics = state.metrics.write().await;
                metrics.failed_requests += 1;
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "status": "error",
                        "error": format!("mp4 encode failed: {e}"),
                    })),
                )
                    .into_response();
            }
            let resolution = match video.height {
                h if h >= 1080 => "1080p".to_string(),
                h if h >= 720 => "720p".to_string(),
                h if h >= 480 => "480p".to_string(),
                h => format!("{}p", h),
            };
            let duration_s = if video.fps > 0.0 {
                video.num_frames as f32 / video.fps
            } else { 0.0 };
            {
                let mut metrics = state.metrics.write().await;
                metrics.successful_requests += 1;
                metrics.total_generation_time_ms += elapsed_ms;
            }
            (
                StatusCode::OK,
                Json(serde_json::json!(GenerateVideoSanaResponse {
                    status: "success".into(),
                    video_url: Some(format!("/static/video/{}.mp4", uuid)),
                    frames: video.num_frames as u32,
                    resolution,
                    duration_s,
                    execution_time_ms: elapsed_ms,
                    error: None,
                })),
            )
                .into_response()
        }
        Err(e) => {
            let mut metrics = state.metrics.write().await;
            metrics.failed_requests += 1;
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "error": format!("SANA-WM inference failed: {e}"),
                })),
            )
                .into_response()
        }
    }
}

#[cfg(not(feature = "metal"))]
async fn generate_video_sana() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "status": "error",
            "error": "video/sana requires metal feature"
        })),
    )
}

/// Encode flat T·H·W·3 u8 RGB frames to an H.264 mp4 by piping to
/// `ffmpeg` on stdin. Requires `ffmpeg` on PATH.
fn encode_rgb_frames_to_mp4(
    frames: &[u8],
    num_frames: usize,
    width: usize,
    height: usize,
    fps: f32,
    out_path: &std::path::Path,
) -> std::result::Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let expected = num_frames.checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| "frame size overflow".to_string())?;
    if frames.len() != expected {
        return Err(format!(
            "frame buffer length {} ≠ T·H·W·3 = {}",
            frames.len(), expected
        ));
    }

    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", &format!("{}x{}", width, height),
            "-r", &format!("{}", fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])
        .arg(out_path.as_os_str())
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn ffmpeg: {e}"))?;

    {
        let stdin = child.stdin.as_mut()
            .ok_or_else(|| "ffmpeg stdin not captured".to_string())?;
        stdin.write_all(frames).map_err(|e| format!("write frames: {e}"))?;
    }

    let out = child.wait_with_output()
        .map_err(|e| format!("wait ffmpeg: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("ffmpeg exit {:?}: {}", out.status.code(), stderr));
    }
    Ok(())
}

// ============================================================================
// Image preprocessing helpers
// ============================================================================

/// Decode a base64-encoded PNG/JPEG into a flat `[3 * H * W]` f32 RGB
/// array in `[C, H, W]` order, normalised to `[0, 1]`. The image is
/// resized to fit within `max_side` while preserving aspect ratio (the
/// shorter side may be smaller). Returns `(chw, width, height)`.
fn decode_image_to_rgb_chw(
    b64: &str,
    max_side: u32,
) -> std::result::Result<(Vec<f32>, usize, usize), String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(b64.trim()))
        .map_err(|e| format!("base64: {e}"))?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("image: {e}"))?;
    let img = if img.width() > max_side || img.height() > max_side {
        img.resize(max_side, max_side, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            chw[0 * h * w + y * w + x] = p[0] as f32 / 255.0;
            chw[1 * h * w + y * w + x] = p[1] as f32 / 255.0;
            chw[2 * h * w + y * w + x] = p[2] as f32 / 255.0;
        }
    }
    Ok((chw, w, h))
}

/// Decode a base64-encoded PNG/JPEG into a `[3 * size * size]` f32 CHW
/// array in `[0, 1]`, **force-resized to exactly `size × size`** (does
/// NOT preserve aspect ratio). SANA-WM's VAE encoder requires a square
/// input matching the trained `image_size` (default 720); aspect-
/// preserving resize would leave shape mismatches that panic downstream.
fn decode_image_to_rgb_chw_exact(
    b64: &str,
    size: u32,
) -> std::result::Result<Vec<f32>, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(b64.trim()))
        .map_err(|e| format!("base64: {e}"))?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("image: {e}"))?;
    let img = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    debug_assert_eq!(w, size as usize);
    debug_assert_eq!(h, size as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            chw[0 * h * w + y * w + x] = p[0] as f32 / 255.0;
            chw[1 * h * w + y * w + x] = p[1] as f32 / 255.0;
            chw[2 * h * w + y * w + x] = p[2] as f32 / 255.0;
        }
    }
    Ok(chw)
}

/// Decode a base64-encoded image into a fixed-size `[3 * size * size]`
/// f32 CHW array, ImageNet-normalised (mean `[0.485, 0.456, 0.406]`,
/// std `[0.229, 0.224, 0.225]`). Used by Hunyuan3D 2.0's DINOv2-Giant
/// front-end which expects 518×518 input.
fn decode_image_imagenet_chw(b64: &str, size: u32) -> std::result::Result<Vec<f32>, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(b64.trim()))
        .map_err(|e| format!("base64: {e}"))?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("image: {e}"))?;
    let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    let s = size as usize;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut chw = vec![0f32; 3 * s * s];
    for y in 0..s {
        for x in 0..s {
            let p = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                chw[c * s * s + y * s + x] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok(chw)
}

/// Parse Florence-2's grounding/object-detection text into structured
/// regions. Florence-2 emits location tokens of the form
/// `<loc_X><loc_Y><loc_X><loc_Y>` (each integer 0–999) following the
/// label, e.g. `tree<loc_120><loc_45><loc_350><loc_810>`. We tolerate
/// tasks that don't emit location tokens by returning an empty list —
/// callers fall back to the raw `text` field.
fn parse_florence2_grounding(text: &str) -> Vec<GroundedRegion> {
    let mut out = Vec::new();
    // Walk the string, splitting on `<loc_*>` markers. Whatever non-marker
    // text precedes a 4-tuple of location tokens is the label.
    let mut cursor = 0;
    let bytes = text.as_bytes();
    let mut buf_label = String::new();
    let mut tokens: Vec<u32> = Vec::with_capacity(4);

    while cursor < bytes.len() {
        if bytes[cursor..].starts_with(b"<loc_") {
            let end = match text[cursor..].find('>') {
                Some(e) => cursor + e,
                None => break,
            };
            let token_str = &text[cursor + 5..end];
            if let Ok(n) = token_str.parse::<u32>() {
                tokens.push(n);
                if tokens.len() == 4 {
                    let label = buf_label.trim().trim_matches(|c: char| !c.is_alphanumeric()).to_string();
                    if !label.is_empty() {
                        out.push(GroundedRegion {
                            label,
                            bbox: [
                                tokens[0] as f32 / 999.0,
                                tokens[1] as f32 / 999.0,
                                tokens[2] as f32 / 999.0,
                                tokens[3] as f32 / 999.0,
                            ],
                            score: 1.0,
                        });
                    }
                    buf_label.clear();
                    tokens.clear();
                }
            }
            cursor = end + 1;
        } else {
            buf_label.push(bytes[cursor] as char);
            cursor += 1;
        }
    }
    out
}

/// Legacy 3D stub — kept for callers that hit the old route name.
async fn _unused_generate_3d_stub() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "status": "error",
            "error": "3D generation is not yet implemented. This endpoint is reserved for future use."
        })),
    )
}
