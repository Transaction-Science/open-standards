//! Unified inference engine for real-time generation.
//!
//! This module provides the high-level API for running inference
//! across all modalities with optimal resource usage.
//!
//! ## Design Goals
//!
//! 1. **Sub-100ms latency** for all modalities
//! 2. **Streaming output** for real-time feedback
//! 3. **Minimal memory footprint** via lazy loading
//! 4. **Zero-copy on UMA** (Apple Silicon)
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │                         Engine                                │
//! │      (High-level API, model management, sessions)             │
//! ├───────────────────────────────────────────────────────────────┤
//! │                    Inference Pipelines                        │
//! │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐ │
//! │  │    LLM    │ │ Diffusion │ │  Video    │ │    Audio      │ │
//! │  │ - Prefill │ │ - LCM     │ │ - I2V     │ │ - TTS         │ │
//! │  │ - Decode  │ │ - IP-Adpt │ │ - T2V     │ │ - AudioLDM    │ │
//! │  │ - KVCache │ │ - CtrlNet │ │ - Interp  │ │ - Music       │ │
//! │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘ │
//! │  ┌───────────────────────────────────────────────────────┐   │
//! │  │               Gaussian 3D                              │   │
//! │  │  - Image-to-3D  - Splatting  - Mesh Export            │   │
//! │  └───────────────────────────────────────────────────────┘   │
//! ├───────────────────────────────────────────────────────────────┤
//! │                       Tokenizer                               │
//! │           (BPE, SentencePiece, Chat Templates)                │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use efficient_genai::inference::Engine;
//!
//! let engine = Engine::new()?;
//!
//! // Load model lazily
//! engine.load_model("llama-7b", ModelConfig::default())?;
//!
//! // Stream tokens
//! let stream = engine.generate_text("Hello", TextParams::default());
//! while let Some(token) = stream.next().await {
//!     print!("{}", token.text);
//! }
//! ```

pub mod engine;
pub mod config;
pub mod model;
mod session;
pub mod llm;
pub mod ggml_graph;
pub mod diffusion;
mod gaussian3d;
mod video;
mod audio;
pub mod whisper;
pub mod musicgen;
pub mod tokenizer;
pub mod formats;
#[cfg(feature = "metal")]
pub mod gpu_ops;
/// Model architecture implementations.
pub mod architecture {
    pub mod unet;
    pub mod flux;
    pub mod t5;
    pub mod dit;
    pub mod auraflow;
    pub mod pixart;
    pub mod chatglm;
    pub mod wan;
    pub mod deepseek;
    pub mod triposr;
    pub mod f5tts;
    pub mod hunyuanvideo;
    pub mod audiogen;
    pub mod flan_t5;
    pub mod trellis;
    pub mod instantmesh;
    pub mod triposg;
    pub mod hunyuan3d;
    pub mod sana_wm;
    pub mod hyworld;
    pub mod florence2;
    pub mod controlnet_forward;
    pub mod worldmirror_forward;
    pub mod sharp;
}

// Core engine
pub use engine::{Engine, TextToken, ImageProgress, Object3D, Camera3D, CameraPath, Mesh};
pub use model::{Model, ModelType, ModelInfo, ModelConfig, Quantization};
pub use session::{Session, SessionConfig};
pub use config::{InferenceConfig, TextParams, ImageParams, ThreeDParams, VideoParams, AudioParams, MeshFormat, LLMConfig, Architecture};

// LLM pipeline
pub use llm::{LLMPipeline, PagedKVCache, MoeRouter, group_by_expert};

// Diffusion pipeline
pub use diffusion::{DiffusionPipeline, DiffusionScheduler, IPAdapter, IPAdapterConfig, ControlNet, ControlType, ModelPredictionType};
pub use diffusion::{DiffusionBackbone, DiTVariant, TextEncoderType, VaeVariant};

// 3D pipeline
pub use gaussian3d::{Gaussian3DPipeline, Gaussian3DProgress, CameraController};
pub use gaussian3d::{compute_view_matrix, compute_projection_matrix};

// Video pipeline
pub use video::{VideoPipeline, Video, VideoFrame, VideoProgress, VideoEditor, OpticalFlow};

// Audio pipeline
pub use audio::{AudioPipeline, Audio, AudioChunk, VoiceConfig, MusicParams, AudioEffects, MelSpectrogram};

// Whisper ASR pipeline
pub use whisper::{WhisperPipeline, WhisperConfig};

// Flux architecture
pub use architecture::flux::{FluxTransformer, FluxConfig, AdaLNModulation};
#[cfg(feature = "metal")]
pub use architecture::flux::FluxGpuTransformer;

// T5 text encoder
#[cfg(feature = "metal")]
pub use architecture::t5::{T5Encoder, T5Config};

// AuraFlow architecture
#[cfg(feature = "metal")]
pub use architecture::auraflow::{AuraFlowTransformer, AuraFlowConfig};

// PixArt-Sigma architecture
#[cfg(feature = "metal")]
pub use architecture::pixart::{PixArtGpuTransformer, PixArtConfig};

// ChatGLM text encoder
#[cfg(feature = "metal")]
pub use architecture::chatglm::{ChatGLMEncoder, ChatGLMConfig};

// MusicGen pipeline
#[cfg(feature = "metal")]
pub use musicgen::{MusicGenPipeline, MusicGenConfig};

// Wan2.1 video DiT + VAE
#[cfg(feature = "metal")]
pub use architecture::wan::{WanDiT, WanConfig, WanVaeConfig, WanVaeDecoder};

// DeepSeek V2 (MLA + MoE)
#[cfg(feature = "metal")]
pub use architecture::deepseek::{DeepSeekV2Pipeline, DeepSeekV2Config};

// TripoSR (3D reconstruction)
#[cfg(feature = "metal")]
pub use architecture::triposr::{TripoSRPipeline, TripoSRConfig};

// F5-TTS (DiT TTS) + Vocos/HiFi-GAN vocoders
#[cfg(feature = "metal")]
pub use architecture::f5tts::{F5TTSPipeline, F5TTSConfig, VocosPipeline, HiFiGANPipeline};

// HunyuanVideo (DiT video)
#[cfg(feature = "metal")]
pub use architecture::hunyuanvideo::{HunyuanVideoPipeline, HunyuanVideoConfig, HunyuanVideoVaeConfig};

// AudioGen / MAGNet (audio generation)
#[cfg(feature = "metal")]
pub use architecture::audiogen::{AudioGenPipeline, AudioGenConfig, MAGNetPipeline, MAGNetConfig};

// Flan-T5 (encoder-decoder)
#[cfg(feature = "metal")]
pub use architecture::flan_t5::{FlanT5Pipeline, FlanT5Config};

// Trellis (image-to-3D via two-stage flow matching)
#[cfg(feature = "metal")]
pub use architecture::trellis::{TrellisPipeline, TrellisConfig, GaussianOutput};

// InstantMesh (image-to-3D mesh)
#[cfg(feature = "metal")]
pub use architecture::instantmesh::{InstantMeshPipeline, InstantMeshConfig, MeshOutput};

// TripoSG (image-to-3D watertight mesh via rectified flow)
#[cfg(feature = "metal")]
pub use architecture::triposg::{TripoSGPipeline, TripoSGConfig};

// Hunyuan3D 2.0 (image-to-3D shape via flow-matching DiT)
#[cfg(feature = "metal")]
pub use architecture::hunyuan3d::{Hunyuan3DPipeline, Hunyuan3DConfig};

// SANA-WM (image + camera action → video via Gated DeltaNet DiT + LTX-2 VAE)
#[cfg(feature = "metal")]
pub use architecture::sana_wm::{SanaWmPipeline, SanaWmConfig, VideoOutput as SanaWmVideoOutput};

// Apple SHARP (fast single-image Gaussian splats)
#[cfg(feature = "metal")]
pub use architecture::sharp::{SharpPipeline, SharpConfig};

// Tokenizer
pub use tokenizer::{Tokenizer, TokenizerType, EncodingResult, BatchEncoder, ChatTemplate, ChatMessage, ChatRole};

