//! Model library scanner and catalog for the dashboard server.
//!
//! Scans a directory tree to auto-detect models, their types, architectures,
//! and sizes. Provides a catalog API for the dashboard to browse and load models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// Types
// ============================================================================

/// Model modality category for UI organization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModelCategory {
    /// Large language models (GPT, LLaMA, Qwen, etc.)
    Llm,
    /// Image diffusion models (SDXL, Flux, etc.)
    Diffusion,
    /// Audio generation (Bark)
    AudioGeneration,
    /// Speech-to-text (Whisper, Moonshine)
    SpeechToText,
    /// Text-to-speech (Kokoro, Chatterbox, Parler-TTS)
    TextToSpeech,
    /// Video generation (Wan, HunyuanVideo, Mochi)
    VideoGeneration,
    /// 3D generation (TripoSR, SAM2, etc.)
    ThreeD,
    /// Text/image encoders (CLIP, T5, E5)
    Encoder,
    /// Audio vocoders (EnCodec, DAC)
    Vocoder,
    /// Unrecognized
    Other,
}

/// Loading status of a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// On disk, not loaded
    Available,
    /// Currently being loaded
    Loading,
    /// In GPU memory, ready for inference
    Loaded,
    /// Failed to load
    Error,
}

/// A single model entry in the library catalog.
#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    /// Unique identifier (directory name)
    pub id: String,
    /// Human-readable display name
    pub name: String,
    /// Modality category
    pub category: ModelCategory,
    /// Architecture string (e.g. "Qwen2ForCausalLM")
    pub architecture: String,
    /// Pipeline type for loading ("llm", "diffusion", "whisper", etc.)
    pub pipeline_type: String,
    /// Total size on disk in bytes
    pub size_bytes: u64,
    /// Human-readable size (e.g. "953 MB")
    pub size_display: String,
    /// Absolute path to the model directory
    pub path: PathBuf,
    /// Whether this model has a tokenizer (tokenizer.json or tokenizer.model)
    pub has_tokenizer: bool,
    /// Whether this model has safetensors weights
    pub has_safetensors: bool,
    /// Whether this model has GGUF weights
    pub has_gguf: bool,
    /// Key config values from config.json
    pub config_summary: HashMap<String, serde_json::Value>,
    /// Current loading status
    pub status: ModelStatus,
    /// Whether this model can be loaded by the server
    pub auto_loadable: bool,
    /// Weight format: "safetensors", "gguf", "diffusion_pipeline", "checkpoint"
    pub format: String,
}

/// The full model library catalog.
pub struct ModelLibrary {
    /// All discovered models, keyed by ID
    pub models: HashMap<String, CatalogEntry>,
    /// Library root path
    pub root_path: PathBuf,
    /// When the library was last scanned
    pub scanned_at: std::time::SystemTime,
}

impl ModelLibrary {
    /// Create an empty library (when no library path is available).
    pub fn empty() -> Self {
        Self {
            models: HashMap::new(),
            root_path: PathBuf::new(),
            scanned_at: std::time::SystemTime::now(),
        }
    }

    /// Total memory used by all currently loaded models.
    pub fn total_loaded_memory(&self) -> u64 {
        self.models
            .values()
            .filter(|e| e.status == ModelStatus::Loaded)
            .map(|e| e.size_bytes)
            .sum()
    }
}

// ============================================================================
// Scanner
// ============================================================================

/// Directories to skip during scanning (not models).
const SKIP_DIRS: &[&str] = &[
    "clusters", "mlx-converted", "huggingface_hub", "models",
    "mlx_models", ".cache", "__pycache__",
];

/// Scan a directory and build the model catalog.
///
/// Iterates one level deep — each subdirectory is treated as a potential model.
pub fn scan_library(root: &Path) -> ModelLibrary {
    let mut models = HashMap::new();

    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("Failed to read model library at {}: {}", root.display(), e);
            return ModelLibrary {
                models,
                root_path: root.to_path_buf(),
                scanned_at: std::time::SystemTime::now(),
            };
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Skip hidden dirs and known non-model dirs
        if dir_name.starts_with('.') || SKIP_DIRS.iter().any(|s| dir_name == *s) {
            continue;
        }

        if let Some(entry) = detect_model_entry(&path, &dir_name) {
            tracing::debug!(
                "Discovered model: {} ({:?}, {})",
                entry.id,
                entry.category,
                entry.size_display
            );
            models.insert(entry.id.clone(), entry);
        }
    }

    tracing::info!(
        "Model library scan complete: {} models found in {}",
        models.len(),
        root.display()
    );

    ModelLibrary {
        models,
        root_path: root.to_path_buf(),
        scanned_at: std::time::SystemTime::now(),
    }
}

/// Detect a model from a directory and create a catalog entry.
fn detect_model_entry(dir: &Path, dir_name: &str) -> Option<CatalogEntry> {
    let dir_lower = dir_name.to_lowercase();

    // Check what files exist
    let has_model_index = dir.join("model_index.json").exists();
    let has_config = dir.join("config.json").exists();
    let has_unet = dir.join("unet").is_dir();
    let has_vae = dir.join("vae").is_dir();
    let has_transformer = dir.join("transformer").is_dir();
    let has_tokenizer_json = dir.join("tokenizer.json").exists();
    let has_tokenizer_model = dir.join("tokenizer.model").exists();
    let has_tokenizer = has_tokenizer_json || has_tokenizer_model;

    // Check for safetensors files
    let has_safetensors = has_any_file_with_ext(dir, "safetensors");
    let has_gguf = has_any_file_with_ext(dir, "gguf");

    // Skip directories with no model files at all
    if !has_safetensors && !has_gguf && !has_model_index && !has_config {
        return None;
    }

    // Read config.json if available
    let config_json = if has_config {
        std::fs::read_to_string(dir.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    } else {
        None
    };

    // Determine category, architecture, and pipeline type
    let (category, architecture, pipeline_type, format) =
        classify_model(dir, &dir_lower, has_model_index, has_unet, has_vae,
                       has_transformer, has_gguf, config_json.as_ref());

    // Calculate size
    let size_bytes = dir_size_bytes(dir);
    let size_display = format_size(size_bytes);

    // Extract config summary
    let config_summary = config_json
        .as_ref()
        .map(extract_config_summary)
        .unwrap_or_default();

    // Determine if auto-loadable
    // GPTQ/AWQ/GGUF-quantized models are NOT auto-loadable — our pipeline
    // doesn't support dequantization and will produce garbage output.
    let is_quantized = config_json
        .as_ref()
        .map(|c| c.get("quantization_config").is_some())
        .unwrap_or(false)
        || dir_name.to_lowercase().contains("gptq")
        || dir_name.to_lowercase().contains("awq");

    let auto_loadable = match category {
        ModelCategory::Llm => has_safetensors && has_tokenizer && !is_quantized,
        ModelCategory::Diffusion => has_safetensors,
        ModelCategory::SpeechToText => has_safetensors && has_tokenizer,
        ModelCategory::AudioGeneration => has_safetensors,
        ModelCategory::TextToSpeech => has_safetensors,
        ModelCategory::VideoGeneration => has_safetensors,
        ModelCategory::ThreeD => has_safetensors,
        _ => false,
    };

    // Generate display name from directory name
    let name = prettify_name(dir_name);

    Some(CatalogEntry {
        id: dir_name.to_string(),
        name,
        category,
        architecture,
        pipeline_type,
        size_bytes,
        size_display,
        path: dir.to_path_buf(),
        has_tokenizer,
        has_safetensors,
        has_gguf,
        config_summary,
        status: ModelStatus::Available,
        auto_loadable,
        format,
    })
}

/// Classify a model directory into category, architecture, pipeline type, and format.
fn classify_model(
    dir: &Path,
    dir_lower: &str,
    has_model_index: bool,
    has_unet: bool,
    has_vae: bool,
    has_transformer: bool,
    has_gguf: bool,
    config: Option<&serde_json::Value>,
) -> (ModelCategory, String, String, String) {
    // Priority 1: model_index.json → Diffusion pipeline
    if has_model_index {
        let arch = read_model_index_class(dir).unwrap_or_else(|| "DiffusionPipeline".into());
        return (ModelCategory::Diffusion, arch, "diffusion".into(), "diffusion_pipeline".into());
    }

    // Priority 2: unet/ + vae/ subdirs → Diffusion pipeline
    if has_unet && has_vae {
        return (ModelCategory::Diffusion, "UNet+VAE".into(), "diffusion".into(), "diffusion_pipeline".into());
    }

    // Priority 2b: transformer/ + vae/ → DiT diffusion pipeline
    if has_transformer && has_vae {
        return (ModelCategory::Diffusion, "DiT+VAE".into(), "diffusion".into(), "diffusion_pipeline".into());
    }

    // Priority 3: config.json architectures field
    if let Some(config) = config {
        if let Some(arch) = config.get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
        {
            let result = classify_by_architecture(arch, dir_lower);
            if result.0 != ModelCategory::Other {
                return result;
            }
        }

        // Also check model_type field
        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            let result = classify_by_model_type(model_type, dir_lower);
            if result.0 != ModelCategory::Other {
                return result;
            }
        }
    }

    // Priority 4: Directory name heuristics
    if let Some(result) = classify_by_dirname(dir_lower) {
        return result;
    }

    // Priority 5: GGUF files → likely LLM
    if has_gguf {
        return (ModelCategory::Llm, "GGUF".into(), "llm".into(), "gguf".into());
    }

    (ModelCategory::Other, "unknown".into(), "unknown".into(), "safetensors".into())
}

/// Classify based on the HuggingFace `architectures[0]` value.
fn classify_by_architecture(arch: &str, dir_lower: &str) -> (ModelCategory, String, String, String) {
    let arch_s = arch.to_string();
    let fmt = "safetensors".to_string();

    // CausalLM models → LLM
    if arch.ends_with("ForCausalLM") || arch.ends_with("LMHeadModel") {
        return (ModelCategory::Llm, arch_s, "llm".into(), fmt);
    }

    // Encoder-decoder models
    match arch {
        "WhisperForConditionalGeneration" => {
            return (ModelCategory::SpeechToText, arch_s, "whisper".into(), fmt);
        }
        "BarkModel" | "BarkForCausalLM" => {
            return (ModelCategory::AudioGeneration, arch_s, "bark".into(), fmt);
        }
        "T5ForConditionalGeneration" => {
            // Flan-T5 is a standalone LLM; plain T5 is an encoder sub-component
            let category = if dir_lower.contains("flan") {
                ModelCategory::Llm
            } else {
                ModelCategory::Encoder
            };
            return (category, arch_s, "t5".into(), fmt);
        }
        "EncodecModel" => {
            return (ModelCategory::Vocoder, arch_s, "codec".into(), fmt);
        }
        _ => {}
    }

    // CLIP / SigLIP → Encoder
    if arch.contains("CLIP") || arch.contains("Siglip") || arch.contains("SigLIP") {
        return (ModelCategory::Encoder, arch_s, "encoder".into(), fmt);
    }

    // Mamba
    if arch.contains("Mamba") {
        return (ModelCategory::Llm, arch_s, "llm".into(), fmt);
    }

    (ModelCategory::Other, arch_s, "unknown".into(), fmt)
}

/// Classify based on the HuggingFace `model_type` field.
fn classify_by_model_type(model_type: &str, dir_lower: &str) -> (ModelCategory, String, String, String) {
    let fmt = "safetensors".to_string();
    let mt = model_type.to_string();

    match model_type {
        "llama" | "mistral" | "qwen2" | "phi" | "phi3" | "gemma" | "gemma2"
        | "gpt2" | "gpt_neox" | "deepseek_v2" | "jamba" | "mamba" | "mamba2" => {
            (ModelCategory::Llm, mt, "llm".into(), fmt)
        }
        "whisper" => (ModelCategory::SpeechToText, mt, "whisper".into(), fmt),
        "bark" => (ModelCategory::AudioGeneration, mt, "bark".into(), fmt),
        "t5" => {
            let cat = if dir_lower.contains("flan") {
                ModelCategory::Llm
            } else {
                ModelCategory::Encoder
            };
            (cat, mt, "t5".into(), fmt)
        }
        "clip" | "siglip" => (ModelCategory::Encoder, mt, "encoder".into(), fmt),
        "encodec" => (ModelCategory::Vocoder, mt, "codec".into(), fmt),
        _ => (ModelCategory::Other, mt, "unknown".into(), fmt),
    }
}

/// Classify based on directory name patterns.
fn classify_by_dirname(dir_lower: &str) -> Option<(ModelCategory, String, String, String)> {
    let fmt = "safetensors".to_string();

    // Diffusion
    for kw in &["sdxl", "flux", "auraflow", "pixart", "kolors", "stable-diffusion", "sd3", "sd-3", "fibo"] {
        if dir_lower.contains(kw) {
            return Some((ModelCategory::Diffusion, kw.to_string(), "diffusion".into(), fmt));
        }
    }

    // Video
    for kw in &["wan", "hunyuanvideo"] {
        if dir_lower.contains(kw) {
            return Some((ModelCategory::VideoGeneration, kw.to_string(), "video".into(), fmt));
        }
    }

    // Audio generation
    for kw in &["bark"] {
        if dir_lower.contains(kw) {
            return Some((ModelCategory::AudioGeneration, kw.to_string(), kw.to_string(), fmt));
        }
    }

    // Speech-to-text
    if dir_lower.contains("whisper") {
        return Some((ModelCategory::SpeechToText, "whisper".into(), "whisper".into(), fmt));
    }

    // TTS
    for kw in &["kokoro", "chatterbox", "parler-tts", "sesame-csm", "outetts", "tts"] {
        if dir_lower.contains(kw) {
            return Some((ModelCategory::TextToSpeech, kw.to_string(), "tts".into(), fmt));
        }
    }

    // 3D
    for kw in &["triposr", "trellis", "instantmesh", "hunyuan3d", "triposg", "sharp"] {
        if dir_lower.contains(kw) {
            return Some((ModelCategory::ThreeD, kw.to_string(), "3d".into(), fmt));
        }
    }

    // Vocoders
    for kw in &["encodec", "dac-codec"] {
        if dir_lower.contains(kw) {
            return Some((ModelCategory::Vocoder, kw.to_string(), "vocoder".into(), fmt));
        }
    }

    // Encoders
    for kw in &["clip", "e5-", "t5-", "t5_"] {
        if dir_lower.contains(kw) {
            return Some((ModelCategory::Encoder, kw.to_string(), "encoder".into(), fmt));
        }
    }

    None
}

/// Read the `_class_name` from model_index.json.
fn read_model_index_class(dir: &Path) -> Option<String> {
    let content = std::fs::read_to_string(dir.join("model_index.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("_class_name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

// ============================================================================
// Helpers
// ============================================================================

/// Check if a directory contains any file with the given extension.
fn has_any_file_with_ext(dir: &Path, ext: &str) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(e) = path.extension().and_then(|e| e.to_str()) {
                    if e == ext {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Calculate total size of a directory recursively.
fn dir_size_bytes(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_file() {
                if let Ok(meta) = entry_path.metadata() {
                    total += meta.len();
                }
            } else if entry_path.is_dir() {
                total += dir_size_bytes(&entry_path);
            }
        }
    }
    total
}

/// Format byte count as human-readable string.
fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;
    const KB: u64 = 1_024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Convert a directory name to a human-readable display name.
fn prettify_name(dir_name: &str) -> String {
    dir_name
        .replace('_', " ")
        .replace('-', " ")
        .split_whitespace()
        .map(|word| {
            // Keep version numbers and abbreviations as-is
            if word.chars().all(|c| c.is_ascii_digit() || c == '.')
                || word.len() <= 3
                || word.chars().all(|c| c.is_uppercase() || c.is_ascii_digit())
            {
                word.to_string()
            } else {
                // Capitalize first letter
                let mut chars = word.chars();
                match chars.next() {
                    Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract key config values for display.
fn extract_config_summary(config: &serde_json::Value) -> HashMap<String, serde_json::Value> {
    let keys = [
        "architectures", "model_type", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
        "max_position_embeddings", "torch_dtype", "intermediate_size",
        "num_local_experts", "quantization_config",
    ];

    let mut summary = HashMap::new();
    for &key in &keys {
        if let Some(val) = config.get(key) {
            summary.insert(key.to_string(), val.clone());
        }
    }
    summary
}

// ============================================================================
// Default Selection
// ============================================================================

/// Select the best default models to load from the library.
///
/// Returns `(llm_id, diffusion_id)`.
pub fn select_defaults(library: &ModelLibrary) -> (Option<String>, Option<String>) {
    // Prefer TinyLlama (tested model), then smallest auto-loadable LLM
    let default_llm = library
        .models
        .values()
        .filter(|e| e.category == ModelCategory::Llm && e.auto_loadable)
        .find(|e| e.id.to_lowercase().contains("tinyllama"))
        .or_else(|| {
            library.models.values()
                .filter(|e| e.category == ModelCategory::Llm && e.auto_loadable)
                .min_by_key(|e| e.size_bytes)
        })
        .map(|e| e.id.clone());

    // Find SDXL Turbo, or fall back to any diffusion model
    let default_diffusion = library
        .models
        .values()
        .filter(|e| e.category == ModelCategory::Diffusion)
        .find(|e| e.id.to_lowercase().contains("sdxl-turbo") || e.id.to_lowercase().contains("sdxl_turbo"))
        .or_else(|| {
            library.models.values()
                .filter(|e| e.category == ModelCategory::Diffusion)
                .min_by_key(|e| e.size_bytes)
        })
        .map(|e| e.id.clone());

    if let Some(ref id) = default_llm {
        tracing::info!("Default LLM: {}", id);
    }
    if let Some(ref id) = default_diffusion {
        tracing::info!("Default diffusion: {}", id);
    }

    (default_llm, default_diffusion)
}
