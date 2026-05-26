//! Model loading and management.

use crate::core::{Error, Modality, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "metal")]
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::metal::{LazyLoader, LazyTensor};
#[cfg(feature = "metal")]
use objc2;
#[cfg(feature = "metal")]
use metal;

/// Type of model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Language model (GPT, LLaMA, etc.)
    LLM,
    /// Text-to-image diffusion
    TextToImage,
    /// Image-to-image
    ImageToImage,
    /// Image-to-3D
    ImageTo3D,
    /// Text-to-audio
    TextToAudio,
    /// Text-to-video
    TextToVideo,
}

impl ModelType {
    /// Get the primary modality for this model type.
    pub fn primary_modality(&self) -> Modality {
        match self {
            Self::LLM => Modality::Text,
            Self::TextToImage | Self::ImageToImage => Modality::Image,
            Self::ImageTo3D => Modality::ThreeD,
            Self::TextToAudio => Modality::Audio,
            Self::TextToVideo => Modality::Video,
        }
    }
}

/// Model information.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Number of parameters
    pub num_parameters: usize,
    /// Size on disk (bytes)
    pub size_bytes: usize,
    /// Architecture name
    pub architecture: String,
    /// Source path
    pub path: PathBuf,
    /// Whether model supports streaming
    pub supports_streaming: bool,
    /// Recommended batch size
    pub recommended_batch_size: usize,
}

/// A loaded model.
pub struct Model {
    /// Model info
    info: ModelInfo,
    /// Weights (lazily loaded on macOS)
    #[cfg(feature = "metal")]
    weights: HashMap<String, LazyTensor>,
    /// Weights (eagerly loaded on other platforms)
    #[cfg(not(feature = "metal"))]
    weights: HashMap<String, Tensor>,
    /// Reverse lookup: HF name → GGUF key (built lazily on first get_weight miss)
    #[cfg(feature = "metal")]
    reverse_map: std::sync::OnceLock<HashMap<String, String>>,
    /// Model config
    config: ModelConfig,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("info", &self.info)
            .field("num_weights", &self.weights.len())
            .finish()
    }
}

/// Model configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA; equals num_heads if not GQA)
    pub num_kv_heads: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// Vocabulary size (for LLMs)
    pub vocab_size: Option<usize>,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE theta base frequency
    pub rope_theta: f32,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// EOS token ID
    pub eos_token_id: u32,
    /// Whether lm_head weight is tied to embedding weight
    pub tie_word_embeddings: bool,
    /// Use flash attention
    pub use_flash_attention: bool,
    /// Quantization (None, INT8, INT4)
    pub quantization: Option<Quantization>,
    /// Number of experts (0 for dense, 8 for Mixtral MoE)
    pub num_experts: usize,
    /// Number of active experts per token (top-k, e.g. 2 for Mixtral)
    pub num_active_experts: usize,
    /// Number of shared experts (DeepSeek V2)
    pub num_shared_experts: usize,
    /// KV LoRA rank for MLA (DeepSeek V2). 0 = standard attention.
    pub kv_lora_rank: usize,
    /// MoE gate weight key pattern.
    pub moe_gate_pattern: String,
    /// SSM inner size (d_inner for Mamba-2)
    pub ssm_inner_size: usize,
    /// SSM state dimension (d_state)
    pub ssm_state_size: usize,
    /// SSM group count (n_groups for grouped B/C)
    pub ssm_group_count: usize,
    /// SSM number of heads (n_head, also dt_rank for Mamba-2)
    pub ssm_time_step_rank: usize,
    /// SSM conv kernel size (d_conv)
    pub ssm_conv_kernel: usize,
    /// Explicit head dimension override (from GGUF attention.key_length).
    /// When 0, falls back to hidden_size/num_heads.
    pub attn_head_dim: usize,
    /// RoPE rotary dimension (for partial RoPE models like Nemotron where n_rot < head_dim)
    pub rope_dim: usize,
}

impl ModelConfig {
    /// Head dimension. Uses explicit override if set, else hidden_size/num_heads.
    pub fn head_dim(&self) -> usize {
        if self.attn_head_dim > 0 { self.attn_head_dim }
        else if self.num_heads > 0 { self.hidden_size / self.num_heads }
        else { 0 }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 11008,
            vocab_size: Some(32000),
            max_seq_len: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            eos_token_id: 2,
            tie_word_embeddings: false,
            use_flash_attention: true,
            quantization: None,
            num_experts: 0,
            num_active_experts: 0,
            num_shared_experts: 0,
            kv_lora_rank: 0,
            moe_gate_pattern: "block_sparse_moe".to_string(),
            ssm_inner_size: 0,
            ssm_state_size: 0,
            ssm_group_count: 0,
            ssm_time_step_rank: 0,
            ssm_conv_kernel: 0,
            attn_head_dim: 0,
            rope_dim: 0,
        }
    }
}

/// Quantization level.
#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    /// 8-bit integers
    INT8,
    /// 4-bit integers
    INT4,
    /// FP8 (E4M3)
    FP8,
}

impl Model {
    /// Load a model from a path.
    ///
    /// Supports four input layouts:
    ///   1. **Single safetensors / GGUF file** — `path` is a regular file.
    ///   2. **Directory with sharded index** — `path/diffusion_pytorch_model.safetensors.index.json`
    ///      or `path/model.safetensors.index.json`.
    ///   3. **Directory with `model.safetensors`** — single bundled file
    ///      at the directory root.
    ///   4. **Diffusers multi-component directory** — `path/model_index.json`
    ///      present plus any subset of `unet/`, `vae/`, `text_encoder/`,
    ///      `text_encoder_2/` containing component safetensors. Each
    ///      component's weights are loaded and merged into one weight
    ///      table (HF diffusers naming is globally unique across components,
    ///      so no prefixing is needed).
    #[cfg(feature = "metal")]
    pub fn load(
        name: &str,
        path: &Path,
        loader: Arc<LazyLoader>,
    ) -> Result<Self> {
        // Diffusers multi-component directory takes precedence — when
        // `model_index.json` is present alongside component subdirs we have
        // to merge multiple safetensors files instead of loading one.
        if path.is_dir() && path.join("model_index.json").exists() {
            return Self::load_diffusers_dir(name, path, loader);
        }

        // Handle directory path — check for sharded safetensors first
        let file_path = if path.is_dir() {
            // Check for sharded index file first
            let sharded_index = path.join("diffusion_pytorch_model.safetensors.index.json");
            let sharded_index_2 = path.join("model.safetensors.index.json");
            if sharded_index.exists() {
                sharded_index
            } else if sharded_index_2.exists() {
                sharded_index_2
            } else {
                path.join("model.safetensors")
            }
        } else {
            path.to_path_buf()
        };

        // Detect model type from path/config
        let model_type = detect_model_type(&file_path)?;

        // Load weights lazily — detect sharded vs single file
        // Handles both ".index.json" and ".index.fp16.json" patterns
        let is_sharded_index = file_path.to_str()
            .map(|s| s.contains(".index.") && s.ends_with(".json"))
            .unwrap_or(false);

        let mut gguf_meta: Option<crate::inference::formats::GgufMetadata> = None;
        let weights = if is_sharded_index {
            if !file_path.exists() {
                return Err(Error::ModelLoad {
                    model: name.into(),
                    message: format!("Sharded index not found: {:?}", file_path).into(),
                    #[cfg(feature = "std")]
                    source: None,
                });
            }
            loader.load_safetensors_sharded_f16(&file_path)?
        } else if file_path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            if !file_path.exists() {
                 return Err(Error::ModelLoad {
                    model: name.into(),
                    message: format!("File not found: {:?}", file_path).into(),
                    #[cfg(feature = "std")]
                    source: None,
                });
            }
            // Auto-detect pre-converted F16 file for NVMe zero-copy mmap
            let fp16_path = {
                let stem = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if !stem.ends_with(".fp16") {
                    Some(file_path.with_file_name(format!("{}.fp16.safetensors", stem)))
                } else {
                    None
                }
            };
            if let Some(ref fp16) = fp16_path {
                if fp16.exists() {
                    tracing::info!("Using pre-converted F16 file for zero-copy mmap: {:?}", fp16);
                    loader.load_safetensors_f16(fp16)?
                } else {
                    loader.load_safetensors_f16(&file_path)?
                }
            } else {
                loader.load_safetensors_f16(&file_path)?
            }
        } else if file_path.extension().map(|e| e == "gguf").unwrap_or(false) {
            if !file_path.exists() {
                 return Err(Error::ModelLoad {
                    model: name.into(),
                    message: format!("File not found: {:?}", file_path).into(),
                    #[cfg(feature = "std")]
                    source: None,
                });
            }
            let (w, meta) = loader.load_gguf_f16(&file_path)?;
            gguf_meta = Some(meta);
            w
        } else {
            return Err(Error::ModelLoad {
                model: name.into(),
                message: "only safetensors and gguf formats are supported".into(),
                #[cfg(feature = "std")]
                source: None,
            });
        };

        // Parse config
        let mut config = parse_model_config(&file_path, &weights)?;

        // Extract SSM config from GGUF metadata
        if let Some(ref meta) = gguf_meta {
            eprintln!("[gguf] metadata keys ({}):", meta.raw.len());
            for (k, _v) in meta.raw.iter().filter(|(k, _)| k.starts_with("nemotron") || k.starts_with("llama") || k.starts_with("general.arch")) {
                eprintln!("  {}", k);
            }
            eprintln!("[gguf] arch={:?}", meta.architecture);
            use crate::inference::formats::gguf::MetadataValue;
            let get_u64 = |key: &str| -> Option<u64> {
                meta.raw.get(key).and_then(|v| match v {
                    MetadataValue::U32(x) => Some(*x as u64),
                    MetadataValue::U64(x) => Some(*x),
                    MetadataValue::I32(x) => Some(*x as u64),
                    MetadataValue::I64(x) => Some(*x as u64),
                    MetadataValue::U16(x) => Some(*x as u64),
                    MetadataValue::U8(x) => Some(*x as u64),
                    _ => None,
                })
            };
            let get_f32 = |key: &str| -> Option<f32> {
                meta.raw.get(key).and_then(|v| match v {
                    MetadataValue::F32(x) => Some(*x),
                    MetadataValue::F64(x) => Some(*x as f32),
                    _ => None,
                })
            };
            // Detect architecture prefix from GGUF metadata
            let arch = meta.architecture.as_deref().unwrap_or("llama");
            let p = |suffix: &str| -> String { format!("{}.{}", arch, suffix) };

            if let Some(v) = get_u64(&p("ssm.inner_size")) { config.ssm_inner_size = v as usize; }
            if let Some(v) = get_u64(&p("ssm.state_size")) { config.ssm_state_size = v as usize; }
            if let Some(v) = get_u64(&p("ssm.group_count")) { config.ssm_group_count = v as usize; }
            if let Some(v) = get_u64(&p("ssm.time_step_rank")) { config.ssm_time_step_rank = v as usize; }
            if let Some(v) = get_u64(&p("ssm.conv_kernel")) { config.ssm_conv_kernel = v as usize; }
            // Override basic config from GGUF (more reliable for GGUF-only models without config.json)
            if let Some(v) = get_u64(&p("embedding_length")).or(meta.embedding_length) { config.hidden_size = v as usize; }
            if let Some(v) = get_u64(&p("block_count")).or(meta.block_count) { config.num_layers = v as usize; }
            if let Some(v) = get_u64(&p("attention.head_count")).or(meta.head_count) { config.num_heads = v as usize; }
            // Explicit head_dim from GGUF (key_length). Nemotron uses 128, not hidden/n_heads.
            if let Some(v) = get_u64(&p("attention.key_length")) { config.attn_head_dim = v as usize; }
            // RoPE dim (partial RoPE for Nemotron: 84 of 128)
            if let Some(v) = get_u64(&p("rope.dimension_count")) { config.rope_dim = v as usize; }
            // head_count_kv may be a per-layer array (Nemotron hybrid SSM+Attn)
            // Use the max non-zero value as the global kv_heads config
            if let Some(raw_val) = meta.raw.get(&p("attention.head_count_kv")) {
                match raw_val {
                    MetadataValue::Array(arr) => {
                        let max_kv = arr.iter().filter_map(|v| match v {
                            MetadataValue::I32(x) => Some(*x as usize),
                            MetadataValue::U32(x) => Some(*x as usize),
                            _ => None,
                        }).max().unwrap_or(0);
                        if max_kv > 0 { config.num_kv_heads = max_kv; }
                    }
                    _ => {
                        if let Some(v) = meta.head_count_kv { if v > 0 { config.num_kv_heads = v as usize; } }
                    }
                }
            } else if let Some(v) = meta.head_count_kv { if v > 0 { config.num_kv_heads = v as usize; } }
            if let Some(v) = get_u64(&p("vocab_size")).or(meta.vocab_size) { config.vocab_size = Some(v as usize); }
            if let Some(v) = get_u64(&p("feed_forward_length")).or(meta.feed_forward_length) { config.intermediate_size = v as usize; }
            // Nemotron-MoE uses per-expert ffn size — use expert_feed_forward_length if available
            if let Some(v) = get_u64(&p("expert_feed_forward_length")) { config.intermediate_size = v as usize; }
            if let Some(v) = get_f32(&p("rope.freq_base")).or(meta.rope_freq_base) { config.rope_theta = v; }
            if let Some(v) = get_f32(&p("attention.layer_norm_rms_epsilon")).or(meta.rms_norm_eps) { config.rms_norm_eps = v; }
            if let Some(v) = get_u64(&p("expert_count")) { config.num_experts = v as usize; }
            if let Some(v) = get_u64(&p("expert_used_count")) { config.num_active_experts = v as usize; }
            if config.ssm_inner_size > 0 {
                eprintln!("[gguf] SSM: d_inner={} d_state={} n_groups={} n_heads={} d_conv={}",
                    config.ssm_inner_size, config.ssm_state_size, config.ssm_group_count,
                    config.ssm_time_step_rank, config.ssm_conv_kernel);
            }
        }

        // Calculate stats
        let num_parameters: usize = weights.values().map(|t: &LazyTensor| t.shape().numel()).sum();
        let size_bytes: usize = weights.values().map(|t: &LazyTensor| t.size()).sum();

        let info = ModelInfo {
            name: name.to_string(),
            model_type,
            num_parameters,
            size_bytes,
            architecture: detect_architecture(&weights),
            path: path.to_path_buf(),
            supports_streaming: matches!(model_type, ModelType::LLM),
            recommended_batch_size: 1,
        };

        Ok(Self {
            info,
            weights,
            config,
            #[cfg(feature = "metal")]
            reverse_map: std::sync::OnceLock::new(),
        })
    }

    /// Load a Hugging Face diffusers multi-component directory.
    ///
    /// Walks the standard component subdirs (`unet`, `vae`, `text_encoder`,
    /// `text_encoder_2`), finds the best safetensors file in each, and
    /// merges all weights into one `Model` instance. Prefers fp16 variants
    /// (`*.fp16.safetensors`) over fp32 to minimise mmap footprint.
    ///
    /// Each component's weight names are unique within HF's diffusers
    /// convention — UNet keys live under `down_blocks.*` / `up_blocks.*`
    /// / `mid_block.*`, VAE under `encoder.*` / `decoder.*` / `quant_conv.*`,
    /// text encoder under `text_model.*`. So merging without prefix is
    /// collision-free.
    #[cfg(feature = "metal")]
    fn load_diffusers_dir(
        name: &str,
        path: &Path,
        loader: Arc<LazyLoader>,
    ) -> Result<Self> {
        let component_dirs: &[&str] = &["unet", "vae", "text_encoder", "text_encoder_2"];
        let mut weights: HashMap<String, LazyTensor> = HashMap::new();
        let mut found_any = false;

        for component in component_dirs {
            let cdir = path.join(component);
            if !cdir.is_dir() {
                continue;
            }
            let file = match pick_diffusers_safetensors(&cdir) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    tracing::warn!(
                        "diffusers loader: no safetensors found in {} — skipping",
                        cdir.display()
                    );
                    continue;
                }
                Err(e) => {
                    // EACCES/EPERM here on macOS almost always means the binary
                    // lost Full Disk Access after a TCC re-evaluation (common
                    // after a deploy that replaces the binary on disk). Be loud
                    // — silent-skip masquerades as "model is fine but missing
                    // weights at inference time", which is much harder to
                    // diagnose downstream.
                    let kind = e.kind();
                    return Err(crate::core::Error::internal(format!(
                        "diffusers loader: cannot read component dir {} ({:?}: {}). \
                         If kind is PermissionDenied on macOS, the binary likely \
                         lost Full Disk Access after a replace — re-grant FDA in \
                         System Settings → Privacy & Security.",
                        cdir.display(), kind, e
                    )));
                }
            };
            tracing::info!("diffusers loader: loading {} from {:?}", component, file);
            let component_weights = loader.load_safetensors_f16(&file)?;
            for (k, v) in component_weights {
                if weights.contains_key(&k) {
                    tracing::warn!(
                        "diffusers loader: weight name collision on '{}' (component '{}') — keeping first",
                        k,
                        component
                    );
                    continue;
                }
                weights.insert(k, v);
            }
            found_any = true;
        }

        if !found_any {
            return Err(Error::ModelLoad {
                model: name.into(),
                message: format!(
                    "diffusers layout detected at {:?} but no component subdir contained loadable safetensors",
                    path
                )
                .into(),
                #[cfg(feature = "std")]
                source: None,
            });
        }

        let model_type = detect_model_type(path).unwrap_or(ModelType::TextToImage);
        let config = parse_model_config(path, &weights)?;

        let num_parameters: usize = weights.values().map(|t: &LazyTensor| t.shape().numel()).sum();
        let size_bytes: usize = weights.values().map(|t: &LazyTensor| t.size()).sum();

        let info = ModelInfo {
            name: name.to_string(),
            model_type,
            num_parameters,
            size_bytes,
            architecture: detect_architecture(&weights),
            path: path.to_path_buf(),
            supports_streaming: matches!(model_type, ModelType::LLM),
            recommended_batch_size: 1,
        };

        Ok(Self {
            info,
            weights,
            config,
            #[cfg(feature = "metal")]
            reverse_map: std::sync::OnceLock::new(),
        })
    }

    /// Load a model (non-macOS fallback).
    #[cfg(not(feature = "metal"))]
    pub fn load(
        _name: &str,
        _path: &Path,
    ) -> Result<Self> {
        Err(Error::unsupported("model loading only implemented for macOS"))
    }

    /// Get model info.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Get model config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get a weight tensor by name.
    /// Tries the original name, then common SDXL prefixes, then GGUF/HF translation.
    #[cfg(feature = "metal")]
    pub fn get_weight(&self, name: &str) -> Option<&LazyTensor> {
        let result = self.weights.get(name)
            .or_else(|| {
                let translated = translate_weight_name(name);
                self.weights.get(&translated)
            })
            .or_else(|| {
                // Use cached reverse map (HF name → GGUF key), built once
                let rmap = self.reverse_map.get_or_init(|| {
                    let mut map = HashMap::new();
                    for key in self.weights.keys() {
                        let hf_name = translate_weight_name(key);
                        if hf_name != *key {
                            map.insert(hf_name, key.clone());
                        }
                    }
                    map
                });
                rmap.get(name).and_then(|gguf_key| self.weights.get(gguf_key))
            });
        if result.is_none() && !name.contains("conditioner") {
            tracing::debug!("get_weight: '{}' not found. Tried: '{}', 'model.diffusion_model.{}', 'first_stage_model.{}'",
                name, name, name, name);
            let first_word = name.split('.').next().unwrap_or(name);
            let matches: Vec<_> = self.weights.keys().filter(|k| k.contains(first_word)).take(5).collect();
            if !matches.is_empty() {
                tracing::debug!("  similar keys: {:?}", matches);
            }
        }
        result
    }

    /// Insert a weight manually (for testing/dummy keys).
    #[cfg(feature = "metal")]
    pub fn insert_weight(&mut self, name: String, tensor: Tensor) {
        if let Some(ptr) = tensor.device_ptr() {
            // Create retained Metal buffer from raw pointer
            let buffer = unsafe {
                use metal::foreign_types::ForeignType;
                use objc2::runtime::AnyObject;
                let ptr = ptr.raw() as *mut AnyObject;
                let retained = objc2::rc::Retained::retain(ptr).expect("Failed to retain object");
                let raw = objc2::rc::Retained::into_raw(retained);
                metal::Buffer::from_ptr(raw as *mut _)
            };
            
            let lazy = LazyTensor::new_resident(
                buffer, 
                tensor.shape().clone(), 
                tensor.dtype(), 
                name.clone()
            );
            
            self.weights.insert(name, lazy);
        }
    }
    
    /// Create a dummy model for testing/server stubbing.
    pub fn dummy(model_type: ModelType) -> Self {
        let info = ModelInfo {
            name: "dummy-model".to_string(),
            model_type,
            num_parameters: 0,
            size_bytes: 0,
            architecture: "dummy".to_string(),
            path: PathBuf::from("dummy.safetensors"),
            supports_streaming: false,
            recommended_batch_size: 1,
        };

        Self {
            info,
            #[cfg(feature = "metal")]
            weights: HashMap::new(),
            #[cfg(not(feature = "metal"))]
            weights: HashMap::new(),
            #[cfg(feature = "metal")]
            reverse_map: std::sync::OnceLock::new(),
            config: ModelConfig::default(),
        }
    }

    /// Get a weight tensor by name (non-Metal fallback).
    #[cfg(not(feature = "metal"))]
    pub fn get_weight(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    /// List all weight names.
    pub fn weight_names(&self) -> Vec<&str> {
        self.weights.keys().map(|s| s.as_str()).collect()
    }

    /// Prefetch all weights into memory.
    #[cfg(feature = "metal")]
    pub fn prefetch(&self) {
        for tensor in self.weights.values() {
            tensor.prefetch();
        }
    }

    /// Prefetch weights (no-op on non-Metal backends).
    #[cfg(not(feature = "metal"))]
    pub fn prefetch(&self) {
        // No-op on non-metal (weights already loaded)
    }

    /// Prefetch weights matching a prefix (async, non-blocking).
    ///
    /// Uses `madvise(MADV_WILLNEED)` for async NVMe read-ahead.
    /// Call this before processing the current block to overlap I/O
    /// with GPU compute on the current block's weights.
    #[cfg(feature = "metal")]
    pub fn prefetch_prefix(&self, prefix: &str) {
        for (name, tensor) in &self.weights {
            if name.starts_with(prefix) {
                tensor.advise_willneed();
            }
        }
    }

    /// Prefetch weights matching a prefix (no-op on non-metal).
    #[cfg(not(feature = "metal"))]
    pub fn prefetch_prefix(&self, _prefix: &str) {
        // No-op on non-metal
    }

    /// Mark weights as not needed (evict hint).
    #[cfg(feature = "metal")]
    pub fn evict_prefix(&self, prefix: &str) {
        for (name, tensor) in &self.weights {
            if name.starts_with(prefix) {
                tensor.advise_dontneed();
            }
        }
    }

    /// Mark weights as not needed (no-op on non-metal).
    #[cfg(not(feature = "metal"))]
    pub fn evict_prefix(&self, _prefix: &str) {
        // No-op on non-metal
    }

    /// Get memory resident ratio (how much is actually in RAM).
    ///
    /// Uses `mincore()` to query actual page residency. For mmap-backed
    /// models, this shows what fraction of weight data is in physical RAM
    /// vs paged out to NVMe/SSD.
    #[cfg(feature = "metal")]
    pub fn memory_resident_ratio(&self) -> f32 {
        if self.weights.is_empty() {
            return 0.0;
        }
        let mut total_size: usize = 0;
        let mut resident_size: f64 = 0.0;
        for tensor in self.weights.values() {
            let size = tensor.size();
            total_size += size;
            resident_size += tensor.residency() * size as f64;
        }
        if total_size == 0 { 0.0 } else { (resident_size / total_size as f64) as f32 }
    }
}

/// Translate weight names between GGUF and HuggingFace conventions.
pub fn translate_weight_name(name: &str) -> String {
    // GGUF -> HuggingFace mapping (top-level only, before layer handling)
    // Use exact prefix/suffix matching to avoid corrupting layer-level names
    let translated = if name == "output.weight" {
        "lm_head.weight".to_string()
    } else {
        name.replace("token_embd.weight", "model.embed_tokens.weight")
            .replace("output_norm.weight", "model.norm.weight")
    };

    // Handle layer-level mappings: blk.{N}.xxx -> model.layers.{N}.xxx
    if translated.starts_with("blk.") {
        // Parse layer number
        let parts: Vec<&str> = translated.splitn(3, '.').collect();
        if parts.len() >= 3 {
            let layer_num = parts[1];
            let rest = parts[2];

            // SSM weights: pass through as-is
            let ssm_keys = ["ssm_in.weight", "ssm_out.weight", "ssm_conv1d.weight",
                "ssm_conv1d.bias", "ssm_dt.bias", "ssm_a", "ssm_d", "ssm_norm.weight",
                "exp_probs_b.bias"];
            if ssm_keys.iter().any(|&k| rest == k) {
                return format!("model.layers.{}.{}", layer_num, rest);
            }

            // Nemotron MoE expert weights
            let hf_rest = rest
                .replace("ffn_gate_inp.weight", "mlp.gate.weight")
                .replace("ffn_up_exps.weight", "mlp.experts_up.weight")
                .replace("ffn_down_exps.weight", "mlp.experts_down.weight")
                .replace("ffn_up_shexp.weight", "mlp.shared_experts.up_proj.weight")
                .replace("ffn_down_shexp.weight", "mlp.shared_experts.down_proj.weight")
                // Standard attention/FFN
                .replace("attn_q.weight", "self_attn.q_proj.weight")
                .replace("attn_k.weight", "self_attn.k_proj.weight")
                .replace("attn_v.weight", "self_attn.v_proj.weight")
                .replace("attn_output.weight", "self_attn.o_proj.weight")
                .replace("ffn_gate.weight", "mlp.gate_proj.weight")
                .replace("ffn_up.weight", "mlp.up_proj.weight")
                .replace("ffn_down.weight", "mlp.down_proj.weight")
                .replace("attn_norm.weight", "input_layernorm.weight")
                .replace("ffn_norm.weight", "post_attention_layernorm.weight");

            return format!("model.layers.{}.{}", layer_num, hf_rest);
        }
    }

    translated
}

/// Pick the best safetensors file inside a diffusers component subdir.
///
/// Preference order (most desirable first):
///   1. `<name>.fp16.safetensors` (fp16, half-size, our load path's
///      preferred precision)
///   2. `<name>.safetensors` (fp32 full precision)
///   3. anything ending in `.safetensors` (fallback)
///
/// `.bin` PyTorch pickle variants are intentionally ignored — efficient-genai
/// only supports safetensors / GGUF.
#[cfg(feature = "metal")]
/// Pick the best safetensors file from a diffusers component dir.
///
/// Returns:
/// - `Ok(Some(path))` when a candidate is found.
/// - `Ok(None)` when the directory was readable but contained no `.safetensors`.
/// - `Err(io)` when the directory itself could not be read (EACCES/EPERM from
///   macOS TCC stripping FDA off a freshly-replaced binary, ENOENT, etc.).
///
/// History: the previous `Option`-returning form used `read_dir(dir).ok()?`,
/// which collapsed EPERM into the same "no candidate" branch as an empty
/// directory. After every prod deploy that triggered a TCC re-evaluation, the
/// loader silently logged "no safetensors found … — skipping" and the server
/// came up with the UNet missing instead of erroring out.
fn pick_diffusers_safetensors(dir: &Path) -> std::io::Result<Option<std::path::PathBuf>> {
    let entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    // 1. fp16 safetensors
    for e in &entries {
        let p = e.path();
        if let Some(s) = p.file_name().and_then(|n| n.to_str()) {
            if s.ends_with(".fp16.safetensors") {
                return Ok(Some(p));
            }
        }
    }
    // 2. plain safetensors (fp32)
    for e in &entries {
        let p = e.path();
        if let Some(s) = p.file_name().and_then(|n| n.to_str()) {
            if s.ends_with(".safetensors") && !s.ends_with(".non_ema.safetensors") {
                return Ok(Some(p));
            }
        }
    }
    // 3. anything safetensors
    for e in &entries {
        let p = e.path();
        if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

fn detect_model_type(path: &Path) -> Result<ModelType> {
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    if name.contains("llama") || name.contains("mistral") || name.contains("phi") {
        Ok(ModelType::LLM)
    } else if name.contains("sdxl") || name.contains("stable-diffusion") || name.contains("flux") {
        Ok(ModelType::TextToImage)
    } else if name.contains("tripo") || name.contains("3d") {
        Ok(ModelType::ImageTo3D)
    } else {
        // Default to LLM
        Ok(ModelType::LLM)
    }
}

#[cfg(feature = "metal")]
fn parse_model_config(path: &Path, weights: &HashMap<String, LazyTensor>) -> Result<ModelConfig> {
    let mut config = ModelConfig::default();

    // Try to load config.json from the same directory
    if let Some(parent) = path.parent() {
        let config_path = parent.join("config.json");
        if config_path.exists() {
            if let Ok(json_str) = std::fs::read_to_string(&config_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    if let Some(v) = json.get("hidden_size").and_then(|v| v.as_u64()) {
                        config.hidden_size = v as usize;
                    }
                    if let Some(v) = json.get("num_hidden_layers").and_then(|v| v.as_u64()) {
                        config.num_layers = v as usize;
                    }
                    if let Some(v) = json.get("num_attention_heads").and_then(|v| v.as_u64()) {
                        config.num_heads = v as usize;
                    }
                    if let Some(v) = json.get("num_key_value_heads").and_then(|v| v.as_u64()) {
                        config.num_kv_heads = v as usize;
                    } else {
                        config.num_kv_heads = config.num_heads;
                    }
                    if let Some(v) = json.get("intermediate_size").and_then(|v| v.as_u64()) {
                        config.intermediate_size = v as usize;
                    }
                    if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) {
                        config.vocab_size = Some(v as usize);
                    }
                    if let Some(v) = json.get("max_position_embeddings").and_then(|v| v.as_u64()) {
                        config.max_seq_len = v as usize;
                    }
                    if let Some(v) = json.get("rope_theta").and_then(|v| v.as_f64()) {
                        config.rope_theta = v as f32;
                    }
                    if let Some(v) = json.get("rms_norm_eps").and_then(|v| v.as_f64()) {
                        config.rms_norm_eps = v as f32;
                    }
                    if let Some(v) = json.get("eos_token_id").and_then(|v| v.as_u64()) {
                        config.eos_token_id = v as u32;
                    }
                    if let Some(v) = json.get("tie_word_embeddings").and_then(|v| v.as_bool()) {
                        config.tie_word_embeddings = v;
                    }
                    // MoE (Mixtral, etc.)
                    if let Some(v) = json.get("num_local_experts").and_then(|v| v.as_u64()) {
                        config.num_experts = v as usize;
                    }
                    if let Some(v) = json.get("num_experts_per_tok").and_then(|v| v.as_u64()) {
                        config.num_active_experts = v as usize;
                    }
                    // DeepSeek V2 MoE fields
                    if let Some(v) = json.get("n_routed_experts").and_then(|v| v.as_u64()) {
                        config.num_experts = v as usize;
                        config.moe_gate_pattern = "mlp".to_string();
                    }
                    if json.get("n_shared_experts").is_some() {
                        if let Some(v) = json.get("n_shared_experts").and_then(|v| v.as_u64()) {
                            config.num_shared_experts = v as usize;
                        }
                    }
                    // DeepSeek V2 MLA fields
                    if let Some(v) = json.get("kv_lora_rank").and_then(|v| v.as_u64()) {
                        config.kv_lora_rank = v as usize;
                    }
                    tracing::info!("Loaded config from {:?}: hidden={}, layers={}, heads={}, kv_heads={}, vocab={:?}, experts={}, shared_experts={}, kv_lora_rank={}",
                        config_path, config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads, config.vocab_size,
                        config.num_experts, config.num_shared_experts, config.kv_lora_rank);
                    return Ok(config);
                }
            }
        }
    }

    // Fallback: infer config from weight shapes
    // Look for embedding weight to get hidden size
    for (name, tensor) in weights {
        if (name.contains("embed") || name.contains("embd")) && !name.contains("pos") && !name.contains("output") {
            if let Some(dim) = tensor.shape().dim(1) {
                config.hidden_size = dim;
            }
            if let Some(vocab) = tensor.shape().dim(0) {
                config.vocab_size = Some(vocab);
            }
            break;
        }
    }

    // Count layers
    let mut max_layer = 0;
    for name in weights.keys() {
        if let Some(layer_str) = name.split('.').find(|s| s.parse::<usize>().is_ok()) {
            if let Ok(layer) = layer_str.parse::<usize>() {
                max_layer = max_layer.max(layer);
            }
        }
    }
    if max_layer > 0 {
        config.num_layers = max_layer + 1;
    }

    // Infer num_heads and head_dim from attention weight shapes
    for (name, tensor) in weights {
        if name.contains("q_proj") || name.contains("query") || name.contains("attn_q") {
            if let (Some(out_dim), Some(in_dim)) = (tensor.shape().dim(0), tensor.shape().dim(1)) {
                // head_dim is typically 64 or 128; try common values
                for head_dim in [128, 96, 80, 64] {
                    if out_dim % head_dim == 0 {
                        config.num_heads = out_dim / head_dim;
                        break;
                    }
                }
            }
            break;
        }
    }
    config.num_kv_heads = config.num_heads; // assume MHA if no config.json

    // If num_heads wasn't determined from attention weights, infer from hidden_size
    // and common architectures
    if config.num_heads == 0 && config.hidden_size > 0 {
        // Common head_dim values: 64 (small), 80 (Phi), 96, 128 (LLaMA/Qwen)
        // Try to infer from hidden_size and common architectures
        for head_dim in [128, 96, 80, 64] {
            if config.hidden_size % head_dim == 0 {
                config.num_heads = config.hidden_size / head_dim;
                break;
            }
        }
    }

    Ok(config)
}

#[cfg(feature = "metal")]
fn detect_architecture(weights: &HashMap<String, LazyTensor>) -> String {
    // Detect architecture from weight naming conventions
    let names: Vec<&str> = weights.keys().map(|s| s.as_str()).collect();

    if names.iter().any(|n| n.contains("token_embd")) {
         // likely GGUF
         "llama-gguf".to_string()
    } else if names.iter().any(|n| n.contains("model.layers")) {
        if names.iter().any(|n| n.contains("mlp.gate_proj")) {
            "llama".to_string()
        } else if names.iter().any(|n| n.contains("mlp.fc1")) {
            "mistral".to_string()
        } else {
            "transformer".to_string()
        }
    } else if names.iter().any(|n| n.contains("unet")) {
        "stable-diffusion".to_string()
    } else if names.iter().any(|n| n.contains("encoder") && n.contains("decoder")) {
        "encoder-decoder".to_string()
    } else {
        "unknown".to_string()
    }
}

#[cfg(not(feature = "metal"))]
fn parse_model_config(_path: &Path, _weights: &HashMap<String, Tensor>) -> Result<ModelConfig> {
    Ok(ModelConfig::default())
}
