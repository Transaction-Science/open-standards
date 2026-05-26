//! Inference configuration.

use super::formats::GgufMetadata;

/// LLM architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// LLaMA family (Llama 1/2/3, CodeLlama, etc.)
    Llama,
    /// Mistral family (Mistral, Mixtral)
    Mistral,
    /// Phi family (Phi-2, Phi-3)
    Phi,
    /// Qwen family
    Qwen,
    /// Gemma family
    Gemma,
    /// DeepSeek family
    DeepSeek,
    /// Unknown architecture
    Unknown,
}

impl Architecture {
    /// Parse architecture from GGUF metadata string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" | "codellama" | "tinyllama" => Self::Llama,
            "mistral" | "mixtral" => Self::Mistral,
            "phi" | "phi2" | "phi3" => Self::Phi,
            "qwen" | "qwen2" => Self::Qwen,
            "gemma" | "gemma2" | "gemma3" => Self::Gemma,
            "deepseek" => Self::DeepSeek,
            _ => Self::Unknown,
        }
    }

    /// Get default RoPE theta for this architecture.
    pub fn default_rope_theta(&self) -> f32 {
        match self {
            Self::Llama => 10000.0,
            Self::Mistral => 10000.0,
            Self::Phi => 10000.0,
            Self::Qwen => 1000000.0, // Qwen2 uses high theta
            Self::Gemma => 10000.0,
            Self::DeepSeek => 10000.0,
            Self::Unknown => 10000.0,
        }
    }

    /// Get default RMS norm epsilon for this architecture.
    pub fn default_rms_norm_eps(&self) -> f32 {
        match self {
            Self::Llama => 1e-5,
            Self::Mistral => 1e-5,
            Self::Phi => 1e-5,
            Self::Qwen => 1e-6,
            Self::Gemma => 1e-6,
            Self::DeepSeek => 1e-6,
            Self::Unknown => 1e-5,
        }
    }

    /// Check if this architecture uses sliding window attention.
    pub fn uses_sliding_window(&self) -> bool {
        matches!(self, Self::Mistral)
    }
}

/// Model configuration for LLM architectures.
#[derive(Debug, Clone)]
pub struct LLMConfig {
    /// Architecture type
    pub architecture: Architecture,
    /// Model name
    pub name: String,
    /// Hidden size (embedding dimension)
    pub hidden_size: usize,
    /// Intermediate size (FFN hidden dimension)
    pub intermediate_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// KV dimension (num_kv_heads * head_dim)
    pub kv_dim: usize,
    /// RoPE theta (frequency base)
    pub rope_theta: f32,
    /// RoPE frequency scale
    pub rope_scale: f32,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// Sliding window size (if applicable)
    pub sliding_window: Option<usize>,
    /// Number of experts (0 = dense model, no MoE).
    pub num_experts: usize,
    /// Number of active experts per token (top-k routing).
    pub num_active_experts: usize,
    /// Whether to normalize top-k routing weights to sum to 1.0.
    pub norm_topk_prob: bool,
    /// Number of shared (always-on) experts (DeepSeek V2).
    pub num_shared_experts: usize,
    /// KV LoRA rank for MLA (DeepSeek V2). 0 = standard attention.
    pub kv_lora_rank: usize,
    /// Non-positional head dimension for MLA (DeepSeek V2).
    pub qk_nope_head_dim: usize,
    /// RoPE head dimension for MLA (DeepSeek V2).
    pub qk_rope_head_dim: usize,
    /// Value head dimension for MLA (DeepSeek V2).
    pub v_head_dim: usize,
    /// MoE expert gate weight key pattern ("block_sparse_moe" for Mixtral, "mlp" for DeepSeek).
    pub moe_gate_pattern: String,
}

impl LLMConfig {
    /// Create model config from GGUF metadata.
    pub fn from_gguf(metadata: &GgufMetadata) -> Self {
        let arch_str = metadata.architecture.as_deref().unwrap_or("unknown");
        let architecture = Architecture::from_str(arch_str);
        let name = metadata.name.clone().unwrap_or_else(|| arch_str.to_string());

        let hidden_size = metadata.embedding_length.unwrap_or(4096) as usize;
        let num_attention_heads = metadata.head_count.unwrap_or(32) as usize;
        let num_kv_heads = metadata.head_count_kv.unwrap_or(num_attention_heads as u64) as usize;
        let num_layers = metadata.block_count.unwrap_or(32) as usize;
        let vocab_size = metadata.vocab_size.unwrap_or(32000) as usize;
        let max_seq_len = metadata.context_length.unwrap_or(4096) as usize;

        let head_dim = hidden_size / num_attention_heads;
        let kv_dim = num_kv_heads * head_dim;

        let intermediate_size = metadata.feed_forward_length.unwrap_or_else(|| {
            match architecture {
                Architecture::Llama | Architecture::Mistral | Architecture::Qwen | Architecture::Gemma => {
                    (hidden_size * 8 / 3) as u64
                }
                _ => (hidden_size * 4) as u64,
            }
        }) as usize;

        let rope_theta = metadata.rope_freq_base.unwrap_or_else(|| architecture.default_rope_theta());
        let rope_scale = metadata.rope_freq_scale.unwrap_or(1.0);
        let rms_norm_eps = metadata.rms_norm_eps.unwrap_or_else(|| architecture.default_rms_norm_eps());
        let sliding_window = if architecture.uses_sliding_window() { Some(4096) } else { None };

        Self {
            architecture,
            name,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_kv_heads,
            num_layers,
            vocab_size,
            max_seq_len,
            head_dim,
            kv_dim,
            rope_theta,
            rope_scale,
            rms_norm_eps,
            sliding_window,
            num_experts: 0,
            num_active_experts: 0,
            norm_topk_prob: false,
            num_shared_experts: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            moe_gate_pattern: "block_sparse_moe".to_string(),
        }
    }

    /// Create model config with explicit values.
    pub fn new(
        architecture: Architecture,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        num_layers: usize,
        vocab_size: usize,
    ) -> Self {
        let head_dim = hidden_size / num_attention_heads;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            architecture,
            name: format!("{:?}", architecture),
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_kv_heads,
            num_layers,
            vocab_size,
            max_seq_len: 4096,
            head_dim,
            kv_dim,
            rope_theta: architecture.default_rope_theta(),
            rope_scale: 1.0,
            rms_norm_eps: architecture.default_rms_norm_eps(),
            sliding_window: if architecture.uses_sliding_window() { Some(4096) } else { None },
            num_experts: 0,
            num_active_experts: 0,
            norm_topk_prob: false,
            num_shared_experts: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            moe_gate_pattern: "block_sparse_moe".to_string(),
        }
    }

    /// Print configuration summary.
    pub fn print_summary(&self) {
        println!("Model: {} ({:?})", self.name, self.architecture);
        println!("  Hidden size: {}", self.hidden_size);
        println!("  Intermediate size: {}", self.intermediate_size);
        println!("  Heads: {} (KV: {})", self.num_attention_heads, self.num_kv_heads);
        println!("  Layers: {}", self.num_layers);
        println!("  Vocab size: {}", self.vocab_size);
        println!("  Max seq len: {}", self.max_seq_len);
        println!("  RoPE theta: {}", self.rope_theta);
        if self.rope_scale != 1.0 {
            println!("  RoPE scale: {}", self.rope_scale);
        }
        if let Some(sw) = self.sliding_window {
            println!("  Sliding window: {}", sw);
        }
    }

    /// TinyLlama 1.1B configuration.
    pub fn tiny_llama() -> Self {
        Self::new(Architecture::Llama, 2048, 5632, 32, 4, 22, 32000)
    }

    /// Llama 2 7B configuration.
    pub fn llama2_7b() -> Self {
        Self::new(Architecture::Llama, 4096, 11008, 32, 32, 32, 32000)
    }

    /// Llama 3 8B configuration.
    pub fn llama3_8b() -> Self {
        let mut config = Self::new(Architecture::Llama, 4096, 14336, 32, 8, 32, 128256);
        config.rope_theta = 500000.0;
        config.max_seq_len = 8192;
        config
    }

    /// Mistral 7B configuration.
    pub fn mistral_7b() -> Self {
        let mut config = Self::new(Architecture::Mistral, 4096, 14336, 32, 8, 32, 32000);
        config.max_seq_len = 32768;
        config
    }

    /// Phi-2 configuration.
    pub fn phi2() -> Self {
        Self::new(Architecture::Phi, 2560, 10240, 32, 32, 32, 51200)
    }

    /// Qwen2 0.5B configuration.
    pub fn qwen2_0_5b() -> Self {
        let mut config = Self::new(Architecture::Qwen, 896, 4864, 14, 2, 24, 151936);
        config.rope_theta = 1000000.0;
        config
    }
}

/// Main inference configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum memory usage (bytes)
    pub max_memory: Option<usize>,
    /// Enable lazy loading
    pub lazy_loading: bool,
    /// Enable weight prefetching
    pub prefetch_weights: bool,
    /// Target latency (milliseconds)
    pub target_latency_ms: u32,
    /// Number of inference threads
    pub num_threads: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_memory: None, // Use all available
            lazy_loading: true,
            prefetch_weights: true,
            target_latency_ms: 100, // Sub-100ms target
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
        }
    }
}

impl InferenceConfig {
    /// Low-memory configuration.
    pub fn low_memory() -> Self {
        Self {
            max_memory: Some(4 * 1024 * 1024 * 1024), // 4GB
            lazy_loading: true,
            prefetch_weights: false,
            target_latency_ms: 200,
            num_threads: 2,
        }
    }

    /// High-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            max_memory: None,
            lazy_loading: true,
            prefetch_weights: true,
            target_latency_ms: 50,
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(8),
        }
    }
}

/// Text generation parameters.
#[derive(Debug, Clone)]
pub struct TextParams {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature (0.0 = deterministic, 1.0+ = more random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Include log probabilities
    pub logprobs: bool,
    /// Stream tokens as generated
    pub stream: bool,
}

impl Default for TextParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.0,
            stop_sequences: Vec::new(),
            logprobs: false,
            stream: true,
        }
    }
}

impl TextParams {
    /// Deterministic generation (greedy decoding).
    pub fn deterministic() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            stop_sequences: Vec::new(),
            logprobs: false,
            stream: true,
        }
    }

    /// Creative generation.
    pub fn creative() -> Self {
        Self {
            max_tokens: 512,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 100,
            repetition_penalty: 1.1,
            stop_sequences: Vec::new(),
            logprobs: false,
            stream: true,
        }
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set repetition penalty.
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Add stop sequence.
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }
}

/// Image generation parameters.
#[derive(Debug, Clone)]
pub struct ImageParams {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Number of inference steps
    pub num_steps: u32,
    /// Guidance scale (CFG)
    pub guidance_scale: f32,
    /// Random seed
    pub seed: Option<u64>,
    /// Negative prompt
    pub negative_prompt: Option<String>,
    /// Use LCM scheduler (fast, 4 steps)
    pub use_lcm: bool,
    /// Use CFG++ guidance (applies guidance in denoised space)
    pub use_cfg_pp: bool,
    /// Scheduler name override (e.g. "heun", "dpmpp_2m_sde", "euler_ancestral_rf")
    pub scheduler: Option<String>,
}

impl Default for ImageParams {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            num_steps: 4, // LCM default
            guidance_scale: 1.0,
            seed: None,
            negative_prompt: None,
            use_lcm: true,
            use_cfg_pp: false,
            scheduler: None,
        }
    }
}

impl ImageParams {
    /// Standard quality (more steps).
    pub fn standard() -> Self {
        Self {
            width: 512,
            height: 512,
            num_steps: 20,
            guidance_scale: 7.5,
            seed: None,
            negative_prompt: None,
            use_lcm: false,
            use_cfg_pp: false,
            scheduler: None,
        }
    }

    /// High resolution.
    pub fn hd() -> Self {
        Self {
            width: 1024,
            height: 1024,
            num_steps: 4,
            guidance_scale: 1.0,
            seed: None,
            negative_prompt: None,
            use_lcm: true,
            use_cfg_pp: false,
            scheduler: None,
        }
    }

    /// Set dimensions.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set negative prompt.
    pub fn with_negative(mut self, negative: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative.into());
        self
    }
}

/// 3D generation parameters.
#[derive(Debug, Clone)]
pub struct ThreeDParams {
    /// Number of Gaussians to generate
    pub num_gaussians: usize,
    /// Output resolution for rendering
    pub render_resolution: u32,
    /// Generate mesh export
    pub export_mesh: bool,
    /// Mesh export format
    pub mesh_format: MeshFormat,
}

impl Default for ThreeDParams {
    fn default() -> Self {
        Self {
            num_gaussians: 500_000,
            render_resolution: 512,
            export_mesh: false,
            mesh_format: MeshFormat::GLB,
        }
    }
}

impl ThreeDParams {
    /// High quality (more Gaussians).
    pub fn high_quality() -> Self {
        Self {
            num_gaussians: 1_000_000,
            render_resolution: 1024,
            export_mesh: false,
            mesh_format: MeshFormat::GLB,
        }
    }

    /// With mesh export.
    pub fn with_mesh(mut self) -> Self {
        self.export_mesh = true;
        self
    }

    /// Set mesh format.
    pub fn with_format(mut self, format: MeshFormat) -> Self {
        self.mesh_format = format;
        self
    }
}

/// Mesh export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat {
    /// GLTF Binary
    GLB,
    /// Wavefront OBJ
    OBJ,
    /// FBX
    FBX,
    /// STL
    STL,
}

/// Video generation parameters.
#[derive(Debug, Clone)]
pub struct VideoParams {
    /// Number of frames
    pub num_frames: usize,
    /// Frame rate
    pub fps: f32,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Motion strength
    pub motion_strength: f32,
}

impl Default for VideoParams {
    fn default() -> Self {
        Self {
            num_frames: 24,
            fps: 24.0,
            width: 512,
            height: 512,
            motion_strength: 0.5,
        }
    }
}

/// Audio generation parameters.
#[derive(Debug, Clone)]
pub struct AudioParams {
    /// Duration in seconds
    pub duration_seconds: f32,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            duration_seconds: 5.0,
            sample_rate: 44100,
            channels: 2,
        }
    }
}
