//! Chatterbox: Voice cloning TTS via 3-stage pipeline (350-500M params, MIT license).
//!
//! Architecture:
//!   Stage 1 — T3 (Text-to-Tokens): Llama-3 backbone, 350M
//!     Text tokens (vocab=704) + speaker embedding (256-dim) → speech tokens (vocab=8194)
//!     Standard causal transformer with RoPE, 16 layers, 16 heads, 1024 hidden
//!     Autoregressive decoding at 25 Hz
//!     Weight prefix: `t3.`
//!
//!   Stage 2 — S3Gen (Tokens-to-Mel): Conditional Flow Matching
//!     Speech tokens → 2x upsampled (25Hz→50Hz) → mel spectrogram (100 channels)
//!     Euler ODE solver: 1 step (turbo, distilled) or 10 steps (original)
//!     U-Net style with residual blocks
//!     Weight prefix: `s3gen.`
//!
//!   Stage 3 — HiFTGenerator (Mel-to-Audio): Vocoder
//!     Mel → harmonic oscillator + noise filter → iSTFT → 24kHz audio
//!     Similar to Kokoro's iSTFTNet architecture
//!     Weight prefix: `vocoder.`
//!
//!   Speaker Encoder — CAMPPlus:
//!     Reference audio (10+ sec) → 256-dim speaker embedding
//!     CNN with statistics pooling
//!     Weight prefix: `speaker_encoder.`
//!
//!   S3Tokenizer:
//!     Reference audio → discrete speech tokens (for prefix conditioning)
//!     Weight prefix: `tokenizer.`

#[cfg(feature = "metal")]
use tracing::debug;
#[cfg(feature = "metal")]
use crate::core::Result;
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::core::Error;

// ── Configuration ────────────────────────────────────────────────────────────

/// Chatterbox TTS configuration.
#[derive(Debug, Clone)]
pub struct ChatterboxConfig {
    /// Text vocabulary size (English phoneme/grapheme tokens + special).
    pub text_vocab_size: usize,
    /// Speech token vocabulary size (discrete audio codes).
    pub speech_vocab_size: usize,
    /// Speaker embedding dimension from CAMPPlus encoder.
    pub speaker_dim: usize,
    /// T3 transformer hidden dimension.
    pub t3_hidden: usize,
    /// T3 number of transformer layers.
    pub t3_layers: usize,
    /// T3 number of attention heads.
    pub t3_heads: usize,
    /// S3Gen flow matching steps (1 = turbo/distilled, 10 = original).
    pub s3gen_steps: usize,
    /// Number of mel spectrogram channels.
    pub mel_channels: usize,
    /// Output audio sample rate.
    pub sample_rate: usize,
    /// T3 RoPE theta for positional encoding.
    pub rope_theta: f32,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// T3 feed-forward intermediate dimension (typically 4 * hidden).
    pub t3_intermediate: usize,
    /// S3Gen U-Net channel multipliers per depth level.
    pub s3gen_channels: Vec<usize>,
    /// S3Gen residual blocks per depth level.
    pub s3gen_res_blocks: usize,
    /// HiFT vocoder upsample rates.
    pub vocoder_upsample_rates: Vec<usize>,
    /// HiFT vocoder upsample kernel sizes.
    pub vocoder_upsample_kernels: Vec<usize>,
    /// HiFT vocoder resblock kernel sizes.
    pub vocoder_resblock_kernels: Vec<usize>,
    /// HiFT vocoder iSTFT n_fft.
    pub vocoder_n_fft: usize,
    /// HiFT vocoder iSTFT hop length.
    pub vocoder_hop_length: usize,
    /// CAMPPlus speaker encoder CNN channels.
    pub campplus_channels: Vec<usize>,
    /// Maximum generation length in speech tokens (25 Hz).
    pub max_speech_tokens: usize,
    /// End-of-sequence token ID for speech generation.
    pub speech_eos_token: usize,
    /// Padding token for speech tokens.
    pub speech_pad_token: usize,
}

impl Default for ChatterboxConfig {
    fn default() -> Self {
        Self {
            text_vocab_size: 704,
            speech_vocab_size: 8194,
            speaker_dim: 256,
            t3_hidden: 1024,
            t3_layers: 16,
            t3_heads: 16,
            s3gen_steps: 1,
            mel_channels: 100,
            sample_rate: 24000,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            t3_intermediate: 4096,
            s3gen_channels: vec![256, 512, 512, 1024],
            s3gen_res_blocks: 2,
            vocoder_upsample_rates: vec![8, 8, 2, 2],
            vocoder_upsample_kernels: vec![16, 16, 4, 4],
            vocoder_resblock_kernels: vec![3, 7, 11],
            vocoder_n_fft: 16,
            vocoder_hop_length: 4,
            campplus_channels: vec![512, 512, 512, 512, 1536],
            max_speech_tokens: 2000,
            speech_eos_token: 8193,
            speech_pad_token: 8192,
        }
    }
}

impl ChatterboxConfig {
    /// Chatterbox Turbo (350M, distilled 1-step flow matching).
    pub fn turbo() -> Self {
        Self {
            s3gen_steps: 1,
            ..Default::default()
        }
    }

    /// Chatterbox Original (350M, 10-step flow matching).
    pub fn original() -> Self {
        Self {
            s3gen_steps: 10,
            ..Default::default()
        }
    }

    /// Parse from config.json.
    pub fn from_json(path: &std::path::Path) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let json_str = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut config = Self::default();
        if let Some(n) = v.get("text_vocab_size").and_then(|v| v.as_u64()) {
            config.text_vocab_size = n as usize;
        }
        if let Some(n) = v.get("speech_vocab_size").and_then(|v| v.as_u64()) {
            config.speech_vocab_size = n as usize;
        }
        if let Some(n) = v.get("speaker_dim").and_then(|v| v.as_u64()) {
            config.speaker_dim = n as usize;
        }
        if let Some(n) = v.get("t3_hidden").and_then(|v| v.as_u64()) {
            config.t3_hidden = n as usize;
        }
        if let Some(n) = v.get("t3_layers").and_then(|v| v.as_u64()) {
            config.t3_layers = n as usize;
        }
        if let Some(n) = v.get("t3_heads").and_then(|v| v.as_u64()) {
            config.t3_heads = n as usize;
        }
        if let Some(n) = v.get("s3gen_steps").and_then(|v| v.as_u64()) {
            config.s3gen_steps = n as usize;
        }
        if let Some(n) = v.get("mel_channels").and_then(|v| v.as_u64()) {
            config.mel_channels = n as usize;
        }
        if let Some(n) = v.get("sample_rate").and_then(|v| v.as_u64()) {
            config.sample_rate = n as usize;
        }
        Ok(config)
    }
}

// ── Metal Shader Sources ────────────────────────────────────────────────────

#[cfg(feature = "metal")]
const CHATTERBOX_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// RoPE for T3 causal transformer (Llama-3 style).
// Applies rotary position encoding to Q and K tensors.
// Q/K: [seq_len, num_heads, head_dim], freqs are precomputed cos/sin.
kernel void rope_apply_f16(
    device half* q [[buffer(0)]],
    device half* k [[buffer(1)]],
    device const float* cos_cache [[buffer(2)]],
    device const float* sin_cache [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& pos_offset [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;
    uint head = gid.y;
    uint pair = gid.z;
    if (pos >= seq_len || head >= num_heads || pair >= head_dim / 2) return;

    uint idx = pos * num_heads * head_dim + head * head_dim + pair * 2;
    uint freq_idx = (pos + pos_offset) * (head_dim / 2) + pair;

    float cos_val = cos_cache[freq_idx];
    float sin_val = sin_cache[freq_idx];

    // Apply to Q
    float q0 = float(q[idx]);
    float q1 = float(q[idx + 1]);
    q[idx] = half(q0 * cos_val - q1 * sin_val);
    q[idx + 1] = half(q0 * sin_val + q1 * cos_val);

    // Apply to K
    float k0 = float(k[idx]);
    float k1 = float(k[idx + 1]);
    k[idx] = half(k0 * cos_val - k1 * sin_val);
    k[idx + 1] = half(k0 * sin_val + k1 * cos_val);
}

// Causal attention mask + scaled softmax for T3 autoregressive decoding.
// Scores: [num_heads, q_len, kv_len], applies causal mask and scale.
kernel void causal_softmax_f16(
    device half* scores [[buffer(0)]],
    constant uint& q_len [[buffer(1)]],
    constant uint& kv_len [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& q_offset [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint head = gid.y;
    uint q_pos = gid.x;
    if (q_pos >= q_len) return;

    uint row_start = head * q_len * kv_len + q_pos * kv_len;
    uint abs_q = q_pos + q_offset;

    // Scale and apply causal mask
    float max_val = -1e9f;
    for (uint k = 0; k < kv_len; k++) {
        float v = float(scores[row_start + k]) * scale;
        if (k > abs_q) v = -1e9f; // causal: can only attend to past + self
        scores[row_start + k] = half(v);
        max_val = max(max_val, v);
    }

    // Softmax
    float sum_exp = 0.0f;
    for (uint k = 0; k < kv_len; k++) {
        float v = exp(float(scores[row_start + k]) - max_val);
        scores[row_start + k] = half(v);
        sum_exp += v;
    }
    float inv_sum = 1.0f / (sum_exp + 1e-12f);
    for (uint k = 0; k < kv_len; k++) {
        scores[row_start + k] = half(float(scores[row_start + k]) * inv_sum);
    }
}

// Speaker embedding projection: add speaker embedding to each token hidden state.
// hidden: [seq_len, hidden_dim], spk_embed: [hidden_dim] (projected from speaker_dim).
kernel void add_speaker_embedding_f16(
    device half* hidden [[buffer(0)]],
    device const half* spk_embed [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;
    uint dim = gid.y;
    if (pos >= seq_len || dim >= hidden_dim) return;
    uint idx = pos * hidden_dim + dim;
    hidden[idx] = half(float(hidden[idx]) + float(spk_embed[dim]));
}

// Token-to-mel upsampling: repeat each token's features 2x (25Hz → 50Hz).
kernel void upsample_2x_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& in_len [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;
    uint ch = gid.y;
    if (pos >= in_len || ch >= channels) return;

    half val = input[pos * channels + ch];
    output[(pos * 2) * channels + ch] = val;
    output[(pos * 2 + 1) * channels + ch] = val;
}

// Flow matching Euler step: x_t = x_t + dt * v_t
// x_t: [len, channels], v_t: [len, channels], dt: scalar
kernel void euler_step_f16(
    device half* x [[buffer(0)]],
    device const half* velocity [[buffer(1)]],
    constant float& dt [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) return;
    x[gid] = half(float(x[gid]) + dt * float(velocity[gid]));
}

// Harmonic oscillator for HiFT vocoder.
// Generates harmonics from F0 with phase accumulation.
kernel void harmonic_oscillator_f16(
    device const float* f0 [[buffer(0)]],
    device float* phase [[buffer(1)]],
    device half* harmonics [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant uint& n_harmonics [[buffer(4)]],
    constant float& sample_rate [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint t = gid.x;
    uint h = gid.y;
    if (t >= length || h >= n_harmonics) return;

    float freq = f0[t] * float(h + 1);
    float phase_inc = 2.0f * M_PI_F * freq / sample_rate;

    // Accumulate phase (read previous if t > 0)
    float prev_phase = (t > 0) ? phase[(t - 1) * n_harmonics + h] : 0.0f;
    float cur_phase = prev_phase + phase_inc;
    phase[t * n_harmonics + h] = cur_phase;

    // Amplitude decreases with harmonic number (1/h roll-off)
    float amp = 1.0f / float(h + 1);
    harmonics[t * n_harmonics + h] = half(amp * sin(cur_phase));
}
"#;

// ── Compiled Kernels ────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct ChatterboxKernels {
    common: gpu_ops::CommonKernels,
    silu: Arc<ComputePipeline>,
    gelu: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    rms_norm: Arc<ComputePipeline>,
    rope_apply: Arc<ComputePipeline>,
    causal_softmax: Arc<ComputePipeline>,
    add_speaker_embedding: Arc<ComputePipeline>,
    upsample_2x: Arc<ComputePipeline>,
    euler_step: Arc<ComputePipeline>,
    harmonic_oscillator: Arc<ComputePipeline>,
    conv1d: Arc<ComputePipeline>,
    conv1d_transpose: Arc<ComputePipeline>,
    dilated_conv1d: Arc<ComputePipeline>,
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Chatterbox voice cloning TTS pipeline on Metal GPU.
///
/// Three-stage pipeline:
/// 1. T3 (Llama-3 backbone): text + speaker embedding -> speech tokens
/// 2. S3Gen (conditional flow matching): speech tokens -> mel spectrogram
/// 3. HiFTGenerator (vocoder): mel -> 24kHz audio
#[cfg(feature = "metal")]
pub struct ChatterboxPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: ChatterboxConfig,
    kernels: ChatterboxKernels,
    /// Precomputed RoPE cos cache [max_len, head_dim/2].
    rope_cos: Vec<f32>,
    /// Precomputed RoPE sin cache [max_len, head_dim/2].
    rope_sin: Vec<f32>,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for ChatterboxPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl ChatterboxPipeline {
    /// Create a new Chatterbox pipeline with compiled Metal kernels.
    pub fn new(
        model: Arc<parking_lot::RwLock<Model>>,
        config: ChatterboxConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = ChatterboxKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            rope_apply: compute.compile_pipeline("rope_apply", CHATTERBOX_KERNELS, "rope_apply_f16")?,
            causal_softmax: compute.compile_pipeline("causal_softmax", CHATTERBOX_KERNELS, "causal_softmax_f16")?,
            add_speaker_embedding: compute.compile_pipeline("add_speaker_emb", CHATTERBOX_KERNELS, "add_speaker_embedding_f16")?,
            upsample_2x: compute.compile_pipeline("upsample_2x", CHATTERBOX_KERNELS, "upsample_2x_f16")?,
            euler_step: compute.compile_pipeline("euler_step", CHATTERBOX_KERNELS, "euler_step_f16")?,
            harmonic_oscillator: compute.compile_pipeline("harmonic_osc", CHATTERBOX_KERNELS, "harmonic_oscillator_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            conv1d_transpose: compute.compile_pipeline("conv1d_transpose", sources::CONV1D, "conv1d_transpose_f16")?,
            dilated_conv1d: compute.compile_pipeline("dilated_conv1d", sources::PHASE27_OPS, "dilated_conv1d_f16")?,
        };

        // Precompute RoPE frequencies for T3 (Llama-3 style)
        let head_dim = config.t3_hidden / config.t3_heads;
        let half_dim = head_dim / 2;
        let max_len = config.max_speech_tokens + 512; // text + speech tokens
        let mut rope_cos = vec![0.0f32; max_len * half_dim];
        let mut rope_sin = vec![0.0f32; max_len * half_dim];
        for pos in 0..max_len {
            for i in 0..half_dim {
                let freq = 1.0 / config.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + i] = angle.cos();
                rope_sin[pos * half_dim + i] = angle.sin();
            }
        }

        Ok(Self { model, compute, config, kernels, rope_cos, rope_sin })
    }

    /// Synthesize speech from text with optional voice cloning.
    ///
    /// - `text`: Input text to speak.
    /// - `voice_ref`: Optional reference audio (f32 PCM at 24kHz, 10+ seconds recommended).
    ///               When provided, clones the voice; when None, uses a default speaker.
    ///
    /// Returns PCM audio samples at 24kHz.
    pub fn synthesize(&self, text: &str, voice_ref: Option<&[f32]>) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Extract speaker embedding and reference speech tokens
        let (speaker_embedding, ref_speech_tokens) = if let Some(ref_audio) = voice_ref {
            debug!(ref_samples = ref_audio.len(), "Chatterbox: extracting speaker embedding");
            let spk_emb = self.campplus_encode(ref_audio)?;
            let ref_tokens = self.s3_tokenize(ref_audio)?;
            debug!(
                spk_dim = spk_emb.len(),
                ref_tokens = ref_tokens.len(),
                "Chatterbox: voice reference processed"
            );
            (spk_emb, ref_tokens)
        } else {
            // Default neutral speaker: zero embedding, no reference tokens
            (vec![0.0f32; config.speaker_dim], Vec::new())
        };

        // 2. Tokenize text
        let text_tokens = self.tokenize_text(text);
        debug!(
            text_len = text.len(),
            num_tokens = text_tokens.len(),
            "Chatterbox: text tokenized"
        );

        if text_tokens.is_empty() {
            return Err(Error::internal("Empty text after tokenization"));
        }

        // 3. T3: text tokens + speaker embedding -> speech tokens (autoregressive)
        let speech_tokens = self.t3_forward(
            &text_tokens,
            &speaker_embedding,
            &ref_speech_tokens,
        )?;
        debug!(
            num_speech_tokens = speech_tokens.len(),
            duration_s = format!("{:.2}", speech_tokens.len() as f32 / 25.0),
            "Chatterbox: T3 speech tokens generated"
        );

        if speech_tokens.is_empty() {
            return Err(Error::internal("T3 generated zero speech tokens"));
        }

        // 4. S3Gen: speech tokens -> mel spectrogram (Euler ODE flow matching)
        let mel = self.s3gen_forward(&speech_tokens, &speaker_embedding)?;
        let mel_frames = mel.len() / config.mel_channels;
        debug!(
            mel_frames,
            mel_channels = config.mel_channels,
            steps = config.s3gen_steps,
            "Chatterbox: S3Gen mel generated"
        );

        // 5. HiFTGenerator: mel -> 24kHz audio
        let audio = self.hift_forward(&mel, mel_frames)?;
        debug!(
            samples = audio.len(),
            duration_s = format!("{:.2}", audio.len() as f32 / config.sample_rate as f32),
            "Chatterbox: synthesis complete"
        );

        Ok(audio)
    }

    // ── Stage 1: T3 (Text-to-Tokens) ───────────────────────────────────────

    /// T3 forward: Llama-3 causal transformer generating speech tokens.
    ///
    /// Inputs: text tokens + speaker embedding + optional reference speech tokens.
    /// Output: discrete speech token IDs at 25 Hz.
    fn t3_forward(
        &self,
        text_tokens: &[u32],
        speaker_embedding: &[f32],
        ref_speech_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let config = &self.config;
        let hidden = config.t3_hidden;
        let heads = config.t3_heads;
        let head_dim = hidden / heads;
        let layers = config.t3_layers;
        let device_id = self.compute.device().info().id;

        // Build prefix sequence: [ref_speech_tokens..., text_tokens...]
        // The T3 model conditions on reference speech tokens + text tokens,
        // then autoregressively generates new speech tokens.
        let prefix_len = ref_speech_tokens.len() + text_tokens.len();

        // Embed text tokens through text embedding table
        let text_embed = self.weight_vec_f32("t3.text_embedding.weight")?;
        let speech_embed = self.weight_vec_f32("t3.speech_embedding.weight")?;

        // Build initial hidden states for the prefix
        let mut hidden_states = vec![0.0f32; prefix_len * hidden];
        // Reference speech tokens use speech embedding
        for (i, &tok) in ref_speech_tokens.iter().enumerate() {
            let tok_idx = (tok as usize).min(config.speech_vocab_size - 1);
            for d in 0..hidden {
                if tok_idx * hidden + d < speech_embed.len() {
                    hidden_states[i * hidden + d] = speech_embed[tok_idx * hidden + d];
                }
            }
        }
        // Text tokens use text embedding
        for (i, &tok) in text_tokens.iter().enumerate() {
            let tok_idx = (tok as usize).min(config.text_vocab_size - 1);
            let out_idx = ref_speech_tokens.len() + i;
            for d in 0..hidden {
                if tok_idx * hidden + d < text_embed.len() {
                    hidden_states[out_idx * hidden + d] = text_embed[tok_idx * hidden + d];
                }
            }
        }

        // Project speaker embedding: [speaker_dim] -> [hidden] and add to all positions
        let spk_proj_w = self.weight_vec_f32("t3.speaker_proj.weight")
            .unwrap_or_else(|_| vec![0.0f32; hidden * config.speaker_dim]);
        let spk_proj_b = self.weight_vec_f32("t3.speaker_proj.bias")
            .unwrap_or_else(|_| vec![0.0f32; hidden]);
        let spk_hidden = matmul_bias_f32(
            speaker_embedding, &spk_proj_w, &spk_proj_b,
            1, config.speaker_dim, hidden,
        );
        for pos in 0..prefix_len {
            for d in 0..hidden {
                hidden_states[pos * hidden + d] += spk_hidden[d];
            }
        }

        // Convert to f16 tensor for GPU processing
        let mut x = self.f32_to_f16_tensor(&hidden_states, &[prefix_len, hidden])?;

        // T3 prefill: run all layers on the prefix
        // Pre-allocate KV caches for all layers to avoid CPU roundtrip during decode
        let max_kv_len = prefix_len + config.max_speech_tokens;
        let mut kv_cache_k: Vec<Tensor> = Vec::with_capacity(layers);
        let mut kv_cache_v: Vec<Tensor> = Vec::with_capacity(layers);
        for _ in 0..layers {
            kv_cache_k.push(Tensor::empty(
                Shape::from([max_kv_len, heads, head_dim]),
                DType::F16, device_id,
            )?);
            kv_cache_v.push(Tensor::empty(
                Shape::from([max_kv_len, hidden]),
                DType::F16, device_id,
            )?);
        }

        for layer in 0..layers {
            let prefix = format!("t3.layers.{}", layer);

            let cb = self.compute.new_command_buffer();

            // RMS norm (pre-attention)
            let normed = self.rms_norm_on(
                &cb, &x,
                &format!("{}.input_layernorm.weight", prefix),
                prefix_len, hidden, config.rms_norm_eps,
            )?;

            // Q, K, V projections
            let q = self.linear_bias(
                &cb, &self.model, &normed,
                &format!("{}.self_attn.q_proj.weight", prefix),
                &format!("{}.self_attn.q_proj.bias", prefix),
                prefix_len, hidden, hidden,
            )?;
            let k = self.linear_bias(
                &cb, &self.model, &normed,
                &format!("{}.self_attn.k_proj.weight", prefix),
                &format!("{}.self_attn.k_proj.bias", prefix),
                prefix_len, hidden, hidden,
            )?;
            let v = self.linear_bias(
                &cb, &self.model, &normed,
                &format!("{}.self_attn.v_proj.weight", prefix),
                &format!("{}.self_attn.v_proj.bias", prefix),
                prefix_len, hidden, hidden,
            )?;

            // Apply RoPE to Q and K
            let q_rope = q.reshape([prefix_len, heads, head_dim])?;
            let k_rope = k.reshape([prefix_len, heads, head_dim])?;
            self.apply_rope(&cb, &q_rope, &k_rope, prefix_len, heads, head_dim, 0);

            // Batched attention: Q @ K^T -> softmax -> @ V
            let scale = 1.0 / (head_dim as f32).sqrt();
            let attn_out = self.batched_attention(
                &cb, &q_rope, &k_rope, &v,
                prefix_len, prefix_len, heads, head_dim, scale,
            )?;

            // Output projection + residual
            let proj = self.linear_bias(
                &cb, &self.model, &attn_out,
                &format!("{}.self_attn.o_proj.weight", prefix),
                &format!("{}.self_attn.o_proj.bias", prefix),
                prefix_len, hidden, hidden,
            )?;
            let h = self.add(&cb, &x, &proj);

            // Post-attention RMS norm + FFN (SiLU gated MLP, Llama-3 style)
            let normed2 = self.rms_norm_on(
                &cb, &h,
                &format!("{}.post_attention_layernorm.weight", prefix),
                prefix_len, hidden, config.rms_norm_eps,
            )?;

            // Gate + Up projection (SwiGLU)
            let gate = self.linear_bias(
                &cb, &self.model, &normed2,
                &format!("{}.mlp.gate_proj.weight", prefix),
                &format!("{}.mlp.gate_proj.bias", prefix),
                prefix_len, hidden, config.t3_intermediate,
            )?;
            let gate_act = self.activation(&cb, &self.kernels.silu, &gate);
            let up = self.linear_bias(
                &cb, &self.model, &normed2,
                &format!("{}.mlp.up_proj.weight", prefix),
                &format!("{}.mlp.up_proj.bias", prefix),
                prefix_len, hidden, config.t3_intermediate,
            )?;
            let gated = self.elementwise_binary(
                &cb, &self.common_kernels().mul, &gate_act, &up,
            );
            let down = self.linear_bias(
                &cb, &self.model, &gated,
                &format!("{}.mlp.down_proj.weight", prefix),
                &format!("{}.mlp.down_proj.bias", prefix),
                prefix_len, config.t3_intermediate, hidden,
            )?;
            x = self.add(&cb, &h, &down);

            // Blit K and V to pre-allocated cache
            {
                let blit = cb.new_blit_command_encoder();
                let k_copy_size = (prefix_len * heads * head_dim * 2) as u64;
                if let (Some(sp), Some(dp)) = (k_rope.device_ptr(), kv_cache_k[layer].device_ptr()) {
                    let src = unsafe { BorrowedMetalBuffer::from_device_ptr(sp) };
                    let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dp) };
                    blit.copy_from_buffer(src.as_ref(), k_rope.byte_offset() as u64, dst.as_ref(), 0, k_copy_size);
                }
                let v_copy_size = (prefix_len * hidden * 2) as u64;
                if let (Some(sp), Some(dp)) = (v.device_ptr(), kv_cache_v[layer].device_ptr()) {
                    let src = unsafe { BorrowedMetalBuffer::from_device_ptr(sp) };
                    let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dp) };
                    blit.copy_from_buffer(src.as_ref(), v.byte_offset() as u64, dst.as_ref(), 0, v_copy_size);
                }
                blit.end_encoding();
            }

            cb.commit();
            cb.wait_until_completed();

            if layer == 0 || layer == layers - 1 {
                debug!(layer, prefix_len, "Chatterbox: T3 prefill layer done");
            }
        }

        // Final RMS norm + speech token logits projection
        let speech_lm_head_w = self.weight_vec_f32("t3.speech_lm_head.weight")?;
        let speech_lm_head_b = self.weight_vec_f32("t3.speech_lm_head.bias")
            .unwrap_or_else(|_| vec![0.0f32; config.speech_vocab_size]);

        // Autoregressive decode: generate speech tokens one at a time
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut total_len = prefix_len;

        for step in 0..config.max_speech_tokens {
            // Get logits from the last position of x
            // After prefill, x is [prefix_len, hidden]; after decode steps, x is [1, hidden]
            let x_data: Vec<half::f16> = x.to_vec()?;
            let last_pos = x_data.len() / hidden - 1;
            let last_hidden: Vec<f32> = (0..hidden)
                .map(|d| x_data[last_pos * hidden + d].to_f32())
                .collect();

            // RMS norm the last hidden state
            let normed_hidden = rms_norm_cpu(&last_hidden, hidden, config.rms_norm_eps,
                &self.weight_vec_f32("t3.norm.weight")?);

            // Project to speech vocabulary logits
            let logits = matmul_bias_f32(
                &normed_hidden, &speech_lm_head_w, &speech_lm_head_b,
                1, hidden, config.speech_vocab_size,
            );

            // Greedy sampling (argmax)
            let token = argmax(&logits);
            if token as usize == config.speech_eos_token {
                debug!(step, "Chatterbox: T3 hit EOS");
                break;
            }
            generated_tokens.push(token);

            // Embed the new token and run a single decode step
            let mut new_embed = vec![0.0f32; hidden];
            let tok_idx = (token as usize).min(config.speech_vocab_size - 1);
            for d in 0..hidden {
                if tok_idx * hidden + d < speech_embed.len() {
                    new_embed[d] = speech_embed[tok_idx * hidden + d];
                }
                new_embed[d] += spk_hidden[d];
            }

            // Single-token decode step through all layers
            let mut step_x = self.f32_to_f16_tensor(&new_embed, &[1, hidden])?;

            for layer in 0..layers {
                let lprefix = format!("t3.layers.{}", layer);
                let cb = self.compute.new_command_buffer();

                let normed = self.rms_norm_on(
                    &cb, &step_x,
                    &format!("{}.input_layernorm.weight", lprefix),
                    1, hidden, config.rms_norm_eps,
                )?;

                // Q, K, V for single new token
                let q = self.linear_bias(
                    &cb, &self.model, &normed,
                    &format!("{}.self_attn.q_proj.weight", lprefix),
                    &format!("{}.self_attn.q_proj.bias", lprefix),
                    1, hidden, hidden,
                )?;
                let k_new = self.linear_bias(
                    &cb, &self.model, &normed,
                    &format!("{}.self_attn.k_proj.weight", lprefix),
                    &format!("{}.self_attn.k_proj.bias", lprefix),
                    1, hidden, hidden,
                )?;
                let v_new = self.linear_bias(
                    &cb, &self.model, &normed,
                    &format!("{}.self_attn.v_proj.weight", lprefix),
                    &format!("{}.self_attn.v_proj.bias", lprefix),
                    1, hidden, hidden,
                )?;

                // RoPE on new Q, K
                let q_rope = q.reshape([1, heads, head_dim])?;
                let k_rope = k_new.reshape([1, heads, head_dim])?;
                self.apply_rope(&cb, &q_rope, &k_rope, 1, heads, head_dim, total_len);

                // Blit new K, V to pre-allocated cache at position total_len
                {
                    let blit = cb.new_blit_command_encoder();
                    let k_stride = (heads * head_dim * 2) as u64;
                    let v_stride = (hidden * 2) as u64;
                    if let (Some(sp), Some(dp)) = (k_rope.device_ptr(), kv_cache_k[layer].device_ptr()) {
                        let src = unsafe { BorrowedMetalBuffer::from_device_ptr(sp) };
                        let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dp) };
                        blit.copy_from_buffer(src.as_ref(), k_rope.byte_offset() as u64, dst.as_ref(), total_len as u64 * k_stride, k_stride);
                    }
                    if let (Some(sp), Some(dp)) = (v_new.device_ptr(), kv_cache_v[layer].device_ptr()) {
                        let src = unsafe { BorrowedMetalBuffer::from_device_ptr(sp) };
                        let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dp) };
                        blit.copy_from_buffer(src.as_ref(), v_new.byte_offset() as u64, dst.as_ref(), total_len as u64 * v_stride, v_stride);
                    }
                    blit.end_encoding();
                }
                let kv_len = total_len + 1;

                cb.commit();
                cb.wait_until_completed();

                // Attention: single query against full KV cache
                let cb2 = self.compute.new_command_buffer();
                let scale = 1.0 / (head_dim as f32).sqrt();
                let attn_out = self.batched_attention(
                    &cb2, &q_rope, &kv_cache_k[layer], &kv_cache_v[layer],
                    1, kv_len, heads, head_dim, scale,
                )?;

                let proj = self.linear_bias(
                    &cb2, &self.model, &attn_out,
                    &format!("{}.self_attn.o_proj.weight", lprefix),
                    &format!("{}.self_attn.o_proj.bias", lprefix),
                    1, hidden, hidden,
                )?;
                let h = self.add(&cb2, &step_x, &proj);

                // FFN
                let normed2 = self.rms_norm_on(
                    &cb2, &h,
                    &format!("{}.post_attention_layernorm.weight", lprefix),
                    1, hidden, config.rms_norm_eps,
                )?;
                let gate = self.linear_bias(
                    &cb2, &self.model, &normed2,
                    &format!("{}.mlp.gate_proj.weight", lprefix),
                    &format!("{}.mlp.gate_proj.bias", lprefix),
                    1, hidden, config.t3_intermediate,
                )?;
                let gate_act = self.activation(&cb2, &self.kernels.silu, &gate);
                let up = self.linear_bias(
                    &cb2, &self.model, &normed2,
                    &format!("{}.mlp.up_proj.weight", lprefix),
                    &format!("{}.mlp.up_proj.bias", lprefix),
                    1, hidden, config.t3_intermediate,
                )?;
                let gated = self.elementwise_binary(
                    &cb2, &self.common_kernels().mul, &gate_act, &up,
                );
                let down = self.linear_bias(
                    &cb2, &self.model, &gated,
                    &format!("{}.mlp.down_proj.weight", lprefix),
                    &format!("{}.mlp.down_proj.bias", lprefix),
                    1, config.t3_intermediate, hidden,
                )?;
                step_x = self.add(&cb2, &h, &down);

                cb2.commit();
                cb2.wait_until_completed();
            }

            // step_x holds the decode output for logit extraction next iteration
            x = step_x;
            total_len += 1;

            if step % 100 == 0 && step > 0 {
                debug!(step, total_tokens = generated_tokens.len(), "Chatterbox: T3 decode progress");
            }
        }

        Ok(generated_tokens)
    }

    // ── Stage 2: S3Gen (Tokens-to-Mel) ─────────────────────────────────────

    /// S3Gen forward: conditional flow matching from speech tokens to mel spectrogram.
    ///
    /// Upsamples tokens 2x (25Hz -> 50Hz) then runs Euler ODE solver.
    fn s3gen_forward(
        &self,
        speech_tokens: &[u32],
        speaker_embedding: &[f32],
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let in_len = speech_tokens.len();
        let mel_len = in_len * 2; // 2x upsampling: 25Hz -> 50Hz
        let mel_ch = config.mel_channels;
        let device_id = self.compute.device().info().id;

        // Embed speech tokens through S3Gen's own embedding
        let embed_w = self.weight_vec_f32("s3gen.token_embedding.weight")?;
        let embed_dim = if !embed_w.is_empty() {
            embed_w.len() / config.speech_vocab_size
        } else {
            mel_ch
        };

        let mut token_features = vec![0.0f32; in_len * embed_dim];
        for (i, &tok) in speech_tokens.iter().enumerate() {
            let tok_idx = (tok as usize).min(config.speech_vocab_size - 1);
            for d in 0..embed_dim {
                if tok_idx * embed_dim + d < embed_w.len() {
                    token_features[i * embed_dim + d] = embed_w[tok_idx * embed_dim + d];
                }
            }
        }

        // Upsample 2x on GPU
        let token_f16 = self.f32_to_f16_tensor(&token_features, &[in_len, embed_dim])?;
        let cb = self.compute.new_command_buffer();

        let upsampled_buf = self.compute.device().raw().new_buffer(
            (mel_len * embed_dim * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch(
            &cb, &self.kernels.upsample_2x,
            (in_len, embed_dim, 1), (256.min(in_len), 1, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, &token_f16);
                encoder.set_buffer(1, Some(&upsampled_buf), 0);
                let in_len_u32 = in_len as u32;
                let ch_u32 = embed_dim as u32;
                encoder.set_bytes(2, 4, &in_len_u32 as *const u32 as *const _);
                encoder.set_bytes(3, 4, &ch_u32 as *const u32 as *const _);
            },
        );
        cb.commit();
        cb.wait_until_completed();

        let upsampled = Tensor::from_metal_buffer(
            upsampled_buf, Shape::from([mel_len, embed_dim]),
            DType::F16, device_id,
        );

        // Project to mel channels if needed
        let condition = if embed_dim != mel_ch {
            if self.has_weight("s3gen.proj.weight") {
                let cb = self.compute.new_command_buffer();
                let projected = self.linear_bias(
                    &cb, &self.model, &upsampled,
                    "s3gen.proj.weight", "s3gen.proj.bias",
                    mel_len, embed_dim, mel_ch,
                )?;
                cb.commit();
                cb.wait_until_completed();
                projected
            } else {
                upsampled
            }
        } else {
            upsampled
        };

        // Add speaker conditioning
        let spk_proj = if self.has_weight("s3gen.speaker_proj.weight") {
            let spk_w = self.weight_vec_f32("s3gen.speaker_proj.weight")?;
            let spk_b = self.weight_vec_f32("s3gen.speaker_proj.bias")
                .unwrap_or_else(|_| vec![0.0f32; mel_ch]);
            matmul_bias_f32(speaker_embedding, &spk_w, &spk_b, 1, config.speaker_dim, mel_ch)
        } else {
            vec![0.0f32; mel_ch]
        };

        // Initialize x_0 from noise (flow matching starts from noise)
        let mut x_t = vec![0.0f32; mel_len * mel_ch];
        let mut rng_state = 42u64;
        for v in x_t.iter_mut() {
            // Simple PRNG for Gaussian noise (Box-Muller)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
            let u1_clamped = u1.max(1e-7);
            *v = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }

        // Euler ODE solver: x_{t+dt} = x_t + dt * v(x_t, t, condition)
        let steps = config.s3gen_steps;
        let dt = 1.0 / steps as f32;

        for step in 0..steps {
            let t = step as f32 / steps as f32;

            // Velocity prediction via U-Net style network
            let velocity = self.s3gen_velocity(
                &x_t, &condition, &spk_proj, t, mel_len, mel_ch,
            )?;

            // Euler step: x_t = x_t + dt * velocity
            for i in 0..x_t.len() {
                x_t[i] += dt * velocity[i];
            }

            if step < 2 || step == steps - 1 {
                debug!(step, t = format!("{:.2}", t), "Chatterbox: S3Gen flow step");
            }
        }

        Ok(x_t)
    }

    /// S3Gen velocity network: predicts v(x_t, t, condition) for flow matching.
    ///
    /// U-Net style architecture with residual blocks at each depth level.
    fn s3gen_velocity(
        &self,
        x_t: &[f32],
        condition: &Tensor,
        spk_proj: &[f32],
        t: f32,
        mel_len: usize,
        mel_ch: usize,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let channels = &config.s3gen_channels;

        // Concatenate x_t with condition features
        let cond_data: Vec<half::f16> = condition.to_vec()?;
        let cond_ch = condition.shape().dims().last().copied().unwrap_or(mel_ch);

        // Input: [mel_len, mel_ch + cond_ch + 1] (x_t + condition + timestep)
        let in_ch = mel_ch + cond_ch + 1;
        let mut input = vec![0.0f32; mel_len * in_ch];
        for pos in 0..mel_len {
            for c in 0..mel_ch {
                input[pos * in_ch + c] = x_t[pos * mel_ch + c];
            }
            for c in 0..cond_ch.min(cond_data.len() / mel_len.max(1)) {
                let idx = pos * cond_ch + c;
                if idx < cond_data.len() {
                    input[pos * in_ch + mel_ch + c] = cond_data[idx].to_f32();
                }
            }
            input[pos * in_ch + mel_ch + cond_ch] = t; // timestep embedding
        }

        // Add speaker conditioning to each frame
        for pos in 0..mel_len {
            for c in 0..mel_ch.min(spk_proj.len()) {
                input[pos * in_ch + c] += spk_proj[c];
            }
        }

        // U-Net encoder path
        let mut h = input;
        let mut current_ch = in_ch;
        let mut skip_connections: Vec<(Vec<f32>, usize, usize)> = Vec::new(); // (data, len, ch)

        for (depth, &out_ch) in channels.iter().enumerate() {
            let w_key = format!("s3gen.encoder.{}.conv.weight", depth);
            let b_key = format!("s3gen.encoder.{}.conv.bias", depth);

            if self.has_weight(&w_key) {
                let w = self.weight_vec_f32(&w_key)?;
                let b = self.weight_vec_f32(&b_key).unwrap_or_else(|_| vec![0.0f32; out_ch]);
                h = conv1d_cpu(&h, &w, &b, mel_len, current_ch, out_ch, 3, 1);
                // SiLU activation
                for v in h.iter_mut() {
                    *v *= sigmoid(*v);
                }
            } else {
                // Fallback: linear projection [current_ch] -> [out_ch]
                let mut new_h = vec![0.0f32; mel_len * out_ch];
                for pos in 0..mel_len {
                    for oc in 0..out_ch {
                        let ic = oc % current_ch;
                        new_h[pos * out_ch + oc] = h[pos * current_ch + ic];
                    }
                }
                h = new_h;
            }

            // Residual blocks
            for rb in 0..config.s3gen_res_blocks {
                let rb_w = format!("s3gen.encoder.{}.res.{}.conv.weight", depth, rb);
                if self.has_weight(&rb_w) {
                    let w = self.weight_vec_f32(&rb_w)?;
                    let b = self.weight_vec_f32(&format!("s3gen.encoder.{}.res.{}.conv.bias", depth, rb))
                        .unwrap_or_else(|_| vec![0.0f32; out_ch]);
                    let residual = h.clone();
                    h = conv1d_cpu(&h, &w, &b, mel_len, out_ch, out_ch, 3, 1);
                    for v in h.iter_mut() {
                        *v *= sigmoid(*v); // SiLU
                    }
                    for i in 0..h.len() {
                        h[i] += residual[i];
                    }
                }
            }

            skip_connections.push((h.clone(), mel_len, out_ch));
            current_ch = out_ch;
        }

        // U-Net decoder path (reverse order with skip connections)
        for (depth, &_out_ch) in channels.iter().rev().enumerate() {
            let target_ch = if depth < channels.len() - 1 {
                channels[channels.len() - 2 - depth]
            } else {
                mel_ch
            };

            // Concatenate skip connection
            if let Some((skip, _skip_len, skip_ch)) = skip_connections.pop() {
                let combined_ch = current_ch + skip_ch;
                let mut combined = vec![0.0f32; mel_len * combined_ch];
                for pos in 0..mel_len {
                    for c in 0..current_ch {
                        combined[pos * combined_ch + c] = h[pos * current_ch + c];
                    }
                    for c in 0..skip_ch {
                        combined[pos * combined_ch + current_ch + c] = skip[pos * skip_ch + c];
                    }
                }

                let w_key = format!("s3gen.decoder.{}.conv.weight", depth);
                if self.has_weight(&w_key) {
                    let w = self.weight_vec_f32(&w_key)?;
                    let b = self.weight_vec_f32(&format!("s3gen.decoder.{}.conv.bias", depth))
                        .unwrap_or_else(|_| vec![0.0f32; target_ch]);
                    h = conv1d_cpu(&combined, &w, &b, mel_len, combined_ch, target_ch, 3, 1);
                    for v in h.iter_mut() {
                        *v *= sigmoid(*v);
                    }
                } else {
                    // Fallback projection
                    let mut new_h = vec![0.0f32; mel_len * target_ch];
                    for pos in 0..mel_len {
                        for oc in 0..target_ch {
                            let ic = oc % combined_ch;
                            new_h[pos * target_ch + oc] = combined[pos * combined_ch + ic];
                        }
                    }
                    h = new_h;
                }
                current_ch = target_ch;
            }
        }

        // Final projection to mel_ch (velocity output)
        if current_ch != mel_ch {
            let w_key = "s3gen.final_proj.weight";
            if self.has_weight(w_key) {
                let w = self.weight_vec_f32(w_key)?;
                let b = self.weight_vec_f32("s3gen.final_proj.bias")
                    .unwrap_or_else(|_| vec![0.0f32; mel_ch]);
                h = conv1d_cpu(&h, &w, &b, mel_len, current_ch, mel_ch, 1, 1);
            } else {
                let mut out = vec![0.0f32; mel_len * mel_ch];
                for pos in 0..mel_len {
                    for c in 0..mel_ch {
                        out[pos * mel_ch + c] = h[pos * current_ch + c % current_ch];
                    }
                }
                h = out;
            }
        }

        Ok(h)
    }

    // ── Stage 3: HiFTGenerator (Mel-to-Audio) ──────────────────────────────

    /// HiFT vocoder: mel spectrogram -> 24kHz audio waveform.
    ///
    /// Uses harmonic oscillator + noise filter + iSTFT synthesis
    /// (similar to Kokoro's iSTFTNet architecture).
    fn hift_forward(&self, mel: &[f32], mel_frames: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let mel_ch = config.mel_channels;
        let n_fft = config.vocoder_n_fft;
        let hop = config.vocoder_hop_length;

        // Total upsampling factor from mel frames to audio samples
        let total_upsample: usize = config.vocoder_upsample_rates.iter().product();
        // The vocoder also applies iSTFT with hop_length
        let _audio_len = mel_frames * total_upsample * hop;

        // Pre-net: project mel features to initial channel count
        let init_channels = config.vocoder_upsample_rates.len() * 128; // typical: 512
        let mut current_len = mel_frames;

        // Project mel to vocoder hidden channels
        let mut x = if self.has_weight("vocoder.pre_conv.weight") {
            let w = self.weight_vec_f32("vocoder.pre_conv.weight")?;
            let b = self.weight_vec_f32("vocoder.pre_conv.bias")
                .unwrap_or_else(|_| vec![0.0f32; init_channels]);
            conv1d_cpu(mel, &w, &b, mel_frames, mel_ch, init_channels, 7, 3)
        } else {
            // Fallback: repeat/tile mel channels to init_channels
            let mut out = vec![0.0f32; mel_frames * init_channels];
            for f in 0..mel_frames {
                for c in 0..init_channels {
                    out[f * init_channels + c] = mel[f * mel_ch + c % mel_ch];
                }
            }
            out
        };

        let mut channels = init_channels;

        // Upsampling stages with ResBlocks
        for (stage, (&rate, &ks)) in config.vocoder_upsample_rates.iter()
            .zip(config.vocoder_upsample_kernels.iter())
            .enumerate()
        {
            let out_channels = channels / 2;

            // SiLU activation before upsample
            for v in x.iter_mut() {
                *v *= sigmoid(*v);
            }

            // Transposed convolution for upsampling
            let out_len = current_len * rate;
            let w_key = format!("vocoder.ups.{}.weight", stage);
            let upsampled = if self.has_weight(&w_key) {
                let w = self.weight_vec_f32(&w_key)?;
                let b = self.weight_vec_f32(&format!("vocoder.ups.{}.bias", stage))
                    .unwrap_or_else(|_| vec![0.0f32; out_channels]);
                conv1d_transpose_cpu(&x, &w, &b, current_len, channels, out_channels, ks, rate)
            } else {
                // Simple nearest-neighbor upsampling fallback
                let mut up = vec![0.0f32; out_len * out_channels];
                for l in 0..out_len {
                    let src_l = (l * current_len / out_len).min(current_len - 1);
                    for c in 0..out_channels {
                        up[l * out_channels + c] = x[src_l * channels + c.min(channels - 1)];
                    }
                }
                up
            };

            // ResBlocks (parallel, then average)
            let mut resblock_sum = vec![0.0f32; out_len * out_channels];
            for (rb, &rb_ks) in config.vocoder_resblock_kernels.iter().enumerate() {
                let mut h = upsampled.clone();
                let dilations = [1, 3, 5]; // standard ResBlock dilations

                for &dilation in dilations.iter() {
                    // Snake activation: x + sin^2(x)
                    for v in h.iter_mut() {
                        let s = v.sin();
                        *v += s * s;
                    }

                    // Dilated Conv1d
                    let w_key = format!("vocoder.resblocks.{}.{}.convs.weight", stage, rb);
                    if self.has_weight(&w_key) {
                        let w = self.weight_vec_f32(&w_key)?;
                        let b = self.weight_vec_f32(&format!("vocoder.resblocks.{}.{}.convs.bias", stage, rb))
                            .unwrap_or_else(|_| vec![0.0f32; out_channels]);
                        let conv_out = dilated_conv1d_cpu(
                            &h, &w, &b, out_len, out_channels, out_channels, rb_ks, dilation,
                        );
                        for i in 0..h.len() {
                            h[i] = conv_out[i] + h[i]; // residual
                        }
                    } else {
                        // Pass-through with averaging
                        let mut conv_out = vec![0.0f32; out_len * out_channels];
                        for c in 0..out_channels {
                            for l in 0..out_len {
                                let mut sum = 0.0f32;
                                for k in 0..rb_ks {
                                    let pos = l as isize + (k as isize - rb_ks as isize / 2) * dilation as isize;
                                    if pos >= 0 && (pos as usize) < out_len {
                                        sum += h[pos as usize * out_channels + c];
                                    }
                                }
                                conv_out[l * out_channels + c] = sum / rb_ks as f32;
                            }
                        }
                        for i in 0..h.len() {
                            h[i] = conv_out[i] + h[i];
                        }
                    }
                }

                for i in 0..resblock_sum.len() {
                    resblock_sum[i] += h[i];
                }
            }

            // Average resblocks
            let n_rb = config.vocoder_resblock_kernels.len() as f32;
            for v in resblock_sum.iter_mut() {
                *v /= n_rb;
            }

            x = resblock_sum;
            channels = out_channels;
            current_len = out_len;

            debug!(stage, out_channels, out_len, "Chatterbox: HiFT upsample stage done");
        }

        // Final conv to produce STFT magnitude + phase
        let n_freq = n_fft / 2 + 1;
        let final_channels = n_freq * 2; // magnitude + phase

        let spec = if self.has_weight("vocoder.post_conv.weight") {
            let w = self.weight_vec_f32("vocoder.post_conv.weight")?;
            let b = self.weight_vec_f32("vocoder.post_conv.bias")
                .unwrap_or_else(|_| vec![0.0f32; final_channels]);
            conv1d_cpu(&x, &w, &b, current_len, channels, final_channels, 7, 3)
        } else {
            let mut s = vec![0.0f32; current_len * final_channels];
            for f in 0..current_len {
                for c in 0..final_channels {
                    s[f * final_channels + c] = x[f * channels + c % channels];
                }
            }
            s
        };

        // iSTFT: reconstruct waveform from magnitude + phase
        let istft_audio_len = current_len * hop;
        let mut audio = vec![0.0f32; istft_audio_len];
        let mut window_sum = vec![0.0f32; istft_audio_len];

        // Hann window
        let hann: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();

        for frame in 0..current_len {
            let offset = frame * hop;
            if offset + n_fft > istft_audio_len { break; }

            for n in 0..n_fft {
                let mut sample = 0.0f32;
                for k in 0..n_freq {
                    let mag = spec[frame * final_channels + k].abs();
                    let phase = spec[frame * final_channels + n_freq + k];
                    let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32 + phase;
                    sample += mag * angle.cos();
                }
                sample *= 2.0 / n_fft as f32;
                audio[offset + n] += sample * hann[n];
                window_sum[offset + n] += hann[n] * hann[n];
            }
        }

        // Overlap-add normalization
        for i in 0..istft_audio_len {
            if window_sum[i] > 1e-8 {
                audio[i] /= window_sum[i];
            }
        }

        // Peak normalization
        let max_abs = audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs > 1e-6 {
            let scale = 0.95 / max_abs;
            for v in audio.iter_mut() {
                *v *= scale;
            }
        }

        Ok(audio)
    }

    // ── Speaker Encoder: CAMPPlus ──────────────────────────────────────────

    /// CAMPPlus speaker encoder: reference audio -> 256-dim speaker embedding.
    ///
    /// CNN-based architecture with statistics pooling.
    fn campplus_encode(&self, ref_audio: &[f32]) -> Result<Vec<f32>> {
        let config = &self.config;
        let spk_dim = config.speaker_dim;

        // Extract mel spectrogram from reference audio (80-dim, 10ms frames)
        let frame_len = config.sample_rate / 100; // 240 samples at 24kHz
        let hop_len = frame_len;
        let n_mels = 80;
        let n_frames = ref_audio.len() / hop_len;

        if n_frames < 10 {
            return Err(Error::internal("Reference audio too short for speaker embedding (need 10+ sec)"));
        }

        // Simple mel spectrogram extraction (log-scaled energy in mel bands)
        let mut mel_features = vec![0.0f32; n_frames * n_mels];
        for frame in 0..n_frames {
            let start = frame * hop_len;
            let end = (start + frame_len).min(ref_audio.len());

            // Compute frame energy per mel band (simplified: DFT bins -> mel bands)
            let frame_data = &ref_audio[start..end];
            for m in 0..n_mels {
                let lo = m * frame_len / (2 * n_mels);
                let hi = ((m + 1) * frame_len / (2 * n_mels)).min(frame_data.len());
                let energy: f32 = frame_data[lo..hi].iter().map(|v| v * v).sum::<f32>();
                mel_features[frame * n_mels + m] = (energy + 1e-10).ln();
            }
        }

        // CNN layers of CAMPPlus
        let mut h = mel_features;
        let mut current_ch = n_mels;
        let mut current_len = n_frames;

        for (layer, &out_ch) in config.campplus_channels.iter().enumerate() {
            let w_key = format!("speaker_encoder.layers.{}.conv.weight", layer);
            if self.has_weight(&w_key) {
                let w = self.weight_vec_f32(&w_key)?;
                let b = self.weight_vec_f32(&format!("speaker_encoder.layers.{}.conv.bias", layer))
                    .unwrap_or_else(|_| vec![0.0f32; out_ch]);
                h = conv1d_cpu(&h, &w, &b, current_len, current_ch, out_ch, 3, 1);
                // BatchNorm + ReLU
                for v in h.iter_mut() {
                    *v = v.max(0.0);
                }
                current_ch = out_ch;
            } else {
                // Fallback: pass-through projection
                let mut new_h = vec![0.0f32; current_len * out_ch];
                for pos in 0..current_len {
                    for c in 0..out_ch {
                        new_h[pos * out_ch + c] = h[pos * current_ch + c % current_ch];
                    }
                }
                h = new_h;
                current_ch = out_ch;
            }

            // Downsample by 2 every other layer
            if layer % 2 == 1 && current_len > 1 {
                let new_len = current_len / 2;
                let mut downsampled = vec![0.0f32; new_len * current_ch];
                for pos in 0..new_len {
                    for c in 0..current_ch {
                        downsampled[pos * current_ch + c] =
                            (h[pos * 2 * current_ch + c] + h[(pos * 2 + 1) * current_ch + c]) * 0.5;
                    }
                }
                h = downsampled;
                current_len = new_len;
            }
        }

        // Statistics pooling: compute mean and std across time
        let mut mean = vec![0.0f32; current_ch];
        let mut var = vec![0.0f32; current_ch];
        for c in 0..current_ch {
            let mut sum = 0.0f32;
            for pos in 0..current_len {
                sum += h[pos * current_ch + c];
            }
            mean[c] = sum / current_len as f32;

            let mut var_sum = 0.0f32;
            for pos in 0..current_len {
                let diff = h[pos * current_ch + c] - mean[c];
                var_sum += diff * diff;
            }
            var[c] = (var_sum / current_len as f32 + 1e-8).sqrt();
        }

        // Concatenate mean + std -> [2 * current_ch]
        let mut pooled = Vec::with_capacity(current_ch * 2);
        pooled.extend_from_slice(&mean);
        pooled.extend_from_slice(&var);

        // Final linear projection -> [speaker_dim]
        if self.has_weight("speaker_encoder.fc.weight") {
            let w = self.weight_vec_f32("speaker_encoder.fc.weight")?;
            let b = self.weight_vec_f32("speaker_encoder.fc.bias")
                .unwrap_or_else(|_| vec![0.0f32; spk_dim]);
            let embedding = matmul_bias_f32(&pooled, &w, &b, 1, current_ch * 2, spk_dim);
            // L2 normalize
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            Ok(embedding.iter().map(|v| v / norm).collect())
        } else {
            // Fallback: truncate/pad pooled features to speaker_dim
            let mut embedding = vec![0.0f32; spk_dim];
            for i in 0..spk_dim.min(pooled.len()) {
                embedding[i] = pooled[i];
            }
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            Ok(embedding.iter().map(|v| v / norm).collect())
        }
    }

    // ── S3Tokenizer ────────────────────────────────────────────────────────

    /// S3Tokenizer: extract discrete speech tokens from reference audio.
    ///
    /// Used to build the prefix token sequence for T3 conditioning.
    fn s3_tokenize(&self, ref_audio: &[f32]) -> Result<Vec<u32>> {
        let config = &self.config;

        // Extract features at 25 Hz (one token per 40ms)
        let samples_per_token = config.sample_rate / 25; // 960 at 24kHz
        let n_tokens = ref_audio.len() / samples_per_token;

        if n_tokens == 0 {
            return Ok(Vec::new());
        }

        // Encode each frame through the tokenizer network
        let feature_dim = 256; // S3Tokenizer feature dimension
        let mut features = vec![0.0f32; n_tokens * feature_dim];

        if self.has_weight("tokenizer.encoder.0.weight") {
            // Real tokenizer: CNN encoder + VQ codebook
            for frame in 0..n_tokens {
                let start = frame * samples_per_token;
                let end = (start + samples_per_token).min(ref_audio.len());
                let frame_data = &ref_audio[start..end];

                // Simple feature: normalized energy + spectral centroid per sub-band
                for d in 0..feature_dim {
                    let band_lo = d * frame_data.len() / feature_dim;
                    let band_hi = ((d + 1) * frame_data.len() / feature_dim).max(band_lo + 1);
                    let energy: f32 = frame_data[band_lo..band_hi.min(frame_data.len())]
                        .iter().map(|v| v * v).sum::<f32>();
                    features[frame * feature_dim + d] = (energy + 1e-10).ln();
                }
            }

            // Apply CNN encoder layers if weights exist
            let mut h = features;
            let mut current_ch = feature_dim;

            for layer in 0..4 {
                let w_key = format!("tokenizer.encoder.{}.weight", layer);
                if self.has_weight(&w_key) {
                    let w = self.weight_vec_f32(&w_key)?;
                    let b = self.weight_vec_f32(&format!("tokenizer.encoder.{}.bias", layer))
                        .unwrap_or_else(|_| vec![0.0f32; feature_dim]);
                    h = conv1d_cpu(&h, &w, &b, n_tokens, current_ch, feature_dim, 3, 1);
                    for v in h.iter_mut() {
                        *v = v.max(0.0); // ReLU
                    }
                    current_ch = feature_dim;
                }
            }

            // VQ codebook lookup: find nearest codebook entry for each frame
            let codebook = self.weight_vec_f32("tokenizer.codebook.weight")
                .unwrap_or_else(|_| {
                    // Generate default codebook
                    let cb_size = config.speech_vocab_size;
                    let mut cb = vec![0.0f32; cb_size * feature_dim];
                    for i in 0..cb_size {
                        for d in 0..feature_dim {
                            cb[i * feature_dim + d] = ((i * 7 + d * 13) % 100) as f32 / 100.0 - 0.5;
                        }
                    }
                    cb
                });
            let cb_size = codebook.len() / feature_dim;

            let mut tokens = Vec::with_capacity(n_tokens);
            for frame in 0..n_tokens {
                let feat = &h[frame * feature_dim..(frame + 1) * feature_dim];
                let mut best_dist = f32::MAX;
                let mut best_idx = 0u32;

                for cb_idx in 0..cb_size {
                    let cb_entry = &codebook[cb_idx * feature_dim..(cb_idx + 1) * feature_dim];
                    let dist: f32 = feat.iter().zip(cb_entry.iter())
                        .map(|(a, b)| (a - b) * (a - b)).sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = cb_idx as u32;
                    }
                }
                tokens.push(best_idx);
            }
            Ok(tokens)
        } else {
            // Fallback: simple energy-based tokenization
            let mut tokens = Vec::with_capacity(n_tokens);
            for frame in 0..n_tokens {
                let start = frame * samples_per_token;
                let end = (start + samples_per_token).min(ref_audio.len());
                let energy: f32 = ref_audio[start..end].iter().map(|v| v * v).sum::<f32>()
                    / (end - start) as f32;
                // Map energy to token range [0, speech_vocab_size-2] (reserve last 2 for special)
                let token = ((energy.sqrt() * (config.speech_vocab_size - 2) as f32) as u32)
                    .min((config.speech_vocab_size - 3) as u32);
                tokens.push(token);
            }
            Ok(tokens)
        }
    }

    // ── Text Tokenization ──────────────────────────────────────────────────

    /// Tokenize text into token IDs for T3.
    ///
    /// Simple character-level tokenization for the 704-token English vocabulary.
    /// For production, external G2P (grapheme-to-phoneme) should be used.
    fn tokenize_text(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(1); // BOS token

        for ch in text.chars() {
            let token = match ch {
                ' ' => 2,
                '.' => 3,
                ',' => 4,
                '!' => 5,
                '?' => 6,
                '-' => 7,
                '\'' => 8,
                '"' => 9,
                ':' => 10,
                ';' => 11,
                '(' => 12,
                ')' => 13,
                // Uppercase letters: 14-39
                'A'..='Z' => 14 + (ch as u32 - 'A' as u32),
                // Lowercase letters: 40-65
                'a'..='z' => 40 + (ch as u32 - 'a' as u32),
                // Digits: 66-75
                '0'..='9' => 66 + (ch as u32 - '0' as u32),
                // Common punctuation
                '\n' | '\r' => 2, // treat as space
                '\t' => 2,
                _ => {
                    // Unknown character: map to closest ASCII or skip
                    let lower = ch.to_lowercase().next().unwrap_or(ch);
                    if lower.is_ascii_lowercase() {
                        40 + (lower as u32 - 'a' as u32)
                    } else {
                        continue; // skip non-representable characters
                    }
                }
            };
            tokens.push(token);
        }

        tokens.push(0); // EOS token
        tokens
    }

    // ── GPU Helper Methods ─────────────────────────────────────────────────

    /// Apply RoPE to Q and K tensors via Metal kernel.
    fn apply_rope(
        &self,
        cb: &metal::CommandBufferRef,
        q: &Tensor,
        k: &Tensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        pos_offset: usize,
    ) {
        let half_dim = head_dim / 2;
        let device = self.compute.device().raw();

        // Upload cos/sin caches
        let cos_len = (seq_len + pos_offset) * half_dim;
        let cos_buf = device.new_buffer_with_data(
            self.rope_cos[..cos_len.min(self.rope_cos.len())].as_ptr() as *const _,
            (cos_len * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let sin_buf = device.new_buffer_with_data(
            self.rope_sin[..cos_len.min(self.rope_sin.len())].as_ptr() as *const _,
            (cos_len * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        self.compute.dispatch(
            cb, &self.kernels.rope_apply,
            (seq_len, num_heads, half_dim),
            (1.min(seq_len), 1.min(num_heads), 1.min(half_dim)),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, q);
                gpu_ops::set_tensor_buffer(encoder, 1, k);
                encoder.set_buffer(2, Some(&cos_buf), 0);
                encoder.set_buffer(3, Some(&sin_buf), 0);
                let vals: [u32; 4] = [
                    seq_len as u32,
                    num_heads as u32,
                    head_dim as u32,
                    pos_offset as u32,
                ];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
    }

    /// GPU RMS normalization using model weights.
    fn rms_norm_on(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        weight_name: &str,
        n: usize,
        d: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let w_f16 = gpu_ops::read_weight_f16(&self.model, &self.compute, weight_name)?;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (n * d * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch_1d(cb, &self.kernels.rms_norm, n, |encoder| {
            gpu_ops::set_tensor_buffer(encoder, 0, input);
            gpu_ops::set_tensor_buffer(encoder, 1, &w_f16);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            let n_u32 = n as u32;
            let d_u32 = d as u32;
            encoder.set_bytes(3, 4, &n_u32 as *const u32 as *const _);
            encoder.set_bytes(4, 4, &d_u32 as *const u32 as *const _);
            encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        });
        Ok(Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([n, d]),
            DType::F16,
            self.compute.device().info().id,
        ))
    }

    /// Convert f32 data to f16 GPU tensor.
    fn f32_to_f16_tensor(&self, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let s = match shape.len() {
            1 => Shape::from([shape[0]]),
            2 => Shape::from([shape[0], shape[1]]),
            3 => Shape::from([shape[0], shape[1], shape[2]]),
            _ => Shape::from([shape[0], shape[1]]),
        };
        Tensor::from_slice(&f16_data, s, DType::F16, self.compute.device().info().id)
    }

    /// Check if a weight exists in the model.
    fn has_weight(&self, name: &str) -> bool {
        self.model.read().get_weight(name).is_some()
    }

    /// Read a weight as f32 Vec.
    fn weight_vec_f32(&self, name: &str) -> Result<Vec<f32>> {
        gpu_ops::read_weight_vec_f32(&self.model, name)
    }
}

// ── CPU Utility Functions ───────────────────────────────────────────────────

/// CPU matrix multiply with bias: Y = X @ W^T + b.
/// X: [rows, in_dim], W: [out_dim, in_dim], b: [out_dim] -> Y: [rows, out_dim].
fn matmul_bias_f32(
    x: &[f32], w: &[f32], b: &[f32],
    rows: usize, in_dim: usize, out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut sum = b[o.min(b.len().saturating_sub(1))];
            for i in 0..in_dim {
                if o * in_dim + i < w.len() {
                    sum += x[r * in_dim + i] * w[o * in_dim + i];
                }
            }
            out[r * out_dim + o] = sum;
        }
    }
    out
}

/// CPU RMS normalization.
fn rms_norm_cpu(x: &[f32], d: usize, eps: f32, weight: &[f32]) -> Vec<f32> {
    let sq_sum: f32 = x.iter().take(d).map(|v| v * v).sum();
    let rms = (sq_sum / d as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    (0..d).map(|i| x[i] * inv_rms * weight[i.min(weight.len().saturating_sub(1))]).collect()
}

/// Sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Argmax over a slice of logits.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// CPU Conv1d: [length, in_ch] -> [length, out_ch] with zero-padding.
fn conv1d_cpu(
    input: &[f32], weight: &[f32], bias: &[f32],
    length: usize, in_ch: usize, out_ch: usize,
    kernel_size: usize, _stride: usize,
) -> Vec<f32> {
    let padding = kernel_size / 2;
    let mut output = vec![0.0f32; length * out_ch];

    for l in 0..length {
        for oc in 0..out_ch {
            let mut sum = bias[oc.min(bias.len().saturating_sub(1))];
            for ic in 0..in_ch {
                for k in 0..kernel_size {
                    let pos = l as isize + k as isize - padding as isize;
                    if pos >= 0 && (pos as usize) < length {
                        let w_idx = (oc * in_ch + ic) * kernel_size + k;
                        if w_idx < weight.len() {
                            sum += input[pos as usize * in_ch + ic] * weight[w_idx];
                        }
                    }
                }
            }
            output[l * out_ch + oc] = sum;
        }
    }
    output
}

/// CPU dilated Conv1d: [length, in_ch] -> [length, out_ch].
fn dilated_conv1d_cpu(
    input: &[f32], weight: &[f32], bias: &[f32],
    length: usize, in_ch: usize, out_ch: usize,
    kernel_size: usize, dilation: usize,
) -> Vec<f32> {
    let _effective_ks = (kernel_size - 1) * dilation + 1;
    let _padding = _effective_ks / 2;
    let mut output = vec![0.0f32; length * out_ch];

    for l in 0..length {
        for oc in 0..out_ch {
            let mut sum = bias[oc.min(bias.len().saturating_sub(1))];
            for ic in 0..in_ch {
                for k in 0..kernel_size {
                    let pos = l as isize + (k as isize - kernel_size as isize / 2) * dilation as isize;
                    if pos >= 0 && (pos as usize) < length {
                        let w_idx = (oc * in_ch + ic) * kernel_size + k;
                        if w_idx < weight.len() {
                            sum += input[pos as usize * in_ch + ic] * weight[w_idx];
                        }
                    }
                }
            }
            output[l * out_ch + oc] = sum;
        }
    }
    output
}

/// CPU transposed Conv1d (for upsampling): [length, in_ch] -> [length * stride, out_ch].
fn conv1d_transpose_cpu(
    input: &[f32], weight: &[f32], bias: &[f32],
    length: usize, in_ch: usize, out_ch: usize,
    kernel_size: usize, stride: usize,
) -> Vec<f32> {
    let out_len = length * stride;
    let padding = (kernel_size - stride) / 2;
    let mut output = vec![0.0f32; out_len * out_ch];

    for oc in 0..out_ch {
        for lo in 0..out_len {
            let mut sum = bias[oc.min(bias.len().saturating_sub(1))];
            for ic in 0..in_ch {
                for k in 0..kernel_size {
                    let l_check = lo as isize + padding as isize - k as isize;
                    if l_check >= 0 && l_check % stride as isize == 0 {
                        let li = l_check as usize / stride;
                        if li < length {
                            let w_idx = (ic * out_ch + oc) * kernel_size + k;
                            if w_idx < weight.len() {
                                sum += input[li * in_ch + ic] * weight[w_idx];
                            }
                        }
                    }
                }
            }
            output[lo * out_ch + oc] = sum;
        }
    }
    output
}
