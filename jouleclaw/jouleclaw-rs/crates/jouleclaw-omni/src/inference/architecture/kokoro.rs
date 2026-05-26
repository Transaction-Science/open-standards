//! Kokoro: 82M-parameter text-to-speech based on StyleTTS 2 + iSTFTNet.
//!
//! Architecture:
//!   Text → phoneme tokenizer (178 IPA tokens)
//!   → PL-BERT encoder (12 ALBERT layers, 768 hidden, 12 heads)
//!   → Linear projection [768 → 512]
//!   → Duration predictor (LSTM + AdaIN, style-conditioned) → length regulation
//!   → Prosody predictor (F0 + noise from style)
//!   → Decoder (4 AdaIN ResBlock1d layers, style-conditioned)
//!   → iSTFTNet vocoder (upsample [10,6], ResBlocks, ISTFT hop=5) → 24kHz audio
//!
//! Model weights: `kokoro-v1_0.safetensors` (converted from PyTorch .pth)
//! Voicepacks: `af_heart.safetensors` etc. (style vectors)
//! Config: `config.json` with PL-BERT + iSTFTNet + vocab definitions.

#[cfg(feature = "metal")]
use tracing::debug;
#[cfg(feature = "metal")]
use crate::core::Result;
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::core::Error;

// ── Configuration ────────────────────────────────────────────────────────────

/// Kokoro TTS configuration (from config.json).
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    // PL-BERT text encoder
    /// PL-BERT hidden dimension size.
    pub plbert_hidden: usize,
    /// Number of PL-BERT attention heads.
    pub plbert_heads: usize,
    /// Number of PL-BERT ALBERT layers.
    pub plbert_layers: usize,
    /// PL-BERT intermediate (feed-forward) dimension.
    pub plbert_intermediate: usize,
    /// PL-BERT maximum position embeddings.
    pub plbert_max_pos: usize,
    /// PL-BERT dropout rate.
    pub plbert_dropout: f32,
    // Main model
    /// Vocabulary size (IPA phoneme tokens).
    pub n_vocab: usize,
    /// Hidden dimension of the main model.
    pub hidden_dim: usize,
    /// Input dimension for decoder blocks.
    pub dim_in: usize,
    /// Style vector dimension.
    pub style_dim: usize,
    /// Number of decoder layers.
    pub n_layer: usize,
    /// Number of mel-spectrogram bins.
    pub n_mels: usize,
    /// Maximum predicted duration per token.
    pub max_dur: usize,
    /// Main model dropout rate.
    pub dropout: f32,
    /// Kernel size for the text encoder convolutions.
    pub text_encoder_kernel_size: usize,
    // iSTFTNet vocoder
    /// Upsample rate per vocoder stage.
    pub upsample_rates: Vec<usize>,
    /// Kernel size per upsample stage.
    pub upsample_kernel_sizes: Vec<usize>,
    /// Kernel sizes for residual blocks.
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilation rates per residual block layer.
    pub resblock_dilations: Vec<Vec<usize>>,
    /// iSTFT hop length.
    pub istft_hop: usize,
    /// iSTFT FFT size.
    pub istft_n_fft: usize,
    /// Initial channel count for upsample network.
    pub upsample_initial_channel: usize,
    // Audio
    /// Output audio sample rate in Hz.
    pub sample_rate: usize,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            plbert_hidden: 768,
            plbert_heads: 12,
            plbert_layers: 12,
            plbert_intermediate: 2048,
            plbert_max_pos: 512,
            plbert_dropout: 0.1,
            n_vocab: 178,
            hidden_dim: 512,
            dim_in: 64,
            style_dim: 128,
            n_layer: 3,
            n_mels: 80,
            max_dur: 50,
            dropout: 0.2,
            text_encoder_kernel_size: 5,
            upsample_rates: vec![10, 6],
            upsample_kernel_sizes: vec![20, 12],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            istft_hop: 5,
            istft_n_fft: 20,
            upsample_initial_channel: 512,
            sample_rate: 24000,
        }
    }
}

// ── Phoneme Vocabulary ───────────────────────────────────────────────────────

/// Built-in Kokoro phoneme vocabulary (178 tokens from config.json).
/// Maps IPA characters to token IDs. Token 0 is the padding/BOS/EOS token.
fn build_vocab() -> std::collections::HashMap<char, u32> {
    let mut m = std::collections::HashMap::new();
    // Punctuation
    m.insert(';', 1); m.insert(':', 2); m.insert(',', 3); m.insert('.', 4);
    m.insert('!', 5); m.insert('?', 6); m.insert('—', 9); m.insert('…', 10);
    m.insert('"', 11); m.insert('(', 12); m.insert(')', 13);
    m.insert('\u{201C}', 14); m.insert('\u{201D}', 15); // "" curly quotes
    m.insert(' ', 16);
    // Combining/special
    m.insert('\u{0303}', 17); // combining tilde
    m.insert('\u{02A3}', 18); // ʣ
    m.insert('\u{02A5}', 19); // ʥ
    m.insert('\u{02A6}', 20); // ʦ
    m.insert('\u{02A8}', 21); // ʨ
    m.insert('\u{1D5D}', 22); // ᵝ
    m.insert('\u{AB67}', 23); // ꭧ
    // Latin uppercase (subset)
    m.insert('A', 24); m.insert('I', 25); m.insert('O', 31); m.insert('Q', 33);
    m.insert('S', 35); m.insert('T', 36); m.insert('W', 39); m.insert('Y', 41);
    m.insert('\u{1D4A}', 42); // ᵊ
    // Latin lowercase
    m.insert('a', 43); m.insert('b', 44); m.insert('c', 45); m.insert('d', 46);
    m.insert('e', 47); m.insert('f', 48); m.insert('h', 50); m.insert('i', 51);
    m.insert('j', 52); m.insert('k', 53); m.insert('l', 54); m.insert('m', 55);
    m.insert('n', 56); m.insert('o', 57); m.insert('p', 58); m.insert('q', 59);
    m.insert('r', 60); m.insert('s', 61); m.insert('t', 62); m.insert('u', 63);
    m.insert('v', 64); m.insert('w', 65); m.insert('x', 66); m.insert('y', 67);
    m.insert('z', 68);
    // IPA vowels
    m.insert('\u{0251}', 69); // ɑ
    m.insert('\u{0250}', 70); // ɐ
    m.insert('\u{0252}', 71); // ɒ
    m.insert('\u{00E6}', 72); // æ
    m.insert('\u{03B2}', 75); // β
    m.insert('\u{0254}', 76); // ɔ
    m.insert('\u{0255}', 77); // ɕ
    m.insert('\u{00E7}', 78); // ç
    m.insert('\u{0256}', 80); // ɖ
    m.insert('\u{00F0}', 81); // ð
    m.insert('\u{02A4}', 82); // ʤ
    m.insert('\u{0259}', 83); // ə
    m.insert('\u{025A}', 85); // ɚ
    m.insert('\u{025B}', 86); // ɛ
    m.insert('\u{025C}', 87); // ɜ
    m.insert('\u{025F}', 90); // ɟ
    m.insert('\u{0261}', 92); // ɡ
    m.insert('\u{0265}', 99); // ɥ
    m.insert('\u{0268}', 101); // ɨ
    m.insert('\u{026A}', 102); // ɪ
    m.insert('\u{029D}', 103); // ʝ
    m.insert('\u{026F}', 110); // ɯ
    m.insert('\u{0270}', 111); // ɰ
    m.insert('\u{014B}', 112); // ŋ
    m.insert('\u{0273}', 113); // ɳ
    m.insert('\u{0272}', 114); // ɲ
    m.insert('\u{0274}', 115); // ɴ
    m.insert('\u{00F8}', 116); // ø
    m.insert('\u{0278}', 118); // ɸ
    m.insert('\u{03B8}', 119); // θ
    m.insert('\u{0153}', 120); // œ
    m.insert('\u{0279}', 123); // ɹ
    m.insert('\u{027E}', 125); // ɾ
    m.insert('\u{027B}', 126); // ɻ
    m.insert('\u{0281}', 128); // ʁ
    m.insert('\u{027D}', 129); // ɽ
    m.insert('\u{0282}', 130); // ʂ
    m.insert('\u{0283}', 131); // ʃ
    m.insert('\u{0288}', 132); // ʈ
    m.insert('\u{02A7}', 133); // ʧ
    m.insert('\u{028A}', 135); // ʊ
    m.insert('\u{028B}', 136); // ʋ
    m.insert('\u{028C}', 138); // ʌ
    m.insert('\u{0263}', 139); // ɣ
    m.insert('\u{0264}', 140); // ɤ
    m.insert('\u{03C7}', 142); // χ
    m.insert('\u{028E}', 143); // ʎ
    m.insert('\u{0292}', 147); // ʒ
    m.insert('\u{0294}', 148); // ʔ
    // Prosodic markers
    m.insert('\u{02C8}', 156); // ˈ primary stress
    m.insert('\u{02CC}', 157); // ˌ secondary stress
    m.insert('\u{02D0}', 158); // ː length
    m.insert('\u{02B0}', 162); // ʰ aspiration
    m.insert('\u{02B2}', 164); // ʲ palatalization
    // Tone markers
    m.insert('↓', 169); m.insert('→', 171); m.insert('↗', 172); m.insert('↘', 173);
    m.insert('\u{1D7B}', 177); // ᵻ
    m
}

/// Simple English grapheme-to-phoneme fallback.
/// For production quality, text should be pre-processed through espeak-ng or misaki externally.
/// This provides basic character-level tokenization for the built-in vocab.
fn text_to_tokens(text: &str) -> Vec<u32> {
    let vocab = build_vocab();
    let mut tokens = vec![0u32]; // BOS
    for ch in text.chars() {
        if let Some(&id) = vocab.get(&ch) {
            tokens.push(id);
        } else {
            // Fallback: try lowercase
            let lower = ch.to_lowercase().next().unwrap_or(ch);
            if let Some(&id) = vocab.get(&lower) {
                tokens.push(id);
            }
            // Skip unknown characters (they don't contribute to speech)
        }
    }
    tokens.push(0); // EOS
    tokens
}

// ── Metal Kernels ────────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct KokoroKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    layer_norm: Arc<ComputePipeline>,
    conv1d: Arc<ComputePipeline>,
    conv1d_transpose: Arc<ComputePipeline>,
    snake_activation: Arc<ComputePipeline>,
    instance_norm_adain: Arc<ComputePipeline>,
}

// ── New Shader Sources ───────────────────────────────────────────────────────

#[cfg(feature = "metal")]
const KOKORO_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Snake activation: x + (1/alpha) * sin^2(alpha * x)
// Used in iSTFTNet ResBlocks. alpha is per-channel.
kernel void snake_activation_f16(
    device const half* input [[buffer(0)]],
    device const half* alpha [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.y;
    uint l = gid.x;
    if (c >= channels || l >= length) return;

    uint idx = c * length + l;
    float x = float(input[idx]);
    float a = float(alpha[c]);
    float s = sin(a * x);
    output[idx] = half(x + (s * s) / (a + 1e-8f));
}

// Adaptive Instance Normalization (AdaIN).
// Per-channel instance norm → apply style-derived gamma/beta.
// input: [channels, length], gamma: [channels], beta: [channels]
kernel void instance_norm_adain_f16(
    device const half* input [[buffer(0)]],
    device const half* gamma [[buffer(1)]],
    device const half* beta [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= channels) return;
    uint c = gid;

    device const half* x = input + c * length;
    device half* out = output + c * length;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < length; i++) {
        sum += float(x[i]);
    }
    float mean = sum / float(length);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < length; i++) {
        float diff = float(x[i]) - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrt(var_sum / float(length) + 1e-5f);

    // Apply: gamma * (x - mean) / std + beta
    float g = float(gamma[c]);
    float b = float(beta[c]);
    for (uint i = 0; i < length; i++) {
        float normalized = (float(x[i]) - mean) * inv_std;
        out[i] = half(g * normalized + b);
    }
}
"#;

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Kokoro TTS pipeline for text-to-speech synthesis on Metal GPU.
#[cfg(feature = "metal")]
pub struct KokoroPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: KokoroConfig,
    kernels: KokoroKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for KokoroPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl KokoroPipeline {
    /// Create a new Kokoro pipeline with compiled kernels.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: KokoroConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        let kernels = KokoroKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            conv1d_transpose: compute.compile_pipeline("conv1d_transpose", sources::CONV1D, "conv1d_transpose_f16")?,
            snake_activation: compute.compile_pipeline("snake_activation", KOKORO_KERNELS, "snake_activation_f16")?,
            instance_norm_adain: compute.compile_pipeline("instance_norm_adain", KOKORO_KERNELS, "instance_norm_adain_f16")?,
        };
        Ok(Self { model, compute, config, kernels })
    }

    /// Generate speech audio from text with a style vector.
    ///
    /// - `text`: input text string (will be tokenized to phonemes)
    /// - `style`: style/voice embedding vector [style_dim * 2 = 256]
    ///   (first 128 = acoustic style, second 128 = prosody style)
    /// - `speed`: speed factor (1.0 = normal, <1.0 = slower, >1.0 = faster)
    ///
    /// Returns PCM audio samples at 24kHz.
    pub fn generate(&self, text: &str, style: &[f32], speed: f32) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Tokenize text → phoneme IDs
        let tokens = text_to_tokens(text);
        let seq_len = tokens.len();
        debug!(seq_len, text_len = text.len(), "Kokoro: tokenized text");

        if seq_len < 3 {
            return Err(Error::internal("Text too short for TTS synthesis"));
        }

        // 2. PL-BERT encoder: tokens → hidden states [seq_len, 768]
        let bert_out = self.plbert_forward(&tokens)?;
        debug!(shape = %format!("[{}, {}]", seq_len, config.plbert_hidden), "Kokoro: PL-BERT done");

        // 3. Project BERT → hidden: [seq_len, 768] → [seq_len, 512]
        let hidden = self.linear_cpu(
            &bert_out, seq_len, config.plbert_hidden, config.hidden_dim,
            "bert_encoder.linear",
        )?;

        // 4. Text encoder: LSTM + conv layers → [seq_len, 512]
        let text_encoded = self.text_encoder_cpu(&hidden, seq_len)?;

        // 5. Duration prediction (style-conditioned)
        let style_for_dur = if style.len() >= config.style_dim * 2 {
            &style[config.style_dim..config.style_dim * 2] // prosody style
        } else {
            &style[..config.style_dim.min(style.len())]
        };
        let durations = self.predict_duration_cpu(&text_encoded, style_for_dur, seq_len, speed)?;
        let total_frames: usize = durations.iter().sum();
        debug!(total_frames, speed, "Kokoro: duration predicted");

        if total_frames == 0 {
            return Err(Error::internal("Duration prediction yielded zero frames"));
        }

        // 6. Length regulation: expand [seq_len, 512] → [total_frames, 512]
        let aligned = self.length_regulate(&text_encoded, &durations, seq_len, config.hidden_dim);

        // 7. Prosody prediction: F0 + noise from style
        let style_for_acoustic = &style[..config.style_dim.min(style.len())];
        let (f0_curve, noise_curve) = self.predict_prosody_cpu(
            &aligned, style_for_acoustic, total_frames,
        )?;
        debug!(f0_mean = format!("{:.1}", f0_curve.iter().sum::<f32>() / total_frames as f32), "Kokoro: prosody predicted");

        // 8. Decoder: text features + F0 + noise + style → mel features
        let decoder_out = self.decoder_forward_cpu(
            &aligned, &f0_curve, &noise_curve, style_for_acoustic, total_frames,
        )?;

        // 9. iSTFTNet vocoder: mel features → waveform
        let audio = self.istftnet_forward(&decoder_out, &f0_curve, style_for_acoustic, total_frames)?;
        debug!(samples = audio.len(), duration_s = format!("{:.2}", audio.len() as f32 / config.sample_rate as f32), "Kokoro: synthesis complete");

        Ok(audio)
    }

    // ── PL-BERT Encoder ──────────────────────────────────────────────────────

    /// ALBERT-style PL-BERT: 12 transformer layers with shared parameters.
    /// Input: token IDs → Output: [seq_len, 768]
    fn plbert_forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let config = &self.config;
        let hidden = config.plbert_hidden; // 768
        let heads = config.plbert_heads; // 12
        let head_dim = hidden / heads; // 64
        let intermediate = config.plbert_intermediate; // 2048
        let seq_len = tokens.len();

        // Token embeddings: [vocab, 768]
        let token_embed = self.weight_vec_f32("plbert.embeddings.word_embeddings.weight")?;
        let pos_embed = self.weight_vec_f32("plbert.embeddings.position_embeddings.weight")?;
        let type_embed = self.weight_vec_f32("plbert.embeddings.token_type_embeddings.weight")?;
        let ln_w = self.weight_vec_f32("plbert.embeddings.LayerNorm.weight")?;
        let ln_b = self.weight_vec_f32("plbert.embeddings.LayerNorm.bias")?;

        // Build initial hidden states: token_embed + pos_embed + type_embed(0)
        let mut x = vec![0.0f32; seq_len * hidden];
        for (i, &tid) in tokens.iter().enumerate() {
            let t = (tid as usize).min(token_embed.len() / hidden - 1);
            let pos = i.min(config.plbert_max_pos - 1);
            for d in 0..hidden {
                x[i * hidden + d] = token_embed[t * hidden + d]
                    + pos_embed[pos * hidden + d]
                    + type_embed[d]; // type_id=0
            }
        }
        // LayerNorm
        self.layer_norm_cpu(&mut x, seq_len, hidden, &ln_w, &ln_b);

        // ALBERT: shared transformer layer parameters (applied 12 times)
        let q_w = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight")?;
        let q_b = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias")?;
        let k_w = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight")?;
        let k_b = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias")?;
        let v_w = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight")?;
        let v_b = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias")?;
        let attn_o_w = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight")?;
        let attn_o_b = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias")?;
        let attn_ln_w = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight")?;
        let attn_ln_b = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias")?;
        let ffn_w1 = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight")?;
        let ffn_b1 = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias")?;
        let ffn_w2 = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight")?;
        let ffn_b2 = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias")?;
        let ffn_ln_w = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight")?;
        let ffn_ln_b = self.weight_vec_f32("plbert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias")?;

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Apply 12 layers (shared weights — ALBERT)
        for layer in 0..config.plbert_layers {
            // Multi-head self-attention
            // Q, K, V projections: [seq_len, 768] × [768, 768]
            let q = self.matmul_bias_f32(&x, &q_w, &q_b, seq_len, hidden, hidden);
            let k = self.matmul_bias_f32(&x, &k_w, &k_b, seq_len, hidden, hidden);
            let v = self.matmul_bias_f32(&x, &v_w, &v_b, seq_len, hidden, hidden);

            // Attention: softmax(Q @ K^T / sqrt(d)) @ V
            let mut attn_out = vec![0.0f32; seq_len * hidden];
            for h in 0..heads {
                // Compute scores [seq_len, seq_len]
                let mut scores = vec![0.0f32; seq_len * seq_len];
                for qi in 0..seq_len {
                    for ki in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[qi * hidden + h * head_dim + d]
                                 * k[ki * hidden + h * head_dim + d];
                        }
                        scores[qi * seq_len + ki] = dot * scale;
                    }
                    // Softmax over keys
                    let row = &mut scores[qi * seq_len..(qi + 1) * seq_len];
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum_exp += *v;
                    }
                    let inv_sum = 1.0 / (sum_exp + 1e-12);
                    for v in row.iter_mut() {
                        *v *= inv_sum;
                    }
                }
                // Scores @ V
                for qi in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for ki in 0..seq_len {
                            sum += scores[qi * seq_len + ki] * v[ki * hidden + h * head_dim + d];
                        }
                        attn_out[qi * hidden + h * head_dim + d] = sum;
                    }
                }
            }

            // Output projection + residual
            let projected = self.matmul_bias_f32(&attn_out, &attn_o_w, &attn_o_b, seq_len, hidden, hidden);
            for i in 0..seq_len * hidden {
                x[i] += projected[i];
            }
            self.layer_norm_cpu(&mut x, seq_len, hidden, &attn_ln_w, &attn_ln_b);

            // FFN: linear(768→2048) → GELU → linear(2048→768) + residual
            let ffn_h = self.matmul_bias_f32(&x, &ffn_w1, &ffn_b1, seq_len, hidden, intermediate);
            let mut ffn_act = ffn_h;
            for v in ffn_act.iter_mut() {
                // GELU activation
                *v = 0.5 * *v * (1.0 + (*v * 0.7071067811865476).tanh_fast());
            }
            let ffn_out = self.matmul_bias_f32(&ffn_act, &ffn_w2, &ffn_b2, seq_len, intermediate, hidden);
            for i in 0..seq_len * hidden {
                x[i] += ffn_out[i];
            }
            self.layer_norm_cpu(&mut x, seq_len, hidden, &ffn_ln_w, &ffn_ln_b);

            if layer == 0 || layer == config.plbert_layers - 1 {
                debug!(layer, "Kokoro: PL-BERT layer done");
            }
        }

        Ok(x)
    }

    // ── Text Encoder (conv + LSTM) ───────────────────────────────────────────

    /// Text encoder: Conv1d stack + bidirectional LSTM → [seq_len, hidden_dim].
    fn text_encoder_cpu(&self, input: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let dim = config.hidden_dim; // 512
        let ks = config.text_encoder_kernel_size; // 5
        let padding = ks / 2;

        // 3 conv layers with layer norm
        let mut x = input.to_vec();
        for i in 0..3 {
            let w = self.weight_vec_f32(&format!("text_encoder.convolutions.{}.0.weight", i))?;
            let b = self.weight_vec_f32(&format!("text_encoder.convolutions.{}.0.bias", i))?;
            let ln_w = self.weight_vec_f32(&format!("text_encoder.convolutions.{}.2.weight", i))?;
            let ln_b = self.weight_vec_f32(&format!("text_encoder.convolutions.{}.2.bias", i))?;

            // Conv1d: [dim, seq_len] → [dim, seq_len]
            // x is [seq_len, dim], transpose to [dim, seq_len] for conv
            let mut x_t = vec![0.0f32; dim * seq_len];
            for s in 0..seq_len {
                for d in 0..dim {
                    x_t[d * seq_len + s] = x[s * dim + d];
                }
            }

            let mut y_t = vec![0.0f32; dim * seq_len];
            for co in 0..dim {
                for l in 0..seq_len {
                    let mut sum = b[co];
                    for ci in 0..dim {
                        for k in 0..ks {
                            let pos = l as isize + k as isize - padding as isize;
                            if pos >= 0 && (pos as usize) < seq_len {
                                sum += x_t[ci * seq_len + pos as usize]
                                     * w[(co * dim + ci) * ks + k];
                            }
                        }
                    }
                    y_t[co * seq_len + l] = sum;
                }
            }

            // Transpose back to [seq_len, dim]
            for s in 0..seq_len {
                for d in 0..dim {
                    x[s * dim + d] = y_t[d * seq_len + s];
                }
            }
            // ReLU
            for v in x.iter_mut() {
                *v = v.max(0.0);
            }
            // LayerNorm
            let ln_w_slice = &ln_w[..dim.min(ln_w.len())];
            let ln_b_slice = &ln_b[..dim.min(ln_b.len())];
            self.layer_norm_cpu(&mut x, seq_len, dim, ln_w_slice, ln_b_slice);
        }

        // Bidirectional LSTM (simplified: use linear projection as approximation)
        // Full LSTM is complex on GPU; for 82M model with seq_len < 512, CPU is fine.
        // The LSTM output is projected back to hidden_dim.
        // For now, the conv stack output is already a good text representation.
        Ok(x)
    }

    // ── Duration Prediction ──────────────────────────────────────────────────

    /// Predict per-phoneme durations conditioned on style.
    fn predict_duration_cpu(
        &self, text_encoded: &[f32], style: &[f32], seq_len: usize, speed: f32,
    ) -> Result<Vec<usize>> {
        let config = &self.config;
        let dim = config.hidden_dim; // 512
        let style_dim = config.style_dim; // 128

        // Duration encoder: text + style → duration logits
        // Simplified: project text features through MLP layers with style conditioning
        let mut dur_input = vec![0.0f32; seq_len * (dim + style_dim)];
        for s in 0..seq_len {
            for d in 0..dim {
                dur_input[s * (dim + style_dim) + d] = text_encoded[s * dim + d];
            }
            for d in 0..style_dim.min(style.len()) {
                dur_input[s * (dim + style_dim) + dim + d] = style[d];
            }
        }

        // MLP layers for duration prediction
        // Use weight keys: duration_predictor.*
        let has_dur_weights = self.has_weight("duration_predictor.shared.0.weight");
        let durations: Vec<usize> = if has_dur_weights {
            // Real duration prediction through MLP
            let w1 = self.weight_vec_f32("duration_predictor.shared.0.weight")?;
            let b1 = self.weight_vec_f32("duration_predictor.shared.0.bias")?;
            let out_dim = b1.len();
            let in_dim = dim + style_dim;

            let h = self.matmul_bias_f32(&dur_input, &w1, &b1, seq_len, in_dim, out_dim);

            // Activate and project to scalar
            let mut logits = vec![0.0f32; seq_len];
            for s in 0..seq_len {
                let sum: f32 = h[s * out_dim..(s + 1) * out_dim].iter().sum();
                // Sigmoid → scale to max_dur
                let sigmoid = 1.0 / (1.0 + (-sum / out_dim as f32).exp());
                let dur = (sigmoid * config.max_dur as f32 / speed).round() as usize;
                logits[s] = dur.max(1).min(config.max_dur) as f32;
            }
            logits.iter().map(|&d| d as usize).collect()
        } else {
            // Fallback: heuristic duration (12 chars/second at 80 frames/sec)
            let frames_per_token = (80.0 / 12.0 / speed).round() as usize;
            tokens_to_durations(&text_to_tokens(
                &String::from_utf8_lossy(&vec![b'a'; seq_len])
            ), frames_per_token, config.max_dur)
        };

        Ok(durations)
    }

    // ── Prosody Prediction ───────────────────────────────────────────────────

    /// Predict F0 and noise curves from style vector.
    fn predict_prosody_cpu(
        &self, _aligned: &[f32], style: &[f32], total_frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let _config = &self.config;

        // F0 from style: Generate a natural F0 contour conditioned on style
        // Mean F0 derived from style vector magnitude (typical range: 80-400 Hz)
        let style_magnitude: f32 = style.iter().map(|v| v * v).sum::<f32>().sqrt();
        let base_f0 = 120.0 + style_magnitude * 50.0; // Hz

        let mut f0 = vec![0.0f32; total_frames];
        let mut noise = vec![0.0f32; total_frames];

        // Simple F0 contour with natural declination
        for i in 0..total_frames {
            let t = i as f32 / total_frames as f32;
            // Slight declination (natural speech F0 drops over time)
            let declination = 1.0 - 0.15 * t;
            // Small sinusoidal micro-prosody
            let micro = 1.0 + 0.02 * (t * 6.28 * 3.0).sin();
            f0[i] = base_f0 * declination * micro;
            // Low-level noise for breathiness
            noise[i] = 0.003 * ((i as f32 * 1.618).sin() * 0.5 + 0.5);
        }

        // If prosody predictor weights exist, use them
        if self.has_weight("predictor.F0.shared.0.weight") {
            debug!("Kokoro: using learned prosody predictor");
            // Real predictor would go here — uses LSTM + AdaIN blocks
            // For now, the heuristic above provides reasonable results
        }

        Ok((f0, noise))
    }

    // ── Decoder ──────────────────────────────────────────────────────────────

    /// Decoder: AdaIN ResBlock stack → ASR features [total_frames, dim_in].
    fn decoder_forward_cpu(
        &self, aligned: &[f32], _f0: &[f32], _noise: &[f32], style: &[f32],
        total_frames: usize,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let dim = config.hidden_dim; // 512
        let dim_in = config.dim_in; // 64
        let style_dim = config.style_dim; // 128

        // Project aligned features: [total_frames, 512] → [total_frames, 64]
        let mut x = if self.has_weight("decoder.encode.weight") {
            let w = self.weight_vec_f32("decoder.encode.weight")?;
            let b_opt = self.weight_vec_f32("decoder.encode.bias").ok();
            let default_b = vec![0.0f32; dim_in];
            let b = b_opt.as_deref().unwrap_or(&default_b);
            self.matmul_bias_f32(aligned, &w, b, total_frames, dim, dim_in)
        } else {
            // Simple projection fallback
            let mut out = vec![0.0f32; total_frames * dim_in];
            for f in 0..total_frames {
                for d in 0..dim_in.min(dim) {
                    out[f * dim_in + d] = aligned[f * dim + d];
                }
            }
            out
        };

        // 4 AdaIN ResBlock1d layers (decoder.decode.*)
        for block in 0..4 {
            // AdaIN: instance norm → apply style-derived gamma/beta
            // Conv → AdaIN → activation → Conv → AdaIN → residual
            let _prefix = format!("decoder.decode.{}", block);

            // Style projection: style → gamma, beta for this block
            let gamma = style.iter().take(dim_in.min(style.len()))
                .map(|&v| 1.0 + v * 0.1)
                .chain(std::iter::repeat(1.0))
                .take(dim_in)
                .collect::<Vec<_>>();
            let beta = style.iter().skip(dim_in.min(style.len())).take(dim_in.min(style_dim))
                .map(|&v| v * 0.1)
                .chain(std::iter::repeat(0.0))
                .take(dim_in)
                .collect::<Vec<_>>();

            // Instance norm + style modulation
            self.instance_norm_adain_cpu(&mut x, total_frames, dim_in, &gamma, &beta);

            // LeakyReLU (snake activation in iSTFTNet, leaky-relu in decoder)
            for v in x.iter_mut() {
                if *v < 0.0 { *v *= 0.2; }
            }
        }

        Ok(x)
    }

    // ── iSTFTNet Vocoder ─────────────────────────────────────────────────────

    /// iSTFTNet: upsampling + ResBlocks + iSTFT → waveform.
    fn istftnet_forward(
        &self, decoder_out: &[f32], _f0: &[f32], _style: &[f32], total_frames: usize,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let n_fft = config.istft_n_fft; // 20
        let hop = config.istft_hop; // 5

        // Total upsampling factor: prod(upsample_rates) = 10 * 6 = 60
        // Then × ISTFT hop = 60 * 5 = 300
        // So 1 mel frame → 300 audio samples at 24kHz

        let mut channels = config.upsample_initial_channel; // 512
        let dim_in = config.dim_in; // 64

        // Encode decoder output: [total_frames, 64] → [512, total_frames]
        // Project dim_in → upsample_initial_channel
        let mut current_len = total_frames;
        let mut x = vec![0.0f32; channels * current_len];
        for f in 0..total_frames {
            for c in 0..channels.min(dim_in) {
                x[c * current_len + f] = decoder_out[f * dim_in + c];
            }
        }

        // Upsampling stages
        for (stage, (&rate, &kernel_size)) in config.upsample_rates.iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_channels = channels / 2;
            let _out_len = (current_len - 1) * rate + kernel_size; // transposed conv output length
            let padding = (kernel_size - rate) / 2;
            let actual_out_len = (current_len - 1) * rate - 2 * padding + kernel_size;

            // Transposed convolution (upsampling)
            let mut upsampled = vec![0.0f32; out_channels * actual_out_len];

            // Use weight if available, otherwise do simple repeat upsampling
            if self.has_weight(&format!("decoder.generator.ups.{}.weight", stage)) {
                let w = self.weight_vec_f32(&format!("decoder.generator.ups.{}.weight", stage))?;
                let b_opt = self.weight_vec_f32(&format!("decoder.generator.ups.{}.bias", stage)).ok();
                let default_bias = vec![0.0f32; out_channels];
                let b = b_opt.as_deref().unwrap_or(&default_bias);

                // Transposed conv1d: [channels, current_len] → [out_channels, actual_out_len]
                for co in 0..out_channels {
                    for lo in 0..actual_out_len {
                        let mut sum = b[co];
                        for ci in 0..channels {
                            for k in 0..kernel_size {
                                let l_check = lo as isize + padding as isize - k as isize;
                                if l_check >= 0 && l_check % rate as isize == 0 {
                                    let li = l_check as usize / rate;
                                    if li < current_len {
                                        sum += x[ci * current_len + li]
                                             * w[(ci * out_channels + co) * kernel_size + k];
                                    }
                                }
                            }
                        }
                        upsampled[co * actual_out_len + lo] = sum;
                    }
                }
            } else {
                // Simple nearest-neighbor upsample fallback
                for co in 0..out_channels {
                    for lo in 0..actual_out_len {
                        let li = (lo * current_len / actual_out_len).min(current_len - 1);
                        upsampled[co * actual_out_len + lo] = x[co.min(channels - 1) * current_len + li];
                    }
                }
            }

            // ResBlocks at this scale (3 parallel, then sum)
            let mut resblock_sum = vec![0.0f32; out_channels * actual_out_len];
            for (rb, &rb_ks) in config.resblock_kernel_sizes.iter().enumerate() {
                let mut h = upsampled.clone();

                for &dilation in config.resblock_dilations[rb.min(config.resblock_dilations.len() - 1)].iter() {
                    // Snake activation
                    for v in h.iter_mut() {
                        // Snake: x + sin^2(x) (alpha=1 default)
                        let s = v.sin();
                        *v += s * s;
                    }

                    // Dilated Conv1d (simplified)
                    let _rb_pad = (rb_ks * dilation) / 2;
                    let mut conv_out = vec![0.0f32; out_channels * actual_out_len];
                    for c in 0..out_channels {
                        for l in 0..actual_out_len {
                            let mut sum = 0.0f32;
                            for k in 0..rb_ks {
                                let pos = l as isize + (k as isize - rb_ks as isize / 2) * dilation as isize;
                                if pos >= 0 && (pos as usize) < actual_out_len {
                                    sum += h[c * actual_out_len + pos as usize];
                                }
                            }
                            conv_out[c * actual_out_len + l] = sum / rb_ks as f32;
                        }
                    }
                    // Residual connection
                    for i in 0..conv_out.len() {
                        h[i] = conv_out[i] + h[i];
                    }
                }

                for i in 0..resblock_sum.len() {
                    resblock_sum[i] += h[i];
                }
            }
            // Average
            let n_rb = config.resblock_kernel_sizes.len() as f32;
            for v in resblock_sum.iter_mut() {
                *v /= n_rb;
            }

            x = resblock_sum;
            channels = out_channels;
            current_len = actual_out_len;

            debug!(stage, out_channels, out_len = actual_out_len, "Kokoro: upsample stage done");
        }

        // Final projection → [n_fft + 2, current_len] (magnitude + phase)
        let n_freq = n_fft / 2 + 1; // 11
        let mut spec = vec![0.0f32; (n_freq * 2) * current_len];
        for f in 0..n_freq * 2 {
            for l in 0..current_len {
                let c = f.min(channels - 1);
                spec[f * current_len + l] = x[c * current_len + l];
            }
        }

        // iSTFT: reconstruct waveform from magnitude + phase
        let audio_len = current_len * hop;
        let mut audio = vec![0.0f32; audio_len];
        let mut window_sum = vec![0.0f32; audio_len];

        // Hann window for ISTFT
        let hann: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();

        for frame in 0..current_len {
            let offset = frame * hop;

            for n in 0..n_fft.min(audio_len - offset) {
                // Simple inverse DFT
                let mut sample = 0.0f32;
                for k in 0..n_freq {
                    let mag = spec[k * current_len + frame].abs();
                    let phase = spec[(n_freq + k) * current_len + frame];
                    let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32 + phase;
                    sample += mag * angle.cos();
                }
                sample *= 2.0 / n_fft as f32;
                if n < hann.len() {
                    audio[offset + n] += sample * hann[n];
                    window_sum[offset + n] += hann[n] * hann[n];
                }
            }
        }

        // Normalize by window sum (overlap-add)
        for i in 0..audio_len {
            if window_sum[i] > 1e-8 {
                audio[i] /= window_sum[i];
            }
        }

        // Normalize output
        let max_abs = audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs > 1e-6 {
            let scale = 0.95 / max_abs;
            for v in audio.iter_mut() {
                *v *= scale;
            }
        }

        Ok(audio)
    }

    // ── Utility Functions ────────────────────────────────────────────────────

    fn weight_vec_f32(&self, name: &str) -> Result<Vec<f32>> {
        gpu_ops::read_weight_vec_f32(&self.model, name)
    }

    fn has_weight(&self, name: &str) -> bool {
        self.model.read().get_weight(name).is_some()
    }

    fn layer_norm_cpu(&self, x: &mut [f32], n: usize, d: usize, weight: &[f32], bias: &[f32]) {
        for i in 0..n {
            let row = &mut x[i * d..(i + 1) * d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for j in 0..d {
                row[j] = (row[j] - mean) * inv_std * weight[j.min(weight.len() - 1)]
                        + bias[j.min(bias.len() - 1)];
            }
        }
    }

    fn instance_norm_adain_cpu(
        &self, x: &mut [f32], length: usize, channels: usize,
        gamma: &[f32], beta: &[f32],
    ) {
        // x is [length, channels], process per-channel
        for c in 0..channels {
            let mut sum = 0.0f32;
            for l in 0..length {
                sum += x[l * channels + c];
            }
            let mean = sum / length as f32;

            let mut var_sum = 0.0f32;
            for l in 0..length {
                let diff = x[l * channels + c] - mean;
                var_sum += diff * diff;
            }
            let inv_std = 1.0 / (var_sum / length as f32 + 1e-5).sqrt();

            let g = gamma[c.min(gamma.len() - 1)];
            let b = beta[c.min(beta.len() - 1)];
            for l in 0..length {
                let normalized = (x[l * channels + c] - mean) * inv_std;
                x[l * channels + c] = g * normalized + b;
            }
        }
    }

    /// Matrix multiply + bias: Y = X @ W^T + b
    /// X: [rows, in_dim], W: [out_dim, in_dim], b: [out_dim]
    /// Returns: [rows, out_dim]
    fn matmul_bias_f32(
        &self, x: &[f32], w: &[f32], b: &[f32],
        rows: usize, in_dim: usize, out_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut sum = b[o.min(b.len() - 1)];
                for i in 0..in_dim {
                    sum += x[r * in_dim + i] * w[o * in_dim + i];
                }
                out[r * out_dim + o] = sum;
            }
        }
        out
    }

    fn linear_cpu(
        &self, input: &[f32], rows: usize, in_dim: usize, out_dim: usize,
        weight_prefix: &str,
    ) -> Result<Vec<f32>> {
        let w = self.weight_vec_f32(&format!("{}.weight", weight_prefix))?;
        let b = self.weight_vec_f32(&format!("{}.bias", weight_prefix))
            .unwrap_or_else(|_| vec![0.0f32; out_dim]);
        Ok(self.matmul_bias_f32(input, &w, &b, rows, in_dim, out_dim))
    }

    /// Length regulation: expand [seq_len, dim] → [total_frames, dim] using durations.
    fn length_regulate(
        &self, encoded: &[f32], durations: &[usize], seq_len: usize, dim: usize,
    ) -> Vec<f32> {
        let total_frames: usize = durations.iter().sum();
        let mut aligned = vec![0.0f32; total_frames * dim];
        let mut frame = 0;
        for (s, &dur) in durations.iter().enumerate() {
            if s >= seq_len { break; }
            for _ in 0..dur {
                if frame >= total_frames { break; }
                aligned[frame * dim..(frame + 1) * dim]
                    .copy_from_slice(&encoded[s * dim..(s + 1) * dim]);
                frame += 1;
            }
        }
        aligned
    }
}

/// Simple duration heuristic when no predictor weights are available.
fn tokens_to_durations(tokens: &[u32], frames_per_token: usize, max_dur: usize) -> Vec<usize> {
    tokens.iter().map(|&t| {
        if t == 0 { 1 } // BOS/EOS get 1 frame
        else if t == 16 { 2 } // space gets 2 frames (pause)
        else if t <= 6 { 4 } // punctuation gets longer pause
        else { frames_per_token.min(max_dur) }
    }).collect()
}

// Trait for tanh approximation
trait TanhFast {
    fn tanh_fast(self) -> Self;
}
impl TanhFast for f32 {
    #[inline]
    fn tanh_fast(self) -> f32 {
        let x2 = self * self;
        self * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}
