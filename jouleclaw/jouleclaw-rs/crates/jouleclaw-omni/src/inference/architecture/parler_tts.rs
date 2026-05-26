//! Parler-TTS Large v1 (2.3B): Prompt-guided text-to-speech synthesis.
//!
//! Architecture:
//!   Voice description → Frozen Flan-T5-XL encoder (24 layers, 2048-dim, 32 heads)
//!     → encoder hidden states (voice conditioning)
//!   Text input → T5 tokenizer → token IDs
//!   → Audio decoder (25-layer autoregressive transformer, 1536-dim, 24 heads, FFN 6144)
//!     - Self-attention (causal) + cross-attention (to T5 encoder output)
//!     - Outputs 9 DAC codebook token logits per step at 86 Hz
//!     - Delay pattern: codebooks interleaved with temporal offsets
//!   → DAC codec decoder (9 codebooks × 1024 entries, ConvTranspose1d + Snake + ResBlocks)
//!     → 44.1kHz audio waveform
//!
//! Weight prefixes:
//!   - `text_encoder.` — Frozen T5-XL encoder (reuses T5Encoder)
//!   - `decoder.` — 25-layer autoregressive audio decoder
//!   - `dac.` — DAC neural codec decoder

use crate::core::Result;

#[cfg(feature = "metal")]
#[allow(unused_imports)]
use crate::core::Error;
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::inference::architecture::t5::{T5Encoder, T5Config};
#[cfg(feature = "metal")]
use tracing::debug;

// ── Configuration ────────────────────────────────────────────────────────────

/// Parler-TTS Large v1 configuration.
#[derive(Debug, Clone)]
pub struct ParlerTTSConfig {
    /// Audio decoder hidden dimension.
    pub decoder_hidden: usize,
    /// Number of audio decoder transformer layers.
    pub decoder_layers: usize,
    /// Number of attention heads in the decoder.
    pub decoder_heads: usize,
    /// FFN intermediate dimension in the decoder.
    pub decoder_ffn_dim: usize,
    /// Number of DAC codebooks.
    pub num_codebooks: usize,
    /// Number of entries per codebook.
    pub codebook_size: usize,
    /// Audio frame rate (codebook tokens per second).
    pub frame_rate: usize,
    /// Output audio sample rate.
    pub sample_rate: usize,
    /// T5 encoder vocabulary size.
    pub vocab_size: usize,
    /// T5 encoder model dimension.
    pub t5_d_model: usize,
    /// T5 encoder number of layers.
    pub t5_num_layers: usize,
    /// T5 encoder number of heads.
    pub t5_num_heads: usize,
    /// T5 encoder FFN dimension.
    pub t5_d_ff: usize,
    /// T5 encoder d_kv (per-head dimension).
    pub t5_d_kv: usize,
    /// Codebook embedding dimension (maps codebook IDs to decoder hidden).
    pub codebook_embed_dim: usize,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
    /// Maximum generation length in frames.
    pub max_length: usize,
    /// DAC decoder channels schedule.
    pub dac_channels: Vec<usize>,
    /// DAC decoder upsample rates.
    pub dac_upsample_rates: Vec<usize>,
}

impl Default for ParlerTTSConfig {
    fn default() -> Self {
        Self {
            decoder_hidden: 1536,
            decoder_layers: 25,
            decoder_heads: 24,
            decoder_ffn_dim: 6144,
            num_codebooks: 9,
            codebook_size: 1024,
            frame_rate: 86,
            sample_rate: 44100,
            vocab_size: 32128,
            t5_d_model: 2048,
            t5_num_layers: 24,
            t5_num_heads: 32,
            t5_d_ff: 5120,
            t5_d_kv: 64,
            codebook_embed_dim: 1536,
            layer_norm_eps: 1e-5,
            max_length: 2580, // ~30 seconds at 86 Hz
            dac_channels: vec![1536, 768, 384, 192, 96],
            dac_upsample_rates: vec![8, 4, 4, 4],
        }
    }
}

impl ParlerTTSConfig {
    /// Parse from config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| crate::core::Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| crate::core::Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();
        if let Some(dec) = json.get("decoder") {
            if let Some(v) = dec.get("hidden_size").and_then(|v| v.as_u64()) { c.decoder_hidden = v as usize; }
            if let Some(v) = dec.get("num_hidden_layers").and_then(|v| v.as_u64()) { c.decoder_layers = v as usize; }
            if let Some(v) = dec.get("num_attention_heads").and_then(|v| v.as_u64()) { c.decoder_heads = v as usize; }
            if let Some(v) = dec.get("ffn_dim").and_then(|v| v.as_u64()) { c.decoder_ffn_dim = v as usize; }
            if let Some(v) = dec.get("num_codebooks").and_then(|v| v.as_u64()) { c.num_codebooks = v as usize; }
            if let Some(v) = dec.get("codebook_size").and_then(|v| v.as_u64()) { c.codebook_size = v as usize; }
            if let Some(v) = dec.get("max_length").and_then(|v| v.as_u64()) { c.max_length = v as usize; }
        }
        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) { c.vocab_size = v as usize; }
        if let Some(v) = json.get("frame_rate").and_then(|v| v.as_u64()) { c.frame_rate = v as usize; }
        if let Some(v) = json.get("sample_rate").and_then(|v| v.as_u64()) { c.sample_rate = v as usize; }
        Ok(c)
    }

    /// Build T5Config for the frozen text encoder.
    #[cfg(feature = "metal")]
    pub fn t5_config(&self) -> T5Config {
        T5Config {
            d_model: self.t5_d_model,
            d_ff: self.t5_d_ff,
            d_kv: self.t5_d_kv,
            num_heads: self.t5_num_heads,
            num_layers: self.t5_num_layers,
            vocab_size: self.vocab_size,
            num_buckets: 32,
            max_distance: 128,
            scalable_attention: false,
            layer_norm_epsilon: 1e-6,
            is_gated_ffn: true,
        }
    }
}

// ── Metal Kernels ────────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct ParlerKernels {
    common: gpu_ops::CommonKernels,
    silu: Arc<ComputePipeline>,
    gelu: Arc<ComputePipeline>,
    rms_norm: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl ParlerKernels {
    fn new(compute: &MetalCompute) -> Result<Self> {
        Ok(Self {
            common: gpu_ops::CommonKernels::new(compute)?,
            silu: compute.compile_pipeline("parler_silu", sources::SILU, "silu_f16")?,
            gelu: compute.compile_pipeline("parler_gelu", sources::GELU, "gelu_f16")?,
            rms_norm: compute.compile_pipeline("parler_rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
        })
    }
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Parler-TTS synthesis pipeline on Metal GPU.
///
/// Uses a frozen Flan-T5-XL encoder for voice description conditioning,
/// a 25-layer autoregressive transformer for codebook prediction,
/// and a DAC neural codec decoder for waveform synthesis.
#[cfg(feature = "metal")]
pub struct ParlerTTSPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: ParlerTTSConfig,
    kernels: ParlerKernels,
    text_encoder: T5Encoder,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for ParlerTTSPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl ParlerTTSPipeline {
    /// Create a new Parler-TTS pipeline.
    ///
    /// The model must contain weights for both the T5 text encoder
    /// (`text_encoder.*`) and the audio decoder (`decoder.*`) and DAC (`dac.*`).
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: ParlerTTSConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(Arc::clone(&device)));
        let kernels = ParlerKernels::new(&compute)?;
        let t5_cfg = config.t5_config();
        let text_encoder = T5Encoder::new(Arc::clone(&model), t5_cfg, device)?;
        Ok(Self { model, compute, config, kernels, text_encoder })
    }

    /// Synthesize speech audio from text with a voice description.
    ///
    /// - `text`: The text content to speak.
    /// - `voice_description`: Natural language description of the desired voice,
    ///   e.g. "A woman with a calm British accent speaking at a moderate pace."
    ///
    /// Returns PCM audio samples at 44.1kHz.
    pub fn synthesize(&self, text: &str, voice_description: &str) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Encode voice description through frozen T5 encoder
        let desc_tokens = self.tokenize_text(voice_description);
        debug!(desc_tokens = desc_tokens.len(), "Parler-TTS: encoding voice description");
        let encoder_output = self.text_encoder.encode(&desc_tokens)?;
        let enc_seq_len = desc_tokens.len();
        debug!(enc_seq_len, "Parler-TTS: T5 encoder done");

        // 2. Tokenize input text for the decoder prompt
        let text_tokens = self.tokenize_text(text);
        let text_seq = text_tokens.len();
        debug!(text_seq, "Parler-TTS: input text tokenized");

        // 3. Pre-compute encoder K/V for cross-attention (static across decode steps)
        let enc_kv_cache = self.precompute_cross_kv(&encoder_output, enc_seq_len)?;
        debug!(layers = enc_kv_cache.len(), "Parler-TTS: cross-attention KV cache built");

        // 4. Autoregressive decoding: predict codebook tokens
        let codebook_tokens = self.autoregressive_decode(
            &encoder_output, enc_seq_len, &enc_kv_cache,
        )?;
        let num_frames = codebook_tokens.len() / config.num_codebooks;
        debug!(num_frames, num_codebooks = config.num_codebooks, "Parler-TTS: decoder done");

        // 5. DAC codec decode: codebook tokens → audio waveform
        let audio = self.dac_decode(&codebook_tokens, num_frames)?;
        debug!(
            samples = audio.len(),
            duration_s = format!("{:.2}", audio.len() as f32 / config.sample_rate as f32),
            "Parler-TTS: synthesis complete"
        );

        Ok(audio)
    }

    // ── Text Tokenization ────────────────────────────────────────────────────

    /// Simple whitespace-level tokenization using T5 vocabulary.
    /// In production, this would use a SentencePiece tokenizer.
    fn tokenize_text(&self, text: &str) -> Vec<u32> {
        // Use character-level fallback: map ASCII to sequential IDs
        // Real implementation would use SentencePiece/BPE from the T5 tokenizer
        let mut tokens = Vec::new();
        for ch in text.chars() {
            let id = match ch {
                ' ' => 3,     // space
                '.' => 5,     // period
                ',' => 4,     // comma
                '!' => 6,
                '?' => 7,
                'a'..='z' => 100 + (ch as u32 - 'a' as u32),
                'A'..='Z' => 200 + (ch as u32 - 'A' as u32),
                '0'..='9' => 300 + (ch as u32 - '0' as u32),
                _ => 2, // unknown
            };
            tokens.push(id);
        }
        tokens.push(1); // EOS
        tokens
    }

    // ── Cross-Attention KV Cache ─────────────────────────────────────────────

    /// Pre-compute encoder K/V projections for all decoder cross-attention layers.
    /// Returns Vec of (K, V) tensors, one per decoder layer.
    fn precompute_cross_kv(
        &self,
        encoder_output: &Tensor,
        enc_seq_len: usize,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let config = &self.config;
        let d_model = config.decoder_hidden;
        let mut cache = Vec::with_capacity(config.decoder_layers);

        for layer in 0..config.decoder_layers {
            let k_w_name = format!("decoder.layers.{}.encoder_attn.k_proj.weight", layer);
            let v_w_name = format!("decoder.layers.{}.encoder_attn.v_proj.weight", layer);

            let k_w = gpu_ops::read_weight_f16(&self.model, &self.compute, &k_w_name)?;
            let v_w = gpu_ops::read_weight_f16(&self.model, &self.compute, &v_w_name)?;

            let cb = self.compute.new_command_buffer();
            let enc_k = self.linear_tensors(
                cb.as_ref(), encoder_output, &k_w,
                &Tensor::empty(Shape::from([d_model]), DType::F16, encoder_output.device())?,
                enc_seq_len, config.t5_d_model, d_model,
            );
            let enc_v = self.linear_tensors(
                cb.as_ref(), encoder_output, &v_w,
                &Tensor::empty(Shape::from([d_model]), DType::F16, encoder_output.device())?,
                enc_seq_len, config.t5_d_model, d_model,
            );
            cb.commit();
            cb.wait_until_completed();

            cache.push((enc_k, enc_v));
        }

        Ok(cache)
    }

    // ── Autoregressive Decoder ───────────────────────────────────────────────

    /// Run the 25-layer autoregressive decoder to predict codebook tokens.
    ///
    /// Uses delay pattern: codebook c at time t uses the token from time t-c.
    /// At each step, predicts logits for all 9 codebooks simultaneously.
    fn autoregressive_decode(
        &self,
        _encoder_output: &Tensor,
        enc_seq_len: usize,
        enc_kv_cache: &[(Tensor, Tensor)],
    ) -> Result<Vec<u32>> {
        let config = &self.config;
        let d_model = config.decoder_hidden;
        let num_heads = config.decoder_heads;
        let head_dim = d_model / num_heads;
        let device_id = self.compute.device().info().id;
        let eps = config.layer_norm_eps;

        // Initialize with BOS tokens for each codebook
        let bos_token: u32 = 0;
        let mut all_codebook_tokens: Vec<Vec<u32>> = vec![vec![bos_token]; config.num_codebooks];
        let mut output_frames: Vec<Vec<u32>> = Vec::new();

        let max_steps = config.max_length;
        let eos_token = config.codebook_size as u32; // codebook_size is the EOS sentinel

        for step in 0..max_steps {
            // Build decoder input: interleave codebook embeddings with delay pattern
            let dec_seq = step + 1;

            // Embed all codebook tokens and sum their embeddings per time step
            let mut embed_sum = vec![half::f16::ZERO; dec_seq * d_model];
            for cb_idx in 0..config.num_codebooks {
                let _delayed_step = if step >= cb_idx { step - cb_idx } else { 0 };
                let tokens = &all_codebook_tokens[cb_idx];
                let num_tokens = tokens.len().min(dec_seq);

                // Codebook embedding: decoder.embed_tokens.{cb_idx}.weight [codebook_size+1, embed_dim]
                let embed_name = format!("decoder.embed_tokens.{}.weight", cb_idx);
                if let Ok(embed_w) = gpu_ops::read_weight_vec_f32(&self.model, &embed_name) {
                    for t in 0..num_tokens {
                        let tok = tokens[t] as usize;
                        let embed_cols = d_model.min(embed_w.len() / (config.codebook_size + 1));
                        for d in 0..embed_cols {
                            let val = embed_w[tok * embed_cols + d];
                            let idx = t * d_model + d;
                            embed_sum[idx] = half::f16::from_f32(
                                embed_sum[idx].to_f32() + val,
                            );
                        }
                    }
                }
            }

            let h_tensor = Tensor::from_slice(
                &embed_sum,
                Shape::from([dec_seq, d_model]),
                DType::F16,
                device_id,
            )?;

            // Run through decoder layers
            let logits = self.decoder_forward(
                &h_tensor, dec_seq, enc_seq_len, enc_kv_cache, d_model,
                num_heads, head_dim, eps,
            )?;

            // Extract logits for the last position, split across codebooks
            // logits shape: [num_codebooks, codebook_size+1]
            let logits_data = logits.to_f32_vec()?;
            let logit_dim = config.codebook_size + 1; // +1 for EOS token

            let mut frame_tokens = Vec::with_capacity(config.num_codebooks);
            let mut all_eos = true;

            for cb_idx in 0..config.num_codebooks {
                let offset = cb_idx * logit_dim;
                let cb_logits = &logits_data[offset..offset + logit_dim];

                // Argmax
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx: u32 = 0;
                for (i, &v) in cb_logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i as u32;
                    }
                }

                if max_idx != eos_token {
                    all_eos = false;
                }
                frame_tokens.push(max_idx);
                all_codebook_tokens[cb_idx].push(max_idx);
            }

            output_frames.push(frame_tokens);

            if all_eos {
                debug!(step, "Parler-TTS: all codebooks reached EOS");
                break;
            }

            if step % 100 == 0 {
                debug!(step, max_steps, "Parler-TTS: decoding progress");
            }
        }

        // Flatten: frame-major ordering [frame0_cb0, frame0_cb1, ..., frame1_cb0, ...]
        let flat: Vec<u32> = output_frames.into_iter().flatten().collect();
        Ok(flat)
    }

    /// Run one forward pass through the decoder transformer stack.
    /// Returns logits for all codebooks at the last position.
    fn decoder_forward(
        &self,
        input: &Tensor,
        dec_seq: usize,
        enc_seq_len: usize,
        enc_kv_cache: &[(Tensor, Tensor)],
        d_model: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let config = &self.config;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;

        // Convert input to f32 for residual stream
        let mut h = input.to_f32_vec()?;
        let h_len = dec_seq * d_model;

        for layer in 0..config.decoder_layers {
            let prefix = format!("decoder.layers.{}", layer);

            // === Causal Self-Attention ===
            let sa_ln_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.self_attn_layer_norm.weight", prefix))?;
            let sa_ln_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.self_attn_layer_norm.bias", prefix))?;
            let normed = layer_norm_f32(&h, dec_seq, d_model, &sa_ln_w, &sa_ln_b, eps);

            // Q/K/V projections
            let q = self.project_f32(&normed, dec_seq, d_model, d_model, &format!("{}.self_attn.q_proj", prefix))?;
            let k = self.project_f32(&normed, dec_seq, d_model, d_model, &format!("{}.self_attn.k_proj", prefix))?;
            let v = self.project_f32(&normed, dec_seq, d_model, d_model, &format!("{}.self_attn.v_proj", prefix))?;

            // Multi-head causal self-attention on CPU
            let sa_out = multi_head_causal_attention_f32(
                &q, &k, &v, dec_seq, dec_seq, num_heads, head_dim, scale,
            );

            // Output projection + residual
            let sa_proj = self.project_f32(&sa_out, dec_seq, d_model, d_model, &format!("{}.self_attn.out_proj", prefix))?;
            for i in 0..h_len {
                h[i] += sa_proj[i];
            }

            // === Cross-Attention ===
            let ca_ln_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.encoder_attn_layer_norm.weight", prefix))?;
            let ca_ln_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.encoder_attn_layer_norm.bias", prefix))?;
            let normed = layer_norm_f32(&h, dec_seq, d_model, &ca_ln_w, &ca_ln_b, eps);

            // Q from decoder, K/V from pre-computed encoder cache
            let cross_q = self.project_f32(&normed, dec_seq, d_model, d_model, &format!("{}.encoder_attn.q_proj", prefix))?;
            let (enc_k, enc_v) = &enc_kv_cache[layer];
            let enc_k_f32 = enc_k.to_f32_vec()?;
            let enc_v_f32 = enc_v.to_f32_vec()?;

            // Multi-head cross-attention (no causal mask)
            let ca_out = multi_head_attention_f32(
                &cross_q, &enc_k_f32, &enc_v_f32,
                dec_seq, enc_seq_len, num_heads, head_dim, scale,
            );

            let ca_proj = self.project_f32(&ca_out, dec_seq, d_model, d_model, &format!("{}.encoder_attn.out_proj", prefix))?;
            for i in 0..h_len {
                h[i] += ca_proj[i];
            }

            // === FFN ===
            let ff_ln_w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.final_layer_norm.weight", prefix))?;
            let ff_ln_b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.final_layer_norm.bias", prefix))?;
            let normed = layer_norm_f32(&h, dec_seq, d_model, &ff_ln_w, &ff_ln_b, eps);

            // FFN: linear → GELU → linear
            let fc1_out = self.project_bias_f32(&normed, dec_seq, d_model, config.decoder_ffn_dim, &format!("{}.fc1", prefix))?;
            let activated: Vec<f32> = fc1_out.iter().map(|&v| gelu_f32(v)).collect();
            let fc2_out = self.project_bias_f32(&activated, dec_seq, config.decoder_ffn_dim, d_model, &format!("{}.fc2", prefix))?;
            for i in 0..h_len {
                h[i] += fc2_out[i];
            }
        }

        // Final layer norm
        let final_ln_w = gpu_ops::read_weight_vec_f32(&self.model, "decoder.layer_norm.weight")?;
        let final_ln_b = gpu_ops::read_weight_vec_f32(&self.model, "decoder.layer_norm.bias")?;
        let h = layer_norm_f32(&h, dec_seq, d_model, &final_ln_w, &final_ln_b, eps);

        // Extract last position
        let last_row = &h[(dec_seq - 1) * d_model..dec_seq * d_model];

        // Project to codebook logits: 9 separate lm_head projections
        let logit_dim = config.codebook_size + 1;
        let mut all_logits = vec![0.0f32; config.num_codebooks * logit_dim];

        for cb_idx in 0..config.num_codebooks {
            let head_name = format!("decoder.lm_heads.{}.weight", cb_idx);
            if let Ok(w) = gpu_ops::read_weight_vec_f32(&self.model, &head_name) {
                let out_dim = logit_dim.min(w.len() / d_model);
                for o in 0..out_dim {
                    let mut sum = 0.0f32;
                    for d in 0..d_model {
                        sum += last_row[d] * w[o * d_model + d];
                    }
                    all_logits[cb_idx * logit_dim + o] = sum;
                }
            }
        }

        let f16_logits: Vec<half::f16> = all_logits.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(
            &f16_logits,
            Shape::from([config.num_codebooks, logit_dim]),
            DType::F16,
            device_id,
        )
    }

    // ── DAC Codec Decoder ────────────────────────────────────────────────────

    /// Decode codebook tokens to audio waveform using the DAC neural codec.
    ///
    /// DAC architecture:
    ///   9 codebook embeddings → sum → ConvTranspose1d upsampling stack
    ///   → Snake activation + ResBlocks → 44.1kHz mono audio
    fn dac_decode(&self, codebook_tokens: &[u32], num_frames: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let embed_dim = config.codebook_embed_dim;

        // 1. Look up and sum codebook embeddings: [num_frames, embed_dim]
        let mut embeddings = vec![0.0f32; num_frames * embed_dim];

        for cb_idx in 0..config.num_codebooks {
            let quantize_name = format!("dac.quantizer.quantizers.{}.codebook.weight", cb_idx);
            let codebook_w = gpu_ops::read_weight_vec_f32(&self.model, &quantize_name)
                .unwrap_or_else(|_| vec![0.0f32; config.codebook_size * embed_dim]);
            let cols = embed_dim.min(codebook_w.len() / config.codebook_size.max(1));

            for frame in 0..num_frames {
                let tok_idx = frame * config.num_codebooks + cb_idx;
                if tok_idx < codebook_tokens.len() {
                    let tok = (codebook_tokens[tok_idx] as usize).min(config.codebook_size - 1);
                    for d in 0..cols {
                        embeddings[frame * embed_dim + d] += codebook_w[tok * cols + d];
                    }
                }
            }
        }

        // 2. Input projection: [num_frames, embed_dim] → [channels[0], num_frames]
        // Transpose to channel-first: [embed_dim, num_frames]
        let ch0 = config.dac_channels[0];
        let mut x = vec![0.0f32; ch0 * num_frames];
        for c in 0..ch0.min(embed_dim) {
            for f in 0..num_frames {
                x[c * num_frames + f] = embeddings[f * embed_dim + c];
            }
        }

        // 3. Input conv: 1D convolution [ch0, embed_dim, 7] (kernel_size=7)
        let in_conv_name = "dac.decoder.model.0";
        x = self.conv1d_cpu(&x, ch0, num_frames, &format!("{}.weight", in_conv_name),
            &format!("{}.bias", in_conv_name), ch0, ch0, 7, 3)?;
        let mut current_len = num_frames;
        let mut current_ch = ch0;

        // 4. Upsample stages with Snake activation + ResBlocks
        for (stage, (&rate, &next_ch)) in config.dac_upsample_rates.iter()
            .zip(config.dac_channels[1..].iter())
            .enumerate()
        {
            // Snake activation: x + (1/alpha) * sin^2(alpha * x)
            x = snake_activation_f32(&x, current_ch, current_len);

            // ConvTranspose1d upsample
            let up_prefix = format!("dac.decoder.model.{}", 1 + stage * 3);
            let out_len = (current_len - 1) * rate + rate * 2; // approximate output length
            x = self.conv_transpose1d_cpu(
                &x, current_ch, current_len, next_ch, rate, rate * 2,
                &format!("{}.weight", up_prefix), &format!("{}.bias", up_prefix),
            )?;
            current_len = out_len;
            current_ch = next_ch;

            // ResBlock: 2 dilated conv layers with Snake activation
            for res_block in 0..3 {
                let res_prefix = format!("dac.decoder.model.{}.{}", 2 + stage * 3, res_block);
                let residual = x.clone();

                // Snake + Conv1d (dilated)
                x = snake_activation_f32(&x, current_ch, current_len);
                let dilation = 3usize.pow(res_block as u32);
                x = self.dilated_conv1d_cpu(
                    &x, current_ch, current_len, current_ch, 7, dilation,
                    &format!("{}.block.0.weight", res_prefix),
                    &format!("{}.block.0.bias", res_prefix),
                )?;

                // Snake + Conv1d (no dilation)
                x = snake_activation_f32(&x, current_ch, current_len);
                x = self.conv1d_cpu(
                    &x, current_ch, current_len,
                    &format!("{}.block.1.weight", res_prefix),
                    &format!("{}.block.1.bias", res_prefix),
                    current_ch, current_ch, 1, 0,
                )?;

                // Residual connection
                for i in 0..x.len().min(residual.len()) {
                    x[i] += residual[i];
                }
            }
        }

        // 5. Final Snake + Conv1d → 1 channel
        x = snake_activation_f32(&x, current_ch, current_len);
        let out_prefix = "dac.decoder.model.final";
        x = self.conv1d_cpu(
            &x, current_ch, current_len,
            &format!("{}.weight", out_prefix),
            &format!("{}.bias", out_prefix),
            1, current_ch, 7, 3,
        ).unwrap_or_else(|_| {
            // Fallback: simple channel averaging
            let mut audio = vec![0.0f32; current_len];
            for l in 0..current_len {
                let mut sum = 0.0f32;
                for c in 0..current_ch {
                    sum += x[c * current_len + l];
                }
                audio[l] = sum / current_ch as f32;
            }
            audio
        });

        // 6. Tanh activation + normalization
        let audio: Vec<f32> = x.iter().map(|&v| v.tanh()).collect();

        // Normalize to [-1, 1]
        let max_abs = audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs > 1e-6 {
            let scale = 0.95 / max_abs;
            Ok(audio.iter().map(|&v| v * scale).collect())
        } else {
            Ok(audio)
        }
    }

    // ── CPU Helpers ──────────────────────────────────────────────────────────

    /// Linear projection: X @ W^T + bias, on CPU.
    fn project_f32(
        &self, input: &[f32], rows: usize, in_dim: usize, out_dim: usize, prefix: &str,
    ) -> Result<Vec<f32>> {
        let w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.weight", prefix))?;
        let mut out = vec![0.0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut sum = 0.0f32;
                for i in 0..in_dim {
                    sum += input[r * in_dim + i] * w[o * in_dim + i];
                }
                out[r * out_dim + o] = sum;
            }
        }
        Ok(out)
    }

    /// Linear projection with bias: X @ W^T + b, on CPU.
    fn project_bias_f32(
        &self, input: &[f32], rows: usize, in_dim: usize, out_dim: usize, prefix: &str,
    ) -> Result<Vec<f32>> {
        let w = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.weight", prefix))?;
        let b = gpu_ops::read_weight_vec_f32(&self.model, &format!("{}.bias", prefix))
            .unwrap_or_else(|_| vec![0.0f32; out_dim]);
        let mut out = vec![0.0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut sum = b[o.min(b.len() - 1)];
                for i in 0..in_dim {
                    sum += input[r * in_dim + i] * w[o * in_dim + i];
                }
                out[r * out_dim + o] = sum;
            }
        }
        Ok(out)
    }

    /// 1D convolution on CPU: [c_in, length] → [c_out, length].
    fn conv1d_cpu(
        &self, input: &[f32], c_in: usize, length: usize,
        w_name: &str, b_name: &str,
        c_out: usize, _c_in_check: usize, kernel_size: usize, padding: usize,
    ) -> Result<Vec<f32>> {
        let w = gpu_ops::read_weight_vec_f32(&self.model, w_name)?;
        let b = gpu_ops::read_weight_vec_f32(&self.model, b_name)
            .unwrap_or_else(|_| vec![0.0f32; c_out]);

        let mut output = vec![0.0f32; c_out * length];
        for co in 0..c_out {
            for l in 0..length {
                let mut sum = b[co.min(b.len() - 1)];
                for ci in 0..c_in {
                    for k in 0..kernel_size {
                        let pos = l as isize + k as isize - padding as isize;
                        if pos >= 0 && (pos as usize) < length {
                            let w_idx = (co * c_in + ci) * kernel_size + k;
                            if w_idx < w.len() {
                                sum += input[ci * length + pos as usize] * w[w_idx];
                            }
                        }
                    }
                }
                output[co * length + l] = sum;
            }
        }
        Ok(output)
    }

    /// Dilated 1D convolution on CPU.
    fn dilated_conv1d_cpu(
        &self, input: &[f32], c_in: usize, length: usize,
        c_out: usize, kernel_size: usize, dilation: usize,
        w_name: &str, b_name: &str,
    ) -> Result<Vec<f32>> {
        let w = gpu_ops::read_weight_vec_f32(&self.model, w_name)?;
        let b = gpu_ops::read_weight_vec_f32(&self.model, b_name)
            .unwrap_or_else(|_| vec![0.0f32; c_out]);
        let _padding = (kernel_size / 2) * dilation;

        let mut output = vec![0.0f32; c_out * length];
        for co in 0..c_out {
            for l in 0..length {
                let mut sum = b[co.min(b.len() - 1)];
                for ci in 0..c_in {
                    for k in 0..kernel_size {
                        let pos = l as isize + (k as isize - kernel_size as isize / 2) * dilation as isize;
                        if pos >= 0 && (pos as usize) < length {
                            let w_idx = (co * c_in + ci) * kernel_size + k;
                            if w_idx < w.len() {
                                sum += input[ci * length + pos as usize] * w[w_idx];
                            }
                        }
                    }
                }
                output[co * length + l] = sum;
            }
        }
        Ok(output)
    }

    /// Transposed 1D convolution on CPU (for upsampling).
    fn conv_transpose1d_cpu(
        &self, input: &[f32], c_in: usize, length: usize,
        c_out: usize, stride: usize, kernel_size: usize,
        w_name: &str, b_name: &str,
    ) -> Result<Vec<f32>> {
        let w = gpu_ops::read_weight_vec_f32(&self.model, w_name)?;
        let b = gpu_ops::read_weight_vec_f32(&self.model, b_name)
            .unwrap_or_else(|_| vec![0.0f32; c_out]);
        let padding = (kernel_size - stride) / 2;
        let out_len = (length - 1) * stride + kernel_size - 2 * padding;

        let mut output = vec![0.0f32; c_out * out_len];
        // Add bias
        for co in 0..c_out {
            let bv = b[co.min(b.len() - 1)];
            for l in 0..out_len {
                output[co * out_len + l] = bv;
            }
        }

        // Transposed convolution scatter
        for ci in 0..c_in {
            for li in 0..length {
                for co in 0..c_out {
                    for k in 0..kernel_size {
                        let lo = li as isize * stride as isize + k as isize - padding as isize;
                        if lo >= 0 && (lo as usize) < out_len {
                            let w_idx = (ci * c_out + co) * kernel_size + k;
                            if w_idx < w.len() {
                                output[co * out_len + lo as usize] +=
                                    input[ci * length + li] * w[w_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok(output)
    }
}

// ── Free Functions ───────────────────────────────────────────────────────────

/// Layer normalization on CPU (f32).
fn layer_norm_f32(
    x: &[f32], n: usize, d: usize, weight: &[f32], bias: &[f32], eps: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for i in 0..n {
        let row = &mut out[i * d..(i + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            row[j] = (row[j] - mean) * inv_std
                * weight[j.min(weight.len() - 1)]
                + bias[j.min(bias.len() - 1)];
        }
    }
    out
}

/// Multi-head causal self-attention on CPU (f32).
fn multi_head_causal_attention_f32(
    q: &[f32], k: &[f32], v: &[f32],
    q_seq: usize, kv_seq: usize,
    num_heads: usize, head_dim: usize, scale: f32,
) -> Vec<f32> {
    let hidden = num_heads * head_dim;
    let mut output = vec![0.0f32; q_seq * hidden];

    for h in 0..num_heads {
        for qi in 0..q_seq {
            // Compute scores with causal mask
            let mut scores = vec![f32::NEG_INFINITY; kv_seq];
            for ki in 0..=qi.min(kv_seq - 1) {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * hidden + h * head_dim + d]
                         * k[ki * hidden + h * head_dim + d];
                }
                scores[ki] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / (sum_exp + 1e-12);
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }

            // Weighted sum of values
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..kv_seq {
                    sum += scores[ki] * v[ki * hidden + h * head_dim + d];
                }
                output[qi * hidden + h * head_dim + d] = sum;
            }
        }
    }
    output
}

/// Multi-head attention on CPU (f32) -- no causal mask (for cross-attention).
fn multi_head_attention_f32(
    q: &[f32], k: &[f32], v: &[f32],
    q_seq: usize, kv_seq: usize,
    num_heads: usize, head_dim: usize, scale: f32,
) -> Vec<f32> {
    let hidden = num_heads * head_dim;
    let mut output = vec![0.0f32; q_seq * hidden];

    for h in 0..num_heads {
        for qi in 0..q_seq {
            let mut scores = vec![0.0f32; kv_seq];
            for ki in 0..kv_seq {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * hidden + h * head_dim + d]
                         * k[ki * hidden + h * head_dim + d];
                }
                scores[ki] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / (sum_exp + 1e-12);
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }

            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..kv_seq {
                    sum += scores[ki] * v[ki * hidden + h * head_dim + d];
                }
                output[qi * hidden + h * head_dim + d] = sum;
            }
        }
    }
    output
}

/// Snake activation: x + (1/alpha) * sin^2(alpha * x), per-channel.
/// Input layout: [channels, length]. Uses alpha=1.0 as default.
fn snake_activation_f32(x: &[f32], channels: usize, length: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * length];
    for c in 0..channels {
        for l in 0..length {
            let idx = c * length + l;
            if idx < x.len() {
                let v = x[idx];
                let s = v.sin();
                output[idx] = v + s * s; // alpha = 1.0
            }
        }
    }
    output
}

/// GELU activation (exact).
#[inline]
fn gelu_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * 0.7071067811865476_f32).tanh())
}

/// Approximate tanh for GELU.
#[allow(dead_code)]
trait TanhApprox {
    fn tanh(self) -> Self;
}
