//! Moonshine: Edge-efficient ASR (27-61M params) processing raw waveforms.
//!
//! Architecture:
//!   Raw 16kHz audio → Conv1d frontend (3 layers, 384x compression)
//!   → Encoder (6 transformer layers, RoPE, GELU)
//!   → Decoder (6 transformer layers, RoPE, SiLU, cross-attention, KV cache)
//!   → Text tokens
//!
//! Key innovation: NO mel spectrogram — raw waveform goes directly through
//! Conv1d layers with large kernel (127) and progressive stride compression.
//! This eliminates FFT preprocessing and enables fully end-to-end learning.
//!
//! Model weights: `model.safetensors` (Moonshine-base ~61M params)
//! Config: `config.json` with encoder/decoder hyperparameters.

use crate::core::{Error, Result};
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

// ── Configuration ────────────────────────────────────────────────────────────

/// Moonshine ASR configuration (from config.json).
#[derive(Debug, Clone)]
pub struct MoonshineConfig {
    /// Hidden size for both encoder and decoder.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of encoder transformer layers.
    pub num_encoder_layers: usize,
    /// Number of decoder transformer layers.
    pub num_decoder_layers: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size for the decoder.
    pub vocab_size: usize,
    /// Maximum target (decoder) sequence length.
    pub max_target_positions: usize,
    /// Token ID for start-of-sequence / beginning of transcription.
    pub decoder_start_token_id: u32,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
    /// RoPE theta base frequency.
    pub rope_theta: f32,
}

impl Default for MoonshineConfig {
    fn default() -> Self {
        // Moonshine-base defaults
        Self {
            hidden_size: 288,
            num_attention_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            intermediate_size: 1152,
            vocab_size: 32768,
            max_target_positions: 448,
            decoder_start_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,
        }
    }
}

impl MoonshineConfig {
    /// Parse Moonshine configuration from a JSON file.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path).map_err(|e|
            Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str).map_err(|e|
            Error::internal(format!("failed to parse config: {}", e)))?;

        let mut config = Self::default();
        if let Some(v) = json.get("hidden_size").and_then(|v| v.as_u64()) { config.hidden_size = v as usize; }
        if let Some(v) = json.get("num_attention_heads").and_then(|v| v.as_u64()) { config.num_attention_heads = v as usize; }
        if let Some(v) = json.get("num_encoder_layers").and_then(|v| v.as_u64()) { config.num_encoder_layers = v as usize; }
        if let Some(v) = json.get("num_decoder_layers").and_then(|v| v.as_u64()) { config.num_decoder_layers = v as usize; }
        if let Some(v) = json.get("intermediate_size").and_then(|v| v.as_u64()) { config.intermediate_size = v as usize; }
        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) { config.vocab_size = v as usize; }
        if let Some(v) = json.get("max_target_positions").and_then(|v| v.as_u64()) { config.max_target_positions = v as usize; }
        if let Some(v) = json.get("decoder_start_token_id").and_then(|v| v.as_u64()) { config.decoder_start_token_id = v as u32; }
        if let Some(v) = json.get("eos_token_id").and_then(|v| v.as_u64()) { config.eos_token_id = v as u32; }
        if let Some(v) = json.get("rope_theta").and_then(|v| v.as_f64()) { config.rope_theta = v as f32; }
        // Also try nested encoder/decoder configs
        if let Some(enc) = json.get("encoder") {
            if let Some(v) = enc.get("hidden_size").and_then(|v| v.as_u64()) { config.hidden_size = v as usize; }
            if let Some(v) = enc.get("num_attention_heads").and_then(|v| v.as_u64()) { config.num_attention_heads = v as usize; }
            if let Some(v) = enc.get("num_hidden_layers").and_then(|v| v.as_u64()) { config.num_encoder_layers = v as usize; }
            if let Some(v) = enc.get("intermediate_size").and_then(|v| v.as_u64()) { config.intermediate_size = v as usize; }
        }
        if let Some(dec) = json.get("decoder") {
            if let Some(v) = dec.get("num_hidden_layers").and_then(|v| v.as_u64()) { config.num_decoder_layers = v as usize; }
            if let Some(v) = dec.get("vocab_size").and_then(|v| v.as_u64()) { config.vocab_size = v as usize; }
        }
        Ok(config)
    }
}

// ── Metal Kernels ────────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct MoonshineKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    silu: Arc<ComputePipeline>,
    conv1d: Arc<ComputePipeline>,
    embedding: Arc<ComputePipeline>,
    rope_single: Arc<ComputePipeline>,
    rope_batched: Arc<ComputePipeline>,
    copy_to_kv_cache: Arc<ComputePipeline>,
    gqa_attention: Arc<ComputePipeline>,
    autoregressive_attention: Arc<ComputePipeline>,
    nchw_to_nhwc: Arc<ComputePipeline>,
}

// ── KV Cache ─────────────────────────────────────────────────────────────────

/// KV cache for Moonshine decoder.
/// Caches self-attention K/V (growing per token) and cross-attention K/V
/// (static after prefill, pre-transposed to HSD format).
#[cfg(feature = "metal")]
struct MoonshineKVCache {
    /// Self-attention K cache per layer: [max_target_positions, num_heads, head_dim]
    self_k: Vec<Tensor>,
    /// Self-attention V cache per layer
    self_v: Vec<Tensor>,
    /// Pre-transposed cross K cache: [num_heads, enc_seq_len, head_dim] (HSD)
    cross_k_hsd: Vec<Tensor>,
    /// Pre-transposed cross V cache: [num_heads, enc_seq_len, head_dim] (HSD)
    cross_v_hsd: Vec<Tensor>,
    /// Number of valid positions in self-attention cache
    seq_len: usize,
}

#[cfg(feature = "metal")]
impl MoonshineKVCache {
    fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_target_positions: usize,
        device_id: crate::hal::DeviceId,
    ) -> Result<Self> {
        let shape = Shape::from([max_target_positions, num_heads, head_dim]);
        let mut self_k = Vec::with_capacity(num_layers);
        let mut self_v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            self_k.push(Tensor::empty(shape.clone(), DType::F16, device_id)?);
            self_v.push(Tensor::empty(shape.clone(), DType::F16, device_id)?);
        }
        Ok(Self {
            self_k,
            self_v,
            cross_k_hsd: Vec::new(),
            cross_v_hsd: Vec::new(),
            seq_len: 0,
        })
    }

    /// Store pre-transposed cross K/V in HSD [num_heads, enc_seq_len, head_dim] format.
    /// Called once during prefill to avoid re-transposing every decode step.
    fn set_cross_hsd(&mut self, layer: usize, k_hsd: Tensor, v_hsd: Tensor) {
        while self.cross_k_hsd.len() <= layer {
            if let Ok(placeholder) = Tensor::zeros_on(
                Shape::from([1]), DType::F16, crate::hal::DeviceId::cpu(),
            ) {
                self.cross_k_hsd.push(placeholder.clone());
                self.cross_v_hsd.push(placeholder);
            }
        }
        self.cross_k_hsd[layer] = k_hsd;
        self.cross_v_hsd[layer] = v_hsd;
    }
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Moonshine edge ASR pipeline for speech-to-text on Metal GPU.
///
/// Processes raw 16kHz waveform (no mel spectrogram) through:
/// 1. Conv1d frontend (3 layers, 384x temporal compression)
/// 2. Encoder (6 transformer layers with RoPE + GELU)
/// 3. Decoder (6 transformer layers with RoPE + SiLU + cross-attention)
/// 4. Greedy token decoding
#[cfg(feature = "metal")]
pub struct MoonshinePipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    config: MoonshineConfig,
    compute: Arc<MetalCompute>,
    kernels: MoonshineKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for MoonshinePipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl MoonshinePipeline {
    /// Create a new Moonshine pipeline with Metal GPU acceleration.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: MoonshineConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));

        let kernels = MoonshineKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            rope_single: compute.compile_pipeline("rope_single", sources::ROPE, "rope_single_f16")?,
            rope_batched: compute.compile_pipeline("rope_batched", sources::PHASE27_OPS, "rope_batched_f16")?,
            copy_to_kv_cache: compute.compile_pipeline("copy_to_kv_cache", sources::ROPE, "copy_to_kv_cache_f16")?,
            gqa_attention: compute.compile_pipeline("gqa_attention", sources::GQA_ATTENTION, "gqa_attention_f16")?,
            autoregressive_attention: compute.compile_pipeline("autoregressive_attention", sources::AUTOREGRESSIVE_ATTENTION, "autoregressive_attention_f16")?,
            nchw_to_nhwc: compute.compile_pipeline("nchw_to_nhwc", sources::TRANSPOSE, "nchw_to_nhwc_f16")?,
        };

        Ok(Self { model, config, compute, kernels })
    }

    /// Transcribe raw audio samples to token IDs.
    ///
    /// - `audio_samples`: PCM samples at the given sample rate (mono, f32 in [-1, 1]).
    /// - `sample_rate`: sample rate of the input audio (will be resampled to 16kHz if needed).
    ///
    /// Returns decoded token IDs. Use a tokenizer to convert to text.
    pub fn transcribe(&self, audio_samples: &[f32], sample_rate: u32) -> Result<Vec<u32>> {
        let config = &self.config;

        // Resample to 16kHz if needed
        let samples_16k = if sample_rate != 16000 {
            resample_to_16k(audio_samples, sample_rate)
        } else {
            audio_samples.to_vec()
        };

        // 1. Conv1d frontend: raw waveform → encoder input sequence
        let encoder_input = self.conv_frontend(&samples_16k)?;
        let enc_seq_len = encoder_input.shape().dim(0).unwrap_or(1);

        // 2. Encoder: transformer layers with RoPE + GELU
        let encoder_out = self.encode(&encoder_input, enc_seq_len)?;

        // 3. Initialize KV cache
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads;
        let mut kv_cache = MoonshineKVCache::new(
            config.num_decoder_layers, num_heads, head_dim,
            config.max_target_positions, self.compute.device().info().id,
        )?;

        // 4. Prefill with start token
        let prefix = vec![config.decoder_start_token_id];
        let logits = self.decode_prefill(&prefix, &encoder_out, enc_seq_len, &mut kv_cache)?;
        let mut next_token = self.argmax_cpu(&logits, config.vocab_size)?;

        let mut tokens: Vec<u32> = Vec::new();

        // 5. Autoregressive decode loop
        for _step in 0..config.max_target_positions {
            if next_token == config.eos_token_id {
                break;
            }
            tokens.push(next_token);
            let logits = self.decode_step(next_token, &mut kv_cache)?;
            next_token = self.argmax_cpu(&logits, config.vocab_size)?;
        }

        Ok(tokens)
    }

    // ========================= CONV1D FRONTEND =========================

    /// Conv1d frontend: raw waveform → [seq_len, hidden_size].
    ///
    /// 3 Conv1d layers with GELU activation:
    ///   Conv1d(1, 288, k=127, s=64) → GELU
    ///   Conv1d(288, 288, k=7, s=3) → GELU
    ///   Conv1d(288, 288, k=3, s=2) → GELU
    /// Total temporal compression: 64 * 3 * 2 = 384x
    ///
    /// Uses CPU for convolutions (small number of frames, large kernels),
    /// then transfers to GPU for the transformer layers.
    fn conv_frontend(&self, audio: &[f32]) -> Result<Tensor> {
        let config = &self.config;
        let d = config.hidden_size;
        let device_id = self.compute.device().info().id;

        // Convert audio to f16 tensor: [1, num_samples] (1 input channel)
        let num_samples = audio.len();

        // --- Conv1 (1 → d, k=127, s=64, padding=63) ---
        let conv1_w = gpu_ops::read_weight_vec_f32(&self.model, "model.encoder.conv1.weight")?;
        let conv1_b = gpu_ops::read_weight_vec_f32(&self.model, "model.encoder.conv1.bias")?;
        let l1 = conv1d_cpu(audio, 1, num_samples, &conv1_w, &conv1_b, d, 127, 64, 63);
        let l1_len = l1.len() / d;
        let l1 = gelu_cpu(&l1);

        // --- Conv2 (d → d, k=7, s=3, padding=3) ---
        let conv2_w = gpu_ops::read_weight_vec_f32(&self.model, "model.encoder.conv2.weight")?;
        let conv2_b = gpu_ops::read_weight_vec_f32(&self.model, "model.encoder.conv2.bias")?;
        let l2 = conv1d_cpu(&l1, d, l1_len, &conv2_w, &conv2_b, d, 7, 3, 3);
        let l2_len = l2.len() / d;
        let l2 = gelu_cpu(&l2);

        // --- Conv3 (d → d, k=3, s=2, padding=1) ---
        let conv3_w = gpu_ops::read_weight_vec_f32(&self.model, "model.encoder.conv3.weight")?;
        let conv3_b = gpu_ops::read_weight_vec_f32(&self.model, "model.encoder.conv3.bias")?;
        let l3 = conv1d_cpu(&l2, d, l2_len, &conv3_w, &conv3_b, d, 3, 2, 1);
        let l3_len = l3.len() / d;
        let l3 = gelu_cpu(&l3);

        // Output is [d, seq_len] in channel-first layout; transpose to [seq_len, d]
        let mut transposed = vec![0.0f32; l3_len * d];
        for s in 0..l3_len {
            for c in 0..d {
                transposed[s * d + c] = l3[c * l3_len + s];
            }
        }

        // Convert to F16 tensor on GPU
        let f16_data: Vec<half::f16> = transposed.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, Shape::from([l3_len, d]), DType::F16, device_id)
    }

    // ========================= ENCODER =========================

    /// Encoder: 6 transformer layers with RoPE + GELU + LayerNorm.
    /// Input: [seq_len, hidden_size], Output: [seq_len, hidden_size].
    fn encode(&self, input: &Tensor, seq_len: usize) -> Result<Tensor> {
        let config = &self.config;
        let d = config.hidden_size;

        // Per-layer command buffers (large intermediates benefit from inter-layer reclamation)
        let mut hidden = input.clone();
        for layer in 0..config.num_encoder_layers {
            let cb = self.compute.new_command_buffer();
            hidden = self.encoder_layer_on(&cb, layer, hidden, seq_len)?;
            cb.commit();
            cb.wait_until_completed();
        }

        // Final layer norm
        let cb = self.compute.new_command_buffer();
        let out = self.layer_norm(
            &cb, &self.model, &hidden,
            "model.encoder.layer_norm.weight",
            "model.encoder.layer_norm.bias",
            seq_len, d, 1e-5,
        )?;
        cb.commit();
        cb.wait_until_completed();

        Ok(out)
    }

    /// Single encoder transformer layer on a shared command buffer.
    ///
    /// Pre-norm architecture:
    ///   LN → Self-Attn (RoPE, non-causal) → residual
    ///   LN → FFN (linear → GELU → linear) → residual
    fn encoder_layer_on(
        &self, cb: &metal::CommandBufferRef, layer: usize, input: Tensor, seq_len: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let d = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = d / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let prefix = format!("model.encoder.layers.{}", layer);

        // Pre-attention LayerNorm
        let normed = self.layer_norm(
            cb, &self.model, &input,
            &format!("{}.layer_norm.weight", prefix),
            &format!("{}.layer_norm.bias", prefix),
            seq_len, d, 1e-5,
        )?;

        // Q/K/V projections
        let q = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.q_proj.weight", prefix),
            &format!("{}.self_attn.q_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let k = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.k_proj.weight", prefix),
            &format!("{}.self_attn.k_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let v = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.v_proj.weight", prefix),
            &format!("{}.self_attn.v_proj.bias", prefix),
            seq_len, d, d,
        )?;

        // Apply RoPE to Q and K (in-place)
        let q = q.reshape([seq_len, num_heads, head_dim])?;
        let k = k.reshape([seq_len, num_heads, head_dim])?;
        self.apply_rope_batch_on(cb, &q, seq_len, num_heads, head_dim);
        self.apply_rope_batch_on(cb, &k, seq_len, num_heads, head_dim);

        // Non-causal self-attention via batched matmul
        let v = v.reshape([seq_len, num_heads, head_dim])?;
        let attn_out = self.batched_attention(
            cb, &q, &k, &v,
            seq_len, seq_len, num_heads, head_dim, scale,
        )?;

        // Output projection + residual
        let o = self.linear_bias(
            cb, &self.model, &attn_out,
            &format!("{}.self_attn.out_proj.weight", prefix),
            &format!("{}.self_attn.out_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let h = self.add(cb, &input, &o);

        // Pre-FFN LayerNorm
        let normed2 = self.layer_norm(
            cb, &self.model, &h,
            &format!("{}.layer_norm1.weight", prefix),
            &format!("{}.layer_norm1.bias", prefix),
            seq_len, d, 1e-5,
        )?;

        // FFN: fc1 → GELU → fc2
        let fc1 = self.linear_bias(
            cb, &self.model, &normed2,
            &format!("{}.mlp.fc1.weight", prefix),
            &format!("{}.mlp.fc1.bias", prefix),
            seq_len, d, config.intermediate_size,
        )?;
        let activated = self.activation(cb, &self.kernels.gelu, &fc1);
        let fc2 = self.linear_bias(
            cb, &self.model, &activated,
            &format!("{}.mlp.fc2.weight", prefix),
            &format!("{}.mlp.fc2.bias", prefix),
            seq_len, config.intermediate_size, d,
        )?;

        // Residual
        Ok(self.add(cb, &h, &fc2))
    }

    // ========================= CACHED DECODER (prefill + step) =========================

    /// Prefill: process all prefix tokens at once, populate KV caches.
    fn decode_prefill(
        &self, tokens: &[u32], encoder_out: &Tensor, enc_seq_len: usize,
        kv_cache: &mut MoonshineKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let d = config.hidden_size;
        let seq_len = tokens.len();

        let cb = self.compute.new_command_buffer();

        // Token embedding
        let h = self.embed_tokens_on(&cb, tokens, d);

        let mut hidden = h;

        // All decoder layers on same CB (with inline blits for KV cache)
        for layer in 0..config.num_decoder_layers {
            hidden = self.decoder_layer_prefill_on(
                &cb, layer, hidden, encoder_out, seq_len, enc_seq_len, kv_cache,
            )?;
        }

        // Final LN + logits
        let normed = self.layer_norm(
            &cb, &self.model, &hidden,
            "model.decoder.layer_norm.weight",
            "model.decoder.layer_norm.bias",
            seq_len, d, 1e-5,
        )?;

        // Project to vocab (tied weights with embedding)
        let logits = self.linear_bias(
            &cb, &self.model, &normed,
            "model.decoder.embed_tokens.weight",
            "model.decoder.lm_head.bias",
            seq_len, d, config.vocab_size,
        ).unwrap_or_else(|_| {
            // If lm_head.bias doesn't exist, use embedding weight without bias
            let embed_w = gpu_ops::read_weight_f16(&self.model, &self.compute, "model.decoder.embed_tokens.weight")
                .unwrap();
            let dummy_bias = Tensor::zeros(Shape::from([config.vocab_size]), DType::F16).unwrap();
            gpu_ops::linear_tensors_on(
                &self.compute, &self.kernels.common.linear,
                &cb, &normed, &embed_w, &dummy_bias,
                seq_len, d, config.vocab_size,
            )
        });

        cb.commit();
        cb.wait_until_completed();
        Ok(logits)
    }

    /// Single decoder prefill layer on a shared command buffer.
    fn decoder_layer_prefill_on(
        &self, cb: &metal::CommandBufferRef, layer: usize, input: Tensor,
        encoder_out: &Tensor, seq_len: usize, enc_seq_len: usize,
        kv_cache: &mut MoonshineKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let d = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = d / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let prefix = format!("model.decoder.layers.{}", layer);

        // 1. Self-attention with RoPE + causal mask
        let normed = self.layer_norm(
            cb, &self.model, &input,
            &format!("{}.self_attn_layer_norm.weight", prefix),
            &format!("{}.self_attn_layer_norm.bias", prefix),
            seq_len, d, 1e-5,
        )?;

        let q = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.q_proj.weight", prefix),
            &format!("{}.self_attn.q_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let k_proj = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.k_proj.weight", prefix),
            &format!("{}.self_attn.k_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let v_proj = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.v_proj.weight", prefix),
            &format!("{}.self_attn.v_proj.bias", prefix),
            seq_len, d, d,
        )?;

        // Apply RoPE to Q, K
        let q = q.reshape([seq_len, num_heads, head_dim])?;
        let k = k_proj.reshape([seq_len, num_heads, head_dim])?;
        let v = v_proj.reshape([seq_len, num_heads, head_dim])?;
        self.apply_rope_batch_on(cb, &q, seq_len, num_heads, head_dim);
        self.apply_rope_batch_on(cb, &k, seq_len, num_heads, head_dim);

        // Blit K/V to self-attention cache
        let stride_row = num_heads * head_dim * 2; // f16 = 2 bytes
        let copy_size = (seq_len * stride_row) as u64;
        {
            let blit = cb.new_blit_command_encoder();
            if let (Some(src_ptr), Some(dst_ptr)) = (k.device_ptr(), kv_cache.self_k[layer].device_ptr()) {
                let src = unsafe { BorrowedMetalBuffer::from_device_ptr(src_ptr) };
                let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dst_ptr) };
                blit.copy_from_buffer(src.as_ref(), k.byte_offset() as u64, dst.as_ref(), 0, copy_size);
            }
            if let (Some(src_ptr), Some(dst_ptr)) = (v.device_ptr(), kv_cache.self_v[layer].device_ptr()) {
                let src = unsafe { BorrowedMetalBuffer::from_device_ptr(src_ptr) };
                let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dst_ptr) };
                blit.copy_from_buffer(src.as_ref(), v.byte_offset() as u64, dst.as_ref(), 0, copy_size);
            }
            blit.end_encoding();
        }
        if layer == 0 {
            kv_cache.seq_len = seq_len;
        }

        // Causal self-attention via gqa_attention_f16 (has built-in causal mask)
        let attn_out = self.prefill_causal_attention_on(
            cb, &q, &k, &v, scale, seq_len, num_heads, head_dim,
        );
        let attn_flat = attn_out.reshape([seq_len, d])?;

        let sa_out = self.linear_bias(
            cb, &self.model, &attn_flat,
            &format!("{}.self_attn.out_proj.weight", prefix),
            &format!("{}.self_attn.out_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let h = self.add(cb, &input, &sa_out);

        // 2. Cross-attention to encoder output
        let normed2 = self.layer_norm(
            cb, &self.model, &h,
            &format!("{}.encoder_attn_layer_norm.weight", prefix),
            &format!("{}.encoder_attn_layer_norm.bias", prefix),
            seq_len, d, 1e-5,
        )?;

        let cross_q = self.linear_bias(
            cb, &self.model, &normed2,
            &format!("{}.encoder_attn.q_proj.weight", prefix),
            &format!("{}.encoder_attn.q_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let cross_k = self.linear_bias(
            cb, &self.model, encoder_out,
            &format!("{}.encoder_attn.k_proj.weight", prefix),
            &format!("{}.encoder_attn.k_proj.bias", prefix),
            enc_seq_len, d, d,
        )?;
        let cross_v = self.linear_bias(
            cb, &self.model, encoder_out,
            &format!("{}.encoder_attn.v_proj.weight", prefix),
            &format!("{}.encoder_attn.v_proj.bias", prefix),
            enc_seq_len, d, d,
        )?;

        // Pre-transpose cross K/V to HSD for fast decode
        let cross_k_shaped = cross_k.reshape([enc_seq_len, num_heads, head_dim])?;
        let cross_v_shaped = cross_v.reshape([enc_seq_len, num_heads, head_dim])?;
        let device_id = self.compute.device().info().id;
        let k_hsd = Tensor::empty(Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
        let v_hsd = Tensor::empty(Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(cb, &cross_k_shaped, &k_hsd, enc_seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd(cb, &cross_v_shaped, &v_hsd, enc_seq_len, num_heads, head_dim);
        kv_cache.set_cross_hsd(layer, k_hsd, v_hsd);

        // Cross-attention (non-causal)
        let cross_q_shaped = cross_q.reshape([seq_len, num_heads, head_dim])?;
        let cross_attn_out = self.batched_attention(
            cb, &cross_q_shaped, &cross_k_shaped, &cross_v_shaped,
            seq_len, enc_seq_len, num_heads, head_dim, scale,
        )?;

        let cross_out = self.linear_bias(
            cb, &self.model, &cross_attn_out,
            &format!("{}.encoder_attn.out_proj.weight", prefix),
            &format!("{}.encoder_attn.out_proj.bias", prefix),
            seq_len, d, d,
        )?;
        let h = self.add(cb, &h, &cross_out);

        // 3. FFN with SiLU activation (decoder uses SiLU, not GELU)
        let normed3 = self.layer_norm(
            cb, &self.model, &h,
            &format!("{}.final_layer_norm.weight", prefix),
            &format!("{}.final_layer_norm.bias", prefix),
            seq_len, d, 1e-5,
        )?;

        let fc1 = self.linear_bias(
            cb, &self.model, &normed3,
            &format!("{}.mlp.fc1.weight", prefix),
            &format!("{}.mlp.fc1.bias", prefix),
            seq_len, d, config.intermediate_size,
        )?;
        let activated = self.activation(cb, &self.kernels.silu, &fc1);
        let fc2 = self.linear_bias(
            cb, &self.model, &activated,
            &format!("{}.mlp.fc2.weight", prefix),
            &format!("{}.mlp.fc2.bias", prefix),
            seq_len, config.intermediate_size, d,
        )?;

        Ok(self.add(cb, &h, &fc2))
    }

    /// Decode a single new token using KV cache.
    fn decode_step(
        &self, token: u32, kv_cache: &mut MoonshineKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let d = config.hidden_size;

        let cb = self.compute.new_command_buffer();

        // Embed single token
        let mut hidden = self.embed_tokens_on(&cb, &[token], d);

        // All decoder layers on same CB
        for layer in 0..config.num_decoder_layers {
            hidden = self.decoder_layer_step_on(&cb, layer, hidden, kv_cache)?;
        }

        // Final LN + logits
        let normed = self.layer_norm(
            &cb, &self.model, &hidden,
            "model.decoder.layer_norm.weight",
            "model.decoder.layer_norm.bias",
            1, d, 1e-5,
        )?;

        let logits = self.linear_bias(
            &cb, &self.model, &normed,
            "model.decoder.embed_tokens.weight",
            "model.decoder.lm_head.bias",
            1, d, config.vocab_size,
        ).unwrap_or_else(|_| {
            let embed_w = gpu_ops::read_weight_f16(&self.model, &self.compute, "model.decoder.embed_tokens.weight")
                .unwrap();
            let dummy_bias = Tensor::zeros(Shape::from([config.vocab_size]), DType::F16).unwrap();
            gpu_ops::linear_tensors_on(
                &self.compute, &self.kernels.common.linear,
                &cb, &normed, &embed_w, &dummy_bias,
                1, d, config.vocab_size,
            )
        });

        cb.commit();
        cb.wait_until_completed();
        Ok(logits)
    }

    /// Single decoder layer for autoregressive step (seq_len=1).
    fn decoder_layer_step_on(
        &self, cb: &metal::CommandBufferRef, layer: usize, input: Tensor,
        kv_cache: &mut MoonshineKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let d = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = d / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_pos = kv_cache.seq_len;
        let prefix = format!("model.decoder.layers.{}", layer);

        // --- Self-attention ---
        let normed = self.layer_norm(
            cb, &self.model, &input,
            &format!("{}.self_attn_layer_norm.weight", prefix),
            &format!("{}.self_attn_layer_norm.bias", prefix),
            1, d, 1e-5,
        )?;

        let q = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.q_proj.weight", prefix),
            &format!("{}.self_attn.q_proj.bias", prefix),
            1, d, d,
        )?;
        let k_new = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.k_proj.weight", prefix),
            &format!("{}.self_attn.k_proj.bias", prefix),
            1, d, d,
        )?;
        let v_new = self.linear_bias(
            cb, &self.model, &normed,
            &format!("{}.self_attn.v_proj.weight", prefix),
            &format!("{}.self_attn.v_proj.bias", prefix),
            1, d, d,
        )?;

        // Apply RoPE to Q, K (single position)
        let q_r = q.reshape([num_heads, head_dim])?;
        let k_r = k_new.reshape([num_heads, head_dim])?;
        self.apply_rope_single_on(cb, &q_r, seq_pos, num_heads, head_dim);
        self.apply_rope_single_on(cb, &k_r, seq_pos, num_heads, head_dim);

        // Blit K/V to cache
        let k_new_r = k_r.reshape([1, num_heads, head_dim])?;
        let v_new_r = v_new.reshape([1, num_heads, head_dim])?;
        let stride_row = num_heads * head_dim * 2;
        let dst_offset = (seq_pos * stride_row) as u64;
        let copy_size = stride_row as u64;
        {
            let blit = cb.new_blit_command_encoder();
            if let (Some(sp), Some(dp)) = (k_new_r.device_ptr(), kv_cache.self_k[layer].device_ptr()) {
                let src = unsafe { BorrowedMetalBuffer::from_device_ptr(sp) };
                let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dp) };
                blit.copy_from_buffer(src.as_ref(), k_new_r.byte_offset() as u64, dst.as_ref(), dst_offset, copy_size);
            }
            if let (Some(sp), Some(dp)) = (v_new_r.device_ptr(), kv_cache.self_v[layer].device_ptr()) {
                let src = unsafe { BorrowedMetalBuffer::from_device_ptr(sp) };
                let dst = unsafe { BorrowedMetalBuffer::from_device_ptr(dp) };
                blit.copy_from_buffer(src.as_ref(), v_new_r.byte_offset() as u64, dst.as_ref(), dst_offset, copy_size);
            }
            blit.end_encoding();
        }
        if layer == 0 {
            kv_cache.seq_len = seq_pos + 1;
        }

        // Autoregressive self-attention: single Q against KV cache
        let self_attn_out = self.autoregressive_attention_on(
            cb, &q_r, &kv_cache.self_k[layer], &kv_cache.self_v[layer],
            scale, seq_pos, num_heads, head_dim,
        );
        let sa_flat = self_attn_out.reshape([1, d])?;
        let sa_out = self.linear_bias(
            cb, &self.model, &sa_flat,
            &format!("{}.self_attn.out_proj.weight", prefix),
            &format!("{}.self_attn.out_proj.bias", prefix),
            1, d, d,
        )?;
        let h = self.add(cb, &input, &sa_out);

        // --- Cross-attention (uses pre-transposed HSD K/V from prefill) ---
        let normed2 = self.layer_norm(
            cb, &self.model, &h,
            &format!("{}.encoder_attn_layer_norm.weight", prefix),
            &format!("{}.encoder_attn_layer_norm.bias", prefix),
            1, d, 1e-5,
        )?;

        let cross_q = self.linear_bias(
            cb, &self.model, &normed2,
            &format!("{}.encoder_attn.q_proj.weight", prefix),
            &format!("{}.encoder_attn.q_proj.bias", prefix),
            1, d, d,
        )?;

        let enc_seq_len = kv_cache.cross_k_hsd[layer].shape().dim(1).unwrap_or(1);
        let cross_attn_out = self.cross_attention_decode_on(
            cb, &cross_q,
            &kv_cache.cross_k_hsd[layer], &kv_cache.cross_v_hsd[layer],
            num_heads, head_dim, enc_seq_len, scale,
        )?;
        let cross_out = self.linear_bias(
            cb, &self.model, &cross_attn_out,
            &format!("{}.encoder_attn.out_proj.weight", prefix),
            &format!("{}.encoder_attn.out_proj.bias", prefix),
            1, d, d,
        )?;
        let h = self.add(cb, &h, &cross_out);

        // --- FFN with SiLU ---
        let normed3 = self.layer_norm(
            cb, &self.model, &h,
            &format!("{}.final_layer_norm.weight", prefix),
            &format!("{}.final_layer_norm.bias", prefix),
            1, d, 1e-5,
        )?;

        let fc1 = self.linear_bias(
            cb, &self.model, &normed3,
            &format!("{}.mlp.fc1.weight", prefix),
            &format!("{}.mlp.fc1.bias", prefix),
            1, d, config.intermediate_size,
        )?;
        let activated = self.activation(cb, &self.kernels.silu, &fc1);
        let fc2 = self.linear_bias(
            cb, &self.model, &activated,
            &format!("{}.mlp.fc2.weight", prefix),
            &format!("{}.mlp.fc2.bias", prefix),
            1, config.intermediate_size, d,
        )?;

        Ok(self.add(cb, &h, &fc2))
    }

    // ========================= ATTENTION HELPERS =========================

    /// Causal prefill attention via gqa_attention_f16.
    /// Q/K/V: [seq_len, num_heads, head_dim]
    fn prefill_causal_attention_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k: &Tensor, v: &Tensor,
        scale: f32, seq_len: usize, num_heads: usize, head_dim: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;
        let output_size = seq_len * num_heads * head_dim * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch(
            cb, &self.kernels.gqa_attention,
            (num_heads, seq_len, 1), (1, 1, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, q);
                gpu_ops::set_tensor_buffer(encoder, 1, k);
                gpu_ops::set_tensor_buffer(encoder, 2, v);
                encoder.set_buffer(3, Some(&output_buffer), 0);

                let sl = seq_len as u32;
                let nq = num_heads as u32;
                let nkv = num_heads as u32;
                let hd = head_dim as u32;

                encoder.set_bytes(4, 4, &sl as *const u32 as *const _);
                encoder.set_bytes(5, 4, &nq as *const u32 as *const _);
                encoder.set_bytes(6, 4, &nkv as *const u32 as *const _);
                encoder.set_bytes(7, 4, &hd as *const u32 as *const _);
                encoder.set_bytes(8, 4, &scale as *const f32 as *const _);
            },
        );

        Tensor::from_metal_buffer(
            output_buffer, Shape::from([seq_len, num_heads, head_dim]),
            DType::F16, device_id,
        )
    }

    /// Autoregressive attention: single-token Q against KV cache.
    /// Q: [num_heads, head_dim], K/V_cache: [max_seq, num_heads, head_dim]
    fn autoregressive_attention_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k_cache: &Tensor, v_cache: &Tensor,
        scale: f32, seq_pos: usize, num_heads: usize, head_dim: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;
        let output_size = num_heads * head_dim * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch_1d(
            cb, &self.kernels.autoregressive_attention, num_heads,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, q);
                gpu_ops::set_tensor_buffer(encoder, 1, k_cache);
                gpu_ops::set_tensor_buffer(encoder, 2, v_cache);
                encoder.set_buffer(3, Some(&output_buffer), 0);

                let sp = seq_pos as u32;
                let nq = num_heads as u32;
                let nkv = num_heads as u32;
                let hd = head_dim as u32;

                encoder.set_bytes(4, 4, &sp as *const u32 as *const _);
                encoder.set_bytes(5, 4, &nq as *const u32 as *const _);
                encoder.set_bytes(6, 4, &nkv as *const u32 as *const _);
                encoder.set_bytes(7, 4, &hd as *const u32 as *const _);
                encoder.set_bytes(8, 4, &scale as *const f32 as *const _);
            },
        );

        Tensor::from_metal_buffer(
            output_buffer, Shape::from([num_heads, head_dim]),
            DType::F16, device_id,
        )
    }

    /// Cross-attention decode with pre-transposed K/V (HSD format).
    /// Q: [1, d_model]. K_hsd/V_hsd: [H, S, D] (pre-transposed during prefill).
    fn cross_attention_decode_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k_hsd: &Tensor, v_hsd: &Tensor,
        num_heads: usize, head_dim: usize, kv_seq_len: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let q = q.reshape([1, num_heads, head_dim])?;
        let device_id = self.compute.device().info().id;

        // Only transpose Q (tiny: [1, H, D] -> [H, 1, D])
        let q_t = Tensor::empty(Shape::from([num_heads, 1, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(cb, &q, &q_t, 1, num_heads, head_dim);

        // Scores = Q_t @ K_hsd^T -> [H, 1, kv_seq]
        let scores = self.batched_qk(cb, &q_t, k_hsd, num_heads, 1, kv_seq_len, head_dim);

        // Scaled softmax
        self.row_softmax(cb, &scores, num_heads, kv_seq_len, scale);

        // Output = Scores @ V_hsd -> [H, 1, head_dim]
        let output_t = self.batched_sv(cb, &scores, v_hsd, num_heads, 1, kv_seq_len, head_dim)?;

        // Transpose [H, 1, D] -> [1, H, D]
        let output = Tensor::empty(Shape::from([1, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(cb, &output_t, &output, 1, num_heads, head_dim);

        output.reshape([1, num_heads * head_dim])
    }

    // ========================= ROPE HELPERS =========================

    /// Apply RoPE in-place to a batch of positions: tensor [seq_len, num_heads, head_dim].
    /// Single GPU dispatch via rope_batched_f16 (replaces N per-position dispatches).
    fn apply_rope_batch_on(
        &self, cb: &metal::CommandBufferRef,
        tensor: &Tensor, seq_len: usize, num_heads: usize, head_dim: usize,
    ) {
        let theta = self.config.rope_theta;
        gpu_ops::rope_batched_on(
            &self.compute, &self.kernels.rope_batched, cb,
            tensor, theta, seq_len, num_heads, head_dim, 0,
        );
    }

    /// Apply RoPE in-place to a single position: tensor [num_heads, head_dim].
    fn apply_rope_single_on(
        &self, cb: &metal::CommandBufferRef,
        tensor: &Tensor, pos: usize, num_heads: usize, head_dim: usize,
    ) {
        let theta = self.config.rope_theta;

        self.compute.dispatch(
            cb, &self.kernels.rope_single,
            (head_dim / 2, num_heads, 1), (1, 1, 1),
            |encoder| {
                if let Some(ptr) = tensor.device_ptr() {
                    let b = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
                    encoder.set_buffer(0, Some(b.as_ref()), tensor.byte_offset() as u64);
                }
                let pos_u32 = pos as u32;
                let nh = num_heads as u32;
                let hd = head_dim as u32;
                encoder.set_bytes(1, 4, &pos_u32 as *const u32 as *const _);
                encoder.set_bytes(2, 4, &nh as *const u32 as *const _);
                encoder.set_bytes(3, 4, &hd as *const u32 as *const _);
                encoder.set_bytes(4, 4, &theta as *const f32 as *const _);
            },
        );
    }

    // ========================= EMBEDDING HELPER =========================

    /// Embed token IDs into hidden vectors. Returns [seq_len, d_model].
    fn embed_tokens_on(
        &self, cb: &metal::CommandBufferRef, token_ids: &[u32], d_model: usize,
    ) -> Tensor {
        let seq_len = token_ids.len();
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;
        let output_size = seq_len * d_model * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let id_buf = device.new_buffer_with_data(
            token_ids.as_ptr() as *const _,
            (seq_len * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let embed_w = self.model.read().get_weight("model.decoder.embed_tokens.weight")
            .expect("embed_tokens weight not found");

        self.compute.dispatch_1d(
            cb, &self.kernels.embedding, seq_len,
            |encoder| {
                encoder.set_buffer(0, Some(embed_w.buffer()), 0);
                encoder.set_buffer(1, Some(&id_buf), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);

                let vocab_size_u32 = self.config.vocab_size as u32;
                let hidden_size_u32 = d_model as u32;
                let seq_len_u32 = seq_len as u32;
                encoder.set_bytes(3, 4, &vocab_size_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &hidden_size_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &seq_len_u32 as *const u32 as *const _);
            },
        );

        Tensor::from_metal_buffer(output_buffer, Shape::from([seq_len, d_model]), DType::F16, device_id)
    }

    // ========================= ARGMAX =========================

    /// CPU argmax for greedy decoding. logits: [seq_len, vocab_size] -- takes last row.
    fn argmax_cpu(&self, logits: &Tensor, vocab_size: usize) -> Result<u32> {
        let data: Vec<half::f16> = logits.to_vec()?;
        // Take the last row (last vocab_size elements)
        let start = data.len().saturating_sub(vocab_size);
        let last_row = &data[start..];
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0u32;
        for (i, &v) in last_row.iter().enumerate() {
            let val = v.to_f32();
            if val > max_val {
                max_val = val;
                max_idx = i as u32;
            }
        }
        Ok(max_idx)
    }
}

// ========================= CPU HELPERS =========================

/// Conv1d on CPU: input [c_in, l_in], weight [c_out, c_in, k], bias [c_out].
/// Returns [c_out, l_out] where l_out = (l_in + 2*padding - k) / stride + 1.
fn conv1d_cpu(
    input: &[f32], c_in: usize, l_in: usize,
    weight: &[f32], bias: &[f32],
    c_out: usize, kernel_size: usize, stride: usize, padding: usize,
) -> Vec<f32> {
    let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
    let mut output = vec![0.0f32; c_out * l_out];

    for co in 0..c_out {
        for lo in 0..l_out {
            let mut sum = if co < bias.len() { bias[co] } else { 0.0 };
            for ci in 0..c_in {
                for k in 0..kernel_size {
                    let pos = lo * stride + k;
                    if pos >= padding && pos < l_in + padding {
                        let in_pos = pos - padding;
                        let in_idx = ci * l_in + in_pos;
                        let w_idx = (co * c_in + ci) * kernel_size + k;
                        if in_idx < input.len() && w_idx < weight.len() {
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            output[co * l_out + lo] = sum;
        }
    }
    output
}

/// GELU activation on CPU.
fn gelu_cpu(input: &[f32]) -> Vec<f32> {
    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let coeff: f32 = 0.044715;
    input.iter().map(|&x| {
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + coeff * x3);
        0.5 * x * (1.0 + inner.tanh())
    }).collect()
}

/// Simple linear resampling from input sample rate to 16kHz.
fn resample_to_16k(audio: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == 16000 || audio.is_empty() {
        return audio.to_vec();
    }
    let ratio = 16000.0 / src_rate as f64;
    let out_len = (audio.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let s0 = audio[idx.min(audio.len() - 1)];
        let s1 = audio[(idx + 1).min(audio.len() - 1)];
        output.push(s0 + (s1 - s0) * frac as f32);
    }
    output
}
