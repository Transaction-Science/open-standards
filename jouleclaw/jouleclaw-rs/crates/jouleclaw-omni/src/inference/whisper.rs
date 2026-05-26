//! Whisper speech recognition pipeline.
//!
//! Implements OpenAI Whisper encoder-decoder transformer on Metal GPU.
//! Supports whisper-small (241M params, 12 encoder + 12 decoder layers).
//!
//! Architecture:
//!   Audio → Mel Spectrogram → Conv1d → Encoder (12 layers) → Decoder (12 layers, cross-attention) → Text

use super::model::Model;
use super::tokenizer::HfTokenizer;
use crate::core::{Error, Result};
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, LazyTensor, BorrowedMetalBuffer};

/// Helper to set a Metal buffer from a Tensor's device_ptr on the encoder.
/// Respects the tensor's byte offset (critical for sliced tensors).
#[cfg(feature = "metal")]
fn set_tensor_buffer(encoder: &metal::ComputeCommandEncoderRef, index: u64, tensor: &Tensor) {
    if let Some(ptr) = tensor.device_ptr() {
        let b = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
        encoder.set_buffer(index, Some(b.as_ref()), tensor.byte_offset() as u64);
    }
}

/// Helper to set a Metal buffer from a LazyTensor on the encoder.
#[cfg(feature = "metal")]
fn set_lazy_buffer(encoder: &metal::ComputeCommandEncoderRef, index: u64, lt: &LazyTensor) {
    encoder.set_buffer(index, Some(lt.buffer()), 0);
}

/// Whisper model configuration (from config.json).
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        // Whisper-small defaults
        Self {
            d_model: 768,
            encoder_layers: 12,
            decoder_layers: 12,
            encoder_attention_heads: 12,
            decoder_attention_heads: 12,
            encoder_ffn_dim: 3072,
            decoder_ffn_dim: 3072,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            vocab_size: 51865,
            decoder_start_token_id: 50258,
            eos_token_id: 50257,
        }
    }
}

impl WhisperConfig {
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path).map_err(|e|
            Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str).map_err(|e|
            Error::internal(format!("failed to parse config: {}", e)))?;

        let mut config = Self::default();
        if let Some(v) = json.get("d_model").and_then(|v| v.as_u64()) { config.d_model = v as usize; }
        if let Some(v) = json.get("encoder_layers").and_then(|v| v.as_u64()) { config.encoder_layers = v as usize; }
        if let Some(v) = json.get("decoder_layers").and_then(|v| v.as_u64()) { config.decoder_layers = v as usize; }
        if let Some(v) = json.get("encoder_attention_heads").and_then(|v| v.as_u64()) { config.encoder_attention_heads = v as usize; }
        if let Some(v) = json.get("decoder_attention_heads").and_then(|v| v.as_u64()) { config.decoder_attention_heads = v as usize; }
        if let Some(v) = json.get("encoder_ffn_dim").and_then(|v| v.as_u64()) { config.encoder_ffn_dim = v as usize; }
        if let Some(v) = json.get("decoder_ffn_dim").and_then(|v| v.as_u64()) { config.decoder_ffn_dim = v as usize; }
        if let Some(v) = json.get("num_mel_bins").and_then(|v| v.as_u64()) { config.num_mel_bins = v as usize; }
        if let Some(v) = json.get("max_source_positions").and_then(|v| v.as_u64()) { config.max_source_positions = v as usize; }
        if let Some(v) = json.get("max_target_positions").and_then(|v| v.as_u64()) { config.max_target_positions = v as usize; }
        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) { config.vocab_size = v as usize; }
        if let Some(v) = json.get("decoder_start_token_id").and_then(|v| v.as_u64()) { config.decoder_start_token_id = v as u32; }
        if let Some(v) = json.get("eos_token_id").and_then(|v| v.as_u64()) { config.eos_token_id = v as u32; }
        Ok(config)
    }
}

/// Compiled Metal kernels for Whisper inference.
#[cfg(feature = "metal")]
struct WhisperKernels {
    linear: Arc<ComputePipeline>,
    layer_norm: Arc<ComputePipeline>,
    gelu: Arc<ComputePipeline>,
    attention: Arc<ComputePipeline>,
    causal_attention: Arc<ComputePipeline>,
    gqa_attention: Arc<ComputePipeline>,
    autoregressive_attention: Arc<ComputePipeline>,
    add: Arc<ComputePipeline>,
    conv1d: Arc<ComputePipeline>,
    embedding: Arc<ComputePipeline>,
    // Batched matmul attention kernels (encoder optimization)
    transpose_shd_to_hsd: Arc<ComputePipeline>,
    transpose_hsd_to_shd: Arc<ComputePipeline>,
    batched_linear: Arc<ComputePipeline>,
    batched_matmul_nn: Arc<ComputePipeline>,
    row_softmax_scale: Arc<ComputePipeline>,
    nchw_to_nhwc: Arc<ComputePipeline>,
}

/// KV cache for Whisper decoder.
/// Caches self-attention K/V (growing per token) and cross-attention K/V (static after prefill).
#[cfg(feature = "metal")]
struct WhisperKVCache {
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
    /// Whether cross-attention K/V have been populated
    cross_cached: bool,
    num_heads: usize,
    head_dim: usize,
}

#[cfg(feature = "metal")]
impl WhisperKVCache {
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
            cross_cached: false,
            num_heads,
            head_dim,
        })
    }

    /// Store pre-transposed cross K/V in HSD [num_heads, enc_seq_len, head_dim] format.
    /// Called once during prefill to avoid re-transposing every decode step.
    fn set_cross_hsd(&mut self, layer: usize, k_hsd: Tensor, v_hsd: Tensor) {
        while self.cross_k_hsd.len() <= layer {
            let placeholder = Tensor::zeros_on(
                Shape::from([1]), DType::F16, crate::hal::DeviceId::cpu(),
            ).unwrap();
            self.cross_k_hsd.push(placeholder.clone());
            self.cross_v_hsd.push(placeholder);
        }
        self.cross_k_hsd[layer] = k_hsd;
        self.cross_v_hsd[layer] = v_hsd;
    }
}

/// Whisper speech recognition pipeline.
pub struct WhisperPipeline {
    model: Arc<Model>,
    config: WhisperConfig,
    tokenizer: Option<Arc<HfTokenizer>>,
    #[cfg(feature = "metal")]
    compute: Arc<MetalCompute>,
    #[cfg(feature = "metal")]
    kernels: WhisperKernels,
}

impl WhisperPipeline {
    #[cfg(feature = "metal")]
    pub fn new(model: Arc<Model>, config: WhisperConfig, device: Arc<MetalDevice>) -> Result<Self> {
        use crate::hal::metal::shader::sources;

        let compute = Arc::new(MetalCompute::new(device));

        let kernels = WhisperKernels {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            attention: compute.compile_pipeline("attention", sources::ATTENTION, "attention_f16")?,
            causal_attention: compute.compile_pipeline("causal_attention", sources::CAUSAL_ATTENTION, "causal_attention_f16")?,
            gqa_attention: compute.compile_pipeline("gqa_attention", sources::GQA_ATTENTION, "gqa_attention_f16")?,
            autoregressive_attention: compute.compile_pipeline("autoregressive_attention", sources::AUTOREGRESSIVE_ATTENTION, "autoregressive_attention_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            transpose_shd_to_hsd: compute.compile_pipeline("transpose_shd_to_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_to_shd: compute.compile_pipeline("transpose_hsd_to_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
            batched_linear: compute.compile_pipeline("batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax_scale: compute.compile_pipeline("row_softmax_scale", sources::LINEAR, "row_softmax_scale_f16")?,
            nchw_to_nhwc: compute.compile_pipeline("nchw_to_nhwc", sources::TRANSPOSE, "nchw_to_nhwc_f16")?,
        };

        Ok(Self {
            model,
            config,
            tokenizer: None,
            compute,
            kernels,
        })
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<HfTokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Transcribe a mel spectrogram to text.
    /// mel: [num_mel_bins, num_frames] (e.g., [80, 3000])
    #[cfg(feature = "metal")]
    pub fn transcribe(&self, mel: &Tensor) -> Result<String> {
        let config = &self.config;

        // 1. Encoder
        let encoder_out = self.encode(mel)?;

        // 2. Initialize KV cache
        let num_heads = config.decoder_attention_heads;
        let head_dim = config.d_model / num_heads;
        let mut kv_cache = WhisperKVCache::new(
            config.decoder_layers, num_heads, head_dim,
            config.max_target_positions, self.compute.device().info().id,
        )?;

        // 3. Forced prefix tokens
        let forced_prefix: Vec<u32> = vec![
            config.decoder_start_token_id,
            50259, // <|en|>
            50359, // <|transcribe|>
            50363, // <|notimestamps|>
        ];

        // 4. Prefill: process all 4 prefix tokens at once, populate caches
        let logits = self.decode_prefill(&forced_prefix, &encoder_out, &mut kv_cache)?;
        let mut next_token = self.argmax_cpu(&logits, config.vocab_size)?;

        let mut tokens = forced_prefix;

        // 5. Decode loop: one token at a time
        for _step in 0..config.max_target_positions {
            if next_token == config.eos_token_id {
                break;
            }
            tokens.push(next_token);
            let logits = self.decode_step(next_token, &mut kv_cache)?;
            next_token = self.argmax_cpu(&logits, config.vocab_size)?;
        }

        // Decode tokens (skip forced prefix)
        let output_tokens = &tokens[4..];
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.decode(output_tokens)
        } else {
            Ok(output_tokens.iter().map(|t| format!("[{}]", t)).collect())
        }
    }

    // ========================= ENCODER =========================

    /// Encoder: Conv1d → GELU → Conv1d → GELU → GPU transpose → +pos → 12× encoder_layer → LN
    /// Uses per-layer CBs for encoder layers (large intermediate tensors benefit from inter-layer
    /// memory reclamation), but merges conv stem + transpose + pos_add into a single CB.
    #[cfg(feature = "metal")]
    fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;

        // Pre-load positional embeddings (CPU side, before GPU commands)
        let pos_embed_lazy = self.get_weight("model.encoder.embed_positions.weight")?;
        let pos_embed = self.lazy_to_tensor(pos_embed_lazy)?;

        // --- CB1: Conv stem + GPU transpose + positional embedding ---
        let cb = self.compute.new_command_buffer();

        // Conv1d(80 → d_model, k=3, padding=1)
        let conv1_w = self.get_weight("model.encoder.conv1.weight")?;
        let conv1_b = self.get_weight("model.encoder.conv1.bias")?;
        let h = self.conv1d_on(&cb, mel, conv1_w, conv1_b,
                              config.num_mel_bins, config.d_model, mel.shape().dim(1).unwrap_or(3000), 3, 1, 1);
        let h = self.gelu_on(&cb, &h);

        // Conv1d(d_model → d_model, k=3, stride=2, padding=1)
        let conv2_w = self.get_weight("model.encoder.conv2.weight")?;
        let conv2_b = self.get_weight("model.encoder.conv2.bias")?;
        let l_after_conv1 = h.shape().dim(1).unwrap_or(3000);
        let h = self.conv1d_on(&cb, &h, conv2_w, conv2_b,
                              config.d_model, config.d_model, l_after_conv1, 3, 2, 1);
        let h = self.gelu_on(&cb, &h);

        // GPU transpose [d_model, seq_len] → [seq_len, d_model]
        let seq_len = h.shape().dim(1).unwrap_or(1500);
        let transposed_size = config.d_model * seq_len * 2;
        let transposed_buffer = device.new_buffer(transposed_size as u64, metal::MTLResourceOptions::StorageModeShared);
        {
            let c_u32 = config.d_model as u32;
            let hw_u32 = seq_len as u32;
            let tg = 16usize;
            self.compute.dispatch(
                &cb, &self.kernels.nchw_to_nhwc,
                ((seq_len + tg - 1) / tg, (config.d_model + tg - 1) / tg, 1), (tg, tg, 1),
                |encoder| {
                    set_tensor_buffer(encoder, 0, &h);
                    encoder.set_buffer(1, Some(&transposed_buffer), 0);
                    encoder.set_bytes(2, 4, &c_u32 as *const u32 as *const _);
                    encoder.set_bytes(3, 4, &hw_u32 as *const u32 as *const _);
                },
            );
        }
        let h = Tensor::from_metal_buffer(transposed_buffer, Shape::from([seq_len, config.d_model]), DType::F16, device_id);

        // Add positional embeddings
        let pos_embed = if seq_len < config.max_source_positions {
            pos_embed.slice(0, 0, seq_len)?
        } else {
            pos_embed
        };
        let hidden = self.add_on(&cb, &h, &pos_embed);
        cb.commit();
        cb.wait_until_completed();

        // --- Per-layer CBs: encoder layers (large intermediates benefit from inter-layer reclamation) ---
        let mut hidden = hidden;
        for layer in 0..config.encoder_layers {
            let cb = self.compute.new_command_buffer();
            hidden = self.encoder_layer_on(&cb, layer, hidden, seq_len)?;
            cb.commit();
            cb.wait_until_completed();
        }

        // --- Final layer norm (merged into last layer for minimal overhead) ---
        let ln_w = self.get_weight("model.encoder.layer_norm.weight")?;
        let ln_b = self.get_weight("model.encoder.layer_norm.bias")?;
        let cb = self.compute.new_command_buffer();
        let out = self.layer_norm_on(&cb, &hidden, ln_w, ln_b, seq_len, config.d_model);
        cb.commit();
        cb.wait_until_completed();

        Ok(out)
    }

    /// Single encoder transformer layer on a shared command buffer (no commit/wait).
    #[cfg(feature = "metal")]
    fn encoder_layer_on(&self, cb: &metal::CommandBufferRef, layer: usize, input: Tensor, seq_len: usize) -> Result<Tensor> {
        let config = &self.config;
        let prefix = format!("model.encoder.layers.{}", layer);

        // Pre-attention LayerNorm
        let ln1_w = self.get_weight(&format!("{}.self_attn_layer_norm.weight", prefix))?;
        let ln1_b = self.get_weight(&format!("{}.self_attn_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &input, ln1_w, ln1_b, seq_len, config.d_model);

        // Self-attention (non-causal for encoder)
        let attn_out = self.self_attention_on(
            cb, &normed, &prefix, "self_attn",
            config.encoder_attention_heads, config.d_model,
            seq_len, seq_len, false,
        )?;

        // Residual
        let h = self.add_on(cb, &input, &attn_out);

        // Pre-FFN LayerNorm
        let ln2_w = self.get_weight(&format!("{}.final_layer_norm.weight", prefix))?;
        let ln2_b = self.get_weight(&format!("{}.final_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln2_w, ln2_b, seq_len, config.d_model);

        // FFN: fc1 → GELU → fc2
        let ffn_out = self.ffn_on(cb, &normed, &prefix, seq_len, config.d_model, config.encoder_ffn_dim)?;

        // Residual
        Ok(self.add_on(cb, &h, &ffn_out))
    }

    // ========================= CACHED DECODER (prefill + step) =========================

    /// Prefill: process all prefix tokens at once, populate KV caches.
    /// Single command buffer for entire prefill: embed + 12 layers (with inline blits) + logits.
    #[cfg(feature = "metal")]
    fn decode_prefill(
        &self, tokens: &[u32], encoder_out: &Tensor, kv_cache: &mut WhisperKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let seq_len = tokens.len();
        let enc_seq_len = encoder_out.shape().dim(0).unwrap_or(1500);

        let pos_lazy = self.get_weight("model.decoder.embed_positions.weight")?;
        let pos_tensor = self.lazy_to_tensor(pos_lazy)?;
        let pos_slice = pos_tensor.slice(0, 0, seq_len)?;

        let cb = self.compute.new_command_buffer();

        // Token + position embedding on shared CB
        let embed_w = self.get_weight("model.decoder.embed_tokens.weight")?;
        let h = self.embed_tokens_on(&cb, tokens, embed_w, config.d_model);
        let mut hidden = self.add_on(&cb, &h, &pos_slice);

        // All 12 decoder layers on same CB (with inline blits for KV cache)
        for layer in 0..config.decoder_layers {
            hidden = self.decoder_layer_prefill_on(
                &cb, layer, hidden, encoder_out, seq_len, enc_seq_len, kv_cache,
            )?;
        }

        // Final LN + logits on same CB
        let ln_w = self.get_weight("model.decoder.layer_norm.weight")?;
        let ln_b = self.get_weight("model.decoder.layer_norm.bias")?;
        let embed_w = self.get_weight("model.decoder.embed_tokens.weight")?;
        let normed = self.layer_norm_on(&cb, &hidden, ln_w, ln_b, seq_len, config.d_model);
        let logits = self.linear_on(&cb, &normed, embed_w, None, seq_len, config.d_model, config.vocab_size);

        cb.commit();
        cb.wait_until_completed();
        Ok(logits)
    }


    /// Single decoder prefill layer on a shared command buffer (no commit/wait).
    /// Inlines blit for self-attention KV cache and stores cross-attention KV references.
    #[cfg(feature = "metal")]
    fn decoder_layer_prefill_on(
        &self, cb: &metal::CommandBufferRef, layer: usize, input: Tensor, encoder_out: &Tensor,
        seq_len: usize, enc_seq_len: usize, kv_cache: &mut WhisperKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let prefix = format!("model.decoder.layers.{}", layer);
        let num_heads = config.decoder_attention_heads;
        let head_dim = config.d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Self-attention: LN + Q/K/V projections
        let ln1_w = self.get_weight(&format!("{}.self_attn_layer_norm.weight", prefix))?;
        let ln1_b = self.get_weight(&format!("{}.self_attn_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &input, ln1_w, ln1_b, seq_len, config.d_model);

        let sa = format!("{}.self_attn", prefix);
        let q_w = self.get_weight(&format!("{}.q_proj.weight", sa))?;
        let q_b = self.get_weight(&format!("{}.q_proj.bias", sa))?;
        let q = self.linear_on(cb, &normed, q_w, Some(q_b), seq_len, config.d_model, config.d_model);
        let k_w = self.get_weight(&format!("{}.k_proj.weight", sa))?;
        let k_proj = self.linear_on(cb, &normed, k_w, None, seq_len, config.d_model, config.d_model);
        let v_w = self.get_weight(&format!("{}.v_proj.weight", sa))?;
        let v_b = self.get_weight(&format!("{}.v_proj.bias", sa))?;
        let v_proj = self.linear_on(cb, &normed, v_w, Some(v_b), seq_len, config.d_model, config.d_model);

        // Reshape for attention and inline blit to self-attention cache
        let q = q.reshape([seq_len, num_heads, head_dim])?;
        let k = k_proj.reshape([seq_len, num_heads, head_dim])?;
        let v = v_proj.reshape([seq_len, num_heads, head_dim])?;

        // Inline blit (was update_self with its own CB)
        let stride_row = num_heads * head_dim * 2;
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

        // Causal self-attention + output proj
        let attn_out = self.prefill_causal_attention_on(cb, &q, &k, &v, scale, seq_len, num_heads, head_dim);
        let attn_flat = attn_out.reshape([seq_len, config.d_model])?;
        let o_w = self.get_weight(&format!("{}.out_proj.weight", sa))?;
        let o_b = self.get_weight(&format!("{}.out_proj.bias", sa))?;
        let sa_out = self.linear_on(cb, &attn_flat, o_w, Some(o_b), seq_len, config.d_model, config.d_model);
        let h = self.add_on(cb, &input, &sa_out);

        // 2. Cross-attention: project Q from decoder, K/V from encoder
        let ln2_w = self.get_weight(&format!("{}.encoder_attn_layer_norm.weight", prefix))?;
        let ln2_b = self.get_weight(&format!("{}.encoder_attn_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln2_w, ln2_b, seq_len, config.d_model);

        let cq_w = self.get_weight(&format!("{}.encoder_attn.q_proj.weight", prefix))?;
        let cq_b = self.get_weight(&format!("{}.encoder_attn.q_proj.bias", prefix))?;
        let cross_q = self.linear_on(cb, &normed, cq_w, Some(cq_b), seq_len, config.d_model, config.d_model);

        let ck_w = self.get_weight(&format!("{}.encoder_attn.k_proj.weight", prefix))?;
        let cross_k = self.linear_on(cb, encoder_out, ck_w, None, enc_seq_len, config.d_model, config.d_model);

        let cv_w = self.get_weight(&format!("{}.encoder_attn.v_proj.weight", prefix))?;
        let cv_b = self.get_weight(&format!("{}.encoder_attn.v_proj.bias", prefix))?;
        let cross_v = self.linear_on(cb, encoder_out, cv_w, Some(cv_b), enc_seq_len, config.d_model, config.d_model);

        // Reshape cross K/V for attention and pre-transpose to HSD for fast decode
        let cross_k_shaped = cross_k.reshape([enc_seq_len, num_heads, head_dim])?;
        let cross_v_shaped = cross_v.reshape([enc_seq_len, num_heads, head_dim])?;
        let device_id = self.compute.device().info().id;
        let k_hsd = Tensor::empty(Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
        let v_hsd = Tensor::empty(Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd_on(cb, &cross_k_shaped, &k_hsd, enc_seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd_on(cb, &cross_v_shaped, &v_hsd, enc_seq_len, num_heads, head_dim);
        kv_cache.set_cross_hsd(layer, k_hsd, v_hsd);

        // Cross-attention + output proj
        let cross_attn_out = self.multi_head_attention_on(
            cb, &cross_q, &cross_k, &cross_v,
            num_heads, head_dim, seq_len, enc_seq_len, scale, false,
        )?;

        let co_w = self.get_weight(&format!("{}.encoder_attn.out_proj.weight", prefix))?;
        let co_b = self.get_weight(&format!("{}.encoder_attn.out_proj.bias", prefix))?;
        let cross_out = self.linear_on(cb, &cross_attn_out, co_w, Some(co_b), seq_len, config.d_model, config.d_model);
        let h = self.add_on(cb, &h, &cross_out);

        // 3. FFN
        let ln3_w = self.get_weight(&format!("{}.final_layer_norm.weight", prefix))?;
        let ln3_b = self.get_weight(&format!("{}.final_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln3_w, ln3_b, seq_len, config.d_model);
        let ffn_out = self.ffn_on(cb, &normed, &prefix, seq_len, config.d_model, config.decoder_ffn_dim)?;
        Ok(self.add_on(cb, &h, &ffn_out))
    }

    /// Decode a single new token using KV cache.
    /// Single command buffer for entire forward pass: embed + 12 layers + logits.
    #[cfg(feature = "metal")]
    fn decode_step(
        &self, token: u32, kv_cache: &mut WhisperKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let pos = kv_cache.seq_len;

        // Embed single token + add position embedding
        let embed_w = self.get_weight("model.decoder.embed_tokens.weight")?;
        let pos_lazy = self.get_weight("model.decoder.embed_positions.weight")?;
        let pos_tensor = self.lazy_to_tensor(pos_lazy)?;
        let pos_slice = pos_tensor.slice(0, pos, pos + 1)?;

        let cb = self.compute.new_command_buffer();
        let h = self.embed_tokens_on(&cb, &[token], embed_w, config.d_model);
        let mut hidden = self.add_on(&cb, &h, &pos_slice);

        // All 12 decoder layers on same CB (compute→blit→compute per layer)
        for layer in 0..config.decoder_layers {
            hidden = self.decoder_layer_step_on(&cb, layer, hidden, kv_cache)?;
        }

        // Final LN + logits on same CB
        let ln_w = self.get_weight("model.decoder.layer_norm.weight")?;
        let ln_b = self.get_weight("model.decoder.layer_norm.bias")?;
        let embed_w = self.get_weight("model.decoder.embed_tokens.weight")?;
        let normed = self.layer_norm_on(&cb, &hidden, ln_w, ln_b, 1, config.d_model);
        let logits = self.linear_on(&cb, &normed, embed_w, None, 1, config.d_model, config.vocab_size);

        cb.commit();
        cb.wait_until_completed();
        Ok(logits)
    }

    /// Single decoder layer on a shared command buffer (no commit/wait).
    /// Same logic as decoder_layer_step but uses an external CB for batching.
    #[cfg(feature = "metal")]
    fn decoder_layer_step_on(
        &self, cb: &metal::CommandBufferRef, layer: usize, input: Tensor, kv_cache: &mut WhisperKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let prefix = format!("model.decoder.layers.{}", layer);
        let num_heads = config.decoder_attention_heads;
        let head_dim = config.d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_pos = kv_cache.seq_len;

        // --- LN + Q/K/V projections ---
        let ln1_w = self.get_weight(&format!("{}.self_attn_layer_norm.weight", prefix))?;
        let ln1_b = self.get_weight(&format!("{}.self_attn_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &input, ln1_w, ln1_b, 1, config.d_model);

        let sa = format!("{}.self_attn", prefix);
        let q_w = self.get_weight(&format!("{}.q_proj.weight", sa))?;
        let q_b = self.get_weight(&format!("{}.q_proj.bias", sa))?;
        let q = self.linear_on(cb, &normed, q_w, Some(q_b), 1, config.d_model, config.d_model);
        let k_w = self.get_weight(&format!("{}.k_proj.weight", sa))?;
        let k_new = self.linear_on(cb, &normed, k_w, None, 1, config.d_model, config.d_model);
        let v_w = self.get_weight(&format!("{}.v_proj.weight", sa))?;
        let v_b = self.get_weight(&format!("{}.v_proj.bias", sa))?;
        let v_new = self.linear_on(cb, &normed, v_w, Some(v_b), 1, config.d_model, config.d_model);

        // --- Blit: K/V cache update ---
        let k_new_r = k_new.reshape([1, num_heads, head_dim])?;
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

        // --- Self-attn + cross-attn + FFN ---
        let q_flat = q.reshape([num_heads, head_dim])?;
        let self_attn_out = self.autoregressive_attention_on(
            cb, &q_flat, &kv_cache.self_k[layer], &kv_cache.self_v[layer],
            scale, seq_pos, num_heads, head_dim,
        );
        let sa_flat = self_attn_out.reshape([1, config.d_model])?;
        let o_w = self.get_weight(&format!("{}.out_proj.weight", sa))?;
        let o_b = self.get_weight(&format!("{}.out_proj.bias", sa))?;
        let sa_out = self.linear_on(cb, &sa_flat, o_w, Some(o_b), 1, config.d_model, config.d_model);
        let h = self.add_on(cb, &input, &sa_out);

        // Cross-attention
        let ln2_w = self.get_weight(&format!("{}.encoder_attn_layer_norm.weight", prefix))?;
        let ln2_b = self.get_weight(&format!("{}.encoder_attn_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln2_w, ln2_b, 1, config.d_model);
        let cq_w = self.get_weight(&format!("{}.encoder_attn.q_proj.weight", prefix))?;
        let cq_b = self.get_weight(&format!("{}.encoder_attn.q_proj.bias", prefix))?;
        let cross_q = self.linear_on(cb, &normed, cq_w, Some(cq_b), 1, config.d_model, config.d_model);

        let enc_seq_len = kv_cache.cross_k_hsd[layer].shape().dim(1).unwrap_or(1500);
        let cross_attn_out = self.cross_attention_decode_on(
            cb, &cross_q, &kv_cache.cross_k_hsd[layer], &kv_cache.cross_v_hsd[layer],
            num_heads, head_dim, enc_seq_len, scale,
        )?;
        let co_w = self.get_weight(&format!("{}.encoder_attn.out_proj.weight", prefix))?;
        let co_b = self.get_weight(&format!("{}.encoder_attn.out_proj.bias", prefix))?;
        let cross_out = self.linear_on(cb, &cross_attn_out, co_w, Some(co_b), 1, config.d_model, config.d_model);
        let h = self.add_on(cb, &h, &cross_out);

        // FFN
        let ln3_w = self.get_weight(&format!("{}.final_layer_norm.weight", prefix))?;
        let ln3_b = self.get_weight(&format!("{}.final_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln3_w, ln3_b, 1, config.d_model);
        let ffn_out = self.ffn_on(cb, &normed, &prefix, 1, config.d_model, config.decoder_ffn_dim)?;
        Ok(self.add_on(cb, &h, &ffn_out))
    }

    // ========================= BUILDING BLOCKS (batched on CB) =========================

    /// Self-attention on a shared command buffer.
    #[cfg(feature = "metal")]
    fn self_attention_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, prefix: &str, attn_name: &str,
        num_heads: usize, d_model: usize,
        q_seq_len: usize, kv_seq_len: usize, causal: bool,
    ) -> Result<Tensor> {
        let head_dim = d_model / num_heads;
        let full_prefix = format!("{}.{}", prefix, attn_name);

        let q_w = self.get_weight(&format!("{}.q_proj.weight", full_prefix))?;
        let q_b = self.get_weight(&format!("{}.q_proj.bias", full_prefix))?;
        let q = self.linear_on(cb, input, q_w, Some(q_b), q_seq_len, d_model, d_model);

        let k_w = self.get_weight(&format!("{}.k_proj.weight", full_prefix))?;
        let k = self.linear_on(cb, input, k_w, None, kv_seq_len, d_model, d_model);

        let v_w = self.get_weight(&format!("{}.v_proj.weight", full_prefix))?;
        let v_b = self.get_weight(&format!("{}.v_proj.bias", full_prefix))?;
        let v = self.linear_on(cb, input, v_w, Some(v_b), kv_seq_len, d_model, d_model);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_out = self.multi_head_attention_on(
            cb, &q, &k, &v, num_heads, head_dim,
            q_seq_len, kv_seq_len, scale, causal,
        )?;

        let o_w = self.get_weight(&format!("{}.out_proj.weight", full_prefix))?;
        let o_b = self.get_weight(&format!("{}.out_proj.bias", full_prefix))?;
        Ok(self.linear_on(cb, &attn_out, o_w, Some(o_b), q_seq_len, d_model, d_model))
    }

    /// Multi-head attention dispatch on a shared command buffer.
    /// Routes to matmul-based path for long non-causal sequences (encoder).
    #[cfg(feature = "metal")]
    fn multi_head_attention_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k: &Tensor, v: &Tensor,
        num_heads: usize, head_dim: usize,
        q_seq_len: usize, kv_seq_len: usize,
        scale: f32, causal: bool,
    ) -> Result<Tensor> {
        // Use batched matmul for non-causal attention with long KV sequences.
        // attention_f16 uses 1 thread per head — catastrophic for large kv_seq_len.
        // Matmul path: 1128 TGs for kv=1499 vs 12 threads for attention_f16 → 94× more parallelism.
        if !causal && kv_seq_len >= 32 {
            return self.multi_head_attention_matmul_on(
                cb, q, k, v, num_heads, head_dim,
                q_seq_len, kv_seq_len, scale,
            );
        }

        let q = q.reshape([q_seq_len, num_heads, head_dim])?;
        let k = k.reshape([kv_seq_len, num_heads, head_dim])?;
        let v = v.reshape([kv_seq_len, num_heads, head_dim])?;

        let device = self.compute.device().raw();
        let output_size = q_seq_len * num_heads * head_dim * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let s_dim: u32 = 1;
        let s_head: u32 = head_dim as u32;
        let s_seq: u32 = (num_heads * head_dim) as u32;
        let s_batch: u32 = (q_seq_len * num_heads * head_dim) as u32;

        let kernel = if causal { &self.kernels.causal_attention } else { &self.kernels.attention };
        let shared_mem_size = kv_seq_len * 4;

        self.compute.dispatch(
            cb, kernel,
            (num_heads, q_seq_len, 1),
            (1, 1, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, &q);
                set_tensor_buffer(encoder, 1, &k);
                set_tensor_buffer(encoder, 2, &v);
                encoder.set_buffer(3, Some(&output_buffer), 0);

                let q_seq_u32 = q_seq_len as u32;
                let head_dim_u32 = head_dim as u32;
                let num_heads_u32 = num_heads as u32;
                let kv_len_u32 = kv_seq_len as u32;

                encoder.set_bytes(4, 4, &q_seq_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &head_dim_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &scale as *const f32 as *const _);
                encoder.set_bytes(7, 4, &num_heads_u32 as *const u32 as *const _);
                encoder.set_bytes(8, 4, &s_batch as *const u32 as *const _);
                encoder.set_bytes(9, 4, &s_head as *const u32 as *const _);
                encoder.set_bytes(10, 4, &s_seq as *const u32 as *const _);
                encoder.set_bytes(11, 4, &s_dim as *const u32 as *const _);

                if !causal {
                    encoder.set_bytes(12, 4, &kv_len_u32 as *const u32 as *const _);
                }

                encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);
            },
        );

        let out = Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([q_seq_len, num_heads, head_dim]),
            DType::F16,
            self.compute.device().info().id,
        );
        out.reshape([q_seq_len, num_heads * head_dim])
    }

    /// FFN on a shared command buffer.
    #[cfg(feature = "metal")]
    fn ffn_on(&self, cb: &metal::CommandBufferRef, input: &Tensor, prefix: &str,
              seq_len: usize, d_model: usize, ffn_dim: usize) -> Result<Tensor> {
        let fc1_w = self.get_weight(&format!("{}.fc1.weight", prefix))?;
        let fc1_b = self.get_weight(&format!("{}.fc1.bias", prefix))?;
        let fc2_w = self.get_weight(&format!("{}.fc2.weight", prefix))?;
        let fc2_b = self.get_weight(&format!("{}.fc2.bias", prefix))?;

        let h = self.linear_on(cb, input, fc1_w, Some(fc1_b), seq_len, d_model, ffn_dim);
        let h = self.gelu_on(cb, &h);
        Ok(self.linear_on(cb, &h, fc2_w, Some(fc2_b), seq_len, ffn_dim, d_model))
    }

    // ========================= CACHED ATTENTION HELPERS =========================

    /// Autoregressive attention: single-token Q against KV cache. Encodes on provided CB.
    /// Q: [num_heads, head_dim], K/V_cache: [max_seq, num_heads, head_dim]
    #[cfg(feature = "metal")]
    fn autoregressive_attention_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k_cache: &Tensor, v_cache: &Tensor,
        scale: f32, seq_pos: usize, num_heads: usize, head_dim: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let output_size = num_heads * head_dim * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch_1d(
            cb, &self.kernels.autoregressive_attention, num_heads,
            |encoder| {
                set_tensor_buffer(encoder, 0, q);
                set_tensor_buffer(encoder, 1, k_cache);
                set_tensor_buffer(encoder, 2, v_cache);
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
            DType::F16, self.compute.device().info().id,
        )
    }

    /// Causal prefill attention via gqa_attention_f16. Encodes on provided CB.
    /// Q/K/V: [seq_len, num_heads, head_dim]
    #[cfg(feature = "metal")]
    fn prefill_causal_attention_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k: &Tensor, v: &Tensor,
        scale: f32, seq_len: usize, num_heads: usize, head_dim: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let output_size = seq_len * num_heads * head_dim * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch(
            cb, &self.kernels.gqa_attention,
            (num_heads, seq_len, 1), (1, 1, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, q);
                set_tensor_buffer(encoder, 1, k);
                set_tensor_buffer(encoder, 2, v);
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
            DType::F16, self.compute.device().info().id,
        )
    }

    // ========================= MATMUL-BASED MULTI-HEAD ATTENTION =========================

    /// Multi-head attention via batched matmul decomposition (for long sequences).
    /// Q/K/V: [seq_len, d_model], reshaped internally to [seq, heads, dim].
    /// Returns: [q_seq_len, d_model].
    ///
    /// Steps: transpose → batched Q@K^T → softmax → batched S@V → transpose back.
    #[cfg(feature = "metal")]
    fn multi_head_attention_matmul_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k: &Tensor, v: &Tensor,
        num_heads: usize, head_dim: usize,
        q_seq_len: usize, kv_seq_len: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let q = q.reshape([q_seq_len, num_heads, head_dim])?;
        let k = k.reshape([kv_seq_len, num_heads, head_dim])?;
        let v = v.reshape([kv_seq_len, num_heads, head_dim])?;

        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;

        // Allocate transpose buffers: [H, S, D]
        let q_t = Tensor::empty(Shape::from([num_heads, q_seq_len, head_dim]), DType::F16, device_id)?;
        let k_t = Tensor::empty(Shape::from([num_heads, kv_seq_len, head_dim]), DType::F16, device_id)?;
        let v_t = Tensor::empty(Shape::from([num_heads, kv_seq_len, head_dim]), DType::F16, device_id)?;

        // Step 1: Transpose Q, K, V from [S, H, D] to [H, S, D]
        self.transpose_shd_to_hsd_on(cb, &q, &q_t, q_seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd_on(cb, &k, &k_t, kv_seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd_on(cb, &v, &v_t, kv_seq_len, num_heads, head_dim);

        // Step 2: Scores = Q' @ K'^T → [H, q_seq, kv_seq]
        let scores_size = num_heads * q_seq_len * kv_seq_len * 2;
        let scores_buffer = device.new_buffer(scores_size as u64, metal::MTLResourceOptions::StorageModeShared);
        {
            let tile: usize = 16;
            let grid_x = (kv_seq_len + tile - 1) / tile;
            let grid_y = (q_seq_len + tile - 1) / tile;
            self.compute.dispatch(
                cb, &self.kernels.batched_linear,
                (grid_x, grid_y, num_heads), (tile, tile, 1),
                |encoder| {
                    set_tensor_buffer(encoder, 0, &q_t);
                    set_tensor_buffer(encoder, 1, &k_t);
                    encoder.set_buffer(2, Some(&scores_buffer), 0);
                    let m = q_seq_len as u32;
                    let n = kv_seq_len as u32;
                    let k_dim = head_dim as u32;
                    encoder.set_bytes(3, 4, &m as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &n as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &k_dim as *const u32 as *const _);
                },
            );
        }

        // Step 3: Row-wise scaled softmax (in-place)
        {
            let total_rows = num_heads * q_seq_len;
            self.compute.dispatch_1d(
                cb, &self.kernels.row_softmax_scale, total_rows,
                |encoder| {
                    encoder.set_buffer(0, Some(&scores_buffer), 0);
                    let rows = total_rows as u32;
                    let cols = kv_seq_len as u32;
                    encoder.set_bytes(1, 4, &rows as *const u32 as *const _);
                    encoder.set_bytes(2, 4, &cols as *const u32 as *const _);
                    encoder.set_bytes(3, 4, &scale as *const f32 as *const _);
                },
            );
        }

        // Step 4: Output = Scores @ V' → [H, q_seq, head_dim]
        let output_t = Tensor::empty(
            Shape::from([num_heads, q_seq_len, head_dim]), DType::F16, device_id)?;
        {
            let tile: usize = 16;
            let grid_x = (head_dim + tile - 1) / tile;
            let grid_y = (q_seq_len + tile - 1) / tile;
            self.compute.dispatch(
                cb, &self.kernels.batched_matmul_nn,
                (grid_x, grid_y, num_heads), (tile, tile, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&scores_buffer), 0);
                    set_tensor_buffer(encoder, 1, &v_t);
                    set_tensor_buffer(encoder, 2, &output_t);
                    let m = q_seq_len as u32;
                    let n = head_dim as u32;
                    let k_dim = kv_seq_len as u32;
                    encoder.set_bytes(3, 4, &m as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &n as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &k_dim as *const u32 as *const _);
                },
            );
        }

        // Step 5: Transpose output [H, S, D] → [S, H, D]
        let output = Tensor::empty(
            Shape::from([q_seq_len, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd_on(cb, &output_t, &output, q_seq_len, num_heads, head_dim);

        output.reshape([q_seq_len, num_heads * head_dim])
    }

    /// Cross-attention decode with pre-transposed K/V (HSD format).
    /// Q: [1, d_model] (single query). K_hsd/V_hsd: [H, S, D] (pre-transposed during prefill).
    /// Skips K/V transpose (saves 24 dispatches/step across 12 layers).
    #[cfg(feature = "metal")]
    fn cross_attention_decode_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k_hsd: &Tensor, v_hsd: &Tensor,
        num_heads: usize, head_dim: usize, kv_seq_len: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let q = q.reshape([1, num_heads, head_dim])?;
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;

        // Only transpose Q (tiny: [1, H, D] → [H, 1, D])
        let q_t = Tensor::empty(Shape::from([num_heads, 1, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd_on(cb, &q, &q_t, 1, num_heads, head_dim);

        // Scores = Q_t @ K_hsd^T → [H, 1, kv_seq]
        let scores_size = num_heads * kv_seq_len * 2;
        let scores_buffer = device.new_buffer(scores_size as u64, metal::MTLResourceOptions::StorageModeShared);
        {
            let tile: usize = 16;
            let grid_x = (kv_seq_len + tile - 1) / tile;
            self.compute.dispatch(
                cb, &self.kernels.batched_linear,
                (grid_x, 1, num_heads), (tile, tile, 1),
                |encoder| {
                    set_tensor_buffer(encoder, 0, &q_t);
                    set_tensor_buffer(encoder, 1, k_hsd);
                    encoder.set_buffer(2, Some(&scores_buffer), 0);
                    let m = 1u32;
                    let n = kv_seq_len as u32;
                    let k_dim = head_dim as u32;
                    encoder.set_bytes(3, 4, &m as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &n as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &k_dim as *const u32 as *const _);
                },
            );
        }

        // Row-wise scaled softmax
        {
            self.compute.dispatch_1d(
                cb, &self.kernels.row_softmax_scale, num_heads,
                |encoder| {
                    encoder.set_buffer(0, Some(&scores_buffer), 0);
                    let rows = num_heads as u32;
                    let cols = kv_seq_len as u32;
                    encoder.set_bytes(1, 4, &rows as *const u32 as *const _);
                    encoder.set_bytes(2, 4, &cols as *const u32 as *const _);
                    encoder.set_bytes(3, 4, &scale as *const f32 as *const _);
                },
            );
        }

        // Output = Scores @ V_hsd → [H, 1, head_dim]
        let output_t = Tensor::empty(
            Shape::from([num_heads, 1, head_dim]), DType::F16, device_id)?;
        {
            let tile: usize = 16;
            let grid_x = (head_dim + tile - 1) / tile;
            self.compute.dispatch(
                cb, &self.kernels.batched_matmul_nn,
                (grid_x, 1, num_heads), (tile, tile, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&scores_buffer), 0);
                    set_tensor_buffer(encoder, 1, v_hsd);
                    set_tensor_buffer(encoder, 2, &output_t);
                    let m = 1u32;
                    let n = head_dim as u32;
                    let k_dim = kv_seq_len as u32;
                    encoder.set_bytes(3, 4, &m as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &n as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &k_dim as *const u32 as *const _);
                },
            );
        }

        // Transpose output [H, 1, D] → [1, H, D]
        let output = Tensor::empty(
            Shape::from([1, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd_on(cb, &output_t, &output, 1, num_heads, head_dim);

        output.reshape([1, num_heads * head_dim])
    }

    /// Transpose [S, H, D] → [H, S, D] on GPU.
    #[cfg(feature = "metal")]
    fn transpose_shd_to_hsd_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, output: &Tensor,
        seq_len: usize, num_heads: usize, head_dim: usize,
    ) {
        let tg_y = 4usize.min(seq_len);
        self.compute.dispatch(
            cb, &self.kernels.transpose_shd_to_hsd,
            (1, (seq_len + tg_y - 1) / tg_y, num_heads),
            (head_dim, tg_y, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                set_tensor_buffer(encoder, 1, output);
                let s = seq_len as u32;
                let h = num_heads as u32;
                let d = head_dim as u32;
                encoder.set_bytes(2, 4, &s as *const u32 as *const _);
                encoder.set_bytes(3, 4, &h as *const u32 as *const _);
                encoder.set_bytes(4, 4, &d as *const u32 as *const _);
            },
        );
    }

    /// Transpose [H, S, D] → [S, H, D] on GPU.
    #[cfg(feature = "metal")]
    fn transpose_hsd_to_shd_on(
        &self, cb: &metal::CommandBufferRef,
        input: &Tensor, output: &Tensor,
        seq_len: usize, num_heads: usize, head_dim: usize,
    ) {
        let tg_y = 4usize.min(seq_len);
        self.compute.dispatch(
            cb, &self.kernels.transpose_hsd_to_shd,
            (1, (seq_len + tg_y - 1) / tg_y, num_heads),
            (head_dim, tg_y, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                set_tensor_buffer(encoder, 1, output);
                let s = seq_len as u32;
                let h = num_heads as u32;
                let d = head_dim as u32;
                encoder.set_bytes(2, 4, &s as *const u32 as *const _);
                encoder.set_bytes(3, 4, &h as *const u32 as *const _);
                encoder.set_bytes(4, 4, &d as *const u32 as *const _);
            },
        );
    }

    // ========================= KERNEL DISPATCH HELPERS (batched) =========================

    /// Linear: Y = X @ W^T + bias. Encodes on the provided command buffer.
    #[cfg(feature = "metal")]
    fn linear_on(&self, cb: &metal::CommandBufferRef, input: &Tensor,
                 weight: &LazyTensor, bias: Option<&LazyTensor>,
                 m: usize, k: usize, n: usize) -> Tensor {
        let device = self.compute.device().raw();
        let output_size = m * n * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let tile: usize = 16;
        let grid_x = (n + tile - 1) / tile;
        let grid_y = (m + tile - 1) / tile;

        self.compute.dispatch(
            cb, &self.kernels.linear,
            (grid_x, grid_y, 1), (tile, tile, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                if let Some(b) = bias {
                    set_lazy_buffer(encoder, 2, b);
                } else {
                    encoder.set_buffer(2, Some(&output_buffer), 0);
                }
                encoder.set_buffer(3, Some(&output_buffer), 0);

                let m_u32 = m as u32;
                let n_u32 = n as u32;
                let k_u32 = k as u32;
                let has_bias_u32: u32 = if bias.is_some() { 1 } else { 0 };

                encoder.set_bytes(4, 4, &m_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &k_u32 as *const u32 as *const _);
                encoder.set_bytes(7, 4, &has_bias_u32 as *const u32 as *const _);
            },
        );

        Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, self.compute.device().info().id)
    }

    /// Layer normalization. Encodes on the provided command buffer.
    #[cfg(feature = "metal")]
    fn layer_norm_on(&self, cb: &metal::CommandBufferRef, input: &Tensor,
                     weight: &LazyTensor, bias: &LazyTensor,
                     n: usize, d: usize) -> Tensor {
        let device = self.compute.device().raw();
        let output_size = n * d * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch_1d(
            cb, &self.kernels.layer_norm, n,
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                set_lazy_buffer(encoder, 2, bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);

                let n_u32 = n as u32;
                let d_u32 = d as u32;
                let eps: f32 = 1e-5;
                encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
            },
        );

        Tensor::from_metal_buffer(output_buffer, Shape::from([n, d]), DType::F16, self.compute.device().info().id)
    }

    /// Conv1d. Encodes on the provided command buffer.
    #[cfg(feature = "metal")]
    fn conv1d_on(&self, cb: &metal::CommandBufferRef, input: &Tensor,
                 weight: &LazyTensor, bias: &LazyTensor,
                 c_in: usize, c_out: usize, l_in: usize,
                 kernel_size: usize, stride: usize, padding: usize) -> Tensor {
        let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
        let device = self.compute.device().raw();
        let output_size = c_out * l_out * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch(
            cb, &self.kernels.conv1d,
            (l_out, c_out, 1), (1, 1, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                set_lazy_buffer(encoder, 2, bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);

                let c_in_u32 = c_in as u32;
                let c_out_u32 = c_out as u32;
                let l_in_u32 = l_in as u32;
                let k_u32 = kernel_size as u32;
                let stride_u32 = stride as u32;
                let padding_u32 = padding as u32;

                encoder.set_bytes(4, 4, &c_in_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_out_u32 as *const u32 as *const _);
                encoder.set_bytes(6, 4, &l_in_u32 as *const u32 as *const _);
                encoder.set_bytes(7, 4, &k_u32 as *const u32 as *const _);
                encoder.set_bytes(8, 4, &stride_u32 as *const u32 as *const _);
                encoder.set_bytes(9, 4, &padding_u32 as *const u32 as *const _);
            },
        );

        Tensor::from_metal_buffer(output_buffer, Shape::from([c_out, l_out]), DType::F16, self.compute.device().info().id)
    }

    /// GELU activation. Encodes on the provided command buffer.
    #[cfg(feature = "metal")]
    fn gelu_on(&self, cb: &metal::CommandBufferRef, input: &Tensor) -> Tensor {
        let numel = input.numel();
        let device = self.compute.device().raw();
        let output_size = numel * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch_1d(
            cb, &self.kernels.gelu, numel,
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(&output_buffer), 0);
            },
        );

        Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, self.compute.device().info().id)
    }

    /// Element-wise add. Encodes on the provided command buffer.
    #[cfg(feature = "metal")]
    fn add_on(&self, cb: &metal::CommandBufferRef, a: &Tensor, b: &Tensor) -> Tensor {
        let numel = a.numel();
        let device = self.compute.device().raw();
        let output_size = numel * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch_1d(
            cb, &self.kernels.add, numel,
            |encoder| {
                set_tensor_buffer(encoder, 0, a);
                set_tensor_buffer(encoder, 1, b);
                encoder.set_buffer(2, Some(&output_buffer), 0);
            },
        );

        Tensor::from_metal_buffer(output_buffer, a.shape().clone(), DType::F16, self.compute.device().info().id)
    }


    /// Embed tokens. Uses its own command buffer (called once per step).
    #[cfg(feature = "metal")]
    fn embed_tokens_on(&self, cb: &metal::CommandBufferRef, token_ids: &[u32],
                       weight: &LazyTensor, d_model: usize) -> Tensor {
        let seq_len = token_ids.len();
        let vocab_size = self.config.vocab_size;
        let device = self.compute.device().raw();
        let output_size = seq_len * d_model * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let id_buf = device.new_buffer_with_data(
            token_ids.as_ptr() as *const _,
            (seq_len * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        self.compute.dispatch_1d(
            cb, &self.kernels.embedding, seq_len,
            |encoder| {
                set_lazy_buffer(encoder, 0, weight);
                encoder.set_buffer(1, Some(&id_buf), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);

                let vocab_size_u32 = vocab_size as u32;
                let hidden_size_u32 = d_model as u32;
                let seq_len_u32 = seq_len as u32;
                encoder.set_bytes(3, 4, &vocab_size_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &hidden_size_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &seq_len_u32 as *const u32 as *const _);
            },
        );

        Tensor::from_metal_buffer(output_buffer, Shape::from([seq_len, d_model]), DType::F16, self.compute.device().info().id)
    }

    /// CPU argmax for greedy decoding. logits: [seq_len, vocab_size] — takes last row.
    #[cfg(feature = "metal")]
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

    /// Helper to get a weight from the model as LazyTensor.
    #[cfg(feature = "metal")]
    fn get_weight(&self, name: &str) -> Result<&LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| Error::internal(format!("weight not found: {}", name)))
    }

    /// Convert a LazyTensor to a Tensor (copies data). Used for positional embeddings.
    #[cfg(feature = "metal")]
    fn lazy_to_tensor(&self, lt: &LazyTensor) -> Result<Tensor> {
        let size = lt.size();
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared);
        unsafe {
            let src = lt.buffer().contents() as *const u8;
            let dst = output_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        output_buffer.did_modify_range(metal::NSRange::new(0, size as u64));
        Ok(Tensor::from_metal_buffer(output_buffer, lt.shape().clone(), lt.dtype(), self.compute.device().info().id))
    }

}

/// Compute log-mel spectrogram from raw audio samples.
/// audio: PCM samples at 16kHz, mono, f32 in [-1, 1].
/// Returns: Tensor [80, num_frames] on the specified device.
pub fn compute_mel_spectrogram(
    audio: &[f32],
    num_mel_bins: usize,
    device_id: crate::hal::DeviceId,
) -> Result<Tensor> {
    let sample_rate = 16000;
    let n_fft = 400;
    let hop_length = 160;
    let max_frames = 3000; // 30 seconds

    // Pad or truncate to 30 seconds
    let target_len = sample_rate * 30;
    let mut padded = vec![0.0f32; target_len];
    let copy_len = audio.len().min(target_len);
    padded[..copy_len].copy_from_slice(&audio[..copy_len]);

    let num_frames = (target_len - n_fft) / hop_length + 1;
    let num_frames = num_frames.min(max_frames);

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
        .collect();

    // STFT magnitude squared via FFT
    let n_freq = n_fft / 2 + 1; // 201
    let mut magnitudes = vec![0.0f32; num_frames * n_freq];

    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_buf = vec![rustfft::num_complex::Complex::new(0.0f32, 0.0f32); n_fft];

    for frame in 0..num_frames {
        let start = frame * hop_length;
        for n in 0..n_fft {
            fft_buf[n] = rustfft::num_complex::Complex::new(padded[start + n] * window[n], 0.0);
        }
        fft.process(&mut fft_buf);
        for freq in 0..n_freq {
            let c = &fft_buf[freq];
            magnitudes[frame * n_freq + freq] = c.re * c.re + c.im * c.im;
        }
    }

    // Mel filterbank (80 filters over 201 frequency bins)
    let mel_filters = build_mel_filterbank(num_mel_bins, n_freq, sample_rate, n_fft);

    // Apply mel filterbank and take log
    let mut mel = vec![0.0f32; num_mel_bins * num_frames];
    for m in 0..num_mel_bins {
        for frame in 0..num_frames {
            let mut sum = 0.0f32;
            for freq in 0..n_freq {
                sum += mel_filters[m * n_freq + freq] * magnitudes[frame * n_freq + freq];
            }
            mel[m * num_frames + frame] = (sum.max(1e-10)).ln();
        }
    }

    // Normalize: Whisper uses max-based normalization
    let max_val = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = (max_val - 8.0).max(-10.0);
    for v in &mut mel {
        *v = ((*v).max(min_val) - max_val) / 4.0 + 1.0;
    }

    // Convert to F16 and create tensor
    let mel_f16: Vec<half::f16> = mel.iter().map(|&v| half::f16::from_f32(v)).collect();
    Tensor::from_slice(&mel_f16, Shape::from([num_mel_bins, num_frames]), DType::F16, device_id)
}

/// Build mel filterbank matrix [num_mel_bins, n_freq].
fn build_mel_filterbank(num_mel_bins: usize, n_freq: usize, sample_rate: usize, n_fft: usize) -> Vec<f32> {
    let f_max = sample_rate as f32 / 2.0;
    let f_min = 0.0f32;

    let hz_to_mel = |f: f32| -> f32 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f32| -> f32 { 700.0 * (10.0f32.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let num_points = num_mel_bins + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_points - 1) as f32)
        .collect();

    let bin_points: Vec<f32> = mel_points.iter()
        .map(|&m| mel_to_hz(m) * n_fft as f32 / sample_rate as f32)
        .collect();

    let mut filters = vec![0.0f32; num_mel_bins * n_freq];
    for m in 0..num_mel_bins {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for freq in 0..n_freq {
            let f = freq as f32;
            if f >= f_left && f <= f_center {
                filters[m * n_freq + freq] = (f - f_left) / (f_center - f_left);
            } else if f > f_center && f <= f_right {
                filters[m * n_freq + freq] = (f_right - f) / (f_right - f_center);
            }
        }
    }

    filters
}
