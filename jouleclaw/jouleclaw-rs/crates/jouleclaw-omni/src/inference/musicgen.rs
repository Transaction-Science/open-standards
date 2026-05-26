//! MusicGen conditional music generation pipeline.
//!
//! Implements Meta's MusicGen on Metal GPU:
//!   Text → T5-base encoder → enc_to_dec_proj → 24-layer transformer decoder
//!   → 4 codebook heads → EnCodec decoder → audio waveform
//!
//! Supports MusicGen-small (591M params, 24 decoder layers, 4 codebooks).

use super::model::Model;
use super::tokenizer::HfTokenizer;
use crate::core::{Error, Result};
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use super::architecture::t5::{T5Encoder, T5Config};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, LazyTensor, BorrowedMetalBuffer};

/// Helper to set a Metal buffer from a Tensor's device_ptr on the encoder.
#[cfg(feature = "metal")]
fn set_tensor_buffer(encoder: &metal::ComputeCommandEncoderRef, index: u64, tensor: &Tensor) {
    if let Some(ptr) = tensor.device_ptr() {
        let b = unsafe { BorrowedMetalBuffer::from_device_ptr(ptr) };
        encoder.set_buffer(index, Some(b.as_ref()), tensor.byte_offset() as u64);
    }
}

#[cfg(feature = "metal")]
fn set_lazy_buffer(encoder: &metal::ComputeCommandEncoderRef, index: u64, lt: &LazyTensor) {
    encoder.set_buffer(index, Some(lt.buffer()), 0);
}

/// MusicGen configuration.
#[derive(Debug, Clone)]
pub struct MusicGenConfig {
    /// Decoder hidden dimension.
    pub d_model: usize,
    /// Decoder attention heads.
    pub num_heads: usize,
    /// Decoder FFN dimension.
    pub ffn_dim: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Number of parallel codebooks.
    pub num_codebooks: usize,
    /// Codebook vocabulary size (tokens per codebook).
    pub codebook_size: usize,
    /// Maximum decoder positions.
    pub max_positions: usize,
    /// T5 text encoder hidden dimension.
    pub t5_d_model: usize,
    /// Audio sampling rate.
    pub audio_sample_rate: usize,
    /// EnCodec codebook dimension.
    pub codec_dim: usize,
    /// Guidance scale for classifier-free guidance.
    pub guidance_scale: f32,
    /// PAD/BOS token ID.
    pub pad_token_id: u32,
}

impl Default for MusicGenConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            num_heads: 16,
            ffn_dim: 4096,
            num_layers: 24,
            num_codebooks: 4,
            codebook_size: 2048,
            max_positions: 2048,
            t5_d_model: 768,
            audio_sample_rate: 32000,
            codec_dim: 128,
            guidance_scale: 3.0,
            pad_token_id: 2048,
        }
    }
}

impl MusicGenConfig {
    /// Parse MusicGen config from config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::internal(format!("failed to parse config: {}", e)))?;

        let mut config = Self::default();

        // Decoder config is nested under "decoder"
        if let Some(dec) = json.get("decoder") {
            if let Some(v) = dec.get("hidden_size").and_then(|v| v.as_u64()) { config.d_model = v as usize; }
            if let Some(v) = dec.get("num_attention_heads").and_then(|v| v.as_u64()) { config.num_heads = v as usize; }
            if let Some(v) = dec.get("ffn_dim").and_then(|v| v.as_u64()) { config.ffn_dim = v as usize; }
            if let Some(v) = dec.get("num_hidden_layers").and_then(|v| v.as_u64()) { config.num_layers = v as usize; }
            if let Some(v) = dec.get("num_codebooks").and_then(|v| v.as_u64()) { config.num_codebooks = v as usize; }
            if let Some(v) = dec.get("vocab_size").and_then(|v| v.as_u64()) { config.codebook_size = v as usize; }
            if let Some(v) = dec.get("max_position_embeddings").and_then(|v| v.as_u64()) { config.max_positions = v as usize; }
        }

        // Text encoder config
        if let Some(enc) = json.get("text_encoder") {
            if let Some(v) = enc.get("d_model").and_then(|v| v.as_u64()) { config.t5_d_model = v as usize; }
        }

        // Audio encoder config
        if let Some(ae) = json.get("audio_encoder") {
            if let Some(v) = ae.get("sampling_rate").and_then(|v| v.as_u64()) { config.audio_sample_rate = v as usize; }
            if let Some(v) = ae.get("codebook_dim").and_then(|v| v.as_u64()) { config.codec_dim = v as usize; }
        }

        Ok(config)
    }
}

/// KV cache for MusicGen decoder.
#[cfg(feature = "metal")]
struct MusicGenKVCache {
    /// Self-attention K cache per layer: [max_positions, num_heads, head_dim]
    self_k: Vec<Tensor>,
    self_v: Vec<Tensor>,
    /// Pre-transposed cross-attention K/V: [num_heads, enc_seq_len, head_dim] (HSD)
    cross_k_hsd: Vec<Tensor>,
    cross_v_hsd: Vec<Tensor>,
    seq_len: usize,
    cross_cached: bool,
    num_heads: usize,
    head_dim: usize,
}

#[cfg(feature = "metal")]
impl MusicGenKVCache {
    fn new(
        num_layers: usize, num_heads: usize, head_dim: usize,
        max_positions: usize, device_id: crate::hal::DeviceId,
    ) -> Result<Self> {
        let shape = Shape::from([max_positions, num_heads, head_dim]);
        let mut self_k = Vec::with_capacity(num_layers);
        let mut self_v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            self_k.push(Tensor::empty(shape.clone(), DType::F16, device_id)?);
            self_v.push(Tensor::empty(shape.clone(), DType::F16, device_id)?);
        }
        Ok(Self {
            self_k, self_v,
            cross_k_hsd: Vec::new(),
            cross_v_hsd: Vec::new(),
            seq_len: 0,
            cross_cached: false,
            num_heads, head_dim,
        })
    }

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

/// Compiled Metal kernels for MusicGen inference.
#[cfg(feature = "metal")]
struct MusicGenKernels {
    linear: Arc<ComputePipeline>,
    layer_norm: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    elu: Arc<ComputePipeline>,
    add: Arc<ComputePipeline>,
    embedding: Arc<ComputePipeline>,
    autoregressive_attention: Arc<ComputePipeline>,
    gqa_attention: Arc<ComputePipeline>,
    conv1d: Arc<ComputePipeline>,
    conv1d_transpose: Arc<ComputePipeline>,
    transpose_shd_to_hsd: Arc<ComputePipeline>,
    transpose_hsd_to_shd: Arc<ComputePipeline>,
    batched_linear: Arc<ComputePipeline>,
    batched_matmul_nn: Arc<ComputePipeline>,
    row_softmax_scale: Arc<ComputePipeline>,
}

/// MusicGen conditional music generation pipeline.
#[cfg(feature = "metal")]
pub struct MusicGenPipeline {
    model: Arc<Model>,
    config: MusicGenConfig,
    text_encoder: T5Encoder,
    tokenizer: Option<Arc<HfTokenizer>>,
    compute: Arc<MetalCompute>,
    kernels: MusicGenKernels,
}

#[cfg(feature = "metal")]
impl MusicGenPipeline {
    /// Create a new MusicGen pipeline.
    pub fn new(
        model: Arc<Model>, config: MusicGenConfig, device: Arc<MetalDevice>,
    ) -> Result<Self> {
        use crate::hal::metal::shader::sources;

        let compute = Arc::new(MetalCompute::new(device.clone()));

        let kernels = MusicGenKernels {
            linear: compute.compile_pipeline("linear", sources::LINEAR, "linear_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            elu: compute.compile_pipeline("elu", sources::GELU, "elu_f16")?,
            add: compute.compile_pipeline("add", sources::ELEMENTWISE, "add_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            autoregressive_attention: compute.compile_pipeline("autoregressive_attention", sources::AUTOREGRESSIVE_ATTENTION, "autoregressive_attention_f16")?,
            gqa_attention: compute.compile_pipeline("gqa_attention", sources::GQA_ATTENTION, "gqa_attention_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            conv1d_transpose: compute.compile_pipeline("conv1d_transpose", sources::CONV1D, "conv1d_transpose_f16")?,
            transpose_shd_to_hsd: compute.compile_pipeline("transpose_shd_to_hsd", sources::LINEAR, "transpose_shd_to_hsd_f16")?,
            transpose_hsd_to_shd: compute.compile_pipeline("transpose_hsd_to_shd", sources::LINEAR, "transpose_hsd_to_shd_f16")?,
            batched_linear: compute.compile_pipeline("batched_linear", sources::LINEAR, "batched_linear_f16")?,
            batched_matmul_nn: compute.compile_pipeline("batched_matmul_nn", sources::LINEAR, "batched_matmul_nn_f16")?,
            row_softmax_scale: compute.compile_pipeline("row_softmax_scale", sources::LINEAR, "row_softmax_scale_f16")?,
        };

        // Create T5-base text encoder with prefix "text_encoder."
        let t5_config = T5Config::t5_base();
        let text_encoder = T5Encoder::new(model.clone(), t5_config, device)?
            .with_prefix("text_encoder.");

        Ok(Self {
            model, config, text_encoder,
            tokenizer: None,
            compute, kernels,
        })
    }

    /// Set the text tokenizer (T5 SentencePiece).
    pub fn with_tokenizer(mut self, tokenizer: Arc<HfTokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Generate music from a text description.
    /// Returns mono f32 audio samples at 32kHz.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Tokenize text prompt
        let token_ids = if let Some(ref tok) = self.tokenizer {
            tok.encode(prompt)?
        } else {
            return Err(Error::internal("no tokenizer set"));
        };
        eprintln!("   Text tokens: {}", token_ids.len());

        // 2. Encode text via T5-base
        let encoder_out = self.text_encoder.encode(&token_ids)?;
        let enc_seq_len = token_ids.len();
        eprintln!("   Encoder output: [{}, {}]", enc_seq_len, config.t5_d_model);

        // 3. Project text embeddings to decoder space
        let proj_w = self.get_weight("enc_to_dec_proj.weight")?;
        let proj_b = self.get_weight("enc_to_dec_proj.bias")?;
        let cb = self.compute.new_command_buffer();
        let encoder_projected = self.linear_on(
            &cb, &encoder_out, proj_w, Some(proj_b),
            enc_seq_len, config.t5_d_model, config.d_model,
        );
        cb.commit();
        cb.wait_until_completed();
        eprintln!("   Projected: [{}, {}]", enc_seq_len, config.d_model);

        // 4. Initialize KV cache
        let head_dim = config.d_model / config.num_heads;
        let mut kv_cache = MusicGenKVCache::new(
            config.num_layers, config.num_heads, head_dim,
            config.max_positions, self.compute.device().info().id,
        )?;

        // 5. Prefill: compute and cache cross-attention K/V for all decoder layers
        self.prefill_cross_attention(&encoder_projected, enc_seq_len, &mut kv_cache)?;
        eprintln!("   Cross-attention cached for {} layers", config.num_layers);

        // 6. Load position embeddings
        let pos_embed_lazy = self.get_weight("decoder.model.decoder.embed_positions.weights")
            .or_else(|_| self.get_weight("decoder.model.decoder.embed_positions.weight"))?;
        let pos_embed = self.lazy_to_tensor(pos_embed_lazy)?;

        // 7. Autoregressive generation loop
        // Start with PAD token (2048) for all 4 codebooks
        let mut codebook_tokens: Vec<Vec<u32>> = (0..config.num_codebooks)
            .map(|_| vec![config.pad_token_id])
            .collect();

        let mut generated_steps = 0usize;
        for step in 0..max_tokens {
            // Sum embeddings from all 4 codebooks for current position
            let current_tokens: Vec<u32> = codebook_tokens.iter()
                .map(|cb_tokens| *cb_tokens.last().unwrap())
                .collect();

            // Get position embedding for this step (offset by 2 for MusicGen)
            let pos_offset = step + 2;
            let pos_slice = pos_embed.slice(0, pos_offset, pos_offset + 1)?
                .reshape([1, config.d_model])?;

            // Compute input embedding: sum of 4 codebook embeddings + position
            let input = self.compute_decoder_input(&current_tokens, &pos_slice)?;

            // Forward through decoder layers
            let logits = self.decode_step(&input, &mut kv_cache)?;

            // Sample from each codebook head
            let mut all_pad = true;
            for c in 0..config.num_codebooks {
                let start = c * config.codebook_size;
                let end = start + config.codebook_size;
                let head_logits = logits.slice(0, start, end)?;
                let token = self.argmax_cpu(&head_logits, config.codebook_size)?;
                codebook_tokens[c].push(token);
                if token != config.pad_token_id {
                    all_pad = false;
                }
            }

            generated_steps = step + 1;
            if all_pad {
                break;
            }
        }
        eprintln!("   Generated {} steps ({} codebook tokens per stream)", generated_steps, generated_steps + 1);

        // 8. Decode codebook tokens to audio via EnCodec
        let audio = self.decode_audio(&codebook_tokens)?;
        eprintln!("   Audio: {} samples ({:.1}s at {}Hz)",
            audio.len(), audio.len() as f64 / config.audio_sample_rate as f64,
            config.audio_sample_rate);

        Ok(audio)
    }

    /// Compute decoder input: sum of 4 codebook embeddings + position embedding.
    fn compute_decoder_input(&self, tokens: &[u32], pos: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let cb = self.compute.new_command_buffer();

        // Embed each codebook token and sum
        let mut sum: Option<Tensor> = None;
        for (c, &token) in tokens.iter().enumerate() {
            let embed_w = self.get_weight(&format!("decoder.model.decoder.embed_tokens.{}.weight", c))?;
            let embedded = self.embed_tokens_on(&cb, &[token], embed_w, config.d_model, config.codebook_size + 1);
            sum = Some(match sum {
                None => embedded,
                Some(prev) => self.add_on(&cb, &prev, &embedded),
            });
        }

        // Add position embedding
        let input = self.add_on(&cb, &sum.unwrap(), pos);
        cb.commit();
        cb.wait_until_completed();
        Ok(input)
    }

    /// Prefill cross-attention K/V cache for all decoder layers.
    fn prefill_cross_attention(
        &self, encoder_out: &Tensor, enc_seq_len: usize,
        kv_cache: &mut MusicGenKVCache,
    ) -> Result<()> {
        let config = &self.config;
        let num_heads = config.num_heads;
        let head_dim = config.d_model / num_heads;

        for layer in 0..config.num_layers {
            let prefix = format!("decoder.model.decoder.layers.{}", layer);

            let cb = self.compute.new_command_buffer();

            // Cross-attention K/V projections
            let ck_w = self.get_weight(&format!("{}.encoder_attn.k_proj.weight", prefix))?;
            let cv_w = self.get_weight(&format!("{}.encoder_attn.v_proj.weight", prefix))?;
            let ck_b = self.model.get_weight(&format!("{}.encoder_attn.k_proj.bias", prefix));
            let cv_b = self.model.get_weight(&format!("{}.encoder_attn.v_proj.bias", prefix));

            let cross_k = self.linear_on(&cb, encoder_out, ck_w, ck_b,
                enc_seq_len, config.d_model, config.d_model);
            let cross_v = self.linear_on(&cb, encoder_out, cv_w, cv_b,
                enc_seq_len, config.d_model, config.d_model);

            // Reshape and transpose to HSD format for fast decode
            let k_reshaped = cross_k.reshape([enc_seq_len, num_heads, head_dim])?;
            let v_reshaped = cross_v.reshape([enc_seq_len, num_heads, head_dim])?;

            let device_id = self.compute.device().info().id;
            let k_hsd = Tensor::empty(
                Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(
                Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;

            self.transpose_shd_to_hsd_on(&cb, &k_reshaped, &k_hsd, enc_seq_len, num_heads, head_dim);
            self.transpose_shd_to_hsd_on(&cb, &v_reshaped, &v_hsd, enc_seq_len, num_heads, head_dim);

            cb.commit();
            cb.wait_until_completed();

            kv_cache.set_cross_hsd(layer, k_hsd, v_hsd);
        }
        kv_cache.cross_cached = true;
        Ok(())
    }

    /// Single-token decode step through all 24 decoder layers.
    /// Returns logits [4 * codebook_size] (concatenated 4 codebook heads).
    fn decode_step(&self, input: &Tensor, kv_cache: &mut MusicGenKVCache) -> Result<Tensor> {
        let config = &self.config;
        let num_heads = config.num_heads;
        let head_dim = config.d_model / num_heads;

        let cb = self.compute.new_command_buffer();
        let mut hidden = input.clone();

        for layer in 0..config.num_layers {
            hidden = self.decoder_layer_step_on(&cb, layer, hidden, kv_cache)?;
        }

        // Final layer norm
        let ln_w = self.get_weight("decoder.model.decoder.layer_norm.weight")?;
        let ln_b = self.get_weight("decoder.model.decoder.layer_norm.bias")?;
        let normed = self.layer_norm_on(&cb, &hidden, ln_w, ln_b, 1, config.d_model);

        // Compute logits for all 4 codebook heads and concatenate
        let mut all_logits = Vec::new();
        for c in 0..config.num_codebooks {
            let head_w = self.get_weight(&format!("decoder.lm_heads.{}.weight", c))?;
            let logits = self.linear_on(&cb, &normed, head_w, None,
                1, config.d_model, config.codebook_size);
            all_logits.push(logits);
        }

        cb.commit();
        cb.wait_until_completed();

        // Concatenate logits on CPU
        if all_logits.len() == 1 {
            return Ok(all_logits.into_iter().next().unwrap());
        }
        let mut cat_data: Vec<half::f16> = Vec::with_capacity(config.num_codebooks * config.codebook_size);
        for logit_tensor in &all_logits {
            let data: Vec<half::f16> = logit_tensor.to_vec()?;
            cat_data.extend_from_slice(&data);
        }
        Tensor::from_slice(&cat_data, Shape::from([config.num_codebooks * config.codebook_size]), DType::F16, self.compute.device().info().id)
    }

    /// Single decoder layer on a shared command buffer.
    fn decoder_layer_step_on(
        &self, cb: &metal::CommandBufferRef, layer: usize, input: Tensor,
        kv_cache: &mut MusicGenKVCache,
    ) -> Result<Tensor> {
        let config = &self.config;
        let prefix = format!("decoder.model.decoder.layers.{}", layer);
        let num_heads = config.num_heads;
        let head_dim = config.d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_pos = kv_cache.seq_len;

        // === Self-attention ===
        let ln1_w = self.get_weight(&format!("{}.self_attn_layer_norm.weight", prefix))?;
        let ln1_b = self.get_weight(&format!("{}.self_attn_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &input, ln1_w, ln1_b, 1, config.d_model);

        let sa = format!("{}.self_attn", prefix);
        let q_w = self.get_weight(&format!("{}.q_proj.weight", sa))?;
        let q_b = self.model.get_weight(&format!("{}.q_proj.bias", sa));
        let q = self.linear_on(cb, &normed, q_w, q_b, 1, config.d_model, config.d_model);

        let k_w = self.get_weight(&format!("{}.k_proj.weight", sa))?;
        let k_b = self.model.get_weight(&format!("{}.k_proj.bias", sa));
        let k_new = self.linear_on(cb, &normed, k_w, k_b, 1, config.d_model, config.d_model);

        let v_w = self.get_weight(&format!("{}.v_proj.weight", sa))?;
        let v_b = self.model.get_weight(&format!("{}.v_proj.bias", sa));
        let v_new = self.linear_on(cb, &normed, v_w, v_b, 1, config.d_model, config.d_model);

        // KV cache update via blit
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

        // Self-attention with cache
        let q_flat = q.reshape([num_heads, head_dim])?;
        let self_attn_out = self.autoregressive_attention_on(
            cb, &q_flat, &kv_cache.self_k[layer], &kv_cache.self_v[layer],
            scale, seq_pos, num_heads, head_dim,
        );
        let sa_flat = self_attn_out.reshape([1, config.d_model])?;
        let o_w = self.get_weight(&format!("{}.out_proj.weight", sa))?;
        let o_b = self.model.get_weight(&format!("{}.out_proj.bias", sa));
        let sa_out = self.linear_on(cb, &sa_flat, o_w, o_b, 1, config.d_model, config.d_model);
        let h = self.add_on(cb, &input, &sa_out);

        // === Cross-attention ===
        let ln2_w = self.get_weight(&format!("{}.encoder_attn_layer_norm.weight", prefix))?;
        let ln2_b = self.get_weight(&format!("{}.encoder_attn_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln2_w, ln2_b, 1, config.d_model);

        let cq_w = self.get_weight(&format!("{}.encoder_attn.q_proj.weight", prefix))?;
        let cq_b = self.model.get_weight(&format!("{}.encoder_attn.q_proj.bias", prefix));
        let cross_q = self.linear_on(cb, &normed, cq_w, cq_b, 1, config.d_model, config.d_model);

        let enc_seq_len = kv_cache.cross_k_hsd[layer].shape().dim(1).unwrap_or(1);
        let cross_attn_out = self.cross_attention_decode_on(
            cb, &cross_q, &kv_cache.cross_k_hsd[layer], &kv_cache.cross_v_hsd[layer],
            num_heads, head_dim, enc_seq_len, scale,
        )?;
        let co_w = self.get_weight(&format!("{}.encoder_attn.out_proj.weight", prefix))?;
        let co_b = self.model.get_weight(&format!("{}.encoder_attn.out_proj.bias", prefix));
        let cross_out = self.linear_on(cb, &cross_attn_out, co_w, co_b, 1, config.d_model, config.d_model);
        let h = self.add_on(cb, &h, &cross_out);

        // === FFN: LayerNorm → fc1 → ReLU → fc2 → residual ===
        let ln3_w = self.get_weight(&format!("{}.final_layer_norm.weight", prefix))?;
        let ln3_b = self.get_weight(&format!("{}.final_layer_norm.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln3_w, ln3_b, 1, config.d_model);

        let fc1_w = self.get_weight(&format!("{}.fc1.weight", prefix))?;
        let fc1_b = self.model.get_weight(&format!("{}.fc1.bias", prefix));
        let fc2_w = self.get_weight(&format!("{}.fc2.weight", prefix))?;
        let fc2_b = self.model.get_weight(&format!("{}.fc2.bias", prefix));

        let ffn_h = self.linear_on(cb, &normed, fc1_w, fc1_b, 1, config.d_model, config.ffn_dim);
        let ffn_h = self.relu_on(cb, &ffn_h);
        let ffn_out = self.linear_on(cb, &ffn_h, fc2_w, fc2_b, 1, config.ffn_dim, config.d_model);
        Ok(self.add_on(cb, &h, &ffn_out))
    }

    // ==================== EnCodec DECODER ====================

    /// Decode codebook tokens to audio waveform.
    /// codebook_tokens: [num_codebooks][seq_len] — generated token IDs.
    /// Returns mono f32 samples at 32kHz.
    fn decode_audio(&self, codebook_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Dequantize: look up codebook embeddings and sum
        // Skip the first token (BOS/PAD) from each codebook
        let seq_len = codebook_tokens[0].len().saturating_sub(1);
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // Sum residual quantizer outputs: [seq_len, codec_dim]
        let mut summed = vec![0.0f32; seq_len * config.codec_dim];
        for q in 0..config.num_codebooks {
            let embed_key = format!("audio_encoder.quantizer.layers.{}.codebook.embed", q);
            if let Some(embed_w) = self.model.get_weight(&embed_key) {
                let embed_data: Vec<half::f16> = unsafe {
                    let ptr = embed_w.buffer().contents() as *const half::f16;
                    std::slice::from_raw_parts(ptr, embed_w.shape().numel()).to_vec()
                };
                for (t, &token) in codebook_tokens[q][1..].iter().enumerate() {
                    let tok = token.min(config.codebook_size as u32 - 1) as usize;
                    for d in 0..config.codec_dim {
                        summed[t * config.codec_dim + d] += embed_data[tok * config.codec_dim + d].to_f32();
                    }
                }
            }
        }

        // 2. Run through EnCodec decoder on CPU
        // The decoder is: conv1d → LSTM → [ELU + ConvTranspose1d + ResBlocks]... → conv1d
        // For initial implementation: CPU-based decoder
        self.encodec_decode_cpu(&summed, seq_len, config.codec_dim)
    }

    /// CPU-based EnCodec decoder (fallback implementation).
    /// Processes quantized embeddings through the neural decoder to produce audio.
    fn encodec_decode_cpu(&self, input: &[f32], seq_len: usize, dim: usize) -> Result<Vec<f32>> {
        let config = &self.config;

        // The EnCodec decoder structure:
        // layers.0: Conv1d(128 → 1024, k=7, padding=3)
        // layers.1: LSTM(1024, 1024, 2 layers)
        // layers.2-14: [ELU + ConvTranspose1d + ResBlocks] × 4
        // layers.15: ELU + Conv1d(64 → 1, k=7, padding=3)
        //
        // For initial implementation, use GPU conv1d/conv1d_transpose + CPU LSTM.

        // Convert input to GPU tensor [codec_dim, seq_len] (channels-first for conv1d)
        let mut channels_first = vec![half::f16::ZERO; dim * seq_len];
        for t in 0..seq_len {
            for d in 0..dim {
                channels_first[d * seq_len + t] = half::f16::from_f32(input[t * dim + d]);
            }
        }
        let h = Tensor::from_slice(&channels_first, Shape::from([dim, seq_len]), DType::F16, self.compute.device().info().id)?;

        // Layer 0: Initial Conv1d(128 → 1024, k=7, padding=3)
        let h = self.encodec_conv1d(&h, "audio_encoder.decoder.layers.0", dim, 1024, seq_len, 7, 1, 3)?;
        let mut current_len = seq_len;
        let mut current_ch = 1024usize;

        // Layer 1: LSTM (run on CPU — sequential nature)
        let h = self.encodec_lstm_cpu(&h, current_ch, current_len)?;

        // Upsampling blocks: ELU → ConvTranspose1d → ResBlocks
        // EnCodec decoder structure (by layer index):
        //   0: Conv1d(128→1024, k=7)   1: LSTM(1024, 2 layers)
        //   2: ELU  3: ConvTranspose(1024→512, k=16, s=8)  4: ResBlock(512)
        //   5: ELU  6: ConvTranspose(512→256, k=10, s=5)   7: ResBlock(256)
        //   8: ELU  9: ConvTranspose(256→128, k=8, s=4)   10: ResBlock(128)
        //  11: ELU 12: ConvTranspose(128→64, k=8, s=4)    13: ResBlock(64)
        //  14: ELU 15: Conv1d(64→1, k=7)
        let ratios = [8usize, 5, 4, 4];
        let channels = [512usize, 256, 128, 64];
        let mut layer_idx = 2usize; // starts at first ELU
        let mut h = h;

        for (_i, (&ratio, &out_ch)) in ratios.iter().zip(channels.iter()).enumerate() {
            // ELU (layer_idx = 2, 5, 8, 11) — no weights
            h = self.elu_gpu(&h)?;
            layer_idx += 1;

            // ConvTranspose1d: upsample (layer_idx = 3, 6, 9, 12)
            let kernel_size = ratio * 2;
            let padding = ratio / 2 + ratio % 2;
            h = self.encodec_conv1d_transpose(
                &h, &format!("audio_encoder.decoder.layers.{}", layer_idx),
                current_ch, out_ch, current_len, kernel_size, ratio, padding,
            )?;
            current_len = (current_len - 1) * ratio - 2 * padding + kernel_size;
            current_ch = out_ch;
            layer_idx += 1;

            // ResBlock (layer_idx = 4, 7, 10, 13) — skip for now (pass-through)
            if self.model.get_weight(&format!("audio_encoder.decoder.layers.{}.block.1.conv.weight_v", layer_idx)).is_some() {
                layer_idx += 1;
            }
        }

        // Final: ELU (layer 14) → Conv1d(64 → 1, k=7, padding=3) (layer 15)
        h = self.elu_gpu(&h)?;
        layer_idx += 1;
        h = self.encodec_conv1d(&h, &format!("audio_encoder.decoder.layers.{}", layer_idx),
            current_ch, 1, current_len, 7, 1, 3)?;

        // Convert to f32 output
        let f16_data: Vec<half::f16> = h.to_vec()?;
        let audio: Vec<f32> = f16_data.iter().map(|v| v.to_f32()).collect();
        Ok(audio)
    }

    /// GPU Conv1d with weight normalization.
    fn encodec_conv1d(
        &self, input: &Tensor, prefix: &str,
        c_in: usize, c_out: usize, l_in: usize,
        kernel_size: usize, stride: usize, padding: usize,
    ) -> Result<Tensor> {
        // Try weight-normalized variant first (weight_g + weight_v)
        let (weight_data, bias) = self.get_weight_normalized(
            &format!("{}.weight_g", prefix),
            &format!("{}.weight_v", prefix),
            &format!("{}.bias", prefix),
            // Also try .conv. prefix
            &format!("{}.conv.weight_g", prefix),
            &format!("{}.conv.weight_v", prefix),
            &format!("{}.conv.bias", prefix),
        )?;

        let cb = self.compute.new_command_buffer();
        let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
        let device = self.compute.device().raw();
        let output_size = c_out * l_out * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch(
            &cb, &self.kernels.conv1d,
            (l_out, c_out, 1), (1, 1, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                set_tensor_buffer(encoder, 1, &weight_data);
                set_tensor_buffer(encoder, 2, &bias);
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

        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([c_out, l_out]), DType::F16, self.compute.device().info().id))
    }

    /// GPU ConvTranspose1d with weight normalization.
    fn encodec_conv1d_transpose(
        &self, input: &Tensor, prefix: &str,
        c_in: usize, c_out: usize, l_in: usize,
        kernel_size: usize, stride: usize, padding: usize,
    ) -> Result<Tensor> {
        let (weight_data, bias) = self.get_weight_normalized(
            &format!("{}.weight_g", prefix),
            &format!("{}.weight_v", prefix),
            &format!("{}.bias", prefix),
            &format!("{}.conv.weight_g", prefix),
            &format!("{}.conv.weight_v", prefix),
            &format!("{}.conv.bias", prefix),
        )?;

        let cb = self.compute.new_command_buffer();
        let l_out = (l_in - 1) * stride - 2 * padding + kernel_size;
        let device = self.compute.device().raw();
        let output_size = c_out * l_out * 2;
        let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch(
            &cb, &self.kernels.conv1d_transpose,
            (l_out, c_out, 1), (1, 1, 1),
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                set_tensor_buffer(encoder, 1, &weight_data);
                set_tensor_buffer(encoder, 2, &bias);
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

        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([c_out, l_out]), DType::F16, self.compute.device().info().id))
    }

    /// Get weight-normalized weight: weight = g * (v / ||v||).
    /// Tries primary keys first, then fallback keys.
    fn get_weight_normalized(
        &self, g_key: &str, v_key: &str, b_key: &str,
        g_key2: &str, v_key2: &str, b_key2: &str,
    ) -> Result<(Tensor, Tensor)> {
        let (g_lt, v_lt, b_lt) = if let Some(g) = self.model.get_weight(g_key) {
            (g, self.model.get_weight(v_key).unwrap(), self.model.get_weight(b_key).unwrap())
        } else if let Some(g) = self.model.get_weight(g_key2) {
            (g, self.model.get_weight(v_key2).unwrap(), self.model.get_weight(b_key2).unwrap())
        } else {
            return Err(Error::internal(format!("weight not found: {} or {}", g_key, g_key2)));
        };

        // Compute weight = g * v / ||v|| on CPU
        // Use shape().numel() not size()/2 — size() may reflect original F32 byte count
        let g_data: Vec<half::f16> = unsafe {
            let ptr = g_lt.buffer().contents() as *const half::f16;
            std::slice::from_raw_parts(ptr, g_lt.shape().numel()).to_vec()
        };
        let v_data: Vec<half::f16> = unsafe {
            let ptr = v_lt.buffer().contents() as *const half::f16;
            std::slice::from_raw_parts(ptr, v_lt.shape().numel()).to_vec()
        };
        let b_data: Vec<half::f16> = unsafe {
            let ptr = b_lt.buffer().contents() as *const half::f16;
            std::slice::from_raw_parts(ptr, b_lt.shape().numel()).to_vec()
        };

        // g is [C_out] or [C_out, 1, 1], v is [C_out, C_in, K]
        let c_out = g_data.len();
        let v_per_filter = v_data.len() / c_out;

        let mut weight = vec![half::f16::ZERO; v_data.len()];
        for co in 0..c_out {
            // Compute L2 norm of v[co]
            let start = co * v_per_filter;
            let end = start + v_per_filter;
            let norm: f32 = v_data[start..end].iter()
                .map(|x| { let f = x.to_f32(); f * f })
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            let g_val = g_data[co].to_f32();
            let scale = g_val / norm;
            for i in start..end {
                weight[i] = half::f16::from_f32(v_data[i].to_f32() * scale);
            }
        }

        let weight_tensor = Tensor::from_slice(&weight, v_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
        let bias_tensor = Tensor::from_slice(&b_data, b_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
        Ok((weight_tensor, bias_tensor))
    }

    /// CPU LSTM forward pass (2 layers, bidirectional=false).
    fn encodec_lstm_cpu(&self, input: &Tensor, hidden_size: usize, seq_len: usize) -> Result<Tensor> {
        // Read input from GPU
        let f16_data: Vec<half::f16> = input.to_vec()?;
        // input is [hidden_size, seq_len] (channels-first), convert to [seq_len, hidden_size]
        let mut x = vec![0.0f32; seq_len * hidden_size];
        for t in 0..seq_len {
            for d in 0..hidden_size {
                x[t * hidden_size + d] = f16_data[d * seq_len + t].to_f32();
            }
        }

        // Process 2 LSTM layers
        for layer in 0..2 {
            let prefix = format!("audio_encoder.decoder.layers.1.lstm.weight_ih_l{}", layer);
            let w_ih = self.read_weight_f32(&prefix)?;
            let w_hh = self.read_weight_f32(&format!("audio_encoder.decoder.layers.1.lstm.weight_hh_l{}", layer))?;
            let b_ih = self.read_weight_f32(&format!("audio_encoder.decoder.layers.1.lstm.bias_ih_l{}", layer))?;
            let b_hh = self.read_weight_f32(&format!("audio_encoder.decoder.layers.1.lstm.bias_hh_l{}", layer))?;

            let mut h = vec![0.0f32; hidden_size];
            let mut c = vec![0.0f32; hidden_size];
            let mut output = vec![0.0f32; seq_len * hidden_size];

            for t in 0..seq_len {
                let xt = &x[t * hidden_size..(t + 1) * hidden_size];

                // gates = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh
                let mut gates = vec![0.0f32; 4 * hidden_size];
                for g in 0..4 * hidden_size {
                    let mut sum = b_ih[g] + b_hh[g];
                    for k in 0..hidden_size {
                        sum += w_ih[g * hidden_size + k] * xt[k];
                        sum += w_hh[g * hidden_size + k] * h[k];
                    }
                    gates[g] = sum;
                }

                // Split into i, f, g, o gates
                for d in 0..hidden_size {
                    let i_gate = sigmoid(gates[d]);
                    let f_gate = sigmoid(gates[hidden_size + d]);
                    let g_gate = gates[2 * hidden_size + d].tanh();
                    let o_gate = sigmoid(gates[3 * hidden_size + d]);

                    c[d] = f_gate * c[d] + i_gate * g_gate;
                    h[d] = o_gate * c[d].tanh();
                }

                output[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&h);
            }

            x = output;
        }

        // Convert back to [hidden_size, seq_len] (channels-first) on GPU
        let mut channels_first = vec![half::f16::ZERO; hidden_size * seq_len];
        for t in 0..seq_len {
            for d in 0..hidden_size {
                channels_first[d * seq_len + t] = half::f16::from_f32(x[t * hidden_size + d]);
            }
        }
        Tensor::from_slice(&channels_first, Shape::from([hidden_size, seq_len]), DType::F16, self.compute.device().info().id)
    }

    /// Read a weight as f32 vector from LazyTensor.
    fn read_weight_f32(&self, name: &str) -> Result<Vec<f32>> {
        let lt = self.get_weight(name)?;
        let f16_data: Vec<half::f16> = unsafe {
            let ptr = lt.buffer().contents() as *const half::f16;
            std::slice::from_raw_parts(ptr, lt.shape().numel()).to_vec()
        };
        Ok(f16_data.iter().map(|v| v.to_f32()).collect())
    }

    /// ELU activation on GPU.
    fn elu_gpu(&self, input: &Tensor) -> Result<Tensor> {
        let numel = input.numel();
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        let cb = self.compute.new_command_buffer();
        self.compute.dispatch_1d(
            &cb, &self.kernels.elu, numel,
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(&output_buffer), 0);
            },
        );
        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, self.compute.device().info().id))
    }

    // ==================== GPU HELPER METHODS ====================

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

    fn relu_on(&self, cb: &metal::CommandBufferRef, input: &Tensor) -> Tensor {
        let numel = input.numel();
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch_1d(
            cb, &self.kernels.relu, numel,
            |encoder| {
                set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(&output_buffer), 0);
            },
        );

        Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, self.compute.device().info().id)
    }

    fn add_on(&self, cb: &metal::CommandBufferRef, a: &Tensor, b: &Tensor) -> Tensor {
        let numel = a.numel();
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((numel * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

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

    fn embed_tokens_on(&self, cb: &metal::CommandBufferRef, token_ids: &[u32],
                       weight: &LazyTensor, d_model: usize, vocab_size: usize) -> Tensor {
        let seq_len = token_ids.len();
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

                let vocab_u32 = vocab_size as u32;
                let hidden_u32 = d_model as u32;
                let seq_u32 = seq_len as u32;
                encoder.set_bytes(3, 4, &vocab_u32 as *const u32 as *const _);
                encoder.set_bytes(4, 4, &hidden_u32 as *const u32 as *const _);
                encoder.set_bytes(5, 4, &seq_u32 as *const u32 as *const _);
            },
        );

        Tensor::from_metal_buffer(output_buffer, Shape::from([seq_len, d_model]), DType::F16, self.compute.device().info().id)
    }

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

    fn cross_attention_decode_on(
        &self, cb: &metal::CommandBufferRef,
        q: &Tensor, k_hsd: &Tensor, v_hsd: &Tensor,
        num_heads: usize, head_dim: usize, kv_seq_len: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let q = q.reshape([1, num_heads, head_dim])?;
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;

        // Transpose Q: [1, H, D] → [H, 1, D]
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

        // Transpose output [H, 1, D] → [1, H*D]
        let output = Tensor::empty(
            Shape::from([1, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd_on(cb, &output_t, &output, 1, num_heads, head_dim);

        output.reshape([1, num_heads * head_dim])
    }

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

    fn argmax_cpu(&self, logits: &Tensor, vocab_size: usize) -> Result<u32> {
        let data: Vec<half::f16> = logits.to_vec()?;
        let last_row = if data.len() > vocab_size {
            &data[data.len() - vocab_size..]
        } else {
            &data
        };
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

    fn get_weight(&self, name: &str) -> Result<&LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| Error::internal(format!("weight not found: {}", name)))
    }

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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
