//! AudioGen and MAGNet: audio generation models from Meta.
//!
//! AudioGen: autoregressive transformer decoder conditioned on T5 text embeddings.
//!   Text → T5-large encoder (d=1024) → output_proj → 48-layer decoder → 4 codebook heads → EnCodec
//!   Uses Meta's internal format: fused QKV (in_proj_weight), sinusoidal positions.
//!
//! MAGNet: Masked Generative Non-autoregressive transformer.
//!   Uses iterative masked token prediction instead of autoregressive decoding.
//!   Generates all codebook tokens simultaneously using masking schedules.
//!   Significantly faster than autoregressive (parallel decoding).
//!
//! Both use EnCodec compression model for audio tokenization/detokenization.

use crate::core::{Error, Result};
use crate::tensor::{DType, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline, LazyTensor, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;
#[cfg(feature = "metal")]
use crate::inference::architecture::t5::{T5Encoder, T5Config};
#[cfg(feature = "metal")]
use crate::inference::tokenizer::HfTokenizer;

/// AudioGen configuration.
#[derive(Debug, Clone)]
pub struct AudioGenConfig {
    /// Decoder hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// FFN dimension.
    pub ffn_dim: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Number of codebooks.
    pub num_codebooks: usize,
    /// Codebook vocabulary size.
    pub codebook_size: usize,
    /// Maximum decoder positions.
    pub max_positions: usize,
    /// T5 text encoder dimension.
    pub t5_d_model: usize,
    /// Audio sample rate.
    pub sample_rate: usize,
    /// EnCodec codebook dimension.
    pub codec_dim: usize,
}

impl Default for AudioGenConfig {
    /// AudioGen-medium defaults.
    fn default() -> Self {
        Self {
            d_model: 1536,
            num_heads: 24,
            ffn_dim: 6144,
            num_layers: 48,
            num_codebooks: 4,
            codebook_size: 2048,
            max_positions: 2048,
            t5_d_model: 1024,
            sample_rate: 16000,
            codec_dim: 128,
        }
    }
}

/// MAGNet configuration.
#[derive(Debug, Clone)]
pub struct MAGNetConfig {
    /// Decoder hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// FFN dimension.
    pub ffn_dim: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Number of codebooks.
    pub num_codebooks: usize,
    /// Codebook vocabulary size.
    pub codebook_size: usize,
    /// T5 text encoder dimension.
    pub t5_d_model: usize,
    /// Audio sample rate.
    pub sample_rate: usize,
    /// Number of masking/decoding steps.
    pub num_decoding_steps: usize,
    /// Masking schedule type: "cosine" or "linear".
    pub masking_schedule: String,
    /// Temperature for initial step.
    pub initial_temperature: f32,
    /// Model size variant ("small" or "medium").
    pub variant: String,
}

impl Default for MAGNetConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            num_heads: 16,
            ffn_dim: 4096,
            num_layers: 24,
            num_codebooks: 4,
            codebook_size: 2048,
            t5_d_model: 768,
            sample_rate: 16000,
            num_decoding_steps: 20,
            masking_schedule: "cosine".to_string(),
            initial_temperature: 3.0,
            variant: "small".to_string(),
        }
    }
}

impl MAGNetConfig {
    /// MAGNet-medium configuration.
    pub fn medium() -> Self {
        Self {
            d_model: 1536,
            num_heads: 24,
            ffn_dim: 6144,
            num_layers: 48,
            variant: "medium".to_string(),
            ..Self::default()
        }
    }
}

// ==================== GPU Helpers ====================

#[cfg(feature = "metal")]
fn set_lazy_buffer(encoder: &metal::ComputeCommandEncoderRef, index: u64, lt: &LazyTensor) {
    encoder.set_buffer(index, Some(lt.buffer()), 0);
}

// ==================== KV Cache ====================

/// KV cache for AudioGen decoder.
#[cfg(feature = "metal")]
struct AudioGenKVCache {
    /// Self-attention K cache per layer: [max_positions, num_heads, head_dim]
    self_k: Vec<Tensor>,
    self_v: Vec<Tensor>,
    /// Pre-transposed cross-attention K/V: [num_heads, enc_seq_len, head_dim] (HSD)
    cross_k_hsd: Vec<Tensor>,
    cross_v_hsd: Vec<Tensor>,
    seq_len: usize,
    cross_cached: bool,
}

#[cfg(feature = "metal")]
impl AudioGenKVCache {
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

// ==================== Compiled Kernels ====================

#[cfg(feature = "metal")]
struct AudioGenKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    embedding: Arc<ComputePipeline>,
    autoregressive_attention: Arc<ComputePipeline>,
    // EnCodec kernels
    conv1d: Arc<ComputePipeline>,
    conv1d_transpose: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    elu: Arc<ComputePipeline>,
}

// ==================== AudioGen Pipeline ====================

/// AudioGen pipeline (autoregressive audio generation).
///
/// Architecture:
///   Text → T5-large encoder → output_proj → 48-layer transformer decoder → 4 codebook heads → EnCodec
///
/// Weight format (Meta's internal naming):
///   - `emb.{0..3}.weight` — codebook embeddings [2049, 1536]
///   - `linears.{0..3}.weight` — codebook output heads [2048, 1536]
///   - `transformer.layers.{L}.self_attn.in_proj_weight` — fused QKV [4608, 1536]
///   - `transformer.layers.{L}.self_attn.out_proj.weight` — [1536, 1536]
///   - `transformer.layers.{L}.cross_attention.in_proj_weight` — fused QKV [4608, 1536]
///   - `transformer.layers.{L}.cross_attention.out_proj.weight` — [1536, 1536]
///   - `transformer.layers.{L}.norm1.{weight,bias}` — self-attn LayerNorm
///   - `transformer.layers.{L}.norm_cross.{weight,bias}` — cross-attn LayerNorm
///   - `transformer.layers.{L}.norm2.{weight,bias}` — FFN LayerNorm
///   - `transformer.layers.{L}.linear1.weight` — FFN up [6144, 1536]
///   - `transformer.layers.{L}.linear2.weight` — FFN down [1536, 6144]
///   - `out_norm.{weight,bias}` — final LayerNorm
///   - `condition_provider.conditioners.description.output_proj.{weight,bias}` — T5→decoder proj
#[cfg(feature = "metal")]
pub struct AudioGenPipeline {
    model: Arc<Model>,
    codec_model: Option<Arc<Model>>,
    config: AudioGenConfig,
    text_encoder: T5Encoder,
    tokenizer: Option<Arc<HfTokenizer>>,
    compute: Arc<MetalCompute>,
    kernels: AudioGenKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for AudioGenPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl AudioGenPipeline {
    /// Create AudioGen pipeline.
    ///
    /// `model` — main model (from model.safetensors, converted from state_dict.bin)
    /// `t5_model` — T5 encoder model (separate safetensors file)
    /// `config` — AudioGen configuration
    /// `device` — Metal device
    pub fn new(
        model: Arc<Model>,
        t5_model: Arc<Model>,
        config: AudioGenConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device.clone()));

        let kernels = AudioGenKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            autoregressive_attention: compute.compile_pipeline("autoregressive_attention", sources::AUTOREGRESSIVE_ATTENTION, "autoregressive_attention_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            conv1d_transpose: compute.compile_pipeline("conv1d_transpose", sources::CONV1D, "conv1d_transpose_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            elu: compute.compile_pipeline("elu", sources::GELU, "elu_f16")?,
        };

        // T5-v1.1-large encoder (d_model=1024)
        let t5_config = T5Config::t5_v1_1_large();
        let text_encoder = T5Encoder::new(t5_model, t5_config, device)?;

        Ok(Self {
            model, codec_model: None, config, text_encoder,
            tokenizer: None, compute, kernels,
        })
    }

    /// Set the text tokenizer (T5 SentencePiece).
    pub fn with_tokenizer(mut self, tokenizer: Arc<HfTokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set the EnCodec codec model (for audio decoding).
    pub fn with_codec(mut self, codec_model: Arc<Model>) -> Self {
        self.codec_model = Some(codec_model);
        self
    }

    /// Generate audio from a text description.
    /// Returns mono f32 audio samples at 16kHz.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Tokenize
        let token_ids = if let Some(ref tok) = self.tokenizer {
            tok.encode(prompt)?
        } else {
            return Err(Error::internal("no tokenizer set"));
        };
        eprintln!("   Text tokens: {}", token_ids.len());

        // 2. Encode text via T5-large
        let encoder_out = self.text_encoder.encode(&token_ids)?;
        let enc_seq_len = token_ids.len();
        eprintln!("   Encoder output: [{}, {}]", enc_seq_len, config.t5_d_model);

        // 3. Project text embeddings to decoder space via output_proj
        let proj_w = self.w("condition_provider.conditioners.description.output_proj.weight")?;
        let proj_b = self.w("condition_provider.conditioners.description.output_proj.bias")?;
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
        let mut kv_cache = AudioGenKVCache::new(
            config.num_layers, config.num_heads, head_dim,
            config.max_positions, self.compute.device().info().id,
        )?;

        // 5. Prefill cross-attention K/V
        self.prefill_cross_attention(&encoder_projected, enc_seq_len, &mut kv_cache)?;
        eprintln!("   Cross-attention cached for {} layers", config.num_layers);

        // 6. Generate sinusoidal position embeddings
        let pos_embed = self.sinusoidal_positions(config.max_positions, config.d_model)?;

        // 7. Autoregressive generation loop
        let pad_token = config.codebook_size as u32; // BOS/PAD = codebook_size (2048)
        let mut codebook_tokens: Vec<Vec<u32>> = (0..config.num_codebooks)
            .map(|_| vec![pad_token])
            .collect();

        let mut generated_steps = 0usize;
        for step in 0..max_tokens {
            // Sum embeddings from all 4 codebooks for current position
            let current_tokens: Vec<u32> = codebook_tokens.iter()
                .map(|cb_tokens| *cb_tokens.last().unwrap())
                .collect();

            // Get position embedding for this step
            let pos_slice = pos_embed.slice(0, step, step + 1)?
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
                if token != pad_token {
                    all_pad = false;
                }
            }

            generated_steps = step + 1;
            if step > 0 && step % 50 == 0 {
                eprintln!("   Step {}/{}", step, max_tokens);
            }
            if all_pad {
                break;
            }
        }
        eprintln!("   Generated {} steps ({} codebook tokens per stream)", generated_steps, generated_steps + 1);

        // 8. Decode codebook tokens to audio via EnCodec
        let audio = self.decode_audio(&codebook_tokens)?;
        eprintln!("   Audio: {} samples ({:.1}s at {}Hz)",
            audio.len(), audio.len() as f64 / config.sample_rate as f64,
            config.sample_rate);

        Ok(audio)
    }

    /// Compute decoder input: sum of 4 codebook embeddings + position embedding.
    fn compute_decoder_input(&self, tokens: &[u32], pos: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let cb = self.compute.new_command_buffer();

        // Embed each codebook token and sum
        let mut sum: Option<Tensor> = None;
        for (c, &token) in tokens.iter().enumerate() {
            // AudioGen: emb.{c}.weight [codebook_size+1, d_model]
            let embed_w = self.w(&format!("emb.{}.weight", c))?;
            let embedded = self.embed_tokens_on(
                &cb, &[token], embed_w, config.d_model, config.codebook_size + 1,
            );
            sum = Some(match sum {
                None => embedded,
                Some(prev) => self.add(&cb, &prev, &embedded),
            });
        }

        // Add position embedding
        let input = self.add(&cb, &sum.unwrap(), pos);
        cb.commit();
        cb.wait_until_completed();
        Ok(input)
    }

    /// Generate sinusoidal position embeddings [max_len, d_model].
    fn sinusoidal_positions(&self, max_len: usize, d_model: usize) -> Result<Tensor> {
        let mut data = vec![half::f16::ZERO; max_len * d_model];
        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
                data[pos * d_model + 2 * i] = half::f16::from_f32(angle.sin() as f32);
                data[pos * d_model + 2 * i + 1] = half::f16::from_f32(angle.cos() as f32);
            }
        }
        Tensor::from_slice(&data, Shape::from([max_len, d_model]), DType::F16, self.compute.device().info().id)
    }

    /// Prefill cross-attention K/V cache for all decoder layers.
    fn prefill_cross_attention(
        &self, encoder_out: &Tensor, enc_seq_len: usize,
        kv_cache: &mut AudioGenKVCache,
    ) -> Result<()> {
        let config = &self.config;
        let num_heads = config.num_heads;
        let head_dim = config.d_model / num_heads;

        for layer in 0..config.num_layers {
            // AudioGen uses fused QKV: cross_attention.in_proj_weight [3*d_model, d_model]
            // Only K and V are needed from encoder (Q comes from decoder at each step)
            let in_proj_w = self.w(&format!(
                "transformer.layers.{}.cross_attention.in_proj_weight", layer
            ))?;

            let cb = self.compute.new_command_buffer();

            // K = encoder_out @ in_proj_weight[d_model..2*d_model, :]^T
            let cross_k = self.linear_with_row_offset_on(
                &cb, encoder_out, in_proj_w,
                enc_seq_len, config.d_model, config.d_model,
                config.d_model, // row offset for K (skip Q rows)
            );
            // V = encoder_out @ in_proj_weight[2*d_model..3*d_model, :]^T
            let cross_v = self.linear_with_row_offset_on(
                &cb, encoder_out, in_proj_w,
                enc_seq_len, config.d_model, config.d_model,
                2 * config.d_model, // row offset for V
            );

            // Transpose to HSD format
            let device_id = self.compute.device().info().id;
            let k_hsd = Tensor::empty(
                Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
            let v_hsd = Tensor::empty(
                Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;

            self.transpose_shd_to_hsd(&cb, &cross_k, &k_hsd, enc_seq_len, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &cross_v, &v_hsd, enc_seq_len, num_heads, head_dim);

            cb.commit();
            cb.wait_until_completed();

            kv_cache.set_cross_hsd(layer, k_hsd, v_hsd);
        }
        kv_cache.cross_cached = true;
        Ok(())
    }

    /// Single-token decode step through all decoder layers.
    /// Returns logits [num_codebooks * codebook_size].
    fn decode_step(&self, input: &Tensor, kv_cache: &mut AudioGenKVCache) -> Result<Tensor> {
        let config = &self.config;
        let num_heads = config.num_heads;
        let head_dim = config.d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let cb = self.compute.new_command_buffer();
        let mut hidden = input.clone();

        for layer in 0..config.num_layers {
            hidden = self.decoder_layer_step_on(&cb, layer, hidden, kv_cache, scale, num_heads, head_dim)?;
        }

        // Final layer norm: out_norm
        let ln_w = self.w("out_norm.weight")?;
        let ln_b = self.w("out_norm.bias")?;
        let normed = self.layer_norm_on(&cb, &hidden, ln_w, ln_b, 1, config.d_model);

        // Compute logits for all codebook heads and concatenate
        let mut all_logits = Vec::new();
        for c in 0..config.num_codebooks {
            let head_w = self.w(&format!("linears.{}.weight", c))?;
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
        kv_cache: &mut AudioGenKVCache,
        scale: f32, num_heads: usize, head_dim: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let prefix = format!("transformer.layers.{}", layer);
        let seq_pos = kv_cache.seq_len;

        // === Self-attention ===
        let ln1_w = self.w(&format!("{}.norm1.weight", prefix))?;
        let ln1_b = self.w(&format!("{}.norm1.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &input, ln1_w, ln1_b, 1, config.d_model);

        // Fused QKV: in_proj_weight [3*d_model, d_model]
        let sa_in_proj = self.w(&format!("{}.self_attn.in_proj_weight", prefix))?;

        // Q = normed @ in_proj[0:d_model, :]^T
        let q = self.linear_with_row_offset_on(
            cb, &normed, sa_in_proj, 1, config.d_model, config.d_model, 0,
        );
        // K = normed @ in_proj[d_model:2*d_model, :]^T
        let k_new = self.linear_with_row_offset_on(
            cb, &normed, sa_in_proj, 1, config.d_model, config.d_model, config.d_model,
        );
        // V = normed @ in_proj[2*d_model:3*d_model, :]^T
        let v_new = self.linear_with_row_offset_on(
            cb, &normed, sa_in_proj, 1, config.d_model, config.d_model, 2 * config.d_model,
        );

        // KV cache update via blit
        let k_new_r = k_new.reshape([1, num_heads, head_dim])?;
        let v_new_r = v_new.reshape([1, num_heads, head_dim])?;
        let stride_row = num_heads * head_dim * 2; // 2 bytes per F16
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
        let o_w = self.w(&format!("{}.self_attn.out_proj.weight", prefix))?;
        let sa_out = self.linear_on(cb, &sa_flat, o_w, None, 1, config.d_model, config.d_model);
        let h = self.add(cb, &input, &sa_out);

        // === Cross-attention ===
        let lnc_w = self.w(&format!("{}.norm_cross.weight", prefix))?;
        let lnc_b = self.w(&format!("{}.norm_cross.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, lnc_w, lnc_b, 1, config.d_model);

        // Cross-attention Q from fused in_proj_weight (only Q part, rows 0..d_model)
        let ca_in_proj = self.w(&format!("{}.cross_attention.in_proj_weight", prefix))?;
        let cross_q = self.linear_with_row_offset_on(
            cb, &normed, ca_in_proj, 1, config.d_model, config.d_model, 0,
        );

        let enc_seq_len = kv_cache.cross_k_hsd[layer].shape().dim(1).unwrap_or(1);
        let cross_attn_out = self.cross_attention_decode_on(
            cb, &cross_q, &kv_cache.cross_k_hsd[layer], &kv_cache.cross_v_hsd[layer],
            num_heads, head_dim, enc_seq_len, scale,
        )?;
        let co_w = self.w(&format!("{}.cross_attention.out_proj.weight", prefix))?;
        let cross_out = self.linear_on(cb, &cross_attn_out, co_w, None, 1, config.d_model, config.d_model);
        let h = self.add(cb, &h, &cross_out);

        // === FFN: LayerNorm → linear1 → GELU → linear2 → residual ===
        let ln2_w = self.w(&format!("{}.norm2.weight", prefix))?;
        let ln2_b = self.w(&format!("{}.norm2.bias", prefix))?;
        let normed = self.layer_norm_on(cb, &h, ln2_w, ln2_b, 1, config.d_model);

        let fc1_w = self.w(&format!("{}.linear1.weight", prefix))?;
        let fc2_w = self.w(&format!("{}.linear2.weight", prefix))?;
        let ffn_h = self.linear_on(cb, &normed, fc1_w, None, 1, config.d_model, config.ffn_dim);
        let ffn_h = self.activation(cb, &self.kernels.gelu, &ffn_h);
        let ffn_out = self.linear_on(cb, &ffn_h, fc2_w, None, 1, config.ffn_dim, config.d_model);
        Ok(self.add(cb, &h, &ffn_out))
    }

    // ==================== EnCodec Decoder ====================

    /// Decode codebook tokens to audio waveform.
    fn decode_audio(&self, codebook_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        let config = &self.config;
        let codec = self.codec_model.as_ref()
            .ok_or_else(|| Error::internal("no codec model set — call with_codec()"))?;

        let seq_len = codebook_tokens[0].len().saturating_sub(1);
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // Dequantize: look up codebook embeddings and sum
        let mut summed = vec![0.0f32; seq_len * config.codec_dim];
        for q in 0..config.num_codebooks {
            let embed_key = format!("quantizer.vq.layers.{}.codebook.embed", q);
            if let Some(embed_w) = codec.get_weight(&embed_key) {
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

        // EnCodec decoder: conv1d → LSTM → [ELU + ConvTranspose1d]×4 → conv1d
        self.encodec_decode_cpu(&summed, seq_len, config.codec_dim, codec)
    }

    /// CPU-based EnCodec decoder.
    fn encodec_decode_cpu(&self, input: &[f32], seq_len: usize, dim: usize, codec: &Model) -> Result<Vec<f32>> {
        // Convert input to channels-first [dim, seq_len]
        let mut channels_first = vec![half::f16::ZERO; dim * seq_len];
        for t in 0..seq_len {
            for d in 0..dim {
                channels_first[d * seq_len + t] = half::f16::from_f32(input[t * dim + d]);
            }
        }
        let h = Tensor::from_slice(&channels_first, Shape::from([dim, seq_len]), DType::F16, self.compute.device().info().id)?;

        // Layer 0: Conv1d(128 → 1024, k=7, padding=3)
        let h = self.encodec_conv1d(&h, "decoder.model.0", dim, 1024, seq_len, 7, 1, 3, codec)?;
        let mut current_len = seq_len;
        let mut current_ch = 1024usize;

        // Layer 1: LSTM (CPU)
        let h = self.encodec_lstm_cpu(&h, current_ch, current_len, codec)?;

        // Upsampling blocks
        let ratios = [8usize, 5, 4, 4];
        let channels = [512usize, 256, 128, 64];
        let mut layer_idx = 2usize;
        let mut h = h;

        for (&ratio, &out_ch) in ratios.iter().zip(channels.iter()) {
            // ELU
            h = self.elu_gpu(&h)?;
            layer_idx += 1;

            // ConvTranspose1d
            let kernel_size = ratio * 2;
            let padding = ratio / 2 + ratio % 2;
            h = self.encodec_conv1d_transpose(
                &h, &format!("decoder.model.{}", layer_idx),
                current_ch, out_ch, current_len, kernel_size, ratio, padding, codec,
            )?;
            current_len = (current_len - 1) * ratio - 2 * padding + kernel_size;
            current_ch = out_ch;
            layer_idx += 1;

            // ResBlock (skip — pass-through)
            if codec.get_weight(&format!("decoder.model.{}.block.1.conv.conv.weight_v", layer_idx)).is_some() {
                layer_idx += 1;
            }
        }

        // Final: ELU → Conv1d(64 → 1, k=7, padding=3)
        h = self.elu_gpu(&h)?;
        layer_idx += 1;
        h = self.encodec_conv1d(&h, &format!("decoder.model.{}", layer_idx),
            current_ch, 1, current_len, 7, 1, 3, codec)?;

        let f16_data: Vec<half::f16> = h.to_vec()?;
        let audio: Vec<f32> = f16_data.iter().map(|v| v.to_f32()).collect();
        Ok(audio)
    }

    /// GPU Conv1d with weight normalization (EnCodec).
    fn encodec_conv1d(
        &self, input: &Tensor, prefix: &str,
        c_in: usize, c_out: usize, l_in: usize,
        kernel_size: usize, stride: usize, padding: usize,
        codec: &Model,
    ) -> Result<Tensor> {
        let (weight_data, bias) = self.get_weight_normalized_codec(prefix, codec)?;

        let cb = self.compute.new_command_buffer();
        let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((c_out * l_out * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch(
            &cb, &self.kernels.conv1d,
            (l_out, c_out, 1), (1, 1, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &weight_data);
                gpu_ops::set_tensor_buffer(encoder, 2, &bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 6] = [c_in as u32, c_out as u32, l_in as u32, kernel_size as u32, stride as u32, padding as u32];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([c_out, l_out]), DType::F16, self.compute.device().info().id))
    }

    /// GPU ConvTranspose1d with weight normalization (EnCodec).
    fn encodec_conv1d_transpose(
        &self, input: &Tensor, prefix: &str,
        c_in: usize, c_out: usize, l_in: usize,
        kernel_size: usize, stride: usize, padding: usize,
        codec: &Model,
    ) -> Result<Tensor> {
        let (weight_data, bias) = self.get_weight_normalized_codec(prefix, codec)?;

        let cb = self.compute.new_command_buffer();
        let l_out = (l_in - 1) * stride - 2 * padding + kernel_size;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((c_out * l_out * 2) as u64, metal::MTLResourceOptions::StorageModeShared);

        self.compute.dispatch(
            &cb, &self.kernels.conv1d_transpose,
            (l_out, c_out, 1), (1, 1, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &weight_data);
                gpu_ops::set_tensor_buffer(encoder, 2, &bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 6] = [c_in as u32, c_out as u32, l_in as u32, kernel_size as u32, stride as u32, padding as u32];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([c_out, l_out]), DType::F16, self.compute.device().info().id))
    }

    /// Get weight-normalized weight from codec model: weight = g * (v / ||v||).
    fn get_weight_normalized_codec(&self, prefix: &str, codec: &Model) -> Result<(Tensor, Tensor)> {
        // Try conv.conv. and convtr.convtr. prefixes
        let (g_lt, v_lt, b_lt) = if let Some(g) = codec.get_weight(&format!("{}.conv.conv.weight_g", prefix)) {
            (g,
             codec.get_weight(&format!("{}.conv.conv.weight_v", prefix)).unwrap(),
             codec.get_weight(&format!("{}.conv.conv.bias", prefix)).unwrap())
        } else if let Some(g) = codec.get_weight(&format!("{}.convtr.convtr.weight_g", prefix)) {
            (g,
             codec.get_weight(&format!("{}.convtr.convtr.weight_v", prefix)).unwrap(),
             codec.get_weight(&format!("{}.convtr.convtr.bias", prefix)).unwrap())
        } else {
            return Err(Error::internal(format!("codec weight not found: {}", prefix)));
        };

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

        let c_out = g_data.len();
        let v_per_filter = v_data.len() / c_out;
        let mut weight = vec![half::f16::ZERO; v_data.len()];
        for co in 0..c_out {
            let start = co * v_per_filter;
            let end = start + v_per_filter;
            let norm: f32 = v_data[start..end].iter()
                .map(|x| { let f = x.to_f32(); f * f })
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            let g_val = g_data[co].to_f32();
            let scale_val = g_val / norm;
            for i in start..end {
                weight[i] = half::f16::from_f32(v_data[i].to_f32() * scale_val);
            }
        }

        let weight_tensor = Tensor::from_slice(&weight, v_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
        let bias_tensor = Tensor::from_slice(&b_data, b_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
        Ok((weight_tensor, bias_tensor))
    }

    /// CPU LSTM forward pass (2 layers, AMX-accelerated gate computation).
    fn encodec_lstm_cpu(&self, input: &Tensor, hidden_size: usize, seq_len: usize, codec: &Model) -> Result<Tensor> {
        use crate::tensor::ops::sgemm_transb_cpu;

        let f16_data: Vec<half::f16> = input.to_vec()?;
        let mut x = vec![0.0f32; seq_len * hidden_size];
        for t in 0..seq_len {
            for d in 0..hidden_size {
                x[t * hidden_size + d] = f16_data[d * seq_len + t].to_f32();
            }
        }

        let gate4 = 4 * hidden_size;

        for layer in 0..2 {
            let w_ih = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.weight_ih_l{}", layer), codec)?;
            let w_hh = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.weight_hh_l{}", layer), codec)?;
            let b_ih = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.bias_ih_l{}", layer), codec)?;
            let b_hh = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.bias_hh_l{}", layer), codec)?;

            // Pre-add biases: combined_bias = b_ih + b_hh
            let mut combined_bias = vec![0.0f32; gate4];
            for g in 0..gate4 { combined_bias[g] = b_ih[g] + b_hh[g]; }

            // Batch all input gate projections: gates_ih = X @ W_ih^T → [seq_len, 4*hidden]
            // w_ih is [4*hidden, hidden], X is [seq_len, hidden]
            let mut gates_ih = vec![0.0f32; seq_len * gate4];
            sgemm_transb_cpu(&x, &w_ih, &mut gates_ih, seq_len, gate4, hidden_size);

            let mut h = vec![0.0f32; hidden_size];
            let mut c = vec![0.0f32; hidden_size];
            let mut output = vec![0.0f32; seq_len * hidden_size];
            let mut gates_hh = vec![0.0f32; gate4];

            for t in 0..seq_len {
                // Hidden-to-gate: gates_hh = h @ W_hh^T → [1, 4*hidden]
                sgemm_transb_cpu(&h, &w_hh, &mut gates_hh, 1, gate4, hidden_size);

                // Combine: gates = gates_ih[t] + gates_hh + combined_bias
                for g in 0..gate4 {
                    let gate_val = gates_ih[t * gate4 + g] + gates_hh[g] + combined_bias[g];
                    let d = g % hidden_size;
                    let gate_idx = g / hidden_size;
                    match gate_idx {
                        0 => { // i gate
                            let i_gate = sigmoid(gate_val);
                            // Will combine below after g_gate
                            gates_hh[g] = i_gate; // reuse buffer for gate values
                        }
                        1 => { // f gate
                            let f_gate = sigmoid(gate_val);
                            c[d] = f_gate * c[d];
                        }
                        2 => { // g gate
                            let g_gate = gate_val.tanh();
                            c[d] += gates_hh[d] * g_gate; // i_gate * g_gate
                        }
                        3 => { // o gate
                            let o_gate = sigmoid(gate_val);
                            h[d] = o_gate * c[d].tanh();
                        }
                        _ => unreachable!(),
                    }
                }
                output[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&h);
            }
            x = output;
        }

        let mut channels_first = vec![half::f16::ZERO; hidden_size * seq_len];
        for t in 0..seq_len {
            for d in 0..hidden_size {
                channels_first[d * seq_len + t] = half::f16::from_f32(x[t * hidden_size + d]);
            }
        }
        Tensor::from_slice(&channels_first, Shape::from([hidden_size, seq_len]), DType::F16, self.compute.device().info().id)
    }

    fn read_codec_weight_f32(&self, name: &str, codec: &Model) -> Result<Vec<f32>> {
        let lt = codec.get_weight(name)
            .ok_or_else(|| Error::internal(format!("codec weight not found: {}", name)))?;
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
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(&output_buffer), 0);
            },
        );
        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, self.compute.device().info().id))
    }

    // ==================== GPU Helper Methods ====================

    fn w(&self, name: &str) -> Result<&LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| Error::internal(format!("weight not found: {}", name)))
    }

    fn linear_on(&self, cb: &metal::CommandBufferRef, input: &Tensor,
                 weight: &LazyTensor, bias: Option<&LazyTensor>,
                 m: usize, k: usize, n: usize) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let tile: usize = 16;

        self.compute.dispatch(
            cb, &self.kernels.common.linear,
            ((n + tile - 1) / tile, (m + tile - 1) / tile, 1), (tile, tile, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                if let Some(b) = bias {
                    set_lazy_buffer(encoder, 2, b);
                } else {
                    encoder.set_buffer(2, Some(&output_buffer), 0);
                }
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 4] = [m as u32, n as u32, k as u32, if bias.is_some() { 1 } else { 0 }];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, self.compute.device().info().id)
    }

    /// Linear projection using a row-offset into the weight matrix.
    /// Used for fused QKV: weight has shape [3*d_model, d_model], and we want
    /// rows [row_offset..row_offset+n] to get Q, K, or V separately.
    fn linear_with_row_offset_on(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        weight: &LazyTensor, m: usize, k: usize, n: usize,
        row_offset: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let tile: usize = 16;
        let byte_offset = row_offset * k * 2; // 2 bytes per F16

        self.compute.dispatch(
            cb, &self.kernels.common.linear,
            ((n + tile - 1) / tile, (m + tile - 1) / tile, 1), (tile, tile, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(weight.buffer()), byte_offset as u64);
                encoder.set_buffer(2, Some(&output_buffer), 0); // dummy bias
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 4] = [m as u32, n as u32, k as u32, 0]; // has_bias=0
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, self.compute.device().info().id)
    }

    fn layer_norm_on(&self, cb: &metal::CommandBufferRef, input: &Tensor,
                     weight: &LazyTensor, bias: &LazyTensor,
                     n: usize, d: usize) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.common.layer_norm, n,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
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

    fn embed_tokens_on(&self, cb: &metal::CommandBufferRef, token_ids: &[u32],
                       weight: &LazyTensor, d_model: usize, vocab_size: usize) -> Tensor {
        let seq_len = token_ids.len();
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((seq_len * d_model * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
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
                let vals: [u32; 3] = [vocab_size as u32, d_model as u32, seq_len as u32];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
                }
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
        let output_buffer = device.new_buffer((num_heads * head_dim * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.autoregressive_attention, num_heads,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, q);
                gpu_ops::set_tensor_buffer(encoder, 1, k_cache);
                gpu_ops::set_tensor_buffer(encoder, 2, v_cache);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 4] = [seq_pos as u32, num_heads as u32, num_heads as u32, head_dim as u32];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
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
        self.transpose_shd_to_hsd(cb, &q, &q_t, 1, num_heads, head_dim);

        // Scores = Q_t @ K_hsd^T → [H, 1, kv_seq]
        let scores_buffer = device.new_buffer((num_heads * kv_seq_len * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        {
            let tile: usize = 16;
            self.compute.dispatch(
                cb, &self.kernels.common.batched_linear,
                ((kv_seq_len + tile - 1) / tile, 1, num_heads), (tile, tile, 1),
                |encoder| {
                    gpu_ops::set_tensor_buffer(encoder, 0, &q_t);
                    gpu_ops::set_tensor_buffer(encoder, 1, k_hsd);
                    encoder.set_buffer(2, Some(&scores_buffer), 0);
                    let vals: [u32; 3] = [1, kv_seq_len as u32, head_dim as u32];
                    for (i, v) in vals.iter().enumerate() {
                        encoder.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
                    }
                },
            );
        }

        // Scaled softmax
        self.compute.dispatch_1d(
            cb, &self.kernels.common.row_softmax_scale, num_heads,
            |encoder| {
                encoder.set_buffer(0, Some(&scores_buffer), 0);
                let rows = num_heads as u32;
                let cols = kv_seq_len as u32;
                encoder.set_bytes(1, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(2, 4, &cols as *const u32 as *const _);
                encoder.set_bytes(3, 4, &scale as *const f32 as *const _);
            },
        );

        // Output = Scores @ V_hsd → [H, 1, head_dim]
        let output_t = Tensor::empty(
            Shape::from([num_heads, 1, head_dim]), DType::F16, device_id)?;
        {
            let tile: usize = 16;
            self.compute.dispatch(
                cb, &self.kernels.common.batched_matmul_nn,
                ((head_dim + tile - 1) / tile, 1, num_heads), (tile, tile, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&scores_buffer), 0);
                    gpu_ops::set_tensor_buffer(encoder, 1, v_hsd);
                    gpu_ops::set_tensor_buffer(encoder, 2, &output_t);
                    let vals: [u32; 3] = [1, head_dim as u32, kv_seq_len as u32];
                    for (i, v) in vals.iter().enumerate() {
                        encoder.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
                    }
                },
            );
        }

        // Transpose [H, 1, D] → [1, H*D]
        let output = Tensor::empty(
            Shape::from([1, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(cb, &output_t, &output, 1, num_heads, head_dim);
        output.reshape([1, num_heads * head_dim])
    }

    fn argmax_cpu(&self, logits: &Tensor, vocab_size: usize) -> Result<u32> {
        let data: Vec<half::f16> = logits.to_vec()?;
        let last_row = if data.len() > vocab_size { &data[data.len() - vocab_size..] } else { &data };
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// MAGNet pipeline (non-autoregressive masked generation).
///
/// Same transformer architecture as AudioGen (48 layers, fused QKV, etc.)
/// but uses bidirectional masked iterative decoding instead of autoregressive.
///
/// Generation loop:
/// 1. Start with all positions MASKED
/// 2. Run full-sequence forward pass (bidirectional self-attention)
/// 3. Predict logits for all positions
/// 4. Unmask the most confident tokens based on cosine schedule
/// 5. Repeat steps 2-4 for `num_decoding_steps` iterations
#[cfg(feature = "metal")]
pub struct MAGNetPipeline {
    model: Arc<Model>,
    codec_model: Option<Arc<Model>>,
    config: MAGNetConfig,
    text_encoder: T5Encoder,
    tokenizer: Option<Arc<HfTokenizer>>,
    compute: Arc<MetalCompute>,
    kernels: MAGNetKernels,
}

#[cfg(feature = "metal")]
struct MAGNetKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    embedding: Arc<ComputePipeline>,
    // EnCodec
    conv1d: Arc<ComputePipeline>,
    conv1d_transpose: Arc<ComputePipeline>,
    elu: Arc<ComputePipeline>,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for MAGNetPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl MAGNetPipeline {
    /// Create MAGNet pipeline.
    ///
    /// `model` — main model (state_dict.safetensors)
    /// `t5_model` — T5 encoder model
    /// `config` — MAGNet configuration
    /// `device` — Metal device
    pub fn new(
        model: Arc<Model>,
        t5_model: Arc<Model>,
        config: MAGNetConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device.clone()));

        let kernels = MAGNetKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_tanh_f16")?,
            embedding: compute.compile_pipeline("embedding", sources::EMBEDDING, "embedding_lookup_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            conv1d_transpose: compute.compile_pipeline("conv1d_transpose", sources::CONV1D, "conv1d_transpose_f16")?,
            elu: compute.compile_pipeline("elu", sources::GELU, "elu_f16")?,
        };

        // T5 encoder (base = 768-dim, or configurable)
        let t5_config = if config.t5_d_model == 1024 {
            T5Config::t5_v1_1_large()
        } else {
            T5Config::t5_v1_1_base()
        };
        let text_encoder = T5Encoder::new(t5_model, t5_config, device)?;

        Ok(Self {
            model, codec_model: None, config, text_encoder,
            tokenizer: None, compute, kernels,
        })
    }

    /// Set the text tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: Arc<HfTokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set the EnCodec codec model.
    pub fn with_codec(mut self, codec_model: Arc<Model>) -> Self {
        self.codec_model = Some(codec_model);
        self
    }

    /// Generate audio using masked iterative decoding.
    /// Returns mono f32 audio samples at 16kHz.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Tokenize
        let token_ids = if let Some(ref tok) = self.tokenizer {
            tok.encode(prompt)?
        } else {
            return Err(Error::internal("no tokenizer set"));
        };
        eprintln!("   Text tokens: {}", token_ids.len());

        // 2. Encode text via T5
        let encoder_out = self.text_encoder.encode(&token_ids)?;
        let enc_seq_len = token_ids.len();
        eprintln!("   Encoder output: [{}, {}]", enc_seq_len, config.t5_d_model);

        // 3. Project to decoder space
        let proj_w = self.w("condition_provider.conditioners.description.output_proj.weight")?;
        let proj_b = self.w("condition_provider.conditioners.description.output_proj.bias")?;
        let cb = self.compute.new_command_buffer();
        let encoder_projected = self.linear_on(
            &cb, &encoder_out, proj_w, Some(proj_b),
            enc_seq_len, config.t5_d_model, config.d_model,
        );
        cb.commit();
        cb.wait_until_completed();

        // 4. Pre-compute cross-attention K/V for all layers (HSD format)
        let num_heads = config.num_heads;
        let head_dim = config.d_model / num_heads;
        let device_id = self.compute.device().info().id;

        let mut cross_k_hsd = Vec::with_capacity(config.num_layers);
        let mut cross_v_hsd = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let in_proj = self.w(&format!(
                "transformer.layers.{}.cross_attention.in_proj_weight", layer
            ))?;
            let cb = self.compute.new_command_buffer();
            let k = self.linear_with_row_offset_on(
                &cb, &encoder_projected, in_proj,
                enc_seq_len, config.d_model, config.d_model, config.d_model,
            );
            let v = self.linear_with_row_offset_on(
                &cb, &encoder_projected, in_proj,
                enc_seq_len, config.d_model, config.d_model, 2 * config.d_model,
            );
            let k_t = Tensor::empty(Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
            let v_t = Tensor::empty(Shape::from([num_heads, enc_seq_len, head_dim]), DType::F16, device_id)?;
            self.transpose_shd_to_hsd(&cb, &k, &k_t, enc_seq_len, num_heads, head_dim);
            self.transpose_shd_to_hsd(&cb, &v, &v_t, enc_seq_len, num_heads, head_dim);
            cb.commit();
            cb.wait_until_completed();
            cross_k_hsd.push(k_t);
            cross_v_hsd.push(v_t);
        }
        eprintln!("   Cross-attention cached for {} layers", config.num_layers);

        // 5. Generate sinusoidal position embeddings
        let max_pos = max_tokens + 1; // +1 for BOS
        let pos_embed = self.sinusoidal_positions(max_pos, config.d_model)?;

        // 6. Masked iterative decoding
        let mask_token = config.codebook_size as u32; // 2048 = mask/BOS token
        let num_steps = config.num_decoding_steps;
        let seq_len = max_tokens;

        // Initialize all tokens as masked
        let mut codebook_tokens: Vec<Vec<u32>> = (0..config.num_codebooks)
            .map(|_| vec![mask_token; seq_len])
            .collect();
        let mut is_masked: Vec<Vec<bool>> = (0..config.num_codebooks)
            .map(|_| vec![true; seq_len])
            .collect();

        for step in 0..num_steps {
            // Ratio of tokens to unmask at this step (cosine schedule)
            let ratio = self.masking_ratio(step, num_steps);
            eprintln!("   Step {}/{}: unmask ratio {:.2}", step + 1, num_steps, ratio);

            // Forward pass through all layers with full sequence
            let logits = self.full_sequence_forward(
                &codebook_tokens, &pos_embed, seq_len,
                &cross_k_hsd, &cross_v_hsd, enc_seq_len,
            )?;

            // For each codebook, unmask most confident positions
            for c in 0..config.num_codebooks {
                let masked_count: usize = is_masked[c].iter().filter(|&&m| m).count();
                if masked_count == 0 { continue; }

                let num_to_unmask = ((masked_count as f32 * ratio).ceil() as usize).max(1);

                // Compute confidence for masked positions
                let mut confidences: Vec<(usize, u32, f32)> = Vec::new();
                for pos in 0..seq_len {
                    if !is_masked[c][pos] { continue; }

                    // Extract logits for this codebook at this position
                    let offset = (pos * config.num_codebooks + c) * config.codebook_size;
                    let head_logits = logits.slice(0, offset, offset + config.codebook_size)?;
                    let data: Vec<half::f16> = head_logits.to_vec()?;

                    // Softmax and argmax
                    let max_val = data.iter().map(|v| v.to_f32()).fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    let mut best_idx = 0u32;
                    let mut best_val = f32::NEG_INFINITY;
                    for (i, &v) in data.iter().enumerate() {
                        let f = v.to_f32();
                        exp_sum += (f - max_val).exp();
                        if f > best_val {
                            best_val = f;
                            best_idx = i as u32;
                        }
                    }
                    let confidence = (best_val - max_val).exp() / exp_sum;
                    confidences.push((pos, best_idx, confidence));
                }

                // Sort by confidence (highest first) and unmask top-k
                confidences.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                for &(pos, token, _) in confidences.iter().take(num_to_unmask) {
                    codebook_tokens[c][pos] = token;
                    is_masked[c][pos] = false;
                }
            }
        }

        eprintln!("   Decoding {} tokens to audio...", seq_len);

        // 7. Decode tokens to audio via EnCodec
        // Prepend BOS token for EnCodec compatibility
        let mut tokens_with_bos: Vec<Vec<u32>> = Vec::new();
        for c in 0..config.num_codebooks {
            let mut t = vec![mask_token];
            t.extend_from_slice(&codebook_tokens[c]);
            tokens_with_bos.push(t);
        }
        let audio = self.decode_audio(&tokens_with_bos)?;
        eprintln!("   Audio: {} samples ({:.1}s at {}Hz)",
            audio.len(), audio.len() as f64 / config.sample_rate as f64, config.sample_rate);

        Ok(audio)
    }

    /// Cosine masking schedule: fraction of remaining tokens to unmask at step `t` of `T`.
    fn masking_ratio(&self, step: usize, total_steps: usize) -> f32 {
        match self.config.masking_schedule.as_str() {
            "cosine" => {
                let s = step as f32 / total_steps as f32;
                let s1 = (step + 1) as f32 / total_steps as f32;
                let cos_s = (s * std::f32::consts::FRAC_PI_2).cos();
                let cos_s1 = (s1 * std::f32::consts::FRAC_PI_2).cos();
                if cos_s > 1e-6 { 1.0 - cos_s1 / cos_s } else { 1.0 }
            }
            _ => {
                // Linear schedule
                1.0 / (total_steps - step) as f32
            }
        }
    }

    /// Full-sequence forward pass through all decoder layers.
    /// Returns concatenated logits: [seq_len * num_codebooks * codebook_size].
    fn full_sequence_forward(
        &self,
        codebook_tokens: &[Vec<u32>],
        pos_embed: &Tensor,
        seq_len: usize,
        cross_k_hsd: &[Tensor],
        cross_v_hsd: &[Tensor],
        enc_seq_len: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let num_heads = config.num_heads;
        let head_dim = config.d_model / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Compute input embeddings: sum of 4 codebook embeddings + position
        let cb = self.compute.new_command_buffer();
        let mut input: Option<Tensor> = None;
        for c in 0..config.num_codebooks {
            let embed_w = self.w(&format!("emb.{}.weight", c))?;
            let embedded = self.embed_tokens_on(
                &cb, &codebook_tokens[c], embed_w, config.d_model, config.codebook_size + 1,
            );
            input = Some(match input {
                None => embedded,
                Some(prev) => self.add(&cb, &prev, &embedded),
            });
        }
        // Add position embeddings [0..seq_len]
        let pos_slice = pos_embed.slice(0, 0, seq_len)?
            .reshape([seq_len, config.d_model])?;
        let hidden = self.add(&cb, &input.unwrap(), &pos_slice);
        cb.commit();
        cb.wait_until_completed();

        // 2. Forward through all decoder layers
        let mut hidden = hidden;
        for layer in 0..config.num_layers {
            hidden = self.magnet_layer_forward(
                layer, hidden, seq_len, cross_k_hsd, cross_v_hsd,
                enc_seq_len, scale, num_heads, head_dim,
            )?;
        }

        // 3. Final layer norm
        let cb = self.compute.new_command_buffer();
        let ln_w = self.w("out_norm.weight")?;
        let ln_b = self.w("out_norm.bias")?;
        let normed = self.layer_norm_on(&cb, &hidden, ln_w, ln_b, seq_len, config.d_model);

        // 4. Compute logits for all codebook heads at all positions
        // Output: [seq_len, num_codebooks * codebook_size]
        let mut all_logits = Vec::new();
        for c in 0..config.num_codebooks {
            let head_w = self.w(&format!("linears.{}.weight", c))?;
            let logits = self.linear_on(&cb, &normed, head_w, None,
                seq_len, config.d_model, config.codebook_size);
            all_logits.push(logits);
        }
        cb.commit();
        cb.wait_until_completed();

        // Concatenate: interleave [pos0_cb0, pos0_cb1, ..., pos1_cb0, ...]
        let total_size = seq_len * config.num_codebooks * config.codebook_size;
        let mut cat_data: Vec<half::f16> = vec![half::f16::ZERO; total_size];
        for c in 0..config.num_codebooks {
            let data: Vec<half::f16> = all_logits[c].to_vec()?;
            // data is [seq_len, codebook_size], copy into interleaved format
            for pos in 0..seq_len {
                let src_offset = pos * config.codebook_size;
                let dst_offset = (pos * config.num_codebooks + c) * config.codebook_size;
                cat_data[dst_offset..dst_offset + config.codebook_size]
                    .copy_from_slice(&data[src_offset..src_offset + config.codebook_size]);
            }
        }
        Tensor::from_slice(&cat_data, Shape::from([total_size]), DType::F16, self.compute.device().info().id)
    }

    /// Single decoder layer with full-sequence bidirectional self-attention.
    fn magnet_layer_forward(
        &self,
        layer: usize,
        input: Tensor,
        seq_len: usize,
        cross_k_hsd: &[Tensor],
        cross_v_hsd: &[Tensor],
        enc_seq_len: usize,
        scale: f32,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let prefix = format!("transformer.layers.{}", layer);

        // === Self-attention (bidirectional — no causal mask) ===
        let cb = self.compute.new_command_buffer();
        let ln1_w = self.w(&format!("{}.norm1.weight", prefix))?;
        let ln1_b = self.w(&format!("{}.norm1.bias", prefix))?;
        let normed = self.layer_norm_on(&cb, &input, ln1_w, ln1_b, seq_len, config.d_model);

        // Fused QKV
        let sa_in_proj = self.w(&format!("{}.self_attn.in_proj_weight", prefix))?;
        let q = self.linear_with_row_offset_on(
            &cb, &normed, sa_in_proj, seq_len, config.d_model, config.d_model, 0,
        );
        let k = self.linear_with_row_offset_on(
            &cb, &normed, sa_in_proj, seq_len, config.d_model, config.d_model, config.d_model,
        );
        let v = self.linear_with_row_offset_on(
            &cb, &normed, sa_in_proj, seq_len, config.d_model, config.d_model, 2 * config.d_model,
        );

        // Batched matmul attention: Q@K^T → softmax → S@V
        let device_id = self.compute.device().info().id;
        let q_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        let k_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        let v_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(&cb, &q, &q_hsd, seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd(&cb, &k, &k_hsd, seq_len, num_heads, head_dim);
        self.transpose_shd_to_hsd(&cb, &v, &v_hsd, seq_len, num_heads, head_dim);

        // Scores = Q @ K^T → [H, S, S]
        let scores = self.batched_qk(&cb, &q_hsd, &k_hsd, num_heads, seq_len, seq_len, head_dim);

        // Scaled softmax (no causal mask — bidirectional)
        self.row_softmax(&cb, &scores, num_heads * seq_len, seq_len, scale);

        // Output = Scores @ V → [H, S, D]
        let attn_out_hsd = self.batched_sv(&cb, &scores, &v_hsd, num_heads, seq_len, seq_len, head_dim);

        // Transpose back to [S, H*D]
        let attn_out = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(&cb, &attn_out_hsd, &attn_out, seq_len, num_heads, head_dim);
        let attn_flat = attn_out.reshape([seq_len, config.d_model])?;

        let o_w = self.w(&format!("{}.self_attn.out_proj.weight", prefix))?;
        let sa_out = self.linear_on(&cb, &attn_flat, o_w, None, seq_len, config.d_model, config.d_model);
        let h = self.add(&cb, &input, &sa_out);
        cb.commit();
        cb.wait_until_completed();

        // === Cross-attention ===
        let cb = self.compute.new_command_buffer();
        let lnc_w = self.w(&format!("{}.norm_cross.weight", prefix))?;
        let lnc_b = self.w(&format!("{}.norm_cross.bias", prefix))?;
        let normed = self.layer_norm_on(&cb, &h, lnc_w, lnc_b, seq_len, config.d_model);

        // Cross Q from fused in_proj
        let ca_in_proj = self.w(&format!("{}.cross_attention.in_proj_weight", prefix))?;
        let cross_q = self.linear_with_row_offset_on(
            &cb, &normed, ca_in_proj, seq_len, config.d_model, config.d_model, 0,
        );

        // Q → [H, S, D]
        let cq_hsd = Tensor::empty(Shape::from([num_heads, seq_len, head_dim]), DType::F16, device_id)?;
        self.transpose_shd_to_hsd(&cb, &cross_q, &cq_hsd, seq_len, num_heads, head_dim);

        // Scores = Q @ K^T → [H, S, enc_seq]
        let cross_scores = self.batched_qk(&cb, &cq_hsd, &cross_k_hsd[layer], num_heads, seq_len, enc_seq_len, head_dim);
        self.row_softmax(&cb, &cross_scores, num_heads * seq_len, enc_seq_len, scale);

        // Output = Scores @ V → [H, S, D]
        let cross_out_hsd = self.batched_sv(&cb, &cross_scores, &cross_v_hsd[layer], num_heads, seq_len, enc_seq_len, head_dim);

        let cross_out_shd = Tensor::empty(Shape::from([seq_len, num_heads, head_dim]), DType::F16, device_id)?;
        self.transpose_hsd_to_shd(&cb, &cross_out_hsd, &cross_out_shd, seq_len, num_heads, head_dim);
        let cross_flat = cross_out_shd.reshape([seq_len, config.d_model])?;

        let co_w = self.w(&format!("{}.cross_attention.out_proj.weight", prefix))?;
        let cross_out = self.linear_on(&cb, &cross_flat, co_w, None, seq_len, config.d_model, config.d_model);
        let h = self.add(&cb, &h, &cross_out);

        // === FFN: LayerNorm → linear1 → GELU → linear2 → residual ===
        let ln2_w = self.w(&format!("{}.norm2.weight", prefix))?;
        let ln2_b = self.w(&format!("{}.norm2.bias", prefix))?;
        let normed = self.layer_norm_on(&cb, &h, ln2_w, ln2_b, seq_len, config.d_model);
        let fc1_w = self.w(&format!("{}.linear1.weight", prefix))?;
        let fc2_w = self.w(&format!("{}.linear2.weight", prefix))?;
        let ffn_h = self.linear_on(&cb, &normed, fc1_w, None, seq_len, config.d_model, config.ffn_dim);
        let ffn_h = self.activation(&cb, &self.kernels.gelu, &ffn_h);
        let ffn_out = self.linear_on(&cb, &ffn_h, fc2_w, None, seq_len, config.ffn_dim, config.d_model);
        let result = self.add(&cb, &h, &ffn_out);
        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    // ==================== EnCodec Decoder ====================

    /// Decode codebook tokens to audio waveform.
    fn decode_audio(&self, codebook_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        let config = &self.config;
        let codec = self.codec_model.as_ref()
            .ok_or_else(|| Error::internal("no codec model set"))?;

        let seq_len = codebook_tokens[0].len().saturating_sub(1);
        if seq_len == 0 { return Ok(Vec::new()); }

        let codec_dim = 128usize;

        // Dequantize: look up codebook embeddings and sum
        let mut summed = vec![0.0f32; seq_len * codec_dim];
        for q in 0..config.num_codebooks {
            let embed_key = format!("quantizer.vq.layers.{}.codebook.embed", q);
            if let Some(embed_w) = codec.get_weight(&embed_key) {
                let embed_data: Vec<half::f16> = unsafe {
                    let ptr = embed_w.buffer().contents() as *const half::f16;
                    std::slice::from_raw_parts(ptr, embed_w.shape().numel()).to_vec()
                };
                // Check if weights are f32 (MAGNet codec) vs f16 (AudioGen converted codec)
                let is_f32 = embed_w.shape().numel() > 0 && {
                    let byte_size = embed_w.buffer().length() as usize;
                    byte_size > embed_w.shape().numel() * 2
                };
                for (t, &token) in codebook_tokens[q][1..].iter().enumerate() {
                    let tok = token.min(config.codebook_size as u32 - 1) as usize;
                    if is_f32 {
                        let f32_ptr = embed_w.buffer().contents() as *const f32;
                        let f32_data = unsafe { std::slice::from_raw_parts(f32_ptr, embed_w.shape().numel()) };
                        for d in 0..codec_dim {
                            summed[t * codec_dim + d] += f32_data[tok * codec_dim + d];
                        }
                    } else {
                        for d in 0..codec_dim {
                            summed[t * codec_dim + d] += embed_data[tok * codec_dim + d].to_f32();
                        }
                    }
                }
            }
        }

        self.encodec_decode_cpu(&summed, seq_len, codec_dim, codec)
    }

    /// CPU-based EnCodec decoder.
    fn encodec_decode_cpu(&self, input: &[f32], seq_len: usize, dim: usize, codec: &Model) -> Result<Vec<f32>> {
        // Convert input to channels-first [dim, seq_len]
        let mut channels_first = vec![half::f16::ZERO; dim * seq_len];
        for t in 0..seq_len {
            for d in 0..dim {
                channels_first[d * seq_len + t] = half::f16::from_f32(input[t * dim + d]);
            }
        }
        let h = Tensor::from_slice(&channels_first, Shape::from([dim, seq_len]), DType::F16, self.compute.device().info().id)?;

        // Layer 0: Conv1d(128 → 1024, k=7, padding=3)
        let h = self.encodec_conv1d(&h, "decoder.model.0", dim, 1024, seq_len, 7, 1, 3, codec)?;
        let mut current_len = seq_len;
        let mut current_ch = 1024usize;

        // Layer 1: LSTM (CPU)
        let h = self.encodec_lstm_cpu(&h, current_ch, current_len, codec)?;

        // Upsampling blocks
        let ratios = [8usize, 5, 4, 4];
        let channels = [512usize, 256, 128, 64];
        let mut layer_idx = 2usize;
        let mut h = h;
        for (&ratio, &out_ch) in ratios.iter().zip(channels.iter()) {
            h = self.elu_gpu(&h)?;
            layer_idx += 1;
            let kernel_size = ratio * 2;
            let padding = ratio / 2 + ratio % 2;
            h = self.encodec_conv1d_transpose(
                &h, &format!("decoder.model.{}", layer_idx),
                current_ch, out_ch, current_len, kernel_size, ratio, padding, codec,
            )?;
            current_len = (current_len - 1) * ratio - 2 * padding + kernel_size;
            current_ch = out_ch;
            layer_idx += 1;
            if codec.get_weight(&format!("decoder.model.{}.block.1.conv.conv.weight_v", layer_idx)).is_some() {
                layer_idx += 1;
            }
        }

        // Final: ELU → Conv1d(64 → 1, k=7, padding=3)
        h = self.elu_gpu(&h)?;
        layer_idx += 1;
        h = self.encodec_conv1d(&h, &format!("decoder.model.{}", layer_idx),
            current_ch, 1, current_len, 7, 1, 3, codec)?;

        let f16_data: Vec<half::f16> = h.to_vec()?;
        Ok(f16_data.iter().map(|v| v.to_f32()).collect())
    }

    /// GPU Conv1d with weight normalization (EnCodec).
    fn encodec_conv1d(
        &self, input: &Tensor, prefix: &str,
        c_in: usize, c_out: usize, l_in: usize,
        kernel_size: usize, stride: usize, padding: usize,
        codec: &Model,
    ) -> Result<Tensor> {
        let (weight_data, bias) = self.get_weight_normalized_codec(prefix, codec)?;
        let cb = self.compute.new_command_buffer();
        let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((c_out * l_out * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch(
            &cb, &self.kernels.conv1d,
            (l_out, c_out, 1), (1, 1, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &weight_data);
                gpu_ops::set_tensor_buffer(encoder, 2, &bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 6] = [c_in as u32, c_out as u32, l_in as u32, kernel_size as u32, stride as u32, padding as u32];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([c_out, l_out]), DType::F16, self.compute.device().info().id))
    }

    /// GPU ConvTranspose1d with weight normalization (EnCodec).
    fn encodec_conv1d_transpose(
        &self, input: &Tensor, prefix: &str,
        c_in: usize, c_out: usize, l_in: usize,
        kernel_size: usize, stride: usize, padding: usize,
        codec: &Model,
    ) -> Result<Tensor> {
        let (weight_data, bias) = self.get_weight_normalized_codec(prefix, codec)?;
        let cb = self.compute.new_command_buffer();
        let l_out = (l_in - 1) * stride - 2 * padding + kernel_size;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((c_out * l_out * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch(
            &cb, &self.kernels.conv1d_transpose,
            (l_out, c_out, 1), (1, 1, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &weight_data);
                gpu_ops::set_tensor_buffer(encoder, 2, &bias);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 6] = [c_in as u32, c_out as u32, l_in as u32, kernel_size as u32, stride as u32, padding as u32];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, Shape::from([c_out, l_out]), DType::F16, self.compute.device().info().id))
    }

    /// Get weight-normalized weight from codec model: weight = g * (v / ||v||).
    /// Handles both f16 and f32 codec weights.
    fn get_weight_normalized_codec(&self, prefix: &str, codec: &Model) -> Result<(Tensor, Tensor)> {
        let (g_lt, v_lt, b_lt) = if let Some(g) = codec.get_weight(&format!("{}.conv.conv.weight_g", prefix)) {
            (g,
             codec.get_weight(&format!("{}.conv.conv.weight_v", prefix)).unwrap(),
             codec.get_weight(&format!("{}.conv.conv.bias", prefix)).unwrap())
        } else if let Some(g) = codec.get_weight(&format!("{}.convtr.convtr.weight_g", prefix)) {
            (g,
             codec.get_weight(&format!("{}.convtr.convtr.weight_v", prefix)).unwrap(),
             codec.get_weight(&format!("{}.convtr.convtr.bias", prefix)).unwrap())
        } else {
            return Err(Error::internal(format!("codec weight not found: {}", prefix)));
        };

        // Detect f32 vs f16 by comparing byte size to numel
        let is_f32 = g_lt.buffer().length() as usize >= g_lt.shape().numel() * 4;

        if is_f32 {
            // Read as f32 directly
            let g_ptr = g_lt.buffer().contents() as *const f32;
            let v_ptr = v_lt.buffer().contents() as *const f32;
            let b_ptr = b_lt.buffer().contents() as *const f32;
            let g_data: Vec<f32> = unsafe { std::slice::from_raw_parts(g_ptr, g_lt.shape().numel()).to_vec() };
            let v_data: Vec<f32> = unsafe { std::slice::from_raw_parts(v_ptr, v_lt.shape().numel()).to_vec() };
            let b_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(b_ptr, b_lt.shape().numel()).to_vec() };

            let c_out = g_data.len();
            let v_per_filter = v_data.len() / c_out;
            let mut weight = vec![half::f16::ZERO; v_data.len()];
            for co in 0..c_out {
                let start = co * v_per_filter;
                let end = start + v_per_filter;
                let norm: f32 = v_data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let scale_val = g_data[co] / norm;
                for i in start..end {
                    weight[i] = half::f16::from_f32(v_data[i] * scale_val);
                }
            }
            let b_data: Vec<half::f16> = b_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
            let weight_tensor = Tensor::from_slice(&weight, v_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
            let bias_tensor = Tensor::from_slice(&b_data, b_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
            Ok((weight_tensor, bias_tensor))
        } else {
            // f16 path (same as AudioGen)
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
            let c_out = g_data.len();
            let v_per_filter = v_data.len() / c_out;
            let mut weight = vec![half::f16::ZERO; v_data.len()];
            for co in 0..c_out {
                let start = co * v_per_filter;
                let end = start + v_per_filter;
                let norm: f32 = v_data[start..end].iter().map(|x| { let f = x.to_f32(); f * f }).sum::<f32>().sqrt().max(1e-12);
                let scale_val = g_data[co].to_f32() / norm;
                for i in start..end {
                    weight[i] = half::f16::from_f32(v_data[i].to_f32() * scale_val);
                }
            }
            let weight_tensor = Tensor::from_slice(&weight, v_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
            let bias_tensor = Tensor::from_slice(&b_data, b_lt.shape().clone(), DType::F16, self.compute.device().info().id)?;
            Ok((weight_tensor, bias_tensor))
        }
    }

    /// CPU LSTM forward pass (2 layers).
    fn encodec_lstm_cpu(&self, input: &Tensor, hidden_size: usize, seq_len: usize, codec: &Model) -> Result<Tensor> {
        let f16_data: Vec<half::f16> = input.to_vec()?;
        let mut x = vec![0.0f32; seq_len * hidden_size];
        for t in 0..seq_len {
            for d in 0..hidden_size {
                x[t * hidden_size + d] = f16_data[d * seq_len + t].to_f32();
            }
        }

        for layer in 0..2 {
            let w_ih = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.weight_ih_l{}", layer), codec)?;
            let w_hh = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.weight_hh_l{}", layer), codec)?;
            let b_ih = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.bias_ih_l{}", layer), codec)?;
            let b_hh = self.read_codec_weight_f32(&format!("decoder.model.1.lstm.bias_hh_l{}", layer), codec)?;

            let mut h = vec![0.0f32; hidden_size];
            let mut c = vec![0.0f32; hidden_size];
            let mut output = vec![0.0f32; seq_len * hidden_size];
            for t in 0..seq_len {
                let xt = &x[t * hidden_size..(t + 1) * hidden_size];
                let mut gates = vec![0.0f32; 4 * hidden_size];
                for g in 0..4 * hidden_size {
                    let mut sum = b_ih[g] + b_hh[g];
                    for k in 0..hidden_size {
                        sum += w_ih[g * hidden_size + k] * xt[k];
                        sum += w_hh[g * hidden_size + k] * h[k];
                    }
                    gates[g] = sum;
                }
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

        let mut channels_first = vec![half::f16::ZERO; hidden_size * seq_len];
        for t in 0..seq_len {
            for d in 0..hidden_size {
                channels_first[d * seq_len + t] = half::f16::from_f32(x[t * hidden_size + d]);
            }
        }
        Tensor::from_slice(&channels_first, Shape::from([hidden_size, seq_len]), DType::F16, self.compute.device().info().id)
    }

    fn read_codec_weight_f32(&self, name: &str, codec: &Model) -> Result<Vec<f32>> {
        let lt = codec.get_weight(name)
            .ok_or_else(|| Error::internal(format!("codec weight not found: {}", name)))?;
        // Detect f32 vs f16
        let is_f32 = lt.buffer().length() as usize >= lt.shape().numel() * 4;
        if is_f32 {
            let ptr = lt.buffer().contents() as *const f32;
            Ok(unsafe { std::slice::from_raw_parts(ptr, lt.shape().numel()).to_vec() })
        } else {
            let ptr = lt.buffer().contents() as *const half::f16;
            let f16_data = unsafe { std::slice::from_raw_parts(ptr, lt.shape().numel()) };
            Ok(f16_data.iter().map(|v| v.to_f32()).collect())
        }
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
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(&output_buffer), 0);
            },
        );
        cb.commit();
        cb.wait_until_completed();
        Ok(Tensor::from_metal_buffer(output_buffer, input.shape().clone(), DType::F16, self.compute.device().info().id))
    }

    // ==================== GPU Helper Methods ====================

    fn w(&self, name: &str) -> Result<&LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| Error::internal(format!("weight not found: {}", name)))
    }

    fn linear_on(&self, cb: &metal::CommandBufferRef, input: &Tensor,
                 weight: &LazyTensor, bias: Option<&LazyTensor>,
                 m: usize, k: usize, n: usize) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let tile: usize = 16;
        self.compute.dispatch(
            cb, &self.kernels.common.linear,
            ((n + tile - 1) / tile, (m + tile - 1) / tile, 1), (tile, tile, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                set_lazy_buffer(encoder, 1, weight);
                if let Some(b) = bias {
                    set_lazy_buffer(encoder, 2, b);
                } else {
                    encoder.set_buffer(2, Some(&output_buffer), 0);
                }
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 4] = [m as u32, n as u32, k as u32, if bias.is_some() { 1 } else { 0 }];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, self.compute.device().info().id)
    }

    fn linear_with_row_offset_on(
        &self, cb: &metal::CommandBufferRef, input: &Tensor,
        weight: &LazyTensor, m: usize, k: usize, n: usize,
        row_offset: usize,
    ) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((m * n * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        let tile: usize = 16;
        let byte_offset = row_offset * k * 2;
        self.compute.dispatch(
            cb, &self.kernels.common.linear,
            ((n + tile - 1) / tile, (m + tile - 1) / tile, 1), (tile, tile, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                encoder.set_buffer(1, Some(weight.buffer()), byte_offset as u64);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 4] = [m as u32, n as u32, k as u32, 0];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([m, n]), DType::F16, self.compute.device().info().id)
    }

    fn layer_norm_on(&self, cb: &metal::CommandBufferRef, input: &Tensor,
                     weight: &LazyTensor, bias: &LazyTensor,
                     n: usize, d: usize) -> Tensor {
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((n * d * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
        self.compute.dispatch_1d(
            cb, &self.kernels.common.layer_norm, n,
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
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

    fn embed_tokens_on(&self, cb: &metal::CommandBufferRef, token_ids: &[u32],
                       weight: &LazyTensor, d_model: usize, vocab_size: usize) -> Tensor {
        let seq_len = token_ids.len();
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer((seq_len * d_model * 2) as u64, metal::MTLResourceOptions::StorageModeShared);
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
                let vals: [u32; 3] = [vocab_size as u32, d_model as u32, seq_len as u32];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((3 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Tensor::from_metal_buffer(output_buffer, Shape::from([seq_len, d_model]), DType::F16, self.compute.device().info().id)
    }

    fn sinusoidal_positions(&self, max_len: usize, d_model: usize) -> Result<Tensor> {
        let mut data = vec![half::f16::ZERO; max_len * d_model];
        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
                data[pos * d_model + 2 * i] = half::f16::from_f32(angle.sin() as f32);
                data[pos * d_model + 2 * i + 1] = half::f16::from_f32(angle.cos() as f32);
            }
        }
        Tensor::from_slice(&data, Shape::from([max_len, d_model]), DType::F16, self.compute.device().info().id)
    }

}
