//! Flan-T5 / T5 encoder-decoder for seq2seq generation.
//!
//! Extends the existing T5 encoder with a decoder for conditional generation:
//!   Input text → T5 encoder (24 layers) → encoder hidden states
//!   → T5 decoder (24 layers, self-attn + cross-attn) → logits → tokens
//!
//! Key differences from encoder-only T5:
//!   - Decoder has causal self-attention (lower triangular mask)
//!   - Decoder has cross-attention to encoder outputs
//!   - Decoder has its own relative position bias
//!   - Uses gated-GELU FFN (same as encoder)
//!   - Shared embedding + tied lm_head

use crate::core::Result;
use crate::inference::architecture::t5::T5Config;

#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::architecture::t5::{T5Encoder, T5Config, borrow_tensor};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, LazyTensor};

/// Flan-T5 configuration.
#[derive(Debug, Clone)]
pub struct FlanT5Config {
    /// Model dimension (encoder and decoder).
    pub d_model: usize,
    /// FFN inner dimension.
    pub d_ff: usize,
    /// Dimension per attention head.
    pub d_kv: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of encoder layers.
    pub num_encoder_layers: usize,
    /// Number of decoder layers.
    pub num_decoder_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of relative position bias buckets.
    pub num_buckets: usize,
    /// Maximum relative distance.
    pub max_distance: usize,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
    /// Decoder start token ID.
    pub decoder_start_token_id: u32,
    /// EOS token ID.
    pub eos_token_id: u32,
    /// PAD token ID.
    pub pad_token_id: u32,
    /// FFN type: "gated-gelu" (Flan-T5) or "relu" (original T5).
    pub feed_forward_proj: String,
    /// Whether to tie word embeddings.
    pub tie_word_embeddings: bool,
}

impl Default for FlanT5Config {
    /// Flan-T5-large defaults.
    fn default() -> Self {
        Self {
            d_model: 1024,
            d_ff: 2816,
            d_kv: 64,
            num_heads: 16,
            num_encoder_layers: 24,
            num_decoder_layers: 24,
            vocab_size: 32128,
            num_buckets: 32,
            max_distance: 128,
            layer_norm_eps: 1e-6,
            decoder_start_token_id: 0,
            eos_token_id: 1,
            pad_token_id: 0,
            feed_forward_proj: "gated-gelu".to_string(),
            tie_word_embeddings: false,
        }
    }
}

impl FlanT5Config {
    /// Parse from config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| crate::core::Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| crate::core::Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();
        if let Some(v) = json.get("d_model").and_then(|v| v.as_u64()) { c.d_model = v as usize; }
        if let Some(v) = json.get("d_ff").and_then(|v| v.as_u64()) { c.d_ff = v as usize; }
        if let Some(v) = json.get("d_kv").and_then(|v| v.as_u64()) { c.d_kv = v as usize; }
        if let Some(v) = json.get("num_heads").and_then(|v| v.as_u64()) { c.num_heads = v as usize; }
        if let Some(v) = json.get("num_layers").and_then(|v| v.as_u64()) { c.num_encoder_layers = v as usize; }
        if let Some(v) = json.get("num_decoder_layers").and_then(|v| v.as_u64()) { c.num_decoder_layers = v as usize; }
        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) { c.vocab_size = v as usize; }
        if let Some(v) = json.get("relative_attention_num_buckets").and_then(|v| v.as_u64()) { c.num_buckets = v as usize; }
        if let Some(v) = json.get("relative_attention_max_distance").and_then(|v| v.as_u64()) { c.max_distance = v as usize; }
        if let Some(v) = json.get("layer_norm_epsilon").and_then(|v| v.as_f64()) { c.layer_norm_eps = v as f32; }
        if let Some(v) = json.get("decoder_start_token_id").and_then(|v| v.as_u64()) { c.decoder_start_token_id = v as u32; }
        if let Some(v) = json.get("eos_token_id").and_then(|v| v.as_u64()) { c.eos_token_id = v as u32; }
        if let Some(v) = json.get("pad_token_id").and_then(|v| v.as_u64()) { c.pad_token_id = v as u32; }
        if let Some(v) = json.get("feed_forward_proj").and_then(|v| v.as_str()) { c.feed_forward_proj = v.to_string(); }
        if let Some(v) = json.get("tie_word_embeddings").and_then(|v| v.as_bool()) { c.tie_word_embeddings = v; }
        Ok(c)
    }

    /// Convert to T5Config for encoder reuse.
    pub fn encoder_config(&self) -> T5Config {
        T5Config {
            d_model: self.d_model,
            d_ff: self.d_ff,
            d_kv: self.d_kv,
            num_heads: self.num_heads,
            num_layers: self.num_encoder_layers,
            vocab_size: self.vocab_size,
            num_buckets: self.num_buckets,
            max_distance: self.max_distance,
            scalable_attention: false,
            layer_norm_epsilon: self.layer_norm_eps,
            is_gated_ffn: self.feed_forward_proj.contains("gated"),
        }
    }
}

/// Flan-T5 encoder-decoder pipeline.
#[cfg(feature = "metal")]
pub struct FlanT5Pipeline {
    /// Shared model weights.
    model: Arc<Model>,
    /// T5Encoder handles encoder forward pass + provides GPU ops.
    encoder: T5Encoder,
    config: FlanT5Config,
}

#[cfg(feature = "metal")]
impl FlanT5Pipeline {
    /// Create Flan-T5 pipeline.
    pub fn new(model: Arc<Model>, config: FlanT5Config, device: Arc<MetalDevice>) -> Result<Self> {
        let enc_config = config.encoder_config();
        let encoder = T5Encoder::new(Arc::clone(&model), enc_config, device)?;
        Ok(Self { model, encoder, config })
    }

    /// Encode input text through T5 encoder.
    pub fn encode(&self, token_ids: &[u32]) -> Result<Tensor> {
        self.encoder.encode(token_ids)
    }

    /// Get weight by name from model.
    fn w(&self, name: &str) -> Result<&LazyTensor> {
        self.model.get_weight(name)
            .ok_or_else(|| crate::core::Error::internal(format!("FlanT5 weight not found: {}", name)))
    }

    /// Decode tokens autoregressively given encoder outputs.
    pub fn decode(
        &self,
        encoder_output: &Tensor,
        enc_seq_len: usize,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let config = &self.config;
        let enc = &self.encoder;
        let device_id = enc.compute.device().info().id;
        let d_model = config.d_model;
        let num_heads = config.num_heads;
        let d_kv = config.d_kv;
        let eps = config.layer_norm_eps;
        let is_gated = config.feed_forward_proj.contains("gated");

        // Pre-compute encoder K/V for cross-attention (static across decode steps)
        let mut enc_kv_cache: Vec<(Tensor, Tensor)> = Vec::with_capacity(config.num_decoder_layers);
        for layer in 0..config.num_decoder_layers {
            let prefix = format!("decoder.block.{}.layer.1", layer);
            let k_w = self.w(&format!("{}.EncDecAttention.k.weight", prefix))?;
            let v_w = self.w(&format!("{}.EncDecAttention.v.weight", prefix))?;
            let enc_k = enc.gpu_linear_lazy(encoder_output, k_w, enc_seq_len, d_model, d_model)?;
            let enc_v = enc.gpu_linear_lazy(encoder_output, v_w, enc_seq_len, d_model, d_model)?;
            enc_kv_cache.push((enc_k, enc_v));
        }

        let mut output_tokens = vec![config.decoder_start_token_id];

        for _step in 0..max_tokens {
            let dec_seq = output_tokens.len();

            // 1. Embed all decoder tokens → F16, then convert to F32 residual
            let embed_w = self.w("shared.weight")
                .or_else(|_| self.w("decoder.embed_tokens.weight"))?;
            let embed_f16 = enc.gpu_embedding(embed_w, &output_tokens, d_model, device_id)?;
            let mut h = enc.gpu_f16_to_f32(&embed_f16)?;

            // 2. Causal mask: [num_heads, dec_seq, dec_seq] with -inf for future positions
            let causal_bias = self.create_causal_bias(num_heads, dec_seq, device_id)?;

            // 3. Decoder blocks
            let mut block0_self_bias: Option<Tensor> = None;

            for layer in 0..config.num_decoder_layers {
                let prefix = format!("decoder.block.{}.layer", layer);

                // --- Causal self-attention ---
                let ln0_w = self.w(&format!("{}.0.layer_norm.weight", prefix))?;
                let normed = enc.gpu_rms_norm_f32(&h, ln0_w, d_model, eps)?;

                let q_w = self.w(&format!("{}.0.SelfAttention.q.weight", prefix))?;
                let k_w = self.w(&format!("{}.0.SelfAttention.k.weight", prefix))?;
                let v_w = self.w(&format!("{}.0.SelfAttention.v.weight", prefix))?;
                let o_w = self.w(&format!("{}.0.SelfAttention.o.weight", prefix))?;

                let q = enc.gpu_linear_lazy(&normed, q_w, dec_seq, d_model, d_model)?;
                let k = enc.gpu_linear_lazy(&normed, k_w, dec_seq, d_model, d_model)?;
                let v = enc.gpu_linear_lazy(&normed, v_w, dec_seq, d_model, d_model)?;

                // Relative position bias for self-attention
                let self_bias = if layer == 0 {
                    let bias_w = self.w(&format!("{}.0.SelfAttention.relative_attention_bias.weight", prefix))?;
                    let b = enc.gpu_relative_position_bias(bias_w, dec_seq, dec_seq, false)?;
                    block0_self_bias = Some(b.clone());
                    b
                } else {
                    block0_self_bias.clone().unwrap()
                };

                // Add causal mask to self-attention bias
                let masked_bias = enc.gpu_add(&self_bias, &causal_bias)?;

                // Self-attention with causal mask (F32 scores pipeline)
                let self_attn = enc.gpu_attention_f32(&q, &k, &v, &masked_bias,
                    dec_seq, num_heads, d_kv)?;
                let projected = enc.gpu_linear_lazy_f32(&self_attn, o_w, dec_seq, d_model, d_model)?;
                h = enc.gpu_residual_add_f32_f32(&h, &projected)?;

                // --- Cross-attention ---
                let ln1_w = self.w(&format!("{}.1.layer_norm.weight", prefix))?;
                let normed = enc.gpu_rms_norm_f32(&h, ln1_w, d_model, eps)?;

                let cross_q_w = self.w(&format!("{}.1.EncDecAttention.q.weight", prefix))?;
                let cross_o_w = self.w(&format!("{}.1.EncDecAttention.o.weight", prefix))?;

                let cross_q = enc.gpu_linear_lazy(&normed, cross_q_w, dec_seq, d_model, d_model)?;
                let (enc_k, enc_v) = &enc_kv_cache[layer];

                // Cross-attention: Q from decoder, K/V from encoder (F32 scores, no bias)
                let cross_attn = enc.gpu_cross_attention_f32(&cross_q, enc_k, enc_v,
                    dec_seq, enc_seq_len, num_heads, d_kv)?;
                let cross_proj = enc.gpu_linear_lazy_f32(&cross_attn, cross_o_w, dec_seq, d_model, d_model)?;
                h = enc.gpu_residual_add_f32_f32(&h, &cross_proj)?;

                // --- FFN ---
                let ln2_w = self.w(&format!("{}.2.layer_norm.weight", prefix))?;
                let normed2 = enc.gpu_rms_norm_f32(&h, ln2_w, d_model, eps)?;

                let ffn_out = if is_gated {
                    let wi0 = self.w(&format!("{}.2.DenseReluDense.wi_0.weight", prefix))?;
                    let wi1 = self.w(&format!("{}.2.DenseReluDense.wi_1.weight", prefix))?;
                    let wo = self.w(&format!("{}.2.DenseReluDense.wo.weight", prefix))?;
                    let gate = enc.gpu_linear_lazy(&normed2, wi0, dec_seq, d_model, config.d_ff)?;
                    let up = enc.gpu_linear_lazy(&normed2, wi1, dec_seq, d_model, config.d_ff)?;
                    let gated = enc.gpu_geglu_f32(&gate, &up, dec_seq * config.d_ff)?;
                    enc.gpu_linear_f32_in_f32_out(&gated, wo, dec_seq, config.d_ff, d_model)?
                } else {
                    let wi = self.w(&format!("{}.2.DenseReluDense.wi.weight", prefix))?;
                    let wo = self.w(&format!("{}.2.DenseReluDense.wo.weight", prefix))?;
                    let hidden = enc.gpu_linear_lazy(&normed2, wi, dec_seq, d_model, config.d_ff)?;
                    let activated = enc.gpu_relu_f32(&hidden, dec_seq * config.d_ff)?;
                    enc.gpu_linear_f32_in_f32_out(&activated, wo, dec_seq, config.d_ff, d_model)?
                };
                h = enc.gpu_residual_add_f32_f32(&h, &ffn_out)?;
            }

            // 4. Final layer norm: F32 → F16
            let final_ln = self.w("decoder.final_layer_norm.weight")?;
            let h_f16 = enc.gpu_rms_norm_f32(&h, final_ln, d_model, eps)?;

            // 5. LM head projection (last token only)
            // h_f16 is [dec_seq, d_model], project last row with byte offset
            let lm_head = self.w("lm_head.weight")
                .or_else(|_| self.w("shared.weight"))?;
            let last_row_offset = (dec_seq - 1) * d_model * 2; // 2 bytes per F16
            let logits = self.gpu_linear_with_offset(enc, &h_f16, lm_head, last_row_offset, 1, d_model, config.vocab_size)?;

            // 6. Argmax on CPU
            let logit_data: Vec<half::f16> = logits.to_vec()?;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx: u32 = 0;
            for (i, &v) in logit_data.iter().enumerate() {
                let f = v.to_f32();
                if f > max_val {
                    max_val = f;
                    max_idx = i as u32;
                }
            }

            output_tokens.push(max_idx);

            if max_idx == config.eos_token_id {
                break;
            }
        }

        Ok(output_tokens)
    }

    /// Full seq2seq: encode input → decode output.
    pub fn generate(&self, input_ids: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        // T5 requires EOS token at end of encoder input
        let mut enc_input = input_ids.to_vec();
        if enc_input.last() != Some(&self.config.eos_token_id) {
            enc_input.push(self.config.eos_token_id);
        }
        let enc_seq_len = enc_input.len();
        let encoder_output = self.encode(&enc_input)?;
        self.decode(&encoder_output, enc_seq_len, max_tokens)
    }

    /// Create causal bias tensor: [num_heads, q_len, q_len] where
    /// valid positions (j <= i) = 0 and future positions (j > i) = -inf.
    fn create_causal_bias(&self, num_heads: usize, seq_len: usize, device_id: crate::hal::DeviceId) -> Result<Tensor> {
        let neg_inf = half::f16::from_f32(-65504.0); // largest finite f16 negative
        let zero = half::f16::ZERO;
        let total = num_heads * seq_len * seq_len;
        let mut data = vec![zero; total];

        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        data[h * seq_len * seq_len + i * seq_len + j] = neg_inf;
                    }
                }
            }
        }

        Tensor::from_slice(
            &data,
            Shape::from([num_heads, seq_len, seq_len]),
            DType::F16,
            device_id,
        )
    }

    /// Linear projection with input byte offset (for extracting a specific row from a larger tensor).
    fn gpu_linear_with_offset(
        &self,
        enc: &T5Encoder,
        input: &Tensor,
        weight: &LazyTensor,
        input_byte_offset: usize,
        m: usize, k: usize, n: usize,
    ) -> Result<Tensor> {
        let output = Tensor::empty(Shape::from([m, n]), DType::F16, input.device())?;
        let cb = enc.compute.new_command_buffer();
        let x_buf = borrow_tensor(input)?;
        let o_buf = borrow_tensor(&output)?;
        let c_m = m as u32;
        let c_n = n as u32;
        let c_k = k as u32;
        let has_bias: u32 = 0;

        enc.compute.dispatch_async(cb.as_ref(), &enc.kernels.linear,
            ((n + 15) / 16, (m + 15) / 16, 1), (16, 16, 1), |encoder| {
                encoder.set_buffer(0, Some(x_buf.as_ref()), input_byte_offset as u64);
                encoder.set_buffer(1, Some(weight.buffer()), 0);
                encoder.set_buffer(2, Some(x_buf.as_ref()), 0); // dummy bias
                encoder.set_buffer(3, Some(o_buf.as_ref()), 0);
                encoder.set_bytes(4, 4, &c_m as *const u32 as *const _);
                encoder.set_bytes(5, 4, &c_n as *const u32 as *const _);
                encoder.set_bytes(6, 4, &c_k as *const u32 as *const _);
                encoder.set_bytes(7, 4, &has_bias as *const u32 as *const _);
            });
        cb.commit();
        cb.wait_until_completed();
        Ok(output)
    }
}
