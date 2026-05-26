//! Sesame CSM: Conversational Speech Model (~1B params, Apache 2.0).
//!
//! Architecture: two-tower approach for conversational speech synthesis.
//!
//!   Backbone Transformer (Llama-based, ~1B):
//!     Processes interleaved text + audio tokens from conversation context.
//!     Multiple speakers tracked via `[spk_0]`, `[spk_1]` tokens.
//!     Outputs hidden states for audio prediction.
//!     Weight prefix: `backbone.`
//!
//!   Decoder Head (smaller transformer):
//!     Takes backbone hidden states.
//!     Generates Mimi audio codec tokens autoregressively.
//!     Multiple codebook prediction (similar to Bark's fine model).
//!     Weight prefix: `decoder.`
//!
//!   Mimi Audio Codec (by Kyutai):
//!     12 codebooks, 2048 entries each, 12.5 Hz frame rate (80ms per frame).
//!     24kHz audio output, streaming-capable.
//!     Weight prefix: `codec.`
//!
//! Conversation context:
//!   Text and audio segments are interleaved with speaker markers.
//!   `[spk_0]`, `[spk_1]` tokens identify speakers for natural turn-taking.
//!
//! Pipeline: text + context -> backbone -> decoder -> Mimi codec decode -> 24kHz audio.

#[cfg(feature = "metal")]
use tracing::debug;
use crate::core::{Error, Result};
#[cfg(feature = "metal")]
use crate::tensor::{DType, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;
#[cfg(feature = "metal")]
use crate::inference::model::Model;
#[cfg(feature = "metal")]
use crate::inference::gpu_ops::{self, MetalPipeline as _};
#[cfg(feature = "metal")]
use crate::hal::metal::{MetalDevice, MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
use crate::hal::metal::shader::sources;

// ── Configuration ────────────────────────────────────────────────────────────

/// Sesame CSM configuration.
#[derive(Debug, Clone)]
pub struct SesameCsmConfig {
    /// Backbone hidden dimension.
    pub backbone_hidden: usize,
    /// Backbone transformer layers.
    pub backbone_layers: usize,
    /// Backbone attention heads.
    pub backbone_heads: usize,
    /// Decoder hidden dimension.
    pub decoder_hidden: usize,
    /// Decoder transformer layers.
    pub decoder_layers: usize,
    /// Number of Mimi codebooks.
    pub num_codebooks: usize,
    /// Codebook vocabulary size.
    pub codebook_size: usize,
    /// Mimi codec frame rate (12.5 Hz, rounded to 12 for integer math).
    pub frame_rate: usize,
    /// Audio sample rate (24kHz).
    pub sample_rate: usize,
    /// Text vocabulary size.
    pub text_vocab_size: usize,
    /// RoPE frequency base.
    pub rope_theta: f32,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Backbone intermediate (MLP) dimension.
    pub backbone_intermediate: usize,
    /// Decoder intermediate (MLP) dimension.
    pub decoder_intermediate: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of GQA key-value heads for backbone.
    pub backbone_kv_heads: usize,
    /// Number of GQA key-value heads for decoder.
    pub decoder_kv_heads: usize,
    /// Number of speaker tokens supported.
    pub max_speakers: usize,
}

impl Default for SesameCsmConfig {
    fn default() -> Self {
        Self {
            backbone_hidden: 2048,
            backbone_layers: 24,
            backbone_heads: 16,
            decoder_hidden: 1024,
            decoder_layers: 8,
            num_codebooks: 12,
            codebook_size: 2048,
            frame_rate: 12,
            sample_rate: 24000,
            text_vocab_size: 32000,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            backbone_intermediate: 5504,
            decoder_intermediate: 2816,
            max_seq_len: 4096,
            backbone_kv_heads: 16,
            decoder_kv_heads: 8,
            max_speakers: 4,
        }
    }
}

impl SesameCsmConfig {
    /// Parse from config.json.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();
        if let Some(v) = json.get("backbone_hidden_size").and_then(|v| v.as_u64()) { c.backbone_hidden = v as usize; }
        if let Some(v) = json.get("backbone_num_layers").and_then(|v| v.as_u64()) { c.backbone_layers = v as usize; }
        if let Some(v) = json.get("backbone_num_heads").and_then(|v| v.as_u64()) { c.backbone_heads = v as usize; }
        if let Some(v) = json.get("backbone_num_kv_heads").and_then(|v| v.as_u64()) { c.backbone_kv_heads = v as usize; }
        if let Some(v) = json.get("backbone_intermediate_size").and_then(|v| v.as_u64()) { c.backbone_intermediate = v as usize; }
        if let Some(v) = json.get("decoder_hidden_size").and_then(|v| v.as_u64()) { c.decoder_hidden = v as usize; }
        if let Some(v) = json.get("decoder_num_layers").and_then(|v| v.as_u64()) { c.decoder_layers = v as usize; }
        if let Some(v) = json.get("decoder_num_kv_heads").and_then(|v| v.as_u64()) { c.decoder_kv_heads = v as usize; }
        if let Some(v) = json.get("decoder_intermediate_size").and_then(|v| v.as_u64()) { c.decoder_intermediate = v as usize; }
        if let Some(v) = json.get("num_codebooks").and_then(|v| v.as_u64()) { c.num_codebooks = v as usize; }
        if let Some(v) = json.get("codebook_size").and_then(|v| v.as_u64()) { c.codebook_size = v as usize; }
        if let Some(v) = json.get("frame_rate").and_then(|v| v.as_u64()) { c.frame_rate = v as usize; }
        if let Some(v) = json.get("sample_rate").and_then(|v| v.as_u64()) { c.sample_rate = v as usize; }
        if let Some(v) = json.get("text_vocab_size").and_then(|v| v.as_u64()) { c.text_vocab_size = v as usize; }
        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) { c.text_vocab_size = v as usize; }
        if let Some(v) = json.get("rope_theta").and_then(|v| v.as_f64()) { c.rope_theta = v as f32; }
        if let Some(v) = json.get("rms_norm_eps").and_then(|v| v.as_f64()) { c.rms_norm_eps = v as f32; }
        if let Some(v) = json.get("max_position_embeddings").and_then(|v| v.as_u64()) { c.max_seq_len = v as usize; }
        if let Some(v) = json.get("max_speakers").and_then(|v| v.as_u64()) { c.max_speakers = v as usize; }
        Ok(c)
    }

    /// Backbone head dimension.
    pub fn backbone_head_dim(&self) -> usize {
        self.backbone_hidden / self.backbone_heads
    }

    /// Decoder head count (derived from hidden / head_dim matching backbone).
    pub fn decoder_heads(&self) -> usize {
        // Decoder uses same head_dim as backbone but fewer heads.
        let head_dim = self.backbone_head_dim();
        self.decoder_hidden / head_dim
    }

    /// Decoder head dimension (same as backbone for weight compatibility).
    pub fn decoder_head_dim(&self) -> usize {
        self.backbone_head_dim()
    }

    /// Total audio token vocab: codebook_size * num_codebooks.
    pub fn audio_vocab_size(&self) -> usize {
        self.codebook_size * self.num_codebooks
    }
}

// ── Compiled Kernels ─────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct SesameCsmKernels {
    common: gpu_ops::CommonKernels,
    silu: Arc<ComputePipeline>,
    rms_norm: Arc<ComputePipeline>,
    rope: Arc<ComputePipeline>,
    mul: Arc<ComputePipeline>,
    conv1d_transpose: Arc<ComputePipeline>,
    rope_batched: Arc<ComputePipeline>,
    kv_head_expand: Arc<ComputePipeline>,
}

// ── Context Segment ──────────────────────────────────────────────────────────

/// A segment in the conversation context.
///
/// Each segment is either text or a reference to audio tokens, annotated
/// with the speaker identity. The backbone processes interleaved segments.
#[derive(Debug, Clone)]
pub enum ContextSegment {
    /// Text segment with speaker ID.
    Text {
        /// The text content.
        text: String,
        /// Speaker index (0 = assistant, 1+ = user).
        speaker: usize,
    },
    /// Audio token segment with speaker ID.
    Audio {
        /// Mimi codec token IDs, shape [num_codebooks, num_frames].
        tokens: Vec<Vec<u32>>,
        /// Speaker index.
        speaker: usize,
    },
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Sesame CSM pipeline for conversational speech synthesis on Metal GPU.
///
/// Two-tower architecture:
/// 1. Backbone (Llama-based, ~1B): processes interleaved text + audio context
/// 2. Decoder head: generates Mimi codec tokens from backbone hidden states
/// 3. Mimi codec: converts codec tokens to 24kHz waveform
#[cfg(feature = "metal")]
pub struct SesameCsmPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: SesameCsmConfig,
    kernels: SesameCsmKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for SesameCsmPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl SesameCsmPipeline {
    /// Create a new Sesame CSM pipeline with compiled Metal kernels.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: SesameCsmConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        let kernels = SesameCsmKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            silu: compute.compile_pipeline("silu", sources::SILU, "silu_f16")?,
            rms_norm: compute.compile_pipeline("rms_norm", sources::RMS_NORM, "rms_norm_f16")?,
            rope: compute.compile_pipeline("rope", sources::ROPE, "rope_f16")?,
            mul: compute.compile_pipeline("mul", sources::ELEMENTWISE, "mul_f16")?,
            conv1d_transpose: compute.compile_pipeline("conv1d_transpose", sources::CONV1D, "conv1d_transpose_f16")?,
            rope_batched: compute.compile_pipeline("rope_batched", sources::PHASE27_OPS, "rope_batched_f16")?,
            kv_head_expand: compute.compile_pipeline("kv_head_expand", sources::PHASE27_OPS, "kv_head_expand_f16")?,
        };
        Ok(Self { model, compute, config, kernels })
    }

    /// Synthesize speech audio from text with conversation context.
    ///
    /// - `text`: the text to synthesize into speech
    /// - `context`: conversation history as `(utterance_text, is_user)` pairs.
    ///   Each pair represents a turn; `is_user=true` for user speech, `false` for assistant.
    ///
    /// Returns PCM audio samples at 24kHz.
    pub fn synthesize(&self, text: &str, context: &[(String, bool)]) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Build context sequence: interleave text/audio segments with speaker tokens
        let context_tokens = self.build_context_tokens(text, context)?;
        let ctx_len = context_tokens.len();
        debug!(ctx_len, "SesameCsm: built context token sequence");

        if ctx_len == 0 {
            return Err(Error::internal("Empty context sequence"));
        }

        // 2. Run backbone: context tokens -> hidden states [ctx_len, backbone_hidden]
        let backbone_hidden = self.backbone_forward(&context_tokens)?;
        debug!(
            shape = %format!("[{}, {}]", ctx_len, config.backbone_hidden),
            "SesameCsm: backbone forward done"
        );

        // 3. Project backbone hidden to decoder input dimension
        let decoder_input = self.project_backbone_to_decoder(&backbone_hidden, ctx_len)?;

        // 4. Run decoder: hidden states -> Mimi codec tokens [num_codebooks, num_frames]
        let target_frames = self.estimate_target_frames(text);
        let codec_tokens = self.decoder_generate(&decoder_input, ctx_len, target_frames)?;
        let num_frames = codec_tokens[0].len();
        debug!(
            num_frames,
            num_codebooks = codec_tokens.len(),
            "SesameCsm: decoder generated codec tokens"
        );

        // 5. Run Mimi decoder: codec tokens -> 24kHz audio
        let audio = self.mimi_decode(&codec_tokens, num_frames)?;
        debug!(
            samples = audio.len(),
            duration_s = format!("{:.2}", audio.len() as f32 / config.sample_rate as f32),
            "SesameCsm: synthesis complete"
        );

        Ok(audio)
    }

    // ── Context Building ─────────────────────────────────────────────────────

    /// Build the backbone input token sequence from text and conversation context.
    ///
    /// Format: [spk_0] text_tokens [spk_1] text_tokens ... [spk_0] current_text_tokens
    /// Speaker 0 = assistant, Speaker 1 = user.
    fn build_context_tokens(&self, text: &str, context: &[(String, bool)]) -> Result<Vec<u32>> {
        let config = &self.config;
        let mut tokens = Vec::with_capacity(config.max_seq_len);

        // Special token IDs (offset past text vocab for speaker markers)
        let spk_base = config.text_vocab_size as u32;
        // [spk_0] = spk_base + 0, [spk_1] = spk_base + 1, etc.

        // Add conversation history
        for (utterance, is_user) in context {
            let speaker_id = if *is_user { 1u32 } else { 0u32 };
            tokens.push(spk_base + speaker_id);

            // Tokenize utterance text (simple character-level for now)
            let text_toks = self.simple_tokenize(utterance);
            tokens.extend_from_slice(&text_toks);

            // Truncate if approaching max length (leave room for current text)
            if tokens.len() > config.max_seq_len - 256 {
                break;
            }
        }

        // Add current text to synthesize (as assistant = speaker 0)
        tokens.push(spk_base); // [spk_0]
        let current_toks = self.simple_tokenize(text);
        tokens.extend_from_slice(&current_toks);

        // Truncate to max sequence length
        tokens.truncate(config.max_seq_len);

        Ok(tokens)
    }

    /// Simple whitespace-aware tokenization fallback.
    ///
    /// In production, the model's SentencePiece/BPE tokenizer should be used.
    /// This provides basic token IDs within the text vocab range.
    fn simple_tokenize(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        for ch in text.chars() {
            // Map characters to token IDs (simple hash within vocab range)
            let id = (ch as u32) % (self.config.text_vocab_size as u32 - 1) + 1;
            tokens.push(id);
        }
        tokens
    }

    /// Estimate the number of target audio frames based on text length.
    ///
    /// Heuristic: ~5 characters per frame at 12.5 Hz (80ms per frame).
    fn estimate_target_frames(&self, text: &str) -> usize {
        let chars = text.chars().count();
        // ~5 chars per frame at natural speech rate, minimum 10 frames
        let frames = (chars as f32 / 5.0).ceil() as usize;
        frames.max(10).min(500) // clamp to reasonable range
    }

    // ── Backbone Transformer (Llama-based) ───────────────────────────────────

    /// Backbone forward pass: context tokens -> hidden states.
    ///
    /// Llama-style transformer with:
    /// - Token + audio codebook embeddings
    /// - RoPE positional encoding
    /// - RMSNorm pre-norm
    /// - SiLU-gated MLP (gate_proj, up_proj, down_proj)
    /// - GQA (grouped query attention)
    fn backbone_forward(&self, tokens: &[u32]) -> Result<Tensor> {
        let config = &self.config;
        let seq_len = tokens.len();
        let hidden = config.backbone_hidden;

        // 1. Token embedding lookup: [seq_len] -> [seq_len, hidden]
        let embed = self.backbone_embed(tokens)?;

        let mut x = embed;

        // 2. Transformer layers
        for layer in 0..config.backbone_layers {
            x = self.backbone_layer(layer, &x, seq_len, 0)?;
        }

        // 3. Final RMSNorm
        let cb = self.compute.new_command_buffer();
        let norm_w = gpu_ops::read_weight_f16(&self.model, &self.compute, "backbone.norm.weight")?;
        let normed = self.gpu_rms_norm(&cb, &x, &norm_w, seq_len, hidden)?;
        cb.commit();
        cb.wait_until_completed();

        Ok(normed)
    }

    /// Backbone token embedding: handles both text tokens and speaker markers.
    fn backbone_embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let config = &self.config;
        let hidden = config.backbone_hidden;
        let device_id = self.compute.device().info().id;
        let seq_len = tokens.len();

        // Read embedding tables
        let text_embed = gpu_ops::read_weight_vec_f32(&self.model, "backbone.embed_tokens.weight")?;
        let spk_embed_opt = gpu_ops::read_weight_vec_f32(&self.model, "backbone.speaker_embed.weight").ok();

        let mut data = vec![0.0f32; seq_len * hidden];
        let spk_base = config.text_vocab_size as u32;

        for (i, &tok) in tokens.iter().enumerate() {
            if tok >= spk_base && tok < spk_base + config.max_speakers as u32 {
                // Speaker token: use speaker embedding if available
                let spk_id = (tok - spk_base) as usize;
                if let Some(ref spk_embed) = spk_embed_opt {
                    let row_start = spk_id * hidden;
                    if row_start + hidden <= spk_embed.len() {
                        data[i * hidden..(i + 1) * hidden]
                            .copy_from_slice(&spk_embed[row_start..row_start + hidden]);
                    }
                }
            } else {
                // Text token: use text embedding
                let tid = tok as usize;
                let row_start = tid * hidden;
                if row_start + hidden <= text_embed.len() {
                    data[i * hidden..(i + 1) * hidden]
                        .copy_from_slice(&text_embed[row_start..row_start + hidden]);
                }
            }
        }

        let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, Shape::from([seq_len, hidden]), DType::F16, device_id)
    }

    /// Single backbone transformer layer (Llama-style).
    fn backbone_layer(
        &self,
        layer_idx: usize,
        input: &Tensor,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let hidden = config.backbone_hidden;
        let heads = config.backbone_heads;
        let kv_heads = config.backbone_kv_heads;
        let head_dim = config.backbone_head_dim();
        let intermediate = config.backbone_intermediate;
        let prefix = format!("backbone.layers.{}", layer_idx);

        // Pre-attention RMSNorm
        let cb = self.compute.new_command_buffer();
        let norm_w = gpu_ops::read_weight_f16(
            &self.model, &self.compute,
            &format!("{}.input_layernorm.weight", prefix),
        )?;
        let normed = self.gpu_rms_norm(&cb, input, &norm_w, seq_len, hidden)?;

        // Q, K, V projections
        let q = self.linear_bias(
            &cb, &self.model, &normed,
            &format!("{}.self_attn.q_proj.weight", prefix),
            &format!("{}.self_attn.q_proj.bias", prefix),
            seq_len, hidden, heads * head_dim,
        ).or_else(|_| {
            // No bias variant
            let w = gpu_ops::read_weight_f16(&self.model, &self.compute,
                &format!("{}.self_attn.q_proj.weight", prefix))?;
            let device = self.compute.device().raw();
            let out_buf = device.new_buffer(
                (seq_len * heads * head_dim * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let tile: usize = 16;
            let n = heads * head_dim;
            self.compute.dispatch(
                &cb, &self.kernels.common.linear,
                ((n + tile - 1) / tile, (seq_len + tile - 1) / tile, 1),
                (tile, tile, 1),
                |enc| {
                    gpu_ops::set_tensor_buffer(enc, 0, &normed);
                    gpu_ops::set_tensor_buffer(enc, 1, &w);
                    gpu_ops::set_tensor_buffer(enc, 2, &w); // dummy bias
                    enc.set_buffer(3, Some(&out_buf), 0);
                    let vals: [u32; 4] = [seq_len as u32, n as u32, hidden as u32, 0];
                    for (j, v) in vals.iter().enumerate() {
                        enc.set_bytes((4 + j) as u64, 4, v as *const u32 as *const _);
                    }
                },
            );
            Ok(Tensor::from_metal_buffer(
                out_buf, Shape::from([seq_len, n]), DType::F16,
                self.compute.device().info().id,
            ))
        })?;

        let k = self.gpu_linear_no_bias(&cb, &normed, &format!("{}.self_attn.k_proj.weight", prefix),
            seq_len, hidden, kv_heads * head_dim)?;
        let v = self.gpu_linear_no_bias(&cb, &normed, &format!("{}.self_attn.v_proj.weight", prefix),
            seq_len, hidden, kv_heads * head_dim)?;
        cb.commit();
        cb.wait_until_completed();

        // Reshape for attention: Q [seq, heads, head_dim], K/V [seq, kv_heads, head_dim]
        let q_shd = q.reshape([seq_len, heads, head_dim])?;
        let k_shd = k.reshape([seq_len, kv_heads, head_dim])?;
        let v_shd = v.reshape([seq_len, kv_heads, head_dim])?;

        // Apply RoPE to Q and K on GPU (in-place, single dispatch each)
        let cb_rope = self.compute.new_command_buffer();
        self.apply_rope_gpu(&cb_rope, &q_shd, seq_len, heads, head_dim, start_pos);
        self.apply_rope_gpu(&cb_rope, &k_shd, seq_len, kv_heads, head_dim, start_pos);

        // Expand KV heads for GQA on GPU
        let k_expanded = gpu_ops::kv_head_expand_on(
            &self.compute, &self.kernels.kv_head_expand, &cb_rope,
            &k_shd, seq_len, kv_heads, heads, head_dim,
        );
        let v_expanded = gpu_ops::kv_head_expand_on(
            &self.compute, &self.kernels.kv_head_expand, &cb_rope,
            &v_shd, seq_len, kv_heads, heads, head_dim,
        );
        cb_rope.commit();
        cb_rope.wait_until_completed();

        // Batched attention via GPU
        let cb = self.compute.new_command_buffer();
        let attn_out = self.batched_attention(
            &cb, &q_shd, &k_expanded, &v_expanded,
            seq_len, seq_len, heads, head_dim,
            1.0 / (head_dim as f32).sqrt(),
        )?;
        cb.commit();
        cb.wait_until_completed();

        let attn_flat = attn_out.reshape([seq_len, heads * head_dim])?;

        // Output projection
        let cb = self.compute.new_command_buffer();
        let o_proj = self.gpu_linear_no_bias(&cb, &attn_flat,
            &format!("{}.self_attn.o_proj.weight", prefix),
            seq_len, heads * head_dim, hidden)?;
        let residual1 = self.add(&cb, input, &o_proj);
        cb.commit();
        cb.wait_until_completed();

        // Post-attention RMSNorm + SiLU-gated MLP
        let cb = self.compute.new_command_buffer();
        let post_norm_w = gpu_ops::read_weight_f16(
            &self.model, &self.compute,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let normed2 = self.gpu_rms_norm(&cb, &residual1, &post_norm_w, seq_len, hidden)?;

        // SiLU-gated MLP: gate = silu(x @ gate_proj) * (x @ up_proj); out = gate @ down_proj
        let gate = self.gpu_linear_no_bias(&cb, &normed2,
            &format!("{}.mlp.gate_proj.weight", prefix),
            seq_len, hidden, intermediate)?;
        let up = self.gpu_linear_no_bias(&cb, &normed2,
            &format!("{}.mlp.up_proj.weight", prefix),
            seq_len, hidden, intermediate)?;

        let gate_silu = self.gpu_silu(&cb, &gate)?;
        let gate_up = self.gpu_mul(&cb, &gate_silu, &up)?;

        let down = self.gpu_linear_no_bias(&cb, &gate_up,
            &format!("{}.mlp.down_proj.weight", prefix),
            seq_len, intermediate, hidden)?;

        let output = self.add(&cb, &residual1, &down);
        cb.commit();
        cb.wait_until_completed();

        Ok(output)
    }

    // ── Backbone-to-Decoder Projection ───────────────────────────────────────

    /// Project backbone hidden states to decoder input dimension.
    ///
    /// backbone_hidden (2048) -> decoder_hidden (1024) via linear projection.
    fn project_backbone_to_decoder(&self, backbone_out: &Tensor, seq_len: usize) -> Result<Tensor> {
        let config = &self.config;
        let cb = self.compute.new_command_buffer();
        let projected = self.gpu_linear_no_bias(
            &cb, backbone_out,
            "decoder.input_proj.weight",
            seq_len, config.backbone_hidden, config.decoder_hidden,
        ).or_else(|_| {
            // Fallback: truncate dimensions if no projection weight exists
            let data: Vec<half::f16> = backbone_out.to_vec()?;
            let device_id = self.compute.device().info().id;
            let mut proj_data = vec![half::f16::ZERO; seq_len * config.decoder_hidden];
            for s in 0..seq_len {
                for d in 0..config.decoder_hidden.min(config.backbone_hidden) {
                    proj_data[s * config.decoder_hidden + d] = data[s * config.backbone_hidden + d];
                }
            }
            Tensor::from_slice(
                &proj_data,
                Shape::from([seq_len, config.decoder_hidden]),
                DType::F16,
                device_id,
            )
        })?;
        cb.commit();
        cb.wait_until_completed();
        Ok(projected)
    }

    // ── Decoder Head ─────────────────────────────────────────────────────────

    /// Decoder autoregressive generation: backbone hidden -> codec tokens.
    ///
    /// The decoder generates Mimi codec tokens one frame at a time, predicting
    /// all codebooks for each frame. Uses the backbone's last hidden state
    /// as the initial conditioning signal.
    fn decoder_generate(
        &self,
        decoder_input: &Tensor,
        ctx_len: usize,
        target_frames: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let config = &self.config;
        let decoder_hidden = config.decoder_hidden;
        let num_codebooks = config.num_codebooks;
        let codebook_size = config.codebook_size;
        let device_id = self.compute.device().info().id;

        // Use last hidden state from backbone as initial decoder input
        let last_hidden_data: Vec<half::f16> = decoder_input.to_vec()?;
        let last_idx = (ctx_len - 1) * decoder_hidden;
        let init_hidden = &last_hidden_data[last_idx..last_idx + decoder_hidden];

        // Initialize codec token storage [num_codebooks][num_frames]
        let mut codec_tokens: Vec<Vec<u32>> = vec![Vec::with_capacity(target_frames); num_codebooks];

        // Autoregressive generation: one frame at a time
        let mut prev_hidden = Tensor::from_slice(
            init_hidden,
            Shape::from([1, decoder_hidden]),
            DType::F16,
            device_id,
        )?;

        for frame in 0..target_frames {
            // Run decoder transformer layers on current hidden state
            let mut h = prev_hidden.clone();
            for layer in 0..config.decoder_layers {
                h = self.decoder_layer(layer, &h, 1, frame)?;
            }

            // Final decoder norm
            let h_normed = if self.model.read().get_weight("decoder.norm.weight").is_some() {
                let cb = self.compute.new_command_buffer();
                let norm_w = gpu_ops::read_weight_f16(&self.model, &self.compute, "decoder.norm.weight")?;
                let result = self.gpu_rms_norm(&cb, &h, &norm_w, 1, decoder_hidden)?;
                cb.commit();
                cb.wait_until_completed();
                result
            } else {
                h.clone()
            };

            // Predict codebook tokens: one GPU linear head per codebook
            for cb_idx in 0..num_codebooks {
                let token_id = self.predict_codebook_token(
                    &h_normed, decoder_hidden, cb_idx, codebook_size,
                )?;
                codec_tokens[cb_idx].push(token_id);
            }

            // Build next decoder input from predicted tokens
            prev_hidden = self.embed_codec_tokens(
                &codec_tokens, frame, decoder_hidden,
            )?;

            if frame % 50 == 0 || frame == target_frames - 1 {
                debug!(frame, target_frames, "SesameCsm: decoder frame");
            }
        }

        Ok(codec_tokens)
    }

    /// Single decoder transformer layer (same Llama structure, smaller).
    fn decoder_layer(
        &self,
        layer_idx: usize,
        input: &Tensor,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let hidden = config.decoder_hidden;
        let heads = config.decoder_heads();
        let kv_heads = config.decoder_kv_heads;
        let head_dim = config.decoder_head_dim();
        let intermediate = config.decoder_intermediate;
        let prefix = format!("decoder.layers.{}", layer_idx);

        // Pre-attention RMSNorm
        let cb = self.compute.new_command_buffer();
        let norm_w = gpu_ops::read_weight_f16(
            &self.model, &self.compute,
            &format!("{}.input_layernorm.weight", prefix),
        )?;
        let normed = self.gpu_rms_norm(&cb, input, &norm_w, seq_len, hidden)?;

        // Q, K, V
        let q = self.gpu_linear_no_bias(&cb, &normed,
            &format!("{}.self_attn.q_proj.weight", prefix),
            seq_len, hidden, heads * head_dim)?;
        let k = self.gpu_linear_no_bias(&cb, &normed,
            &format!("{}.self_attn.k_proj.weight", prefix),
            seq_len, hidden, kv_heads * head_dim)?;
        let v = self.gpu_linear_no_bias(&cb, &normed,
            &format!("{}.self_attn.v_proj.weight", prefix),
            seq_len, hidden, kv_heads * head_dim)?;
        cb.commit();
        cb.wait_until_completed();

        // Reshape + RoPE
        let q_shd = q.reshape([seq_len, heads, head_dim])?;
        let k_shd = k.reshape([seq_len, kv_heads, head_dim])?;
        let v_shd = v.reshape([seq_len, kv_heads, head_dim])?;

        // RoPE + GQA expansion on GPU
        let cb_rope = self.compute.new_command_buffer();
        self.apply_rope_gpu(&cb_rope, &q_shd, seq_len, heads, head_dim, start_pos);
        self.apply_rope_gpu(&cb_rope, &k_shd, seq_len, kv_heads, head_dim, start_pos);
        let k_expanded = gpu_ops::kv_head_expand_on(
            &self.compute, &self.kernels.kv_head_expand, &cb_rope,
            &k_shd, seq_len, kv_heads, heads, head_dim,
        );
        let v_expanded = gpu_ops::kv_head_expand_on(
            &self.compute, &self.kernels.kv_head_expand, &cb_rope,
            &v_shd, seq_len, kv_heads, heads, head_dim,
        );
        cb_rope.commit();
        cb_rope.wait_until_completed();

        // Attention
        let cb = self.compute.new_command_buffer();
        let attn_out = self.batched_attention(
            &cb, &q_shd, &k_expanded, &v_expanded,
            seq_len, seq_len, heads, head_dim,
            1.0 / (head_dim as f32).sqrt(),
        )?;
        cb.commit();
        cb.wait_until_completed();

        let attn_flat = attn_out.reshape([seq_len, heads * head_dim])?;

        // O projection + residual
        let cb = self.compute.new_command_buffer();
        let o_proj = self.gpu_linear_no_bias(&cb, &attn_flat,
            &format!("{}.self_attn.o_proj.weight", prefix),
            seq_len, heads * head_dim, hidden)?;
        let residual1 = self.add(&cb, input, &o_proj);
        cb.commit();
        cb.wait_until_completed();

        // Post-attention RMSNorm + SiLU-gated MLP
        let cb = self.compute.new_command_buffer();
        let post_norm_w = gpu_ops::read_weight_f16(
            &self.model, &self.compute,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let normed2 = self.gpu_rms_norm(&cb, &residual1, &post_norm_w, seq_len, hidden)?;

        let gate = self.gpu_linear_no_bias(&cb, &normed2,
            &format!("{}.mlp.gate_proj.weight", prefix),
            seq_len, hidden, intermediate)?;
        let up = self.gpu_linear_no_bias(&cb, &normed2,
            &format!("{}.mlp.up_proj.weight", prefix),
            seq_len, hidden, intermediate)?;

        let gate_silu = self.gpu_silu(&cb, &gate)?;
        let gate_up = self.gpu_mul(&cb, &gate_silu, &up)?;

        let down = self.gpu_linear_no_bias(&cb, &gate_up,
            &format!("{}.mlp.down_proj.weight", prefix),
            seq_len, intermediate, hidden)?;

        let output = self.add(&cb, &residual1, &down);
        cb.commit();
        cb.wait_until_completed();

        Ok(output)
    }

    /// Predict a single codebook token from decoder hidden state via GPU linear.
    fn predict_codebook_token(
        &self,
        h_tensor: &Tensor,
        hidden_dim: usize,
        codebook_idx: usize,
        codebook_size: usize,
    ) -> Result<u32> {
        let head_name = format!("decoder.codebook_heads.{}.weight", codebook_idx);

        let logits_data = if self.model.read().get_weight(&head_name).is_some() {
            let w = gpu_ops::read_weight_f16(&self.model, &self.compute, &head_name)?;
            let dummy_bias = Tensor::empty(Shape::from([codebook_size]), DType::F16, self.compute.device().info().id)?;
            let cb = self.compute.new_command_buffer();
            let logits = self.linear_tensors(
                cb.as_ref(), h_tensor, &w, &dummy_bias,
                1, hidden_dim, codebook_size,
            );
            cb.commit();
            cb.wait_until_completed();
            let data: Vec<half::f16> = logits.to_vec()?;
            data.iter().map(|v| v.to_f32()).collect::<Vec<f32>>()
        } else {
            vec![0.0f32; codebook_size]
        };

        let mut best_id = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits_data.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i as u32;
            }
        }

        Ok(best_id)
    }

    /// Embed codec tokens from frame `frame_idx` into decoder hidden dimension.
    ///
    /// Sums all codebook embeddings for the current frame to produce a single vector.
    fn embed_codec_tokens(
        &self,
        codec_tokens: &[Vec<u32>],
        frame_idx: usize,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        let config = &self.config;
        let device_id = self.compute.device().info().id;
        let mut combined = vec![0.0f32; hidden_dim];

        for cb_idx in 0..config.num_codebooks {
            if frame_idx < codec_tokens[cb_idx].len() {
                let token_id = codec_tokens[cb_idx][frame_idx] as usize;
                let embed_name = format!("decoder.codec_input_embed.{}.weight", cb_idx);
                if let Ok(embed) = gpu_ops::read_weight_vec_f32(&self.model, &embed_name) {
                    let row_start = token_id * hidden_dim;
                    if row_start + hidden_dim <= embed.len() {
                        for d in 0..hidden_dim {
                            combined[d] += embed[row_start + d];
                        }
                    }
                }
            }
        }

        let f16_data: Vec<half::f16> = combined.iter().map(|&v| half::f16::from_f32(v)).collect();
        Tensor::from_slice(&f16_data, Shape::from([1, hidden_dim]), DType::F16, device_id)
    }

    // ── Mimi Audio Codec Decoder ─────────────────────────────────────────────

    /// Decode Mimi codec tokens to 24kHz audio waveform.
    ///
    /// Mimi architecture:
    /// 1. Codebook dequantization: token IDs -> codebook embeddings [num_frames, codec_dim]
    /// 2. Sum across codebooks (residual VQ pattern)
    /// 3. ConvTranspose1d decoder stack: upsample from frame rate to sample rate
    /// 4. Output: raw PCM samples at 24kHz
    fn mimi_decode(&self, codec_tokens: &[Vec<u32>], num_frames: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let codec_dim = 256; // Mimi internal embedding dimension

        // 1. Dequantize: look up codebook embeddings and sum (residual VQ)
        let mut quantized = vec![0.0f32; num_frames * codec_dim];

        for cb_idx in 0..config.num_codebooks {
            let codebook_name = format!("codec.quantizer.codebooks.{}.weight", cb_idx);
            let codebook_data = gpu_ops::read_weight_vec_f32(&self.model, &codebook_name)
                .unwrap_or_else(|_| vec![0.0f32; config.codebook_size * codec_dim]);

            for frame in 0..num_frames {
                let token = codec_tokens[cb_idx].get(frame).copied().unwrap_or(0) as usize;
                let token = token.min(config.codebook_size - 1);
                let row_start = token * codec_dim;
                if row_start + codec_dim <= codebook_data.len() {
                    for d in 0..codec_dim {
                        quantized[frame * codec_dim + d] += codebook_data[row_start + d];
                    }
                }
            }
        }

        // 2. Decoder convolutional stack: upsample from 12.5 Hz to 24kHz
        //    Total upsample ratio: 24000 / 12.5 = 1920
        //    Typical: 4 stages with rates [8, 6, 5, 8] (product = 1920)
        let upsample_rates = [8, 6, 5, 8];
        let upsample_kernels = [16, 12, 10, 16];

        let mut current = quantized;
        let mut current_len = num_frames;
        let mut current_channels = codec_dim;

        for (stage, (&rate, &kernel_size)) in upsample_rates.iter()
            .zip(upsample_kernels.iter())
            .enumerate()
        {
            let stage_prefix = format!("codec.decoder.layers.{}", stage);
            let out_channels = if stage < upsample_rates.len() - 1 {
                current_channels / 2
            } else {
                1 // Final stage outputs mono audio
            };
            let out_len = current_len * rate;

            let has_weights = self.model.read().get_weight(
                &format!("{}.conv_transpose.weight", stage_prefix)
            ).is_some();

            let mut output = vec![0.0f32; out_channels * out_len];

            if has_weights {
                // Real ConvTranspose1d
                let w = gpu_ops::read_weight_vec_f32(
                    &self.model,
                    &format!("{}.conv_transpose.weight", stage_prefix),
                )?;
                let b = gpu_ops::read_weight_vec_f32(
                    &self.model,
                    &format!("{}.conv_transpose.bias", stage_prefix),
                ).unwrap_or_else(|_| vec![0.0f32; out_channels]);
                let padding = (kernel_size - rate) / 2;

                for co in 0..out_channels {
                    for lo in 0..out_len {
                        let mut sum = b[co.min(b.len() - 1)];
                        for ci in 0..current_channels {
                            for k in 0..kernel_size {
                                let l_check = lo as isize + padding as isize - k as isize;
                                if l_check >= 0 && l_check % rate as isize == 0 {
                                    let li = l_check as usize / rate;
                                    if li < current_len {
                                        sum += current[ci * current_len + li]
                                             * w[(ci * out_channels + co) * kernel_size + k];
                                    }
                                }
                            }
                        }
                        output[co * out_len + lo] = sum;
                    }
                }
            } else {
                // Fallback: nearest-neighbor upsample + channel projection
                for co in 0..out_channels {
                    for lo in 0..out_len {
                        let li = (lo * current_len / out_len).min(current_len - 1);
                        let ci = co.min(current_channels - 1);
                        output[co * out_len + lo] = current[ci * current_len + li];
                    }
                }
            }

            // Post-upsample activation (ELU for Mimi)
            for v in output.iter_mut() {
                if *v < 0.0 {
                    *v = (*v).exp() - 1.0; // ELU(x) = exp(x) - 1 for x < 0
                }
            }

            // Transpose current from [channels, length] to match next stage
            current = output;
            current_len = out_len;
            current_channels = out_channels;

            debug!(stage, out_channels, out_len, "SesameCsm: Mimi decoder stage");
        }

        // 3. If multi-channel output remains, average to mono
        let audio = if current_channels > 1 {
            let mut mono = vec![0.0f32; current_len];
            for l in 0..current_len {
                let mut sum = 0.0f32;
                for c in 0..current_channels {
                    sum += current[c * current_len + l];
                }
                mono[l] = sum / current_channels as f32;
            }
            mono
        } else {
            current[..current_len].to_vec()
        };

        // 4. Normalize output
        let max_abs = audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mut normalized = audio;
        if max_abs > 1e-6 {
            let scale = 0.95 / max_abs;
            for v in normalized.iter_mut() {
                *v *= scale;
            }
        }

        Ok(normalized)
    }

    // ── GPU Helper Methods ───────────────────────────────────────────────────

    /// GPU linear without bias: Y = X @ W^T.
    fn gpu_linear_no_bias(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        weight_name: &str,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor> {
        let w = gpu_ops::read_weight_f16(&self.model, &self.compute, weight_name)?;
        let device = self.compute.device().raw();
        let out_buf = device.new_buffer(
            (m * n * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let tile: usize = 16;
        self.compute.dispatch(
            cb,
            &self.kernels.common.linear,
            ((n + tile - 1) / tile, (m + tile - 1) / tile, 1),
            (tile, tile, 1),
            |enc| {
                gpu_ops::set_tensor_buffer(enc, 0, input);
                gpu_ops::set_tensor_buffer(enc, 1, &w);
                gpu_ops::set_tensor_buffer(enc, 2, &w); // dummy bias (has_bias = 0)
                enc.set_buffer(3, Some(&out_buf), 0);
                let vals: [u32; 4] = [m as u32, n as u32, k as u32, 0]; // has_bias = 0
                for (i, v) in vals.iter().enumerate() {
                    enc.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Ok(Tensor::from_metal_buffer(
            out_buf,
            Shape::from([m, n]),
            DType::F16,
            self.compute.device().info().id,
        ))
    }

    /// GPU RMSNorm: output[i] = (x[i] / rms) * weight[i].
    ///
    /// Kernel signature: input(0), weight(1), output(2), N(3), D(4), eps(5).
    /// Dispatches one thread per row (seq_len threads).
    fn gpu_rms_norm(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        weight: &Tensor,
        seq_len: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let device_id = self.compute.device().info().id;
        let output = Tensor::empty(Shape::from([seq_len, hidden]), DType::F16, device_id)?;
        let eps = self.config.rms_norm_eps;
        let n_u32 = seq_len as u32;
        let d_u32 = hidden as u32;
        self.compute.dispatch_1d(cb, &self.kernels.rms_norm, seq_len, |enc| {
            gpu_ops::set_tensor_buffer(enc, 0, input);
            gpu_ops::set_tensor_buffer(enc, 1, weight);
            gpu_ops::set_tensor_buffer(enc, 2, &output);
            enc.set_bytes(3, 4, &n_u32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &d_u32 as *const u32 as *const _);
            enc.set_bytes(5, 4, &eps as *const f32 as *const _);
        });
        Ok(output)
    }

    /// GPU SiLU activation: x * sigmoid(x).
    fn gpu_silu(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
    ) -> Result<Tensor> {
        Ok(gpu_ops::activation_on(&self.compute, &self.kernels.silu, cb, input))
    }

    /// GPU element-wise multiply.
    fn gpu_mul(
        &self,
        cb: &metal::CommandBufferRef,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor> {
        Ok(gpu_ops::elementwise_binary_on(&self.compute, &self.kernels.mul, cb, a, b))
    }

    /// Apply RoPE positional encoding on GPU via batched kernel.
    fn apply_rope_gpu(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) {
        gpu_ops::rope_batched_on(
            &self.compute, &self.kernels.rope_batched,
            cb, input,
            self.config.rope_theta,
            seq_len, num_heads, head_dim, start_pos,
        );
    }

    /// Expand KV heads for GQA: repeat each KV head to match query head count.
    ///
    /// Input: [seq_len, kv_heads, head_dim]
    /// Output: [seq_len, num_heads, head_dim]
    #[allow(dead_code)]
    fn expand_kv_heads(
        &self,
        kv: &Tensor,
        seq_len: usize,
        kv_heads: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        if kv_heads == num_heads {
            return Ok(kv.clone());
        }

        let device_id = self.compute.device().info().id;
        let data: Vec<half::f16> = kv.to_vec()?;
        let repeat = num_heads / kv_heads;
        let mut expanded = vec![half::f16::ZERO; seq_len * num_heads * head_dim];

        for s in 0..seq_len {
            for kv_h in 0..kv_heads {
                let src_offset = s * kv_heads * head_dim + kv_h * head_dim;
                for r in 0..repeat {
                    let dst_h = kv_h * repeat + r;
                    let dst_offset = s * num_heads * head_dim + dst_h * head_dim;
                    expanded[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&data[src_offset..src_offset + head_dim]);
                }
            }
        }

        Tensor::from_slice(
            &expanded,
            Shape::from([seq_len, num_heads, head_dim]),
            DType::F16,
            device_id,
        )
    }

    /// Read a weight as f32 Vec.
    #[allow(dead_code)]
    fn weight_vec_f32(&self, name: &str) -> Result<Vec<f32>> {
        gpu_ops::read_weight_vec_f32(&self.model, name)
    }

    /// Check if a weight exists in the model.
    #[allow(dead_code)]
    fn has_weight(&self, name: &str) -> bool {
        self.model.read().get_weight(name).is_some()
    }
}
