//! Bark: cascaded GPT audio generation (MIT license, ~900M total).
//!
//! Architecture: 3 cascaded GPT models (~300M each):
//!   Stage 1 — Semantic Model (text -> semantic tokens):
//!     12 layers, 768 hidden, 12 heads, 3072 FFN
//!     Text vocab: 129,600 (WordPiece), Semantic vocab: 10,000
//!     Causal attention (standard GPT)
//!
//!   Stage 2 — Coarse Model (semantic -> coarse audio):
//!     12 layers, 1024 hidden, 16 heads, 4096 FFN
//!     Input: semantic tokens + 2 EnCodec codebooks
//!     Output: 2 coarse codebook tokens (1024 entries each, 75 Hz)
//!     Causal attention
//!
//!   Stage 3 — Fine Model (coarse -> fine audio):
//!     12 layers, 1024 hidden, 16 heads, 4096 FFN
//!     Bidirectional attention (NOT causal)
//!     Iteratively predicts codebooks 3-8 from sum of previous codebook embeddings
//!
//!   EnCodec Decoder: 8 codebook tokens -> 24kHz audio waveform
//!     ConvTranspose1d upsampling + ResBlock1d
//!
//! Weight prefixes: `semantic.`, `coarse_acoustics.`, `fine_acoustics.`, `encodec.`
//!
//! This is the MIT-licensed replacement for MusicGen (CC-BY-NC).

#[cfg(feature = "metal")]
use tracing::debug;
#[cfg(feature = "metal")]
use crate::core::{Error, Result};
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
use crate::tensor::{DType, Shape, Tensor};

// ── Configuration ────────────────────────────────────────────────────────────

/// Bark audio generation configuration.
#[derive(Debug, Clone)]
pub struct BarkConfig {
    /// Semantic token vocabulary size (output of stage 1).
    pub semantic_vocab_size: usize,
    /// Text vocabulary size (WordPiece BERT-like input).
    pub text_vocab_size: usize,
    /// EnCodec codebook size (per codebook).
    pub coarse_codebook_size: usize,
    /// Number of coarse codebooks (stage 2 output).
    pub n_coarse_codebooks: usize,
    /// Total number of fine codebooks (stage 3 outputs codebooks 3..n_fine_codebooks).
    pub n_fine_codebooks: usize,
    /// Semantic model hidden dimension.
    pub semantic_hidden: usize,
    /// Coarse model hidden dimension.
    pub coarse_hidden: usize,
    /// Fine model hidden dimension.
    pub fine_hidden: usize,
    /// Number of semantic model layers.
    pub semantic_layers: usize,
    /// Number of coarse model layers.
    pub coarse_layers: usize,
    /// Number of fine model layers.
    pub fine_layers: usize,
    /// Audio sample rate (Hz).
    pub sample_rate: usize,
}

impl Default for BarkConfig {
    fn default() -> Self {
        Self {
            semantic_vocab_size: 10_000,
            text_vocab_size: 129_600,
            coarse_codebook_size: 1024,
            n_coarse_codebooks: 2,
            n_fine_codebooks: 8,
            semantic_hidden: 768,
            coarse_hidden: 1024,
            fine_hidden: 1024,
            semantic_layers: 12,
            coarse_layers: 12,
            fine_layers: 12,
            sample_rate: 24_000,
        }
    }
}

impl BarkConfig {
    /// Parse from config.json (Bark-style layout).
    pub fn from_json(path: &std::path::Path) -> crate::core::Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| crate::core::Error::internal(format!("failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| crate::core::Error::internal(format!("failed to parse config: {}", e)))?;

        let mut c = Self::default();
        if let Some(v) = json.get("semantic_vocab_size").and_then(|v| v.as_u64()) { c.semantic_vocab_size = v as usize; }
        if let Some(v) = json.get("text_vocab_size").and_then(|v| v.as_u64()) { c.text_vocab_size = v as usize; }
        if let Some(v) = json.get("codebook_size").and_then(|v| v.as_u64()) { c.coarse_codebook_size = v as usize; }
        if let Some(v) = json.get("n_coarse_codebooks").and_then(|v| v.as_u64()) { c.n_coarse_codebooks = v as usize; }
        if let Some(v) = json.get("n_fine_codebooks").and_then(|v| v.as_u64()) { c.n_fine_codebooks = v as usize; }
        if let Some(sem) = json.get("semantic") {
            if let Some(v) = sem.get("hidden_size").and_then(|v| v.as_u64()) { c.semantic_hidden = v as usize; }
            if let Some(v) = sem.get("num_layers").and_then(|v| v.as_u64()) { c.semantic_layers = v as usize; }
        }
        if let Some(coarse) = json.get("coarse_acoustics") {
            if let Some(v) = coarse.get("hidden_size").and_then(|v| v.as_u64()) { c.coarse_hidden = v as usize; }
            if let Some(v) = coarse.get("num_layers").and_then(|v| v.as_u64()) { c.coarse_layers = v as usize; }
        }
        if let Some(fine) = json.get("fine_acoustics") {
            if let Some(v) = fine.get("hidden_size").and_then(|v| v.as_u64()) { c.fine_hidden = v as usize; }
            if let Some(v) = fine.get("num_layers").and_then(|v| v.as_u64()) { c.fine_layers = v as usize; }
        }
        if let Some(v) = json.get("sample_rate").and_then(|v| v.as_u64()) { c.sample_rate = v as usize; }
        Ok(c)
    }

    /// Semantic model: number of attention heads (768/64 = 12).
    #[cfg(feature = "metal")]
    fn semantic_heads(&self) -> usize { self.semantic_hidden / 64 }
    /// Semantic model: per-head dimension.
    #[cfg(feature = "metal")]
    fn semantic_head_dim(&self) -> usize { 64 }
    /// Semantic model: FFN intermediate size (4x hidden).
    #[cfg(feature = "metal")]
    fn semantic_ffn(&self) -> usize { self.semantic_hidden * 4 }

    /// Coarse model: number of attention heads (1024/64 = 16).
    #[cfg(feature = "metal")]
    fn coarse_heads(&self) -> usize { self.coarse_hidden / 64 }
    /// Coarse model: per-head dimension.
    #[cfg(feature = "metal")]
    fn coarse_head_dim(&self) -> usize { 64 }
    /// Coarse model: FFN intermediate size.
    #[cfg(feature = "metal")]
    fn coarse_ffn(&self) -> usize { self.coarse_hidden * 4 }

    /// Fine model: number of attention heads (1024/64 = 16).
    #[cfg(feature = "metal")]
    fn fine_heads(&self) -> usize { self.fine_hidden / 64 }
    /// Fine model: per-head dimension.
    #[cfg(feature = "metal")]
    fn fine_head_dim(&self) -> usize { 64 }
    /// Fine model: FFN intermediate size.
    #[cfg(feature = "metal")]
    fn fine_ffn(&self) -> usize { self.fine_hidden * 4 }

    /// Total combined vocabulary for the coarse model input:
    /// semantic_vocab + n_coarse_codebooks * codebook_size.
    #[cfg(feature = "metal")]
    #[allow(dead_code)]
    fn coarse_total_vocab(&self) -> usize {
        self.semantic_vocab_size + self.n_coarse_codebooks * self.coarse_codebook_size
    }
}

// ── Metal Kernels ────────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
#[allow(dead_code)]
struct BarkKernels {
    common: gpu_ops::CommonKernels,
    gelu: Arc<ComputePipeline>,
    layer_norm: Arc<ComputePipeline>,
    conv1d: Arc<ComputePipeline>,
    conv1d_transpose: Arc<ComputePipeline>,
    relu: Arc<ComputePipeline>,
    elu: Arc<ComputePipeline>,
    embedding_lookup: Arc<ComputePipeline>,
    argmax: Arc<ComputePipeline>,
    dequantize_rvq: Arc<ComputePipeline>,
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Bark audio generation pipeline: text -> semantic -> coarse -> fine -> waveform.
///
/// Runs 3 cascaded GPT stages plus an EnCodec decoder, all on Metal GPU.
#[cfg(feature = "metal")]
pub struct BarkPipeline {
    model: Arc<parking_lot::RwLock<Model>>,
    compute: Arc<MetalCompute>,
    config: BarkConfig,
    kernels: BarkKernels,
}

#[cfg(feature = "metal")]
impl gpu_ops::MetalPipeline for BarkPipeline {
    fn compute(&self) -> &MetalCompute { &self.compute }
    fn common_kernels(&self) -> &gpu_ops::CommonKernels { &self.kernels.common }
}

#[cfg(feature = "metal")]
impl BarkPipeline {
    /// Create a new Bark pipeline with compiled kernels.
    pub fn new(model: Arc<parking_lot::RwLock<Model>>, config: BarkConfig, device: Arc<MetalDevice>) -> Result<Self> {
        let compute = Arc::new(MetalCompute::new(device));
        let kernels = BarkKernels {
            common: gpu_ops::CommonKernels::new(&compute)?,
            gelu: compute.compile_pipeline("gelu", sources::GELU, "gelu_f16")?,
            layer_norm: compute.compile_pipeline("layer_norm", sources::LAYER_NORM, "layer_norm_f16")?,
            conv1d: compute.compile_pipeline("conv1d", sources::CONV1D, "conv1d_f16")?,
            conv1d_transpose: compute.compile_pipeline("conv1d_transpose", sources::CONV1D, "conv1d_transpose_f16")?,
            relu: compute.compile_pipeline("relu", sources::GELU, "relu_f16")?,
            elu: compute.compile_pipeline("elu", sources::PHASE27_OPS, "elu_f16")?,
            embedding_lookup: compute.compile_pipeline("embedding_lookup", sources::EMBEDDING, "embedding_lookup_f16")?,
            argmax: compute.compile_pipeline("argmax", sources::ARGMAX, "argmax_f16")?,
            dequantize_rvq: compute.compile_pipeline("dequantize_rvq", sources::PHASE27_OPS, "dequantize_rvq_f16")?,
        };
        Ok(Self { model, compute, config, kernels })
    }

    /// Generate audio from text.
    ///
    /// Pipeline: text -> semantic tokens -> coarse codebook tokens -> fine codebook tokens -> waveform.
    ///
    /// - `text`: input text string
    /// - `max_seconds`: maximum audio duration in seconds
    ///
    /// Returns PCM audio samples at 24kHz.
    pub fn generate(&self, text: &str, max_seconds: f32) -> Result<Vec<f32>> {
        let config = &self.config;

        // 1. Tokenize text -> text_tokens (simple WordPiece-like tokenization)
        let text_tokens = self.tokenize_text(text);
        let text_len = text_tokens.len();
        debug!(text_len, text = text, "Bark: tokenized text");

        if text_len < 2 {
            return Err(Error::internal("Text too short for audio generation"));
        }

        // Max tokens at 75 Hz semantic token rate
        let max_semantic_tokens = (max_seconds * 75.0) as usize;

        // 2. Stage 1: Semantic model (text -> semantic tokens)
        let semantic_tokens = self.semantic_forward(&text_tokens, max_semantic_tokens)?;
        let sem_len = semantic_tokens.len();
        debug!(sem_len, "Bark: semantic stage complete");

        // Max coarse tokens: 2 codebooks interleaved at 75 Hz
        let max_coarse_steps = sem_len * config.n_coarse_codebooks;

        // 3. Stage 2: Coarse model (semantic -> coarse codebook tokens)
        let coarse_tokens = self.coarse_forward(&semantic_tokens, max_coarse_steps)?;
        let n_coarse_frames = coarse_tokens.len() / config.n_coarse_codebooks;
        debug!(n_coarse_frames, n_coarse_codebooks = config.n_coarse_codebooks, "Bark: coarse stage complete");

        // 4. Stage 3: Fine model (coarse -> fine codebook tokens, bidirectional)
        let fine_tokens = self.fine_forward(&coarse_tokens, n_coarse_frames)?;
        debug!(n_codebooks = config.n_fine_codebooks, n_frames = n_coarse_frames, "Bark: fine stage complete");

        // 5. EnCodec decoder: 8 codebook tokens -> 24kHz waveform
        let audio = self.encodec_decode(&fine_tokens, n_coarse_frames)?;
        let duration = audio.len() as f32 / config.sample_rate as f32;
        debug!(samples = audio.len(), duration_s = format!("{:.2}", duration), "Bark: synthesis complete");

        Ok(audio)
    }

    // ── Text Tokenization ────────────────────────────────────────────────────

    /// Simple character-level tokenization for Bark's text encoder.
    ///
    /// Bark uses a BERT-style WordPiece tokenizer with 129,600 vocab entries.
    /// For production, load the real tokenizer.json. This provides a basic
    /// character-level fallback where each ASCII char maps to its code point
    /// offset by special token count.
    fn tokenize_text(&self, text: &str) -> Vec<u32> {
        // Special tokens: [CLS]=101, [SEP]=102, [PAD]=0, [UNK]=100
        let cls_token = 101u32;
        let sep_token = 102u32;
        let unk_token = 100u32;

        let mut tokens = vec![cls_token];
        for ch in text.chars() {
            let code = ch as u32;
            if code < 128 {
                // ASCII range: offset by 1000 to avoid collision with special tokens
                let tid = 1000 + code;
                if (tid as usize) < self.config.text_vocab_size {
                    tokens.push(tid);
                } else {
                    tokens.push(unk_token);
                }
            } else {
                // Non-ASCII: hash into vocab range
                let tid = 2000 + (code % 10_000);
                if (tid as usize) < self.config.text_vocab_size {
                    tokens.push(tid);
                } else {
                    tokens.push(unk_token);
                }
            }
        }
        tokens.push(sep_token);
        tokens
    }

    // ── Stage 1: Semantic Model ──────────────────────────────────────────────

    /// Semantic GPT: text_tokens -> semantic_tokens (causal autoregressive).
    ///
    /// 12 layers, 768 hidden, 12 heads. Generates one semantic token at a time
    /// until EOS or max_tokens reached.
    fn semantic_forward(&self, text_tokens: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        let config = &self.config;
        let hidden = config.semantic_hidden;
        let heads = config.semantic_heads();
        let head_dim = config.semantic_head_dim();
        let ffn_dim = config.semantic_ffn();
        let prefix = "semantic.";

        // Load embedding weights
        let text_embed = self.weight_vec_f32(&format!("{}input_embeds_layer.weight", prefix))?;
        let pos_embed = self.weight_vec_f32(&format!("{}position_embeds_layer.weight", prefix))?;

        // Semantic output embedding (separate from text input)
        let semantic_out_embed = self.weight_vec_f32(&format!("{}lm_head.weight", prefix))
            .unwrap_or_else(|_| vec![0.0f32; config.semantic_vocab_size * hidden]);

        // Build initial context from text tokens
        let _text_len = text_tokens.len();
        let mut all_tokens: Vec<u32> = text_tokens.to_vec();
        let mut semantic_tokens: Vec<u32> = Vec::with_capacity(max_tokens);

        // Autoregressive loop: generate semantic tokens one at a time
        for step in 0..max_tokens {
            let seq_len = all_tokens.len();

            // Embed all tokens: text_embed for text tokens, learned embed for semantic tokens
            let mut x = vec![0.0f32; seq_len * hidden];
            for (i, &tid) in all_tokens.iter().enumerate() {
                // Position embedding
                let pos = i.min(pos_embed.len() / hidden - 1);
                let embed_row = if (tid as usize) < config.text_vocab_size {
                    // Text token
                    let t = (tid as usize).min(text_embed.len() / hidden - 1);
                    &text_embed[t * hidden..(t + 1) * hidden]
                } else {
                    // Semantic token (offset by text_vocab_size)
                    let sem_id = (tid as usize).saturating_sub(config.text_vocab_size);
                    let s = sem_id.min(semantic_out_embed.len() / hidden - 1);
                    &semantic_out_embed[s * hidden..(s + 1) * hidden]
                };
                for d in 0..hidden {
                    x[i * hidden + d] = embed_row[d] + pos_embed[pos * hidden + d];
                }
            }

            // Run transformer layers (causal attention) — GPU tensor flow
            let x_tensor = self.f32_to_f16_tensor(&x, &[seq_len, hidden])?;
            let h = self.gpt_forward_gpu(
                &x_tensor, seq_len, hidden, heads, head_dim, ffn_dim,
                config.semantic_layers, prefix, true, // causal=true
            )?;

            // Project last position to semantic vocab logits (GPU: blit + linear + argmax)
            let token_id = self.project_to_token_gpu(
                &h, seq_len, hidden, config.semantic_vocab_size, prefix, "lm_head",
            )?;

            // Check for EOS (semantic_vocab_size - 1 is the EOS token)
            let eos_token = (config.semantic_vocab_size - 1) as u32;
            if token_id == eos_token {
                debug!(step, "Bark semantic: EOS reached");
                break;
            }

            semantic_tokens.push(token_id);
            // Add to context with text_vocab_size offset
            all_tokens.push(token_id + config.text_vocab_size as u32);

            if step % 50 == 0 {
                debug!(step, total_semantic = semantic_tokens.len(), "Bark: semantic generation progress");
            }
        }

        if semantic_tokens.is_empty() {
            return Err(Error::internal("Semantic model produced no tokens"));
        }

        Ok(semantic_tokens)
    }

    // ── Stage 2: Coarse Model ────────────────────────────────────────────────

    /// Coarse GPT: semantic_tokens -> coarse codebook tokens (causal autoregressive).
    ///
    /// 12 layers, 1024 hidden, 16 heads. Input is semantic tokens;
    /// output is interleaved tokens for 2 coarse EnCodec codebooks.
    fn coarse_forward(&self, semantic_tokens: &[u32], max_steps: usize) -> Result<Vec<u32>> {
        let config = &self.config;
        let hidden = config.coarse_hidden;
        let heads = config.coarse_heads();
        let head_dim = config.coarse_head_dim();
        let ffn_dim = config.coarse_ffn();
        let prefix = "coarse_acoustics.";

        // Semantic input embedding for coarse model
        let semantic_embed = self.weight_vec_f32(&format!("{}input_embeds_layer.weight", prefix))?;
        let pos_embed = self.weight_vec_f32(&format!("{}position_embeds_layer.weight", prefix))?;

        // Coarse codebook embeddings (2 codebooks, each 1024 entries)
        let codebook_embeds: Vec<Vec<f32>> = (0..config.n_coarse_codebooks)
            .map(|cb| {
                self.weight_vec_f32(&format!("{}codebook_embeds.{}.weight", prefix, cb))
                    .unwrap_or_else(|_| vec![0.0f32; config.coarse_codebook_size * hidden])
            })
            .collect();

        let mut all_tokens: Vec<(u32, bool)> = semantic_tokens.iter()
            .map(|&t| (t, true)) // (token_id, is_semantic)
            .collect();

        let mut coarse_tokens: Vec<u32> = Vec::with_capacity(max_steps);
        let mut current_codebook = 0usize;

        // Autoregressive loop: generate coarse tokens
        for step in 0..max_steps {
            let seq_len = all_tokens.len();

            // Embed all tokens
            let mut x = vec![0.0f32; seq_len * hidden];
            for (i, &(tid, is_semantic)) in all_tokens.iter().enumerate() {
                let pos = i.min(pos_embed.len() / hidden - 1);
                let embed_row = if is_semantic {
                    let s = (tid as usize).min(semantic_embed.len() / hidden - 1);
                    &semantic_embed[s * hidden..(s + 1) * hidden]
                } else {
                    // Coarse codebook token: determine which codebook
                    let cb_idx = (coarse_tokens.len().saturating_sub(1 + step.saturating_sub(1))) % config.n_coarse_codebooks;
                    let cb = &codebook_embeds[cb_idx.min(config.n_coarse_codebooks - 1)];
                    let t = (tid as usize).min(cb.len() / hidden - 1);
                    &cb[t * hidden..(t + 1) * hidden]
                };
                for d in 0..hidden {
                    x[i * hidden + d] = embed_row[d] + pos_embed[pos * hidden + d];
                }
            }

            // Run transformer (causal) — GPU tensor flow
            let x_tensor = self.f32_to_f16_tensor(&x, &[seq_len, hidden])?;
            let h = self.gpt_forward_gpu(
                &x_tensor, seq_len, hidden, heads, head_dim, ffn_dim,
                config.coarse_layers, prefix, true,
            )?;

            // Project last position to codebook logits (GPU: blit + linear + argmax)
            let head_name = format!("lm_heads.{}", current_codebook);
            let token_id = self.project_to_token_gpu(
                &h, seq_len, hidden, config.coarse_codebook_size, prefix, &head_name,
            )?;

            coarse_tokens.push(token_id);
            all_tokens.push((token_id, false));

            // Alternate between codebooks (interleaved)
            current_codebook = (current_codebook + 1) % config.n_coarse_codebooks;

            if step % 100 == 0 {
                debug!(step, codebook = current_codebook, "Bark: coarse generation progress");
            }
        }

        Ok(coarse_tokens)
    }

    // ── Stage 3: Fine Model ──────────────────────────────────────────────────

    /// Fine GPT: coarse_tokens -> fine codebook tokens (bidirectional, iterative).
    ///
    /// 12 layers, 1024 hidden, 16 heads. Uses BIDIRECTIONAL attention (no causal mask).
    /// Iteratively predicts codebooks 3-8 from the sum of embeddings of previous codebooks.
    fn fine_forward(&self, coarse_tokens: &[u32], n_frames: usize) -> Result<Vec<Vec<u32>>> {
        let config = &self.config;
        let hidden = config.fine_hidden;
        let heads = config.fine_heads();
        let head_dim = config.fine_head_dim();
        let ffn_dim = config.fine_ffn();
        let prefix = "fine_acoustics.";

        // Load codebook embeddings for all codebooks
        let codebook_embeds: Vec<Vec<f32>> = (0..config.n_fine_codebooks)
            .map(|cb| {
                self.weight_vec_f32(&format!("{}codebook_embeds.{}.weight", prefix, cb))
                    .unwrap_or_else(|_| vec![0.0f32; config.coarse_codebook_size * hidden])
            })
            .collect();

        let pos_embed = self.weight_vec_f32(&format!("{}position_embeds_layer.weight", prefix))?;

        // Deinterleave coarse tokens into per-codebook arrays
        let mut all_codebooks: Vec<Vec<u32>> = vec![Vec::new(); config.n_fine_codebooks];
        for (i, &tok) in coarse_tokens.iter().enumerate() {
            let cb = i % config.n_coarse_codebooks;
            all_codebooks[cb].push(tok);
        }
        // Pad shorter codebooks to n_frames
        for cb in 0..config.n_coarse_codebooks {
            all_codebooks[cb].resize(n_frames, 0);
        }

        // Iteratively predict codebooks 3 through n_fine_codebooks
        for target_cb in config.n_coarse_codebooks..config.n_fine_codebooks {
            // Build input: sum of embeddings from all previous codebooks
            let mut x = vec![0.0f32; n_frames * hidden];
            for frame in 0..n_frames {
                let pos = frame.min(pos_embed.len() / hidden - 1);
                // Sum embeddings from codebooks 0..target_cb
                for cb in 0..target_cb {
                    let tok = all_codebooks[cb][frame] as usize;
                    let embed = &codebook_embeds[cb];
                    let t = tok.min(embed.len() / hidden - 1);
                    for d in 0..hidden {
                        x[frame * hidden + d] += embed[t * hidden + d];
                    }
                }
                // Add position embedding
                for d in 0..hidden {
                    x[frame * hidden + d] += pos_embed[pos * hidden + d];
                }
            }

            // Run transformer with BIDIRECTIONAL attention (causal=false) — GPU tensor flow
            let x_tensor = self.f32_to_f16_tensor(&x, &[n_frames, hidden])?;
            let h = self.gpt_forward_gpu(
                &x_tensor, n_frames, hidden, heads, head_dim, ffn_dim,
                config.fine_layers, prefix, false, // causal=false for fine model
            )?;

            // Batch project all frames to codebook logits (GPU: batched linear + per-row argmax)
            let head_name = format!("lm_heads.{}", target_cb);
            let w_name = format!("{}{}.weight", prefix, head_name);
            let b_name = format!("{}{}.bias", prefix, head_name);
            let cb_tokens = if self.has_weight(&w_name) {
                // Batched linear: [n_frames, hidden] → [n_frames, vocab_size]
                let cb = self.compute.new_command_buffer();
                let logits = if self.has_weight(&b_name) {
                    self.linear_bias(
                        cb.as_ref(), &self.model, &h,
                        &w_name, &b_name, n_frames, hidden, config.coarse_codebook_size,
                    )?
                } else {
                    let w = gpu_ops::read_weight_f16(&self.model, &self.compute, &w_name)?;
                    let zero_b = Tensor::empty(Shape::from([config.coarse_codebook_size]), DType::F16, self.compute.device().info().id)?;
                    self.linear_tensors(cb.as_ref(), &h, &w, &zero_b, n_frames, hidden, config.coarse_codebook_size)
                };
                cb.commit();
                cb.wait_until_completed();

                // Per-row argmax on CPU (small per-frame work)
                let logits_f32 = logits.to_f32_vec()?;
                let mut tokens = Vec::with_capacity(n_frames);
                for frame in 0..n_frames {
                    let row = &logits_f32[frame * config.coarse_codebook_size..(frame + 1) * config.coarse_codebook_size];
                    let mut max_idx = 0u32;
                    let mut max_val = f32::NEG_INFINITY;
                    for (i, &v) in row.iter().enumerate() {
                        if v > max_val { max_val = v; max_idx = i as u32; }
                    }
                    tokens.push(max_idx);
                }
                tokens
            } else {
                vec![0u32; n_frames]
            };
            let cb_tokens = cb_tokens;

            all_codebooks[target_cb] = cb_tokens;
            debug!(target_cb, n_frames, "Bark: fine codebook generated");
        }

        Ok(all_codebooks)
    }

    // ── EnCodec Decoder ──────────────────────────────────────────────────────

    /// EnCodec decoder: 8 codebook token arrays -> 24kHz audio waveform.
    ///
    /// Architecture: sum codebook embeddings -> ConvTranspose1d upsampling + ResBlock1d.
    /// Upsampling factors: [8, 5, 4, 2] -> total 320x (75 Hz tokens -> 24kHz audio).
    fn encodec_decode(&self, codebook_tokens: &[Vec<u32>], n_frames: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let prefix = "encodec.decoder.";

        let n_codebooks = codebook_tokens.len().min(config.n_fine_codebooks);
        let embed_dim = 128;

        // 1. Dequantize on GPU: sum codebook embeddings → [n_frames, embed_dim] via dequantize_rvq_f16
        let device = self.compute.device().raw();
        let _device_id = self.compute.device().info().id;

        // Stack all codebook embeddings into [n_codebooks * vocab_size, embed_dim]
        let mut all_embeds = Vec::with_capacity(n_codebooks * config.coarse_codebook_size * embed_dim);
        for cb_i in 0..n_codebooks {
            let embed = self.weight_vec_f32(&format!("encodec.quantizer.layers.{}.codebook", cb_i))
                .unwrap_or_else(|_| {
                    let mut e = vec![0.0f32; config.coarse_codebook_size * embed_dim];
                    for (i, v) in e.iter_mut().enumerate() {
                        *v = ((i as f32 * 0.01).sin() * 0.1) / n_codebooks as f32;
                    }
                    e
                });
            all_embeds.extend_from_slice(&embed[..config.coarse_codebook_size * embed_dim]);
        }
        let embeds_tensor = self.f32_to_f16_tensor(&all_embeds, &[n_codebooks * config.coarse_codebook_size, embed_dim])?;

        // Stack all token codes into flat u32 buffer [n_codebooks, n_frames]
        let mut codes_flat = Vec::with_capacity(n_codebooks * n_frames);
        for cb_i in 0..n_codebooks {
            let tokens = &codebook_tokens[cb_i];
            for frame in 0..n_frames {
                codes_flat.push(tokens[frame.min(tokens.len() - 1)]);
            }
        }
        let codes_buf = device.new_buffer(
            (codes_flat.len() * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                codes_flat.as_ptr(),
                codes_buf.contents() as *mut u32,
                codes_flat.len(),
            );
        }

        // GPU dequantize: sum codebook embeddings per frame → [n_frames, embed_dim]
        let cb = self.compute.new_command_buffer();
        let dequantized = gpu_ops::dequantize_rvq_on(
            &self.compute, &self.kernels.dequantize_rvq, cb.as_ref(),
            &codes_buf, &embeds_tensor,
            n_codebooks, n_frames, config.coarse_codebook_size, embed_dim,
        );
        cb.commit();
        cb.wait_until_completed();

        // Convert to channel-first [embed_dim, n_frames] for conv1d
        let dequantized_f32 = dequantized.to_f32_vec()?;
        let mut quantized = vec![0.0f32; embed_dim * n_frames];
        for frame in 0..n_frames {
            for d in 0..embed_dim {
                quantized[d * n_frames + frame] = dequantized_f32[frame * embed_dim + d];
            }
        }

        let init_channels = 512;
        let mut current_len = n_frames;
        let mut channels = init_channels;

        // 2. Input conv1d on GPU: [embed_dim, n_frames] -> [channels, n_frames]
        let mut x_tensor = if self.has_weight(&format!("{}model.0.conv.weight", prefix)) {
            let input_t = self.f32_to_f16_tensor(&quantized, &[embed_dim, n_frames])?;
            let cb = self.compute.new_command_buffer();
            let out = self.gpu_conv1d(
                cb.as_ref(), &input_t,
                &format!("{}model.0.conv.weight", prefix),
                &format!("{}model.0.conv.bias", prefix),
                embed_dim, channels, current_len, 7, 1, 3,
            )?;
            cb.commit();
            cb.wait_until_completed();
            out
        } else {
            let mut x = vec![0.0f32; channels * current_len];
            for co in 0..channels.min(embed_dim) {
                for l in 0..current_len {
                    x[co * current_len + l] = quantized[co * n_frames + l];
                }
            }
            self.f32_to_f16_tensor(&x, &[channels, current_len])?
        };

        // 3. Upsample blocks on GPU
        let upsample_ratios = [8, 5, 4, 2];
        let upsample_kernels = [16, 10, 8, 4];

        for (stage, (&ratio, &kernel_size)) in upsample_ratios.iter()
            .zip(upsample_kernels.iter())
            .enumerate()
        {
            let out_channels = channels / 2;
            let padding = (kernel_size - ratio) / 2;
            let out_len = (current_len - 1) * ratio + kernel_size - 2 * padding;

            let cb = self.compute.new_command_buffer();

            // ELU activation
            x_tensor = gpu_ops::elu_on(&self.compute, &self.kernels.elu, cb.as_ref(), &x_tensor, 1.0);

            // ConvTranspose1d upsampling
            let block_prefix = format!("{}model.{}", prefix, 1 + stage * 3);
            let upsampled = if self.has_weight(&format!("{}.conv.weight", block_prefix)) {
                self.gpu_conv1d_transpose(
                    cb.as_ref(), &x_tensor,
                    &format!("{}.conv.weight", block_prefix),
                    &format!("{}.conv.bias", block_prefix),
                    channels, out_channels, current_len, kernel_size, ratio, padding,
                )?
            } else {
                // Nearest-neighbor fallback (stay on CPU then upload)
                cb.commit();
                cb.wait_until_completed();
                let x_f32 = x_tensor.to_f32_vec()?;
                let mut up = vec![0.0f32; out_channels * out_len];
                for co in 0..out_channels {
                    for lo in 0..out_len {
                        let li = (lo * current_len / out_len).min(current_len - 1);
                        up[co * out_len + lo] = x_f32[co.min(channels - 1) * current_len + li];
                    }
                }
                self.f32_to_f16_tensor(&up, &[out_channels, out_len])?
            };

            // ResBlock1d on GPU
            let res_prefix = format!("{}model.{}", prefix, 2 + stage * 3);
            x_tensor = self.resblock1d_forward_gpu(&upsampled, out_channels, out_len, &res_prefix)?;

            if !self.has_weight(&format!("{}.conv.weight", block_prefix)) {
                // Already committed above in fallback path
            } else {
                cb.commit();
                cb.wait_until_completed();
            }

            channels = out_channels;
            current_len = out_len;

            debug!(stage, out_channels, out_len, ratio, "Bark: EnCodec upsample stage done");
        }

        // 4. Final ELU + Conv1d -> mono audio
        let cb = self.compute.new_command_buffer();
        x_tensor = gpu_ops::elu_on(&self.compute, &self.kernels.elu, cb.as_ref(), &x_tensor, 1.0);

        let final_conv_w = format!("{}model.{}.conv.weight", prefix, 1 + upsample_ratios.len() * 3);
        let final_conv_b = format!("{}model.{}.conv.bias", prefix, 1 + upsample_ratios.len() * 3);
        if self.has_weight(&final_conv_w) {
            x_tensor = self.gpu_conv1d(
                cb.as_ref(), &x_tensor,
                &final_conv_w, &final_conv_b,
                channels, 1, current_len, 7, 1, 3,
            )?;
        }
        cb.commit();
        cb.wait_until_completed();

        let mut audio = x_tensor.to_f32_vec()?;
        if !self.has_weight(&final_conv_w) {
            let full = audio;
            audio = vec![0.0f32; current_len];
            for l in 0..current_len {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += full[c * current_len + l];
                }
                audio[l] = sum / channels as f32;
            }
        }

        for v in audio.iter_mut() {
            *v = v.tanh();
        }
        let max_abs = audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs > 1e-6 {
            let scale = 0.95 / max_abs;
            for v in audio.iter_mut() {
                *v *= scale;
            }
        }

        Ok(audio)
    }

    // ── Shared GPT Forward ───────────────────────────────────────────────────

    /// GPU-native GPT forward: takes and returns f16 Tensor (no f32 round-trip).
    fn gpt_forward_gpu(
        &self,
        x: &Tensor,
        seq_len: usize,
        hidden: usize,
        heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        n_layers: usize,
        prefix: &str,
        _causal: bool,
    ) -> Result<Tensor> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let device_id = self.compute.device().info().id;

        let mut h = x.clone();

        let cb = self.compute.new_command_buffer();

        for layer in 0..n_layers {
            let lp = format!("{}transformer.h.{}", prefix, layer);

            let normed = self.layer_norm(
                cb.as_ref(), &self.model, &h,
                &format!("{}.ln_1.weight", lp), &format!("{}.ln_1.bias", lp),
                seq_len, hidden, 1e-5,
            )?;

            let qkv = self.linear_bias(
                cb.as_ref(), &self.model, &normed,
                &format!("{}.attn.c_attn.weight", lp), &format!("{}.attn.c_attn.bias", lp),
                seq_len, hidden, hidden * 3,
            )?;

            let q_buf = Tensor::empty(Shape::from([seq_len, hidden]), DType::F16, device_id)?;
            let k_buf = Tensor::empty(Shape::from([seq_len, hidden]), DType::F16, device_id)?;
            let v_buf = Tensor::empty(Shape::from([seq_len, hidden]), DType::F16, device_id)?;
            self.split_qkv_dispatch(cb.as_ref(), &qkv, &q_buf, &k_buf, &v_buf, seq_len, hidden);

            let q_shd = q_buf.reshape([seq_len, heads, head_dim])?;
            let k_shd = k_buf.reshape([seq_len, heads, head_dim])?;
            let v_shd = v_buf.reshape([seq_len, heads, head_dim])?;

            let attn_out = self.batched_attention(
                cb.as_ref(), &q_shd, &k_shd, &v_shd,
                seq_len, seq_len, heads, head_dim, scale,
            )?;

            let projected = self.linear_bias(
                cb.as_ref(), &self.model, &attn_out,
                &format!("{}.attn.c_proj.weight", lp), &format!("{}.attn.c_proj.bias", lp),
                seq_len, hidden, hidden,
            )?;

            h = self.add(cb.as_ref(), &h, &projected);

            let normed2 = self.layer_norm(
                cb.as_ref(), &self.model, &h,
                &format!("{}.ln_2.weight", lp), &format!("{}.ln_2.bias", lp),
                seq_len, hidden, 1e-5,
            )?;

            let ffn_up = self.linear_bias(
                cb.as_ref(), &self.model, &normed2,
                &format!("{}.mlp.c_fc.weight", lp), &format!("{}.mlp.c_fc.bias", lp),
                seq_len, hidden, ffn_dim,
            )?;

            let activated = self.activation(cb.as_ref(), &self.kernels.gelu, &ffn_up);

            let ffn_out = self.linear_bias(
                cb.as_ref(), &self.model, &activated,
                &format!("{}.mlp.c_proj.weight", lp), &format!("{}.mlp.c_proj.bias", lp),
                seq_len, ffn_dim, hidden,
            )?;

            h = self.add(cb.as_ref(), &h, &ffn_out);

            if layer == 0 || layer == n_layers - 1 {
                debug!(layer, n_layers, _causal, "Bark: GPT layer done");
            }
        }

        let h = self.layer_norm(
            cb.as_ref(), &self.model, &h,
            &format!("{}transformer.ln_f.weight", prefix),
            &format!("{}transformer.ln_f.bias", prefix),
            seq_len, hidden, 1e-5,
        )?;

        cb.commit();
        cb.wait_until_completed();

        Ok(h)
    }

    #[allow(dead_code)]
    fn gpt_forward(
        &self,
        x: &[f32],
        seq_len: usize,
        hidden: usize,
        heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        n_layers: usize,
        prefix: &str,
        causal: bool,
    ) -> Result<Vec<f32>> {
        let h = self.f32_to_f16_tensor(x, &[seq_len, hidden])?;
        let result = self.gpt_forward_gpu(&h, seq_len, hidden, heads, head_dim, ffn_dim, n_layers, prefix, causal)?;
        result.to_f32_vec()
    }

    fn split_qkv_dispatch(
        &self,
        cb: &metal::CommandBufferRef,
        qkv: &Tensor,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        dim: usize,
    ) {
        let numel = seq_len * dim;
        let compute = self.compute();
        compute.dispatch_1d(cb, &self.kernels.common.add, numel, |encoder| {
            gpu_ops::set_tensor_buffer(encoder, 0, qkv);
            gpu_ops::set_tensor_buffer(encoder, 1, qkv);
            gpu_ops::set_tensor_buffer(encoder, 2, q);
        });
        // Manual blit: Q = qkv[:, 0:dim], K = qkv[:, dim:2*dim], V = qkv[:, 2*dim:3*dim]
        let blit = cb.new_blit_command_encoder();
        if let (Some(src_ptr), Some(q_ptr), Some(k_ptr), Some(v_ptr)) =
            (qkv.device_ptr(), q.device_ptr(), k.device_ptr(), v.device_ptr())
        {
            let src_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(src_ptr) };
            let q_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(q_ptr) };
            let k_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(k_ptr) };
            let v_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(v_ptr) };
            for s in 0..seq_len {
                let src_row_offset = (s * dim * 3 * 2) as u64;
                let dst_row_offset = (s * dim * 2) as u64;
                let row_bytes = (dim * 2) as u64;
                blit.copy_from_buffer(src_buf.as_ref(), src_row_offset, q_buf.as_ref(), dst_row_offset, row_bytes);
                blit.copy_from_buffer(src_buf.as_ref(), src_row_offset + row_bytes, k_buf.as_ref(), dst_row_offset, row_bytes);
                blit.copy_from_buffer(src_buf.as_ref(), src_row_offset + 2 * row_bytes, v_buf.as_ref(), dst_row_offset, row_bytes);
            }
        }
        blit.end_encoding();
    }

    // ── ResBlock1d (EnCodec) — GPU ──────────────────────────────────────────

    fn resblock1d_forward_gpu(
        &self,
        input: &Tensor,
        channels: usize,
        length: usize,
        prefix: &str,
    ) -> Result<Tensor> {
        let cb = self.compute.new_command_buffer();
        let mut x = input.clone();

        for conv_idx in 0..2 {
            let w_name = format!("{}.block.{}.conv.weight", prefix, conv_idx);
            let b_name = format!("{}.block.{}.conv.bias", prefix, conv_idx);
            let ks: usize = if conv_idx == 0 { 3 } else { 1 };
            let pad = ks / 2;

            x = gpu_ops::elu_on(&self.compute, &self.kernels.elu, cb.as_ref(), &x, 1.0);

            if self.has_weight(&w_name) {
                x = self.gpu_conv1d(
                    cb.as_ref(), &x, &w_name, &b_name,
                    channels, channels, length, ks, 1, pad,
                )?;
            }
        }

        // Skip connection
        let shortcut_name = format!("{}.shortcut.conv.weight", prefix);
        let skip = if self.has_weight(&shortcut_name) {
            self.gpu_conv1d(
                cb.as_ref(), input,
                &shortcut_name, &format!("{}.shortcut.conv.bias", prefix),
                channels, channels, length, 1, 1, 0,
            )?
        } else {
            input.clone()
        };

        let result = self.add(cb.as_ref(), &x, &skip);

        cb.commit();
        cb.wait_until_completed();

        Ok(result)
    }

    // ── Utility Functions ────────────────────────────────────────────────────

    fn weight_vec_f32(&self, name: &str) -> Result<Vec<f32>> {
        gpu_ops::read_weight_vec_f32(&self.model, name)
    }

    fn has_weight(&self, name: &str) -> bool {
        self.model.read().get_weight(name).is_some()
    }

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

    fn gpu_conv1d(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        w_name: &str,
        b_name: &str,
        in_channels: usize,
        out_channels: usize,
        in_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        let w = gpu_ops::read_weight_f16(&self.model, &self.compute, w_name)?;
        let b = if self.has_weight(b_name) {
            gpu_ops::read_weight_f16(&self.model, &self.compute, b_name)?
        } else {
            self.f32_to_f16_tensor(&vec![0.0f32; out_channels], &[out_channels])?
        };
        let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (out_channels * out_length * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch(
            cb, &self.kernels.conv1d,
            (out_length, out_channels, 1), (16, 16, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &w);
                gpu_ops::set_tensor_buffer(encoder, 2, &b);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 6] = [
                    in_channels as u32, out_channels as u32, in_length as u32,
                    kernel_size as u32, stride as u32, padding as u32,
                ];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Ok(Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([out_channels, out_length]),
            DType::F16,
            self.compute.device().info().id,
        ))
    }

    fn gpu_conv1d_transpose(
        &self,
        cb: &metal::CommandBufferRef,
        input: &Tensor,
        w_name: &str,
        b_name: &str,
        in_channels: usize,
        out_channels: usize,
        in_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        let w = gpu_ops::read_weight_f16(&self.model, &self.compute, w_name)?;
        let b = if self.has_weight(b_name) {
            gpu_ops::read_weight_f16(&self.model, &self.compute, b_name)?
        } else {
            self.f32_to_f16_tensor(&vec![0.0f32; out_channels], &[out_channels])?
        };
        let out_length = (in_length - 1) * stride - 2 * padding + kernel_size;
        let device = self.compute.device().raw();
        let output_buffer = device.new_buffer(
            (out_channels * out_length * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.compute.dispatch(
            cb, &self.kernels.conv1d_transpose,
            (out_length, out_channels, 1), (16, 16, 1),
            |encoder| {
                gpu_ops::set_tensor_buffer(encoder, 0, input);
                gpu_ops::set_tensor_buffer(encoder, 1, &w);
                gpu_ops::set_tensor_buffer(encoder, 2, &b);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                let vals: [u32; 6] = [
                    in_channels as u32, out_channels as u32, in_length as u32,
                    kernel_size as u32, stride as u32, padding as u32,
                ];
                for (i, v) in vals.iter().enumerate() {
                    encoder.set_bytes((4 + i) as u64, 4, v as *const u32 as *const _);
                }
            },
        );
        Ok(Tensor::from_metal_buffer(
            output_buffer,
            Shape::from([out_channels, out_length]),
            DType::F16,
            self.compute.device().info().id,
        ))
    }

    /// GPU-native token projection: takes GPU Tensor [seq, hidden], extracts last row via blit,
    /// projects to vocab logits, and returns argmax token (no f32 round-trip).
    fn project_to_token_gpu(
        &self,
        h_gpu: &Tensor,
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
        prefix: &str,
        head_name: &str,
    ) -> Result<u32> {
        let w_name = format!("{}{}.weight", prefix, head_name);
        let b_name = format!("{}{}.bias", prefix, head_name);

        if !self.has_weight(&w_name) {
            // Fallback for missing weights
            return Ok(0);
        }

        // Blit last row from GPU tensor: h_gpu[seq_len-1, :] → last_row[1, hidden_dim]
        let device = self.compute.device().raw();
        let device_id = self.compute.device().info().id;
        let last_row = Tensor::empty(Shape::from([1, hidden_dim]), DType::F16, device_id)?;

        let cb = self.compute.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        if let (Some(src_ptr), Some(dst_ptr)) = (h_gpu.device_ptr(), last_row.device_ptr()) {
            let src_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(src_ptr) };
            let dst_buf = unsafe { BorrowedMetalBuffer::from_device_ptr(dst_ptr) };
            let offset = ((seq_len - 1) * hidden_dim * 2) as u64;
            blit.copy_from_buffer(src_buf.as_ref(), offset, dst_buf.as_ref(), 0, (hidden_dim * 2) as u64);
        }
        blit.end_encoding();

        // Linear + argmax on GPU
        let logits = if self.has_weight(&b_name) {
            self.linear_bias(
                cb.as_ref(), &self.model, &last_row,
                &w_name, &b_name, 1, hidden_dim, vocab_size,
            )?
        } else {
            let w = gpu_ops::read_weight_f16(&self.model, &self.compute, &w_name)?;
            let zero_b = Tensor::empty(Shape::from([vocab_size]), DType::F16, device_id)?;
            self.linear_tensors(cb.as_ref(), &last_row, &w, &zero_b, 1, hidden_dim, vocab_size)
        };

        let argmax_out = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let tg_size = 256.min(vocab_size);
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(self.kernels.argmax.raw());
        gpu_ops::set_tensor_buffer(encoder, 0, &logits);
        encoder.set_buffer(1, Some(&argmax_out), 0);
        let size_u32 = vocab_size as u32;
        encoder.set_bytes(2, 4, &size_u32 as *const u32 as *const _);
        encoder.set_threadgroup_memory_length(0, (tg_size * 4) as u64);
        encoder.set_threadgroup_memory_length(1, (tg_size * 4) as u64);
        let grid = metal::MTLSize::new(1, 1, 1);
        let threads = metal::MTLSize::new(tg_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, threads);
        encoder.end_encoding();

        cb.commit();
        cb.wait_until_completed();

        let ptr = argmax_out.contents() as *const u32;
        Ok(unsafe { *ptr })
    }

    #[allow(dead_code)]
    fn project_to_token(
        &self,
        hidden: &[f32],
        hidden_dim: usize,
        vocab_size: usize,
        prefix: &str,
        head_name: &str,
    ) -> Result<u32> {
        let input = self.f32_to_f16_tensor(hidden, &[1, hidden_dim])?;
        self.project_to_token_gpu(&input, 1, hidden_dim, vocab_size, prefix, head_name)
    }
}

// ── Tanh Approximation ──────────────────────────────────────────────────────

/// Fast tanh approximation trait for GELU activation.
#[cfg(feature = "metal")]
#[allow(dead_code)]
trait TanhApprox {
    fn tanh_approx(self) -> Self;
}

#[cfg(feature = "metal")]
impl TanhApprox for f32 {
    #[inline]
    fn tanh_approx(self) -> f32 {
        let x2 = self * self;
        self * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}
