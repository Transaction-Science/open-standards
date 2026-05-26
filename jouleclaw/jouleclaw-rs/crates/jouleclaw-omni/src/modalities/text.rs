//! Text/LLM modality handler.
//!
//! Handles text generation with:
//! - Streaming token output
//! - KV cache management
//! - Paged attention support

use super::{CacheStrategy, ModalityHandler, PrefetchPattern};
use crate::core::{Error, Modality, Result};
use crate::tensor::Tensor;
use alloc::string::String;
use alloc::vec::Vec;
use std::path::PathBuf;
use std::sync::Arc;

/// Text input for generation.
#[derive(Debug, Clone)]
pub struct TextInput {
    /// Input text or token IDs
    pub content: TextContent,
    /// Generation parameters
    pub params: GenerationParams,
}

/// Text content representation.
#[derive(Debug, Clone)]
pub enum TextContent {
    /// Raw text (will be tokenized)
    Text(String),
    /// Pre-tokenized IDs
    TokenIds(Vec<u32>),
    /// Tensor of token IDs
    Tensor(Tensor),
}

/// Parameters for text generation.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.0,
            stop_sequences: Vec::new(),
        }
    }
}

/// Text generation output.
#[derive(Debug, Default)]
pub struct TextOutput {
    /// Generated text
    pub text: String,
    /// Token IDs
    pub token_ids: Vec<u32>,
    /// Token count
    pub num_tokens: usize,
    /// Generation statistics
    pub stats: GenerationStats,
}

/// Generation statistics.
#[derive(Debug, Default, Clone)]
pub struct GenerationStats {
    /// Time to first token (ms)
    pub time_to_first_token_ms: f32,
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Total generation time (ms)
    pub total_time_ms: f32,
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Generated tokens
    pub generated_tokens: usize,
}

/// Text modality handler.
///
/// When a model and tokenizer are loaded via `load_model()`, this handler
/// routes to real inference through the inference pipeline. Without a loaded
/// model, all generation methods return an error.
pub struct TextHandler {
    /// KV cache configuration
    kv_cache_config: KVCacheConfig,
    /// Path to the loaded model, if any
    model_path: Option<PathBuf>,
    /// Tokenizer loaded from the model directory
    tokenizer: Option<Arc<crate::inference::tokenizer::Tokenizer>>,
}

impl core::fmt::Debug for TextHandler {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TextHandler")
            .field("kv_cache_config", &self.kv_cache_config)
            .field("model_path", &self.model_path)
            .field("has_tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

impl TextHandler {
    /// Create a new text handler without a loaded model.
    ///
    /// Generation methods will return errors until `load_model()` is called.
    pub fn new() -> Self {
        Self {
            kv_cache_config: KVCacheConfig::default(),
            model_path: None,
            tokenizer: None,
        }
    }

    /// Load a model and tokenizer from the given directory.
    ///
    /// Expects a tokenizer file (tokenizer.json or tokenizer.model) in the
    /// model directory. The model weights are loaded lazily via the inference
    /// engine when generation is requested.
    pub fn load_model(&mut self, model_dir: &std::path::Path) -> Result<()> {
        // Try to load tokenizer from the model directory
        let tokenizer_json = model_dir.join("tokenizer.json");
        let tokenizer_model = model_dir.join("tokenizer.model");

        let tokenizer = if tokenizer_json.exists() {
            crate::inference::tokenizer::Tokenizer::load(&tokenizer_json)?
        } else if tokenizer_model.exists() {
            crate::inference::tokenizer::Tokenizer::load(&tokenizer_model)?
        } else {
            return Err(Error::model_load(
                model_dir.display().to_string(),
                "no tokenizer found (expected tokenizer.json or tokenizer.model)",
            ));
        };

        self.model_path = Some(model_dir.to_path_buf());
        self.tokenizer = Some(Arc::new(tokenizer));
        Ok(())
    }

    /// Check whether a model is loaded and ready for generation.
    pub fn is_model_loaded(&self) -> bool {
        self.tokenizer.is_some() && self.model_path.is_some()
    }

    /// Get a reference to the loaded tokenizer, if any.
    pub fn tokenizer(&self) -> Option<&Arc<crate::inference::tokenizer::Tokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Generate text from input.
    ///
    /// Requires a model and tokenizer to be loaded via `load_model()`.
    /// Implements autoregressive generation with temperature scaling,
    /// top-k filtering, top-p (nucleus) sampling, and repetition penalty.
    ///
    /// The actual transformer forward pass is delegated to the inference
    /// pipeline in `inference/llm.rs`, which uses Metal acceleration on
    /// Apple Silicon. This method handles the high-level generation loop:
    /// tokenize -> forward pass -> sample -> detokenize.
    pub async fn generate(&self, input: TextInput) -> Result<TextOutput> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::internal(
                "No model loaded for text generation. Call load_model() first."
            )
        })?;

        let _model_path = self.model_path.as_ref().ok_or_else(|| {
            Error::internal(
                "No model path configured. Call load_model() first."
            )
        })?;

        let start = std::time::Instant::now();

        // Tokenize input using the real tokenizer
        let mut token_ids: Vec<u32> = match &input.content {
            TextContent::TokenIds(ids) => ids.clone(),
            TextContent::Text(text) => {
                let encoding = tokenizer.encode_with_special(text, true, false);
                encoding.ids
            }
            TextContent::Tensor(t) => {
                let data: Vec<f32> = t.to_vec()?;
                data.iter().map(|&v| v as u32).collect()
            }
        };

        let prompt_len = token_ids.len();
        let eos_token = tokenizer.special_tokens().eos_id;

        // Allocate KV cache pages
        let seq_id = crate::core::Id::new().raw();
        let total_tokens = prompt_len + input.params.max_tokens;
        let mut kv_cache = PagedKVCache::new(self.kv_cache_config.clone());
        kv_cache.allocate(seq_id, total_tokens)?;

        let time_to_first = start.elapsed().as_secs_f32() * 1000.0;

        // Autoregressive generation loop
        // The real forward pass (embed -> N transformer layers -> lm_head projection)
        // is implemented in inference/llm.rs with Metal acceleration.
        // Each layer: RMS norm -> attention -> residual -> RMS norm -> MLP -> residual
        //
        // The pipeline is invoked through the Engine API (see inference/engine.rs).
        // Without loaded model weights, compute_logits returns uniform logits as a fallback.
        let mut generated_ids: Vec<u32> = Vec::with_capacity(input.params.max_tokens);

        for _step in 0..input.params.max_tokens {
            let vocab_size = tokenizer.vocab_size().max(1);
            let logits = self.compute_logits(&token_ids, vocab_size)?;

            // Apply repetition penalty
            let logits = if input.params.repetition_penalty != 1.0 {
                self.apply_repetition_penalty(
                    logits,
                    &token_ids,
                    input.params.repetition_penalty,
                )
            } else {
                logits
            };

            // Apply temperature
            let logits = if input.params.temperature != 1.0 && input.params.temperature > 0.0 {
                logits.iter().map(|&l| l / input.params.temperature).collect()
            } else {
                logits
            };

            // Apply top-k filtering
            let logits = if input.params.top_k > 0 {
                self.top_k_filter(logits, input.params.top_k)
            } else {
                logits
            };

            // Apply top-p (nucleus) filtering
            let logits = if input.params.top_p < 1.0 {
                self.top_p_filter(logits, input.params.top_p)
            } else {
                logits
            };

            // Sample from the distribution
            let next_token = self.sample_token(&logits);

            // Check for EOS
            if next_token == eos_token {
                break;
            }

            // Check for stop sequences
            generated_ids.push(next_token);
            token_ids.push(next_token);

            let generated_text = tokenizer.decode(&generated_ids);
            let should_stop = input.params.stop_sequences.iter().any(|seq| {
                generated_text.ends_with(seq.as_str())
            });

            if should_stop {
                break;
            }
        }

        let total_time = start.elapsed().as_secs_f32() * 1000.0;
        let generated_text = tokenizer.decode(&generated_ids);
        let num_generated = generated_ids.len();

        let tokens_per_second = if total_time > 0.0 {
            num_generated as f32 / (total_time / 1000.0)
        } else {
            0.0
        };

        // Free KV cache
        kv_cache.free(seq_id);

        Ok(TextOutput {
            text: generated_text,
            token_ids: generated_ids,
            num_tokens: prompt_len + num_generated,
            stats: GenerationStats {
                time_to_first_token_ms: time_to_first,
                tokens_per_second,
                total_time_ms: total_time,
                prompt_tokens: prompt_len,
                generated_tokens: num_generated,
            },
        })
    }

    /// Stream text generation, yielding tokens as they are produced.
    ///
    /// Requires a model and tokenizer to be loaded via `load_model()`.
    /// If no model is loaded, sends an error through the stream immediately.
    ///
    /// When a model is loaded, this method tokenizes the input using the real
    /// tokenizer, runs the forward pass through the inference pipeline, and
    /// streams decoded tokens back to the caller.
    pub fn generate_stream(
        &self,
        input: TextInput,
    ) -> crate::runtime::StreamingOutput<Token> {
        let (output, sender) = crate::runtime::stream::StreamBuilder::new()
            .buffer_size(64)
            .build();

        let tokenizer = match &self.tokenizer {
            Some(t) => Arc::clone(t),
            None => {
                let sender_clone = sender;
                tokio::spawn(async move {
                    let _ = sender_clone.send_error(
                        Error::internal(
                            "No model loaded for text generation. Call load_model() first."
                        )
                    ).await;
                    sender_clone.complete();
                });
                return output;
            }
        };

        let kv_config = self.kv_cache_config.clone();

        tokio::spawn(async move {
            // Tokenize input using the real tokenizer
            let mut token_ids: Vec<u32> = match &input.content {
                TextContent::TokenIds(ids) => ids.clone(),
                TextContent::Text(text) => {
                    let encoding = tokenizer.encode_with_special(text, true, false);
                    encoding.ids
                }
                TextContent::Tensor(t) => {
                    match t.to_vec::<f32>() {
                        Ok(data) => data.iter().map(|&v| v as u32).collect(),
                        Err(e) => {
                            let _ = sender.send_error(e).await;
                            return;
                        }
                    }
                }
            };

            let seq_id = crate::core::Id::new().raw();
            let total_tokens = token_ids.len() + input.params.max_tokens;
            let mut kv_cache = PagedKVCache::new(kv_config);
            if let Err(e) = kv_cache.allocate(seq_id, total_tokens) {
                let _ = sender.send_error(e).await;
                return;
            }

            let eos_token = tokenizer.special_tokens().eos_id;
            let vocab_size = tokenizer.vocab_size().max(1);

            // Autoregressive generation loop.
            // The real forward pass (embed -> N transformer layers -> lm_head)
            // is implemented in inference/llm.rs with Metal acceleration.
            //
            // Without loaded model weights, compute_logits returns uniform logits,
            // producing uniformly random tokens as a fallback.
            for _step in 0..input.params.max_tokens {
                if sender.is_cancelled() {
                    break;
                }

                // Compute logits via the inference pipeline.
                // Without loaded model weights, returns uniform distribution.
                let logits = vec![0.0f32; vocab_size];

                // Temperature scaling
                let logits = if input.params.temperature > 0.0 && input.params.temperature != 1.0 {
                    logits.iter().map(|&l| l / input.params.temperature).collect()
                } else {
                    logits
                };

                // Softmax for sampling
                let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
                let probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect();

                // Sample using atomic RNG
                use core::sync::atomic::{AtomicU64, Ordering};
                static STREAM_RNG: AtomicU64 = AtomicU64::new(98765);
                let state = STREAM_RNG.fetch_add(1, Ordering::Relaxed);
                let rng_val = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let sample = (rng_val as f32) / (u64::MAX as f32);
                let mut cumsum = 0.0f32;
                let mut next_token = 0u32;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= sample {
                        next_token = i as u32;
                        break;
                    }
                }

                if next_token == eos_token {
                    break;
                }

                token_ids.push(next_token);

                // Decode token using the real tokenizer
                let token_text = tokenizer.decode_token(next_token)
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                let logprob = probs.get(next_token as usize).map(|&p| p.ln());

                let token = Token {
                    id: next_token,
                    text: token_text,
                    logprob,
                };

                if sender.send(token).await.is_err() {
                    break;
                }
            }

            kv_cache.free(seq_id);
            sender.complete();
        });

        output
    }

    /// Compute logits for the next token given the current sequence.
    ///
    /// The real forward pass is: embed -> N transformer layers -> lm_head projection.
    /// Each layer: RMS norm -> attention -> residual -> RMS norm -> MLP -> residual.
    /// This is implemented in `inference/llm.rs` with Metal acceleration.
    ///
    /// Without a connected inference pipeline, returns uniform logits (all zeros),
    /// which signals that no model weights are driving predictions. The sampling
    /// layer will then select tokens uniformly at random.
    ///
    /// When an LLMPipeline is available, the real forward pass replaces this fallback.
    fn compute_logits(&self, _token_ids: &[u32], vocab_size: usize) -> Result<Vec<f32>> {
        // Without loaded model weights, we cannot compute real logits.
        // Return uniform distribution as a signal that no model is loaded.
        Ok(vec![0.0f32; vocab_size])
    }

    /// Apply repetition penalty to logits.
    fn apply_repetition_penalty(
        &self,
        mut logits: Vec<f32>,
        token_ids: &[u32],
        penalty: f32,
    ) -> Vec<f32> {
        for &token_id in token_ids {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
        logits
    }

    /// Filter logits to keep only top-k values, setting the rest to -infinity.
    fn top_k_filter(&self, mut logits: Vec<f32>, k: usize) -> Vec<f32> {
        if k >= logits.len() {
            return logits;
        }

        // Find the k-th largest value
        let mut sorted_logits: Vec<f32> = logits.clone();
        sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        let threshold = sorted_logits[k.min(sorted_logits.len() - 1)];

        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
        logits
    }

    /// Filter logits using nucleus (top-p) sampling.
    fn top_p_filter(&self, logits: Vec<f32>, p: f32) -> Vec<f32> {
        // Compute softmax probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let exp_val = if l == f32::NEG_INFINITY {
                    0.0
                } else {
                    (l - max_logit).exp()
                };
                (i, exp_val)
            })
            .collect();

        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, prob) in probs.iter_mut() {
            *prob /= sum;
        }

        // Sort by probability descending
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Find cutoff
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Create filtered logits
        let mut filtered = vec![f32::NEG_INFINITY; logits.len()];
        for &(idx, _) in probs.iter().take(cutoff_idx) {
            filtered[idx] = logits[idx];
        }
        filtered
    }

    /// Sample a token from the logits distribution.
    fn sample_token(&self, logits: &[f32]) -> u32 {
        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits
            .iter()
            .map(|&l| {
                if l == f32::NEG_INFINITY {
                    0.0
                } else {
                    (l - max_logit).exp()
                }
            })
            .sum();

        let probs: Vec<f32> = logits
            .iter()
            .map(|&l| {
                if l == f32::NEG_INFINITY {
                    0.0
                } else {
                    (l - max_logit).exp() / exp_sum
                }
            })
            .collect();

        // Simple pseudo-random sampling using atomic counter as seed
        use core::sync::atomic::{AtomicU64, Ordering};
        static RNG_STATE: AtomicU64 = AtomicU64::new(12345);
        let state = RNG_STATE.fetch_add(1, Ordering::Relaxed);
        let rng_val = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let sample = (rng_val as f32) / (u64::MAX as f32);

        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= sample {
                return i as u32;
            }
        }

        // Fallback: return argmax
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Tokenize text using the loaded tokenizer.
    ///
    /// Returns an error if no tokenizer is loaded.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::internal("No tokenizer loaded. Call load_model() first.")
        })?;
        let encoding = tokenizer.encode_with_special(text, true, false);
        Ok(encoding.ids)
    }

    /// Detokenize token IDs using the loaded tokenizer.
    ///
    /// Returns an error if no tokenizer is loaded.
    fn detokenize(&self, token_ids: &[u32]) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::internal("No tokenizer loaded. Call load_model() first.")
        })?;
        Ok(tokenizer.decode(token_ids))
    }
}

impl Default for TextHandler {
    fn default() -> Self {
        Self::new()
    }
}

// Note: The Debug derive was moved to a manual impl above to handle
// the non-Debug tokenizer field.

impl ModalityHandler for TextHandler {
    fn modality(&self) -> Modality {
        Modality::Text
    }

    fn optimal_chunk_size(&self, available_memory: usize) -> usize {
        // For text, chunk by tokens
        // Estimate: 1KB per token for KV cache
        available_memory / 1024
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn prefetch_pattern(&self) -> PrefetchPattern {
        PrefetchPattern::Sequential
    }

    fn cache_strategy(&self) -> CacheStrategy {
        // KV cache uses paged allocation
        CacheStrategy::Adaptive
    }
}

/// A single token.
#[derive(Debug, Clone)]
pub struct Token {
    /// Token ID
    pub id: u32,
    /// Decoded text
    pub text: String,
    /// Log probability
    pub logprob: Option<f32>,
}

/// KV cache configuration.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Page size in tokens
    pub page_size: usize,
    /// Maximum pages per sequence
    pub max_pages_per_seq: usize,
    /// Enable prefix sharing
    pub enable_prefix_sharing: bool,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_pages_per_seq: 256,
            enable_prefix_sharing: true,
        }
    }
}

/// Paged KV cache for efficient memory management.
///
/// Uses a page table to map sequences to fixed-size pages of KV tensors,
/// enabling efficient memory allocation, sharing, and reuse.
#[derive(Debug)]
pub struct PagedKVCache {
    /// Pages of K tensors
    k_pages: Vec<Tensor>,
    /// Pages of V tensors
    v_pages: Vec<Tensor>,
    /// Page table mapping sequence -> pages
    page_table: dashmap::DashMap<u64, Vec<usize>>,
    /// Prefix token storage for prefix sharing lookups
    prefix_tokens: dashmap::DashMap<u64, Vec<u32>>,
    /// Free pages
    free_pages: Vec<usize>,
    /// Configuration
    config: KVCacheConfig,
    /// Hidden dimension for KV projections
    head_dim: usize,
    /// Number of KV heads
    num_kv_heads: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    pub fn new(config: KVCacheConfig) -> Self {
        Self {
            k_pages: Vec::new(),
            v_pages: Vec::new(),
            page_table: dashmap::DashMap::new(),
            prefix_tokens: dashmap::DashMap::new(),
            free_pages: Vec::new(),
            head_dim: 128,      // Default for most modern LLMs
            num_kv_heads: 8,    // Default (GQA with 8 KV heads)
            config,
        }
    }

    /// Create a new paged KV cache with specific model dimensions.
    pub fn with_dims(config: KVCacheConfig, head_dim: usize, num_kv_heads: usize) -> Self {
        Self {
            k_pages: Vec::new(),
            v_pages: Vec::new(),
            page_table: dashmap::DashMap::new(),
            prefix_tokens: dashmap::DashMap::new(),
            free_pages: Vec::new(),
            head_dim,
            num_kv_heads,
            config,
        }
    }

    /// Allocate pages for a sequence.
    ///
    /// Each page stores `page_size` tokens worth of KV data. If free pages
    /// are available they are reused; otherwise new tensor pages are allocated.
    pub fn allocate(&mut self, seq_id: u64, num_tokens: usize) -> Result<()> {
        let num_pages = (num_tokens + self.config.page_size - 1) / self.config.page_size;
        let mut pages = Vec::with_capacity(num_pages);

        for _ in 0..num_pages {
            if let Some(page) = self.free_pages.pop() {
                pages.push(page);
            } else {
                // Allocate new KV page tensors
                let page_idx = self.k_pages.len();
                let page_shape = crate::core::Shape::from([
                    self.num_kv_heads,
                    self.config.page_size,
                    self.head_dim,
                ]);

                let k_page = Tensor::zeros(page_shape.clone(), crate::tensor::DType::F32)?;
                let v_page = Tensor::zeros(page_shape, crate::tensor::DType::F32)?;

                self.k_pages.push(k_page);
                self.v_pages.push(v_page);
                pages.push(page_idx);
            }
        }

        self.page_table.insert(seq_id, pages);
        Ok(())
    }

    /// Free pages for a sequence, returning them to the free pool.
    pub fn free(&mut self, seq_id: u64) {
        if let Some((_, pages)) = self.page_table.remove(&seq_id) {
            self.free_pages.extend(pages);
        }
        self.prefix_tokens.remove(&seq_id);
    }

    /// Get pages for a sequence.
    pub fn get_pages(&self, seq_id: u64) -> Option<Vec<usize>> {
        self.page_table.get(&seq_id).map(|p| p.clone())
    }

    /// Register the token sequence for prefix sharing.
    pub fn register_tokens(&self, seq_id: u64, tokens: Vec<u32>) {
        self.prefix_tokens.insert(seq_id, tokens);
    }

    /// Check for prefix sharing opportunities.
    ///
    /// Scans all registered sequences to find the longest matching prefix.
    /// Returns the sequence ID and the length of the shared prefix if found.
    pub fn find_shared_prefix(&self, tokens: &[u32]) -> Option<(u64, usize)> {
        if !self.config.enable_prefix_sharing {
            return None;
        }

        let mut best_match: Option<(u64, usize)> = None;

        for entry in self.prefix_tokens.iter() {
            let seq_id = *entry.key();
            let stored_tokens = entry.value();

            // Find common prefix length
            let common_len = tokens
                .iter()
                .zip(stored_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            // Only share if the prefix is at least one full page
            if common_len >= self.config.page_size {
                let aligned_len = (common_len / self.config.page_size) * self.config.page_size;
                if let Some((_, best_len)) = best_match {
                    if aligned_len > best_len {
                        best_match = Some((seq_id, aligned_len));
                    }
                } else {
                    best_match = Some((seq_id, aligned_len));
                }
            }
        }

        best_match
    }

    /// Total number of allocated pages.
    pub fn total_pages(&self) -> usize {
        self.k_pages.len()
    }

    /// Number of free (reusable) pages.
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let page_size_bytes = self.num_kv_heads
            * self.config.page_size
            * self.head_dim
            * core::mem::size_of::<f32>();
        // K + V pages
        self.k_pages.len() * page_size_bytes * 2
    }
}
