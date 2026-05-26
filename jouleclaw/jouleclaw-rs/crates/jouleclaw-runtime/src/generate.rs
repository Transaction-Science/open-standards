//! High-level text generation utility.
//!
//! Wraps:
//! - tokenization (SPM or BPE)
//! - prefill via decode_step (one large step) or via the simpler one-shot
//!   prefill graph when no KV cache is needed
//! - autoregressive decoding (one token per step, growing the KV cache)
//! - sampling with configurable strategy
//! - detokenization
//!
//! into one ergonomic call. The generation is fully deterministic given
//! the sampler's seed: identical (model, prompt, sampling) inputs yield
//! identical outputs across runs and across machines.

use jouleclaw_loader_gguf::decode::build_decode_step_graph;
use jouleclaw_loader_gguf::kv_cache::KvCache;
use jouleclaw_loader_gguf::kv_cache_inplace::{
    build_decode_step_graph_inplace, build_decode_step_graph_inplace_const,
    InPlaceKvCache, ShortConvStateCache, KV_POS_INPUT,
};
use jouleclaw_loader_gguf::llama::{LlamaConfig, LoadError};
use jouleclaw_loader_gguf::sample::{sample_logits_with_history, SamplingConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::GgufModel;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use crate::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;

/// Configuration for a single generation call.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of tokens to generate. Generation stops earlier
    /// if EOS is sampled.
    pub max_new_tokens: usize,
    /// Sampling strategy (greedy, temperature, top-k, top-p).
    pub sampling: SamplingConfig,
    /// Whether to prepend BOS to the prompt. Most Llama-family models
    /// expect BOS; set false for raw continuation.
    pub add_bos: bool,
    /// Whether to use SPM (Llama 1/2 style) or BPE (Llama 3 / Mistral /
    /// Qwen style) tokenization. Tries to auto-detect from the vocab's
    /// `bpe_merges` field; explicit setting overrides.
    pub tokenizer_kind: TokenizerKind,
    /// KV cache implementation: in-place (constant memory) or concat
    /// (grows per step). Default: in-place.
    pub cache_kind: KvCacheKind,
    /// Maximum total sequence length (prompt + generated). Used by the
    /// in-place cache to size the preallocated buffer. If `None`, defaults
    /// to `prompt_len + max_new_tokens + 16` (small headroom for safety).
    /// Ignored when `cache_kind` is `Concat`.
    pub max_seq: Option<usize>,
    /// Stop strings: if any of these appears as a suffix of the
    /// generated text, generation halts before emitting the rest of
    /// the current token. Useful for chat templates ("\nUser:",
    /// "<|im_end|>", etc).
    pub stop_strings: Vec<String>,
}

/// Which KV cache implementation `generate()` should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheKind {
    /// Preallocated buffer of size `max_seq`. Constant memory per step.
    /// Recommended for production decode. Default.
    InPlace,
    /// Functional, grows per step via Concat. Simpler; fine for short
    /// prompts and tests.
    Concat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// Auto-detect: use BPE if merges are present, otherwise SPM.
    Auto,
    Spm,
    Bpe,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            sampling: SamplingConfig::greedy(),
            add_bos: true,
            tokenizer_kind: TokenizerKind::Auto,
            cache_kind: KvCacheKind::InPlace,
            max_seq: None,
            stop_strings: Vec::new(),
        }
    }
}

/// One generation result.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Output text (decoded; does not include the prompt).
    pub text: String,
    /// Output token IDs (does not include the prompt).
    pub tokens: Vec<u32>,
    /// True if generation stopped because EOS was sampled.
    pub stopped_at_eos: bool,
    /// Number of prompt tokens (after tokenization).
    pub prompt_token_count: usize,
}

/// Errors during generation.
#[derive(Debug)]
pub enum GenerateError {
    /// Loader error (typically when building the decode-step graph for
    /// a non-Llama architecture).
    Load(LoadError),
    /// Runtime error during graph compile or execute.
    Runtime(jouleclaw_core::error::Error),
    /// Tokenizer produced zero tokens for the prompt — refuse to generate
    /// (would imply BOS-only input, which most callers don't want).
    EmptyPromptTokens,
}

impl From<LoadError> for GenerateError {
    fn from(e: LoadError) -> Self { Self::Load(e) }
}

impl From<jouleclaw_core::error::Error> for GenerateError {
    fn from(e: jouleclaw_core::error::Error) -> Self { Self::Runtime(e) }
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(e) => write!(f, "load error: {}", e),
            Self::Runtime(e) => write!(f, "runtime error: {:?}", e),
            Self::EmptyPromptTokens => write!(f, "prompt tokenized to zero tokens"),
        }
    }
}

impl std::error::Error for GenerateError {}

/// Generate text from a prompt using the model.
///
/// Pipeline: tokenize prompt → prefill (one decode step over all prompt
/// tokens) → sample first new token → loop (one decode step per new
/// token, sampling each) → detokenize.
///
/// The KV cache is allocated fresh per call. To reuse a cache across calls
/// (e.g., for multi-turn chat), use the lower-level decode-step builders
/// directly.
pub fn generate(
    model: &GgufModel,
    vocab: &Vocab,
    prompt: &str,
    cfg: &GenerateConfig,
) -> Result<GenerateResult, GenerateError> {
    let tokenizer_kind = resolve_tokenizer_kind(vocab, cfg.tokenizer_kind);

    let prompt_tokens = match tokenizer_kind {
        TokenizerKind::Spm => vocab.encode_spm(prompt, cfg.add_bos),
        TokenizerKind::Bpe => vocab.encode_bpe_regex(prompt, cfg.add_bos),
        TokenizerKind::Auto => unreachable!(),
    };
    if prompt_tokens.is_empty() {
        return Err(GenerateError::EmptyPromptTokens);
    }
    let prompt_token_count = prompt_tokens.len();

    // Pick the cache path.
    let runtime = Runtime::boot();
    let (generated_tokens, stopped_at_eos) = match cfg.cache_kind {
        KvCacheKind::Concat => generate_with_concat_cache(
            model, vocab, &prompt_tokens, &runtime, cfg, tokenizer_kind,
        )?,
        KvCacheKind::InPlace => generate_with_inplace_cache(
            model, vocab, &prompt_tokens, &runtime, cfg, tokenizer_kind,
        )?,
    };

    let raw_text = match tokenizer_kind {
        TokenizerKind::Spm => vocab.decode_spm(&generated_tokens),
        TokenizerKind::Bpe => vocab.decode_bpe(&generated_tokens),
        TokenizerKind::Auto => unreachable!(),
    };
    // If a stop string fired, trim the output text up to (but not including)
    // the stop string. The tokens are left untouched.
    let text = if !cfg.stop_strings.is_empty() {
        if let Some(pos) = find_stop_string(&raw_text, &cfg.stop_strings) {
            raw_text[..pos].to_string()
        } else {
            raw_text
        }
    } else {
        raw_text
    };

    Ok(GenerateResult {
        text,
        tokens: generated_tokens,
        stopped_at_eos,
        prompt_token_count,
    })
}

/// Find the earliest occurrence of any stop string in `text`, returning
/// the byte position. Returns None if none matched.
pub(crate) fn find_stop_string(text: &str, stops: &[String]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for s in stops {
        if s.is_empty() { continue; }
        if let Some(p) = text.find(s.as_str()) {
            best = Some(match best { Some(b) => b.min(p), None => p });
        }
    }
    best
}

/// Generation path using the concat-based KV cache. Simpler; grows per step.
fn generate_with_concat_cache(
    model: &GgufModel,
    vocab: &Vocab,
    prompt_tokens: &[u32],
    runtime: &Runtime,
    cfg: &GenerateConfig,
    tokenizer_kind: TokenizerKind,
) -> Result<(Vec<u32>, bool), GenerateError> {
    let mut cache = KvCache::empty(LlamaConfig::from_metadata(model)?.block_count);

    let prompt_tensor = tokens_to_tensor(prompt_tokens);
    let prefill_logits = run_concat_step(
        model, runtime, &mut cache, prompt_tensor, prompt_tokens.len())?;
    let vocab_size = vocab.len();

    let last_row = &prefill_logits[(prompt_tokens.len() - 1) * vocab_size..];
    let mut next = sample_logits_with_history(last_row, &cfg.sampling, prompt_tokens);

    let mut generated = Vec::with_capacity(cfg.max_new_tokens);
    let mut stopped = false;
    for _ in 0..cfg.max_new_tokens {
        if Some(next) == vocab.eos_id { stopped = true; break; }
        generated.push(next);

        // Stop-string check (post-token).
        if !cfg.stop_strings.is_empty() {
            let so_far = match tokenizer_kind {
                TokenizerKind::Spm => vocab.decode_spm(&generated),
                TokenizerKind::Bpe => vocab.decode_bpe(&generated),
                TokenizerKind::Auto => unreachable!(),
            };
            if let Some(stop_at) = find_stop_string(&so_far, &cfg.stop_strings) {
                // Trim text by re-tokenizing — but since we're returning
                // the tokens, we just stop here. The trimming happens at
                // the caller level (the final text is detokenized fresh).
                let _ = stop_at;
                stopped = true;
                break;
            }
        }

        let single = tokens_to_tensor(&[next]);
        let logits = run_concat_step(model, runtime, &mut cache, single, 1)?;
        let recent: Vec<u32> = prompt_tokens.iter().chain(generated.iter()).copied().collect();
        next = sample_logits_with_history(&logits, &cfg.sampling, &recent);
    }
    Ok((generated, stopped))
}

/// Generation path using the in-place KV cache. Constant memory per step.
fn generate_with_inplace_cache(
    model: &GgufModel,
    vocab: &Vocab,
    prompt_tokens: &[u32],
    runtime: &Runtime,
    cfg: &GenerateConfig,
    tokenizer_kind: TokenizerKind,
) -> Result<(Vec<u32>, bool), GenerateError> {
    // Compute max_seq for the preallocated buffer.
    let max_seq = cfg.max_seq.unwrap_or_else(|| {
        prompt_tokens.len() + cfg.max_new_tokens + 16
    });
    if max_seq < prompt_tokens.len() + cfg.max_new_tokens {
        return Err(GenerateError::Load(LoadError::UnsupportedArchitecture(
            format!("max_seq {} too small for prompt {} + max_new_tokens {}",
                max_seq, prompt_tokens.len(), cfg.max_new_tokens))));
    }

    let mut cache = InPlaceKvCache::for_model(model, max_seq)?;

    let prompt_tensor = tokens_to_tensor(prompt_tokens);
    let prefill = run_inplace_step(
        model, runtime, &mut cache, prompt_tensor, prompt_tokens.len())?;
    let vocab_size = vocab.len();

    let last_row = &prefill.logits[(prompt_tokens.len() - 1) * vocab_size..];
    let mut next = sample_logits_with_history(last_row, &cfg.sampling, prompt_tokens);

    let mut generated = Vec::with_capacity(cfg.max_new_tokens);
    let mut stopped = false;
    for _ in 0..cfg.max_new_tokens {
        if Some(next) == vocab.eos_id { stopped = true; break; }
        generated.push(next);

        if !cfg.stop_strings.is_empty() {
            let so_far = match tokenizer_kind {
                TokenizerKind::Spm => vocab.decode_spm(&generated),
                TokenizerKind::Bpe => vocab.decode_bpe(&generated),
                TokenizerKind::Auto => unreachable!(),
            };
            if find_stop_string(&so_far, &cfg.stop_strings).is_some() {
                stopped = true;
                break;
            }
        }

        let single = tokens_to_tensor(&[next]);
        let step = run_inplace_step(model, runtime, &mut cache, single, 1)?;
        let recent: Vec<u32> = prompt_tokens.iter().chain(generated.iter()).copied().collect();
        next = sample_logits_with_history(&step.logits, &cfg.sampling, &recent);
    }
    Ok((generated, stopped))
}

pub(crate) fn resolve_tokenizer_kind(vocab: &Vocab, requested: TokenizerKind) -> TokenizerKind {
    match requested {
        TokenizerKind::Auto => {
            // `tokenizer.ggml.model` is the authoritative signal: "llama"
            // = SentencePiece (U+2581 spaces), "gpt2" = BPE with Ġ
            // byte-mapping. The presence of `tokenizer.ggml.merges` is
            // not enough — many SPM-based Llama models also ship a
            // merges array, and decoding their output through `decode_bpe`
            // produces missing-space garbage ("TheGQofGFrance" instead of
            // "The Q of G France").
            match vocab.model_name.as_str() {
                "gpt2" => TokenizerKind::Bpe,
                "llama" => TokenizerKind::Spm,
                _ => {
                    if vocab.bpe_merges.is_some() { TokenizerKind::Bpe }
                    else { TokenizerKind::Spm }
                }
            }
        }
        explicit => explicit,
    }
}

pub(crate) fn tokens_to_tensor(ids: &[u32]) -> Tensor {
    let bytes: Vec<u8> = ids.iter()
        .flat_map(|&id| (id as i32).to_le_bytes())
        .collect();
    Tensor {
        meta: TensorMeta::new(Dtype::I32, &[ids.len()]),
        storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
    }
}

/// Run one concat-based decode step: build graph, compile, execute,
/// stash updated K/V back into cache, return logits.
fn run_concat_step(
    model: &GgufModel,
    runtime: &Runtime,
    cache: &mut KvCache,
    new_tokens: Tensor,
    new_seq: usize,
) -> Result<Vec<f32>, GenerateError> {
    let step = build_decode_step_graph(model, cache, new_seq)?;
    let compiled = compile(step.graph, &runtime.kernels)?;

    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), new_tokens);
    if cache.current_seq > 0 {
        for layer in 0..step.config.block_count {
            let k = cache.k_for(layer).expect("K present").clone();
            let v = cache.v_for(layer).expect("V present").clone();
            inputs.insert(step.k_input_names[layer].clone(), k);
            inputs.insert(step.v_input_names[layer].clone(), v);
        }
    }

    let res = execute(&compiled, inputs, ExecutionOptions::default())?;

    for layer in 0..step.config.block_count {
        let k = res.outputs.get(&step.k_output_names[layer])
            .expect("kv_out_k").clone();
        let v = res.outputs.get(&step.v_output_names[layer])
            .expect("kv_out_v").clone();
        cache.put(layer, k, v);
    }

    Ok(res.outputs.get(&step.logits_output_name)
        .expect("logits").as_f32_vec())
}

/// Result of one in-place decode step: the next-row logits plus the
/// kernel-reported joule total for this step's `execute()` call. The
/// joules figure aggregates every `KernelResult.joules` produced by
/// the runtime during this step — it's the "actual" side of the
/// estimated-vs-actual divergence the cascade's calibration ledger
/// learns from.
pub struct InPlaceStepResult {
    pub logits: Vec<f32>,
    pub joules: f64,
}

/// Run one in-place decode step: build graph, compile, execute, stash
/// the updated buffers back into cache (preserving shape), return
/// `(logits, joules)`.
pub(crate) fn run_inplace_step(
    model: &GgufModel,
    runtime: &Runtime,
    cache: &mut InPlaceKvCache,
    new_tokens: Tensor,
    new_seq: usize,
) -> Result<InPlaceStepResult, GenerateError> {
    let step = build_decode_step_graph_inplace(model, cache, new_seq)?;
    let compiled = compile(step.graph, &runtime.kernels)?;

    // Take the K/V buffers OUT of the cache, leaving placeholder empties.
    // This makes the Arc strong_count == 1 at execution time, which lets
    // the executor's `scatter_inplace` aliasing fire (stealing the storage
    // instead of allocating fresh output). The cache will have valid
    // buffers again after `replace_buffers` below.
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), new_tokens);
    for layer in 0..step.config.block_count {
        let (k, v) = cache.take_buffers(layer);
        inputs.insert(step.k_input_names[layer].clone(), k);
        inputs.insert(step.v_input_names[layer].clone(), v);
    }

    let res = execute(&compiled, inputs, ExecutionOptions::default())?;

    for layer in 0..step.config.block_count {
        let k = res.outputs.get(&step.k_output_names[layer])
            .expect("kv_out_k").clone();
        let v = res.outputs.get(&step.v_output_names[layer])
            .expect("kv_out_v").clone();
        cache.replace_buffers(layer, k, v);
    }
    cache.advance(new_seq);

    Ok(InPlaceStepResult {
        logits: res.outputs.get(&step.logits_output_name)
            .expect("logits").as_f32_vec(),
        joules: res.trace.joule_accounting.total_joules,
    })
}

/// One compiled constant-topology decode step + the I/O names needed
/// to bind it. Built once per distinct `new_seq` and reused for every
/// subsequent step with that `new_seq` (decode is always `new_seq=1`,
/// so after prefill the loop reuses a single compiled graph).
pub struct CachedDecodeStep {
    compiled: crate::compile::CompiledGraph,
    block_count: usize,
    k_input_names: Vec<String>,
    v_input_names: Vec<String>,
    k_output_names: Vec<String>,
    v_output_names: Vec<String>,
    logits_output_name: String,
    /// Per-layer shortconv state input names. `Some(name)` for LFM2
    /// recurrent layers, `None` for attention layers. Empty Vec for
    /// non-LFM2 models.
    shortconv_state_input_names: Vec<Option<String>>,
    shortconv_state_output_names: Vec<Option<String>>,
}

/// Per-conversation compile cache keyed by `new_seq`. Holds the
/// const-topology decode graphs so the per-token build+compile happens
/// at most once per distinct `new_seq` instead of every step.
#[derive(Default)]
pub struct DecodeStepCache {
    by_new_seq: std::collections::HashMap<usize, CachedDecodeStep>,
    /// Sequential variant (one compiled graph per layer + embed + head).
    /// Populated lazily by `run_inplace_step_sequential`. Independent of
    /// `by_new_seq` so a Conversation can use either path.
    sequential_by_new_seq: std::collections::HashMap<usize, CachedSequentialStep>,
}

impl DecodeStepCache {
    pub fn new() -> Self { Self::default() }
    pub fn clear(&mut self) {
        self.by_new_seq.clear();
        self.sequential_by_new_seq.clear();
    }
    pub fn len(&self) -> usize { self.by_new_seq.len() }
    pub fn is_empty(&self) -> bool { self.by_new_seq.is_empty() }
}

/// Compiled sequential decode step: one embed graph + N layer graphs
/// + one head graph. Built once per `new_seq`. The layer graphs are
/// distinct compiled artifacts (each layer's weights are baked in as
/// constants), so compile cost is `n_layers + 2` × per-graph compile.
pub struct CachedSequentialStep {
    embed_compiled: crate::compile::CompiledGraph,
    embed_x_output: String,
    /// Per-layer compiled graphs, indexed by layer (0..block_count).
    layer_compiled: Vec<crate::compile::CompiledGraph>,
    layer_io: Vec<LayerIoNames>,
    head_compiled: crate::compile::CompiledGraph,
    head_x_input: String,
    head_logits_output: String,
    block_count: usize,
}

struct LayerIoNames {
    x_in: String,
    k_in: String,
    v_in: String,
    x_out: String,
    k_out: String,
    v_out: String,
}

/// Constant-topology in-place decode step with a compile-once cache.
/// First call for a given `new_seq` builds + compiles the
/// fixed-shape graph; every later call with the same `new_seq` reuses
/// it, binding only the per-step inputs (`token_ids`, `kv_pos =
/// cached_seq`, and the K/V buffers). Numerically identical to
/// `run_inplace_step` (full-buffer attention + dynamic causal offset
/// reproduces the sliced result exactly).
pub fn run_inplace_step_cached(
    model: &GgufModel,
    runtime: &Runtime,
    cache: &mut InPlaceKvCache,
    shortconv: &mut ShortConvStateCache,
    step_cache: &mut DecodeStepCache,
    new_tokens: Tensor,
    new_seq: usize,
) -> Result<InPlaceStepResult, GenerateError> {
    // When KV cache is quantized, the monolithic graph would need all
    // n_layers worth of fp32 working buffers alive simultaneously (one
    // per layer in the executor's inputs HashMap). Routing to the
    // sequential path keeps only one layer's fp32 in memory at a time,
    // which is the whole point of the int8 lever — peak memory drops
    // from ~16.6 MB (current int8) to ~5.1 MB at Bonsai's max_seq=128.
    // Numerical equivalence is bit-identical (see prism's
    // sequential_parity test).
    if cache.quant == jouleclaw_loader_gguf::kv_cache_inplace::KvQuant::Int8 {
        return run_inplace_step_sequential(
            model, runtime, cache, step_cache, new_tokens, new_seq);
    }

    let cached_seq = cache.current_seq;

    if !step_cache.by_new_seq.contains_key(&new_seq) {
        let step = build_decode_step_graph_inplace_const(model, cache, new_seq)?;
        let compiled = compile(step.graph, &runtime.kernels)?;
        step_cache.by_new_seq.insert(new_seq, CachedDecodeStep {
            compiled,
            block_count: step.config.block_count,
            k_input_names: step.k_input_names,
            v_input_names: step.v_input_names,
            k_output_names: step.k_output_names,
            v_output_names: step.v_output_names,
            logits_output_name: step.logits_output_name,
            shortconv_state_input_names: step.shortconv_state_input_names,
            shortconv_state_output_names: step.shortconv_state_output_names,
        });
    }
    let cs = step_cache.by_new_seq.get(&new_seq).expect("just inserted");

    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), new_tokens);
    // Dynamic KV position (I32 [1] = cached_seq) — the only thing that
    // changes token-to-token now that the graph is constant-topology.
    inputs.insert(
        KV_POS_INPUT.into(),
        Tensor {
            meta: jouleclaw_core::tensor::TensorMeta::new(
                jouleclaw_core::tensor::Dtype::I32, &[1]),
            storage: std::sync::Arc::new(jouleclaw_core::tensor::TensorStorage { bytes: (cached_seq as i32).to_le_bytes().to_vec(), mapped: None }),
        },
    );
    for layer in 0..cs.block_count {
        let (k, v) = cache.take_buffers(layer);
        inputs.insert(cs.k_input_names[layer].clone(), k);
        inputs.insert(cs.v_input_names[layer].clone(), v);
    }
    // LFM2 shortconv state: feed in per-layer rolling window for any
    // recurrent layer. Non-LFM2 models have empty
    // `shortconv_state_input_names` so this is a no-op.
    for layer in 0..cs.shortconv_state_input_names.len() {
        if let Some(name) = &cs.shortconv_state_input_names[layer] {
            let state = shortconv.take_state(layer)
                .ok_or_else(|| GenerateError::Load(LoadError::UnsupportedArchitecture(
                    format!("shortconv state missing for layer {}", layer))))?;
            inputs.insert(name.clone(), state);
        }
    }

    let res = execute(&cs.compiled, inputs, ExecutionOptions::default())?;

    for layer in 0..cs.block_count {
        let k = res.outputs.get(&cs.k_output_names[layer]).expect("kv_out_k").clone();
        let v = res.outputs.get(&cs.v_output_names[layer]).expect("kv_out_v").clone();
        cache.replace_buffers(layer, k, v);
    }
    for layer in 0..cs.shortconv_state_output_names.len() {
        if let Some(name) = &cs.shortconv_state_output_names[layer] {
            let state = res.outputs.get(name)
                .expect("shortconv_state_out").clone();
            shortconv.replace_state(layer, state);
        }
    }
    cache.advance(new_seq);

    Ok(InPlaceStepResult {
        logits: res.outputs.get(&cs.logits_output_name)
            .expect("logits").as_f32_vec(),
        joules: res.trace.joule_accounting.total_joules,
    })
}

/// Sequential variant of [`run_inplace_step_cached`]. Equivalent
/// numerics — the per-layer arithmetic is bit-identical to the
/// monolithic graph — but with the layer loop hoisted into Rust so
/// only one layer's K/V is in fp32 working memory at any moment.
/// Enables the cache to hold a single shared fp32 working buffer
/// (paired with int8 cold storage) instead of `n_layers` fresh
/// allocations per step.
///
/// This commit ships the substrate: graph builders + sequential
/// executor + bit-parity test against the monolithic path. The
/// follow-up commit wires `KvQuant::Int8` to use this and adds the
/// shared work buffer.
pub fn run_inplace_step_sequential(
    model: &GgufModel,
    runtime: &Runtime,
    cache: &mut InPlaceKvCache,
    step_cache: &mut DecodeStepCache,
    new_tokens: Tensor,
    new_seq: usize,
) -> Result<InPlaceStepResult, GenerateError> {
    use jouleclaw_loader_gguf::kv_cache_inplace::{
        build_embed_only_graph, build_head_only_graph, build_layer_only_graph,
    };

    let cached_seq = cache.current_seq;

    // First call for this `new_seq`: build + compile all the pieces.
    if !step_cache.sequential_by_new_seq.contains_key(&new_seq) {
        let embed = build_embed_only_graph(model, new_seq)?;
        let embed_compiled = compile(embed.graph, &runtime.kernels)?;

        let head = build_head_only_graph(model, new_seq)?;
        let head_compiled = compile(head.graph, &runtime.kernels)?;

        let mut layer_compiled = Vec::with_capacity(embed.config.block_count);
        let mut layer_io = Vec::with_capacity(embed.config.block_count);
        for layer in 0..embed.config.block_count {
            let lg = build_layer_only_graph(model, cache, layer, new_seq)?;
            let compiled = compile(lg.graph, &runtime.kernels)?;
            layer_io.push(LayerIoNames {
                x_in: lg.x_input_name,
                k_in: lg.k_input_name,
                v_in: lg.v_input_name,
                x_out: lg.x_output_name,
                k_out: lg.k_output_name,
                v_out: lg.v_output_name,
            });
            layer_compiled.push(compiled);
        }
        step_cache.sequential_by_new_seq.insert(new_seq, CachedSequentialStep {
            embed_compiled,
            embed_x_output: embed.x_output_name,
            layer_compiled, layer_io,
            head_compiled,
            head_x_input: head.x_input_name,
            head_logits_output: head.logits_output_name,
            block_count: embed.config.block_count,
        });
    }
    let cs = step_cache.sequential_by_new_seq.get(&new_seq).expect("just inserted");

    let mut joules_total = 0.0;

    // Helper: dynamic KV pos input.
    let kv_pos_tensor = || Tensor {
        meta: jouleclaw_core::tensor::TensorMeta::new(jouleclaw_core::tensor::Dtype::I32, &[1]),
        storage: std::sync::Arc::new(jouleclaw_core::tensor::TensorStorage { bytes: (cached_seq as i32).to_le_bytes().to_vec(), mapped: None }),
    };

    // ── 1. Embed ──
    let mut embed_inputs = HashMap::new();
    embed_inputs.insert("token_ids".into(), new_tokens);
    let embed_res = execute(&cs.embed_compiled, embed_inputs, ExecutionOptions::default())?;
    joules_total += embed_res.trace.joule_accounting.total_joules;
    let mut x = embed_res.outputs.get(&cs.embed_x_output)
        .expect("embed x output").clone();

    // ── 2. Layers ──
    for layer in 0..cs.block_count {
        let (k_in, v_in) = cache.take_buffers(layer);
        let io = &cs.layer_io[layer];

        let mut layer_inputs = HashMap::new();
        layer_inputs.insert(io.x_in.clone(), x);
        layer_inputs.insert(io.k_in.clone(), k_in);
        layer_inputs.insert(io.v_in.clone(), v_in);
        layer_inputs.insert(KV_POS_INPUT.into(), kv_pos_tensor());

        let layer_res = execute(
            &cs.layer_compiled[layer], layer_inputs, ExecutionOptions::default())?;
        joules_total += layer_res.trace.joule_accounting.total_joules;

        let x_out = layer_res.outputs.get(&io.x_out).expect("layer x output").clone();
        let k_out = layer_res.outputs.get(&io.k_out).expect("layer k output").clone();
        let v_out = layer_res.outputs.get(&io.v_out).expect("layer v output").clone();

        cache.replace_buffers(layer, k_out, v_out);
        x = x_out;
    }

    // ── 3. Head ──
    let mut head_inputs = HashMap::new();
    head_inputs.insert(cs.head_x_input.clone(), x);
    let head_res = execute(&cs.head_compiled, head_inputs, ExecutionOptions::default())?;
    joules_total += head_res.trace.joule_accounting.total_joules;

    cache.advance(new_seq);

    Ok(InPlaceStepResult {
        logits: head_res.outputs.get(&cs.head_logits_output)
            .expect("logits").as_f32_vec(),
        joules: joules_total,
    })
}
