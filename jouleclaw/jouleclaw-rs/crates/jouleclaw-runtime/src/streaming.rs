//! Streaming generation API and multi-turn conversation support.
//!
//! The one-shot `generate()` returns the complete result as a single
//! string. That's fine for batch generation but no good for an interactive
//! chat where you want tokens to appear as they're generated.
//!
//! This module provides two interfaces:
//!
//! - `generate_stream()`: returns a `GenerateStream` iterator. Each call to
//!   `next()` runs one decode step, samples one token, and yields it.
//!   Iteration ends when EOS is sampled or `max_new_tokens` is reached.
//!
//! - `Conversation`: a stateful wrapper holding a persistent KV cache.
//!   Successive `extend()` calls add to the same conversation without
//!   reprocessing the previous turns — the cache carries that state.
//!
//! Both build on the in-place KV cache infrastructure from Phase 1.15.
//! Memory is constant throughout the conversation (bounded by `max_seq`),
//! and per-step attention work scales with conversation length, not
//! buffer size (the Slice optimization from Phase 1.17).
//!
//! ## Streaming example
//!
//! ```rust,ignore
//! let mut stream = generate_stream(&model, &vocab, "Tell me about", &cfg)?;
//! while let Some(token_result) = stream.next() {
//!     let tok = token_result?;
//!     print!("{}", tok.text);  // appear as they're generated
//!     if tok.is_eos { break; }
//! }
//! ```
//!
//! ## Multi-turn example
//!
//! ```rust,ignore
//! let mut conv = Conversation::new(&model, &vocab, 4096)?;
//! conv.extend("Hi, what's your name?", &cfg)?.for_each(|t| print!("{}", t.unwrap().text));
//! conv.extend("And how old are you?", &cfg)?.for_each(|t| print!("{}", t.unwrap().text));
//! // The cache holds context from both turns.
//! ```

use crate::generate::{
    resolve_tokenizer_kind, run_inplace_step_cached, tokens_to_tensor,
    DecodeStepCache, GenerateConfig, GenerateError, TokenizerKind,
};
use crate::Runtime;
use jouleclaw_loader_gguf::kv_cache_inplace::{
    InPlaceKvCache, KvSnapshot, ShortConvStateCache, ShortConvStateSnapshot,
};
use jouleclaw_loader_gguf::sample::sample_logits_with_history;
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::GgufModel;

/// One token yielded by `GenerateStream`.
#[derive(Debug, Clone)]
pub struct StreamedToken {
    /// The token's numeric ID.
    pub id: u32,
    /// The token's text, ready to print. Note that for BPE tokenizers a
    /// single token may be a partial UTF-8 byte sequence; the streaming
    /// caller should be tolerant of that. For SPM, each token is a
    /// complete code point or piece.
    pub text: String,
    /// True if this is the EOS token. Iteration ends after this is yielded
    /// (it's not actually yielded — EOS causes the stream to end).
    pub is_eos: bool,
    /// Position within the conversation. Useful for progress bars or
    /// computing remaining budget against `max_new_tokens`.
    pub position: usize,
}

/// Token-by-token streaming generation.
///
/// Each `next()` call advances the conversation by one token. The stream
/// borrows the model/vocab/runtime by reference; the lifetime ties it to
/// those resources.
pub struct GenerateStream<'a> {
    model: &'a GgufModel,
    vocab: &'a Vocab,
    runtime: Runtime,
    cache: InPlaceKvCache,
    /// LFM2 shortconv state (per recurrent layer). Empty for non-LFM2.
    shortconv: ShortConvStateCache,
    cfg: GenerateConfig,
    tokenizer_kind: TokenizerKind,
    next_token: Option<u32>,
    tokens_yielded: usize,
    finished: bool,
    /// Token history for repetition/frequency/presence penalty. Includes
    /// the prompt and everything generated so far.
    recent_tokens: Vec<u32>,
    /// Sum of `KernelResult.joules` across every `execute()` call
    /// during this stream (prefill + each decode step). This is the
    /// **measured** energy figure — what the calibration ledger sees
    /// as the "actual" side of the estimated-vs-actual ratio.
    cumulative_joules: f64,
    /// Compile-once cache for the constant-topology decode graph. The
    /// per-token build+compile happens at most once per distinct
    /// `new_seq` (prefill length, then 1 for every decode step).
    step_cache: DecodeStepCache,
}

impl<'a> GenerateStream<'a> {
    /// Sum of kernel-reported joules across every step run so far.
    pub fn cumulative_joules(&self) -> f64 { self.cumulative_joules }
}

impl<'a> GenerateStream<'a> {
    fn new(
        model: &'a GgufModel,
        vocab: &'a Vocab,
        prompt: &str,
        cfg: &GenerateConfig,
    ) -> Result<Self, GenerateError> {
        let tokenizer_kind = resolve_tokenizer_kind(vocab, cfg.tokenizer_kind);

        let prompt_tokens = match tokenizer_kind {
            TokenizerKind::Spm => vocab.encode_spm(prompt, cfg.add_bos),
            TokenizerKind::Bpe => vocab.encode_bpe_regex(prompt, cfg.add_bos),
            TokenizerKind::Auto => unreachable!(),
        };
        if prompt_tokens.is_empty() {
            return Err(GenerateError::EmptyPromptTokens);
        }

        let max_seq = cfg.max_seq.unwrap_or_else(|| {
            prompt_tokens.len() + cfg.max_new_tokens + 16
        });

        let runtime = Runtime::boot();
        let mut cache = InPlaceKvCache::for_model(model, max_seq)?;
        let mut shortconv = ShortConvStateCache::for_model(model)?;

        // Prefill: one constant-topology decode step over all prompt
        // tokens (compiled + cached under new_seq = prompt_len).
        let prompt_tensor = tokens_to_tensor(&prompt_tokens);
        let mut step_cache = DecodeStepCache::new();
        let prefill = run_inplace_step_cached(
            model, &runtime, &mut cache, &mut shortconv, &mut step_cache,
            prompt_tensor, prompt_tokens.len())?;

        // Sample first new token from the last prompt position.
        let vocab_size = vocab.len();
        let last_row = &prefill.logits[(prompt_tokens.len() - 1) * vocab_size..];
        let first_next = sample_logits_with_history(last_row, &cfg.sampling, &prompt_tokens);

        Ok(Self {
            model, vocab, runtime, cache, shortconv,
            cfg: cfg.clone(),
            tokenizer_kind,
            next_token: Some(first_next),
            tokens_yielded: 0,
            finished: false,
            recent_tokens: prompt_tokens,
            cumulative_joules: prefill.joules,
            step_cache,
        })
    }
}

impl<'a> Iterator for GenerateStream<'a> {
    type Item = Result<StreamedToken, GenerateError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished { return None; }
        if self.tokens_yielded >= self.cfg.max_new_tokens {
            self.finished = true;
            return None;
        }
        let id = self.next_token?;

        // EOS terminates the stream without yielding.
        if Some(id) == self.vocab.eos_id {
            self.finished = true;
            return None;
        }

        let text = match self.tokenizer_kind {
            TokenizerKind::Spm => self.vocab.decode_spm(&[id]),
            TokenizerKind::Bpe => self.vocab.decode_bpe(&[id]),
            TokenizerKind::Auto => unreachable!(),
        };
        let position = self.tokens_yielded;
        self.tokens_yielded += 1;
        // Append to history so the next sample sees this token.
        self.recent_tokens.push(id);

        // Decode one more step to set up the next iteration. Reuses
        // the cached const graph (compiled once for new_seq=1).
        let single = tokens_to_tensor(&[id]);
        let step_result = run_inplace_step_cached(
            self.model, &self.runtime, &mut self.cache,
            &mut self.shortconv, &mut self.step_cache, single, 1);
        match step_result {
            Ok(step) => {
                self.cumulative_joules += step.joules;
                self.next_token = Some(sample_logits_with_history(
                    &step.logits, &self.cfg.sampling, &self.recent_tokens));
            }
            Err(e) => {
                self.finished = true;
                return Some(Err(e));
            }
        }

        Some(Ok(StreamedToken {
            id, text,
            is_eos: false,
            position,
        }))
    }
}

/// Start a streaming generation. The returned iterator yields one
/// `StreamedToken` per call to `next()`.
pub fn generate_stream<'a>(
    model: &'a GgufModel,
    vocab: &'a Vocab,
    prompt: &str,
    cfg: &GenerateConfig,
) -> Result<GenerateStream<'a>, GenerateError> {
    GenerateStream::new(model, vocab, prompt, cfg)
}

/// A multi-turn conversation that reuses the KV cache across turns.
///
/// The cache persists model context between `extend()` calls. The second
/// turn builds on top of the first turn's KV state without reprocessing
/// the first turn's prompt. This is what makes interactive chat efficient.
pub struct Conversation<'a> {
    model: &'a GgufModel,
    vocab: &'a Vocab,
    runtime: Runtime,
    cache: InPlaceKvCache,
    /// LFM2 shortconv state (per recurrent layer). Empty for non-LFM2
    /// archs, so this is a near-zero-cost wrapper on attention-only
    /// models. For hybrid LFM2, this is the rolling-window state that
    /// makes streaming decode produce the same output as a fresh
    /// prefill.
    shortconv: ShortConvStateCache,
    /// Total tokens consumed (prompt + generated) across all turns.
    /// `cache.current_seq` mirrors this.
    pub total_tokens: usize,
    /// Maximum sequence length the cache was sized for.
    pub max_seq: usize,
    /// All tokens consumed (prompt + generated) across all turns. Used
    /// as `recent_tokens` history for the sampler's penalty terms.
    recent_tokens: Vec<u32>,
    /// Sum of `KernelResult.joules` across every `execute()` call in
    /// this conversation (every prefill + every decode step in every
    /// `extend`). This is the **measured** energy the calibration
    /// ledger sees on the cascade side.
    cumulative_joules: f64,
    /// Compile-once cache for the constant-topology decode graphs,
    /// persisted across `extend()` turns (a follow-up turn's decode
    /// steps reuse the `new_seq=1` graph compiled in the first turn).
    step_cache: DecodeStepCache,
}

impl<'a> Conversation<'a> {
    /// Create a new conversation with a fresh cache of the given size.
    pub fn new(
        model: &'a GgufModel,
        vocab: &'a Vocab,
        max_seq: usize,
    ) -> Result<Self, GenerateError> {
        Self::with_runtime(model, vocab, max_seq, Runtime::boot())
    }

    /// Like [`Self::new`] but with an explicit runtime — lets callers pick
    /// e.g. [`Runtime::reference_only`] when an accelerated backend has a
    /// known correctness gap (the AppleAmx f32 drift) that would
    /// contaminate a correctness oracle.
    pub fn with_runtime(
        model: &'a GgufModel,
        vocab: &'a Vocab,
        max_seq: usize,
        runtime: Runtime,
    ) -> Result<Self, GenerateError> {
        Self::with_runtime_and_quant(
            model, vocab, max_seq, runtime, jouleclaw_loader_gguf::kv_cache_inplace::KvQuant::None)
    }

    /// Like [`Self::with_runtime`] but with an explicit KV cache quant
    /// scheme. Use [`jouleclaw_loader_gguf::kv_cache_inplace::KvQuant::Int8`]
    /// to reduce cold KV-cache storage by ~4×; per the cache-side
    /// roundtrip test, the per-element quant error stays bounded by
    /// `max_abs(row) / 254` (~0.4% relative for typical attention K/V
    /// distributions).
    pub fn with_runtime_and_quant(
        model: &'a GgufModel,
        vocab: &'a Vocab,
        max_seq: usize,
        runtime: Runtime,
        kv_quant: jouleclaw_loader_gguf::kv_cache_inplace::KvQuant,
    ) -> Result<Self, GenerateError> {
        let cache = InPlaceKvCache::for_model_with_quant(model, max_seq, kv_quant)?;
        let shortconv = ShortConvStateCache::for_model(model)?;
        Ok(Self {
            model, vocab, runtime, cache, shortconv,
            total_tokens: 0,
            max_seq,
            recent_tokens: Vec::new(),
            cumulative_joules: 0.0,
            step_cache: DecodeStepCache::new(),
        })
    }

    /// Cold-storage byte footprint of the KV cache. Read by benchmarks
    /// comparing fp32 vs int8 cache configurations.
    pub fn kv_cache_bytes(&self) -> usize {
        self.cache.cache_bytes()
    }

    /// Capture this conversation's full state — KV cache, shortconv
    /// state, token history, position — into an immutable
    /// [`ConversationCheckpoint`]. The checkpoint can later be passed
    /// to [`Self::from_checkpoint`] to resume from that exact point
    /// without re-running prefill on the cached tokens.
    ///
    /// Used by [`PrefixCache`] to remember system-prompt + early-turn
    /// prefills and replay them when a new conversation shares the
    /// same prefix.
    pub fn checkpoint(&self) -> ConversationCheckpoint {
        ConversationCheckpoint {
            kv: self.cache.snapshot(),
            shortconv: self.shortconv.snapshot(),
            tokens: self.recent_tokens.clone(),
            cumulative_joules: self.cumulative_joules,
        }
    }

    /// Restore a conversation from a checkpoint. The new conversation
    /// holds independent (copied) KV / shortconv buffers, so further
    /// `extend` calls don't mutate the snapshot. `max_seq` is read from
    /// the snapshot; pass the same `runtime` you'd use otherwise.
    pub fn from_checkpoint(
        model: &'a GgufModel,
        vocab: &'a Vocab,
        runtime: Runtime,
        checkpoint: &ConversationCheckpoint,
    ) -> Result<Self, GenerateError> {
        let cache = InPlaceKvCache::from_snapshot(&checkpoint.kv);
        let shortconv = ShortConvStateCache::from_snapshot(&checkpoint.shortconv);
        Ok(Self {
            model, vocab, runtime, cache, shortconv,
            total_tokens: checkpoint.tokens.len(),
            max_seq: checkpoint.kv.max_seq,
            recent_tokens: checkpoint.tokens.clone(),
            cumulative_joules: checkpoint.cumulative_joules,
            step_cache: DecodeStepCache::new(),
        })
    }

    /// Cumulative kernel-reported joules across every prefill + decode
    /// step run by this conversation. Read by [`mrl::GgufTextEmbedder`]
    /// and `PrismTier::gguf_answer` to feed the cascade's calibration
    /// ledger an honest `actual` instead of a static estimate.
    pub fn cumulative_joules(&self) -> f64 { self.cumulative_joules }

    /// Extend the conversation with Prompt Lookup Decoding enabled.
    /// Behaves like [`extend_tokens`](Self::extend_tokens) but speculates
    /// up to `pld_cfg.max_lookahead` tokens per forward pass via n-gram
    /// matches in the conversation history.
    ///
    /// Returns a [`PldOutcome`] with the generated tokens, the total
    /// kernel-reported joules across prefill + every PLD forward, and a
    /// per-step acceptance histogram. The mean of `accepted_per_step` is
    /// the effective wall-clock speedup-per-pass — `1.0` means PLD
    /// landed no hits (no echo in this workload), higher means PLD won.
    pub fn extend_pld_tokens(
        &mut self,
        prompt_tokens: Vec<u32>,
        cfg: &GenerateConfig,
        pld_cfg: &crate::pld::PldConfig,
    ) -> Result<crate::pld::PldOutcome, GenerateError> {
        use crate::pld::{find_draft, PldOutcome};
        use jouleclaw_loader_gguf::sample::sample_logits_with_history;

        if prompt_tokens.is_empty() {
            return Err(GenerateError::EmptyPromptTokens);
        }
        // Worst-case overshoot during a tentative PLD forward.
        let head = self.total_tokens
            + prompt_tokens.len()
            + cfg.max_new_tokens
            + pld_cfg.max_lookahead;
        if head > self.max_seq {
            return Err(GenerateError::Load(
                jouleclaw_loader_gguf::llama::LoadError::UnsupportedArchitecture(format!(
                    "conversation overflow (PLD): have {} cached + {} prompt + {} max_new \
                     + {} pld lookahead > max_seq {}",
                    self.total_tokens, prompt_tokens.len(),
                    cfg.max_new_tokens, pld_cfg.max_lookahead, self.max_seq))));
        }

        // ---- Prefill ----
        let prompt_tensor = tokens_to_tensor(&prompt_tokens);
        let prefill = run_inplace_step_cached(
            self.model, &self.runtime, &mut self.cache,
            &mut self.shortconv, &mut self.step_cache,
            prompt_tensor, prompt_tokens.len())?;
        let mut joules_total = prefill.joules;
        self.cumulative_joules += prefill.joules;
        self.total_tokens += prompt_tokens.len();
        self.recent_tokens.extend(&prompt_tokens);

        let vocab_size = self.vocab.len();
        let last_row = &prefill.logits[(prompt_tokens.len() - 1) * vocab_size..];
        let mut next_token = sample_logits_with_history(
            last_row, &cfg.sampling, &self.recent_tokens);
        let eos = self.vocab.eos_id;

        let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let mut accepted_per_step: Vec<usize> = Vec::new();

        while generated.len() < cfg.max_new_tokens {
            if Some(next_token) == eos { break; }

            // Emit next_token as this step's first new token. Its KV
            // is not yet in the cache — it goes in as input position 0
            // of the upcoming forward.
            generated.push(next_token);
            self.total_tokens += 1;
            self.recent_tokens.push(next_token);
            if generated.len() >= cfg.max_new_tokens { break; }

            // Lookup a draft over the running history.
            let draft = find_draft(&self.recent_tokens, pld_cfg);

            // Forward inputs = [next_token, draft_1, ..., draft_K],
            // capped so we don't overshoot max_new_tokens.
            let mut forward_inputs: Vec<u32> = Vec::with_capacity(1 + draft.len());
            forward_inputs.push(next_token);
            forward_inputs.extend(&draft);
            let remaining = cfg.max_new_tokens - generated.len();
            let max_forward = (1 + remaining).min(forward_inputs.len());
            forward_inputs.truncate(max_forward);
            let n_rows = forward_inputs.len();

            // Single tentative forward — cache advances by n_rows.
            let pre_seq = self.cache.current_seq;
            let tensor = tokens_to_tensor(&forward_inputs);
            let step = run_inplace_step_cached(
                self.model, &self.runtime, &mut self.cache,
                &mut self.shortconv, &mut self.step_cache,
                tensor, n_rows)?;
            joules_total += step.joules;
            self.cumulative_joules += step.joules;

            // Sample one prediction per row. Row i = "what comes after
            // forward position i". Penalty history at row i must
            // include every token the model conditioned on through
            // position i — that's recent_tokens (which already has
            // next_token at position 0) PLUS forward_inputs[1..=i]
            // (positions 1 through i, inclusive). The earlier `[1..i]`
            // (exclusive) form skipped position i and produced subtly
            // different sampling than the equivalent single-token
            // decode would have.
            let mut sampled: Vec<u32> = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let row = &step.logits[i * vocab_size..(i + 1) * vocab_size];
                let tok = if i == 0 {
                    sample_logits_with_history(row, &cfg.sampling, &self.recent_tokens)
                } else {
                    let mut hist: Vec<u32> = self.recent_tokens.clone();
                    hist.extend(&forward_inputs[1..=i]);
                    sample_logits_with_history(row, &cfg.sampling, &hist)
                };
                sampled.push(tok);
            }

            // Acceptance: s_1 unconditionally accepted (it would also
            // be the no-PLD prediction). For i in 1..n_rows: the draft
            // was correct iff forward_inputs[i] == sampled[i-1].
            let mut accepted_count = 1usize;
            for i in 1..n_rows {
                if forward_inputs[i] == sampled[i - 1] {
                    accepted_count += 1;
                } else {
                    break;
                }
            }

            // Emit s_1 .. s_{accepted_count-1} into `generated`; the
            // last accepted sample becomes the next iteration's
            // `next_token`. Stop early on EOS or max_new_tokens.
            let mut early_exit: Option<usize> = None;
            for i in 1..accepted_count {
                if Some(sampled[i - 1]) == eos {
                    early_exit = Some(i);
                    break;
                }
                generated.push(sampled[i - 1]);
                self.total_tokens += 1;
                self.recent_tokens.push(sampled[i - 1]);
                if generated.len() >= cfg.max_new_tokens {
                    early_exit = Some(i);
                    break;
                }
            }
            if let Some(stop_i) = early_exit {
                self.cache.current_seq = pre_seq + stop_i;
                accepted_per_step.push(stop_i);
                return Ok(PldOutcome { tokens: generated, joules: joules_total, accepted_per_step });
            }

            // The last accepted sample seeds the next loop iteration.
            next_token = sampled[accepted_count - 1];

            // Rewind cache to drop the rejected drafts' stale K/V.
            self.cache.current_seq = pre_seq + accepted_count;
            accepted_per_step.push(accepted_count);
        }

        Ok(crate::pld::PldOutcome {
            tokens: generated,
            joules: joules_total,
            accepted_per_step,
        })
    }

    /// Reset to the empty state. The buffer's allocation is preserved
    /// (the bytes are zeroed) so the next turn starts fresh without
    /// reallocation.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.total_tokens = 0;
        self.recent_tokens.clear();
        self.cumulative_joules = 0.0;
    }

    /// Append `prompt` to the conversation, run generation, and return an
    /// iterator that yields one `StreamedToken` per call. The cache state
    /// is updated as tokens are produced; the iterator borrows the
    /// conversation mutably for its lifetime.
    ///
    /// IMPORTANT: don't add BOS to follow-up turns — the cache already
    /// contains the first turn's BOS context. The default `add_bos=true`
    /// is for fresh conversations; pass `add_bos: false` on subsequent
    /// turns.
    pub fn extend<'s>(
        &'s mut self,
        prompt: &str,
        cfg: &GenerateConfig,
    ) -> Result<ConversationStream<'s, 'a>, GenerateError> {
        let tokenizer_kind = resolve_tokenizer_kind(self.vocab, cfg.tokenizer_kind);

        let prompt_tokens = match tokenizer_kind {
            TokenizerKind::Spm => self.vocab.encode_spm(prompt, cfg.add_bos),
            TokenizerKind::Bpe => self.vocab.encode_bpe_regex(prompt, cfg.add_bos),
            TokenizerKind::Auto => unreachable!(),
        };
        self.extend_with_prompt_tokens(prompt_tokens, tokenizer_kind, cfg)
    }

    /// Like `extend`, but takes pre-tokenized prompt IDs. Use this when
    /// the prompt contains chat-template markers that need atomic
    /// special-token encoding (`encode_user_turn` / `encode_with_specials`)
    /// — plain BPE would shatter `<|im_start|>` etc. into byte subwords.
    pub fn extend_tokens<'s>(
        &'s mut self,
        prompt_tokens: Vec<u32>,
        cfg: &GenerateConfig,
    ) -> Result<ConversationStream<'s, 'a>, GenerateError> {
        let tokenizer_kind = resolve_tokenizer_kind(self.vocab, cfg.tokenizer_kind);
        self.extend_with_prompt_tokens(prompt_tokens, tokenizer_kind, cfg)
    }

    /// Prefill a long prompt in fixed-size chunks. Numerically
    /// equivalent to one monolithic `extend_tokens` call (the per-chunk
    /// in-place scatter + masked attention produces bit-identical
    /// argmax — see `chunked_prefill.rs`), but each forward pass only
    /// builds attention scratch for `chunk_size` queries instead of
    /// the full prompt. For an 8K-token prompt with `chunk_size=256`,
    /// peak attention working memory drops ~32×.
    ///
    /// Trade-off: per-chunk dispatch overhead means short prompts
    /// (< ~64 tokens) run slightly slower chunked than monolithic.
    /// Use chunked when the prompt is large enough that the
    /// monolithic attention scratch dominates memory.
    ///
    /// Returns a stream over the final chunk's logits; intermediate
    /// chunk streams are consumed and dropped internally.
    pub fn extend_tokens_chunked<'s>(
        &'s mut self,
        prompt_tokens: Vec<u32>,
        chunk_size: usize,
        cfg: &GenerateConfig,
    ) -> Result<ConversationStream<'s, 'a>, GenerateError> {
        if prompt_tokens.is_empty() {
            return Err(GenerateError::EmptyPromptTokens);
        }
        let chunk_size = chunk_size.max(1);
        let n = prompt_tokens.len();
        let tokenizer_kind = resolve_tokenizer_kind(self.vocab, cfg.tokenizer_kind);
        if n <= chunk_size {
            return self.extend_with_prompt_tokens(prompt_tokens, tokenizer_kind, cfg);
        }
        // Prefill every chunk except the last; drop the stream each
        // time (intermediate samples are unused).
        let mut i = 0usize;
        while i + chunk_size < n {
            let chunk = prompt_tokens[i..i + chunk_size].to_vec();
            let _ = self.extend_with_prompt_tokens(chunk, tokenizer_kind, cfg)?;
            i += chunk_size;
        }
        // Final chunk: return its stream so the caller can begin decoding.
        let last = prompt_tokens[i..].to_vec();
        self.extend_with_prompt_tokens(last, tokenizer_kind, cfg)
    }

    fn extend_with_prompt_tokens<'s>(
        &'s mut self,
        prompt_tokens: Vec<u32>,
        tokenizer_kind: TokenizerKind,
        cfg: &GenerateConfig,
    ) -> Result<ConversationStream<'s, 'a>, GenerateError> {
        if prompt_tokens.is_empty() {
            return Err(GenerateError::EmptyPromptTokens);
        }
        if self.total_tokens + prompt_tokens.len() + cfg.max_new_tokens > self.max_seq {
            // Honest error rather than a confusing scatter-bounds panic.
            return Err(GenerateError::Load(
                jouleclaw_loader_gguf::llama::LoadError::UnsupportedArchitecture(format!(
                    "conversation overflow: have {} cached + {} prompt + {} max_new \
                     > max_seq {}",
                    self.total_tokens, prompt_tokens.len(),
                    cfg.max_new_tokens, self.max_seq))));
        }

        // Prefill the new portion of the conversation (cached const
        // graph, keyed by this turn's prompt length).
        let prompt_tensor = tokens_to_tensor(&prompt_tokens);
        let prefill = run_inplace_step_cached(
            self.model, &self.runtime, &mut self.cache,
            &mut self.shortconv, &mut self.step_cache,
            prompt_tensor, prompt_tokens.len())?;
        self.cumulative_joules += prefill.joules;
        self.total_tokens += prompt_tokens.len();
        // Append the new prompt tokens to the running history.
        self.recent_tokens.extend(&prompt_tokens);

        let vocab_size = self.vocab.len();
        let last_row = &prefill.logits[(prompt_tokens.len() - 1) * vocab_size..];
        let first_next = sample_logits_with_history(
            last_row, &cfg.sampling, &self.recent_tokens);

        Ok(ConversationStream {
            conv: self,
            cfg: cfg.clone(),
            tokenizer_kind,
            next_token: Some(first_next),
            tokens_yielded: 0,
            finished: false,
        })
    }
}

/// Trained-drafter speculative decoding: a smaller paired model
/// (`drafter`) proposes K tokens autoregressively; the bigger model
/// (`target`) verifies them in one forward pass of K+1 positions.
/// Accept the longest matching prefix; rewind both caches to the
/// acceptance boundary; continue. Same accept/reject math as PLD
/// but the draft comes from a learned distribution instead of an
/// n-gram lookup.
///
/// `target` and `drafter` must be fresh (no prior turns committed)
/// and share a tokenizer — same model family is the typical
/// guarantee (Bonsai 1.7B as drafter for Bonsai 4B / 8B; Gemma 4
/// ships dedicated drafters for the same reason).
///
/// Returns the generated tokens plus per-model joule receipts and
/// the per-step acceptance histogram. `mean_acceptance × cost_ratio
/// = wall-clock speedup` against single-token target decode; see
/// `drafter.rs` for the full math.
pub fn extend_with_drafter(
    target: &mut Conversation,
    drafter: &mut Conversation,
    prompt_tokens: Vec<u32>,
    cfg: &GenerateConfig,
    spec_cfg: &crate::drafter::DrafterConfig,
) -> Result<crate::drafter::DrafterOutcome, GenerateError> {
    use crate::drafter::DrafterOutcome;
    use jouleclaw_loader_gguf::sample::sample_logits_with_history;

    if prompt_tokens.is_empty() {
        return Err(GenerateError::EmptyPromptTokens);
    }
    let k = spec_cfg.max_lookahead;
    if k == 0 {
        return Err(GenerateError::Load(
            jouleclaw_loader_gguf::llama::LoadError::UnsupportedArchitecture(
                "DrafterConfig::max_lookahead must be >= 1".into())));
    }

    // Worst-case overshoot during verify (cache holds K+1 transient
    // positions before rewind). Reserve room on both caches.
    let target_head = target.total_tokens + prompt_tokens.len()
        + cfg.max_new_tokens + k;
    if target_head > target.max_seq {
        return Err(GenerateError::Load(
            jouleclaw_loader_gguf::llama::LoadError::UnsupportedArchitecture(format!(
                "target conversation overflow (drafter): have {} + {} prompt + {} max_new \
                 + {} drafter lookahead > max_seq {}",
                target.total_tokens, prompt_tokens.len(),
                cfg.max_new_tokens, k, target.max_seq))));
    }
    let drafter_head = drafter.total_tokens + prompt_tokens.len()
        + cfg.max_new_tokens + k;
    if drafter_head > drafter.max_seq {
        return Err(GenerateError::Load(
            jouleclaw_loader_gguf::llama::LoadError::UnsupportedArchitecture(format!(
                "drafter conversation overflow: have {} + {} prompt + {} max_new \
                 + {} lookahead > max_seq {}",
                drafter.total_tokens, prompt_tokens.len(),
                cfg.max_new_tokens, k, drafter.max_seq))));
    }

    // ── Prefill both models with the same prompt ──
    let prompt_tensor = tokens_to_tensor(&prompt_tokens);
    let target_prefill = run_inplace_step_cached(
        target.model, &target.runtime, &mut target.cache,
        &mut target.shortconv, &mut target.step_cache,
        prompt_tensor.clone(), prompt_tokens.len())?;
    let mut target_joules = target_prefill.joules;
    target.cumulative_joules += target_prefill.joules;
    target.total_tokens += prompt_tokens.len();
    target.recent_tokens.extend(&prompt_tokens);

    let drafter_prefill = run_inplace_step_cached(
        drafter.model, &drafter.runtime, &mut drafter.cache,
        &mut drafter.shortconv, &mut drafter.step_cache,
        prompt_tensor, prompt_tokens.len())?;
    let mut drafter_joules = drafter_prefill.joules;
    drafter.cumulative_joules += drafter_prefill.joules;
    drafter.total_tokens += prompt_tokens.len();
    drafter.recent_tokens.extend(&prompt_tokens);

    let vocab_size = target.vocab.len();
    let last_row = &target_prefill.logits[(prompt_tokens.len() - 1) * vocab_size..];
    let mut next_token = sample_logits_with_history(
        last_row, &cfg.sampling, &target.recent_tokens);

    let eos = target.vocab.eos_id;

    let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
    let mut accepted_per_step: Vec<usize> = Vec::new();

    while generated.len() < cfg.max_new_tokens {
        if Some(next_token) == eos { break; }

        // Emit next_token. Push to both models' recent_tokens so the
        // sampler penalties stay in sync.
        generated.push(next_token);
        target.total_tokens += 1;
        target.recent_tokens.push(next_token);
        drafter.total_tokens += 1;
        drafter.recent_tokens.push(next_token);
        if generated.len() >= cfg.max_new_tokens { break; }

        // ── 1. Drafter generates K candidate tokens ──
        let pre_drafter_seq = drafter.cache.current_seq;
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut drafter_token = next_token;
        for _ in 0..k {
            let single = tokens_to_tensor(&[drafter_token]);
            let step = run_inplace_step_cached(
                drafter.model, &drafter.runtime,
                &mut drafter.cache, &mut drafter.shortconv,
                &mut drafter.step_cache, single, 1)?;
            drafter_joules += step.joules;
            drafter.cumulative_joules += step.joules;
            // Sample drafter's next prediction. Hist = drafter's
            // committed history + drafts already proposed this step.
            // We DON'T push to drafter.recent_tokens during this loop
            // because the outer code already pushed `next_token`, and
            // pushing again here would double-count. The accepted
            // drafts get pushed below in the emit loop.
            let mut hist: Vec<u32> = drafter.recent_tokens.clone();
            hist.extend(&drafts);
            let next_draft = sample_logits_with_history(
                &step.logits, &cfg.sampling, &hist);
            drafter_token = next_draft;
            drafts.push(next_draft);
        }
        // drafter.cache.current_seq is now pre_drafter_seq + k.

        // ── 2. Target verifies [next_token, d_1, ..., d_K] in one forward ──
        let mut forward_inputs: Vec<u32> = Vec::with_capacity(1 + k);
        forward_inputs.push(next_token);
        forward_inputs.extend(&drafts);
        let remaining = cfg.max_new_tokens - generated.len();
        let max_forward = (1 + remaining).min(forward_inputs.len());
        forward_inputs.truncate(max_forward);
        let n_rows = forward_inputs.len();

        let pre_target_seq = target.cache.current_seq;
        let tensor = tokens_to_tensor(&forward_inputs);
        let target_step = run_inplace_step_cached(
            target.model, &target.runtime,
            &mut target.cache, &mut target.shortconv,
            &mut target.step_cache, tensor, n_rows)?;
        target_joules += target_step.joules;
        target.cumulative_joules += target_step.joules;

        // ── 3. Sample target's prediction at each row ──
        //
        // Row i predicts what comes AFTER position i (= forward_inputs[i]).
        // For sampler-penalty alignment with the drafter, hist at row i
        // must include forward_inputs[1..=i] (inclusive) — every token
        // the model conditioned on, including position i itself. Using
        // `[1..i]` (exclusive) is wrong: at iter i the drafter's hist
        // already contains drafts[0..i] = d_1..d_i, so the target must
        // match. Off-by-one here breaks self-drafting acceptance.
        let mut sampled: Vec<u32> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row = &target_step.logits[i * vocab_size..(i + 1) * vocab_size];
            let tok = if i == 0 {
                sample_logits_with_history(row, &cfg.sampling, &target.recent_tokens)
            } else {
                let mut hist: Vec<u32> = target.recent_tokens.clone();
                hist.extend(&forward_inputs[1..=i]);
                sample_logits_with_history(row, &cfg.sampling, &hist)
            };
            sampled.push(tok);
        }

        // ── 4. Acceptance ──
        let mut accepted_count = 1usize;
        for i in 1..n_rows {
            if forward_inputs[i] == sampled[i - 1] {
                accepted_count += 1;
            } else {
                break;
            }
        }

        // ── 5. Emit s_1..s_{accepted_count-1}, set next_token = s_{accepted_count} ──
        let mut early_exit: Option<usize> = None;
        for i in 1..accepted_count {
            if Some(sampled[i - 1]) == eos {
                early_exit = Some(i);
                break;
            }
            generated.push(sampled[i - 1]);
            target.total_tokens += 1;
            target.recent_tokens.push(sampled[i - 1]);
            drafter.total_tokens += 1;
            drafter.recent_tokens.push(sampled[i - 1]);
            if generated.len() >= cfg.max_new_tokens {
                early_exit = Some(i);
                break;
            }
        }
        if let Some(stop_i) = early_exit {
            target.cache.current_seq = pre_target_seq + stop_i;
            drafter.cache.current_seq = pre_drafter_seq + stop_i;
            accepted_per_step.push(stop_i);
            return Ok(DrafterOutcome {
                tokens: generated,
                target_joules, drafter_joules,
                accepted_per_step,
            });
        }
        next_token = sampled[accepted_count - 1];

        // ── 6. Rewind both caches to the accepted boundary ──
        target.cache.current_seq = pre_target_seq + accepted_count;
        drafter.cache.current_seq = pre_drafter_seq + accepted_count;

        accepted_per_step.push(accepted_count);
    }

    Ok(DrafterOutcome {
        tokens: generated,
        target_joules, drafter_joules,
        accepted_per_step,
    })
}

/// Streaming iterator yielded by `Conversation::extend()`.
pub struct ConversationStream<'s, 'a: 's> {
    conv: &'s mut Conversation<'a>,
    cfg: GenerateConfig,
    tokenizer_kind: TokenizerKind,
    next_token: Option<u32>,
    tokens_yielded: usize,
    finished: bool,
}

impl<'s, 'a> ConversationStream<'s, 'a> {
    /// The next token sampled from the last prefill position, without
    /// yet consuming a stream step. Useful for one-shot oracles that
    /// just want the argmax after prefill.
    pub fn peek_next_token(&self) -> Option<u32> { self.next_token }
}

impl<'s, 'a> Iterator for ConversationStream<'s, 'a> {
    type Item = Result<StreamedToken, GenerateError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished { return None; }
        if self.tokens_yielded >= self.cfg.max_new_tokens {
            self.finished = true;
            return None;
        }
        let id = self.next_token?;

        if Some(id) == self.conv.vocab.eos_id {
            self.finished = true;
            return None;
        }

        let text = match self.tokenizer_kind {
            TokenizerKind::Spm => self.conv.vocab.decode_spm(&[id]),
            TokenizerKind::Bpe => self.conv.vocab.decode_bpe(&[id]),
            TokenizerKind::Auto => unreachable!(),
        };
        let position = self.conv.total_tokens;
        self.tokens_yielded += 1;
        self.conv.total_tokens += 1;
        self.conv.recent_tokens.push(id);

        let single = tokens_to_tensor(&[id]);
        let step_result = run_inplace_step_cached(
            self.conv.model, &self.conv.runtime, &mut self.conv.cache,
            &mut self.conv.shortconv, &mut self.conv.step_cache, single, 1);
        match step_result {
            Ok(step) => {
                self.conv.cumulative_joules += step.joules;
                self.next_token = Some(sample_logits_with_history(
                    &step.logits, &self.cfg.sampling, &self.conv.recent_tokens));
            }
            Err(e) => {
                self.finished = true;
                return Some(Err(e));
            }
        }

        Some(Ok(StreamedToken {
            id, text,
            is_eos: false,
            position,
        }))
    }
}

// ──────────────────────────────────────────────────────────────────────
// Prefix cache: replay previously-prefilled prompts across requests.

/// Immutable snapshot of a [`Conversation`]'s full state at the point
/// of capture. Holds the KV cache, shortconv state, and token history
/// — everything needed to resume identically. Stored inside
/// [`PrefixCache`] entries; cheaply `Clone`-able internally
/// (`Vec<u8>` copies aren't free but are dwarfed by re-prefill cost).
///
/// Serializable for on-disk persistence — see
/// [`PrefixCache::save_to_file`] / [`PrefixCache::load_from_file`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversationCheckpoint {
    pub kv: KvSnapshot,
    pub shortconv: ShortConvStateSnapshot,
    /// The exact token sequence prefilled to produce this checkpoint.
    /// PrefixCache uses this to verify a new request actually starts
    /// with these tokens before reusing the snapshot.
    pub tokens: Vec<u32>,
    /// Cumulative kernel-reported joules through the checkpointed
    /// turn. Restored conversations carry this forward so the cascade
    /// calibration ledger sees a continuous total.
    pub cumulative_joules: f64,
}

impl ConversationCheckpoint {
    /// Resident-bytes accounting — used by [`PrefixCache`] eviction.
    pub fn bytes(&self) -> usize {
        self.kv.bytes()
            + self.shortconv.states.iter()
                .map(|s| s.as_ref().map(|b| b.len()).unwrap_or(0)).sum::<usize>()
            + self.tokens.len() * 4
    }

    /// Length of the cached prefix in tokens.
    pub fn prefix_len(&self) -> usize { self.tokens.len() }
}

/// LRU-bounded cache of [`ConversationCheckpoint`]s, keyed by token
/// prefix. The first turn of every conversation hits this cache: if
/// some previous request prefilled an exact prefix of the incoming
/// prompt, we restore from the checkpoint and only prefill the
/// suffix — saving the full prefill cost on the cached portion.
///
/// Use case fit: shared system prompts across users, agent loops that
/// reuse the same scaffolding, multi-turn chats where the early turns
/// rarely change. The win is proportional to (cached_tokens /
/// total_tokens) of a typical request.
///
/// Insertion is explicit (the caller decides what's worth caching).
/// Lookup picks the LONGEST exact-prefix match. Eviction is LRU,
/// bounded by total resident bytes.
pub struct PrefixCache {
    entries: Vec<PrefixEntry>,
    max_total_bytes: usize,
    current_bytes: usize,
    /// Monotonic counter for LRU ordering — newer entries have higher
    /// `last_used`. Cheaper than wall-clock and bit-reproducible.
    tick: u64,
}

struct PrefixEntry {
    checkpoint: ConversationCheckpoint,
    last_used: u64,
}

impl PrefixCache {
    /// Empty cache with a resident-bytes budget. A 100 MB budget fits
    /// ~1-3 Bonsai-class checkpoints depending on prefix length.
    pub fn new(max_total_bytes: usize) -> Self {
        Self { entries: Vec::new(), max_total_bytes, current_bytes: 0, tick: 0 }
    }

    /// Look up the longest cached checkpoint whose `tokens` are a
    /// prefix of `query`. Returns the checkpoint and the matched
    /// length. Updates LRU ordering on hit.
    pub fn lookup(&mut self, query: &[u32]) -> Option<(&ConversationCheckpoint, usize)> {
        self.tick += 1;
        let mut best_idx: Option<usize> = None;
        let mut best_len = 0usize;
        for (i, entry) in self.entries.iter().enumerate() {
            let n = entry.checkpoint.tokens.len();
            if n > query.len() { continue; }
            if entry.checkpoint.tokens.as_slice() == &query[..n] && n > best_len {
                best_len = n;
                best_idx = Some(i);
            }
        }
        let idx = best_idx?;
        self.entries[idx].last_used = self.tick;
        Some((&self.entries[idx].checkpoint, best_len))
    }

    /// Insert a checkpoint. If the same token sequence is already
    /// cached, this replaces it. Evicts LRU entries until total
    /// resident bytes fit the budget.
    pub fn insert(&mut self, checkpoint: ConversationCheckpoint) {
        self.tick += 1;
        // Replace any exact-tokens duplicate.
        if let Some(i) = self.entries.iter().position(|e|
            e.checkpoint.tokens == checkpoint.tokens
        ) {
            self.current_bytes -= self.entries[i].checkpoint.bytes();
            self.current_bytes += checkpoint.bytes();
            self.entries[i] = PrefixEntry { checkpoint, last_used: self.tick };
            self.evict_to_budget();
            return;
        }
        self.current_bytes += checkpoint.bytes();
        self.entries.push(PrefixEntry { checkpoint, last_used: self.tick });
        self.evict_to_budget();
    }

    fn evict_to_budget(&mut self) {
        while self.current_bytes > self.max_total_bytes && !self.entries.is_empty() {
            let oldest = self.entries.iter().enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i).unwrap();
            self.current_bytes -= self.entries[oldest].checkpoint.bytes();
            self.entries.remove(oldest);
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn current_bytes(&self) -> usize { self.current_bytes }
    pub fn max_total_bytes(&self) -> usize { self.max_total_bytes }

    /// Persist every cached checkpoint to a single bincode file.
    /// Atomic-ish: writes to `<path>.tmp` first, then renames, so a
    /// crash mid-write leaves the previous file intact.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let payload: Vec<ConversationCheckpoint> = self.entries.iter()
            .map(|e| e.checkpoint.clone()).collect();
        let bytes = bincode::serialize(&payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load checkpoints previously written by [`Self::save_to_file`].
    /// The returned cache uses `max_total_bytes`; entries that don't
    /// fit are dropped by the standard LRU eviction (file order is
    /// preserved, with each insert bumping the LRU tick).
    pub fn load_from_file(
        path: &std::path::Path,
        max_total_bytes: usize,
    ) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let checkpoints: Vec<ConversationCheckpoint> =
            bincode::deserialize(&bytes).map_err(|e|
                std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut cache = Self::new(max_total_bytes);
        for cp in checkpoints {
            cache.insert(cp);
        }
        Ok(cache)
    }
}
