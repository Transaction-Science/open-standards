//! `PrismTier` — joule cascade tier for 1-bit / ternary models.
//!
//! R28.0 shipped the kernel; R28.1 ships the end-to-end inference path.
//! A `PrismTier` constructed with [`PrismTier::from_decoder`] runs a
//! complete decoder-only transformer forward pass through ternary
//! weights (see [`crate::forward::TernaryDecoder`]) and answers Text
//! queries. With no decoder loaded the tier preserves R28.0 behavior:
//! refuses with a structured "R28.1 not loaded" reason so the cascade
//! sees the tier and routes around it cleanly.
//!
//! The coordinate is:
//!
//!   Z = Z2_3   (zone — small-to-medium statistical inference)
//!   E = Reactive
//!   T = L1_Measure   (well above Landauer, measurable per-op)
//!   I = Tokens
//!   V = Statistical  (single samples may be wrong; the distribution is the claim)
//!   R = Facts        (parameters encode propositions)
//!   P = { Sample, MlpForward }
//!
//! What's still ahead (R28.1.1+):
//!
//! - Real BitNet b1.58 / Bonsai weights from disk (currently uses
//!   [`crate::model::synthetic_model`] random-weight checkpoints — the
//!   pipeline is live, the answers are noise).
//! - KV cache (forward pass is `O(seq²)` instead of `O(seq)`).
//! - SentencePiece tokenizer (current byte-level vocab is a stand-in).
//! - Calibrated `confidence_floor` from a held-out eval.

use std::path::Path;
use std::time::Duration;

use jouleclaw_cascade::*;
use jouleclaw_loader_gguf::llama::LlamaConfig;
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::GgufModel;

use crate::bit::BitMatrix;
use crate::forward::TernaryDecoder;
use crate::ternary::TernaryMatrix;

/// GGUF-backed dispatch path for `PrismTier` — the production
/// integration point. A real PrismML model (Q2_0 g128 ternary, qwen3
/// arch) is parsed once at load; each `try_answer` builds a fresh
/// `Conversation` over the in-place KV cache and decodes greedy text.
/// The ternary kernels (`MatMulTernary`, `LookupTernary`) carry the
/// thesis: weights stay packed, the inner product is sign-select +
/// accumulate, energy is metered honestly. This is the "model as
/// peripheral, runtime as AI" surface the joule cascade dispatches.
pub struct GgufBackend {
    pub model: GgufModel,
    pub vocab: Vocab,
    pub config: LlamaConfig,
    pub max_seq: usize,
    /// Cached `joules_per_token` from the cost model below, computed
    /// once at load so `estimate_cost` is O(1).
    joules_per_token: f64,
    /// Auto-detected chat template (None for base models with no
    /// `tokenizer.chat_template` metadata). When set, `gguf_answer`
    /// wraps the user's query in the template before tokenizing so
    /// instruction-tuned models hit their trained input distribution
    /// instead of completion-style noise.
    chat_template: Option<jouleclaw_runtime::ChatTemplate>,
}

/// Weight format of a prism-quantized layer.
#[derive(Debug)]
pub enum PrismLayer {
    /// {-1, 0, +1} weights.
    Ternary(TernaryMatrix),
    /// {-1, +1} weights.
    Bit(BitMatrix),
}

impl PrismLayer {
    pub fn rows(&self) -> usize {
        match self {
            Self::Ternary(m) => m.rows(),
            Self::Bit(m) => m.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Ternary(m) => m.cols(),
            Self::Bit(m) => m.cols(),
        }
    }

    /// Static joule cost estimate for one forward pass of this layer.
    pub fn matvec_joules(&self) -> f64 {
        match self {
            Self::Ternary(m) => m.matvec_joules(),
            Self::Bit(m) => m.matvec_joules(),
        }
    }

    /// y = W * x.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) -> Result<(), String> {
        match self {
            Self::Ternary(m) => m.matvec(x, y).map_err(|e| e.to_string()),
            Self::Bit(m) => m.matvec(x, y).map_err(|e| e.to_string()),
        }
    }
}

/// Cascade tier wrapping a prism-quantized model. Sits at Z2_3·Statistical.
///
/// Dispatch priority (highest first): `gguf` → `decoder` → refuse.
/// The GGUF path is the real-model integration; the standalone
/// decoder path is the pre-GGUF synthetic-model surface kept for
/// existing tests.
pub struct PrismTier {
    /// Standalone PrismLayer stack — predates the decoder integration
    /// (R28.0 surface). Useful for register-and-refuse tests and the
    /// pre-R28.1 cost-estimation paths.
    pub layers: Vec<PrismLayer>,
    /// Full decoder-only transformer with ternary weights. When present
    /// (and no GGUF backend), `try_answer` tokenizes the query, runs
    /// `decoder.generate_greedy`, and returns the decoded text.
    pub decoder: Option<TernaryDecoder>,
    /// GGUF-backed dispatch — the real production surface. When
    /// `Some`, `try_answer` runs through the jouleclaw-runtime streaming
    /// API (in-place KV cache + ternary kernels).
    pub gguf: Option<GgufBackend>,
    /// Number of tokens to generate beyond the prompt. Default 16.
    pub max_new_tokens: usize,
    /// When `Some`, the GGUF dispatch path uses Prompt Lookup Decoding
    /// to speculatively decode multiple tokens per forward pass via
    /// prompt n-gram matches. ~1.3-1.6× wall-clock speedup on
    /// echo-heavy outputs (RAG, "rephrase this", agentic replay); no
    /// effect on purely novel generation beyond a single extra
    /// empty-draft forward per step.
    pub pld: Option<jouleclaw_runtime::PldConfig>,
    /// KV cache storage precision. `None` (fp32, the default) keeps
    /// the maximum-fidelity decode path. `Int8` reduces cold-storage
    /// memory by ~4× at the cost of a sub-millisecond per-step
    /// quant/dequant cycle — useful for long contexts or multi-model
    /// servers where idle cache memory is the constraint.
    pub kv_quant: jouleclaw_loader_gguf::kv_cache_inplace::KvQuant,
    /// Trained-drafter speculative decoding. When `Some`, the GGUF
    /// dispatch builds two Conversations (target = this tier's GGUF,
    /// drafter = the second GGUF) and routes through
    /// `jouleclaw_runtime::extend_with_drafter`. Mutually exclusive with
    /// `pld` — drafter wins if both are set (it's strictly more
    /// general).
    ///
    /// Pre-condition: target and drafter must share a tokenizer. The
    /// Bonsai family (1.7B / 4B / 8B, all qwen3 arch) qualifies. The
    /// builder validates vocab size as a sanity check.
    pub drafter: Option<GgufBackend>,
    /// Companion config for `drafter`. Default K=4 (Leviathan et al.
    /// sweet spot). Ignored when `drafter` is `None`.
    pub drafter_cfg: jouleclaw_runtime::DrafterConfig,
    /// Stable model ID surfaced as `TierId::L3(L3ModelId(model_id))`.
    pub model_id: u32,
}

impl PrismTier {
    /// Empty tier — declares the cell on the coordinate space but refuses
    /// all queries. Useful for exercising router calibration before the
    /// real model lands.
    pub fn empty(model_id: u32) -> Self {
        Self {
            layers: Vec::new(),
            decoder: None,
            gguf: None,
            max_new_tokens: 16,
            pld: None,
            kv_quant: jouleclaw_loader_gguf::kv_cache_inplace::KvQuant::None,
            drafter: None,
            drafter_cfg: jouleclaw_runtime::DrafterConfig { max_lookahead: 4 },
            model_id,
        }
    }

    /// Build a PrismTier from a stack of layers. Retained for backward
    /// compatibility with R28.0 callers; prefer `from_decoder` for R28.1.
    pub fn from_layers(model_id: u32, layers: Vec<PrismLayer>) -> Self {
        Self {
            layers,
            decoder: None,
            gguf: None,
            max_new_tokens: 16,
            pld: None,
            kv_quant: jouleclaw_loader_gguf::kv_cache_inplace::KvQuant::None,
            drafter: None,
            drafter_cfg: jouleclaw_runtime::DrafterConfig { max_lookahead: 4 },
            model_id,
        }
    }

    /// Build a PrismTier backed by a full ternary decoder. Queries on
    /// `Text` input will run through the decoder and return generated text.
    pub fn from_decoder(model_id: u32, decoder: TernaryDecoder) -> Self {
        Self {
            layers: Vec::new(),
            decoder: Some(decoder),
            gguf: None,
            max_new_tokens: 16,
            pld: None,
            kv_quant: jouleclaw_loader_gguf::kv_cache_inplace::KvQuant::None,
            drafter: None,
            drafter_cfg: jouleclaw_runtime::DrafterConfig { max_lookahead: 4 },
            model_id,
        }
    }

    /// Build a PrismTier backed by a real PrismML GGUF model on disk
    /// (`Ternary-Bonsai-*-Q2_0.gguf` and friends). The model is parsed
    /// once; subsequent `try_answer` calls run the decode/streaming
    /// path with the ternary kernels and report the per-token joule
    /// cost from the static cost model below.
    pub fn from_gguf<P: AsRef<Path>>(
        model_id: u32,
        path: P,
        max_seq: usize,
    ) -> Result<Self, GgufTierError> {
        let model = jouleclaw_loader_gguf::read_gguf_file(path.as_ref())
            .map_err(GgufTierError::Parse)?;
        let vocab = Vocab::from_gguf(&model)
            .map_err(|_| GgufTierError::Vocab)?;
        let config = LlamaConfig::from_metadata(&model)
            .map_err(GgufTierError::Config)?;
        let joules_per_token = ternary_joules_per_token(&config);
        let chat_template = jouleclaw_runtime::ChatTemplate::detect_from_model(&model);
        Ok(Self {
            layers: Vec::new(),
            decoder: None,
            gguf: Some(GgufBackend {
                model, vocab, config, max_seq, joules_per_token,
                chat_template,
            }),
            max_new_tokens: 16,
            pld: None,
            kv_quant: jouleclaw_loader_gguf::kv_cache_inplace::KvQuant::None,
            drafter: None,
            drafter_cfg: jouleclaw_runtime::DrafterConfig { max_lookahead: 4 },
            model_id,
        })
    }

    /// How many tokens of output to generate beyond the prompt.
    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// Enable Prompt Lookup Decoding on the GGUF dispatch path.
    /// `cfg = PldConfig::default()` is the standard 3-gram match /
    /// 3-token lookahead — the original PLD paper's recipe. Bigger
    /// lookahead trades cache rewind cost for higher peak speedup;
    /// shorter ngram trades precision for hit rate.
    pub fn with_pld(mut self, cfg: jouleclaw_runtime::PldConfig) -> Self {
        self.pld = Some(cfg);
        self
    }

    /// Load a smaller paired model to act as a drafter for trained
    /// speculative decoding. The target (this tier's GGUF) verifies
    /// `K+1` positions in one forward pass per step, accepting the
    /// longest matching prefix; the drafter generates K candidates
    /// autoregressively in K cheap forwards. Per the
    /// `bonsai_17b_drafting_bonsai_4b` integration test:
    ///
    ///   - Bonsai-1.7B as drafter for Bonsai-4B target
    ///   - Same qwen3 tokenizer family (validated at load time)
    ///   - 1.5-2× wall-clock speedup on longer generations
    ///
    /// Pre-conditions:
    ///   - This tier must already have a GGUF target loaded via
    ///     `from_gguf` first.
    ///   - The drafter's tokenizer must match (same vocab size is the
    ///     load-time check; full tokenizer equivalence is the caller's
    ///     responsibility).
    ///
    /// Returns the tier with the drafter wired. Errors on GGUF parse
    /// failure or vocab size mismatch.
    pub fn with_drafter<P: AsRef<Path>>(
        mut self,
        drafter_path: P,
        drafter_max_seq: usize,
    ) -> Result<Self, GgufTierError> {
        let target = self.gguf.as_ref().ok_or_else(|| {
            GgufTierError::Vocab // closest existing error; mean "no target loaded"
        })?;
        let target_vocab_len = target.vocab.len();

        let model = jouleclaw_loader_gguf::read_gguf_file(drafter_path.as_ref())
            .map_err(GgufTierError::Parse)?;
        let vocab = Vocab::from_gguf(&model)
            .map_err(|_| GgufTierError::Vocab)?;
        if vocab.len() != target_vocab_len {
            return Err(GgufTierError::Vocab);
        }
        let config = LlamaConfig::from_metadata(&model)
            .map_err(GgufTierError::Config)?;
        let joules_per_token = ternary_joules_per_token(&config);
        let chat_template = jouleclaw_runtime::ChatTemplate::detect_from_model(&model);

        self.drafter = Some(GgufBackend {
            model, vocab, config, max_seq: drafter_max_seq,
            joules_per_token, chat_template,
        });
        Ok(self)
    }

    /// Set the drafter lookahead K (number of tokens drafted per
    /// verify pass). Default 4. Ignored if `drafter` is not set.
    pub fn with_drafter_lookahead(mut self, k: usize) -> Self {
        self.drafter_cfg = jouleclaw_runtime::DrafterConfig { max_lookahead: k };
        self
    }

    /// Select the KV cache storage precision. Default is fp32.
    ///
    /// `KvQuant::Int8` is a deliberate **memory-for-latency** trade.
    /// Internally it routes through the sequential decode executor
    /// (`run_inplace_step_sequential`) so only one layer's fp32 K/V
    /// is alive in working memory at a time — necessary for the
    /// memory savings to actually materialize.
    ///
    /// - **Memory wins**: ~3.88× cold-storage savings on the cache
    ///   itself (28 MB → 7.2 MB at Bonsai-1.7B max_seq=128;
    ///   448 MB → 116 MB at max_seq=2048). Total RAM at peak,
    ///   counting transient working buffers:
    ///     fp32 mode at max_seq=128:        12 MB
    ///     int8 sequential at max_seq=128:   5.1 MB (cache + 1 layer's fp32)
    ///     fp32 mode at max_seq=2048:      448 MB
    ///     int8 sequential at max_seq=2048: ~120 MB
    /// - **Latency costs**: 50-70% wall-clock overhead. The sequential
    ///   executor runs n_layers + 2 separate execute() calls per decode
    ///   step (one per layer + embed + head) instead of one monolithic
    ///   call. Per-call dispatch overhead accumulates. Bonsai-1.7B
    ///   measured: 4.0-4.7 s fp32 vs 5.9-6.5 s int8 sequential.
    /// - **Correctness**: bit-identical (same per-layer arithmetic) to
    ///   fp32 in the sequential path itself; verified by the
    ///   `sequential_parity` test. The int8 quant noise (~0.4%
    ///   relative per element) can shift greedy sampling to a
    ///   different valid token (e.g., "Paris" in English vs French)
    ///   but preserves semantic correctness on strong-fact prompts.
    ///
    /// Enable when cache memory is the constraint (long contexts,
    /// multi-model servers, edge devices with <4 GB RAM). Stay fp32
    /// when latency matters more.
    pub fn with_kv_quant(
        mut self,
        quant: jouleclaw_loader_gguf::kv_cache_inplace::KvQuant,
    ) -> Self {
        self.kv_quant = quant;
        self
    }

    /// Per-token energy for the GGUF-backed path: ~43 mJ/tok on
    /// Bonsai-1.7B. Times `max_new_tokens` gives the steady-state cost
    /// the cascade should budget. Charged at the same ternary rates as
    /// the kernel (`2.5e-11 J/add`, `1e-10 J` per per-128-block scale
    /// multiply) — so the tier's joule receipt is consistent with what
    /// the runtime actually executes.
    pub fn gguf_joules(&self) -> Option<f64> {
        let b = self.gguf.as_ref()?;
        Some(b.joules_per_token * (self.max_new_tokens as f64).max(1.0))
    }

    /// First-order forward-pass cost. If a decoder is loaded, sums the
    /// matvec joules across every weight matrix in every block + the
    /// embedding and lm_head. Otherwise falls back to the per-layer
    /// sum (R28.0 path).
    pub fn forward_joules(&self) -> f64 {
        if let Some(d) = &self.decoder {
            // Per-position cost. The forward pass touches all weights
            // once per token; with no KV cache we re-run the full
            // pass for each generated token. Multiply by max_new_tokens
            // to estimate one generation.
            let per_pos = d.embed.matvec_joules()
                + d.lm_head.matvec_joules()
                + d.blocks.iter().map(|b| {
                    b.w_q.matvec_joules()
                        + b.w_k.matvec_joules()
                        + b.w_v.matvec_joules()
                        + b.w_o.matvec_joules()
                        + b.w_gate.matvec_joules()
                        + b.w_up.matvec_joules()
                        + b.w_down.matvec_joules()
                }).sum::<f64>();
            // Token-by-token (quadratic without KV cache): if prompt has
            // p tokens and we generate n new, average sequence length is
            // p + n/2. For a generic estimate, scale by max_new_tokens.
            per_pos * (self.max_new_tokens as f64).max(1.0)
        } else {
            self.layers.iter().map(|l| l.matvec_joules()).sum::<f64>()
        }
    }

    /// Estimated wall-clock latency. Heuristic; R32+ calibration replaces this.
    pub fn forward_latency(&self) -> Duration {
        let base_ns = if let Some(d) = &self.decoder {
            (d.n_layers as u64) * 50_000 * (self.max_new_tokens as u64)
        } else {
            (self.layers.len() as u64) * 200
        };
        Duration::from_nanos(base_ns.max(1))
    }
}

impl Tier for PrismTier {
    fn id(&self) -> TierId {
        TierId::L3(L3ModelId(self.model_id))
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        let _text = match &q.input {
            QueryInput::Text(s) => s,
            _ => return None,
        };
        let joules = if let Some(j) = self.gguf_joules() {
            j
        } else if self.decoder.is_some() {
            self.forward_joules()
        } else if self.layers.is_empty() {
            10e-9 // dispatch floor
        } else {
            self.forward_joules()
        };
        Some(TierEstimate {
            joules,
            latency: self.forward_latency(),
            // Statistical: single samples are not provably correct.
            // R28.1.1 will replace 0.5 with a calibrated value from a held-out eval.
            confidence_floor: 0.5,
        })
    }

    fn try_answer(&mut self, q: &Query, _budget: f64) -> Result<Answer, AnswerError> {
        // GGUF-backed dispatch is the production path. The model is
        // borrowed long-lived by this tier; per call we spin a fresh
        // Runtime + Conversation (both cheap) so cache state is reset
        // per query and there's no cross-query interference.
        if self.gguf.is_some() {
            return self.gguf_answer(q);
        }

        // R28.1 hot path: run the decoder if one is loaded.
        if let Some(decoder) = &self.decoder {
            let text = match &q.input {
                QueryInput::Text(s) => s,
                _ => return Ok(refused(
                    self.id(),
                    0.0,
                    RefusalReason::Inapplicable,
                )),
            };
            let prompt_tokens = TernaryDecoder::encode_bytes(text);
            let all_tokens = decoder.generate_greedy(&prompt_tokens, self.max_new_tokens);
            // Output = the generated continuation (everything after the prompt).
            let continuation = &all_tokens[prompt_tokens.len()..];
            let out_text = TernaryDecoder::decode_bytes(continuation);
            let cost = self.forward_joules();
            return Ok(Answer {
                output: AnswerOutput::Text(out_text),
                tier_used: self.id(),
                joules_spent: cost,
                confidence: 0.5,
                trace: hit_trace(self.id(), cost),
                verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
            });
        }

        // R28.0 path: kernel present (or empty), no decoder. Refuse with
        // a structured reason so the cascade falls through to a higher-cost
        // tier rather than producing a wrong answer.
        let cost = if self.layers.is_empty() { 10e-9 } else { self.forward_joules() };
        let reason = if self.layers.is_empty() {
            RefusalReason::TierSpecific(
                "PrismTier kernel ready; load weights via from_decoder to enable inference".into(),
            )
        } else {
            RefusalReason::TierSpecific(format!(
                "PrismTier has {} quantized layer(s) but no decoder; use from_decoder instead",
                self.layers.len()
            ))
        };
        let _ = q;
        Ok(refused(self.id(), cost, reason))
    }

    fn coord(&self) -> Option<jouleclaw_cascade::coord::Coord> {
        use jouleclaw_cascade::coord::{
            Coord, Encoding, Entity, Interface, NamedPrimitive, PrimitiveSet, Thermo,
            Verify, Zone,
        };
        Some(
            Coord::new(
                Zone::Z2_3,
                Entity::Reactive,
                Thermo::L1_Measure,
                Interface::Tokens,
                Verify::Statistical,
                Encoding::Facts,
            )
            .with_primitives(PrimitiveSet::of(&[
                NamedPrimitive::MlpForward,
                NamedPrimitive::Sample,
            ])),
        )
    }
}

fn refused(tier: TierId, joules: f64, reason: RefusalReason) -> Answer {
    let mut trace = ExecutionTrace::default();
    trace.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Refused(reason.clone()),
        joules,
    });
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: tier,
        joules_spent: joules,
        confidence: 0.0,
        trace,
        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
    }
}

fn hit_trace(tier: TierId, joules: f64) -> ExecutionTrace {
    let mut t = ExecutionTrace::default();
    t.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Hit,
        joules,
    });
    t
}

// ============================================================
// GGUF dispatch path
// ============================================================

/// Errors building a GGUF-backed [`PrismTier`].
#[derive(Debug)]
pub enum GgufTierError {
    /// Failed to parse the GGUF file (corrupt, wrong format, …).
    Parse(jouleclaw_loader_gguf::ParseError),
    /// Failed to derive a `LlamaConfig` from metadata (unsupported arch
    /// or missing keys).
    Config(jouleclaw_loader_gguf::llama::LoadError),
    /// Failed to load the tokenizer vocabulary.
    Vocab,
}

impl std::fmt::Display for GgufTierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "GGUF parse error: {:?}", e),
            Self::Config(e) => write!(f, "GGUF config error: {}", e),
            Self::Vocab => write!(f, "could not load tokenizer vocab from GGUF"),
        }
    }
}

impl std::error::Error for GgufTierError {}

/// Static per-token energy for the GGUF path, derived from
/// `LlamaConfig`. Counts every ternary weight matmul a decode step
/// performs: q/k/v/o per layer, gate/up/down per layer, plus the
/// `[d → V]` LM head. Charged at the kernel's own rates so the joule
/// receipt the cascade reports is consistent with what the runtime
/// actually executes. (`token_embd` lookup is `O(d)` per token and
/// rounds away below this.)
fn ternary_joules_per_token(c: &LlamaConfig) -> f64 {
    const J_ADD: f64 = 2.5e-11;
    const J_SCALE_MUL: f64 = 1e-10;
    const BLK: f64 = 128.0;
    let d = c.embedding_length as f64;
    let d_q = (c.head_count * c.head_dim) as f64;
    let d_kv = (c.head_count_kv * c.head_dim) as f64;
    let ffn = c.feed_forward_length as f64;
    let v = c.vocab_size as f64;
    let layers = c.block_count as f64;

    let per_layer_adds = d_q * d        // q_proj  [d_q × d]
        + d_kv * d                       // k_proj
        + d_kv * d                       // v_proj
        + d * d_q                        // o_proj  [d × d_q]
        + ffn * d                        // ffn_gate
        + ffn * d                        // ffn_up
        + d * ffn;                       // ffn_down
    let total_adds = layers * per_layer_adds + v * d; // + lm_head
    total_adds * J_ADD + (total_adds / BLK) * J_SCALE_MUL
}

impl PrismTier {
    /// Run a Text query through the GGUF-backed dispatch path. Builds
    /// a fresh Runtime + Conversation, prefills the prompt, decodes
    /// `max_new_tokens` greedy tokens, returns the continuation.
    fn gguf_answer(&mut self, q: &Query) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) => s.clone(),
            _ => return Ok(refused(
                self.id(), 0.0, RefusalReason::Inapplicable)),
        };
        let backend = self.gguf.as_ref()
            .expect("gguf_answer requires gguf backend");
        let max_new = self.max_new_tokens;
        let max_seq = backend.max_seq;
        let per_tok = backend.joules_per_token;

        use jouleclaw_loader_gguf::sample::SamplingConfig;
        use jouleclaw_runtime::generate::{GenerateConfig, KvCacheKind, TokenizerKind};
        use jouleclaw_runtime::streaming::Conversation;
        use jouleclaw_runtime::{encode_user_turn, Runtime};

        // Mild repetition penalty keeps small models from collapsing into
        // "the village in a small village in a small village …" under
        // greedy decoding. 1.15 is the llama.cpp default sweet spot:
        // strong enough to break runaway loops, weak enough to leave
        // coherent local repetition alone.
        let sampling = SamplingConfig::greedy().with_repetition_penalty(1.15);
        let chat_template = backend.chat_template;
        let cfg = GenerateConfig {
            max_new_tokens: max_new,
            add_bos: chat_template.is_none(),
            tokenizer_kind: TokenizerKind::Auto,
            cache_kind: KvCacheKind::InPlace,
            sampling,
            max_seq: Some(max_seq),
            stop_strings: vec![],
        };

        // With adaptive kernel selection (`Kernel::prefers` + the
        // compile-time picker), AppleAmx now declares itself `Weak` for
        // sub-1M-flop matmuls — so the tiny attention scores stay on
        // the reference scalar loop while any genuinely large MatMul
        // routes to Accelerate. `Runtime::boot` is now strictly
        // not-worse than `reference_only` for this workload.
        let mut conv = match Conversation::with_runtime_and_quant(
            &backend.model, &backend.vocab, max_seq, Runtime::boot(), self.kv_quant,
        ) {
            Ok(c) => c,
            Err(e) => return Ok(refused(
                self.id(), 0.0,
                RefusalReason::TierSpecific(format!("conv init failed: {}", e)))),
        };

        let mut tokens: Vec<u32> = Vec::with_capacity(max_new);
        // Drafter compute that isn't captured by conv.cumulative_joules()
        // (which only tracks the target's KV cache). The drafter
        // Conversation has its own cumulative_joules; we add it onto
        // the receipt at the bottom of this block.
        let mut drafter_extra_joules: f64 = 0.0;
        let (cont_text, measured_joules) = {
            // If the model exposes a chat template, wrap the query in it
            // and pre-tokenize with atomic special-token reconstruction —
            // plain BPE would shatter `<|im_start|>`/`<|user|>` into byte
            // subtokens, defeating the instruct tuning. Otherwise feed
            // the raw text and let the runtime tokenize.

            // Pre-tokenize so both the PLD and streaming paths see the
            // same input. `encode_user_turn` is template-aware;
            // otherwise dispatch by vocab tokenizer kind.
            let prompt_tokens: Vec<u32> = match chat_template {
                Some(template) => encode_user_turn(template, &text, &backend.vocab, true),
                None => match backend.vocab.model_name.as_str() {
                    "llama" => backend.vocab.encode_spm(&text, cfg.add_bos),
                    _ => backend.vocab.encode_bpe_regex(&text, cfg.add_bos),
                },
            };

            if let Some(drafter_backend) = self.drafter.as_ref() {
                // Drafter spec-decode path. Build a second Conversation
                // for the drafter (same Runtime::boot, separate cache).
                // extend_with_drafter handles the lockstep; PLD is
                // ignored if both are set (drafter is strictly more
                // general).
                let mut drafter_conv = match Conversation::with_runtime_and_quant(
                    &drafter_backend.model, &drafter_backend.vocab,
                    drafter_backend.max_seq, Runtime::boot(), self.kv_quant,
                ) {
                    Ok(c) => c,
                    Err(e) => return Ok(refused(
                        self.id(), 0.0,
                        RefusalReason::TierSpecific(format!("drafter init failed: {}", e)))),
                };
                match jouleclaw_runtime::extend_with_drafter(
                    &mut conv, &mut drafter_conv,
                    prompt_tokens, &cfg, &self.drafter_cfg,
                ) {
                    Ok(outcome) => {
                        tokens.extend(outcome.tokens);
                        // The target conversation's cumulative_joules
                        // already counts the verify forwards. The
                        // drafter conv's cumulative_joules counts the
                        // drafts. Total receipt = target + drafter.
                        drafter_extra_joules = drafter_conv.cumulative_joules();
                    }
                    Err(e) => return Ok(refused(
                        self.id(), 0.0,
                        RefusalReason::TierSpecific(format!("drafter extend failed: {}", e)))),
                }
            } else if let Some(pld_cfg) = self.pld {
                // PLD path: batched extend, no streaming. Returns the
                // generated tokens, measured joules, and the per-step
                // acceptance histogram. mean(accepted_per_step) > 1.0
                // means PLD landed hits and we got real speedup.
                match conv.extend_pld_tokens(prompt_tokens, &cfg, &pld_cfg) {
                    Ok(outcome) => {
                        tokens.extend(outcome.tokens);
                    }
                    Err(e) => return Ok(refused(
                        self.id(), 0.0,
                        RefusalReason::TierSpecific(format!("PLD extend failed: {}", e)))),
                }
            } else {
                // Streaming path (no PLD, no drafter).
                let stream = match conv.extend_tokens(prompt_tokens, &cfg) {
                    Ok(s) => s,
                    Err(e) => return Ok(refused(
                        self.id(), 0.0,
                        RefusalReason::TierSpecific(format!("extend failed: {}", e)))),
                };
                for st in stream {
                    match st {
                        Ok(t) => tokens.push(t.id),
                        Err(e) => return Ok(refused(
                            self.id(), per_tok * tokens.len() as f64,
                            RefusalReason::TierSpecific(format!("decode failed: {}", e)))),
                    }
                }
            }
            // Match decode to encoder: SPM vocabs (Llama 1/2) use U+2581
            // for word-start; decode_bpe would emit those as raw bytes,
            // collapsing spaces. Dispatch on the vocab's declared tokenizer.
            let txt = match backend.vocab.model_name.as_str() {
                "llama" => backend.vocab.decode_spm(&tokens),
                _ => backend.vocab.decode_bpe(&tokens),
            };
            // `cumulative_joules` is the sum of every kernel's reported
            // `KernelResult.joules` across prefill + every decode step
            // — the **measured** energy, distinct from the static
            // per-token estimate. Reporting this (not `per_tok *
            // n_tokens`) is what gives the cascade's calibration ledger
            // a real estimated-vs-actual ratio to learn `learned_mu`
            // from. The static estimate stays in `estimate_cost`
            // (pre-dispatch budgeting); the receipt is now honest.
            (txt, conv.cumulative_joules())
        };

        // Add drafter compute onto the receipt. For the non-drafter
        // path this is 0.0 and a no-op; for the drafter path it folds
        // in the drafter Conversation's cumulative joules so the
        // cascade calibration ledger sees the FULL compute cost
        // (target verify forwards + drafter generation forwards).
        let measured_joules = measured_joules + drafter_extra_joules;

        // Fall back to the static estimate only if the runtime
        // reported zero (no kernel accounting available).
        let joules_spent = if measured_joules > 0.0 {
            measured_joules
        } else {
            per_tok * tokens.len().max(1) as f64
        };
        Ok(Answer {
            output: AnswerOutput::Text(cont_text),
            tier_used: self.id(),
            joules_spent,
            confidence: 0.5,
            trace: hit_trace(self.id(), joules_spent),
            verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn empty_tier_declares_coord_and_floor_cost() {
        let tier = PrismTier::empty(42);
        let q = text_query("anything");
        let est = tier.estimate_cost(&q).expect("estimate should be Some");
        assert!(est.joules > 0.0);
        assert!(est.joules < 1e-6, "empty tier should report sub-µJ floor");
        assert_eq!(tier.id(), TierId::L3(L3ModelId(42)));
        let coord = tier.coord().expect("coord should be Some");
        assert!(matches!(coord.zone, jouleclaw_cascade::coord::Zone::Z2_3));
        assert!(matches!(
            coord.verify,
            jouleclaw_cascade::coord::Verify::Statistical
        ));
    }

    #[test]
    fn empty_tier_refuses_with_structured_reason() {
        let mut tier = PrismTier::empty(0);
        let ans = tier.try_answer(&text_query("hello"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(
                    msg.contains("from_decoder"),
                    "refusal should point users to from_decoder: {}",
                    msg
                );
            }
            other => panic!("expected structured refusal, got {:?}", other),
        }
    }

    #[test]
    fn decoder_loaded_tier_hits_with_text_output() {
        use crate::model::{synthetic_model, ModelConfig};

        let decoder = synthetic_model(ModelConfig::tiny_byte(), 0xABCDEF).unwrap();
        let mut tier = PrismTier::from_decoder(99, decoder).with_max_new_tokens(4);
        let q = text_query("hi");
        let ans = tier.try_answer(&q, 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(s) => {
                assert!(!s.is_empty(), "decoder should produce some output");
                // Random weights, so we can't assert content — just shape.
                // 4 new tokens at 1 byte each (mostly; multi-byte utf-8 may
                // collapse), so the string is at most 4 chars when decoded
                // as ascii but may be lossy-utf8-replaced for non-ascii bytes.
            }
            other => panic!("expected Text, got {:?}", other),
        }
        assert_eq!(ans.tier_used, TierId::L3(L3ModelId(99)));
        assert!(ans.joules_spent > 0.0);
    }

    #[test]
    fn decoder_path_is_deterministic_per_seed() {
        use crate::model::{synthetic_model, ModelConfig};

        let d1 = synthetic_model(ModelConfig::tiny_byte(), 555).unwrap();
        let d2 = synthetic_model(ModelConfig::tiny_byte(), 555).unwrap();
        let mut t1 = PrismTier::from_decoder(0, d1).with_max_new_tokens(6);
        let mut t2 = PrismTier::from_decoder(0, d2).with_max_new_tokens(6);
        let a = t1.try_answer(&text_query("test"), 1.0).unwrap();
        let b = t2.try_answer(&text_query("test"), 1.0).unwrap();
        match (a.output, b.output) {
            (AnswerOutput::Text(sa), AnswerOutput::Text(sb)) => {
                assert_eq!(sa, sb, "same seed + same prompt must yield same output");
            }
            _ => panic!("both should be Text"),
        }
    }

    #[test]
    fn loaded_tier_cost_scales_with_layer_size() {
        let ternary_small = TernaryMatrix::from_f32(8, 16, &vec![1.0_f32; 128]).unwrap();
        let ternary_large = TernaryMatrix::from_f32(64, 64, &vec![1.0_f32; 4096]).unwrap();

        let small = PrismTier::from_layers(1, vec![PrismLayer::Ternary(ternary_small)]);
        let large = PrismTier::from_layers(1, vec![PrismLayer::Ternary(ternary_large)]);

        assert!(large.forward_joules() > small.forward_joules());
        assert!(large.forward_joules() < 1e-3, "still well under 1 mJ at this size");
    }

    #[test]
    fn mixed_ternary_and_bit_layers_compose() {
        let t = TernaryMatrix::from_f32(4, 8, &vec![1.0_f32; 32]).unwrap();
        let b = BitMatrix::from_f32(4, 4, &vec![1.0_f32; 16]).unwrap();
        let tier = PrismTier::from_layers(
            7,
            vec![PrismLayer::Ternary(t), PrismLayer::Bit(b)],
        );
        assert_eq!(tier.layers.len(), 2);
        let est = tier.estimate_cost(&text_query("x")).unwrap();
        assert!(est.joules >= tier.forward_joules() - 1e-15);
    }

    #[test]
    fn cascade_can_register_prism_tier() {
        // Smoke test: PrismTier slots into the cascade like any other Tier.
        let mut cascade = Cascade::new();
        cascade.register(Box::new(PrismTier::empty(0)));
        let mut rt = Runtime::new_without_l0(cascade);
        // No L4 fallback registered, so the cascade will surface either
        // NoTierSatisfied (preferred) or a tier-level refusal as an
        // error. We only care that registration succeeded and a query
        // can flow through without panicking.
        let _ = rt.answer(text_query("compute something"));
    }
}
