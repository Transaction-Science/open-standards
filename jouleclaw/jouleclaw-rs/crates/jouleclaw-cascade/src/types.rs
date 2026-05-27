//! Core cascade types: `Query`, `Answer`, and `TierId`.
//!
//! See `specs/r0.1-query-answer-tier.md` for the design rationale.
//! This module defines the load-bearing types at the cascade boundary.
//! Every tier consumes `Query` and produces `Answer`. The runtime walks
//! tiers in cost order, asking each whether it can answer.

use std::time::Duration;

// ============================================================
// Query
// ============================================================

/// A single question submitted to the runtime.
#[derive(Debug, Clone)]
pub struct Query {
    pub input: QueryInput,
    pub budget: JouleBudget,
    pub quality: QualityFloor,
    pub context: ContextRef,
    pub deadline: Option<Duration>,
}

/// The question's payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryInput {
    Text(String),
    /// Pre-serialized structured data (JSON-equivalent bytes). Kept as
    /// bytes so `Query` is `Hash` — needed for L0 cache keying. The
    /// serialization format is canonical JSON (sorted keys, no
    /// whitespace) so structurally-identical inputs produce identical
    /// hashes.
    Structured(Vec<u8>),
    Binary(Vec<u8>),
    /// R31: opaque image bytes (PNG/JPEG/etc.). Format detection and
    /// preprocessing happen inside the receiving tier. Kept as bytes
    /// for L0 hashability.
    Image(Vec<u8>),
    /// R31: opaque audio bytes (WAV/FLAC/MP3/etc.).
    Audio(Vec<u8>),
    /// R31: a combined multimodal query — text plus zero-or-more images
    /// and audio clips. Models that consume all three modalities in one
    /// forward pass (LMMs / LFM-VL class) receive this variant.
    Multimodal {
        text: String,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
    },
}

/// Hard ceiling and soft target for joule cost.
///
/// The runtime fails with `BudgetExhausted` if `hard_limit` would be
/// exceeded. It prefers tiers whose estimated cost stays under
/// `soft_target` but does not fail on `soft_target` overrun.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JouleBudget {
    pub hard_limit: f64,
    pub soft_target: f64,
}

impl JouleBudget {
    /// 1 µJ hard, 100 nJ soft — for L0/L1-class queries.
    pub fn trivial() -> Self {
        Self { hard_limit: 1e-6, soft_target: 1e-7 }
    }
    /// 1 mJ hard, 100 µJ soft — for L1/L2-class queries.
    pub fn cheap() -> Self {
        Self { hard_limit: 1e-3, soft_target: 1e-4 }
    }
    /// 1 J hard, 100 mJ soft — for L3-class queries.
    pub fn standard() -> Self {
        Self { hard_limit: 1.0, soft_target: 1e-1 }
    }
    /// 100 J hard, 10 J soft — for L4-class queries.
    pub fn expensive() -> Self {
        Self { hard_limit: 1e2, soft_target: 1e1 }
    }

    /// Joules remaining given `spent`.
    pub fn remaining(&self, spent: f64) -> f64 {
        (self.hard_limit - spent).max(0.0)
    }
}

/// Minimum confidence required for a tier to claim the answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityFloor {
    pub min_confidence: f32,
    pub accept_partial: bool,
}

impl QualityFloor {
    /// Floor for factual lookups: 0.99 confidence required.
    pub fn exact() -> Self {
        Self { min_confidence: 0.99, accept_partial: false }
    }
    /// Floor for chat: 0.7 confidence required, partial OK.
    pub fn chat() -> Self {
        Self { min_confidence: 0.7, accept_partial: true }
    }
    /// Always accept any tier (for tests and exploration).
    pub fn any() -> Self {
        Self { min_confidence: 0.0, accept_partial: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub [u8; 16]);

impl SessionId {
    pub fn zero() -> Self {
        Self([0u8; 16])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextFingerprint(pub [u8; 32]);

impl ContextFingerprint {
    pub fn empty() -> Self {
        Self([0u8; 32])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextRef {
    pub session_id: SessionId,
    pub history_fingerprint: ContextFingerprint,
}

impl ContextRef {
    /// A fresh, sessionless context.
    pub fn fresh() -> Self {
        Self {
            session_id: SessionId::zero(),
            history_fingerprint: ContextFingerprint::empty(),
        }
    }
}

// ============================================================
// Answer
// ============================================================

/// The result of answering a query.
#[derive(Debug, Clone)]
pub struct Answer {
    pub output: AnswerOutput,
    pub tier_used: TierId,
    pub joules_spent: f64,
    pub confidence: f32,
    pub trace: ExecutionTrace,
    /// Whether this answer's correctness is verified at decision time
    /// (the default) or pending later verification. Most tiers leave
    /// this as `Resolved`; tiers whose work resolves over time
    /// (`E=Active` or `V=Delayed` synthesizers) return `Pending(token)`
    /// and expect the caller to call `Runtime::resolve(token, outcome)`
    /// when the outcome is known.
    pub verification: crate::verification::VerificationStatus,
}

impl Answer {
    /// Mark an existing answer as having pending verification.
    /// Convenience for tiers that want to convert a `Resolved` answer
    /// into a `Pending` one without re-constructing the struct.
    pub fn with_pending_verification(mut self,
        token: crate::verification::VerificationToken) -> Self
    {
        self.verification = crate::verification::VerificationStatus::Pending(token);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnswerOutput {
    Text(String),
    Structured(Vec<u8>),
    /// A tier refused to claim the answer. The runtime continues to
    /// the next tier in the plan.
    Refused(RefusalReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefusalReason {
    LowConfidence(u32),  // confidence as q7.24 fixed point for Eq
    TierSpecific(String),
    Inapplicable,
}

impl RefusalReason {
    pub fn low_confidence(c: f32) -> Self {
        Self::LowConfidence((c.clamp(0.0, 1.0) * (1 << 24) as f32) as u32)
    }
    pub fn confidence_value(&self) -> Option<f32> {
        match self {
            Self::LowConfidence(c) => Some(*c as f32 / (1 << 24) as f32),
            _ => None,
        }
    }
}

/// A record of which tier(s) were attempted, in order, with per-tier
/// joule spend. The successful tier (if any) is last.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    pub attempts: Vec<TraceEntry>,
}

#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub tier: TierId,
    pub outcome: TraceOutcome,
    pub joules: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceOutcome {
    Hit,                                 // tier answered
    Refused(RefusalReason),
    SkippedBudget,                       // tier estimate > remaining budget
    SkippedQuality,                      // tier confidence floor < quality floor
    SkippedInapplicable,                 // tier estimate_cost returned None
}

// ============================================================
// Tier identification
// ============================================================

/// Tier identification across the JouleClaw cascade.
///
/// The coarse five-class taxonomy (`L0` Cache / `L1` Lawful / `L2` Embed /
/// `L3` Model / `L4` Wire) is the wire-stable identifier used by
/// `jouleclaw-prov` receipts and conformance vectors. The fractional
/// variants below (ported from `verity-cascade`) expand the internal
/// surface to the full L0–L10 taxonomy without changing the receipt
/// format: every fractional variant maps deterministically to one of
/// the five coarse classes via [`TierId::joule_class`] and the
/// [`map_to_coarse`] helper, so receipts continue to carry the stable
/// five-class tag while routers, calibration, and observability get
/// the richer surface.
///
/// Tier ordering (ascending energy class):
/// - `L0`            — content-addressed cache hit
/// - `L0.1` `FactLut`            — deterministic fact LUT
/// - `L0.25` `FormulaFirst`      — primary structural-relationship resolution
/// - `L0.5` `ToolCompute`        — pure deterministic tool dispatch
/// - `L0.75` `SsmRouter`         — local SSM intent classifier
/// - `L1(...)`                   — coarse lawful primitive
/// - `L1.25` `GraphRag`          — deterministic graph-enriched retrieval
/// - `L1.375` `StructContrast`   — second formula pass with graph context
/// - `L1.5` `SsmReader`          — local SSM reading comprehension
/// - `L2(...)`                   — coarse embed/retrieval model
/// - `L2.5` `NeuralRerank`       — neural reranker (ColBERT/SPLADE)
/// - `L3(...)`                   — coarse local stochastic model
/// - `L4(...)`                   — coarse remote frontier model
/// - `L4.5` `Proof`              — deterministic constraint/proof solver
/// - `L5..=L10`                  — meta-cognitive control plane
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierId {
    /// Coarse L0 — content-addressed cache hit, picojoules.
    L0,
    /// Coarse L1 — deterministic lawful primitive parameterised by
    /// which primitive resolves the query.
    L1(L1Primitive),
    /// Coarse L2 — embedding / nearest-neighbour retrieval model.
    L2(L2ModelId),
    /// Coarse L3 — local stochastic model (SSM / ternary / multimodal).
    L3(L3ModelId),
    /// Coarse L4 — remote frontier RPC.
    L4(L4ModelId),

    // ─── Fractional sub-tiers (L0 Cache class) ─────────────────────
    /// L0.1 — deterministic fact lookup table. Pure HashMap; if the
    /// answer is in the table, nothing else fires. Picojoules.
    L0_1FactLut,

    // ─── Fractional sub-tiers (L1 Lawful class) ────────────────────
    /// L0.25 — primary structural-contrast formula pass. Resolves
    /// queries by entity relationship alone, with no GraphRAG or model
    /// dependency.
    L0_25FormulaFirst,
    /// L0.5 — deterministic pure-function tool dispatch (numeric,
    /// string, date, set ops). Zero hallucination by construction.
    L0_5ToolCompute,
    /// L0.75 — local SSM intent router; classifies which downstream
    /// lawful path applies, without committing to a full read.
    L0_75SsmRouter,

    // ─── Fractional sub-tiers (L2 Embed class) ─────────────────────
    /// L1.25 — deterministic entity extraction plus knowledge-graph
    /// enrichment; supplies richer context for downstream tiers.
    L1_25GraphRag,
    /// L1.375 — second structural-contrast pass with graph-enriched
    /// entities; decomposes similarity per-dimension.
    L1_375StructContrast,
    /// L1.5 — local SSM reading-comprehension QA over retrieved
    /// passages.
    L1_5SsmReader,
    /// L2.5 — neural reranking (ColBERT / SPLADE class) over the
    /// federated/local retrieval candidate set.
    L2_5NeuralRerank,

    // ─── Fractional sub-tiers (L3 Model class) ─────────────────────
    /// L4.5 — deterministic constraint / proof solver (SAT, Sudoku,
    /// type inference, scheduling). Emits a verifiable proof receipt;
    /// resolves "find a state that fits" problems at constraint-solver
    /// energy, not LLM energy.
    L4_5Proof,

    // ─── Meta-cognitive control plane (L5–L10) ─────────────────────
    /// L5 — learned routing over episode memory and phasor similarity.
    L5Routing,
    /// L6 — multi-step recursive cascade agent.
    L6Agent,
    /// L7 — asynchronous meta-cognitive reflection and learning.
    L7Reflection,
    /// L8 — self-tuning damped control loops for routing weights.
    L8Tuner,
    /// L9 — pathology detection (oscillation, starvation, runaway).
    L9Supervisor,
    /// L10 — energy / cost budget enforcement and hard limits.
    L10Governor,
}

/// Where a cached entry currently lives.
///
/// The memory hierarchy is independent of the compute cascade tiers.
/// L0 cache hits can come from any memory tier; the joule cost
/// differs.
///
/// `Hot`        — small, in-process HashMap. ~6 nJ per lookup.
/// `Warm`       — bounded LRU in-process. ~10-20 nJ per lookup
///                (slightly slower due to LRU bookkeeping).
/// `Cold`       — disk-resident or compressed. ~1 µJ per lookup
///                (random disk I/O dominates).
/// `Persistent` — write-ahead-logged, durable across crashes.
///                Practically the same cost as Cold for reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    Hot,
    Warm,
    Cold,
    Persistent,
}

impl MemoryTier {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hot => "Hot",
            Self::Warm => "Warm",
            Self::Cold => "Cold",
            Self::Persistent => "Persistent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum L1Primitive {
    CacheLookup,
    Tokenize,
    Detokenize,
    Regex,
    Parse,
    TemplateFill,
    Retrieve,
    Execute,
    /// Pattern-lang's lawful lexicon — composed CortexIR primitives dispatched
    /// deterministically (gcd, fibonacci, is_prime, …). See joule-l1::lawful.
    Lawful,
    /// Belief-state / value-iteration planner over a world model. The
    /// constraint-satisfaction-shaped sibling to `Lawful`: rather than
    /// composing pre-proven primitives, it searches a discrete state
    /// space for a path that satisfies the goal. See `joule-l1::planner_tier`.
    Plan,
    /// Deterministic world-model step / rollout. Given a state and an
    /// action, returns the next state — same energy class as `Execute`
    /// but typed for world transitions. See `joule-l1::world_model_tier`.
    WorldModel,
    /// Energy-based constraint-satisfaction solver. Takes a problem with
    /// a crisp energy function (sudoku, SAT, scheduling, layout, …),
    /// searches the state space for a zero-energy state. The Konna /
    /// LeCun-EBM analog inside pattern-lang's cascade. See `ebm::EbmTier`.
    Ebm,
    /// Intent→verified-code synthesis. Pattern-lang's flagship: parse
    /// intent, retrieve vocabulary patterns, compose a typed DAG, emit
    /// source in a target language. Deterministic, no sampling. See
    /// `synth::SynthesisTier`.
    Synthesize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L2ModelId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L3ModelId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L4ModelId(pub u32);

/// The coarse five-class taxonomy that gates the wire-stable receipt
/// format in `jouleclaw-prov`. Every fractional `TierId` variant maps
/// to exactly one of these.
///
/// This enum mirrors `jouleclaw_prov::CascadeTier` shape-for-shape but
/// is defined locally so `jouleclaw-cascade` does not need to depend on
/// `jouleclaw-prov`. The [`map_to_coarse`] free function converts a
/// rich `TierId` to a `jouleclaw_prov::CascadeTier` for receipt
/// emission; consumers that already pull in `jouleclaw-prov` should
/// prefer that helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JouleClass {
    /// L0 family — content-addressed cache hits. Picojoules.
    Cache,
    /// L1 family — deterministic lawful primitives and the L0.x
    /// pre-L1 layers that compose into them. Nanojoules.
    Lawful,
    /// L2 family — embedding, retrieval, reranking. Sub-millijoules.
    Embed,
    /// L3 family — local stochastic models and deterministic proof
    /// gates that stand in for them. Joules to tens of joules.
    Model,
    /// L4 family — remote frontier RPC. Tens of joules and up.
    Wire,
    /// L5–L10 — meta-cognitive control plane. Not a runtime tier in
    /// the receipt sense; surfaced here so `joule_class` is total.
    Meta,
}

impl TierId {
    /// The coarse family this tier belongs to. Mirrors the wire tag
    /// for the five-class taxonomy; included for backward
    /// compatibility with code written before the fractional variants
    /// landed.
    pub fn family(&self) -> &'static str {
        match self {
            TierId::L0 | TierId::L0_1FactLut => "L0",
            TierId::L1(_)
            | TierId::L0_25FormulaFirst
            | TierId::L0_5ToolCompute
            | TierId::L0_75SsmRouter => "L1",
            TierId::L2(_)
            | TierId::L1_25GraphRag
            | TierId::L1_375StructContrast
            | TierId::L1_5SsmReader
            | TierId::L2_5NeuralRerank => "L2",
            TierId::L3(_) | TierId::L4_5Proof => "L3",
            TierId::L4(_) => "L4",
            TierId::L5Routing
            | TierId::L6Agent
            | TierId::L7Reflection
            | TierId::L8Tuner
            | TierId::L9Supervisor
            | TierId::L10Governor => "Lmeta",
        }
    }

    /// Stable wire tag for this tier — the string that flows into
    /// receipts, dashboards, and conformance vectors. Fractional
    /// variants use their dotted form (e.g. `"L0.25"`, `"L4.5"`); the
    /// five coarse variants use the bare `"L0".."L4"` tags so existing
    /// receipts remain byte-identical.
    pub fn wire_tag(&self) -> &'static str {
        match self {
            TierId::L0 => "L0",
            TierId::L1(_) => "L1",
            TierId::L2(_) => "L2",
            TierId::L3(_) => "L3",
            TierId::L4(_) => "L4",
            TierId::L0_1FactLut => "L0.1",
            TierId::L0_25FormulaFirst => "L0.25",
            TierId::L0_5ToolCompute => "L0.5",
            TierId::L0_75SsmRouter => "L0.75",
            TierId::L1_25GraphRag => "L1.25",
            TierId::L1_375StructContrast => "L1.375",
            TierId::L1_5SsmReader => "L1.5",
            TierId::L2_5NeuralRerank => "L2.5",
            TierId::L4_5Proof => "L4.5",
            TierId::L5Routing => "L5",
            TierId::L6Agent => "L6",
            TierId::L7Reflection => "L7",
            TierId::L8Tuner => "L8",
            TierId::L9Supervisor => "L9",
            TierId::L10Governor => "L10",
        }
    }

    /// Human-readable short name for this tier. Used in prose and
    /// dashboards; receipts should use [`TierId::wire_tag`] instead.
    pub fn name(&self) -> &'static str {
        match self {
            TierId::L0 => "Cache",
            TierId::L1(_) => "Lawful",
            TierId::L2(_) => "Embed",
            TierId::L3(_) => "Model",
            TierId::L4(_) => "Wire",
            TierId::L0_1FactLut => "FactLut",
            TierId::L0_25FormulaFirst => "FormulaFirst",
            TierId::L0_5ToolCompute => "ToolCompute",
            TierId::L0_75SsmRouter => "SsmRouter",
            TierId::L1_25GraphRag => "GraphRag",
            TierId::L1_375StructContrast => "StructContrast",
            TierId::L1_5SsmReader => "SsmReader",
            TierId::L2_5NeuralRerank => "NeuralRerank",
            TierId::L4_5Proof => "Proof",
            TierId::L5Routing => "Routing",
            TierId::L6Agent => "Agent",
            TierId::L7Reflection => "Reflection",
            TierId::L8Tuner => "Tuner",
            TierId::L9Supervisor => "Supervisor",
            TierId::L10Governor => "Governor",
        }
    }

    /// The coarse class this tier rolls up into for receipt emission.
    /// Receipts are stable at the five-class taxonomy; the fractional
    /// surface is internal. Meta tiers (`L5..=L10`) map to
    /// [`JouleClass::Meta`] and are not emitted in receipts — they
    /// govern the cascade, they do not answer queries.
    pub fn joule_class(&self) -> JouleClass {
        match self {
            TierId::L0 | TierId::L0_1FactLut => JouleClass::Cache,
            TierId::L1(_)
            | TierId::L0_25FormulaFirst
            | TierId::L0_5ToolCompute
            | TierId::L0_75SsmRouter => JouleClass::Lawful,
            TierId::L2(_)
            | TierId::L1_25GraphRag
            | TierId::L1_375StructContrast
            | TierId::L1_5SsmReader
            | TierId::L2_5NeuralRerank => JouleClass::Embed,
            TierId::L3(_) | TierId::L4_5Proof => JouleClass::Model,
            TierId::L4(_) => JouleClass::Wire,
            TierId::L5Routing
            | TierId::L6Agent
            | TierId::L7Reflection
            | TierId::L8Tuner
            | TierId::L9Supervisor
            | TierId::L10Governor => JouleClass::Meta,
        }
    }

    /// Whether this tier is a meta-cognitive control-plane layer
    /// (`L5..=L10`) rather than an execution tier. Meta tiers shape
    /// routing, learning, supervision, and budget enforcement; they
    /// do not directly resolve queries.
    pub fn is_meta(&self) -> bool {
        matches!(
            self,
            TierId::L5Routing
                | TierId::L6Agent
                | TierId::L7Reflection
                | TierId::L8Tuner
                | TierId::L9Supervisor
                | TierId::L10Governor
        )
    }
}

/// The coarse five-class tier used in JouleClaw receipts.
///
/// Shape-for-shape mirror of `jouleclaw_prov::CascadeTier` (same
/// variant names) so receipt-emitting crates can construct one with
/// `match coarse { CoarseTier::L0Cache => CascadeTier::L0Cache, ... }`
/// without `jouleclaw-cascade` having to depend on `jouleclaw-prov`.
///
/// **Why duplicated, not re-exported?** Receipts are pinned to a
/// stable five-class taxonomy, and `jouleclaw-cascade` is the lower
/// crate in the dependency graph (receipt emitters depend on the
/// cascade, not the other way around). Re-defining the shape here
/// keeps the dependency edges one-way and lets `jouleclaw-prov` evolve
/// receipt-side concerns (serialization, signing) without rebuilding
/// the cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoarseTier {
    /// Coarse L0 — content-addressed cache.
    L0Cache,
    /// Coarse L1 — deterministic lawful primitive.
    L1Lawful,
    /// Coarse L2 — embedding / retrieval / reranking.
    L2Embed,
    /// Coarse L3 — local stochastic model (or deterministic proof
    /// gate occupying the same energy class).
    L3Model,
    /// Coarse L4 — remote frontier RPC.
    L4Wire,
}

/// Map a rich [`TierId`] to the wire-stable [`CoarseTier`] used in
/// receipts.
///
/// JouleClaw receipts are pinned to the five-class taxonomy (`L0
/// Cache` / `L1 Lawful` / `L2 Embed` / `L3 Model` / `L4 Wire`) so the
/// receipt format stays byte-stable across releases as the internal
/// tier surface grows. The fractional and meta variants introduced
/// alongside the L0–L10 taxonomy fold into the five coarse classes
/// via this helper.
///
/// Meta tiers (`L5..=L10`) have no receipt representation — they
/// govern the cascade, they do not answer queries — and are
/// conservatively folded into `L0Cache` so callers that hand a meta
/// tier to the receipt builder by mistake still produce a well-formed
/// receipt rather than panicking. Callers SHOULD gate on
/// [`TierId::is_meta`] first and skip receipt emission for meta
/// dispatches.
///
/// Receipt-emitting crates (such as `jouleclaw-prov`) consume this by
/// pattern-matching `CoarseTier` into their own `CascadeTier` — the
/// two enums are intentionally shape-identical so the bridge is a
/// six-arm `match`.
pub fn map_to_coarse(tier: TierId) -> CoarseTier {
    match tier.joule_class() {
        JouleClass::Cache => CoarseTier::L0Cache,
        JouleClass::Lawful => CoarseTier::L1Lawful,
        JouleClass::Embed => CoarseTier::L2Embed,
        JouleClass::Model => CoarseTier::L3Model,
        JouleClass::Wire => CoarseTier::L4Wire,
        // Meta tiers are not emitted in receipts (they govern the
        // cascade, they do not answer queries). Fold them to the
        // cheapest coarse class so a well-formed receipt is still
        // produced if a caller forgets to gate on `is_meta` first.
        JouleClass::Meta => CoarseTier::L0Cache,
    }
}

// ============================================================
// Errors
// ============================================================

#[derive(Debug, Clone)]
pub enum AnswerError {
    /// Budget exhausted before any tier could produce an answer.
    BudgetExhausted {
        spent: f64,
        limit: f64,
        attempted_tiers: Vec<(TierId, f64)>,
    },
    /// Wall-clock deadline missed.
    DeadlineExceeded {
        elapsed: Duration,
        deadline: Duration,
    },
    /// Every tier refused.
    NoTierSatisfied {
        refusals: Vec<(TierId, RefusalReason)>,
    },
    /// A tier failed to execute. Bug-level.
    TierFailed {
        tier: TierId,
        cause: String,
    },
}

impl std::fmt::Display for AnswerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExhausted { spent, limit, .. } =>
                write!(f, "budget exhausted: spent {:.3e} J, limit {:.3e} J", spent, limit),
            Self::DeadlineExceeded { elapsed, deadline } =>
                write!(f, "deadline exceeded: elapsed {:?}, deadline {:?}", elapsed, deadline),
            Self::NoTierSatisfied { refusals } =>
                write!(f, "no tier satisfied query: {} tiers refused", refusals.len()),
            Self::TierFailed { tier, cause } =>
                write!(f, "tier {:?} failed: {}", tier, cause),
        }
    }
}

impl std::error::Error for AnswerError {}
