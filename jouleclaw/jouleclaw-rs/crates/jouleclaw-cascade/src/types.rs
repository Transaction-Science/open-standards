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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierId {
    L0,
    L1(L1Primitive),
    L2(L2ModelId),
    L3(L3ModelId),
    L4(L4ModelId),
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

impl TierId {
    pub fn family(&self) -> &'static str {
        match self {
            TierId::L0 => "L0",
            TierId::L1(_) => "L1",
            TierId::L2(_) => "L2",
            TierId::L3(_) => "L3",
            TierId::L4(_) => "L4",
        }
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
