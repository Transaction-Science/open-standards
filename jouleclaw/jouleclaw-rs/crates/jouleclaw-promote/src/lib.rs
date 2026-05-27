//! Yang→yin promotion — the mechanism that makes JouleClaw cheaper the
//! more it is used.
//!
//! The cascade's expensive path is the statistical compartment (the
//! model tiers, L3/L4). A naive system re-pays that cost every time the
//! same question is asked — the perpetual model tax. JouleClaw refuses
//! to: when a compartment answer is **verified**, it is *promoted* into
//! a permanent, provenance-stamped deterministic entry. The next time
//! that exact input arrives, a [`PromotedTier`] resolves it at
//! lookup-table energy (nanojoules) and the model is never consulted
//! again. The statistical surface shrinks as the deterministic surface
//! learns — "author once, resolve forever."
//!
//! This is distinct from the runtime's volatile L0 cache in three ways
//! that matter:
//!
//! 1. **Verified-only.** Only answers a verifier approved are promoted —
//!    promotions are facts, not guesses.
//! 2. **Permanent + auditable.** Promotions are never evicted and every
//!    one is recorded in a [`PromotionLog`] — the curated, provenance-
//!    keyed dataset is itself the high-value asset.
//! 3. **Provenance-keyed.** The key is a BLAKE3 content hash of the
//!    query input (provenance-as-cache); the stored entry remembers
//!    which model tier originally produced it and when.
//!
//! ## Wiring
//!
//! - Register a [`PromotedTier`] as the front (cheapest) tier of the
//!   cascade, sharing a store via `Arc<Mutex<_>>`.
//! - After a model tier answers and a verifier approves, call
//!   [`PromotionGate::consider`] with the same shared store. Verified
//!   high-confidence compartment answers are promoted; everything else
//!   is left alone.
//!
//! No dependency on the verifier or receipt crates — the caller passes
//! the boolean verdict in, keeping this crate a thin, composable
//! mechanism.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, JouleClass, Query, QueryInput,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;

/// Energy charged for a promoted (deterministic LUT) resolution. A few
/// nanojoules — a content-addressed map lookup. Compare with the
/// megajoule-class cost of the model tier the promotion replaces.
pub const PROMOTED_JOULES: f64 = 5e-9;

/// Default minimum confidence a verified answer needs before it is
/// promoted to a permanent deterministic fact. Promotions are forever,
/// so the bar is high.
pub const DEFAULT_PROMOTE_CONFIDENCE: f32 = 0.9;

/// Content-addressable key: BLAKE3 of the canonical query input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromotionKey([u8; 32]);

impl PromotionKey {
    /// Derive the key from a query's input. Deterministic and
    /// modality-aware: text, structured, binary, image, audio, and
    /// multimodal inputs each hash under a distinct domain tag so two
    /// different modalities with identical bytes never collide.
    pub fn of(query: &Query) -> Self {
        let mut h = blake3::Hasher::new();
        match &query.input {
            QueryInput::Text(t) => {
                h.update(b"T");
                h.update(t.as_bytes());
            }
            QueryInput::Structured(b) => {
                h.update(b"S");
                h.update(b);
            }
            QueryInput::Binary(b) => {
                h.update(b"B");
                h.update(b);
            }
            QueryInput::Image(b) => {
                h.update(b"I");
                h.update(b);
            }
            QueryInput::Audio(b) => {
                h.update(b"A");
                h.update(b);
            }
            QueryInput::Multimodal { text, images, audio } => {
                h.update(b"M");
                h.update(text.as_bytes());
                for im in images {
                    h.update(b"i");
                    h.update(im);
                }
                for au in audio {
                    h.update(b"a");
                    h.update(au);
                }
            }
        }
        Self(*h.finalize().as_bytes())
    }

    /// Lowercase hex form, for logs and audit export.
    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

/// A promoted deterministic fact: a verified compartment answer that now
/// resolves without the model.
#[derive(Debug, Clone)]
pub struct PromotedEntry {
    /// The verified answer payload, served verbatim on a hit.
    pub output: AnswerOutput,
    /// Confidence at promotion time.
    pub confidence: f32,
    /// The model/compartment tier that originally produced this answer.
    pub origin_tier: TierId,
    /// Unix-seconds when the promotion happened.
    pub promoted_at_secs: u64,
    /// How many times this promoted fact has been reused (each reuse is
    /// one model invocation avoided).
    pub hits: u64,
}

/// An append-only audit record of a single promotion — the curated,
/// provenance-keyed dataset that promotion produces as a side effect.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PromotionLogEntry {
    /// Hex content key of the promoted query.
    pub key_hex: String,
    /// Wire tag of the origin model tier (e.g. `"L3"`, `"L4"`).
    pub origin_tier: String,
    /// Confidence at promotion.
    pub confidence: f32,
    /// Unix-seconds of promotion.
    pub promoted_at_secs: u64,
}

/// The store of promoted facts plus the promotion log. Implementors may
/// back this with disk, a DB, or a registry; the in-memory default is
/// permanent for the process lifetime.
pub trait PromotionStore: Send {
    /// Look up a promoted fact and, on a hit, count the reuse.
    fn lookup(&mut self, key: &PromotionKey) -> Option<PromotedEntry>;
    /// Record a promotion (and append to the log).
    fn record(&mut self, key: PromotionKey, entry: PromotedEntry, log: PromotionLogEntry);
    /// Number of promoted facts held.
    fn len(&self) -> usize;
    /// Total model invocations avoided so far (sum of hits).
    fn invocations_avoided(&self) -> u64;
}

/// In-memory, never-evicted promotion store.
#[derive(Debug, Default)]
pub struct InMemoryPromotionStore {
    facts: HashMap<[u8; 32], PromotedEntry>,
    log: Vec<PromotionLogEntry>,
}

impl InMemoryPromotionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// The append-only promotion log — the curated labelled dataset.
    pub fn log(&self) -> &[PromotionLogEntry] {
        &self.log
    }

    /// Export the promotion log as JSON (the audit / dataset artifact).
    pub fn export_log_json(&self) -> Result<String, PromoteError> {
        serde_json::to_string(&self.log).map_err(|e| PromoteError::Serialize(e.to_string()))
    }
}

impl PromotionStore for InMemoryPromotionStore {
    fn lookup(&mut self, key: &PromotionKey) -> Option<PromotedEntry> {
        let entry = self.facts.get_mut(&key.0)?;
        entry.hits += 1;
        Some(entry.clone())
    }

    fn record(&mut self, key: PromotionKey, entry: PromotedEntry, log: PromotionLogEntry) {
        // Promotions are permanent and idempotent: first verified answer
        // for a key wins; we do not overwrite a fact already proven.
        if self.facts.contains_key(&key.0) {
            return;
        }
        self.facts.insert(key.0, entry);
        self.log.push(log);
    }

    fn len(&self) -> usize {
        self.facts.len()
    }

    fn invocations_avoided(&self) -> u64 {
        self.facts.values().map(|e| e.hits).sum()
    }
}

/// Errors from the promotion machinery.
#[derive(Debug, thiserror::Error)]
pub enum PromoteError {
    #[error("failed to serialize promotion log: {0}")]
    Serialize(String),
}

/// A shared promotion store, held by both the [`PromotedTier`] (reads)
/// and the [`PromotionGate`] (writes).
pub type SharedStore<S> = Arc<Mutex<S>>;

/// Convenience: a fresh shared in-memory store.
pub fn shared_in_memory() -> SharedStore<InMemoryPromotionStore> {
    Arc::new(Mutex::new(InMemoryPromotionStore::new()))
}

/// The front deterministic tier: resolves a query for free (LUT energy)
/// if it has been promoted, otherwise refuses so the cascade continues.
pub struct PromotedTier<S: PromotionStore> {
    store: SharedStore<S>,
}

impl<S: PromotionStore> PromotedTier<S> {
    pub fn new(store: SharedStore<S>) -> Self {
        Self { store }
    }
}

impl<S: PromotionStore + 'static> Tier for PromotedTier<S> {
    fn id(&self) -> TierId {
        // Promoted facts are deterministic lookups — the cache class.
        TierId::L0_1FactLut
    }

    fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
        // Always cheap to *check*. A miss refuses in `try_answer`; a hit
        // serves a verified, high-confidence answer — hence the high
        // confidence floor.
        Some(TierEstimate {
            joules: PROMOTED_JOULES,
            latency: Duration::from_micros(1),
            confidence_floor: DEFAULT_PROMOTE_CONFIDENCE,
        })
    }

    fn try_answer(&mut self, q: &Query, _budget_remaining: f64) -> Result<Answer, AnswerError> {
        let key = PromotionKey::of(q);
        let hit = self
            .store
            .lock()
            .map_err(|e| AnswerError::TierFailed {
                tier: TierId::L0_1FactLut,
                cause: format!("promotion store lock poisoned: {e}"),
            })?
            .lookup(&key);
        match hit {
            Some(entry) => Ok(Answer {
                output: entry.output,
                tier_used: TierId::L0_1FactLut,
                joules_spent: PROMOTED_JOULES,
                confidence: entry.confidence,
                trace: ExecutionTrace::default(),
                verification: VerificationStatus::Resolved,
            }),
            None => Ok(Answer {
                output: AnswerOutput::Refused(RefusalReason::Inapplicable),
                tier_used: TierId::L0_1FactLut,
                joules_spent: PROMOTED_JOULES,
                confidence: 0.0,
                trace: ExecutionTrace::default(),
                verification: VerificationStatus::Resolved,
            }),
        }
    }
}

/// Promotes verified compartment answers into the shared store.
///
/// Call [`consider`](Self::consider) after a model tier answers AND a
/// verifier has ruled on it. Only verified, high-confidence answers that
/// came from the statistical compartment (model/wire class) are
/// promoted — promoting an already-deterministic answer would be
/// pointless, and promoting an unverified one would poison the cache.
pub struct PromotionGate<S: PromotionStore> {
    store: SharedStore<S>,
    min_confidence: f32,
}

impl<S: PromotionStore> PromotionGate<S> {
    pub fn new(store: SharedStore<S>) -> Self {
        Self {
            store,
            min_confidence: DEFAULT_PROMOTE_CONFIDENCE,
        }
    }

    /// Override the promotion confidence bar.
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self
    }

    /// Whether `answer` (for `query`) is eligible to be promoted, given
    /// the verifier's `verified` verdict. Pure predicate, no side
    /// effects — useful for tracing/metrics.
    pub fn is_eligible(&self, answer: &Answer, verified: bool) -> bool {
        verified
            && answer.confidence >= self.min_confidence
            && !matches!(answer.output, AnswerOutput::Refused(_))
            && matches!(
                answer.tier_used.joule_class(),
                JouleClass::Model | JouleClass::Wire
            )
    }

    /// Consider `answer` for promotion. Returns `true` if it was
    /// promoted (newly recorded), `false` otherwise. `now_secs` is the
    /// caller's clock (kept injectable for determinism in tests).
    pub fn consider(
        &mut self,
        query: &Query,
        answer: &Answer,
        verified: bool,
        now_secs: u64,
    ) -> bool {
        if !self.is_eligible(answer, verified) {
            return false;
        }
        let key = PromotionKey::of(query);
        let entry = PromotedEntry {
            output: answer.output.clone(),
            confidence: answer.confidence,
            origin_tier: answer.tier_used,
            promoted_at_secs: now_secs,
            hits: 0,
        };
        let log = PromotionLogEntry {
            key_hex: key.to_hex(),
            origin_tier: answer.tier_used.wire_tag().to_string(),
            confidence: answer.confidence,
            promoted_at_secs: now_secs,
        };
        let mut guard = match self.store.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let before = guard.len();
        guard.record(key, entry, log);
        guard.len() > before
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, L3ModelId, QualityFloor, QueryInput,
    };

    fn text(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn model_answer(text_out: &str, conf: f32) -> Answer {
        Answer {
            output: AnswerOutput::Text(text_out.to_string()),
            tier_used: TierId::L3(L3ModelId(0)),
            joules_spent: 2.0,
            confidence: conf,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        }
    }

    #[test]
    fn key_is_deterministic_and_modality_aware() {
        assert_eq!(PromotionKey::of(&text("hi")), PromotionKey::of(&text("hi")));
        assert_ne!(PromotionKey::of(&text("hi")), PromotionKey::of(&text("ho")));
        // Same bytes, different modality → different key.
        let t = Query {
            input: QueryInput::Text("x".into()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let b = Query {
            input: QueryInput::Binary(b"x".to_vec()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert_ne!(PromotionKey::of(&t), PromotionKey::of(&b));
    }

    #[test]
    fn promoted_tier_misses_before_promotion() {
        let store = shared_in_memory();
        let mut tier = PromotedTier::new(store);
        let ans = tier.try_answer(&text("capital of france"), 1.0).unwrap();
        assert!(matches!(ans.output, AnswerOutput::Refused(_)));
    }

    #[test]
    fn verified_model_answer_is_promoted_then_served_deterministically() {
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        let mut tier = PromotedTier::new(store.clone());

        let q = text("capital of france");
        let a = model_answer("Paris", 0.95);

        // Verified → promoted.
        assert!(gate.consider(&q, &a, true, 1000));
        // Now the front tier serves it deterministically at LUT energy.
        let served = tier.try_answer(&q, 1.0).unwrap();
        match served.output {
            AnswerOutput::Text(t) => assert_eq!(t, "Paris"),
            other => panic!("expected promoted text, got {other:?}"),
        }
        assert_eq!(served.tier_used, TierId::L0_1FactLut);
        assert!(served.joules_spent < a.joules_spent / 1_000_000.0); // 2 J → 5 nJ
        assert!((served.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn unverified_is_not_promoted() {
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        assert!(!gate.consider(&text("q"), &model_answer("a", 0.99), false, 1));
        assert_eq!(store.lock().unwrap().len(), 0);
    }

    #[test]
    fn low_confidence_is_not_promoted() {
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        assert!(!gate.consider(&text("q"), &model_answer("a", 0.5), true, 1));
        assert_eq!(store.lock().unwrap().len(), 0);
    }

    #[test]
    fn deterministic_origin_is_not_promoted() {
        // An answer that already came from a deterministic tier should
        // not be promoted — there is no model tax to avoid.
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        let det = Answer {
            output: AnswerOutput::Text("4".into()),
            tier_used: TierId::L0_5ToolCompute, // Lawful class
            joules_spent: 15e-6,
            confidence: 1.0,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        };
        assert!(!gate.consider(&text("2+2"), &det, true, 1));
    }

    #[test]
    fn refused_answer_is_not_promoted() {
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        let refused = Answer {
            output: AnswerOutput::Refused(RefusalReason::Inapplicable),
            tier_used: TierId::L3(L3ModelId(0)),
            joules_spent: 2.0,
            confidence: 0.95,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        };
        assert!(!gate.consider(&text("q"), &refused, true, 1));
    }

    #[test]
    fn promotion_is_idempotent() {
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        let q = text("q");
        assert!(gate.consider(&q, &model_answer("first", 0.95), true, 1));
        // Second consider for the same key does not overwrite or grow.
        assert!(!gate.consider(&q, &model_answer("second", 0.99), true, 2));
        assert_eq!(store.lock().unwrap().len(), 1);
    }

    #[test]
    fn reuse_counts_invocations_avoided() {
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        let mut tier = PromotedTier::new(store.clone());
        let q = text("hot query");
        gate.consider(&q, &model_answer("ans", 0.95), true, 1);
        for _ in 0..5 {
            let _ = tier.try_answer(&q, 1.0).unwrap();
        }
        assert_eq!(store.lock().unwrap().invocations_avoided(), 5);
    }

    #[test]
    fn promotion_log_records_the_dataset() {
        let store = shared_in_memory();
        let mut gate = PromotionGate::new(store.clone());
        gate.consider(&text("q1"), &model_answer("a1", 0.95), true, 100);
        gate.consider(&text("q2"), &model_answer("a2", 0.95), true, 200);
        let guard = store.lock().unwrap();
        assert_eq!(guard.log().len(), 2);
        assert_eq!(guard.log()[0].origin_tier, "L3");
        let json = guard.export_log_json().unwrap();
        assert!(json.contains("\"origin_tier\":\"L3\""));
    }

    #[test]
    fn end_to_end_via_cascade_runtime() {
        use jouleclaw_cascade::tier::{Cascade, Runtime};

        let store = shared_in_memory();
        // Pre-promote a fact (as if a model answered + verifier approved).
        let mut gate = PromotionGate::new(store.clone());
        let q = text("what is the capital of france");
        gate.consider(&q, &model_answer("Paris", 0.97), true, 1);

        // Register the promoted tier as the cascade front.
        let mut cascade = Cascade::new();
        cascade.register(Box::new(PromotedTier::new(store.clone())));
        let mut rt = Runtime::new_without_l0(cascade);

        let ans = rt.answer(text("what is the capital of france")).unwrap();
        match ans.output {
            AnswerOutput::Text(t) => assert_eq!(t, "Paris"),
            other => panic!("expected promoted answer, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L0_1FactLut);
    }
}
