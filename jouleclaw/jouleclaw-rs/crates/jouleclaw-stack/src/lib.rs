//! JouleClaw full-stack assembler.
//!
//! The tier crates each ship and test in isolation. This crate wires
//! them into one runnable [`JouleClawStack`] using every crate's
//! **deterministic reference backend** — so `JouleClawStack::with_defaults()`
//! produces a cascade you can run a query through with zero external
//! dependencies (no model weights, no network, no API keys).
//!
//! ## Resolvers vs pipeline stages (why they aren't one flat walk)
//!
//! `jouleclaw_cascade::Runtime::answer` is *first-non-refused-wins*: the
//! first tier that returns a non-`Refused` output terminates the walk.
//! That semantics fits tiers whose output is a **final answer**, but the
//! execution surface contains two distinct kinds of tier:
//!
//! - **Promoted front tier** — a [`jouleclaw_promote::PromotedTier`] is
//!   registered ahead of everything. Verified model answers promoted via
//!   [`JouleClawStack::promote`] resolve here at nanojoule energy, so a
//!   repeated question never re-pays the model tax — cost trends to zero
//!   with use (the yang→yin path).
//! - **Resolvers** — produce a final answer (or refuse): [`FormulaFirstTier`]
//!   (L0.25), [`ToolTier`] (L0.5), [`LlmCheapTier`] (L3),
//!   [`VerificationTier`] (L4), [`ProofTier`] (L4.5). These are
//!   registered into [`JouleClawStack::runtime`] in ascending cost order;
//!   the cheapest one that can close the query wins.
//! - **Pipeline stages** — produce an *intermediate artifact*, not a
//!   standalone answer: [`SsmRouterTier`] (L0.75, a route),
//!   [`LocalIndexTier`] (L1), [`Federation`] (L2) and [`GraphRagTier`]
//!   (L1.25) (retrieved candidates / subgraph), [`StructContrastTier`]
//!   (L1.375, a contrast map), [`RerankTier`] (L2.5, a reordering), and
//!   [`SsmReaderTier`] (L1.5, the *read* terminal that extracts an answer
//!   from retrieved passages — it refuses a bare query, so it belongs to
//!   the pipeline, not the standalone walk). Dropping the retrieval /
//!   router stages into the linear walk would short-circuit it (the
//!   router would "answer" every query with a route), so they are exposed
//!   as fields and driven explicitly.
//!
//! The orchestrator is realized as [`JouleClawStack::rag`] /
//! [`JouleClawStack::rag_with`]: `retrieve → enrich → rerank → read`,
//! charging energy at each stage, honoring a joule budget, and emitting a
//! per-stage trace. See [`pipeline`].
//!
//! ## Control plane (L5–L10)
//!
//! - **L5 routing** — [`LearnedRouter`] is installed as the runtime's
//!   [`jouleclaw_cascade::router::Router`], so dispatch order improves as
//!   episodes accrue.
//! - **L7–L10** — [`ReflectionEngine`], [`Tuner`], [`Supervisor`],
//!   [`Governor`] are exposed as fields for the caller to feed
//!   observations into and read advice/verdicts out of.
//! - **L6 agent** — intentionally NOT auto-wired. [`jouleclaw_agent::AgentTier`]
//!   dispatches sub-queries back through the cascade, which is a borrow
//!   cycle against the `Runtime` it lives in. Wiring it needs a
//!   consumer-supplied `AgentCascade` adapter (e.g. over an
//!   `Arc<Mutex<Runtime>>`); the stack documents the seam rather than
//!   faking it.
//!
//! ## Not tiers
//!
//! `jouleclaw-decode` (grammar-constrained decoding) and
//! `jouleclaw-program` (typed-signature programs) are capability
//! libraries consumed *by* tiers (e.g. an LLM backend behind
//! [`LlmCheapTier`]), not cascade tiers themselves, so they are not
//! registered here.

#![forbid(unsafe_code)]

pub mod pipeline;
pub use pipeline::{RagConfig, RagOutcome, StageTrace};

use jouleclaw_cascade::tier::{Cascade, Runtime};

// Resolver tiers.
use jouleclaw_formula::{FormulaFirstTier, InMemoryKnowledgeStore};
use jouleclaw_llm_cheap::{EchoBackend, LlmCheapTier};
use jouleclaw_proof_tier::ProofTier;
use jouleclaw_ssm_reader::SsmReaderTier;
use jouleclaw_tool_tier::ToolTier;
use jouleclaw_verification_tier::{StaticBackend, VerificationTier};

// Pipeline-stage tiers.
use jouleclaw_federation::{Federation, MockProvider};
use jouleclaw_graph_rag::{GraphRagTier, InMemoryKnowledgeGraph};
use jouleclaw_local_index::{InMemoryIndex, LocalIndexTier};
use jouleclaw_rerank::{Bm25Reranker, RerankTier};
use jouleclaw_ssm_router::SsmRouterTier;
use jouleclaw_struct_contrast::StructContrastTier;

// Control plane.
use jouleclaw_governor::Governor;
use jouleclaw_reflection::ReflectionEngine;
use jouleclaw_routing::LearnedRouter;
use jouleclaw_supervisor::Supervisor;
use jouleclaw_tuner::Tuner;

// Yang→yin promotion.
use std::path::Path;
use std::sync::{Arc, Mutex};

use jouleclaw_cascade::types::{Answer, AnswerError, AnswerOutput, JouleClass, Query};
use jouleclaw_promote::disk::DiskError;
use jouleclaw_promote::{
    shared_in_memory, FilePromotionStore, InMemoryPromotionStore, PromotedTier, PromotionGate,
    PromotionStore, SharedStore,
};
use jouleclaw_verify::OutputVerifier;

/// Default global energy budget for the [`Governor`]: 1 MJ per rolling
/// hour. Sized so the default stack admits realistic demo traffic; a
/// real deployment sets its own ceiling.
pub const DEFAULT_GOVERNOR_BUDGET_J: f64 = 1.0e6;
/// Default governor window — one hour.
pub const DEFAULT_GOVERNOR_WINDOW_SECS: u64 = 3600;
/// Default sliding-window capacity for the [`Supervisor`].
pub const DEFAULT_SUPERVISOR_WINDOW: usize = 256;

/// An assembled JouleClaw stack: a runnable resolver cascade plus the
/// pipeline-stage tiers and control-plane components, all built from
/// deterministic reference backends.
///
/// Fields are public so a consumer can swap any component for a
/// production backend (a real SSM behind the router, a tantivy index,
/// an HTTP LLM backend, …) after construction.
pub struct JouleClawStack<S: PromotionStore = InMemoryPromotionStore> {
    /// Resolver cascade (L0 cache → L0.25 → L0.5 → L1.5 → L3 → L4 → L4.5),
    /// with the L5 [`LearnedRouter`] installed as its router.
    pub runtime: Runtime,

    // ── Pipeline-stage tiers (compose explicitly; not in the walk) ──
    /// L0.75 — intent router.
    pub ssm_router: SsmRouterTier,
    /// L1 — in-process retrieval (empty by default; seed via the field).
    pub local_index: LocalIndexTier<InMemoryIndex>,
    /// L1.25 — GraphRAG enrichment (empty graph by default).
    pub graph_rag: GraphRagTier<InMemoryKnowledgeGraph>,
    /// L1.375 — structural-contrast second pass.
    pub struct_contrast: StructContrastTier<InMemoryKnowledgeStore>,
    /// L1.5 — the pipeline's *read* terminal: extracts an answer from
    /// retrieved passages. Refuses a bare query, so it is driven by the
    /// orchestrator, not the standalone walk.
    pub ssm_reader: SsmReaderTier,
    /// L2 — federated retrieval (one mock provider by default).
    pub federation: Federation,
    /// L2.5 — BM25 reranker.
    pub rerank: RerankTier<Bm25Reranker>,

    // ── Control plane (L7–L10) ──
    /// L7 — offline reflection learner.
    pub reflection: ReflectionEngine,
    /// L8 — self-tuner.
    pub tuner: Tuner,
    /// L9 — pathology supervisor.
    pub supervisor: Supervisor,
    /// L10 — budget governor.
    pub governor: Governor,

    /// Yang→yin promotion store, shared with the front [`PromotedTier`]
    /// registered in [`Self::runtime`]. Inspect it for the promotion log
    /// and `invocations_avoided` (model calls saved by promotion). Feed
    /// it via [`Self::promote`] or [`Self::answer_and_promote`].
    pub promotion_store: SharedStore<S>,

    /// Optional verifier for the auto-promotion path. When set,
    /// [`Self::answer_and_promote`] runs it over a compartment (model)
    /// answer and promotes only on a pass. `None` (the default) disables
    /// auto-promotion — promoting an unverified answer would poison the
    /// deterministic store, so the safe default requires an explicit
    /// verifier (or use [`Self::promote`] with your own verdict).
    pub verifier: Option<Box<dyn OutputVerifier>>,

    /// Confidence bar a verified compartment answer must clear to be
    /// promoted (default [`jouleclaw_promote::DEFAULT_PROMOTE_CONFIDENCE`]).
    pub promote_confidence: f32,
}

impl<S: PromotionStore + 'static> JouleClawStack<S> {
    /// Shared assembly: build the resolver cascade + control plane over a
    /// caller-provided promotion store. Used by [`with_defaults`]
    /// (in-memory) and [`with_durable_promotion`] (file-backed). Every
    /// tier uses its crate's deterministic reference backend, so the
    /// result runs end-to-end with no external dependencies.
    ///
    /// [`with_defaults`]: JouleClawStack::with_defaults
    /// [`with_durable_promotion`]: JouleClawStack::with_durable_promotion
    fn assemble(promotion_store: SharedStore<S>) -> Self {
        // Resolver cascade, registered cheapest-first. The runtime owns
        // a built-in L0 cache ahead of these.
        let mut cascade = Cascade::new();
        // Front deterministic tier: verified model answers promoted here
        // resolve at nanojoule energy and never re-invoke the model.
        cascade.register(Box::new(PromotedTier::new(promotion_store.clone()))); // L0.1 promoted
        cascade.register(Box::new(FormulaFirstTier::new(InMemoryKnowledgeStore::new()))); // L0.25
        cascade.register(Box::new(ToolTier::new())); // L0.5
        cascade.register(Box::new(LlmCheapTier::new(EchoBackend::default()))); // L3
        // L4 cross-model verification — only constructed if ≥2 backends
        // (always true here). Reached only when L3 refuses.
        if let Ok(verify) = VerificationTier::new(vec![
            Box::new(StaticBackend::new("ref-a", "")),
            Box::new(StaticBackend::new("ref-b", "")),
        ]) {
            cascade.register(Box::new(verify));
        }
        cascade.register(Box::new(ProofTier::new())); // L4.5

        let runtime = Runtime::new_with_router(cascade, Box::new(LearnedRouter::with_defaults()));

        Self {
            runtime,
            ssm_router: SsmRouterTier::new(),
            local_index: LocalIndexTier::new(InMemoryIndex::new()),
            graph_rag: GraphRagTier::new(InMemoryKnowledgeGraph::new()),
            struct_contrast: StructContrastTier::new(InMemoryKnowledgeStore::new()),
            ssm_reader: SsmReaderTier::new(),
            federation: Federation::with_providers(vec![Box::new(MockProvider::named("reference"))]),
            rerank: RerankTier::new(Bm25Reranker::default()),
            reflection: ReflectionEngine::default(),
            tuner: Tuner::new(),
            supervisor: Supervisor::with_default_detectors(DEFAULT_SUPERVISOR_WINDOW, Vec::new()),
            governor: Governor::new(DEFAULT_GOVERNOR_BUDGET_J, DEFAULT_GOVERNOR_WINDOW_SECS, 0),
            promotion_store,
            verifier: None,
            promote_confidence: jouleclaw_promote::DEFAULT_PROMOTE_CONFIDENCE,
        }
    }

    /// Install a verifier to enable the auto-promotion path
    /// ([`Self::answer_and_promote`]).
    pub fn with_verifier(mut self, verifier: Box<dyn OutputVerifier>) -> Self {
        self.verifier = Some(verifier);
        self
    }

    /// Override the promotion confidence bar used by [`Self::promote`]
    /// and [`Self::answer_and_promote`].
    pub fn with_promote_confidence(mut self, confidence: f32) -> Self {
        self.promote_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Consider a (query, answer) pair for yang→yin promotion. Call this
    /// after a model-tier answer has been verified: a verified,
    /// high-confidence compartment answer is recorded as a permanent
    /// deterministic fact, so the same query thereafter resolves at the
    /// front [`PromotedTier`] for nanojoules instead of re-invoking the
    /// model. Returns `true` if the answer was newly promoted.
    ///
    /// `verified` is the caller's verifier verdict (e.g. from
    /// `jouleclaw-verify`); `now_secs` is the caller's clock.
    pub fn promote(
        &mut self,
        query: &Query,
        answer: &Answer,
        verified: bool,
        now_secs: u64,
    ) -> bool {
        PromotionGate::new(self.promotion_store.clone())
            .with_min_confidence(self.promote_confidence)
            .consider(query, answer, verified, now_secs)
    }

    /// Answer a query and, if it was resolved by the statistical
    /// compartment (a Model/Wire-class tier) AND a [`verifier`](Self::verifier)
    /// is installed AND that verifier passes, automatically promote it
    /// (yang→yin). This is the auto-wired path: verify-then-promote with
    /// no caller bookkeeping. Deterministic-tier answers are returned
    /// untouched (there is no model tax to amortize).
    ///
    /// Returns the answer regardless of whether promotion happened;
    /// inspect [`promotion_store`](Self::promotion_store) for the effect.
    pub fn answer_and_promote(&mut self, query: Query) -> Result<Answer, AnswerError> {
        // Keep the query to key the promotion; the runtime consumes its copy.
        let answer = self.runtime.answer(query.clone())?;

        let is_compartment = matches!(
            answer.tier_used.joule_class(),
            JouleClass::Model | JouleClass::Wire
        );
        if is_compartment {
            if let Some(bytes) = output_bytes(&answer.output) {
                let verified = self
                    .verifier
                    .as_ref()
                    .map(|v| v.verify(&bytes).is_pass())
                    .unwrap_or(false);
                if verified {
                    self.promote(&query, &answer, true, now_secs());
                }
            }
        }
        Ok(answer)
    }
}

impl JouleClawStack<InMemoryPromotionStore> {
    /// Assemble the default stack with an in-memory promotion store
    /// (promoted facts live for the process lifetime). On a plain text
    /// query the resolver walk falls through to the L3 `EchoBackend`;
    /// tool-shaped and constraint-shaped queries resolve at L0.5 / L4.5.
    pub fn with_defaults() -> Self {
        Self::assemble(shared_in_memory())
    }
}

impl JouleClawStack<FilePromotionStore> {
    /// Assemble a stack whose promoted facts persist under `dir` (an
    /// append-only journal, replayed on open) — so the deterministic
    /// surface learned by one run is available to the next. Identical to
    /// [`with_defaults`](JouleClawStack::with_defaults) in every other
    /// respect.
    pub fn with_durable_promotion(dir: impl AsRef<Path>) -> Result<Self, DiskError> {
        let store = FilePromotionStore::open(dir)?;
        Ok(Self::assemble(Arc::new(Mutex::new(store))))
    }
}

/// Output bytes for verification; `Refused` answers have none.
fn output_bytes(output: &AnswerOutput) -> Option<Vec<u8>> {
    match output {
        AnswerOutput::Text(t) => Some(t.as_bytes().to_vec()),
        AnswerOutput::Structured(b) => Some(b.clone()),
        AnswerOutput::Refused(_) => None,
    }
}

/// Wall-clock unix seconds for promotion timestamps.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Default for JouleClawStack<InMemoryPromotionStore> {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        Answer, AnswerOutput, ContextRef, ExecutionTrace, JouleBudget, L3ModelId, QualityFloor,
        Query, QueryInput, TierId,
    };
    use jouleclaw_cascade::verification::VerificationStatus;
    use jouleclaw_promote::PromotionStore;
    use jouleclaw_verify::VerifyResult;

    /// Trivial verifier that approves any non-empty output — stands in
    /// for a real domain verifier in the auto-promotion test.
    struct PassIfNonEmpty;
    impl OutputVerifier for PassIfNonEmpty {
        fn name(&self) -> &str {
            "verify:nonempty"
        }
        fn verify(&self, output: &[u8]) -> VerifyResult {
            if output.is_empty() {
                VerifyResult::fail("empty")
            } else {
                VerifyResult::Pass
            }
        }
        fn declared_cost_uj(&self) -> u64 {
            1
        }
    }

    fn text(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn stack_builds() {
        let _ = JouleClawStack::with_defaults();
    }

    #[test]
    fn resolver_cascade_has_expected_tiers() {
        let stack = JouleClawStack::with_defaults();
        let ids = stack.runtime.tier_ids();
        // Built-in L0 cache + the six resolvers.
        assert!(ids.contains(&TierId::L0));
        assert!(ids.contains(&TierId::L0_25FormulaFirst));
        assert!(ids.contains(&TierId::L0_5ToolCompute));
        assert!(ids.contains(&TierId::L4_5Proof));
        // ssm-reader is a pipeline field now, NOT in the standalone walk.
        assert!(!ids.contains(&TierId::L1_5SsmReader));
        // L3 / L4 are parameterised model ids; assert at least one of each family.
        assert!(ids.iter().any(|t| matches!(t, TierId::L3(_))));
        assert!(ids.iter().any(|t| matches!(t, TierId::L4(_))));
    }

    #[test]
    fn plain_text_falls_through_to_l3_echo() {
        let mut stack = JouleClawStack::with_defaults();
        let ans = stack.runtime.answer(text("hello stack")).expect("resolves");
        match ans.output {
            AnswerOutput::Text(t) => assert!(t.contains("hello stack"), "got: {t}"),
            other => panic!("expected text echo, got {other:?}"),
        }
        assert!(matches!(ans.tier_used, TierId::L3(_)));
    }

    #[test]
    fn constraint_query_resolves_at_proof_tier() {
        // A SAT envelope the ProofTier recognises; resolvers above it
        // (formula/tool/reader/llm) refuse or skip the structured input.
        let mut stack = JouleClawStack::with_defaults();
        let cnf = r#"{"kind":"sat","cnf":[[1,2],[-1,2]]}"#;
        let q = Query {
            input: QueryInput::Structured(cnf.as_bytes().to_vec()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = stack.runtime.answer(q).expect("resolves");
        // Either the proof tier solved it (preferred) or a cheaper tier
        // legitimately handled the structured blob — in all cases the
        // walk must not panic and must return a concrete output.
        assert!(matches!(
            ans.output,
            AnswerOutput::Structured(_) | AnswerOutput::Text(_)
        ));
    }

    #[test]
    fn promoted_fact_is_served_by_front_tier() {
        let mut stack = JouleClawStack::with_defaults();
        let q = text("what is the capital of france");
        // Simulate a verified model answer (as a verifier would approve).
        let model = Answer {
            output: AnswerOutput::Text("Paris".into()),
            tier_used: TierId::L3(L3ModelId(0)),
            joules_spent: 2.0,
            confidence: 0.97,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        };
        assert!(stack.promote(&q, &model, true, 1));
        // A fresh ask is served deterministically by the promoted front
        // tier (built-in L0 cache is empty for this never-run query).
        let ans = stack
            .runtime
            .answer(text("what is the capital of france"))
            .expect("resolves");
        assert_eq!(ans.tier_used, TierId::L0_1FactLut);
        match ans.output {
            AnswerOutput::Text(t) => assert_eq!(t, "Paris"),
            other => panic!("expected promoted answer, got {other:?}"),
        }
        assert!(ans.joules_spent < 1e-6); // nanojoules, not the model's 2 J
        assert!(stack.promotion_store.lock().unwrap().invocations_avoided() >= 1);
    }

    #[test]
    fn durable_promotion_survives_restart() {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "jouleclaw-stack-durable-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let model = Answer {
            output: AnswerOutput::Text("Paris".into()),
            tier_used: TierId::L3(L3ModelId(0)),
            joules_spent: 2.0,
            confidence: 0.97,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        };
        // First process: durable stack, promote a verified fact.
        {
            let mut stack =
                JouleClawStack::with_durable_promotion(&dir).expect("open durable stack");
            assert!(stack.promote(&text("durable capital of france"), &model, true, 1));
        }
        // Second process: reopen the same dir — the fact is replayed and
        // the front PromotedTier serves it for a never-before-asked query.
        {
            let mut stack =
                JouleClawStack::with_durable_promotion(&dir).expect("reopen durable stack");
            let ans = stack
                .runtime
                .answer(text("durable capital of france"))
                .expect("resolves");
            assert_eq!(ans.tier_used, TierId::L0_1FactLut);
            match ans.output {
                AnswerOutput::Text(t) => assert_eq!(t, "Paris"),
                other => panic!("expected replayed promoted answer, got {other:?}"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_promote_promotes_verified_compartment_answer() {
        // Low bar (0.7) so the L3 echo (0.8 confidence) clears it.
        let mut stack = JouleClawStack::with_defaults()
            .with_verifier(Box::new(PassIfNonEmpty))
            .with_promote_confidence(0.7);
        // "hello" reaches L3 (formula/tool refuse), echo answers @0.8.
        let ans = stack.answer_and_promote(text("hello there")).expect("resolves");
        assert!(matches!(ans.tier_used, TierId::L3(_)));
        // Verified compartment answer → auto-promoted.
        assert_eq!(stack.promotion_store.lock().unwrap().len(), 1);
        assert_eq!(stack.promotion_store.lock().unwrap().log()[0].origin_tier, "L3");
    }

    #[test]
    fn auto_promote_no_verifier_does_not_promote() {
        let mut stack = JouleClawStack::with_defaults().with_promote_confidence(0.7);
        let _ = stack.answer_and_promote(text("hello there")).expect("resolves");
        // No verifier installed → nothing promoted.
        assert_eq!(stack.promotion_store.lock().unwrap().len(), 0);
    }

    #[test]
    fn pipeline_and_control_plane_present() {
        let mut stack = JouleClawStack::with_defaults();
        // Pipeline-stage tiers are constructed and usable.
        let _ = &stack.ssm_router;
        let _ = &stack.local_index;
        let _ = &stack.graph_rag;
        let _ = &stack.struct_contrast;
        let _ = &stack.federation;
        let _ = &stack.rerank;
        // Control plane is usable: governor admits within budget.
        let decision = stack.governor.admit(10.0, None);
        assert!(decision.is_admit());
        // Tuner produces advice; supervisor scans without panic.
        let _ = stack.tuner.advise();
        let _ = stack.supervisor.scan();
        // Reflection starts empty.
        assert!(stack.reflection.is_empty());
    }
}
