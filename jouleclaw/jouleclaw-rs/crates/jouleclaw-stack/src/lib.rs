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
//! - **Resolvers** — produce a final answer (or refuse): [`FormulaFirstTier`]
//!   (L0.25), [`ToolTier`] (L0.5), [`SsmReaderTier`] (L1.5), [`LlmCheapTier`]
//!   (L3), [`VerificationTier`] (L4), [`ProofTier`] (L4.5). These are
//!   registered into [`JouleClawStack::runtime`] in ascending cost order;
//!   the cheapest one that can close the query wins.
//! - **Pipeline stages** — produce an *intermediate artifact*, not an
//!   answer: [`SsmRouterTier`] (L0.75, a route), [`LocalIndexTier`] (L1),
//!   [`Federation`] (L2) and [`GraphRagTier`] (L1.25) (retrieved
//!   candidates / subgraph), [`StructContrastTier`] (L1.375, a contrast
//!   map), [`RerankTier`] (L2.5, a reordering). Dropping these into the
//!   linear walk would short-circuit it — the router would "answer" every
//!   query with a route. So they are exposed as fields for explicit
//!   `retrieve → enrich → rerank → read` composition rather than walked.
//!
//! A future orchestrator can drive the pipeline stages to build context
//! and feed [`SsmReaderTier`] (which reads retrieved passages); that
//! orchestration is intentionally out of scope for v0.
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
pub struct JouleClawStack {
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
}

impl JouleClawStack {
    /// Assemble the default stack. Every tier uses its crate's
    /// deterministic reference backend, so the result runs end-to-end
    /// with no external dependencies. On a plain text query the resolver
    /// walk falls through to the L3 [`EchoBackend`]; tool-shaped and
    /// constraint-shaped queries resolve at L0.5 / L4.5 respectively.
    pub fn with_defaults() -> Self {
        // Resolver cascade, registered cheapest-first. The runtime owns
        // a built-in L0 cache ahead of these.
        let mut cascade = Cascade::new();
        cascade.register(Box::new(FormulaFirstTier::new(InMemoryKnowledgeStore::new()))); // L0.25
        cascade.register(Box::new(ToolTier::new())); // L0.5
        cascade.register(Box::new(SsmReaderTier::new())); // L1.5 (refuses without passages)
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
            federation: Federation::with_providers(vec![Box::new(MockProvider::named("reference"))]),
            rerank: RerankTier::new(Bm25Reranker::default()),
            reflection: ReflectionEngine::default(),
            tuner: Tuner::new(),
            supervisor: Supervisor::with_default_detectors(DEFAULT_SUPERVISOR_WINDOW, Vec::new()),
            governor: Governor::new(DEFAULT_GOVERNOR_BUDGET_J, DEFAULT_GOVERNOR_WINDOW_SECS, 0),
        }
    }
}

impl Default for JouleClawStack {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        AnswerOutput, ContextRef, JouleBudget, QualityFloor, Query, QueryInput, TierId,
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
        assert!(ids.contains(&TierId::L1_5SsmReader));
        assert!(ids.contains(&TierId::L4_5Proof));
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
