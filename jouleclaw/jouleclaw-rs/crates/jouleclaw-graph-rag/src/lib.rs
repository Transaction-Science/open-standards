//! # jouleclaw-graph-rag
//!
//! L1.25 of the JouleClaw cascade — **GraphRAG** deterministic entity
//! extraction and knowledge-graph enrichment.
//!
//! Sits between L0.75 (`SsmRouter`) and L1.375 (`StructContrast`). The
//! L1.25 tier runs in the ~500 µJ envelope: it extracts entity
//! candidates from the raw query text, resolves them against a
//! consumer-supplied [`KnowledgeGraph`], collects their immediate
//! neighbourhood, and emits a structured sub-graph that downstream
//! tiers consume to build richer prompts without having to re-query
//! the store.
//!
//! ## Doctrine
//!
//! - **Inference is the last resort.** L1.25 has no model — only
//!   pattern-matched extraction plus a lookup. If it can supply enough
//!   context to push the SSM reader above its confidence threshold,
//!   the cascade skips remote frontier calls entirely.
//! - **Fresh retrieval beats frozen weights.** The graph is consulted
//!   live; nothing is memorised in tier state.
//! - **Honest provenance.** Energy is reported as
//!   [`Provenance::Estimator`] — this tier has no hardware shunt and
//!   no claim to anything tighter.
//!
//! ## Wiring it up
//!
//! ```rust,no_run
//! use jouleclaw_cascade::tier::{Cascade, Runtime};
//! use jouleclaw_cascade::types::{
//!     ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
//! };
//! use jouleclaw_graph_rag::{
//!     Edge, Entity, EntityId, GraphRagTier, InMemoryKnowledgeGraph,
//! };
//!
//! let mut g = InMemoryKnowledgeGraph::new();
//! g.insert_entity(Entity {
//!     id: EntityId::new("urn:rust"),
//!     name: "Rust".into(),
//!     kind: "Language".into(),
//!     description: "systems language".into(),
//! });
//! g.insert_entity(Entity {
//!     id: EntityId::new("urn:cargo"),
//!     name: "Cargo".into(),
//!     kind: "Tool".into(),
//!     description: "package manager".into(),
//! });
//! g.insert_edge(Edge {
//!     from: EntityId::new("urn:rust"),
//!     to: EntityId::new("urn:cargo"),
//!     label: "ships_with".into(),
//!     weight: 1.0,
//! });
//!
//! let mut cascade = Cascade::new();
//! cascade.register(Box::new(GraphRagTier::new(g)));
//! let mut runtime = Runtime::new(cascade);
//!
//! let _ = runtime.answer(Query {
//!     input: QueryInput::Text("Rust and Cargo together".into()),
//!     budget: JouleBudget::cheap(),
//!     quality: QualityFloor::chat(),
//!     context: ContextRef::fresh(),
//!     deadline: None,
//! });
//! ```
//!
//! ## What was ported
//!
//! Ported from
//! `joulesperbit/crates/verity-cascade/src/layers/l125_graph_rag.rs`
//! (~400 LOC). The donor's three-pattern extractor (proper-noun
//! sequences, CamelCase / snake_case / kebab-case technical terms, and
//! quantity-with-unit pairs) is preserved verbatim. The donor's
//! confidence map is preserved in spirit: more entities + more edges
//! → higher confidence.
//!
//! What was inlined or replaced:
//!
//! - **`FusedResult` snippets** — the donor's co-occurrence graph was
//!   built from snippet overlap. JouleClaw replaces snippets with a
//!   trait-shaped [`KnowledgeGraph`] so consumers plug their own store.
//! - **`PowerSampler`** — replaced by the `Tier::try_answer` contract's
//!   reported `joules_spent`; the runtime's outer energy meter
//!   measures actual spend.
//! - **`blake3` content hashing** — not needed in the JouleClaw port;
//!   the cascade's L0 cache already handles content-addressing.
//!
//! [`Provenance::Estimator`]: jouleclaw_energy::Provenance::Estimator
//! [`KnowledgeGraph`]: graph::KnowledgeGraph

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod extract;
pub mod graph;
pub mod tier;

pub use extract::{
    extract_entity_candidates, has_mixed_case, is_all_upper, is_capitalized,
    is_stopword, EntityCandidate, EntityClass,
};
pub use graph::{
    Edge, Entity, EntityId, InMemoryKnowledgeGraph, KnowledgeGraph,
};
pub use tier::{
    GraphRagEntity, GraphRagError, GraphRagOutput, GraphRagTier,
    DEFAULT_NEIGHBORHOOD_DEPTH, GRAPH_RAG_CONFIDENCE_FLOOR,
    GRAPH_RAG_JOULES, GRAPH_RAG_LATENCY, MAX_EDGES_OUT, MAX_ENTITIES_OUT,
};
