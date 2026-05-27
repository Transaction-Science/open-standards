//! # jouleclaw-formula
//!
//! L0.25 of the JouleClaw cascade — **formula-first** structural-relationship
//! resolution.
//!
//! The formula tier runs immediately after the L0 cache and before everything
//! downstream. It extracts entity candidates from the raw query text,
//! resolves them in a [`KnowledgeStore`], and computes structural
//! relationships via the contrast formula `R(A, B) = cos(E(A − μ), E(B − μ))`.
//! When the formula resolves with high confidence the cascade skips every
//! later tier — the backbone of the deterministic-first doctrine.
//!
//! ## Doctrine
//!
//! - **Inference is the fallback, not the primary intelligence.** Over time,
//!   as the knowledge store grows, this tier should resolve an increasing
//!   share of queries (the "promotion curve").
//! - **Fresh retrieval beats frozen weights.** The formula consults a live
//!   store; it does not memorise.
//! - **Deterministic-first, then estimator-bounded.** Energy is reported as
//!   [`Provenance::Estimator`] because this tier has no hardware shunt of
//!   its own; the runtime's outer energy meter measures the actual spend.
//!
//! ## Wiring it up
//!
//! ```rust,no_run
//! use jouleclaw_cascade::tier::{Cascade, Runtime};
//! use jouleclaw_cascade::types::{
//!     ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
//! };
//! use jouleclaw_formula::{Concept, FormulaFirstTier, InMemoryKnowledgeStore};
//!
//! let mut store = InMemoryKnowledgeStore::new();
//! store.insert(Concept {
//!     id: "urn:fire".into(),
//!     name: "fire".into(),
//!     traits: vec![1.0, -1.0],
//! });
//! store.insert(Concept {
//!     id: "urn:water".into(),
//!     name: "water".into(),
//!     traits: vec![-1.0, 1.0],
//! });
//!
//! let mut cascade = Cascade::new();
//! cascade.register(Box::new(FormulaFirstTier::new(store)));
//! let mut runtime = Runtime::new(cascade);
//!
//! let _ = runtime.answer(Query {
//!     input: QueryInput::Text("compare fire and water".into()),
//!     budget: JouleBudget::cheap(),
//!     quality: QualityFloor::chat(),
//!     context: ContextRef::fresh(),
//!     deadline: None,
//! });
//! ```
//!
//! ## What was ported
//!
//! Ported from `joulesperbit/crates/verity-cascade/src/layers/l025_formula_first.rs`
//! (~833 LOC). The donor's algorithms — n-gram entity extraction, the
//! structural-vs-factual query classifier, the entity-focused query heuristic,
//! and the NCD zero-shot fallback shape — are preserved verbatim.
//!
//! What was inlined or stubbed:
//!
//! - **`KnowledgeStore`** — defined locally as a portable trait; the donor's
//!   `verity-contrast` impl can adapt itself to it.
//! - **`FormulaHit`** — a small local struct replacing the donor's
//!   `verity_federation::fuser::FusedResult`.
//! - **NCD** — `crate::mdl::concept_ncd` (zstd-backed) is replaced by a
//!   trigram-Jaccard approximation in [`ncd`]. Preserves ordering, not
//!   absolute calibration.
//! - **SNN path** — the donor's `execute_snn` is NOT ported. SNN stores are
//!   OpenIE biological-trait IP and excluded from the JouleClaw open
//!   standard.
//! - **`PowerSampler`** — the donor's joule sampling is replaced by the
//!   `Tier::try_answer` contract's reported `joules_spent`; the runtime
//!   does the outer measurement via `jouleclaw-energy`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod extract;
pub mod knowledge;
pub mod ncd;
pub mod tier;

pub use extract::{
    extract_entities, is_stop_word, is_structural_query,
    query_is_about_entity, Extraction,
};
pub use knowledge::{
    Concept, ContrastDimension, ContrastMap, ContrastRelation,
    InMemoryKnowledgeStore, KnowledgeStore, Similarity,
};
pub use ncd::concept_ncd;
pub use tier::{
    structured_contrasts, FormulaError, FormulaFirstTier, FormulaHit,
    FormulaSidecar, FormulaSidecarSink, StructuredContrast,
    FORMULA_CONFIDENCE_FLOOR, FORMULA_JOULES, FORMULA_LATENCY,
};
