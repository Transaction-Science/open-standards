//! # jouleclaw-struct-contrast
//!
//! **L1.375** of the JouleClaw cascade — the *second* structural-contrast
//! formula pass, run after L1.25 GraphRag has enriched the query with
//! graph-extracted entity names.
//!
//! Where [`jouleclaw_formula`] (L0.25) takes raw query text and runs its
//! own n-gram entity extraction, **L1.375** consumes an already-curated
//! list of entities (the L1.25 output) and re-runs the structural-contrast
//! formula `R(A, B) = cos(E(A − μ), E(B − μ))` over that richer context.
//! The per-dimension breakdown is exposed verbatim — every dimension is
//! classified as `Align` / `Oppose` / `Partial` / `Unknown` (axiom 4 in
//! the donor `verity-cascade`).
//!
//! ## Doctrine
//!
//! - **Inference is the fallback, not the primary intelligence.** L1.375
//!   only fires when L1.25 has produced a graph-enriched context. If the
//!   query carries no such payload, the tier reports
//!   [`Tier::estimate_cost`] = `None` and the cascade walks past it.
//! - **Per-dimension transparency.** The donor's "contrast map" — the
//!   classification of every dimension as Align / Oppose / Partial /
//!   Unknown — is the load-bearing output. Downstream tiers (SSM reader,
//!   model, wire) use it to know what to *seek*, not just what was found.
//! - **Estimator provenance.** Same as L0.25: no hardware shunt, so
//!   reported energy carries [`Provenance::Estimator`]. The runtime's
//!   outer meter records the actual spend.
//!
//! ## Payload contract
//!
//! L1.25 emits a JSON-encoded [`StructContrastInput`] inside
//! `QueryInput::Structured(bytes)`:
//!
//! ```json
//! {
//!   "kind": "jouleclaw.struct_contrast/v1",
//!   "query": "compare fire and water",
//!   "entities": ["fire", "water", "steam"]
//! }
//! ```
//!
//! Any other shape — text-only queries, binary, multimodal, or structured
//! payloads with a different `kind` — is `Inapplicable`. The cascade
//! continues to the next tier.
//!
//! ## Wiring it up
//!
//! ```rust,no_run
//! use jouleclaw_cascade::tier::Cascade;
//! use jouleclaw_formula::{Concept, InMemoryKnowledgeStore};
//! use jouleclaw_struct_contrast::StructContrastTier;
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
//! cascade.register(Box::new(StructContrastTier::new(store)));
//! ```
//!
//! ## What was ported
//!
//! Ported from `joulesperbit/crates/verity-cascade/src/layers/l1375_structural_contrast.rs`
//! (~201 LOC). The donor's algorithm — entity lookup, pairwise contrast,
//! coverage-driven confidence, per-dimension breakdown — is preserved.
//!
//! What was inlined or remapped:
//!
//! - **`KnowledgeStore` + `ContrastMap` + `ContrastRelation`** — reused from
//!   `jouleclaw-formula` rather than redefined.
//! - **Entity-name extraction** — removed. L1.25 supplies the names; the
//!   donor's whitespace + "- " GraphRAG-snippet parsing is no longer this
//!   tier's responsibility.
//! - **`FusedResult` + `PowerSampler`** — gone. Output is JouleClaw's
//!   `AnswerOutput::Structured(serde_json bytes)` plus a human-readable
//!   text summary; energy spend is reported through the `Tier` contract.
//! - **`blake3` dependency** — dropped. The donor used it to seed a
//!   content hash on the synthetic `FusedResult`; L1.375 emits structured
//!   JSON, the caller hashes that if needed.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod context;
pub mod tier;

pub use context::{StructContrastInput, STRUCT_CONTRAST_KIND};
pub use tier::{
    ContrastVerdict, DimensionVerdict, PairContrast, StructContrastError,
    StructContrastSidecar, StructContrastSidecarSink, StructContrastTier,
    CONTRAST_CONFIDENCE_FLOOR, CONTRAST_JOULES, CONTRAST_LATENCY,
};
