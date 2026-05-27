//! JouleClaw L1 — local-index retrieval tier.
//!
//! Client-energy-only retrieval against a local document corpus. Sits
//! immediately below the L2 federation / L3 model tiers in the cascade
//! and answers any factual query whose ground truth already lives on
//! the device. Wire spend is zero by construction; the only joules
//! charged to the budget are those drawn by the local index pipeline.
//!
//! ## Why a tier?
//!
//! Cascading is "inference is the last resort, fresh retrieval beats
//! frozen weights" all the way down. The L1 LocalIndex tier exists so
//! that any query the operator's own corpus can satisfy — service
//! manuals, internal wikis, ingest-time documentation, prior agent
//! receipts — never escalates to a federated search or an LLM. The
//! donor `verity-cascade::layers::l1_index` measured this path at
//! ~890 µJ end-to-end on its tantivy reference impl; JouleClaw inherits
//! the envelope.
//!
//! ## Standard surface
//!
//! This crate is **not** a search-engine implementation. It is the
//! *tier adapter* that wires a consumer-supplied [`LocalIndex`] into
//! the JouleClaw cascade. Consumers implement [`LocalIndex`] over
//! whatever backend they like — tantivy, sled, embedded-postgres,
//! a SQLite FTS5 file, a memory-mapped FST — and hand the implementer
//! to [`LocalIndexTier`].
//!
//! For reference / smoke tests / Pi-class deployments where bringing
//! tantivy into the binary is unwanted, the crate ships a default
//! [`InMemoryIndex`]: pure-Rust BM25-shaped scoring over a `Vec<Document>`
//! held in process memory. No external deps, no model weights,
//! microjoule-class per query. Production deployments should swap to a
//! durable index (tantivy/sled/etc.) via the trait.
//!
//! ## Query envelope
//!
//! The tier consumes [`jouleclaw_cascade::types::QueryInput::Text`]
//! directly — the query string IS the input. The structured response
//! shape is:
//!
//! ```json
//! {
//!   "hits": [
//!     { "doc_id": "doc-7", "text": "…matching passage…", "score": 4.213 },
//!     …
//!   ],
//!   "k": 10
//! }
//! ```

#![forbid(unsafe_code)]

pub mod index;
pub mod inmem;
pub mod tier;

pub use index::{IndexError, IndexHit, LocalIndex};
pub use inmem::{Document, InMemoryIndex, InMemoryParams};
pub use tier::{
    LocalIndexOutput, LocalIndexTier, DEFAULT_K, LOCAL_INDEX_CONFIDENCE_FLOOR,
    LOCAL_INDEX_JOULES, LOCAL_INDEX_LATENCY,
};
