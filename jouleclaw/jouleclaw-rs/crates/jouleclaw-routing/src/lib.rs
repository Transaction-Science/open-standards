//! L5 — learned routing (meta-cognitive control plane).
//!
//! The cascade walker (`jouleclaw_cascade::Runtime`) walks tiers in
//! registration order unless a [`jouleclaw_cascade::Router`] supplies a
//! dispatch order. L5 is that router, with a memory: it records which
//! tier *won* for each query and, on the next query, orders tiers by
//! what worked for similar queries before.
//!
//! This is the cheapest meta-tier — picking an order costs a handful of
//! token-set comparisons, not a model call. The whole point of the
//! cascade is that the *first* tier to fire should usually be the one
//! that resolves the query; L5 learns which tier that is per query
//! class so the expensive tiers are only reached when the cheap ones
//! genuinely cannot answer.
//!
//! ## Determinism
//!
//! Identical router state + identical query → identical plan. The
//! similarity metric is token-set Jaccard (no floats in the ordering
//! key beyond the win-rate tally), so plans are reproducible across
//! runs and platforms.
//!
//! ## Two implementations
//!
//! - [`LearnedRouter`] — token-Jaccard nearest-episode voting. The
//!   default; works with raw query text.
//! - [`PhasorRouter`] — hashes the query into a fixed-width "phasor"
//!   fingerprint and votes by Hamming proximity. A consumer can supply
//!   real phasor embeddings later via [`PhasorEmbedder`]; the default
//!   uses a deterministic FNV-derived fingerprint so the crate is
//!   self-contained.

#![forbid(unsafe_code)]

mod learned;
mod phasor;

pub use learned::{Episode, LearnedRouter};
pub use phasor::{Phasor, PhasorEmbedder, PhasorRouter, HashPhasorEmbedder};

/// The (negligible) joule cost charged for producing a routing plan.
/// L5 lives in the meta-cognitive control plane; its energy is client
/// CPU only and is dwarfed by every execution tier it orders.
pub const ROUTING_JOULES: f64 = 8e-9;
