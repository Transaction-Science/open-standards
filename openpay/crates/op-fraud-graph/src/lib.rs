//! # `op-fraud-graph` — graph-based fraud detection
//!
//! Sibling to [`op-fraud`](../op_fraud/index.html) (per-transaction scoring)
//! and [`op-screening`](../op_screening/index.html) (sanctions / PEP). Where
//! `op-fraud` answers *"is this one payment risky?"* by feature-vectoring a
//! single attempt, this crate answers *"is this entity (or this cluster of
//! entities) part of a fraud ring?"* by treating identifiers as vertices in
//! a graph and looking at the *structure* of their connections.
//!
//! ## Design
//!
//! Five layers:
//!
//! 1. **Ingest & entity resolution** ([`entity`], [`bloom`]) — Raw payment
//!    identifiers (PAN-hash, email-hash, device fingerprint, IP, address,
//!    phone, BIN+last4) become [`Entity`] vertices. A [`bloom::EntityBloom`]
//!    filter accelerates "have we seen this fingerprint?" lookups before
//!    paying the cost of a graph mutation.
//!
//! 2. **Graph** ([`graph`], [`edge`]) — [`FraudGraph`] is an adjacency-list
//!    representation. Edges carry an [`EdgeKind`] (shared instrument,
//!    co-transaction, billing→shipping link) and a `weight` for downstream
//!    algorithms. Updates are streaming-friendly (incremental insert; no
//!    full rebuild).
//!
//! 3. **Algorithms** ([`components`], [`pagerank`], [`louvain`]) — Standard
//!    graph primitives chosen for fraud workloads: union-find for connected
//!    components (ring discovery), PageRank for influence/centrality
//!    (which mule account is the hub?), and a modularity-greedy Louvain
//!    pass for community detection (which cluster of accounts forms a
//!    single behavioural unit?).
//!
//! 4. **Heuristics** ([`ring`], [`velocity`], [`synthetic`],
//!    [`laundering`]) — Domain detectors on top of the algorithmic layer.
//!    `ring` flags payment instruments shared across an unusual number of
//!    merchants. `velocity` is a rolling per-entity transaction counter
//!    with a fixed window. `synthetic` scores entity vertices for the
//!    classic synthetic-identity signature (fresh PII, thin links, sudden
//!    activity). `laundering` looks for structuring (sub-threshold
//!    splitting), smurfing (many small entities into one) and mule
//!    networks (long thin transfer chains).
//!
//! 5. **Score** ([`score`]) — A small calibrated combiner that turns the
//!    above heuristics into an [`score::EntityRisk`] and a
//!    [`score::TransactionRisk`]. Compatible in spirit with public
//!    descriptions of GraphFraud / Riskified-style graph features:
//!    component size, ring participation, velocity, centrality,
//!    layering depth.
//!
//! ## Privacy
//!
//! Raw identifiers are hashed before they enter [`entity::EntityKey`]. The
//! graph stores only the hash and a coarse [`entity::EntityKind`] tag. PII
//! never lands on a vertex. This is the same posture as `op-fraud`: the
//! detector cannot leak what it does not have.
//!
//! ## What this crate does NOT do
//!
//! - It does not persist the graph. Operators wire `op-graph` (Minigraf)
//!   or their own store. This crate is the in-memory analysis layer.
//! - It does not train an ML model. The detectors are deterministic and
//!   inspectable; results are reproducible across runs.
//! - It does not call external services.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]

pub mod bloom;
pub mod components;
pub mod edge;
pub mod entity;
pub mod error;
pub mod graph;
pub mod laundering;
pub mod louvain;
pub mod pagerank;
pub mod ring;
pub mod score;
pub mod synthetic;
pub mod velocity;

pub use bloom::EntityBloom;
pub use components::{ComponentId, ConnectedComponents};
pub use edge::{Edge, EdgeKind};
pub use entity::{Entity, EntityKey, EntityKind};
pub use error::{Error, Result};
pub use graph::{FraudGraph, VertexId};
pub use laundering::{LayeringPattern, LaunderingDetector};
pub use louvain::{Community, Louvain};
pub use pagerank::PageRank;
pub use ring::{Ring, RingDetector};
pub use score::{EntityRisk, RiskBand, TransactionRisk};
pub use synthetic::SyntheticIdentityScorer;
pub use velocity::{VelocityCounter, VelocityWindow};
