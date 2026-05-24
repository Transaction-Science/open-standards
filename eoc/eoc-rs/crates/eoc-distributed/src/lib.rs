//! `eoc-distributed` — distributed-inference scheduling, worker pool,
//! and cluster coordination for energy-optimized AI compute.
//!
//! Modern serving stacks combine four loosely-related concerns:
//!
//! * **Worker pool** — heterogeneous accelerators (CPU/GPU/TPU/NPU)
//!   serving one or more models.
//! * **Cluster coordination** — membership, heartbeats, failure
//!   detection.
//! * **Routing** — pick a worker for each incoming request.
//! * **Batching** — group decode iterations across requests (vLLM /
//!   Triton style).
//!
//! This crate covers all four with a deterministic, dependency-light
//! Rust implementation. Every routing decision is driven by a simple
//! [`Load`](worker::Load) snapshot, so swapping in tensor-parallel /
//! pipeline-parallel / expert-parallel backends is additive.
//!
//! ## Joule-aware placement
//!
//! The router's default strategy is [`Strategy::JouleWeighted`](router::Strategy)
//! — for each candidate it projects micro-joules-per-token × expected
//! tokens and picks the minimum, tie-broken by busyness. The
//! [`Strategy::CarbonWeighted`](router::Strategy) variant additionally
//! multiplies in the worker's local gCO2e/kWh so the lowest-grid-zone
//! worker wins when joule efficiency is comparable.
//!
//! ## KV-cache locality
//!
//! [`KvCacheAwareRouter`](kv_cache_aware::KvCacheAwareRouter) keeps a
//! `session_id -> worker_id` index. When a multi-turn caller comes back,
//! it lands on the worker that still holds the prefix KV-cache, dodging
//! a 10-100× joule re-prefill.
//!
//! ## Continuous batching
//!
//! [`ContinuousBatcher`](batch::ContinuousBatcher) is a vLLM-style
//! admit-and-decode loop with a KV-block budget and a Triton-style
//! "max_queue_delay" approximation.
//!
//! ## Auto-scaling
//!
//! [`ReplicaController`](replica::ReplicaController) implements a
//! Ray-Serve-style reactive controller with cooldown and min/max
//! replicas.
//!
//! ## Topology
//!
//! [`Topology`](topology::Topology) covers mesh / ring / tree wiring
//! for tensor-parallel, pipeline-parallel, and broadcast/reduce
//! collectives. Pretty-printed via `Display`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod batch;
pub mod cluster;
pub mod error;
pub mod kv_cache_aware;
pub mod replica;
pub mod router;
pub mod scheduler;
pub mod topology;
pub mod worker;

pub use batch::{BatchConfig, BatchRequest, ContinuousBatcher};
pub use cluster::{
    Cluster, FixedBudgetDetector, Heartbeat, HeartbeatDetector, Node, NodeStatus,
};
pub use error::{DistributedError, Result};
pub use kv_cache_aware::{KvCacheAwareRouter, LocalRequest};
pub use replica::{ReplicaController, ScaleConfig, ScaleDecision};
pub use router::{Request, Router, Strategy};
pub use scheduler::{WorkItem, WorkStealingScheduler};
pub use topology::{Topology, TopologyKind};
pub use worker::{Accelerator, Capability, InMemoryWorker, Load, Worker};
