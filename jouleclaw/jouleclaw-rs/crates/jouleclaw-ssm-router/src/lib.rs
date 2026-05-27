//! # jouleclaw-ssm-router — L0.75 SSM router tier
//!
//! Local intent classifier that decides which downstream tier(s) should
//! handle a query, *without* committing to a full read. Sits between the
//! L0.5 tool-compute tier and the L1 lawful tier in the JouleClaw
//! cascade. Class-typical cost: ~100 µJ, ~100 µs.
//!
//! The tier reports a route — it does not answer the query. Callers can
//! consume the `routed_to` hint in the structured answer to short-circuit
//! the cascade, or simply ignore the tier (refused) and let the runtime
//! keep walking.
//!
//! ## Architecture
//!
//! The donor (`verity-cascade::layers::l075_ssm_router`) calls into a
//! real SSM engine (Mamba-3 class) when one is loaded. JouleClaw is the
//! open-standard layer: it carries an [`IntentClassifier`] trait so
//! production deployments can plug in a Liquid / Mamba / Hyena backend,
//! and a deterministic hash-based [`KeywordClassifier`] default for
//! v0.1.
//!
//! The default classifier is *deterministic* (same query → same intent,
//! always) and *cheap* (sub-microsecond pure CPU). It combines:
//!
//! - keyword cues (`tool_signals`, `greetings`, question words),
//! - Shannon character / word entropy (low ↔ lookup, high ↔ reasoning),
//! - blake3-stable tie-breaking so two near-equal classifications never
//!   flip-flop across runs.
//!
//! ## Wiring
//!
//! ```ignore
//! use jouleclaw_cascade::tier::{Cascade, Runtime};
//! use jouleclaw_ssm_router::SsmRouterTier;
//!
//! let mut cascade = Cascade::new();
//! cascade.register(Box::new(SsmRouterTier::new()));
//! let mut rt = Runtime::new_without_l0(cascade);
//! ```
//!
//! Plug in a custom classifier:
//!
//! ```ignore
//! struct MyMambaClassifier { /* … */ }
//! impl jouleclaw_ssm_router::IntentClassifier for MyMambaClassifier { /* … */ }
//!
//! let tier = SsmRouterTier::with_classifier(Box::new(MyMambaClassifier { /* … */ }));
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod classifier;
mod tier;

pub use classifier::{
    Intent, IntentClassifier, KeywordClassifier, QueryEntropy, RouteHint, query_entropy,
};
pub use tier::{
    SSM_ROUTER_CONFIDENCE_FLOOR, SSM_ROUTER_JOULES, SSM_ROUTER_LATENCY, SsmRouterError,
    SsmRouterTier,
};
