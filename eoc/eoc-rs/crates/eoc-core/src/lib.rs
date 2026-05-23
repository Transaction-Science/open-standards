//! Core types for EOC — Energy-Optimized Compute.
//!
//! This crate is the type backbone for the four-stage memoizing cascade
//! (cache → key-value → graph → neural). It deliberately has no runtime
//! dependencies beyond serde/blake3/thiserror/async-trait so it can compile
//! to `wasm32-unknown-unknown` unchanged.

#![forbid(unsafe_code)]

pub mod error;
pub mod joule_cost;
pub mod query;
pub mod receipt;
pub mod response;
pub mod stage;

pub use error::{Error, Result};
pub use joule_cost::{JouleCost, JouleSource};
pub use query::{Query, QueryId};
pub use receipt::Receipt;
pub use response::Response;
pub use stage::Stage;
