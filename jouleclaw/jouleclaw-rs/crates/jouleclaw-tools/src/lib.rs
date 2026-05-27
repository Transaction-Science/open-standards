//! Deterministic computation toolkit — zero-cost, zero-hallucination tools.
//!
//! Pure functions covering math, text, color, encoding, DevOps, code generation,
//! security, biology, logic, dates, hashing, units, geography, and more. Every
//! function costs effectively 0 energy (no GPU, no LLM, no network), returns in
//! <1 ms, and is deterministic — the "calculator next to the AI".
//!
//! ## Cascade placement
//!
//! This crate is the **L0.5 tool-compute tier** of the JouleClaw cascade
//! (`Cache → Lawful → Embed → Model → Wire`). It sits between the L0.1 fact
//! lookup tier (`jouleclaw-lut`) and the L1 deterministic standard library
//! (`jouleclaw-deterministic`):
//!
//! - **L0**   `jouleclaw-cache`           — memoised results
//! - **L0.1** `jouleclaw-lut`             — registered fact lookup
//! - **L0.5** `jouleclaw-tools`           — *this crate*: 484 deterministic compute tools
//! - **L1**   `jouleclaw-deterministic`   — ~48 standard-library primitives
//! - **L2**   `jouleclaw-mrl`             — Matryoshka embeddings
//! - **L3**   model tiers (liquid / prism / lmm / ebm / omni)
//! - **L4**   `jouleclaw-wire`            — wire / network egress (last resort)
//!
//! ## Tool selection is NOT a tool concern
//!
//! Operators MUST verify the chosen tool's matcher claim *before* executing.
//! That is: *which* tool to run for a given user intent is an upstream
//! **L0.1 fact lookup** or matcher concern (e.g. via `jouleclaw-lut` or a
//! dedicated classifier). The tools in this crate are pure compute — they do
//! not advertise themselves, score themselves against intents, or guard
//! against being called for the wrong reason. Garbage in, deterministic
//! garbage out. The matcher's claim ("this tool fits this request") is what
//! the verifier-in-the-loop (`jouleclaw-verify`) must check; the tool's own
//! output, once invoked, is by construction faithful to its inputs.
//!
//! ## Usage
//!
//! ```rust
//! use jouleclaw_tools::{DeterministicToolKind, execute};
//!
//! let tool = DeterministicToolKind::Math { expression: "2 + 2".into() };
//! let result = execute(&tool).unwrap();
//! assert!(result.contains("4"));
//! ```

#![forbid(unsafe_code)]

mod tools;

// Re-export the public API.
pub use tools::*;
