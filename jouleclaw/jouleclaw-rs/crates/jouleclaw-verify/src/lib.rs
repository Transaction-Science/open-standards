//! # jouleclaw-verify
//!
//! Verifier-in-the-loop. **AI proposes, verifier disposes.**
//!
//! When the JouleClaw cascade has no choice but to fire L3 (a local
//! stochastic model) or L4 (a remote frontier RPC), the doctrine says
//! we constrain that open-ended inference as much as possible and
//! gate its output through a chain of *deterministic* verifiers. Only
//! output that passes every verifier in the chain is allowed to close
//! the cascade walk; refusal triggers retry-with-cheaper-tier or
//! returns `Unresolvable` to the caller.
//!
//! This is the formal-methods pattern — the same trick used by Lean
//! tactic search, bounded model checkers (BMC), SymCode, ProofNet++,
//! and the wider "proposes / disposes" literature — packaged as a
//! tiny composable trait surface:
//!
//! - [`OutputVerifier`] — one deterministic check, named, with a
//!   declared microjoule cost so receipts stay honest.
//! - [`VerifierChain`] — ordered fail-fast composition. The first
//!   verifier to refuse wins; prior passes are still recorded as
//!   `ToolTouch` rows so the receipt shows the work that *was* done.
//! - Three reference verifiers: [`RegexVerifier`],
//!   [`JsonSchemaVerifier`], [`BlakeHashVerifier`].
//!
//! ## The verifier-honesty load-bearing assumption
//!
//! Every verifier appears in the receipt by name (`verify:<tag>`) and
//! contributes its `declared_cost_uj` to the receipt's `joules_uj`
//! total. **A verifier that lies — that returns `Pass` for output it
//! did not actually check, or that under-reports its cost — corrupts
//! the resulting receipt.** Receipts are the thermodynamic ledger;
//! the integrity of "capability per joule, not capability per
//! parameter" hangs on every verifier in the chain being truthful
//! about both its verdict and its cost.
//!
//! Implementations that wrap untrusted code (sandboxed evaluators,
//! networked oracles, third-party crates) should declare their cost
//! generously and floor their `energy_provenance` to
//! [`Provenance::Estimator`][jouleclaw_energy::Provenance::Estimator]
//! — which this crate does by default. Receipts produced through this
//! crate will never claim hardware-shunt-grade energy honesty for a
//! verifier touch.
//!
//! ## What this crate is NOT
//!
//! - Not a full JSON Schema draft-2020 implementation. The bundled
//!   schema verifier checks required top-level keys and their
//!   coarse types only; reach for `jsonschema` if you need draft
//!   compliance.
//! - Not a proof-search engine. We dispose, we do not propose.
//! - Not a transport — verifiers run in-process, synchronously.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod chain;
pub mod error;
pub mod hash_verifier;
pub mod json_schema_verifier;
pub mod regex_verifier;
pub mod verifier;

pub use chain::{ChainResult, VerifierChain};
pub use error::VerifyError;
pub use hash_verifier::BlakeHashVerifier;
pub use json_schema_verifier::{JsonSchemaVerifier, JsonType};
pub use regex_verifier::RegexVerifier;
pub use verifier::{OutputVerifier, VerifyResult};
