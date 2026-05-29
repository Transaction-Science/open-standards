//! ARL core — the AI Readiness Level claim model, with the cross-axis
//! gates and controlled vocabulary enforced in code.
//!
//! ARL is a universal, vendor-neutral measurement standard: it scores any
//! AI system the way the Technology Readiness Level scale scores any
//! technology. This crate is deliberately **standalone** — it is tied to
//! no model, runtime, or vendor. It models a complete four-axis claim and
//! refuses to let an *invalid* or *unmeasurable* one stand:
//!
//! - the four axes — [`ValidationDepth`] (1–9), [`ConvergenceClass`]
//!   (A–E), [`EnergyProfile`] (joules), [`SecurityClass`] (S0–S4);
//! - the **cross-axis gates** ([`Claim::validate`]) — a high validation
//!   depth is unreachable without a matching convergence class and
//!   security class, energy non-disclosure caps the score, security
//!   methodology non-disclosure caps the security class, and methodology
//!   must predate the claim;
//! - the **controlled vocabulary** ([`crate::lexicon`]) — terms with no
//!   single operational definition (AGI, superintelligence, consciousness, …)
//!   cannot anchor a claim, because they cannot be measured; terms with a
//!   measurable operational sense are flagged to confirm that sense is meant.
//!   ARL takes no position on the terms themselves, only on their measurability.
//!
//! A `Claim` that passes [`validate`](Claim::validate) is a well-formed
//! ARL claim, scoped to what can be measured. The returned [`Violation`]s
//! say exactly why a claim that does not pass falls outside that scope.

#![forbid(unsafe_code)]

pub mod axes;
pub mod claim;
pub mod lexicon;

pub use axes::{ConvergenceClass, EnergyProfile, SecurityClass, ValidationDepth};
pub use claim::{Claim, Violation};
pub use lexicon::{LexiconFinding, Severity};
