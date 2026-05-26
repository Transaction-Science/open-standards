//! Liquid — Closed-form Continuous-time (CfC) cells and the LiquidTier.
//!
//! The Liquid AI thesis (Hasani, Lechner, Amini, et al., MIT CSAIL → Liquid AI Inc.):
//! a recurrent cell whose dynamics derive from a continuous-time differential
//! equation, but whose forward pass is a closed-form algebraic expression
//! (no ODE solver, no Runge-Kutta steps). The result is a small-parameter
//! sequence model that approximates the expressivity of much larger
//! transformers on time-series-like tasks, at a fraction of the cost,
//! and that can be deployed on wearables, phones, and Pi-class hardware
//! down to sub-1GB footprints (LFM2.5-1.2B-Thinking).
//!
//! Reference: Hasani, Lechner et al., "Closed-form Continuous-time Neural
//! Models," Nature Machine Intelligence (2022). The CfC update we
//! implement here is the discrete-step form of that paper's equation:
//!
//!   z       = [x_t ; u_t]                         (concat state + input)
//!   gate    = σ( -(W_f · z + b_f) - θ_t )         (time gate)
//!   content = tanh( W_g · z + b_g )               (new content)
//!   alt     = tanh( W_h · z + b_h )               (alternative state)
//!   x_{t+1} = gate ⊙ content + (1 - gate) ⊙ alt
//!
//! No multiplicative coupling with x_t inside the gate; pure feed-forward
//! gating against the concatenation. That's what makes it closed-form vs
//! the original Liquid Time-Constant (LTC) formulation, which needs a
//! numerical ODE step.
//!
//! This crate provides:
//!
//! - [`cell::CfcCell`] — a single CfC cell. Pure-Rust f32 math; no SIMD,
//!   no `unsafe`. Deterministic per-platform (uses the std `tanh`/`exp`
//!   implementations).
//!
//! - [`model::LiquidModel`] — a stack of cells with optional input/output
//!   projections. The forward pass is a sequence of `cell.step` calls.
//!
//! - [`tier::LiquidTier`] — joule cascade tier shell. The kernel is in
//!   place; full model integration (tokenizer, embedding lookup, sampling,
//!   weight loading) is R29.1.
//!
//! See also: [`prism`](https://github.com/) for the orthogonal efficiency
//! axis (weight precision). A production Liquid + Prism composition uses
//! CfC dynamics with ternary weights — but that's a deliberate R28+R29
//! interaction, not a hidden coupling.

pub mod cell;
pub mod lm;
pub mod model;
pub mod tier;

pub use cell::CfcCell;
pub use lm::{synthetic_lm, LiquidLanguageModel, LmConfig};
pub use model::LiquidModel;
pub use tier::LiquidTier;
