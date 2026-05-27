//! # jouleclaw-program
//!
//! Typed-signature programs for [JouleClaw][jouleclaw] — DSPy-style modules +
//! compiler, written from scratch in pure Rust. **No PyO3. No Python.**
//!
//! ## Why
//!
//! In JouleClaw, model calls (L3+) are the expensive last resort. When you do
//! call one, you want it as a typed, composable program — not a string prompt
//! that someone hand-tweaks. A "program" is a directed graph of typed modules,
//! each with declared input/output fields.
//!
//! The compiler turns that program into a minimal sequence of cascade
//! dispatches (preferring cheaper tiers when a tier can satisfy a typed
//! signature without invoking a model). v0.1 ships the typed-signature
//! surface, the module trait, three built-in modules (`Predict`,
//! `ChainOfThought`, `ProgramOfThought`), a compiler that lowers programs into
//! a flat dispatch plan, and a runner that executes the plan against a
//! caller-supplied [`Backend`].
//!
//! Integration with `jouleclaw-cascade` happens in a later phase. This crate
//! is intentionally self-contained.
//!
//! [jouleclaw]: https://jouleclaw.transaction.science

#![forbid(unsafe_code)]

pub mod compiler;
pub mod error;
pub mod grammar;
pub mod module;
pub mod program;
pub mod record;
pub mod runner;
pub mod signature;

pub use compiler::{Compiled, Compiler, Dispatch};
pub use error::{Error, Result};
pub use grammar::GrammarHandle;
pub use module::{
    Backend, BackendResponse, ChainOfThought, CodeRunner, Module, ModuleKind, Predict,
    ProgramOfThought,
};
pub use program::{Edge, NamedModule, Port, Program};
pub use record::{Record, Value};
pub use runner::Runner;
pub use signature::{Field, FieldType, Signature};
