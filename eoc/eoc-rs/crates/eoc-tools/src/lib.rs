//! EOC tool-use / function-calling substrate.
//!
//! Modern LLM workflows are not single-turn completions: the model emits
//! a structured *tool-call request*, an orchestrator runs the tool, the
//! result is fed back in, and the loop repeats until the model returns a
//! plain answer. To be a complete substrate EOC ingests the protocol
//! surface that sits underneath that pattern:
//!
//! * [`schema`] — canonical [`ToolSchema`] derived from a Rust type via
//!   `schemars` (`#[derive(JsonSchema)]`).
//! * [`tool`] — the [`Tool`] trait + [`ToolRegistry`] dispatcher.
//! * Per-vendor adapters ([`anthropic_schema`], [`openai_schema`],
//!   [`google_schema`], [`mistral_schema`], [`cohere_schema`]) that
//!   translate the canonical schema to/from each provider's wire shape.
//! * [`parallel`] — bounded-concurrency parallel tool execution.
//! * [`tool_loop`] — the multi-turn orchestration loop with joule-cost
//!   accumulation.
//! * [`builtins`] — sandboxed reference tools (shell, http, sql, file,
//!   search).
//! * [`safety`] — pre/post invocation policies (rate limit, jailbreak
//!   match, PII redaction).
//!
//! The crate is `#![forbid(unsafe_code)]` and contains no `unwrap`
//! outside of `#[test]` blocks.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod anthropic_schema;
pub mod builtins;
pub mod cohere_schema;
pub mod error;
pub mod google_schema;
pub mod mistral_schema;
pub mod openai_schema;
pub mod parallel;
pub mod safety;
pub mod schema;
pub mod tool;
pub mod tool_loop;

pub use error::{ToolError, ToolResult};
pub use parallel::{ParallelConfig, ToolCallResult, execute_parallel};
pub use safety::{Decision, ToolPolicy};
pub use schema::{ToolSchema, schema_for};
pub use tool::{Tool, ToolCallRequest, ToolRegistry};
pub use tool_loop::{SchemaTranslator, ToolLoop};
