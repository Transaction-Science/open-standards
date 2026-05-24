//! Typed errors for agent loops.

use thiserror::Error;

/// Errors emitted by an agent loop.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The supplied LLM provider returned an error.
    #[error("provider error: {0}")]
    Provider(String),

    /// A tool invocation returned an error.
    #[error("tool `{tool}` failed: {reason}")]
    Tool {
        /// The tool name.
        tool: String,
        /// Human-readable reason.
        reason: String,
    },

    /// Parsing the model's structured output failed.
    #[error("parse error: {0}")]
    Parse(String),

    /// A planner emitted a malformed plan.
    #[error("plan error: {0}")]
    Plan(String),

    /// A loop exceeded its hard iteration cap with no termination.
    #[error("iteration cap reached: {0}")]
    MaxIterations(usize),

    /// The joule meter could not be read.
    #[error("meter error: {0}")]
    Meter(String),

    /// Generic backend / I/O.
    #[error("backend error: {0}")]
    Backend(String),
}

/// Convenience alias.
pub type AgentResult<T> = std::result::Result<T, AgentError>;

impl From<eoc_core::Error> for AgentError {
    fn from(e: eoc_core::Error) -> Self {
        AgentError::Backend(e.to_string())
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(e: serde_json::Error) -> Self {
        AgentError::Parse(e.to_string())
    }
}
