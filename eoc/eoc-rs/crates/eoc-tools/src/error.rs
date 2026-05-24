//! Typed errors for the tool-use substrate.

use thiserror::Error;

/// Errors that can arise from schema generation, dispatch, vendor
/// translation, parallel execution, or the orchestration loop.
#[derive(Debug, Error)]
pub enum ToolError {
    /// The requested tool is not registered.
    #[error("tool not registered: {0}")]
    NotRegistered(String),

    /// Argument validation against the tool's JSON Schema failed.
    #[error("argument validation failed for `{tool}`: {reason}")]
    InvalidArguments {
        /// Tool name.
        tool: String,
        /// Human-readable reason.
        reason: String,
    },

    /// The tool itself returned an error.
    #[error("tool execution failed: {0}")]
    Execution(String),

    /// A vendor schema translation failed.
    #[error("schema translation error: {0}")]
    SchemaTranslation(String),

    /// Failed to parse a vendor response for tool-call requests.
    #[error("vendor response parse error: {0}")]
    VendorParse(String),

    /// A tool call timed out.
    #[error("tool `{0}` timed out")]
    Timeout(String),

    /// A safety policy denied the call.
    #[error("policy denied call to `{tool}`: {reason}")]
    PolicyDenied {
        /// Tool name.
        tool: String,
        /// Human-readable reason.
        reason: String,
    },

    /// A sandbox primitive rejected the call (e.g. allowlist).
    #[error("sandbox denial in `{tool}`: {reason}")]
    SandboxDenied {
        /// Tool name.
        tool: String,
        /// Human-readable reason.
        reason: String,
    },

    /// I/O failure.
    #[error("io error: {0}")]
    Io(String),

    /// HTTP failure.
    #[error("http error: {0}")]
    Http(String),

    /// JSON encode/decode failure.
    #[error("serde error: {0}")]
    Serde(String),

    /// The tool loop exceeded its iteration cap.
    #[error("tool loop hit max_iterations ({0})")]
    MaxIterations(usize),
}

impl From<serde_json::Error> for ToolError {
    fn from(e: serde_json::Error) -> Self {
        ToolError::Serde(e.to_string())
    }
}

impl From<std::io::Error> for ToolError {
    fn from(e: std::io::Error) -> Self {
        ToolError::Io(e.to_string())
    }
}

impl From<reqwest::Error> for ToolError {
    fn from(e: reqwest::Error) -> Self {
        ToolError::Http(e.to_string())
    }
}

/// Convenience `Result` alias.
pub type ToolResult<T> = std::result::Result<T, ToolError>;
