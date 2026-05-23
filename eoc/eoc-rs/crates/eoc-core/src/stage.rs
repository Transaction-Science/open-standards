//! The four stages of the EOC cascade.

use serde::{Deserialize, Serialize};

/// Stage that resolved a query.
///
/// Stages are ordered cheapest → most expensive. The cascade attempts each
/// in order; the first stage that returns `Some(Response)` wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    /// LRU / content-addressed cache (cache hits are nearly free).
    Cache,
    /// Key-value lookup, including embedding-similarity match.
    Kv,
    /// Graph / triple store retrieval (DCY-style).
    Graph,
    /// Neural inference — the last and most expensive stage.
    Neural,
}

impl Stage {
    /// Stable string identifier — used in receipts and CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Stage::Cache => "cache",
            Stage::Kv => "kv",
            Stage::Graph => "graph",
            Stage::Neural => "neural",
        }
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
