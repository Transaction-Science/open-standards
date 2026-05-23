//! Query — the input to the cascade.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A 32-byte content-addressed query identifier (BLAKE3 of the prompt).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QueryId(pub [u8; 32]);

impl QueryId {
    /// Derive a `QueryId` from a prompt string (BLAKE3).
    pub fn from_prompt(prompt: &str) -> Self {
        Self(*blake3::hash(prompt.as_bytes()).as_bytes())
    }

    /// Hex encoding for logging / display.
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for b in &self.0 {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }
}

impl std::fmt::Display for QueryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hex = self.to_hex();
        // Print short form by default; full hex with `:#`.
        if f.alternate() {
            f.write_str(&hex)
        } else {
            f.write_str(&hex[..16])
        }
    }
}

/// A query submitted to the cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Content-addressed identifier (derived from the prompt by default).
    pub id: QueryId,
    /// Human-readable prompt text.
    pub prompt: String,
    /// Optional embedding vector — used by the KV stage for similarity match.
    pub embedding: Option<Vec<f32>>,
    /// Free-form metadata (tenant, request-id, model hint, etc.).
    pub metadata: BTreeMap<String, String>,
}

impl Query {
    /// Construct a query from a prompt, deriving the id automatically.
    pub fn new(prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        Self {
            id: QueryId::from_prompt(&prompt),
            prompt,
            embedding: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Attach an embedding (consumes `self`).
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Insert a metadata pair (consumes `self`).
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}
