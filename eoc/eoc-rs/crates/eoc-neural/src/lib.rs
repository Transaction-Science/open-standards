//! EOC stage 4 — neural inference.
//!
//! The most expensive stage and the stage of last resort. This crate
//! defines the trait surface and ships an `EchoBackend` for testing.
//! Production deployments wire in real backends (llama.cpp, Ollama,
//! Anthropic API, Triton, vLLM, etc.).

#![forbid(unsafe_code)]

use async_trait::async_trait;
use eoc_cache::Stage;
use eoc_core::{JouleCost, Query, Response, Stage as StageKind};

/// A pluggable neural inference backend.
#[async_trait]
pub trait NeuralBackend: Send + Sync {
    /// Run inference for `q` and return a `Response`. Backends are
    /// expected to attach the most accurate joule cost they can compute;
    /// the cascade will trust whatever value is reported.
    async fn infer(&self, q: &Query) -> Response;
}

/// Synthetic backend used for tests and demos. Echoes the prompt back
/// with a configurable joule cost.
pub struct EchoBackend {
    /// Estimated cost in micro-joules per inference.
    pub estimated_microjoules: u64,
    /// Optional prefix added to the echoed payload.
    pub prefix: String,
}

impl EchoBackend {
    /// Construct an `EchoBackend` with a default prefix and synthetic cost.
    pub fn new() -> Self {
        Self {
            estimated_microjoules: 50_000_000, // 50 J — order-of-magnitude
            prefix: "echo: ".to_string(),
        }
    }

    /// Set the estimated cost.
    pub fn with_cost(mut self, microjoules: u64) -> Self {
        self.estimated_microjoules = microjoules;
        self
    }

    /// Set the prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }
}

impl Default for EchoBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NeuralBackend for EchoBackend {
    async fn infer(&self, q: &Query) -> Response {
        Response::new(
            q.id,
            format!("{}{}", self.prefix, q.prompt),
            StageKind::Neural,
            JouleCost::estimated(self.estimated_microjoules),
        )
    }
}

/// The neural stage adapter — wraps any `NeuralBackend` as a `Stage`.
///
/// The neural stage *always* returns `Some` — it is the bottom of the
/// cascade and the answerer of last resort.
pub struct NeuralStage {
    backend: Box<dyn NeuralBackend>,
}

impl NeuralStage {
    /// Wrap a backend.
    pub fn new(backend: Box<dyn NeuralBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Stage for NeuralStage {
    async fn try_resolve(&self, q: &Query) -> Option<Response> {
        Some(self.backend.infer(q).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_backend_echoes() {
        let backend = EchoBackend::new().with_cost(123).with_prefix("> ");
        let stage = NeuralStage::new(Box::new(backend));
        let q = Query::new("hello");
        let r = stage.try_resolve(&q).await.expect("neural always answers");
        assert_eq!(r.payload, "> hello");
        assert_eq!(r.stage, StageKind::Neural);
        assert_eq!(r.joule_cost.microjoules, 123);
    }
}
