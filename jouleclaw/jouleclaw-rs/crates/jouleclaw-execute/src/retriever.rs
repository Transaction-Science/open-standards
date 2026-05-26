//! The `Retriever` trait every store implements (spec §5.3).
//!
//! A retriever's RAP names one or more methods (e.g.
//! `wikidata_primary`, `wikidata_alt_predicate`); the RAP executor
//! dispatches by method name. Implementations expose those methods
//! through a single async entry point taking a `(method, parameters)`
//! pair, so the executor stays generic over backends.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use jouleclaw_schema::{RetrievedItem, SubQuery};

/// Errors a retriever can return from a single step invocation.
#[derive(Debug)]
pub enum RetrieverError {
    /// The retriever doesn't know how to dispatch the requested
    /// method name. The RAP executor treats this as `ON_ERROR`.
    UnknownMethod(String),
    /// Network / IO failure.
    Backend(String),
    /// Backend response couldn't be parsed.
    ParseFailed(String),
    /// Backend deliberately refused (e.g. rate limit).
    Refused(String),
}

impl std::fmt::Display for RetrieverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownMethod(m) => write!(f, "unknown method: {m}"),
            Self::Backend(s) => write!(f, "backend error: {s}"),
            Self::ParseFailed(s) => write!(f, "parse failed: {s}"),
            Self::Refused(s) => write!(f, "refused: {s}"),
        }
    }
}

impl std::error::Error for RetrieverError {}

/// A retriever knows how to invoke one or more RAP-named methods
/// against a backing store, returning [`RetrievedItem`]s annotated
/// with provenance and the seven `KnowledgeAxes`.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Stable id matching `RetrieverCapability.retriever_id`.
    fn retriever_id(&self) -> &str;

    /// Run one RAP-step method against this retriever. `parameters`
    /// is the per-step `parameters` map from the RAP definition,
    /// opaque to the executor.
    async fn call(
        &self,
        method: &str,
        subquery: &SubQuery,
        parameters: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Vec<RetrievedItem>, RetrieverError>;
}

/// Registry mapping `retriever_id` → `Retriever` impl. The
/// orchestrator looks up the retriever for each sub-query's first
/// target_store at dispatch time.
#[derive(Clone, Default)]
pub struct RetrieverRegistry {
    inner: BTreeMap<String, Arc<dyn Retriever>>,
}

impl RetrieverRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, retriever: Arc<dyn Retriever>) {
        self.inner.insert(retriever.retriever_id().to_string(), retriever);
    }

    pub fn get(&self, id: &str) -> Option<&Arc<dyn Retriever>> {
        self.inner.get(id)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl std::fmt::Debug for RetrieverRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetrieverRegistry")
            .field("ids", &self.inner.keys().collect::<Vec<_>>())
            .finish()
    }
}
