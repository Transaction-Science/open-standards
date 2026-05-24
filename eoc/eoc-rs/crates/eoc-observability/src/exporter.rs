//! Span exporters.
//!
//! Exporters are the sink for finished spans. The reference implementation
//! ships an [`InMemoryExporter`] that just collects spans in a `Vec` for
//! testing. Real exporters (OTLP/HTTP, OTLP/gRPC, Jaeger, Zipkin, LangSmith,
//! LangFuse, ...) live in their own modules and implement the same trait.

use crate::error::ObsResult;
use crate::span::Span;
use std::sync::{Arc, Mutex};

/// Span exporter trait.
///
/// Implementations must be `Send + Sync` because the
/// [`crate::processor::BatchSpanProcessor`] is shared across worker threads.
pub trait SpanExporter: Send + Sync {
    /// Hand a batch of finished spans to the exporter. Implementations should
    /// not block for unbounded time.
    fn export(&self, batch: &[Span]) -> ObsResult<()>;

    /// Shut the exporter down. After this call, `export` may return an error.
    fn shutdown(&self) -> ObsResult<()> {
        Ok(())
    }
}

/// In-memory exporter: stashes spans in a `Mutex<Vec>` for tests.
#[derive(Debug, Default, Clone)]
pub struct InMemoryExporter {
    spans: Arc<Mutex<Vec<Span>>>,
}

impl InMemoryExporter {
    /// Construct a fresh, empty exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the currently-collected spans.
    pub fn finished_spans(&self) -> Vec<Span> {
        match self.spans.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Drop all collected spans.
    pub fn reset(&self) {
        if let Ok(mut g) = self.spans.lock() {
            g.clear();
        }
    }
}

impl SpanExporter for InMemoryExporter {
    fn export(&self, batch: &[Span]) -> ObsResult<()> {
        match self.spans.lock() {
            Ok(mut g) => {
                g.extend_from_slice(batch);
                Ok(())
            }
            Err(_) => Err(crate::error::ObsError::Exporter(
                "in-memory exporter poisoned".to_string(),
            )),
        }
    }
}
