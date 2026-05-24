//! Batch span processor.
//!
//! The [`BatchSpanProcessor`] buffers finished spans and flushes them to a
//! [`SpanExporter`] in batches of `max_batch_size` (or sooner, via
//! [`BatchSpanProcessor::force_flush`]). The reference implementation is
//! synchronous: there is no background timer thread, so callers control when
//! flushes happen. This keeps the crate runtime-agnostic (no tokio
//! dependency), which matters for embedded and WASM builds.

use crate::error::ObsResult;
use crate::exporter::SpanExporter;
use crate::span::Span;
use std::sync::{Arc, Mutex};

/// Configuration for [`BatchSpanProcessor`].
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of spans buffered before an auto-flush.
    pub max_batch_size: usize,
    /// Maximum queue depth before spans get dropped. `0` = unbounded.
    pub max_queue_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            max_queue_size: 2048,
        }
    }
}

/// Synchronous batch span processor.
pub struct BatchSpanProcessor {
    exporter: Arc<dyn SpanExporter>,
    config: BatchConfig,
    queue: Mutex<Vec<Span>>,
    dropped: Mutex<u64>,
}

impl std::fmt::Debug for BatchSpanProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchSpanProcessor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl BatchSpanProcessor {
    /// Construct with default config.
    pub fn new(exporter: Arc<dyn SpanExporter>) -> Self {
        Self::with_config(exporter, BatchConfig::default())
    }

    /// Construct with an explicit config.
    pub fn with_config(exporter: Arc<dyn SpanExporter>, config: BatchConfig) -> Self {
        Self {
            exporter,
            config,
            queue: Mutex::new(Vec::new()),
            dropped: Mutex::new(0),
        }
    }

    /// Submit a finished span. If the queue is full, the span is dropped
    /// and the dropped counter is incremented. If the queue reaches
    /// `max_batch_size`, an auto-flush is triggered.
    pub fn on_end(&self, span: Span) -> ObsResult<()> {
        let needs_flush = {
            let mut q = self.queue.lock().map_err(|_| {
                crate::error::ObsError::Exporter("processor poisoned".to_string())
            })?;
            if self.config.max_queue_size > 0 && q.len() >= self.config.max_queue_size {
                drop(q);
                if let Ok(mut d) = self.dropped.lock() {
                    *d = d.saturating_add(1);
                }
                return Ok(());
            }
            q.push(span);
            q.len() >= self.config.max_batch_size
        };
        if needs_flush {
            self.force_flush()?;
        }
        Ok(())
    }

    /// Force the buffered spans to the exporter immediately.
    pub fn force_flush(&self) -> ObsResult<()> {
        let drained: Vec<Span> = {
            let mut q = self.queue.lock().map_err(|_| {
                crate::error::ObsError::Exporter("processor poisoned".to_string())
            })?;
            std::mem::take(&mut *q)
        };
        if drained.is_empty() {
            return Ok(());
        }
        self.exporter.export(&drained)
    }

    /// Shutdown: flush, then shut the exporter down.
    pub fn shutdown(&self) -> ObsResult<()> {
        self.force_flush()?;
        self.exporter.shutdown()
    }

    /// Number of spans dropped due to a full queue.
    pub fn dropped_count(&self) -> u64 {
        self.dropped.lock().map(|g| *g).unwrap_or(0)
    }

    /// Number of spans currently buffered (not yet exported).
    pub fn pending_count(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }
}
