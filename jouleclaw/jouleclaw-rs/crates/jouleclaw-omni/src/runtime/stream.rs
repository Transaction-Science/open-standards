//! Streaming output for real-time generation.

use crate::core::{Error, Result};
use alloc::vec::Vec;
use core::pin::Pin;
use core::task::{Context, Poll};
use futures::Stream;

/// A streaming output that produces items incrementally.
///
/// This is the core abstraction for real-time generation.
/// Results are yielded as they become available, enabling
/// sub-100ms perceived latency even for large outputs.
pub struct StreamingOutput<T> {
    /// Channel receiver for items
    receiver: tokio::sync::mpsc::Receiver<Result<T>>,
    /// Handle to the generation task
    handle: StreamHandle,
}

impl<T> StreamingOutput<T> {
    /// Create a new streaming output.
    pub fn new(
        receiver: tokio::sync::mpsc::Receiver<Result<T>>,
        handle: StreamHandle,
    ) -> Self {
        Self { receiver, handle }
    }

    /// Get the stream handle.
    pub fn handle(&self) -> &StreamHandle {
        &self.handle
    }

    /// Cancel the stream.
    pub fn cancel(&self) {
        self.handle.cancel();
    }

    /// Check if the stream is complete.
    pub fn is_complete(&self) -> bool {
        self.handle.is_complete()
    }

    /// Collect all items (blocks until complete).
    pub async fn collect(mut self) -> Result<Vec<T>> {
        let mut items = Vec::new();
        while let Some(item) = self.receiver.recv().await {
            items.push(item?);
        }
        Ok(items)
    }
}

impl<T> Stream for StreamingOutput<T> {
    type Item = Result<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_recv(cx)
    }
}

/// Handle for controlling a streaming operation.
#[derive(Debug, Clone)]
pub struct StreamHandle {
    /// Unique stream ID
    id: crate::core::Id,
    /// Cancellation flag
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Completion flag
    complete: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl StreamHandle {
    /// Create a new stream handle.
    pub fn new() -> Self {
        Self {
            id: crate::core::Id::new(),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            complete: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Get the stream ID.
    pub fn id(&self) -> crate::core::Id {
        self.id
    }

    /// Cancel the stream.
    pub fn cancel(&self) {
        self.cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Mark as complete.
    pub fn mark_complete(&self) {
        self.complete.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if complete.
    pub fn is_complete(&self) -> bool {
        self.complete.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Default for StreamHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating streaming outputs.
pub struct StreamBuilder<T> {
    buffer_size: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T> StreamBuilder<T> {
    /// Create a new stream builder.
    pub fn new() -> Self {
        Self {
            buffer_size: 32,
            _marker: core::marker::PhantomData,
        }
    }

    /// Set the buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Build the stream.
    pub fn build(self) -> (StreamingOutput<T>, StreamSender<T>) {
        let (tx, rx) = tokio::sync::mpsc::channel(self.buffer_size);
        let handle = StreamHandle::new();

        let output = StreamingOutput::new(rx, handle.clone());
        let sender = StreamSender::new(tx, handle);

        (output, sender)
    }
}

impl<T> Default for StreamBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Sender side of a streaming output.
#[derive(Clone)]
pub struct StreamSender<T> {
    sender: tokio::sync::mpsc::Sender<Result<T>>,
    handle: StreamHandle,
}

impl<T> StreamSender<T> {
    /// Create a new sender.
    pub fn new(
        sender: tokio::sync::mpsc::Sender<Result<T>>,
        handle: StreamHandle,
    ) -> Self {
        Self { sender, handle }
    }

    /// Send an item.
    pub async fn send(&self, item: T) -> Result<()> {
        if self.handle.is_cancelled() {
            return Err(Error::execution("stream", "cancelled"));
        }

        self.sender
            .send(Ok(item))
            .await
            .map_err(|_| Error::execution("stream", "receiver dropped"))
    }

    /// Send an error.
    pub async fn send_error(&self, error: Error) -> Result<()> {
        self.sender
            .send(Err(error))
            .await
            .map_err(|_| Error::execution("stream", "receiver dropped"))
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.handle.is_cancelled()
    }

    /// Mark the stream as complete.
    pub fn complete(self) {
        self.handle.mark_complete();
        // Sender drops, closing the channel
    }
}

/// Progress information for streaming operations.
#[derive(Debug, Clone, Copy)]
pub struct StreamProgress {
    /// Items produced so far
    pub produced: usize,
    /// Estimated total items (if known)
    pub total: Option<usize>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl StreamProgress {
    /// Calculate progress percentage (0.0 to 1.0).
    pub fn percentage(&self) -> Option<f32> {
        self.total.map(|t| self.produced as f32 / t as f32)
    }

    /// Calculate items per second.
    pub fn items_per_second(&self) -> f32 {
        if self.elapsed_ms == 0 {
            0.0
        } else {
            self.produced as f32 / (self.elapsed_ms as f32 / 1000.0)
        }
    }

    /// Estimate remaining time in milliseconds.
    pub fn eta_ms(&self) -> Option<u64> {
        let total = self.total?;
        let remaining = total.saturating_sub(self.produced);
        let rate = self.items_per_second();

        if rate == 0.0 {
            None
        } else {
            Some((remaining as f32 / rate * 1000.0) as u64)
        }
    }
}
