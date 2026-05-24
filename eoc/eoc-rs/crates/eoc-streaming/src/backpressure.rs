//! Bounded streams with explicit high/low watermarks.
//!
//! Most provider streams emit faster than a consumer can render or
//! re-serialize them. We wrap a bounded MPSC channel with watermarks
//! so a producer can pause when occupancy crosses the high watermark
//! and resume when it drops below the low watermark. The blocking
//! semantics are inherited from `tokio::sync::mpsc::Sender::send`,
//! which awaits when the channel is full.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{StreamError, StreamResult};
use crate::stream::Event;

/// High/low watermarks expressed as channel occupancy.
#[derive(Debug, Clone, Copy)]
pub struct Watermarks {
    /// Channel capacity.
    pub capacity: usize,
    /// Pause-producer threshold. Must be ≤ `capacity`.
    pub high: usize,
    /// Resume-producer threshold. Must be ≤ `high`.
    pub low: usize,
}

impl Watermarks {
    /// Construct watermarks with `(capacity, capacity*3/4, capacity/2)`.
    pub fn default_for(capacity: usize) -> Self {
        let cap = capacity.max(2);
        Self {
            capacity: cap,
            high: (cap * 3) / 4,
            low: cap / 2,
        }
    }
}

/// Bounded stream with watermark-tracked occupancy.
#[derive(Debug)]
pub struct BoundedStream {
    tx: tokio::sync::mpsc::Sender<Event>,
    rx: tokio::sync::mpsc::Receiver<Event>,
    occupancy: Arc<AtomicUsize>,
    watermarks: Watermarks,
}

impl BoundedStream {
    /// Construct a bounded stream with the supplied watermarks.
    pub fn new(watermarks: Watermarks) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(watermarks.capacity);
        Self {
            tx,
            rx,
            occupancy: Arc::new(AtomicUsize::new(0)),
            watermarks,
        }
    }

    /// Get a producer handle.
    pub fn producer(&self) -> Producer {
        Producer {
            tx: self.tx.clone(),
            occupancy: self.occupancy.clone(),
            watermarks: self.watermarks,
        }
    }

    /// Receive the next event, decrementing occupancy.
    pub async fn recv(&mut self) -> Option<Event> {
        let ev = self.rx.recv().await?;
        self.occupancy.fetch_sub(1, Ordering::AcqRel);
        Some(ev)
    }

    /// Current occupancy snapshot.
    pub fn occupancy(&self) -> usize {
        self.occupancy.load(Ordering::Acquire)
    }

    /// Watermarks in use.
    pub fn watermarks(&self) -> Watermarks {
        self.watermarks
    }
}

/// Producer half of a [`BoundedStream`].
#[derive(Clone, Debug)]
pub struct Producer {
    tx: tokio::sync::mpsc::Sender<Event>,
    occupancy: Arc<AtomicUsize>,
    watermarks: Watermarks,
}

impl Producer {
    /// Awaitable send. Returns immediately under the low watermark; may
    /// await arbitrarily long if the receiver is slow.
    pub async fn send(&self, ev: Event) -> StreamResult<()> {
        self.tx.send(ev).await.map_err(|_| StreamError::Closed)?;
        self.occupancy.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Non-blocking send. Returns [`StreamError::Backpressure`] if the
    /// channel is at capacity.
    pub fn try_send(&self, ev: Event) -> StreamResult<()> {
        use tokio::sync::mpsc::error::TrySendError;
        self.tx.try_send(ev).map_err(|e| match e {
            TrySendError::Full(_) => StreamError::Backpressure,
            TrySendError::Closed(_) => StreamError::Closed,
        })?;
        self.occupancy.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// True if occupancy is at or above the high watermark.
    pub fn should_pause(&self) -> bool {
        self.occupancy.load(Ordering::Acquire) >= self.watermarks.high
    }

    /// True if occupancy is at or below the low watermark.
    pub fn should_resume(&self) -> bool {
        self.occupancy.load(Ordering::Acquire) <= self.watermarks.low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::Role;

    #[tokio::test]
    async fn watermark_tracks_occupancy() {
        let wm = Watermarks {
            capacity: 4,
            high: 3,
            low: 1,
        };
        let mut s = BoundedStream::new(wm);
        let p = s.producer();
        for _ in 0..3 {
            p.send(Event::MessageStart {
                id: None,
                role: Role::Assistant,
            })
            .await
            .unwrap();
        }
        assert!(p.should_pause());
        s.recv().await.unwrap();
        s.recv().await.unwrap();
        s.recv().await.unwrap();
        assert!(p.should_resume());
    }
}
