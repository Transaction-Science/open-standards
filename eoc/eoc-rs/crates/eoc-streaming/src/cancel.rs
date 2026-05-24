//! Stream cancellation.
//!
//! A [`CancelToken`] is a cheap, clonable handle on a shared
//! cancellation flag. The HTTP/WS bridge holds one; when a client
//! disconnects, it flips the flag and notifies waiters. Upstream
//! mappers and the cascade observe the flag between events and abort
//! the in-flight request.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

/// Cancellation handle for a single stream.
#[derive(Clone, Debug)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl CancelToken {
    /// Construct a fresh, un-cancelled token.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Flip to cancelled and wake any waiters.
    pub fn cancel(&self) {
        if !self.flag.swap(true, Ordering::AcqRel) {
            self.notify.notify_waiters();
        }
    }

    /// Snapshot the cancelled state.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Wait until cancelled.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancel_signals_waiter() {
        let tok = CancelToken::new();
        let tok2 = tok.clone();
        let h = tokio::spawn(async move {
            tok2.cancelled().await;
            true
        });
        assert!(!tok.is_cancelled());
        tok.cancel();
        assert!(h.await.unwrap());
        assert!(tok.is_cancelled());
    }

    #[tokio::test]
    async fn cancel_is_idempotent() {
        let tok = CancelToken::new();
        tok.cancel();
        tok.cancel();
        assert!(tok.is_cancelled());
    }
}
