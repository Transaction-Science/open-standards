//! Producer-side emission of webhook events.
//!
//! The dispatcher in this crate already knows how to send a
//! [`WebhookEvent`] to every subscribed endpoint; what was missing
//! was the *call site* — a thin trait domain code can call to
//! "publish this thing happened," without taking on a hard
//! dependency on the dispatcher's full machinery.
//!
//! ## Usage
//!
//! ```ignore
//! let emitter: Arc<dyn EventEmitter> = Arc::new(
//!     WebhookEmitter::new(Arc::new(dispatcher))
//! );
//! // Inside a handler, after a successful state change:
//! emitter.emit("refund.created", refund_json_bytes, now_unix_secs);
//! ```
//!
//! Production deployments hold one `Arc<dyn EventEmitter>` per
//! process; tests use [`NoOpEmitter`] to keep deterministic.

use std::sync::Arc;

use crate::dispatcher::WebhookDispatcher;
use crate::event::WebhookEvent;

/// The producer-side emission interface.
///
/// Calling code passes a freshly-built payload and the dispatcher
/// fans it out to every subscribed endpoint. Errors are
/// **swallowed and logged** — webhook delivery is fire-and-forget
/// from the producer's perspective; failures don't roll back the
/// caller's transaction.
pub trait EventEmitter: Send + Sync {
    /// Publish an event. `event_type` is the operator-defined
    /// taxonomy key (`"refund.created"`, `"subscription.canceled"`,
    /// etc.); `payload` is opaque bytes (typically JSON, but the
    /// crate doesn't enforce that).
    fn emit(&self, event_type: &str, payload: Vec<u8>, now_unix_secs: u64);
}

/// `EventEmitter` that delegates to a [`WebhookDispatcher`].
pub struct WebhookEmitter {
    dispatcher: Arc<WebhookDispatcher>,
}

impl WebhookEmitter {
    /// Construct.
    #[must_use]
    pub fn new(dispatcher: Arc<WebhookDispatcher>) -> Self {
        Self { dispatcher }
    }
}

impl EventEmitter for WebhookEmitter {
    fn emit(&self, event_type: &str, payload: Vec<u8>, now_unix_secs: u64) {
        let event = WebhookEvent::new(event_type, payload, now_unix_secs);
        if let Err(e) = self.dispatcher.dispatch(event) {
            // Producer-side failure (typically a store I/O fault).
            // Webhook delivery is not transactionally bound to the
            // caller's mutation, so we log and continue.
            tracing::warn!(
                error = %e,
                event_type,
                "webhook emission failed; caller's state change is unaffected"
            );
        }
    }
}

/// `EventEmitter` that drops every event on the floor. Used in
/// tests where webhook side effects aren't part of the assertion,
/// and as the default when an operator hasn't configured webhooks.
#[derive(Default, Clone, Copy)]
pub struct NoOpEmitter;

impl NoOpEmitter {
    /// Construct.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl EventEmitter for NoOpEmitter {
    fn emit(&self, _event_type: &str, _payload: Vec<u8>, _now_unix_secs: u64) {
        // Intentional no-op.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::WebhookDispatcher;
    use crate::endpoint::Endpoint;
    use crate::retry::ExponentialBackoffPolicy;
    use crate::store::{InMemoryWebhookStore, WebhookStore};
    use crate::transport::MockTransport;

    #[test]
    fn webhook_emitter_dispatches_to_subscribers() {
        let store: Arc<dyn WebhookStore> = Arc::new(InMemoryWebhookStore::new());
        let mock = Arc::new(MockTransport::new());
        mock.push_ok();
        let transport: Arc<dyn crate::transport::HttpTransport> = mock.clone();
        let retry: Arc<dyn crate::retry::RetryPolicy> =
            Arc::new(ExponentialBackoffPolicy::stripe_like());
        let ep = Endpoint::new(
            "https://example.test/hook",
            b"shared-secret".to_vec(),
            vec!["refund.created".into()],
        )
        .unwrap();
        store.put_endpoint(ep).unwrap();
        let dispatcher = Arc::new(WebhookDispatcher::new(store, transport, retry));
        let emitter = WebhookEmitter::new(dispatcher);
        emitter.emit("refund.created", b"{\"id\":\"x\"}".to_vec(), 1_700_000_000);
        // One outbound request should have hit the mock transport.
        assert_eq!(mock.captured_count(), 1);
        let captured = mock.take_captured();
        assert_eq!(captured[0].url, "https://example.test/hook");
    }

    #[test]
    fn noop_emitter_is_silent() {
        let e = NoOpEmitter::new();
        e.emit("anything", b"".to_vec(), 0);
        // No assertion needed — the test is that nothing panicked
        // and there's no observable side effect.
    }
}
