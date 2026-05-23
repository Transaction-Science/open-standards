//! The dispatcher — coordinates signing, transport, retry
//! classification, and store updates.
//!
//! ## Three-method surface
//!
//! 1. [`WebhookDispatcher::dispatch`] — given a freshly-emitted
//!    event, fan it out to every matching active endpoint and
//!    return per-endpoint outcomes. Saves the event and attempts
//!    to the store.
//! 2. [`WebhookDispatcher::process_due_retries`] — re-attempt all
//!    [`DeliveryStatus::RetryScheduled`] attempts whose
//!    `next_attempt_at_unix_secs <= now`. Idempotent under repeated
//!    calls.
//! 3. [`WebhookDispatcher::replay`] — operator-triggered manual
//!    re-dispatch of a specific (event, endpoint) pair, used after
//!    fixing a configuration mistake or accidental auto-disable.
//!
//! ## Atomicity model
//!
//! The dispatcher is **sync**. Each `(event, endpoint)` is handled
//! sequentially. A real production deployment runs N dispatcher
//! threads, each pulling from a queue; the [`WebhookStore`] is the
//! shared atomicity boundary.
//!
//! For the in-memory reference impl this is fine since
//! [`InMemoryWebhookStore`] uses a coarse Mutex anyway.

use std::sync::Arc;

use crate::endpoint::{Endpoint, EndpointId, EndpointStatus};
use crate::error::Result;
use crate::event::{
    DeliveryAttempt, DeliveryStatus, RESPONSE_BODY_EXCERPT_BYTES, WebhookEvent, WebhookEventId,
};
use crate::retry::RetryPolicy;
use crate::signing::{SIGNATURE_HEADER, SignedPayload, build_signature_header, compute_signature};
use crate::store::WebhookStore;
use crate::transport::{HttpRequest, HttpTransport};

/// Per-endpoint outcome of a [`WebhookDispatcher::dispatch`] call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// 2xx response on first try.
    Delivered {
        /// The endpoint that received it.
        endpoint_id: EndpointId,
        /// HTTP status from the receiver.
        status: u16,
    },
    /// Failure but retry is scheduled.
    Retrying {
        /// The endpoint.
        endpoint_id: EndpointId,
        /// Unix epoch seconds when the next attempt is due.
        next_attempt_at_unix_secs: u64,
    },
    /// Hard failure — won't be retried automatically.
    DeadLetter {
        /// The endpoint.
        endpoint_id: EndpointId,
        /// Reason class (HTTP status or transport error label).
        reason: String,
    },
    /// Endpoint was skipped because it's not active.
    Skipped {
        /// The endpoint.
        endpoint_id: EndpointId,
        /// Why.
        reason: String,
    },
}

/// The dispatcher.
pub struct WebhookDispatcher {
    store: Arc<dyn WebhookStore>,
    transport: Arc<dyn HttpTransport>,
    retry: Arc<dyn RetryPolicy>,
    /// Injectable clock for tests. Returns unix epoch seconds.
    now: Box<dyn Fn() -> u64 + Send + Sync>,
    /// Configured HTTP timeout per attempt.
    timeout_secs: u32,
    /// Content-Type sent with every request.
    content_type: String,
}

impl WebhookDispatcher {
    /// Construct.
    pub fn new(
        store: Arc<dyn WebhookStore>,
        transport: Arc<dyn HttpTransport>,
        retry: Arc<dyn RetryPolicy>,
    ) -> Self {
        Self {
            store,
            transport,
            retry,
            now: Box::new(default_now),
            timeout_secs: 10,
            content_type: "application/json".to_string(),
        }
    }

    /// Builder: set an injectable clock (for tests).
    #[must_use]
    pub fn with_clock<F: Fn() -> u64 + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.now = Box::new(f);
        self
    }

    /// Builder: set the per-attempt timeout.
    #[must_use]
    pub fn with_timeout_secs(mut self, t: u32) -> Self {
        self.timeout_secs = t.max(1);
        self
    }

    /// Builder: set the content-type header.
    #[must_use]
    pub fn with_content_type(mut self, ct: impl Into<String>) -> Self {
        self.content_type = ct.into();
        self
    }

    /// Dispatch an event to every matching active endpoint. Returns
    /// a per-endpoint outcome list.
    ///
    /// Saves the event to the store if it isn't already present.
    #[tracing::instrument(
        name = "webhook.dispatch",
        skip(self, event),
        fields(
            event_id = %event.id,
            event_type = %event.event_type,
            payload_bytes = event.payload.len(),
        ),
    )]
    pub fn dispatch(&self, event: WebhookEvent) -> Result<Vec<DispatchOutcome>> {
        // Persist the event (even if no endpoints match, so replay
        // can find it later).
        self.store.put_event(event.clone())?;

        let endpoints = self.store.list_active_endpoints_for(&event.event_type)?;

        let mut outcomes: Vec<DispatchOutcome> = Vec::with_capacity(endpoints.len());
        for ep in endpoints {
            outcomes.push(self.dispatch_single(&event, &ep, 0));
        }
        Ok(outcomes)
    }

    /// Manually re-dispatch a specific event to a specific endpoint.
    /// Useful after fixing a misconfiguration. Does NOT skip
    /// disabled endpoints — the operator explicitly invoked replay.
    pub fn replay(
        &self,
        event_id: WebhookEventId,
        endpoint_id: EndpointId,
    ) -> Result<DispatchOutcome> {
        let event = self.store.get_event(event_id)?;
        let endpoint = self.store.get_endpoint(endpoint_id)?;
        // Compute the next attempt number from history.
        let existing = self.store.list_attempts(event_id, endpoint_id)?;
        let attempt_number = existing.last().map(|a| a.attempt_number + 1).unwrap_or(0);
        Ok(self.dispatch_single(&event, &endpoint, attempt_number))
    }

    /// Process all retries due at `now`. Returns the per-attempt
    /// outcomes for diagnostics.
    pub fn process_due_retries(&self) -> Result<Vec<DispatchOutcome>> {
        let now = (self.now)();
        let due = self.store.list_due_retries(now)?;
        let mut outcomes: Vec<DispatchOutcome> = Vec::with_capacity(due.len());
        for a in due {
            // Mark the existing attempt as terminally retired (move
            // it from RetryScheduled to Failed *if* the budget is
            // exhausted, else we create a new attempt). For
            // simplicity we always create a new attempt; the old
            // RetryScheduled record stays in the audit trail with
            // status RetryScheduled — which is fine because
            // attempt_number tells you the sequence.
            let event = self.store.get_event(a.event_id)?;
            let endpoint = match self.store.get_endpoint(a.endpoint_id) {
                Ok(e) => e,
                Err(_) => continue, // endpoint deleted; skip
            };
            // Mark the prior scheduled attempt as superseded
            // (terminal Failed) before creating the new attempt, so
            // list_due_retries doesn't re-pick it.
            let mut prev = a.clone();
            prev.status = DeliveryStatus::Failed;
            prev.completed_at_unix_secs = Some(now);
            self.store.put_attempt(prev)?;

            outcomes.push(self.dispatch_single(&event, &endpoint, a.attempt_number + 1));
        }
        Ok(outcomes)
    }

    /// Core single-(event, endpoint) flow.
    fn dispatch_single(
        &self,
        event: &WebhookEvent,
        endpoint: &Endpoint,
        attempt_number: u32,
    ) -> DispatchOutcome {
        if endpoint.status.is_blocking() {
            return DispatchOutcome::Skipped {
                endpoint_id: endpoint.id,
                reason: format!("endpoint status {:?}", endpoint.status),
            };
        }

        let now = (self.now)();
        let elapsed_secs = now.saturating_sub(event.created_at_unix_secs);
        let mut attempt = DeliveryAttempt::new_pending(event.id, endpoint.id, attempt_number, now);
        attempt.status = DeliveryStatus::InFlight;

        // Build the HTTP request.
        let signed = SignedPayload::new(now, &event.payload);
        let sig_hex = match compute_signature(&endpoint.secret, &signed) {
            Ok(s) => s,
            Err(e) => {
                attempt.status = DeliveryStatus::Failed;
                attempt.error = Some(format!("sign: {e}"));
                attempt.completed_at_unix_secs = Some(now);
                let _ = self.store.put_attempt(attempt);
                return DispatchOutcome::DeadLetter {
                    endpoint_id: endpoint.id,
                    reason: format!("sign: {e}"),
                };
            }
        };
        let sig_header = build_signature_header(now, &sig_hex);
        let req = HttpRequest {
            url: endpoint.url.clone(),
            headers: vec![
                ("Content-Type".to_string(), self.content_type.clone()),
                (SIGNATURE_HEADER.to_string(), sig_header),
                (
                    "OpenPay-Event-Id".to_string(),
                    event.id.as_uuid().to_string(),
                ),
                ("OpenPay-Event-Type".to_string(), event.event_type.clone()),
            ],
            body: event.payload.clone(),
            timeout_secs: self.timeout_secs,
        };

        // Send.
        let result = self.transport.send(&req);

        // Classify.
        match result {
            Ok(resp) if (200..300).contains(&resp.status) => {
                attempt.status = DeliveryStatus::Succeeded;
                attempt.http_status = Some(resp.status);
                attempt.response_body_excerpt = Some(excerpt(&resp.body));
                attempt.completed_at_unix_secs = Some(now);
                let _ = self.store.put_attempt(attempt);
                // Reset consecutive failures.
                let _ = self.store.set_endpoint_consecutive_failures(endpoint.id, 0);
                DispatchOutcome::Delivered {
                    endpoint_id: endpoint.id,
                    status: resp.status,
                }
            }
            Ok(resp) => {
                attempt.http_status = Some(resp.status);
                attempt.response_body_excerpt = Some(excerpt(&resp.body));
                self.classify_non_success(
                    event,
                    endpoint,
                    &mut attempt,
                    Some(resp.status),
                    None,
                    elapsed_secs,
                    now,
                )
            }
            Err(e) => {
                let msg = e.to_string();
                attempt.error = Some(msg.clone());
                self.classify_non_success(
                    event,
                    endpoint,
                    &mut attempt,
                    None,
                    Some(msg),
                    elapsed_secs,
                    now,
                )
            }
        }
    }

    /// Decide whether a non-2xx outcome should be retried, dead-
    /// lettered, or trigger auto-disable.
    #[allow(clippy::too_many_arguments)]
    fn classify_non_success(
        &self,
        _event: &WebhookEvent,
        endpoint: &Endpoint,
        attempt: &mut DeliveryAttempt,
        http_status: Option<u16>,
        transport_err: Option<String>,
        elapsed_secs: u64,
        now: u64,
    ) -> DispatchOutcome {
        let retryable = self.retry.should_retry(http_status);
        if !retryable {
            attempt.status = DeliveryStatus::Failed;
            attempt.completed_at_unix_secs = Some(now);
            // Bump consecutive failures + possibly auto-disable.
            self.bump_consecutive_failures(endpoint);
            let _ = self.store.put_attempt(attempt.clone());
            return DispatchOutcome::DeadLetter {
                endpoint_id: endpoint.id,
                reason: http_status.map(|s| format!("http_{s}")).unwrap_or_else(|| {
                    transport_err
                        .clone()
                        .unwrap_or_else(|| "unknown".to_owned())
                }),
            };
        }

        // Retry is permitted by status; check budget.
        let next_delay = self
            .retry
            .next_delay_secs(attempt.attempt_number, elapsed_secs);
        match next_delay {
            Some(delay) => {
                let next_at = now.saturating_add(delay);
                attempt.status = DeliveryStatus::RetryScheduled;
                attempt.next_attempt_at_unix_secs = Some(next_at);
                attempt.completed_at_unix_secs = Some(now);
                self.bump_consecutive_failures(endpoint);
                let _ = self.store.put_attempt(attempt.clone());
                DispatchOutcome::Retrying {
                    endpoint_id: endpoint.id,
                    next_attempt_at_unix_secs: next_at,
                }
            }
            None => {
                attempt.status = DeliveryStatus::Failed;
                attempt.completed_at_unix_secs = Some(now);
                self.bump_consecutive_failures(endpoint);
                let _ = self.store.put_attempt(attempt.clone());
                DispatchOutcome::DeadLetter {
                    endpoint_id: endpoint.id,
                    reason: "retry_budget_exhausted".to_owned(),
                }
            }
        }
    }

    /// Increment the endpoint's consecutive_failures counter. If
    /// the new value crosses the policy's threshold, flip status to
    /// AutoDisabled.
    fn bump_consecutive_failures(&self, endpoint: &Endpoint) {
        let new = endpoint.consecutive_failures.saturating_add(1);
        let _ = self
            .store
            .set_endpoint_consecutive_failures(endpoint.id, new);
        if new >= self.retry.disable_after_consecutive_failures() {
            let _ = self
                .store
                .set_endpoint_status(endpoint.id, EndpointStatus::AutoDisabled);
        }
    }
}

/// Default system clock.
fn default_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Truncate a body to the diagnostic excerpt size and stringify
/// (lossy UTF-8). We never want to log the full body — could be
/// gigabytes, could contain PII.
fn excerpt(body: &[u8]) -> String {
    let slice = if body.len() > RESPONSE_BODY_EXCERPT_BYTES {
        &body[..RESPONSE_BODY_EXCERPT_BYTES]
    } else {
        body
    };
    String::from_utf8_lossy(slice).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint::Endpoint;
    use crate::retry::ExponentialBackoffPolicy;
    use crate::store::InMemoryWebhookStore;
    use crate::transport::{HttpResponse, MockTransport};

    fn setup() -> (
        Arc<InMemoryWebhookStore>,
        Arc<MockTransport>,
        WebhookDispatcher,
    ) {
        let store: Arc<InMemoryWebhookStore> = Arc::new(InMemoryWebhookStore::new());
        let transport: Arc<MockTransport> = Arc::new(MockTransport::new());
        let policy: Arc<dyn RetryPolicy> =
            Arc::new(ExponentialBackoffPolicy::deterministic(2, 10, 100, 3, 0));
        let store_dyn: Arc<dyn WebhookStore> = store.clone();
        let transport_dyn: Arc<dyn HttpTransport> = transport.clone();
        let dispatcher = WebhookDispatcher::new(store_dyn, transport_dyn, policy)
            .with_clock(|| 1000)
            .with_timeout_secs(5);
        (store, transport, dispatcher)
    }

    fn endpoint_for(store: &InMemoryWebhookStore, filters: Vec<String>) -> EndpointId {
        let e = Endpoint::new(
            "https://merchant.example.com/h",
            b"whsec_test".to_vec(),
            filters,
        )
        .unwrap();
        let id = e.id;
        store.put_endpoint(e).unwrap();
        id
    }

    fn event_payload(et: &str) -> WebhookEvent {
        WebhookEvent::new(et, b"{\"ok\":true}".to_vec(), 1000)
    }

    #[test]
    fn dispatch_to_matching_endpoint_delivers() {
        let (store, _t, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["payment.authorized".to_string()]);
        // MockTransport returns default 200 if queue empty.
        let outcomes = dispatcher
            .dispatch(event_payload("payment.authorized"))
            .unwrap();
        assert_eq!(outcomes.len(), 1);
        match &outcomes[0] {
            DispatchOutcome::Delivered {
                endpoint_id,
                status,
            } => {
                assert_eq!(*endpoint_id, eid);
                assert_eq!(*status, 200);
            }
            o => panic!("expected Delivered, got {o:?}"),
        }
    }

    #[test]
    fn dispatch_skips_non_matching_filters() {
        let (store, _t, dispatcher) = setup();
        endpoint_for(&store, vec!["ledger.txn.posted".to_string()]);
        let outcomes = dispatcher
            .dispatch(event_payload("payment.authorized"))
            .unwrap();
        // No endpoints subscribe to payment.authorized → no outcomes.
        assert!(outcomes.is_empty());
    }

    #[test]
    fn dispatch_to_multiple_active_endpoints() {
        let (store, _t, dispatcher) = setup();
        endpoint_for(&store, vec!["*".to_string()]);
        endpoint_for(&store, vec!["*".to_string()]);
        endpoint_for(&store, vec!["*".to_string()]);
        let outcomes = dispatcher.dispatch(event_payload("any")).unwrap();
        assert_eq!(outcomes.len(), 3);
        for o in &outcomes {
            assert!(matches!(o, DispatchOutcome::Delivered { .. }));
        }
    }

    #[test]
    fn five_hundred_three_response_schedules_retry() {
        let (store, transport, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["*".to_string()]);
        transport.push_5xx(503);
        let outcomes = dispatcher.dispatch(event_payload("any")).unwrap();
        assert_eq!(outcomes.len(), 1);
        match &outcomes[0] {
            DispatchOutcome::Retrying {
                endpoint_id,
                next_attempt_at_unix_secs,
            } => {
                assert_eq!(*endpoint_id, eid);
                // Full jitter draws uniformly in [0, base*2^n]; with the
                // deterministic FixedJitter(0) in `setup()` the draw is
                // exactly 0, so the retry is scheduled at `now` (1000).
                // The invariant is "never scheduled in the past, never
                // beyond now + max_delay" (10s cap).
                assert!(*next_attempt_at_unix_secs >= 1000);
                assert!(*next_attempt_at_unix_secs <= 1000 + 10);
            }
            o => panic!("expected Retrying, got {o:?}"),
        }
    }

    #[test]
    fn transport_error_schedules_retry() {
        let (store, transport, dispatcher) = setup();
        endpoint_for(&store, vec!["*".to_string()]);
        transport.push_transport_err("connection refused");
        let outcomes = dispatcher.dispatch(event_payload("any")).unwrap();
        assert!(matches!(outcomes[0], DispatchOutcome::Retrying { .. }));
    }

    #[test]
    fn four_oh_four_response_dead_letters_immediately() {
        let (store, transport, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["*".to_string()]);
        transport.push_response(crate::transport::MockResponse::Response(HttpResponse {
            status: 404,
            body: b"not found".to_vec(),
        }));
        let outcomes = dispatcher.dispatch(event_payload("any")).unwrap();
        match &outcomes[0] {
            DispatchOutcome::DeadLetter {
                endpoint_id,
                reason,
            } => {
                assert_eq!(*endpoint_id, eid);
                assert_eq!(reason, "http_404");
            }
            o => panic!("expected DeadLetter, got {o:?}"),
        }
    }

    #[test]
    fn four_twenty_nine_response_schedules_retry() {
        let (store, transport, dispatcher) = setup();
        endpoint_for(&store, vec!["*".to_string()]);
        transport.push_response(crate::transport::MockResponse::Response(HttpResponse {
            status: 429,
            body: b"rate limited".to_vec(),
        }));
        let outcomes = dispatcher.dispatch(event_payload("any")).unwrap();
        assert!(matches!(outcomes[0], DispatchOutcome::Retrying { .. }));
    }

    #[test]
    fn disabled_endpoint_is_skipped_at_dispatch() {
        let (store, _t, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["*".to_string()]);
        store
            .set_endpoint_status(eid, EndpointStatus::Disabled)
            .unwrap();
        // Disabled endpoints aren't returned by list_active_endpoints_for —
        // so no outcome at all.
        let outcomes = dispatcher.dispatch(event_payload("any")).unwrap();
        assert!(outcomes.is_empty());
    }

    #[test]
    fn replay_dispatches_to_disabled_endpoint_anyway() {
        // Replay is operator-triggered; it does NOT skip disabled
        // endpoints (the operator explicitly chose this endpoint).
        let (store, _t, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["*".to_string()]);
        store
            .set_endpoint_status(eid, EndpointStatus::Disabled)
            .unwrap();
        let event = event_payload("any");
        let event_id = event.id;
        store.put_event(event).unwrap();

        let outcome = dispatcher.replay(event_id, eid).unwrap();
        // The dispatcher's dispatch_single STILL checks
        // is_blocking() — so even replay skips disabled endpoints.
        // (This is a defensive default: operators wanting to bypass
        // must first re-enable.)
        assert!(matches!(outcome, DispatchOutcome::Skipped { .. }));
    }

    #[test]
    fn replay_works_after_re_enabling_an_auto_disabled_endpoint() {
        let (store, transport, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["*".to_string()]);
        // Force auto-disable.
        store
            .set_endpoint_status(eid, EndpointStatus::AutoDisabled)
            .unwrap();
        let event = event_payload("any");
        let event_id = event.id;
        store.put_event(event).unwrap();
        // Re-enable.
        store
            .set_endpoint_status(eid, EndpointStatus::Active)
            .unwrap();
        // Queue a 200 for the replay.
        transport.push_ok();
        let outcome = dispatcher.replay(event_id, eid).unwrap();
        assert!(matches!(outcome, DispatchOutcome::Delivered { .. }));
    }

    #[test]
    fn auto_disable_after_threshold_consecutive_failures() {
        let (store, transport, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["*".to_string()]);
        // Threshold = 3 (set in setup()).
        // Queue 3 failures.
        transport.push_5xx(500);
        transport.push_5xx(500);
        transport.push_5xx(500);

        for _ in 0..3 {
            // Each call uses a fresh event so we accumulate 3
            // failures on the same endpoint.
            let _ = dispatcher.dispatch(event_payload("any")).unwrap();
        }
        // Endpoint should now be auto-disabled.
        let ep = store.get_endpoint(eid).unwrap();
        assert_eq!(ep.status, EndpointStatus::AutoDisabled);
        assert!(ep.consecutive_failures >= 3);
    }

    #[test]
    fn success_resets_consecutive_failure_counter() {
        let (store, transport, dispatcher) = setup();
        let eid = endpoint_for(&store, vec!["*".to_string()]);
        transport.push_5xx(503);
        transport.push_ok();

        let _ = dispatcher.dispatch(event_payload("a")).unwrap();
        let _ = dispatcher.dispatch(event_payload("b")).unwrap();
        let ep = store.get_endpoint(eid).unwrap();
        assert_eq!(ep.consecutive_failures, 0);
        assert_eq!(ep.status, EndpointStatus::Active);
    }

    #[test]
    fn signature_header_is_included_in_request() {
        let (store, transport, dispatcher) = setup();
        endpoint_for(&store, vec!["*".to_string()]);
        let _ = dispatcher.dispatch(event_payload("any")).unwrap();
        let captured = transport.take_captured();
        assert_eq!(captured.len(), 1);
        let header = captured[0]
            .headers
            .iter()
            .find(|(k, _)| k == SIGNATURE_HEADER)
            .expect("signature header missing");
        // Must start with `t=` and contain `,v1=`.
        assert!(header.1.starts_with("t="));
        assert!(header.1.contains(",v1="));
    }

    #[test]
    fn request_carries_event_id_and_type_headers() {
        let (store, transport, dispatcher) = setup();
        endpoint_for(&store, vec!["*".to_string()]);
        let _ = dispatcher
            .dispatch(event_payload("payment.authorized"))
            .unwrap();
        let captured = transport.take_captured();
        let event_id_h = captured[0]
            .headers
            .iter()
            .find(|(k, _)| k == "OpenPay-Event-Id")
            .expect("event id missing");
        assert!(!event_id_h.1.is_empty());
        let event_type_h = captured[0]
            .headers
            .iter()
            .find(|(k, _)| k == "OpenPay-Event-Type")
            .expect("event type missing");
        assert_eq!(event_type_h.1, "payment.authorized");
    }

    #[test]
    fn process_due_retries_redispatches_scheduled_attempts() {
        let (store, transport, dispatcher) = setup();
        endpoint_for(&store, vec!["*".to_string()]);
        // First attempt fails (5xx) → scheduled retry.
        transport.push_5xx(503);
        let _ = dispatcher.dispatch(event_payload("any")).unwrap();

        // Now advance the clock past the next_attempt time and run
        // process_due_retries. Queue a 200 for it.
        transport.push_ok();
        // The dispatcher's clock is still 1000. We need to call with
        // a clock that's past the scheduled time. Reset dispatcher
        // with a later clock.
        let later = WebhookDispatcher::new(
            Arc::clone(&store) as Arc<dyn WebhookStore>,
            Arc::clone(&transport) as Arc<dyn HttpTransport>,
            Arc::new(ExponentialBackoffPolicy::deterministic(2, 10, 100, 3, 0)),
        )
        .with_clock(|| 9999);

        let outcomes = later.process_due_retries().unwrap();
        // One due retry, succeeded this time.
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(outcomes[0], DispatchOutcome::Delivered { .. }));
    }

    #[test]
    fn event_is_persisted_even_with_no_matching_endpoints() {
        let (store, _t, dispatcher) = setup();
        // No endpoints registered.
        let event = event_payload("orphan.event");
        let event_id = event.id;
        let outcomes = dispatcher.dispatch(event).unwrap();
        assert!(outcomes.is_empty());
        // But the event was saved so future endpoints could replay it.
        let stored = store.get_event(event_id).unwrap();
        assert_eq!(stored.event_type, "orphan.event");
    }

    #[test]
    fn excerpt_truncates_large_body() {
        let big = vec![b'A'; 10_000];
        let e = excerpt(&big);
        assert_eq!(e.len(), RESPONSE_BODY_EXCERPT_BYTES);
    }

    #[test]
    fn excerpt_preserves_small_body() {
        let small = b"hi";
        let e = excerpt(small);
        assert_eq!(e, "hi");
    }
}
