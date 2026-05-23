//! Webhook events and per-(event, endpoint) delivery attempts.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque event id. UUID v4.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WebhookEventId(pub Uuid);

impl WebhookEventId {
    /// Generate a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID.
    #[must_use]
    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    /// The underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for WebhookEventId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for WebhookEventId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// A webhook event. The payload is **opaque** — the crate doesn't
/// know JSON from CBOR from raw bytes. Operators encode upstream.
///
/// The `created_at_unix_secs` field is the authoritative event time
/// (used as the retry window anchor). It's caller-supplied for
/// deterministic replay and to handle backfill scenarios.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Stable id used for downstream deduplication.
    pub id: WebhookEventId,
    /// Event type (`"payment.authorized"`, `"ledger.transaction.posted"`,
    /// etc.). Free-form; operators define their taxonomy.
    pub event_type: String,
    /// Opaque body that will be POSTed.
    pub payload: Vec<u8>,
    /// Caller-supplied creation time (unix epoch seconds).
    pub created_at_unix_secs: u64,
}

impl WebhookEvent {
    /// Construct.
    #[must_use]
    pub fn new(event_type: impl Into<String>, payload: Vec<u8>, created_at_unix_secs: u64) -> Self {
        Self {
            id: WebhookEventId::new(),
            event_type: event_type.into(),
            payload,
            created_at_unix_secs,
        }
    }

    /// Builder: set a specific id (for replay or migration).
    #[must_use]
    pub fn with_id(mut self, id: WebhookEventId) -> Self {
        self.id = id;
        self
    }
}

/// Opaque attempt id.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeliveryAttemptId(pub Uuid);

impl DeliveryAttemptId {
    /// Generate a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID.
    #[must_use]
    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }
}

impl Default for DeliveryAttemptId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for DeliveryAttemptId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Lifecycle status of a single delivery attempt.
///
/// Note: a [`WebhookEvent`] may produce multiple
/// [`DeliveryAttempt`]s across retries — the attempts themselves
/// are immutable records; status updates create a new attempt with
/// updated state.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Queued, not yet attempted.
    Pending,
    /// In flight (between dispatch and response).
    InFlight,
    /// HTTP 2xx received.
    Succeeded,
    /// Retryable failure; another attempt is scheduled.
    RetryScheduled,
    /// Retry budget exhausted; this is a dead letter. Operator
    /// must `replay` to dispatch again.
    Failed,
}

impl DeliveryStatus {
    /// True if no further state transitions are expected.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed)
    }
}

/// A single delivery attempt for an (event, endpoint) pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliveryAttempt {
    /// Stable id.
    pub id: DeliveryAttemptId,
    /// Which event this attempt is for.
    pub event_id: WebhookEventId,
    /// Which endpoint this attempt is for.
    pub endpoint_id: crate::endpoint::EndpointId,
    /// 0 for the first try, N for the Nth retry.
    pub attempt_number: u32,
    /// Current status.
    pub status: DeliveryStatus,
    /// HTTP response status code, if a response was received.
    pub http_status: Option<u16>,
    /// HTTP response body excerpt (truncated for storage).
    pub response_body_excerpt: Option<String>,
    /// Wall-clock time the attempt was started (unix epoch seconds).
    pub started_at_unix_secs: u64,
    /// Wall-clock time the attempt completed (unix epoch seconds).
    pub completed_at_unix_secs: Option<u64>,
    /// Earliest wall-clock time this attempt may be re-dispatched
    /// (for `RetryScheduled`).
    pub next_attempt_at_unix_secs: Option<u64>,
    /// Free-form error context.
    pub error: Option<String>,
}

/// Maximum bytes of response body kept for diagnostics.
pub const RESPONSE_BODY_EXCERPT_BYTES: usize = 512;

impl DeliveryAttempt {
    /// Construct a fresh pending attempt.
    #[must_use]
    pub fn new_pending(
        event_id: WebhookEventId,
        endpoint_id: crate::endpoint::EndpointId,
        attempt_number: u32,
        started_at_unix_secs: u64,
    ) -> Self {
        Self {
            id: DeliveryAttemptId::new(),
            event_id,
            endpoint_id,
            attempt_number,
            status: DeliveryStatus::Pending,
            http_status: None,
            response_body_excerpt: None,
            started_at_unix_secs,
            completed_at_unix_secs: None,
            next_attempt_at_unix_secs: None,
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint::EndpointId;

    #[test]
    fn event_id_round_trip() {
        let u = Uuid::new_v4();
        assert_eq!(WebhookEventId::from_uuid(u).as_uuid(), u);
    }

    #[test]
    fn event_id_display_is_uuid() {
        let u = Uuid::new_v4();
        assert_eq!(format!("{}", WebhookEventId::from_uuid(u)), u.to_string());
    }

    #[test]
    fn fresh_event_has_new_id() {
        let a = WebhookEvent::new("test.event", b"{}".to_vec(), 0);
        let b = WebhookEvent::new("test.event", b"{}".to_vec(), 0);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn with_id_overrides() {
        let custom = WebhookEventId::new();
        let e = WebhookEvent::new("t", vec![], 0).with_id(custom);
        assert_eq!(e.id, custom);
    }

    #[test]
    fn delivery_status_terminal_correctness() {
        assert!(!DeliveryStatus::Pending.is_terminal());
        assert!(!DeliveryStatus::InFlight.is_terminal());
        assert!(!DeliveryStatus::RetryScheduled.is_terminal());
        assert!(DeliveryStatus::Succeeded.is_terminal());
        assert!(DeliveryStatus::Failed.is_terminal());
    }

    #[test]
    fn new_pending_attempt_initial_state() {
        let eid = WebhookEventId::new();
        let epid = EndpointId::new();
        let a = DeliveryAttempt::new_pending(eid, epid, 0, 100);
        assert_eq!(a.status, DeliveryStatus::Pending);
        assert_eq!(a.attempt_number, 0);
        assert_eq!(a.event_id, eid);
        assert_eq!(a.endpoint_id, epid);
        assert_eq!(a.started_at_unix_secs, 100);
        assert!(a.completed_at_unix_secs.is_none());
        assert!(a.http_status.is_none());
        assert!(a.error.is_none());
    }
}
