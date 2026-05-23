//! Webhook endpoints.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque endpoint id.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EndpointId(pub Uuid);

impl EndpointId {
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

impl Default for EndpointId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for EndpointId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Endpoint lifecycle status.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EndpointStatus {
    /// Accepting deliveries.
    Active,
    /// Operator-disabled. No deliveries.
    Disabled,
    /// Auto-disabled after `disable_after_consecutive_failures`
    /// consecutive failures.
    AutoDisabled,
}

impl EndpointStatus {
    /// True if the endpoint should not receive new deliveries.
    #[must_use]
    pub const fn is_blocking(self) -> bool {
        matches!(self, Self::Disabled | Self::AutoDisabled)
    }
}

/// A merchant-configured webhook destination.
///
/// The `event_filters` field is a list of event-type globs. The
/// dispatcher delivers an event to an endpoint iff at least one
/// filter matches the event's `event_type`. The wildcard `"*"`
/// matches everything.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    /// Stable id.
    pub id: EndpointId,

    /// Destination URL. Must be `https://...` in production;
    /// `http://localhost:*` is permitted for local development.
    pub url: String,

    /// Shared secret used for HMAC signing. Operators rotate this
    /// out-of-band; we don't model rotation in this phase.
    pub secret: Vec<u8>,

    /// List of event-type filters. `"*"` matches all events; any
    /// other string is matched literally against the event's
    /// `event_type`. Empty list = receives nothing (the dispatcher
    /// treats this as a hard skip).
    pub event_filters: Vec<String>,

    /// Lifecycle status.
    pub status: EndpointStatus,

    /// Number of consecutive failures so far. Reset to 0 on any
    /// successful delivery. Triggers auto-disable when it crosses
    /// the policy's threshold.
    pub consecutive_failures: u32,

    /// Free-form metadata.
    pub metadata: Vec<(String, String)>,
}

impl Endpoint {
    /// Construct a fresh endpoint.
    ///
    /// # Errors
    /// [`crate::Error::InvalidUrl`] if `url` is empty or doesn't start
    /// with `http://` or `https://`.
    /// [`crate::Error::InvalidInput`] if `secret` is empty.
    pub fn new(
        url: impl Into<String>,
        secret: Vec<u8>,
        event_filters: Vec<String>,
    ) -> crate::Result<Self> {
        let url = url.into();
        if url.is_empty() {
            return Err(crate::Error::InvalidUrl("empty".into()));
        }
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err(crate::Error::InvalidUrl(format!("not http(s): {url}")));
        }
        if secret.is_empty() {
            return Err(crate::Error::InvalidInput("endpoint secret empty".into()));
        }
        Ok(Self {
            id: EndpointId::new(),
            url,
            secret,
            event_filters,
            status: EndpointStatus::Active,
            consecutive_failures: 0,
            metadata: Vec::new(),
        })
    }

    /// Does this endpoint subscribe to the given event type?
    #[must_use]
    pub fn matches(&self, event_type: &str) -> bool {
        self.event_filters
            .iter()
            .any(|f| f == "*" || f == event_type)
    }

    /// Builder: append metadata.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_endpoint() -> Endpoint {
        Endpoint::new(
            "https://merchant.example.com/hooks",
            b"whsec_test_v1".to_vec(),
            vec!["*".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn fresh_endpoint_is_active_with_no_failures() {
        let e = ok_endpoint();
        assert_eq!(e.status, EndpointStatus::Active);
        assert_eq!(e.consecutive_failures, 0);
    }

    #[test]
    fn empty_url_rejected() {
        let r = Endpoint::new("", b"s".to_vec(), vec![]);
        assert!(matches!(r, Err(crate::Error::InvalidUrl(_))));
    }

    #[test]
    fn non_http_url_rejected() {
        let r = Endpoint::new("ftp://x", b"s".to_vec(), vec![]);
        assert!(matches!(r, Err(crate::Error::InvalidUrl(_))));
    }

    #[test]
    fn http_localhost_permitted_for_dev() {
        let r = Endpoint::new("http://localhost:8080/h", b"s".to_vec(), vec![]);
        assert!(r.is_ok());
    }

    #[test]
    fn empty_secret_rejected() {
        let r = Endpoint::new("https://x", vec![], vec![]);
        assert!(matches!(r, Err(crate::Error::InvalidInput(_))));
    }

    #[test]
    fn wildcard_matches_anything() {
        let e = ok_endpoint();
        assert!(e.matches("payment.authorized"));
        assert!(e.matches("ledger.transaction.posted"));
        assert!(e.matches("anything.else"));
    }

    #[test]
    fn specific_filter_matches_exactly() {
        let e = Endpoint::new(
            "https://x.example",
            b"s".to_vec(),
            vec!["payment.authorized".to_string()],
        )
        .unwrap();
        assert!(e.matches("payment.authorized"));
        assert!(!e.matches("payment.refunded"));
        assert!(!e.matches("ledger.txn.posted"));
    }

    #[test]
    fn empty_filters_match_nothing() {
        let e = Endpoint::new("https://x.example", b"s".to_vec(), vec![]).unwrap();
        assert!(!e.matches("anything"));
    }

    #[test]
    fn status_is_blocking_for_disabled_variants() {
        assert!(!EndpointStatus::Active.is_blocking());
        assert!(EndpointStatus::Disabled.is_blocking());
        assert!(EndpointStatus::AutoDisabled.is_blocking());
    }

    #[test]
    fn metadata_builder_accumulates() {
        let e = ok_endpoint()
            .with_metadata("team", "platform")
            .with_metadata("env", "prod");
        assert_eq!(e.metadata.len(), 2);
    }

    #[test]
    fn endpoint_id_display_is_uuid() {
        let u = Uuid::new_v4();
        assert_eq!(format!("{}", EndpointId::from_uuid(u)), u.to_string());
    }
}
