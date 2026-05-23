//! [`ScoringContext`] — caller-supplied signals beyond the Payment.
//!
//! Fraud detection needs more than the payment itself. It needs:
//!
//! - Velocity: how many payments has this customer/device made recently?
//! - Recency: time since last payment
//! - Device fingerprint: device ID, OS, app version
//! - Geo: country code, optionally lat/lon
//! - Time-of-day patterns
//!
//! The orchestrator computes these from its own storage and passes them
//! in. This crate doesn't query databases; it does pure-function scoring.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// Caller-supplied context for fraud scoring.
///
/// All fields are optional; missing data degrades the score signal but
/// does not fail scoring. The heuristic scorer treats `None` as
/// "neutral" (no signal either way).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoringContext {
    /// Number of payments made by this customer in the last hour.
    /// `None` = unknown (e.g. first-time customer with no history).
    pub velocity_1h: Option<u32>,
    /// Number of payments made by this customer in the last 24h.
    pub velocity_24h: Option<u32>,
    /// Number of payments from this device in the last hour.
    pub device_velocity_1h: Option<u32>,
    /// Seconds since this customer's previous payment.
    /// `None` = no prior payment recorded.
    pub seconds_since_last_payment: Option<u64>,
    /// ISO 3166-1 alpha-2 country code (e.g. `"US"`, `"BR"`, `"DE"`).
    pub geo_country: Option<String>,
    /// Whether the geo of this payment matches the customer's usual geo.
    /// `None` = unknown.
    pub geo_matches_history: Option<bool>,
    /// Stable device id (operator-generated, e.g. UUID v4 hashed). Goes
    /// through SHA-256 in feature extraction.
    pub device_id: Option<String>,
    /// App or SDK version (helps catch downgrade attacks).
    pub app_version: Option<String>,
    /// True if the customer's account is newer than `new_customer_threshold`
    /// (default: 7 days).
    pub is_new_customer: Option<bool>,
    /// Authentication recency in seconds (how long since the customer
    /// last performed strong auth).
    pub seconds_since_auth: Option<u64>,
    /// Time of the transaction. Defaults to now if unset.
    pub timestamp: Option<OffsetDateTime>,
}

impl ScoringContext {
    /// Minimal context with no signals. Useful as a baseline / testing.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Mark all velocity fields as known-zero. Use when we've checked the
    /// customer's history and confirmed they have no prior payments.
    #[must_use]
    pub fn fresh_customer() -> Self {
        Self {
            velocity_1h: Some(0),
            velocity_24h: Some(0),
            device_velocity_1h: Some(0),
            seconds_since_last_payment: None,
            is_new_customer: Some(true),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_context_has_all_none() {
        let c = ScoringContext::empty();
        assert!(c.velocity_1h.is_none());
        assert!(c.velocity_24h.is_none());
        assert!(c.device_velocity_1h.is_none());
        assert!(c.seconds_since_last_payment.is_none());
        assert!(c.geo_country.is_none());
        assert!(c.is_new_customer.is_none());
    }

    #[test]
    fn fresh_customer_marks_velocity_zero() {
        let c = ScoringContext::fresh_customer();
        assert_eq!(c.velocity_1h, Some(0));
        assert_eq!(c.velocity_24h, Some(0));
        assert_eq!(c.device_velocity_1h, Some(0));
        assert_eq!(c.is_new_customer, Some(true));
        // seconds_since_last_payment stays None (no prior payment exists)
        assert!(c.seconds_since_last_payment.is_none());
    }

    #[test]
    fn context_round_trips_through_json() {
        let mut c = ScoringContext::empty();
        c.velocity_1h = Some(3);
        c.geo_country = Some("US".into());
        c.is_new_customer = Some(false);
        let json = serde_json::to_string(&c).unwrap();
        let back: ScoringContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back.velocity_1h, Some(3));
        assert_eq!(back.geo_country.as_deref(), Some("US"));
        assert_eq!(back.is_new_customer, Some(false));
    }
}
