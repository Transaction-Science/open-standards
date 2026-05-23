//! Per-transaction cryptogram primitives.
//!
//! Network tokens are useless without a cryptogram: the cryptogram is
//! what authenticates the *use* of the token at authorization time.
//! Each network has its own format, but all of them share the same
//! external shape — an opaque blob, an Electronic Commerce Indicator
//! (ECI), and an expiry.
//!
//! - **Visa** issues a TAVV (Token Authentication Verification Value).
//! - **Mastercard** issues an AAV (Accountholder Authentication
//!   Value), surfaced as the UCAF (Universal Cardholder
//!   Authentication Field).
//! - **Amex** issues an AEVV.
//!
//! `OpenPay` normalizes these into a single [`Cryptogram`] struct.
//! Adapters fill `avv` with the network-specific value; downstream
//! code treats it as an opaque base64 string and forwards it to the
//! acquirer.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// A per-transaction cryptogram bound to a network token + amount.
///
/// Cryptograms are single-use and time-bounded. A cryptogram that has
/// passed [`Cryptogram::expires_at`] must be re-fetched from the
/// provider before use; reusing an expired cryptogram will hard-decline
/// at the network.
///
/// > **Note on the time type.** The OpenPay issue spec requested
/// > `chrono::DateTime<Utc>`. The rest of this workspace standardizes
/// > on `time::OffsetDateTime` (see workspace `Cargo.toml`); we use
/// > `time::OffsetDateTime` here to stay consistent and avoid pulling
/// > in a second time crate. Callers that need chrono compatibility
/// > can convert via the standard `time → chrono` helpers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cryptogram {
    /// The Authentication Verification Value, base64-encoded. Format
    /// is network-specific; treat as opaque.
    pub avv: String,
    /// Electronic Commerce Indicator. Network-issued numeric code
    /// (e.g. `"05"` for fully-authenticated Visa, `"02"` for
    /// fully-authenticated Mastercard). Forwarded verbatim to the
    /// acquirer.
    pub eci: String,
    /// When this cryptogram expires. Network policy: typically a few
    /// minutes from issuance. Re-fetch on expiry.
    pub expires_at: OffsetDateTime,
}

impl Cryptogram {
    /// Construct.
    #[must_use]
    pub const fn new(avv: String, eci: String, expires_at: OffsetDateTime) -> Self {
        Self {
            avv,
            eci,
            expires_at,
        }
    }

    /// True if the cryptogram has passed its expiry. Comparison is
    /// against the supplied `now` so callers can inject a clock for
    /// determinism in tests.
    #[must_use]
    pub fn is_expired(&self, now: OffsetDateTime) -> bool {
        now >= self.expires_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::Duration;

    #[test]
    fn cryptogram_expiry_check() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let c = Cryptogram::new("AAAA".into(), "05".into(), now + Duration::minutes(5));
        assert!(!c.is_expired(now));
        assert!(c.is_expired(now + Duration::minutes(6)));
    }
}
