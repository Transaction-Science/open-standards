//! [`Route`] — what a router selects.
//!
//! A `Route` is the unit of choice in this crate. It is conceptually
//! a richer cousin of `op-orchestrator::RailChoice` — same `(rail,
//! driver)` core plus the metadata cost / MCC / retry reasoning
//! needs.
//!
//! The orchestrator's `RailChoice` is intentionally minimal because
//! its router is static. Here we model:
//!
//! - The driver id (PSP name).
//! - The rail kind (Card / A2A / Wallet / Qr / Crypto).
//! - The destination country (ISO 3166-1 alpha-2), because
//!   interchange and scheme-fee tables are country-specific.
//! - An auth-rate score in basis points (operator-supplied prior).
//!   Used by [`LeastCostRouter`](crate::LeastCostRouter) when the
//!   `auth_rate_bias` knob is enabled to break ties / outweigh cost
//!   for routes with materially better acceptance.
//!
//! Operators bridge a `Route` back to the orchestrator's
//! `RailChoice` by reading `driver` and `rail`.

use op_core::RailKind;
use serde::{Deserialize, Serialize};

/// Driver identifier (PSP name). Newtype around `String` so the
/// type system flags accidental swaps with country / MCC strings.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DriverId(pub String);

impl DriverId {
    /// Construct a new driver id.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow as `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for DriverId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for DriverId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl core::fmt::Display for DriverId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A candidate route — driver, rail, country, and an auth-rate
/// prior in basis points (0–10000 bp = 0–100%).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Route {
    /// Driver id (PSP name).
    pub driver: DriverId,

    /// Rail family. Reused from `op-core` so we don't fork the
    /// taxonomy.
    pub rail: RailKind,

    /// ISO 3166-1 alpha-2 destination country, if known. Used by
    /// interchange tier tables.
    pub country: Option<String>,

    /// Operator-supplied auth-rate prior, in basis points
    /// (`0..=10_000`). `None` means "no prior available" and the
    /// router treats it as the neutral midpoint when comparing.
    pub auth_rate_bps: Option<u16>,
}

impl Route {
    /// Construct a route.
    #[must_use]
    pub const fn new(driver: DriverId, rail: RailKind) -> Self {
        Self {
            driver,
            rail,
            country: None,
            auth_rate_bps: None,
        }
    }

    /// Builder: set destination country.
    #[must_use]
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Builder: set auth-rate prior in basis points.
    ///
    /// `bps` is clamped to `0..=10_000`.
    #[must_use]
    pub fn with_auth_rate_bps(mut self, bps: u16) -> Self {
        self.auth_rate_bps = Some(bps.min(10_000));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_builder_chains() {
        let r = Route::new(DriverId::new("stripe"), RailKind::Card)
            .with_country("US")
            .with_auth_rate_bps(9_500);
        assert_eq!(r.driver.as_str(), "stripe");
        assert_eq!(r.rail, RailKind::Card);
        assert_eq!(r.country.as_deref(), Some("US"));
        assert_eq!(r.auth_rate_bps, Some(9_500));
    }

    #[test]
    fn auth_rate_clamps() {
        let r = Route::new(DriverId::new("x"), RailKind::Card).with_auth_rate_bps(50_000);
        assert_eq!(r.auth_rate_bps, Some(10_000));
    }

    #[test]
    fn driver_id_from_str() {
        let d: DriverId = "adyen".into();
        assert_eq!(d.as_str(), "adyen");
        assert_eq!(format!("{d}"), "adyen");
    }
}
