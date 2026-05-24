//! Core types for carbon intensity readings and zone catalogs.
//!
//! A [`CarbonIntensity`] is a gCO2e/kWh number plus provenance (which
//! provider, which zone, average vs marginal, when it was sampled). All
//! providers in this crate normalize their feed to this type.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Which carbon-intensity provider sourced this reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderKind {
    /// Electricity Maps (free + paid tiers).
    ElectricityMaps,
    /// WattTime (marginal emissions / MOER).
    WattTime,
    /// CO2 Signal (consumer-grade Electricity Maps mirror).
    Co2Signal,
    /// IEA per-country baseline (no live feed).
    IeaBaseline,
    /// Hand-injected (testing / offline replay).
    Mock,
}

impl ProviderKind {
    /// Stable short tag, useful in logs.
    pub fn tag(&self) -> &'static str {
        match self {
            ProviderKind::ElectricityMaps => "em",
            ProviderKind::WattTime => "wt",
            ProviderKind::Co2Signal => "co2sig",
            ProviderKind::IeaBaseline => "iea",
            ProviderKind::Mock => "mock",
        }
    }
}

/// Average vs marginal emissions. Electricity Maps publishes both; WattTime
/// is marginal-first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntensityKind {
    /// Average grid intensity for the period.
    Average,
    /// Marginal — the next kWh of demand would emit this much.
    Marginal,
}

/// A balancing-authority / country / sub-region identifier. We keep this
/// as a wrapper around a short string because every provider uses its own
/// catalog and we don't want to lose round-trip fidelity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Zone(pub String);

impl Zone {
    /// Construct a zone from any string-like value.
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }

    /// Borrow as `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Zone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for Zone {
    fn from(s: &str) -> Self {
        Zone(s.to_string())
    }
}

/// A single carbon-intensity sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarbonIntensity {
    /// Zone the reading covers.
    pub zone: Zone,
    /// gCO2e per kWh.
    pub g_co2e_per_kwh: f64,
    /// Average vs marginal.
    pub kind: IntensityKind,
    /// Provider that returned this reading.
    pub provider: ProviderKind,
    /// When the reading was observed (UTC).
    pub at: DateTime<Utc>,
}

impl CarbonIntensity {
    /// Convenience constructor for tests / mocks.
    pub fn new(
        zone: impl Into<Zone>,
        g_co2e_per_kwh: f64,
        kind: IntensityKind,
        provider: ProviderKind,
        at: DateTime<Utc>,
    ) -> Self {
        Self {
            zone: zone.into(),
            g_co2e_per_kwh,
            kind,
            provider,
            at,
        }
    }

    /// A "now" reading for the world-average grid — only useful as a
    /// last-resort fallback.
    pub fn world_average_now() -> Self {
        Self {
            zone: Zone::new("WORLD"),
            g_co2e_per_kwh: crate::WORLD_AVERAGE_G_CO2E_PER_KWH,
            kind: IntensityKind::Average,
            provider: ProviderKind::IeaBaseline,
            at: Utc::now(),
        }
    }
}

/// A short-horizon forecast point used by the scheduler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastPoint {
    /// When this prediction takes effect.
    pub at: DateTime<Utc>,
    /// Predicted gCO2e/kWh.
    pub g_co2e_per_kwh: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_roundtrip() {
        let z: Zone = "US-CAISO_NORTH".into();
        assert_eq!(z.as_str(), "US-CAISO_NORTH");
        assert_eq!(z.to_string(), "US-CAISO_NORTH");
    }

    #[test]
    fn world_average_is_iea() {
        let ci = CarbonIntensity::world_average_now();
        assert_eq!(ci.provider, ProviderKind::IeaBaseline);
        assert!(ci.g_co2e_per_kwh > 0.0);
    }

    #[test]
    fn provider_tags_are_unique() {
        let tags = [
            ProviderKind::ElectricityMaps.tag(),
            ProviderKind::WattTime.tag(),
            ProviderKind::Co2Signal.tag(),
            ProviderKind::IeaBaseline.tag(),
            ProviderKind::Mock.tag(),
        ];
        let mut sorted = tags.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), tags.len());
    }
}
