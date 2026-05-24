//! Electricity Maps client.
//!
//! Live HTTP lives behind the `http` feature. By default the
//! [`MockElectricityMaps`] backend services every call from an
//! in-memory map, which is what the unit tests exercise.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

use crate::error::{CarbonError, Result};
use crate::intensity::{CarbonIntensity, ForecastPoint, IntensityKind, ProviderKind, Zone};

/// Anything that can answer Electricity Maps queries.
#[async_trait]
pub trait ElectricityMapsBackend: Send + Sync {
    /// Latest reading for `zone` — corresponds to
    /// `GET /v3/carbon-intensity/latest?zone=...`.
    async fn latest(&self, zone: &Zone) -> Result<CarbonIntensity>;

    /// Short-horizon forecast for `zone` — corresponds to
    /// `GET /v3/carbon-intensity/forecast?zone=...`.
    async fn forecast(&self, zone: &Zone) -> Result<Vec<ForecastPoint>>;
}

/// In-process mock — what the tests use and what every offline / CI
/// environment will fall back to.
#[derive(Debug, Default)]
pub struct MockElectricityMaps {
    /// Latest reading per zone (gCO2e/kWh).
    latest: HashMap<String, f64>,
    /// Forecast curve per zone (gCO2e/kWh at +1h, +2h, ...).
    forecasts: HashMap<String, Vec<f64>>,
}

impl MockElectricityMaps {
    /// Empty backend; load with [`with_latest`](Self::with_latest) and
    /// [`with_forecast`](Self::with_forecast).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the latest reading for a zone.
    pub fn with_latest(mut self, zone: impl Into<String>, g_co2e_per_kwh: f64) -> Self {
        self.latest.insert(zone.into(), g_co2e_per_kwh);
        self
    }

    /// Set a forecast curve (hourly values) for a zone.
    pub fn with_forecast(mut self, zone: impl Into<String>, curve: Vec<f64>) -> Self {
        self.forecasts.insert(zone.into(), curve);
        self
    }

    /// Parse a raw `/v3/carbon-intensity/latest` JSON body into a
    /// [`CarbonIntensity`]. Public so tests can verify wire-format
    /// roundtripping without spinning up an HTTP server.
    pub fn parse_latest(json: &str) -> Result<CarbonIntensity> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let zone = v
            .get("zone")
            .and_then(|x| x.as_str())
            .ok_or_else(|| CarbonError::Decode("missing zone".into()))?;
        let g = v
            .get("carbonIntensity")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| CarbonError::Decode("missing carbonIntensity".into()))?;
        let at = v
            .get("datetime")
            .and_then(|x| x.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);
        Ok(CarbonIntensity::new(
            zone,
            g,
            IntensityKind::Average,
            ProviderKind::ElectricityMaps,
            at,
        ))
    }
}

#[async_trait]
impl ElectricityMapsBackend for MockElectricityMaps {
    async fn latest(&self, zone: &Zone) -> Result<CarbonIntensity> {
        let g = self
            .latest
            .get(zone.as_str())
            .copied()
            .ok_or_else(|| CarbonError::UnknownZone(zone.as_str().to_string()))?;
        Ok(CarbonIntensity::new(
            zone.clone(),
            g,
            IntensityKind::Average,
            ProviderKind::ElectricityMaps,
            Utc::now(),
        ))
    }

    async fn forecast(&self, zone: &Zone) -> Result<Vec<ForecastPoint>> {
        let curve = self
            .forecasts
            .get(zone.as_str())
            .ok_or_else(|| CarbonError::UnknownZone(zone.as_str().to_string()))?;
        let now = Utc::now();
        Ok(curve
            .iter()
            .enumerate()
            .map(|(i, g)| ForecastPoint {
                at: now + Duration::hours(i as i64 + 1),
                g_co2e_per_kwh: *g,
            })
            .collect())
    }
}

/// Live HTTP client. Compiled only with the `http` feature.
#[cfg(feature = "http")]
pub struct LiveElectricityMaps {
    api_key: String,
    base: String,
    http: reqwest::Client,
}

#[cfg(feature = "http")]
impl LiveElectricityMaps {
    /// Construct a client for `https://api.electricitymap.org`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base: "https://api.electricitymap.org".into(),
            http: reqwest::Client::new(),
        }
    }

    /// Override the base URL (testing).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }
}

#[cfg(feature = "http")]
#[async_trait]
impl ElectricityMapsBackend for LiveElectricityMaps {
    async fn latest(&self, zone: &Zone) -> Result<CarbonIntensity> {
        let url = format!("{}/v3/carbon-intensity/latest?zone={}", self.base, zone);
        let body = self
            .http
            .get(&url)
            .header("auth-token", &self.api_key)
            .send()
            .await
            .map_err(|e| CarbonError::Http(e.to_string()))?
            .text()
            .await
            .map_err(|e| CarbonError::Http(e.to_string()))?;
        MockElectricityMaps::parse_latest(&body)
    }

    async fn forecast(&self, zone: &Zone) -> Result<Vec<ForecastPoint>> {
        let url = format!("{}/v3/carbon-intensity/forecast?zone={}", self.base, zone);
        let body = self
            .http
            .get(&url)
            .header("auth-token", &self.api_key)
            .send()
            .await
            .map_err(|e| CarbonError::Http(e.to_string()))?
            .text()
            .await
            .map_err(|e| CarbonError::Http(e.to_string()))?;
        let v: serde_json::Value = serde_json::from_str(&body)?;
        let arr = v
            .get("forecast")
            .and_then(|x| x.as_array())
            .ok_or_else(|| CarbonError::Decode("missing forecast".into()))?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let at = item
                .get("datetime")
                .and_then(|x| x.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|d| d.with_timezone(&Utc))
                .ok_or_else(|| CarbonError::Decode("missing datetime".into()))?;
            let g = item
                .get("carbonIntensity")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| CarbonError::Decode("missing carbonIntensity".into()))?;
            out.push(ForecastPoint {
                at,
                g_co2e_per_kwh: g,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_latest_roundtrip() {
        let m = MockElectricityMaps::new().with_latest("FR", 60.0);
        let ci = m.latest(&Zone::new("FR")).await.expect("ok");
        assert_eq!(ci.g_co2e_per_kwh, 60.0);
        assert_eq!(ci.provider, ProviderKind::ElectricityMaps);
    }

    #[tokio::test]
    async fn mock_unknown_zone_errors() {
        let m = MockElectricityMaps::new();
        let r = m.latest(&Zone::new("XX")).await;
        assert!(matches!(r, Err(CarbonError::UnknownZone(_))));
    }
}
