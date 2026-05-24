//! WattTime client. Marginal emissions / MOER signal.
//!
//! WattTime is balancing-authority centric — `ba=CAISO_NORTH`, etc. —
//! and returns a marginal operating emissions rate. Like
//! [`crate::electricity_maps`], the mock backend is the default and the
//! live HTTP path is `http`-feature-gated.

use async_trait::async_trait;
use chrono::{Duration, Utc};
use std::collections::HashMap;

use crate::error::{CarbonError, Result};
use crate::intensity::{CarbonIntensity, ForecastPoint, IntensityKind, ProviderKind, Zone};

/// Anything that can answer WattTime queries.
#[async_trait]
pub trait WattTimeBackend: Send + Sync {
    /// `/index?ba=...` — current MOER for the balancing authority.
    async fn moer_now(&self, ba: &Zone) -> Result<CarbonIntensity>;

    /// `/forecast?ba=...` — short-horizon MOER forecast.
    async fn moer_forecast(&self, ba: &Zone) -> Result<Vec<ForecastPoint>>;
}

/// In-process mock.
#[derive(Debug, Default)]
pub struct MockWattTime {
    moer: HashMap<String, f64>,
    forecasts: HashMap<String, Vec<f64>>,
}

impl MockWattTime {
    /// Empty backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin the current MOER for a balancing authority.
    pub fn with_moer(mut self, ba: impl Into<String>, g_co2e_per_kwh: f64) -> Self {
        self.moer.insert(ba.into(), g_co2e_per_kwh);
        self
    }

    /// Pin a forecast curve for a balancing authority.
    pub fn with_forecast(mut self, ba: impl Into<String>, curve: Vec<f64>) -> Self {
        self.forecasts.insert(ba.into(), curve);
        self
    }
}

#[async_trait]
impl WattTimeBackend for MockWattTime {
    async fn moer_now(&self, ba: &Zone) -> Result<CarbonIntensity> {
        let g = self
            .moer
            .get(ba.as_str())
            .copied()
            .ok_or_else(|| CarbonError::UnknownZone(ba.as_str().to_string()))?;
        Ok(CarbonIntensity::new(
            ba.clone(),
            g,
            IntensityKind::Marginal,
            ProviderKind::WattTime,
            Utc::now(),
        ))
    }

    async fn moer_forecast(&self, ba: &Zone) -> Result<Vec<ForecastPoint>> {
        let curve = self
            .forecasts
            .get(ba.as_str())
            .ok_or_else(|| CarbonError::UnknownZone(ba.as_str().to_string()))?;
        let now = Utc::now();
        Ok(curve
            .iter()
            .enumerate()
            .map(|(i, g)| ForecastPoint {
                at: now + Duration::minutes(5 * (i as i64 + 1)),
                g_co2e_per_kwh: *g,
            })
            .collect())
    }
}

/// Live HTTP client. Compiled only with the `http` feature.
#[cfg(feature = "http")]
pub struct LiveWattTime {
    token: String,
    base: String,
    http: reqwest::Client,
}

#[cfg(feature = "http")]
impl LiveWattTime {
    /// Construct a client. WattTime needs a bearer token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base: "https://api2.watttime.org/v2".into(),
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
impl WattTimeBackend for LiveWattTime {
    async fn moer_now(&self, ba: &Zone) -> Result<CarbonIntensity> {
        let url = format!("{}/index?ba={}", self.base, ba);
        let body = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|e| CarbonError::Http(e.to_string()))?
            .text()
            .await
            .map_err(|e| CarbonError::Http(e.to_string()))?;
        let v: serde_json::Value = serde_json::from_str(&body)?;
        // WattTime returns "moer" in lbs CO2 / MWh by default. We
        // convert to gCO2e/kWh: 1 lb/MWh ≈ 0.4536 g/kWh.
        let lbs_per_mwh = v
            .get("moer")
            .and_then(|x| x.as_f64())
            .or_else(|| v.get("value").and_then(|x| x.as_f64()))
            .ok_or_else(|| CarbonError::Decode("missing moer".into()))?;
        let g_per_kwh = lbs_per_mwh * 0.4536;
        Ok(CarbonIntensity::new(
            ba.clone(),
            g_per_kwh,
            IntensityKind::Marginal,
            ProviderKind::WattTime,
            Utc::now(),
        ))
    }

    async fn moer_forecast(&self, ba: &Zone) -> Result<Vec<ForecastPoint>> {
        let url = format!("{}/forecast?ba={}", self.base, ba);
        let body = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
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
                .get("point_time")
                .and_then(|x| x.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|d| d.with_timezone(&Utc))
                .ok_or_else(|| CarbonError::Decode("missing point_time".into()))?;
            let lbs = item
                .get("value")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| CarbonError::Decode("missing value".into()))?;
            out.push(ForecastPoint {
                at,
                g_co2e_per_kwh: lbs * 0.4536,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_moer_marginal() {
        let m = MockWattTime::new().with_moer("CAISO_NORTH", 250.0);
        let ci = m.moer_now(&Zone::new("CAISO_NORTH")).await.expect("ok");
        assert_eq!(ci.kind, IntensityKind::Marginal);
        assert_eq!(ci.g_co2e_per_kwh, 250.0);
    }
}
