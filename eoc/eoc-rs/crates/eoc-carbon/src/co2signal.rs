//! CO2 Signal — the free, consumer-grade Electricity Maps mirror.
//!
//! Wire format is essentially Electricity Maps' `data.carbonIntensity`
//! field plus a `countryCode`. The endpoint is
//! `https://api.co2signal.com/v1/latest?countryCode=...`.

use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;

use crate::error::{CarbonError, Result};
use crate::intensity::{CarbonIntensity, IntensityKind, ProviderKind, Zone};

/// Anything that can answer CO2 Signal queries.
#[async_trait]
pub trait Co2SignalBackend: Send + Sync {
    /// Latest reading for a country code.
    async fn latest(&self, country: &Zone) -> Result<CarbonIntensity>;
}

/// In-process mock backend.
#[derive(Debug, Default)]
pub struct MockCo2Signal {
    latest: HashMap<String, f64>,
}

impl MockCo2Signal {
    /// Empty mock.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin a country's intensity.
    pub fn with_country(mut self, country: impl Into<String>, g_co2e_per_kwh: f64) -> Self {
        self.latest.insert(country.into(), g_co2e_per_kwh);
        self
    }

    /// Parse a raw `/v1/latest` body, exposed so tests can verify
    /// wire-format fidelity.
    pub fn parse_latest(json: &str) -> Result<CarbonIntensity> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let country = v
            .get("countryCode")
            .and_then(|x| x.as_str())
            .ok_or_else(|| CarbonError::Decode("missing countryCode".into()))?;
        let g = v
            .pointer("/data/carbonIntensity")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| CarbonError::Decode("missing data.carbonIntensity".into()))?;
        Ok(CarbonIntensity::new(
            country,
            g,
            IntensityKind::Average,
            ProviderKind::Co2Signal,
            Utc::now(),
        ))
    }
}

#[async_trait]
impl Co2SignalBackend for MockCo2Signal {
    async fn latest(&self, country: &Zone) -> Result<CarbonIntensity> {
        let g = self
            .latest
            .get(country.as_str())
            .copied()
            .ok_or_else(|| CarbonError::UnknownZone(country.as_str().to_string()))?;
        Ok(CarbonIntensity::new(
            country.clone(),
            g,
            IntensityKind::Average,
            ProviderKind::Co2Signal,
            Utc::now(),
        ))
    }
}

/// Live HTTP client. Compiled only with the `http` feature.
#[cfg(feature = "http")]
pub struct LiveCo2Signal {
    api_key: String,
    base: String,
    http: reqwest::Client,
}

#[cfg(feature = "http")]
impl LiveCo2Signal {
    /// Construct a client against `https://api.co2signal.com`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base: "https://api.co2signal.com".into(),
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
impl Co2SignalBackend for LiveCo2Signal {
    async fn latest(&self, country: &Zone) -> Result<CarbonIntensity> {
        let url = format!("{}/v1/latest?countryCode={}", self.base, country);
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
        MockCo2Signal::parse_latest(&body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_lookup() {
        let m = MockCo2Signal::new().with_country("DE", 380.0);
        let ci = m.latest(&Zone::new("DE")).await.expect("ok");
        assert_eq!(ci.g_co2e_per_kwh, 380.0);
        assert_eq!(ci.provider, ProviderKind::Co2Signal);
    }

    #[test]
    fn parse_real_shape() {
        let json = r#"{"countryCode":"DE","data":{"carbonIntensity":380.5}}"#;
        let ci = MockCo2Signal::parse_latest(json).expect("ok");
        assert_eq!(ci.g_co2e_per_kwh, 380.5);
    }
}
