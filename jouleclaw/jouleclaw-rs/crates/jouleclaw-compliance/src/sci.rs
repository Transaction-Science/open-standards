//! Software Carbon Intensity (SCI) score calculation per ISO/IEC 21031.
//!
//! Implements the Green Software Foundation's SCI formula:
//!
//! `SCI = ((E × I) + M) / R`
//!
//! where:
//! - `E` — operational energy (kWh)
//! - `I` — marginal grid carbon intensity (gCO2 / kWh)
//! - `M` — embodied carbon amortised over the unit (gCO2)
//! - `R` — functional units (requests, transactions, ...)
//!
//! The result `SCI` is `gCO2 / functional_unit`. Lower is better.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Deployment context for an SCI calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciConfig {
    /// Cloud region or data-centre location (e.g. `"eu-west-1"`).
    pub region: String,
    /// Hardware model identifier.
    pub hardware_model: String,
    /// Expected hardware lifetime in years.
    pub lifetime_years: f64,
    /// Functional-unit name (`"request"`, `"transaction"`, ...).
    pub functional_unit_name: String,
}

/// Estimated embodied carbon broken down by lifecycle stage.
/// All values in grams CO2 equivalent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedCarbonEstimate {
    /// Manufacturing carbon (gCO2e).
    pub manufacturing_gco2: f64,
    /// Transport / logistics carbon (gCO2e).
    pub transport_gco2: f64,
    /// End-of-life processing carbon (gCO2e).
    pub end_of_life_gco2: f64,
}

impl EmbodiedCarbonEstimate {
    /// Total embodied carbon across all stages.
    pub fn total(&self) -> f64 {
        self.manufacturing_gco2 + self.transport_gco2 + self.end_of_life_gco2
    }
}

/// Runtime measurements fed into the SCI formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciMeasurements {
    /// Energy consumed during the period (kWh).
    pub energy_kwh: f64,
    /// Marginal grid carbon intensity (gCO2 / kWh).
    pub carbon_intensity_gco2_kwh: f64,
    /// Functional units served.
    pub functional_unit_count: f64,
}

/// Fully computed SCI score with all input components retained for audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciScore {
    /// Energy consumed (kWh).
    pub energy_kwh: f64,
    /// Carbon intensity (gCO2 / kWh).
    pub carbon_intensity_gco2_kwh: f64,
    /// Embodied carbon (gCO2).
    pub embodied_carbon_gco2: f64,
    /// Functional unit count.
    pub functional_unit_count: f64,
    /// Operational carbon `E × I` (gCO2).
    pub operational_carbon_gco2: f64,
    /// Final SCI value `((E × I) + M) / R` (gCO2 / functional unit).
    pub sci_value: f64,
    /// Configuration used for this calculation.
    pub config: SciConfig,
}

/// Validation errors when computing SCI.
#[derive(Debug, thiserror::Error)]
pub enum SciError {
    /// Functional-unit count was zero.
    #[error("functional unit count must be greater than zero")]
    ZeroFunctionalUnits,
    /// Energy value was negative.
    #[error("energy must not be negative, got {0}")]
    NegativeEnergy(f64),
    /// Carbon-intensity value was negative.
    #[error("carbon intensity must not be negative, got {0}")]
    InvalidCarbonIntensity(f64),
}

/// Compute SCI without input validation. Faster, but will return
/// `inf` / `NaN` on degenerate input.
pub fn calculate_sci(
    config: SciConfig,
    measurements: &SciMeasurements,
    embodied: &EmbodiedCarbonEstimate,
) -> SciScore {
    let operational = measurements.energy_kwh * measurements.carbon_intensity_gco2_kwh;
    let sci_value = (operational + embodied.total()) / measurements.functional_unit_count;
    SciScore {
        energy_kwh: measurements.energy_kwh,
        carbon_intensity_gco2_kwh: measurements.carbon_intensity_gco2_kwh,
        embodied_carbon_gco2: embodied.total(),
        functional_unit_count: measurements.functional_unit_count,
        operational_carbon_gco2: operational,
        sci_value,
        config,
    }
}

/// Compute SCI after validating all inputs.
pub fn calculate_sci_checked(
    config: SciConfig,
    measurements: &SciMeasurements,
    embodied: &EmbodiedCarbonEstimate,
) -> Result<SciScore, SciError> {
    if measurements.functional_unit_count == 0.0 {
        return Err(SciError::ZeroFunctionalUnits);
    }
    if measurements.energy_kwh < 0.0 {
        return Err(SciError::NegativeEnergy(measurements.energy_kwh));
    }
    if measurements.carbon_intensity_gco2_kwh < 0.0 {
        return Err(SciError::InvalidCarbonIntensity(
            measurements.carbon_intensity_gco2_kwh,
        ));
    }
    Ok(calculate_sci(config, measurements, embodied))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(region: &str) -> SciConfig {
        SciConfig {
            region: region.to_string(),
            hardware_model: "TestServer-1U".to_string(),
            lifetime_years: 4.0,
            functional_unit_name: "request".to_string(),
        }
    }

    fn emb() -> EmbodiedCarbonEstimate {
        EmbodiedCarbonEstimate {
            manufacturing_gco2: 1000.0,
            transport_gco2: 200.0,
            end_of_life_gco2: 50.0,
        }
    }

    #[test]
    fn basic_sci_calculation() {
        let m = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 1000.0,
        };
        let score = calculate_sci(cfg("us-east-1"), &m, &emb());
        // (100*400 + 1250) / 1000 = 41.25
        assert!((score.sci_value - 41.25).abs() < 1e-10);
    }

    #[test]
    fn sci_with_zero_embodied() {
        let zero = EmbodiedCarbonEstimate {
            manufacturing_gco2: 0.0,
            transport_gco2: 0.0,
            end_of_life_gco2: 0.0,
        };
        let m = SciMeasurements {
            energy_kwh: 50.0,
            carbon_intensity_gco2_kwh: 200.0,
            functional_unit_count: 100.0,
        };
        let score = calculate_sci(cfg("eu-west-1"), &m, &zero);
        assert!((score.sci_value - 100.0).abs() < 1e-10);
    }

    #[test]
    fn embodied_total() {
        let e = EmbodiedCarbonEstimate {
            manufacturing_gco2: 500.0,
            transport_gco2: 100.0,
            end_of_life_gco2: 25.0,
        };
        assert!((e.total() - 625.0).abs() < 1e-10);
    }

    #[test]
    fn sci_checked_zero_functional_units() {
        let m = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 0.0,
        };
        let r = calculate_sci_checked(cfg("us-east-1"), &m, &emb());
        assert!(matches!(r, Err(SciError::ZeroFunctionalUnits)));
    }

    #[test]
    fn sci_checked_negative_energy() {
        let m = SciMeasurements {
            energy_kwh: -10.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 100.0,
        };
        let r = calculate_sci_checked(cfg("us-east-1"), &m, &emb());
        assert!(matches!(r, Err(SciError::NegativeEnergy(_))));
    }

    #[test]
    fn sci_checked_negative_intensity() {
        let m = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: -50.0,
            functional_unit_count: 100.0,
        };
        let r = calculate_sci_checked(cfg("us-east-1"), &m, &emb());
        assert!(matches!(r, Err(SciError::InvalidCarbonIntensity(_))));
    }

    #[test]
    fn operational_carbon_calculation() {
        let m = SciMeasurements {
            energy_kwh: 25.0,
            carbon_intensity_gco2_kwh: 300.0,
            functional_unit_count: 10.0,
        };
        let score = calculate_sci(cfg("eu-west-1"), &m, &emb());
        assert!((score.operational_carbon_gco2 - 7500.0).abs() < 1e-10);
    }

    #[test]
    fn config_round_trip() {
        let c = cfg("us-east-1");
        let json = serde_json::to_string(&c).expect("serialize");
        let d: SciConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.region, "us-east-1");
    }

    #[test]
    fn score_round_trip() {
        let m = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 400.0,
            functional_unit_count: 1000.0,
        };
        let s = calculate_sci(cfg("us-east-1"), &m, &emb());
        let json = serde_json::to_string(&s).expect("serialize");
        let d: SciScore = serde_json::from_str(&json).expect("deserialize");
        assert!((d.sci_value - s.sci_value).abs() < 1e-10);
    }

    #[test]
    fn high_vs_low_intensity_orders_correctly() {
        let m_h = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 800.0,
            functional_unit_count: 1000.0,
        };
        let m_l = SciMeasurements {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 50.0,
            functional_unit_count: 1000.0,
        };
        let h = calculate_sci(cfg("coal-region"), &m_h, &emb());
        let l = calculate_sci(cfg("hydro-region"), &m_l, &emb());
        assert!(h.sci_value > l.sci_value);
    }
}
