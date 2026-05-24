//! Bridge to the Green Software Foundation **Software Carbon Intensity
//! (SCI)** specification, v1.
//!
//! SCI is defined as:
//!
//! ```text
//! SCI = ((E * I) + M) / R
//! ```
//!
//! where:
//! - `E` is energy consumed by the software (kWh),
//! - `I` is the location-based marginal grid intensity (gCO2e/kWh),
//! - `M` is embodied emissions amortized over the request (gCO2e),
//! - `R` is the functional unit (e.g. one request, one inference, one
//!   correct answer for an eval).
//!
//! `eoc-eval` already references SCI; this module is the explicit
//! computation surface, and when the `eval-bridge` feature is on it
//! exposes a constructor that turns an [`eoc_eval::EvalReport`] into an
//! [`SciScore`] with `R = number of correct cases`.

use serde::{Deserialize, Serialize};

use crate::account::{CarbonAccount, account_request};
use crate::intensity::CarbonIntensity;
use crate::pue::Pue;
use eoc_core::JouleCost;

/// Inputs to one SCI computation.
#[derive(Debug, Clone)]
pub struct SciInputs<'a> {
    /// Measured joules at the IT layer (from [`eoc_meter`]).
    pub joule_cost: &'a JouleCost,
    /// Grid intensity (marginal preferred, average acceptable).
    pub intensity: &'a CarbonIntensity,
    /// Facility PUE.
    pub pue: &'a Pue,
    /// Embodied emissions amortized to this request, in gCO2e. Set to
    /// 0.0 if you don't track it yet.
    pub embodied_g_co2e: f64,
    /// Functional units served by this request (number of inferences,
    /// number of correct answers, etc.). Must be > 0; values ≤ 0 are
    /// clamped to 1 to keep the score finite.
    pub functional_units: f64,
}

/// A computed SCI score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SciScore {
    /// Energy term (kWh).
    pub energy_kwh: f64,
    /// Operational gCO2e (energy × intensity × PUE).
    pub operational_g_co2e: f64,
    /// Embodied gCO2e passed in.
    pub embodied_g_co2e: f64,
    /// `R` denominator actually used.
    pub functional_units: f64,
    /// The SCI value (gCO2e per functional unit).
    pub sci: f64,
}

impl SciScore {
    /// Compute an SCI score from joules + intensity + PUE + embodied.
    pub fn compute(inputs: SciInputs<'_>) -> Self {
        let account: CarbonAccount = account_request(inputs.joule_cost, inputs.intensity, inputs.pue);
        let energy_kwh = account.facility_joules / 3_600_000.0;
        let operational_g_co2e = account.g_co2e;
        let r = if inputs.functional_units > 0.0 {
            inputs.functional_units
        } else {
            1.0
        };
        let sci = (operational_g_co2e + inputs.embodied_g_co2e) / r;
        Self {
            energy_kwh,
            operational_g_co2e,
            embodied_g_co2e: inputs.embodied_g_co2e,
            functional_units: r,
            sci,
        }
    }
}

/// `eoc-eval` bridge — turns an `EvalReport` into an SCI score with
/// `R = correct`. Compiled only with the `eval-bridge` Cargo feature.
///
/// `EvalReport` carries `joules_per_correct` in micro-joules and a
/// `correct` count, so we reconstruct the total IT-layer energy as
/// `joules_per_correct * correct`.
#[cfg(feature = "eval-bridge")]
pub fn sci_from_eval_report(
    report: &eoc_eval::EvalReport,
    intensity: &CarbonIntensity,
    pue: &Pue,
    embodied_g_co2e: f64,
) -> SciScore {
    let correct = report.correct.max(1);
    let total_microjoules = (report.joules_per_correct as u128)
        .saturating_mul(correct as u128)
        .min(u64::MAX as u128) as u64;
    let cost = JouleCost::measured(total_microjoules);
    SciScore::compute(SciInputs {
        joule_cost: &cost,
        intensity,
        pue,
        embodied_g_co2e,
        functional_units: correct as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intensity::{IntensityKind, ProviderKind};
    use chrono::Utc;

    #[test]
    fn r_clamped_when_zero() {
        let cost = JouleCost::measured(1_000_000); // 1 J
        let ci = CarbonIntensity::new(
            "US",
            400.0,
            IntensityKind::Average,
            ProviderKind::Mock,
            Utc::now(),
        );
        let s = SciScore::compute(SciInputs {
            joule_cost: &cost,
            intensity: &ci,
            pue: &Pue::new(1.2),
            embodied_g_co2e: 0.0,
            functional_units: 0.0,
        });
        assert_eq!(s.functional_units, 1.0);
        assert!(s.sci > 0.0);
    }
}
