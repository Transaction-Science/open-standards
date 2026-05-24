//! Per-request carbon accounting.
//!
//! `(joules / 3.6e6) * gCO2_per_kWh * PUE → gCO2e`
//!
//! Inputs are typed: joules come from [`eoc_meter`] / [`eoc_core::JouleCost`],
//! intensity comes from [`crate::intensity::CarbonIntensity`], overhead
//! comes from [`crate::pue::Pue`].

use eoc_core::JouleCost;
use serde::{Deserialize, Serialize};

use crate::intensity::CarbonIntensity;
use crate::pue::Pue;

/// One joule in kWh.
const JOULES_PER_KWH: f64 = 3_600_000.0;

/// Carbon attributable to a single resolved request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarbonAccount {
    /// Joules spent at the IT layer (i.e. before PUE multiplier).
    pub it_joules: f64,
    /// Joules spent at the facility layer (i.e. `it_joules * pue`).
    pub facility_joules: f64,
    /// Grid intensity used for the conversion (gCO2e/kWh).
    pub g_co2e_per_kwh: f64,
    /// PUE multiplier applied.
    pub pue: f64,
    /// Final gCO2e number.
    pub g_co2e: f64,
}

/// Anything that turns an `it_joules + intensity + pue` tuple into a
/// [`CarbonAccount`]. Lives as a trait so the eval-bridge and the
/// scheduler can share an accounting strategy.
pub trait CarbonAccounting {
    /// Compute gCO2e for a single request.
    fn account(&self, joule_cost: &JouleCost, intensity: &CarbonIntensity, pue: &Pue)
    -> CarbonAccount;
}

/// The default accounting strategy: straight multiplication. No
/// renewable credits, no offsets, no time-weighting. This is the
/// number a regulator would compute.
#[derive(Debug, Default, Clone, Copy)]
pub struct StandardAccounting;

impl CarbonAccounting for StandardAccounting {
    fn account(
        &self,
        joule_cost: &JouleCost,
        intensity: &CarbonIntensity,
        pue: &Pue,
    ) -> CarbonAccount {
        let it_joules = joule_cost.joules();
        let facility_joules = it_joules * pue.value();
        let kwh = facility_joules / JOULES_PER_KWH;
        let g_co2e = kwh * intensity.g_co2e_per_kwh;
        CarbonAccount {
            it_joules,
            facility_joules,
            g_co2e_per_kwh: intensity.g_co2e_per_kwh,
            pue: pue.value(),
            g_co2e,
        }
    }
}

/// Convenience free function — the most common call shape.
pub fn account_request(
    joule_cost: &JouleCost,
    intensity: &CarbonIntensity,
    pue: &Pue,
) -> CarbonAccount {
    StandardAccounting.account(joule_cost, intensity, pue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intensity::{IntensityKind, ProviderKind};
    use chrono::Utc;
    use eoc_core::JouleSource;

    fn intensity(g: f64) -> CarbonIntensity {
        CarbonIntensity::new("US", g, IntensityKind::Average, ProviderKind::Mock, Utc::now())
    }

    #[test]
    fn spec_example_1000j_400g_pue_1_2() {
        // 1000 J / 3.6e6 = 2.777...e-4 kWh
        // * 400 g/kWh = 0.1111... g
        // * 1.2 PUE = 0.1333... g
        let cost = JouleCost {
            microjoules: 1_000_000_000,
            source: JouleSource::Measured,
        };
        let acc = account_request(&cost, &intensity(400.0), &Pue::new(1.2));
        assert!((acc.g_co2e - 0.133333).abs() < 1e-4, "got {}", acc.g_co2e);
    }

    #[test]
    fn zero_joules_zero_carbon() {
        let acc = account_request(&JouleCost::zero(), &intensity(800.0), &Pue::new(1.4));
        assert_eq!(acc.g_co2e, 0.0);
    }
}
