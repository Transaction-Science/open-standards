//! Cost attribution: USD + gCO2e from joules.
//!
//! These are pure-function helpers; they do NOT live-fetch grid intensity
//! (`eoc-carbon` does that). Callers either supply a static intensity or
//! plug in the carbon crate.

use eoc_core::JouleCost;

/// Grams of CO2e per kilowatt-hour. A typical 2024 global average is
/// ~480 gCO2e/kWh; specific grids range from ~30 (Quebec hydro) to ~1000
/// (high-coal regions).
#[derive(Debug, Clone, Copy)]
pub struct CarbonIntensityGCo2ePerKwh(pub f64);

/// USD per kilowatt-hour, e.g. retail electricity rate. US 2024 average
/// is ~$0.16/kWh.
#[derive(Debug, Clone, Copy)]
pub struct EnergyPriceUsdPerKwh(pub f64);

/// Power-usage-effectiveness multiplier (facility / IT). A modern hyperscale
/// DC runs ~1.1-1.2; on-prem typically 1.5-2.0.
#[derive(Debug, Clone, Copy)]
pub struct Pue(pub f64);

impl Default for Pue {
    fn default() -> Self {
        Pue(1.0)
    }
}

/// Per-request attribution record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostAttribution {
    /// Energy at the IT layer in joules.
    pub it_joules: f64,
    /// Energy at the facility layer in joules (it_joules * PUE).
    pub facility_joules: f64,
    /// PUE multiplier used.
    pub pue: f64,
    /// gCO2e/kWh intensity used.
    pub g_co2e_per_kwh: f64,
    /// USD/kWh price used.
    pub usd_per_kwh: f64,
    /// Total gCO2e for this request.
    pub g_co2e: f64,
    /// Total USD for this request.
    pub usd: f64,
}

/// Convert joules to kilowatt-hours. `1 kWh = 3.6e6 J`.
pub fn joules_to_kwh(joules: f64) -> f64 {
    joules / 3.6e6
}

/// Attribute USD + gCO2e cost to one joule reading.
pub fn attribute(
    cost: &JouleCost,
    intensity: CarbonIntensityGCo2ePerKwh,
    price: EnergyPriceUsdPerKwh,
    pue: Pue,
) -> CostAttribution {
    let it_joules = cost.joules();
    let facility_joules = it_joules * pue.0;
    let facility_kwh = joules_to_kwh(facility_joules);
    let g_co2e = facility_kwh * intensity.0;
    let usd = facility_kwh * price.0;
    CostAttribution {
        it_joules,
        facility_joules,
        pue: pue.0,
        g_co2e_per_kwh: intensity.0,
        usd_per_kwh: price.0,
        g_co2e,
        usd,
    }
}
