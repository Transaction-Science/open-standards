//! End-to-end: joules (from eoc-meter stub) → CarbonAccount → SCI value.
//!
//! We exercise the joule pipeline by using `eoc_meter::detect()` which
//! always returns *some* counter on every platform, then drive the SCI
//! computation with a deterministic test reading.

use chrono::Utc;
use eoc_carbon::intensity::{CarbonIntensity, IntensityKind, ProviderKind};
use eoc_carbon::pue::Pue;
use eoc_carbon::sci_bridge::{SciInputs, SciScore};
use eoc_core::JouleCost;

#[test]
fn full_request_to_sci_uses_meter_and_carbon_pipeline() {
    // Step 1 — joule meter contract. detect() must always succeed and
    // every counter must respond to read_microjoules().
    let counter = eoc_meter::detect();
    let _ = counter.read_microjoules();
    assert!(!counter.name().is_empty());

    // Step 2 — pretend the meter reported 2000 J on the stub backend.
    let joule_cost = JouleCost::measured(2_000_000_000);

    // Step 3 — grid intensity for the request's region.
    let intensity = CarbonIntensity::new(
        "EU-FR",
        60.0,
        IntensityKind::Average,
        ProviderKind::Mock,
        Utc::now(),
    );

    // Step 4 — PUE for the datacenter.
    let pue = Pue::new(1.1);

    // Step 5 — one inference served, R = 1.
    let sci = SciScore::compute(SciInputs {
        joule_cost: &joule_cost,
        intensity: &intensity,
        pue: &pue,
        embodied_g_co2e: 0.05,
        functional_units: 1.0,
    });

    // 2000 J * 1.1 = 2200 J = 2200 / 3.6e6 kWh ≈ 6.111e-4 kWh.
    // operational = 6.111e-4 * 60 = 0.03667 gCO2e.
    // SCI = (0.03667 + 0.05) / 1 = 0.08667.
    assert!((sci.energy_kwh - 6.1111e-4).abs() < 1e-6, "got {}", sci.energy_kwh);
    assert!((sci.operational_g_co2e - 0.03667).abs() < 1e-4);
    assert_eq!(sci.embodied_g_co2e, 0.05);
    assert!((sci.sci - 0.08667).abs() < 1e-4);
}
