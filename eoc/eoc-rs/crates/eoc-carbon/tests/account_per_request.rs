//! Spec example: 1000 J at 400 gCO2e/kWh, PUE 1.2 → 0.1333... gCO2e.

use chrono::Utc;
use eoc_carbon::account::account_request;
use eoc_carbon::intensity::{CarbonIntensity, IntensityKind, ProviderKind};
use eoc_carbon::pue::Pue;
use eoc_core::{JouleCost, JouleSource};

#[test]
fn one_kilojoule_400g_pue_1_2() {
    let cost = JouleCost {
        microjoules: 1_000_000_000, // 1000 J
        source: JouleSource::Measured,
    };
    let intensity = CarbonIntensity::new(
        "EXAMPLE",
        400.0,
        IntensityKind::Average,
        ProviderKind::Mock,
        Utc::now(),
    );
    let pue = Pue::new(1.2);
    let acc = account_request(&cost, &intensity, &pue);
    // 1000 / 3.6e6 * 400 * 1.2 = 0.13333...
    assert!(
        (acc.g_co2e - 0.13333333).abs() < 1e-5,
        "got {} gCO2e",
        acc.g_co2e
    );
    assert!((acc.facility_joules - 1200.0).abs() < 1e-9);
    assert_eq!(acc.pue, 1.2);
    assert_eq!(acc.g_co2e_per_kwh, 400.0);
}
