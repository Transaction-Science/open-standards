//! Verifies that a real-shape Electricity Maps `/v3/carbon-intensity/latest`
//! body parses into a `CarbonIntensity` without any HTTP roundtrip.

use eoc_carbon::electricity_maps::MockElectricityMaps;
use eoc_carbon::intensity::{IntensityKind, ProviderKind};

#[test]
fn parse_em_latest_body() {
    let body = r#"{
        "zone": "FR",
        "carbonIntensity": 58,
        "datetime": "2026-05-24T12:00:00Z",
        "updatedAt": "2026-05-24T12:01:00Z",
        "emissionFactorType": "lifecycle",
        "isEstimated": false,
        "estimationMethod": null
    }"#;
    let ci = MockElectricityMaps::parse_latest(body).expect("parse ok");
    assert_eq!(ci.zone.as_str(), "FR");
    assert_eq!(ci.g_co2e_per_kwh, 58.0);
    assert_eq!(ci.kind, IntensityKind::Average);
    assert_eq!(ci.provider, ProviderKind::ElectricityMaps);
}

#[test]
fn missing_zone_decodes_to_error() {
    let body = r#"{"carbonIntensity": 100}"#;
    let r = MockElectricityMaps::parse_latest(body);
    assert!(r.is_err());
}
