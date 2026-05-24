//! Picks the greenest of three regions over a 6-hour horizon.

use chrono::{Duration, Utc};
use eoc_carbon::intensity::{ForecastPoint, Zone};
use eoc_carbon::scheduler::{RegionForecast, RegionScheduler};

fn region(name: &str, now: chrono::DateTime<Utc>, values: &[f64]) -> RegionForecast {
    RegionForecast {
        zone: Zone::new(name),
        forecast: values
            .iter()
            .enumerate()
            .map(|(i, g)| ForecastPoint {
                at: now + Duration::hours(i as i64 + 1),
                g_co2e_per_kwh: *g,
            })
            .collect(),
    }
}

#[test]
fn picks_lowest_mean_region() {
    let now = Utc::now();
    let candidates = vec![
        region("US-VA",         now, &[400.0, 410.0, 420.0, 430.0, 440.0, 450.0]),
        region("EU-FR",         now, &[ 60.0,  65.0,  55.0,  70.0,  62.0,  58.0]),
        region("AU-NSW",        now, &[600.0, 610.0, 620.0, 630.0, 640.0, 650.0]),
    ];
    let s = RegionScheduler::new(Duration::hours(6));
    let pick = s.pick_at(&candidates, now).expect("ok");
    assert_eq!(pick.zone.as_str(), "EU-FR");
}

#[test]
fn empty_candidate_set_errors() {
    let s = RegionScheduler::new(Duration::hours(1));
    let r = s.pick_at(&[], Utc::now());
    assert!(r.is_err());
}
