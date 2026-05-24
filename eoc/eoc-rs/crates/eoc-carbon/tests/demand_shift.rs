//! Queue background work for the lowest-intensity window inside a 12h horizon.

use chrono::{Duration, Utc};
use eoc_carbon::intensity::ForecastPoint;
use eoc_carbon::scheduler::DemandShifter;

#[test]
fn shifts_to_greenest_window_in_next_12h() {
    let now = Utc::now();
    // 12 hourly points; minimum at hour 8.
    let curve: Vec<ForecastPoint> = (1..=12)
        .map(|h| ForecastPoint {
            at: now + Duration::hours(h),
            g_co2e_per_kwh: if h == 8 { 90.0 } else { 300.0 + (h as f64) * 5.0 },
        })
        .collect();
    let d = DemandShifter::new(Duration::hours(12));
    let decision = d.pick_window(&curve, now).expect("ok");
    assert_eq!(decision.g_co2e_per_kwh, 90.0);
    assert_eq!(decision.at, now + Duration::hours(8));
}

#[test]
fn ignores_points_outside_horizon() {
    let now = Utc::now();
    let curve = vec![
        ForecastPoint { at: now + Duration::hours(2), g_co2e_per_kwh: 400.0 },
        ForecastPoint { at: now + Duration::hours(20), g_co2e_per_kwh: 50.0 }, // outside 12h
    ];
    let d = DemandShifter::new(Duration::hours(12));
    let decision = d.pick_window(&curve, now).expect("ok");
    assert_eq!(decision.g_co2e_per_kwh, 400.0);
}
