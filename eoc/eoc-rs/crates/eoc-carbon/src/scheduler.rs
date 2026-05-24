//! Region-aware scheduling and demand shifting.
//!
//! * [`RegionScheduler`] picks the region with the lowest forecast
//!   intensity over the next N minutes.
//! * [`DemandShifter`] queues background work to run in the greenest
//!   window of the next N hours, returning a [`ShiftDecision`].
//!
//! Both operate on raw [`ForecastPoint`] streams so they can be driven
//! by Electricity Maps, WattTime, or any other source.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{CarbonError, Result};
use crate::intensity::{ForecastPoint, Zone};

/// One candidate region in a scheduling decision.
#[derive(Debug, Clone)]
pub struct RegionForecast {
    /// Zone tag (matches whatever the provider uses).
    pub zone: Zone,
    /// Forecast curve, sorted ascending by `at`.
    pub forecast: Vec<ForecastPoint>,
}

impl RegionForecast {
    /// Mean gCO2e/kWh over points within `horizon` of `now`. Returns
    /// `None` if no points fall in the window.
    pub fn mean_over(&self, now: DateTime<Utc>, horizon: Duration) -> Option<f64> {
        let cutoff = now + horizon;
        let points: Vec<&ForecastPoint> = self
            .forecast
            .iter()
            .filter(|p| p.at <= cutoff)
            .collect();
        if points.is_empty() {
            return None;
        }
        let n = points.len() as f64;
        let sum: f64 = points.iter().map(|p| p.g_co2e_per_kwh).sum();
        Some(sum / n)
    }
}

/// Picks the greenest region from a set of candidates.
#[derive(Debug, Default)]
pub struct RegionScheduler {
    horizon: Duration,
}

impl RegionScheduler {
    /// Construct a scheduler that averages over the next `horizon`.
    pub fn new(horizon: Duration) -> Self {
        Self { horizon }
    }

    /// Pick the region with the lowest mean forecast intensity over the
    /// configured horizon. Ties go to the first region in `candidates`.
    pub fn pick<'a>(&self, candidates: &'a [RegionForecast]) -> Result<&'a RegionForecast> {
        self.pick_at(candidates, Utc::now())
    }

    /// Same as [`pick`](Self::pick) but with an explicit "now" — useful
    /// for deterministic tests.
    pub fn pick_at<'a>(
        &self,
        candidates: &'a [RegionForecast],
        now: DateTime<Utc>,
    ) -> Result<&'a RegionForecast> {
        if candidates.is_empty() {
            return Err(CarbonError::Empty);
        }
        let mut best: Option<(&RegionForecast, f64)> = None;
        for c in candidates {
            let Some(m) = c.mean_over(now, self.horizon) else {
                continue;
            };
            match best {
                None => best = Some((c, m)),
                Some((_, bm)) if m < bm => best = Some((c, m)),
                _ => {}
            }
        }
        best.map(|(c, _)| c)
            .ok_or_else(|| CarbonError::NoData("no forecast within horizon".into()))
    }
}

/// A scheduling result for [`DemandShifter`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShiftDecision {
    /// When to run the work.
    pub at: DateTime<Utc>,
    /// Predicted intensity at that time.
    pub g_co2e_per_kwh: f64,
}

/// Queues background work for the greenest window inside a longer
/// horizon (typically 12 hours).
#[derive(Debug)]
pub struct DemandShifter {
    horizon: Duration,
}

impl Default for DemandShifter {
    fn default() -> Self {
        Self {
            horizon: Duration::hours(12),
        }
    }
}

impl DemandShifter {
    /// Construct with an explicit horizon.
    pub fn new(horizon: Duration) -> Self {
        Self { horizon }
    }

    /// Pick the lowest-intensity point in the forecast that falls within
    /// the horizon from `now`.
    pub fn pick_window(
        &self,
        forecast: &[ForecastPoint],
        now: DateTime<Utc>,
    ) -> Result<ShiftDecision> {
        let cutoff = now + self.horizon;
        let mut best: Option<&ForecastPoint> = None;
        for p in forecast.iter().filter(|p| p.at >= now && p.at <= cutoff) {
            match best {
                None => best = Some(p),
                Some(b) if p.g_co2e_per_kwh < b.g_co2e_per_kwh => best = Some(p),
                _ => {}
            }
        }
        best.map(|p| ShiftDecision {
            at: p.at,
            g_co2e_per_kwh: p.g_co2e_per_kwh,
        })
        .ok_or_else(|| CarbonError::NoData("no forecast in horizon".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(now: DateTime<Utc>, vs: &[f64]) -> Vec<ForecastPoint> {
        vs.iter()
            .enumerate()
            .map(|(i, g)| ForecastPoint {
                at: now + Duration::hours(i as i64 + 1),
                g_co2e_per_kwh: *g,
            })
            .collect()
    }

    #[test]
    fn picks_lowest_window() {
        let now = Utc::now();
        let curve = pts(now, &[300.0, 250.0, 200.0, 150.0, 220.0]);
        let d = DemandShifter::new(Duration::hours(12))
            .pick_window(&curve, now)
            .expect("ok");
        assert_eq!(d.g_co2e_per_kwh, 150.0);
    }

    #[test]
    fn empty_set_errors() {
        let s = RegionScheduler::new(Duration::hours(1));
        let r = s.pick(&[]);
        assert!(matches!(r, Err(CarbonError::Empty)));
    }
}
