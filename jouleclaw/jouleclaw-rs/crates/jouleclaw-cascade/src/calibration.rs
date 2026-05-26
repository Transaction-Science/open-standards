//! Budget enforcement and tier calibration (R7).
//!
//! Three pieces:
//!
//!   1. **Pre-flight check.** The runtime skips a tier whose estimate
//!      exceeds the remaining budget. Already present from R0.
//!
//!   2. **Post-flight check.** After a tier reports its actual spend,
//!      the runtime verifies the spend stayed within the budget it
//!      was given. If a tier overruns its budget cap, the runtime
//!      records this as a budget violation and refuses to count the
//!      tier's answer.
//!
//!   3. **Calibration.** Every tier dispatch records the
//!      `(estimate, actual)` pair. Over time, this data feeds the
//!      tier's cost-model constants. R7 records the data and exposes
//!      stats; R7+ may add self-update loops.

use crate::types::*;
use std::collections::HashMap;

/// Per-tier accumulator of `(estimate, actual)` joule pairs. Used to
/// answer: is this tier's cost model honest? How much does it
/// overestimate or underestimate?
#[derive(Debug, Clone, Default)]
pub struct TierCalibration {
    pub samples: u64,
    /// Sum of (actual / estimate) ratios. A tier whose estimates are
    /// exactly right will have `ratio_sum == samples`. A tier that
    /// consistently underestimates by 2× will have `ratio_sum == 2 *
    /// samples`.
    pub ratio_sum: f64,
    /// Largest single (actual / estimate) ratio observed. Captures
    /// worst-case overshoot for budget-planning purposes.
    pub max_ratio: f64,
    /// Number of times a tier's actual spend exceeded its budget cap.
    pub budget_violations: u64,
    /// Cumulative joules estimated and actually spent.
    pub total_estimated: f64,
    pub total_actual: f64,
}

impl TierCalibration {
    pub fn record(&mut self, estimated: f64, actual: f64) {
        self.samples += 1;
        self.total_estimated += estimated;
        self.total_actual += actual;
        if estimated > 0.0 {
            let r = actual / estimated;
            self.ratio_sum += r;
            if r > self.max_ratio { self.max_ratio = r; }
        }
    }

    pub fn record_violation(&mut self) {
        self.budget_violations += 1;
    }

    /// Mean (actual / estimate) ratio. >1 means tier underestimates.
    pub fn mean_ratio(&self) -> f64 {
        if self.samples == 0 { return 1.0; }
        self.ratio_sum / self.samples as f64
    }
}

/// Calibration data across all tiers in the runtime.
#[derive(Debug, Clone, Default)]
pub struct CalibrationReport {
    pub per_tier: HashMap<TierId, TierCalibration>,
    /// Calibration keyed by Synthesis cell ID. Lets us learn μ at
    /// cell granularity rather than just per-TierId — two tiers with
    /// the same coordinate share the same μ correction.
    pub per_cell: HashMap<u16, TierCalibration>,
}

impl CalibrationReport {
    pub fn record(&mut self, tier: TierId, estimated: f64, actual: f64) {
        self.per_tier.entry(tier).or_default().record(estimated, actual);
    }

    /// Record an observation keyed by both `TierId` and Synthesis
    /// cell. The per-cell aggregation is what feeds the learned-μ
    /// correction factor.
    pub fn record_with_coord(
        &mut self,
        tier: TierId,
        coord: &crate::coord::Coord,
        estimated: f64,
        actual: f64,
    ) {
        self.per_tier.entry(tier).or_default().record(estimated, actual);
        self.per_cell.entry(coord.cell_id()).or_default().record(estimated, actual);
    }

    pub fn record_violation(&mut self, tier: TierId) {
        self.per_tier.entry(tier).or_default().record_violation();
    }

    /// Learned μ-correction for a Synthesis cell. The factor by which
    /// the tier's declared μ should be multiplied to match observed
    /// reality. Returns 1.0 (no correction) when no data has been
    /// observed for this cell.
    ///
    /// Formula: `mu_correction = mean(actual / estimated)`. If a tier
    /// declared μ=1.0 and observations show it's actually 1.8× more
    /// expensive than estimates predict, `mu_correction` returns 1.8.
    /// The effective μ is then `declared_μ × mu_correction = 1.8`.
    pub fn learned_mu(&self, coord: &crate::coord::Coord) -> f64 {
        match self.per_cell.get(&coord.cell_id()) {
            Some(cal) if cal.samples >= 3 => cal.mean_ratio(),
            _ => 1.0,
        }
    }

    /// All cells with enough samples to have a learned μ. Returns
    /// `(cell_id, learned_mu, sample_count)`.
    pub fn learned_cells(&self, min_samples: u64) -> Vec<(u16, f64, u64)> {
        let mut out: Vec<_> = self.per_cell.iter()
            .filter(|(_, cal)| cal.samples >= min_samples)
            .map(|(cell, cal)| (*cell, cal.mean_ratio(), cal.samples))
            .collect();
        out.sort_unstable_by_key(|(cell, _, _)| *cell);
        out
    }

    /// Summarize for diagnostic printing. Returns lines.
    pub fn summary(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut tiers: Vec<_> = self.per_tier.keys().collect();
        tiers.sort_by_key(|t| format!("{:?}", t));
        for tid in tiers {
            let cal = &self.per_tier[tid];
            out.push(format!(
                "{:<25} n={:<6} mean_ratio={:.2} max_ratio={:.2} \
                 total_est={:.3e} total_actual={:.3e} violations={}",
                format!("{:?}", tid),
                cal.samples, cal.mean_ratio(), cal.max_ratio,
                cal.total_estimated, cal.total_actual, cal.budget_violations,
            ));
        }
        out
    }

    /// A tier whose mean_ratio drifts too far from 1.0 is producing
    /// dishonest estimates. Returns tiers needing attention.
    pub fn dishonest_tiers(&self, mean_ratio_tolerance: f64)
        -> Vec<(TierId, f64)>
    {
        let mut out = Vec::new();
        for (tid, cal) in &self.per_tier {
            if cal.samples < 5 { continue; }   // not enough data
            let r = cal.mean_ratio();
            if (r - 1.0).abs() > mean_ratio_tolerance {
                out.push((*tid, r));
            }
        }
        out
    }
}
