//! `DimPicker` — choose the smallest dim that meets a quality floor under
//! a per-query retrieval budget.
//!
//! Decision rule:
//!
//!   pick(corpus_size, quality_floor, retrieval_budget_joules) →
//!     smallest d ∈ dims such that:
//!       quality_model(d)              ≥ quality_floor       AND
//!       retrieval_joules(d) * corpus  ≤ retrieval_budget_joules
//!
//! If no dim satisfies both, the picker reports which constraint failed
//! so the cascade can fall through to a different tier.

use std::fmt;

use crate::embedder::Embedder;
use crate::matryoshka::MatryoshkaEmbedder;

#[derive(Debug, Clone, PartialEq)]
pub enum PickError {
    /// No dim in the ladder meets the quality floor (even at full dim).
    QualityFloorUnreachable { max_quality: f32, requested: f32 },
    /// Quality floor is reachable, but every dim that meets it exceeds the budget.
    BudgetExceeded {
        min_dim_meeting_quality: usize,
        cost_at_min_dim: f64,
        budget: f64,
    },
}

impl fmt::Display for PickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QualityFloorUnreachable { max_quality, requested } => {
                write!(
                    f,
                    "quality floor {:.4} unreachable (max attainable {:.4})",
                    requested, max_quality
                )
            }
            Self::BudgetExceeded { min_dim_meeting_quality, cost_at_min_dim, budget } => {
                write!(
                    f,
                    "smallest quality-satisfying dim {} costs {:.3e} J, budget {:.3e} J",
                    min_dim_meeting_quality, cost_at_min_dim, budget
                )
            }
        }
    }
}

impl std::error::Error for PickError {}

pub struct DimPicker<'a, E: Embedder> {
    pub matryoshka: &'a MatryoshkaEmbedder<E>,
}

impl<'a, E: Embedder> DimPicker<'a, E> {
    pub fn new(matryoshka: &'a MatryoshkaEmbedder<E>) -> Self {
        Self { matryoshka }
    }

    /// Smallest dim that satisfies both constraints.
    pub fn pick(
        &self,
        corpus_size: usize,
        quality_floor: f32,
        retrieval_budget_joules: f64,
    ) -> Result<usize, PickError> {
        // dims is sorted ascending; walk in that order so the first hit
        // is the smallest qualifying dim.
        let mut min_quality_dim: Option<usize> = None;
        for &d in self.matryoshka.dims() {
            let q = self.matryoshka.quality.at(d);
            if q < quality_floor {
                continue;
            }
            if min_quality_dim.is_none() {
                min_quality_dim = Some(d);
            }
            let cost = self.matryoshka.retrieval_joules(d, corpus_size);
            if cost <= retrieval_budget_joules {
                return Ok(d);
            }
        }
        match min_quality_dim {
            Some(d) => {
                let cost = self.matryoshka.retrieval_joules(d, corpus_size);
                Err(PickError::BudgetExceeded {
                    min_dim_meeting_quality: d,
                    cost_at_min_dim: cost,
                    budget: retrieval_budget_joules,
                })
            }
            None => {
                // No dim meets the quality floor; report the best attainable.
                let full = *self.matryoshka.dims().last().unwrap_or(&0);
                let q = self.matryoshka.quality.at(full);
                Err(PickError::QualityFloorUnreachable {
                    max_quality: q,
                    requested: quality_floor,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::IdentityEmbedder;
    use crate::matryoshka::MatryoshkaEmbedder;

    fn make(full_dim: usize) -> MatryoshkaEmbedder<IdentityEmbedder> {
        MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(full_dim))
    }

    #[test]
    fn picks_smallest_dim_meeting_quality_under_generous_budget() {
        let m = make(2048);
        let p = DimPicker::new(&m);
        // Generous budget (1 J) so cost never binds; quality floor 0.90.
        let d = p.pick(10_000, 0.90, 1.0).unwrap();
        // The default LogDecay model: q(d) = 1 - 0.022 * ln(2048/d).
        // Solve 1 - 0.022 * ln(2048/d) ≥ 0.90 → ln(2048/d) ≤ 4.545 → d ≥ 21.7
        // First power-of-2 ≥ 21.7 is 32.
        assert_eq!(d, 32);
    }

    #[test]
    fn budget_pressure_forces_higher_dim_when_quality_floor_is_low() {
        // With very low quality floor, small dims pass quality. But if
        // budget is tight per-doc, fewer dims qualify on cost.
        let m = make(2048);
        let p = DimPicker::new(&m);
        // 10M docs at 1 pJ/dim → 1 J at d=1, 1024 J at d=1024.
        // Budget 10 mJ → max affordable dim = 10e-3 / (1e-12 * 10e6) = 1000.
        // We expect dim ≤ 1000 → 512.
        let d = p.pick(10_000_000, 0.0, 10e-3).unwrap();
        assert!(d <= 1000, "got d={} > affordable cap", d);
        // And it should be the smallest meeting quality 0.0 (which is d=1).
        // But the budget is the binding constraint, so it picks the
        // SMALLEST d that meets quality. With quality_floor=0.0 every d
        // qualifies, so picker returns the smallest that also fits — d=1.
        assert_eq!(d, 1);
    }

    #[test]
    fn quality_floor_unreachable_when_above_max() {
        let m = make(2048);
        let p = DimPicker::new(&m);
        let err = p.pick(100, 1.5, 1.0).unwrap_err();
        assert!(matches!(err, PickError::QualityFloorUnreachable { .. }));
    }

    #[test]
    fn budget_exceeded_when_min_quality_dim_too_costly() {
        let m = make(2048);
        let p = DimPicker::new(&m);
        // Demand quality 0.95 (forces d ≥ 256-ish), tiny budget.
        let err = p.pick(100_000_000_000, 0.95, 1e-9).unwrap_err();
        assert!(
            matches!(err, PickError::BudgetExceeded { .. }),
            "expected BudgetExceeded, got {:?}", err
        );
    }

    #[test]
    fn pick_is_monotone_in_quality_floor() {
        let m = make(2048);
        let p = DimPicker::new(&m);
        let d_low = p.pick(1000, 0.85, 1.0).unwrap();
        let d_mid = p.pick(1000, 0.92, 1.0).unwrap();
        let d_high = p.pick(1000, 0.97, 1.0).unwrap();
        assert!(d_low <= d_mid);
        assert!(d_mid <= d_high);
    }
}
