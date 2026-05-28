//! The four ARL axes. Each is anchored in math or physics that does not
//! drift across time, languages, or regimes.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors constructing an axis value.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AxisError {
    /// Validation depth outside 1–9.
    #[error("validation depth must be 1–9, got {0}")]
    DepthOutOfRange(u8),
}

// ─────────────────────────────────────────────────────────────────────
// Validation Depth (1–9) — statistics
// ─────────────────────────────────────────────────────────────────────

/// How thoroughly the readiness claim has been tested, 1–9. Adapts the
/// Technology Readiness Level scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct ValidationDepth(u8);

impl ValidationDepth {
    /// Construct a depth in 1–9.
    pub fn new(level: u8) -> Result<Self, AxisError> {
        if (1..=9).contains(&level) {
            Ok(Self(level))
        } else {
            Err(AxisError::DepthOutOfRange(level))
        }
    }

    /// The numeric level, 1–9.
    pub fn level(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for ValidationDepth {
    type Error = AxisError;
    fn try_from(level: u8) -> Result<Self, Self::Error> {
        Self::new(level)
    }
}

impl From<ValidationDepth> for u8 {
    fn from(d: ValidationDepth) -> u8 {
        d.0
    }
}

impl Default for ValidationDepth {
    /// ARL 1 — principle observed — is the floor.
    fn default() -> Self {
        Self(1)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Convergence Class (A–E) — stochastic process theory
// ─────────────────────────────────────────────────────────────────────

/// How stochastic the system is on the certified task. `A` is the most
/// deterministic; `E` (uncharacterized) is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceClass {
    /// Deterministic equivalent (≥100 identical/equivalent runs).
    A,
    /// Bounded convergent (variance + failure rate characterized, N ≥ 30).
    B,
    /// Bounded with characterized failures inside a documented envelope.
    C,
    /// Divergent on extension (stable short, diverges with length/depth).
    D,
    /// Uncharacterized. Variance never measured. Default.
    E,
}

impl ConvergenceClass {
    /// Rank where lower = more deterministic (`A`=0 … `E`=4).
    pub fn rank(self) -> u8 {
        match self {
            ConvergenceClass::A => 0,
            ConvergenceClass::B => 1,
            ConvergenceClass::C => 2,
            ConvergenceClass::D => 3,
            ConvergenceClass::E => 4,
        }
    }

    /// True if `self` is at least as good (deterministic) as `floor`.
    /// `Class C or better` ⇒ `at_least(C)` is true for A, B, C.
    pub fn at_least(self, floor: ConvergenceClass) -> bool {
        self.rank() <= floor.rank()
    }
}

impl Default for ConvergenceClass {
    fn default() -> Self {
        ConvergenceClass::E
    }
}

// ─────────────────────────────────────────────────────────────────────
// Security Class (S0–S4) — information theory + cryptography
// ─────────────────────────────────────────────────────────────────────

/// Measured resistance to adversarial conditions. `S0` (uncharacterized)
/// is the default; `S4` (complete auditability) is the ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityClass {
    /// Uncharacterized. No security measurement. Default.
    S0,
    /// Adversarial robustness measured (attack success rate published).
    S1,
    /// S1 + output integrity cryptographically attested.
    S2,
    /// S2 + measured input/state confidentiality.
    S3,
    /// S3 + complete auditability.
    S4,
}

impl SecurityClass {
    /// Rank where higher = stronger (`S0`=0 … `S4`=4).
    pub fn rank(self) -> u8 {
        match self {
            SecurityClass::S0 => 0,
            SecurityClass::S1 => 1,
            SecurityClass::S2 => 2,
            SecurityClass::S3 => 3,
            SecurityClass::S4 => 4,
        }
    }

    /// True if `self` is at least as strong as `floor`.
    pub fn at_least(self, floor: SecurityClass) -> bool {
        self.rank() >= floor.rank()
    }
}

impl Default for SecurityClass {
    fn default() -> Self {
        SecurityClass::S0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Energy Profile (joules) — thermodynamics
// ─────────────────────────────────────────────────────────────────────

/// The three energy numbers, or an explicit refusal to disclose. Refusing
/// to disclose is a *valid state* — it just caps the achievable ARL at 3
/// (enforced by [`crate::Claim::validate`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnergyProfile {
    /// Energy not disclosed. Caps the score at ARL 3.
    Undisclosed,
    /// All three figures disclosed.
    Disclosed {
        /// Training energy amortized over deployment lifetime, MWh per
        /// deployment-year.
        training_mwh_per_year: f64,
        /// Mean per-task inference energy, kJ/task.
        inference_kj_mean: f64,
        /// Standard deviation of per-task inference energy, kJ/task.
        inference_kj_std: f64,
        /// Sample size behind the inference figures (must be ≥ 100).
        inference_n: u32,
        /// Total cost of operation = inference × PUE, kJ/task.
        total_kj: f64,
        /// Deployment facility PUE.
        pue: f64,
        /// Grid carbon intensity at the deployment location, gCO₂eq/kWh.
        grid_gco2_per_kwh: f64,
    },
}

impl EnergyProfile {
    pub fn is_disclosed(&self) -> bool {
        matches!(self, EnergyProfile::Disclosed { .. })
    }
}

impl Default for EnergyProfile {
    fn default() -> Self {
        EnergyProfile::Undisclosed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_range_enforced() {
        assert!(ValidationDepth::new(0).is_err());
        assert!(ValidationDepth::new(10).is_err());
        assert_eq!(ValidationDepth::new(6).unwrap().level(), 6);
    }

    #[test]
    fn convergence_ordering() {
        // "Class C or better" includes A, B, C — not D, E.
        assert!(ConvergenceClass::A.at_least(ConvergenceClass::C));
        assert!(ConvergenceClass::C.at_least(ConvergenceClass::C));
        assert!(!ConvergenceClass::D.at_least(ConvergenceClass::C));
        assert!(!ConvergenceClass::E.at_least(ConvergenceClass::C));
    }

    #[test]
    fn security_ordering() {
        assert!(SecurityClass::S4.at_least(SecurityClass::S2));
        assert!(SecurityClass::S2.at_least(SecurityClass::S2));
        assert!(!SecurityClass::S1.at_least(SecurityClass::S2));
    }

    #[test]
    fn defaults_are_the_uncharacterized_floor() {
        assert_eq!(ConvergenceClass::default(), ConvergenceClass::E);
        assert_eq!(SecurityClass::default(), SecurityClass::S0);
        assert_eq!(ValidationDepth::default().level(), 1);
        assert!(!EnergyProfile::default().is_disclosed());
    }
}
