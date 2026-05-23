//! Joule accounting for a single resolved query.

use serde::{Deserialize, Serialize};

/// Where the joule reading came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JouleSource {
    /// Reading from a hardware energy counter (RAPL, NVML, powermetrics).
    Measured,
    /// Synthesized estimate (no hardware counter available).
    Estimated,
}

/// Energy attributable to resolving one query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JouleCost {
    /// Energy in micro-joules (1e-6 J). u64 covers ~5 million joules.
    pub microjoules: u64,
    /// Provenance of the reading.
    pub source: JouleSource,
}

impl JouleCost {
    /// A zero-cost reading — typical for in-process cache hits.
    pub fn zero() -> Self {
        Self {
            microjoules: 0,
            source: JouleSource::Measured,
        }
    }

    /// An estimated cost — used when no hardware counter is available.
    pub fn estimated(microjoules: u64) -> Self {
        Self {
            microjoules,
            source: JouleSource::Estimated,
        }
    }

    /// A measured cost — used when a hardware counter reading is attached.
    pub fn measured(microjoules: u64) -> Self {
        Self {
            microjoules,
            source: JouleSource::Measured,
        }
    }

    /// Convert micro-joules to joules.
    pub fn joules(&self) -> f64 {
        (self.microjoules as f64) / 1_000_000.0
    }
}

impl std::fmt::Display for JouleCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tag = match self.source {
            JouleSource::Measured => "measured",
            JouleSource::Estimated => "estimated",
        };
        write!(f, "{} µJ ({tag})", self.microjoules)
    }
}
