//! Energy cost intrinsic to the envelope.
//!
//! Every envelope carries both a *measured* cost (what the producing
//! node observed) and an *estimated* cost (what the publisher claims
//! when measurement is unavailable). Cascade auditors compare the two.

use serde::{Deserialize, Serialize};

/// Energy cost attributed to producing or transmitting this envelope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JouleCost {
    /// Joules measured by hardware, expressed in microjoules.
    pub measured_microjoules: u64,
    /// Joules estimated when measurement is unavailable, in microjoules.
    pub estimated_microjoules: u64,
}

impl JouleCost {
    /// Constructor for a fully-measured cost (no estimation).
    pub fn measured(microjoules: u64) -> Self {
        Self {
            measured_microjoules: microjoules,
            estimated_microjoules: 0,
        }
    }

    /// Constructor for an estimated-only cost.
    pub fn estimated(microjoules: u64) -> Self {
        Self {
            measured_microjoules: 0,
            estimated_microjoules: microjoules,
        }
    }

    /// Total cost = measured + estimated. Auditors may diverge from
    /// this; the field is provided for ergonomic display.
    pub fn total_microjoules(&self) -> u128 {
        self.measured_microjoules as u128 + self.estimated_microjoules as u128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_sums_components() {
        let c = JouleCost {
            measured_microjoules: 100,
            estimated_microjoules: 50,
        };
        assert_eq!(c.total_microjoules(), 150);
    }
}
