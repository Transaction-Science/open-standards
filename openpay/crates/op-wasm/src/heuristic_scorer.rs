//! JS-exposed heuristic fraud scorer.
//!
//! The scorer itself is pure Rust — no model load, no async, no
//! browser API surface. JS calls it like any other class.

use wasm_bindgen::prelude::*;

/// Rule-based fraud scorer. See `op_fraud::HeuristicScorer` for the
/// scoring details.
#[wasm_bindgen]
pub struct HeuristicScorer {
    inner: op_fraud::HeuristicScorer,
}

#[wasm_bindgen]
impl HeuristicScorer {
    /// Construct.
    #[wasm_bindgen(constructor)]
    pub fn new() -> HeuristicScorer {
        HeuristicScorer {
            inner: op_fraud::HeuristicScorer::new(),
        }
    }

    /// Scorer name for telemetry. Stable across versions until a
    /// breaking change forces a bump.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        use op_fraud::Scorer;
        self.inner.name().to_owned()
    }
}

impl Default for HeuristicScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_stable() {
        let s = HeuristicScorer::new();
        assert_eq!(s.name(), "heuristic-v1");
    }

    #[test]
    fn default_constructor() {
        let s = HeuristicScorer::default();
        assert_eq!(s.name(), "heuristic-v1");
    }
}
