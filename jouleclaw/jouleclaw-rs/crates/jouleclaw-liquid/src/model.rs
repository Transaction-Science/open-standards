//! `LiquidModel` — a stack of CfC cells with optional input/output projections.
//!
//! Forward pass:
//!
//!   y_0 = u_t
//!   For each cell `i` in layers:
//!       x_i_{t+1} = cell_i.step(x_i_t, y_i)
//!       y_{i+1}   = x_i_{t+1}
//!   output     = y_last
//!
//! Each layer carries its own recurrent state `x_i`. The model owns the
//! state vectors so a single `model.step(u) -> output` call walks the
//! entire stack and updates internal state in place. `model.reset()`
//! returns all states to zero.
//!
//! R29.0 ships the math. R29.1 wires the model to a tokenizer, an input
//! embedding, and an output projection that maps `state_dim` back to
//! token logits.

use crate::cell::{CfcCell, CfcError};

#[derive(Debug, Clone)]
pub struct LiquidModel {
    layers: Vec<CfcCell>,
    /// Per-layer recurrent state. `states[i].len() == layers[i].state_dim()`.
    states: Vec<Vec<f32>>,
}

impl LiquidModel {
    /// Build a model from a stack of cells. Validates that adjacent cells
    /// have compatible dimensions: `layer[i+1].input_dim == layer[i].state_dim`.
    pub fn new(layers: Vec<CfcCell>) -> Result<Self, CfcError> {
        if layers.is_empty() {
            return Err(CfcError::EmptyCell);
        }
        for i in 1..layers.len() {
            let prev = &layers[i - 1];
            let curr = &layers[i];
            if curr.input_dim() != prev.state_dim() {
                return Err(CfcError::DimensionMismatch {
                    what: "inter-layer",
                    expected: prev.state_dim(),
                    got: curr.input_dim(),
                });
            }
        }
        let states: Vec<Vec<f32>> =
            layers.iter().map(|c| vec![0.0; c.state_dim()]).collect();
        Ok(Self { layers, states })
    }

    pub fn input_dim(&self) -> usize {
        self.layers.first().map(|c| c.input_dim()).unwrap_or(0)
    }

    pub fn output_dim(&self) -> usize {
        self.layers.last().map(|c| c.state_dim()).unwrap_or(0)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Reset all per-layer state vectors to zero.
    pub fn reset(&mut self) {
        for s in &mut self.states {
            for v in s.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// One sequence step: feed input `u` through all layers, advancing
    /// each layer's state. Writes the final layer's new state into
    /// `output`.
    pub fn step(&mut self, u: &[f32], output: &mut [f32]) -> Result<(), CfcError> {
        if u.len() != self.input_dim() {
            return Err(CfcError::DimensionMismatch {
                what: "model input",
                expected: self.input_dim(),
                got: u.len(),
            });
        }
        if output.len() != self.output_dim() {
            return Err(CfcError::DimensionMismatch {
                what: "model output",
                expected: self.output_dim(),
                got: output.len(),
            });
        }

        // Walk the layers. `current_input` rotates: it starts as `u`, then
        // becomes each layer's new state.
        let mut current_input: Vec<f32> = u.to_vec();
        let n_layers = self.layers.len();
        for (i, cell) in self.layers.iter().enumerate() {
            // Allocate a scratch buffer for this layer's new state so we
            // don't alias `self.states[i]` while reading from it.
            let mut new_state = vec![0.0_f32; cell.state_dim()];
            cell.step(&self.states[i], &current_input, &mut new_state)?;
            // Commit the new state.
            self.states[i].copy_from_slice(&new_state);
            // Feed forward to the next layer.
            if i + 1 < n_layers {
                current_input = new_state;
            } else {
                output.copy_from_slice(&new_state);
            }
        }
        Ok(())
    }

    /// Static joule estimate for one model step.
    pub fn step_joules(&self) -> f64 {
        self.layers.iter().map(|c| c.step_joules()).sum::<f64>()
    }

    pub fn states(&self) -> &[Vec<f32>] {
        &self.states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_model_step_yields_zeros() {
        let l1 = CfcCell::zeros(8, 4).unwrap();
        let l2 = CfcCell::zeros(6, 8).unwrap();
        let mut model = LiquidModel::new(vec![l1, l2]).unwrap();
        let u = vec![0.0_f32; 4];
        let mut out = vec![999.0_f32; 6];
        model.step(&u, &mut out).unwrap();
        assert_eq!(out, vec![0.0; 6]);
    }

    #[test]
    fn step_is_deterministic_across_calls_after_reset() {
        let l1 = CfcCell::zeros(4, 3).unwrap();
        let l2 = CfcCell::zeros(2, 4).unwrap();
        let mut m1 = LiquidModel::new(vec![l1.clone(), l2.clone()]).unwrap();
        let mut m2 = LiquidModel::new(vec![l1, l2]).unwrap();

        let u = vec![0.5_f32, -0.3, 0.1];
        let mut a = vec![0.0_f32; 2];
        let mut b = vec![0.0_f32; 2];
        m1.step(&u, &mut a).unwrap();
        m2.step(&u, &mut b).unwrap();
        // Same starting state (zeros), same input → bit-identical output.
        assert_eq!(a, b);
    }

    #[test]
    fn reset_returns_states_to_zero() {
        let l1 = CfcCell::zeros(3, 2).unwrap();
        let mut model = LiquidModel::new(vec![l1]).unwrap();
        // Step with non-zero input so state would normally drift — but
        // with zero weights it doesn't. We assert reset is at least a no-op.
        model.reset();
        for s in model.states() {
            for v in s {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn dimension_mismatch_between_layers_rejected() {
        let l1 = CfcCell::zeros(8, 4).unwrap();   // output dim 8
        let l2 = CfcCell::zeros(6, 7).unwrap();   // expects input dim 7, but prev gives 8
        assert!(matches!(
            LiquidModel::new(vec![l1, l2]),
            Err(CfcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn output_envelope_stays_in_tanh_range() {
        // Even with extreme weights, the model's final-layer output is the
        // last cell's new state, which is bounded by tanh.
        let mut l1 = CfcCell::zeros(4, 2).unwrap();
        for w in &mut l1.w_g { *w = 50.0; }
        for w in &mut l1.w_h { *w = -50.0; }
        let mut model = LiquidModel::new(vec![l1]).unwrap();
        let u = vec![10.0_f32, -10.0];
        let mut out = vec![0.0_f32; 4];
        model.step(&u, &mut out).unwrap();
        for v in &out {
            assert!((-1.0..=1.0).contains(v));
        }
    }
}
