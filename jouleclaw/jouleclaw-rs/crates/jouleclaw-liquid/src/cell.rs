//! A single Closed-form Continuous-time (CfC) cell.
//!
//! Discrete-step update:
//!
//!   z       = [x_t ; u_t]                   (concatenated state + input)
//!   pre_f   = W_f · z + b_f                 (time-gate logits)
//!   pre_g   = W_g · z + b_g                 (content logits)
//!   pre_h   = W_h · z + b_h                 (alternative-state logits)
//!   gate    = σ(-(pre_f + θ_t))             (per-channel time gate)
//!   x_{t+1} = gate ⊙ tanh(pre_g) + (1-gate) ⊙ tanh(pre_h)
//!
//! All three weight matrices are stored row-major and shaped
//! `[state_dim, state_dim + input_dim]`. The per-channel time-gate bias
//! `θ_t` has length `state_dim`. The cell is f32-only; cross-platform
//! bit-reproducibility depends on the platform's libm conformance — for
//! a strict guarantee, R29.1 will pin to a software `tanh`/`exp`.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CfcError {
    DimensionMismatch { what: &'static str, expected: usize, got: usize },
    EmptyCell,
}

impl fmt::Display for CfcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { what, expected, got } => {
                write!(f, "{}: expected {}, got {}", what, expected, got)
            }
            Self::EmptyCell => write!(f, "cell has zero state or input dim"),
        }
    }
}

impl std::error::Error for CfcError {}

/// A single CfC cell with three weight matrices and four biases.
#[derive(Debug, Clone)]
pub struct CfcCell {
    state_dim: usize,
    input_dim: usize,
    /// W_f: time-gate weights, row-major, shape [state_dim, state_dim + input_dim].
    pub w_f: Vec<f32>,
    /// W_g: content weights, same shape.
    pub w_g: Vec<f32>,
    /// W_h: alternative-state weights, same shape.
    pub w_h: Vec<f32>,
    /// Biases, each length `state_dim`.
    pub b_f: Vec<f32>,
    pub b_g: Vec<f32>,
    pub b_h: Vec<f32>,
    /// Per-channel time-gate bias θ_t, length `state_dim`. In the CfC paper
    /// this absorbs both the original LTC time-constant τ and a learned
    /// shift; here it's a single per-channel parameter.
    pub theta_t: Vec<f32>,
}

impl CfcCell {
    /// Construct a cell with all zero weights and biases. Useful as a
    /// starting point and for sanity checks.
    pub fn zeros(state_dim: usize, input_dim: usize) -> Result<Self, CfcError> {
        if state_dim == 0 || input_dim == 0 {
            return Err(CfcError::EmptyCell);
        }
        let w_size = state_dim * (state_dim + input_dim);
        Ok(Self {
            state_dim,
            input_dim,
            w_f: vec![0.0; w_size],
            w_g: vec![0.0; w_size],
            w_h: vec![0.0; w_size],
            b_f: vec![0.0; state_dim],
            b_g: vec![0.0; state_dim],
            b_h: vec![0.0; state_dim],
            theta_t: vec![0.0; state_dim],
        })
    }

    pub fn state_dim(&self) -> usize { self.state_dim }
    pub fn input_dim(&self) -> usize { self.input_dim }

    /// In-place forward step. `x_out` may alias `x_in` only if the caller
    /// is fine with that — internally we use a scratch buffer.
    pub fn step(&self, x_in: &[f32], u: &[f32], x_out: &mut [f32]) -> Result<(), CfcError> {
        if x_in.len() != self.state_dim {
            return Err(CfcError::DimensionMismatch {
                what: "state",
                expected: self.state_dim,
                got: x_in.len(),
            });
        }
        if u.len() != self.input_dim {
            return Err(CfcError::DimensionMismatch {
                what: "input",
                expected: self.input_dim,
                got: u.len(),
            });
        }
        if x_out.len() != self.state_dim {
            return Err(CfcError::DimensionMismatch {
                what: "output",
                expected: self.state_dim,
                got: x_out.len(),
            });
        }
        let z_dim = self.state_dim + self.input_dim;
        // Build concatenated z = [x_in ; u] on the stack-ish scratch.
        // For large dims this would heap-allocate; that's acceptable for
        // R29.0 — a hot-path optimization in R29.1.
        let mut z = Vec::with_capacity(z_dim);
        z.extend_from_slice(x_in);
        z.extend_from_slice(u);

        // Three matvecs in parallel sequence.
        let pre_f = matvec_bias(self.state_dim, z_dim, &self.w_f, &z, &self.b_f);
        let pre_g = matvec_bias(self.state_dim, z_dim, &self.w_g, &z, &self.b_g);
        let pre_h = matvec_bias(self.state_dim, z_dim, &self.w_h, &z, &self.b_h);

        for i in 0..self.state_dim {
            let gate = sigmoid(-(pre_f[i] + self.theta_t[i]));
            let content = pre_g[i].tanh();
            let alt = pre_h[i].tanh();
            x_out[i] = gate * content + (1.0 - gate) * alt;
        }
        Ok(())
    }

    /// Static joule estimate for one step. Cost model:
    ///   - three matvecs of size state_dim × (state_dim + input_dim)
    ///   - 3 element-wise activations per state channel (sigmoid + 2 tanhs)
    ///   - 1 blend per state channel
    /// At ~10 pJ per f32 FMA, ~30 pJ per activation, ~5 pJ per blend.
    pub fn step_joules(&self) -> f64 {
        const FMA_PJ: f64 = 10.0;
        const ACT_PJ: f64 = 30.0;
        const BLEND_PJ: f64 = 5.0;
        const DISPATCH_FLOOR_NJ: f64 = 50.0;
        let z_dim = (self.state_dim + self.input_dim) as f64;
        let matvec_ops = 3.0 * (self.state_dim as f64) * z_dim;
        let activations = 3.0 * (self.state_dim as f64);
        let blends = self.state_dim as f64;
        (matvec_ops * FMA_PJ + activations * ACT_PJ + blends * BLEND_PJ) * 1e-12
            + DISPATCH_FLOOR_NJ * 1e-9
    }
}

/// y = W · x + b, row-major. Returns a new Vec sized `rows`.
fn matvec_bias(rows: usize, cols: usize, w: &[f32], x: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(b.len(), rows);
    let mut y = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut acc = b[r];
        let row_offset = r * cols;
        for c in 0..cols {
            acc += w[row_offset + c] * x[c];
        }
        y.push(acc);
    }
    y
}

/// Numerically-stable f32 sigmoid: σ(x) = 1 / (1 + exp(-x)). The std `exp`
/// is deterministic per-platform. For x outside [-50, 50] we saturate to
/// avoid `exp` overflow/underflow producing platform-dependent NaNs.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x > 50.0 {
        1.0
    } else if x < -50.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_cell_step_yields_zeros() {
        // All-zero weights and biases: pre_f = pre_g = pre_h = 0,
        // gate = σ(0) = 0.5, tanh(0) = 0, so x_out = 0.5 * 0 + 0.5 * 0 = 0.
        let cell = CfcCell::zeros(4, 3).unwrap();
        let x = vec![0.0_f32; 4];
        let u = vec![0.0_f32; 3];
        let mut x_out = vec![999.0_f32; 4]; // poison
        cell.step(&x, &u, &mut x_out).unwrap();
        assert_eq!(x_out, vec![0.0; 4]);
    }

    #[test]
    fn zeros_cell_step_with_nonzero_input_still_yields_zeros() {
        // Weights are zero so the input never reaches the activations.
        let cell = CfcCell::zeros(4, 3).unwrap();
        let x = vec![1.0_f32; 4];
        let u = vec![-2.5_f32; 3];
        let mut x_out = vec![0.0_f32; 4];
        cell.step(&x, &u, &mut x_out).unwrap();
        for v in &x_out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn step_is_deterministic_across_calls() {
        // Build a deterministic non-zero cell.
        let mut cell = CfcCell::zeros(3, 2).unwrap();
        for (i, w) in cell.w_f.iter_mut().enumerate() {
            *w = (i as f32) * 0.1 - 0.5;
        }
        for (i, w) in cell.w_g.iter_mut().enumerate() {
            *w = (i as f32) * -0.07 + 0.3;
        }
        for (i, w) in cell.w_h.iter_mut().enumerate() {
            *w = (i as f32) * 0.05;
        }
        cell.b_f = vec![0.1, -0.2, 0.3];
        cell.b_g = vec![0.0, 0.4, -0.1];
        cell.theta_t = vec![0.05, -0.05, 0.0];

        let x = vec![0.7_f32, -0.3, 0.0];
        let u = vec![1.0_f32, -0.5];
        let mut a = vec![0.0_f32; 3];
        let mut b = vec![0.0_f32; 3];
        cell.step(&x, &u, &mut a).unwrap();
        cell.step(&x, &u, &mut b).unwrap();
        // Bit-identical determinism.
        assert_eq!(a.to_vec(), b.to_vec(), "same inputs must give same outputs");
    }

    #[test]
    fn step_bounded_by_tanh_envelope() {
        // The blend gate * tanh + (1-gate) * tanh sits inside the tanh
        // envelope, so each output channel must be in (-1, 1) regardless
        // of weight magnitudes.
        let mut cell = CfcCell::zeros(8, 4).unwrap();
        for w in &mut cell.w_g { *w = 100.0; }
        for w in &mut cell.w_h { *w = -100.0; }
        let x = vec![1.0_f32; 8];
        let u = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 8];
        cell.step(&x, &u, &mut out).unwrap();
        for v in &out {
            assert!((-1.0..=1.0).contains(v), "tanh envelope violated: {}", v);
        }
    }

    #[test]
    fn dimension_errors_surface() {
        let cell = CfcCell::zeros(4, 3).unwrap();
        let x = vec![0.0_f32; 4];
        let u = vec![0.0_f32; 2]; // wrong: expected 3
        let mut out = vec![0.0_f32; 4];
        assert!(matches!(
            cell.step(&x, &u, &mut out),
            Err(CfcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn step_joules_scales_with_dimensions() {
        let small = CfcCell::zeros(8, 8).unwrap();
        let large = CfcCell::zeros(64, 64).unwrap();
        assert!(large.step_joules() > small.step_joules());
        // Sanity: both well under a microjoule for these sizes.
        assert!(small.step_joules() < 1e-6);
    }

    #[test]
    fn sigmoid_saturates_safely() {
        assert_eq!(sigmoid(100.0), 1.0);
        assert_eq!(sigmoid(-100.0), 0.0);
        let mid = sigmoid(0.0);
        assert!((mid - 0.5).abs() < 1e-7);
    }
}
