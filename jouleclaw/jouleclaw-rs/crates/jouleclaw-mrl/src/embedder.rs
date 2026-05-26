//! `Embedder` — abstract interface for whatever produces a fixed-dim
//! vector from a query. The MRL wrapper composes with any implementor.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum EmbedderError {
    InputTooShort { min: usize, got: usize },
    InputTooLong { max: usize, got: usize },
    Other(String),
}

impl fmt::Display for EmbedderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooShort { min, got } => {
                write!(f, "input too short: need ≥ {} elements, got {}", min, got)
            }
            Self::InputTooLong { max, got } => {
                write!(f, "input too long: max {} elements, got {}", max, got)
            }
            Self::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for EmbedderError {}

pub trait Embedder: Send + Sync {
    /// Output dimensionality at full resolution. The MRL wrapper assumes
    /// any prefix of length d ≤ full_dim() is itself a valid embedding.
    fn full_dim(&self) -> usize;

    /// Produce a `full_dim()`-dimensional vector for the given input.
    /// Implementations must be deterministic: same input → same output.
    fn embed(&self, input: &[f32]) -> Result<Vec<f32>, EmbedderError>;

    /// Static joule cost estimate for one forward pass. Used by the
    /// cascade router to size the embed step against the budget.
    /// Does *not* include downstream retrieval cost — see
    /// `MatryoshkaEmbedder::retrieval_joules_per_doc`.
    fn embed_joules(&self) -> f64;
}

/// A deterministic identity-style embedder: the output's first `full_dim`
/// entries are the input's first `full_dim` entries (or zero-padded).
/// Useful for tests and for exercising the dim-ladder logic without a
/// trained model.
pub struct IdentityEmbedder {
    dim: usize,
}

impl IdentityEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for IdentityEmbedder {
    fn full_dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, input: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        // Take the first `dim` entries of the input, zero-pad if too short.
        let mut out = vec![0.0_f32; self.dim];
        let n = input.len().min(self.dim);
        out[..n].copy_from_slice(&input[..n]);
        Ok(out)
    }

    fn embed_joules(&self) -> f64 {
        // No real work; just the dispatch floor.
        50e-9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_embedder_outputs_input_prefix() {
        let e = IdentityEmbedder::new(4);
        let out = e.embed(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn identity_embedder_zero_pads() {
        let e = IdentityEmbedder::new(8);
        let out = e.embed(&[1.0, 2.0]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn identity_embedder_is_deterministic() {
        let e = IdentityEmbedder::new(4);
        let a = e.embed(&[0.1, -0.2, 0.3, 0.0]).unwrap();
        let b = e.embed(&[0.1, -0.2, 0.3, 0.0]).unwrap();
        assert_eq!(a, b);
    }
}
