//! L2 service interfaces — what L2-class models expose to the rest of
//! the cascade.
//!
//! L2 models live in the `joule-l2` crate. The traits here let other
//! cascade components (`HistoryLayer` for semantic retrieval, `Router`
//! for intent classification) consume L2 services without depending on
//! the heavy `joule-l2` crate.
//!
//! Each service trait has a cost-estimation method so the cascade can
//! decide whether to pay for it within a query's joule budget. Calls
//! that would exceed budget produce `L2Error::BudgetExhausted`.
//!
//! Determinism: same input + same model → same output + same cost.

use crate::types::*;

/// Errors from L2 services.
#[derive(Debug, Clone)]
pub enum L2Error {
    BudgetExhausted { spent: f64, limit: f64 },
    InvalidInput(String),
    BackendFailed(String),
}

impl std::fmt::Display for L2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExhausted { spent, limit } =>
                write!(f, "L2 budget exhausted: spent {:.3e}, limit {:.3e}", spent, limit),
            Self::InvalidInput(s) => write!(f, "L2 invalid input: {}", s),
            Self::BackendFailed(s) => write!(f, "L2 backend: {}", s),
        }
    }
}

impl std::error::Error for L2Error {}

/// An embedding model. Given a piece of text, produces a fixed-dim
/// vector. Used by the history layer for semantic retrieval and by
/// the router for intent classification (via embedding similarity).
pub trait EmbeddingService: Send {
    /// Dimensionality of the produced vectors.
    fn d_model(&self) -> usize;

    /// Estimate the joule cost of embedding `text`.
    fn estimate_cost(&self, text: &str) -> f64;

    /// Compute the embedding. Result is L2-normalized so that cosine
    /// similarity reduces to dot product.
    fn embed(&mut self, text: &str, budget: f64) -> Result<EmbeddingResult, L2Error>;
}

#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    pub vector: Vec<f32>,
    pub joules_spent: f64,
}

/// Intent classifier. Given a query's text, produces a probability
/// distribution over a fixed set of intents.
pub trait IntentClassifier: Send {
    /// The intent labels this classifier emits, in fixed order.
    fn intent_labels(&self) -> &[&'static str];

    fn estimate_cost(&self, text: &str) -> f64;

    fn classify(&mut self, text: &str, budget: f64) -> Result<ClassificationResult, L2Error>;
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Probabilities, indexed by `intent_labels()` position.
    pub probabilities: Vec<f32>,
    /// Index of the highest-probability intent.
    pub top: usize,
    /// Confidence in the top intent (= probabilities[top]).
    pub top_confidence: f32,
    pub joules_spent: f64,
}

/// Cosine similarity between two L2-normalized vectors. Reduces to dot
/// product. Both must have the same length.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

// Use the Query type in scope.
#[allow(dead_code)]
fn _force_use(_q: &Query) {}
