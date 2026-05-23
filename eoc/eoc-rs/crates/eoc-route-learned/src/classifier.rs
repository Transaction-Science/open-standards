//! Multinomial logistic-regression router.
//!
//! Linear-softmax classifier over query embeddings:
//!
//! ```text
//!     logits = W · embedding + b
//!     probs  = softmax(logits)
//! ```
//!
//! Trained by stochastic gradient descent on cross-entropy loss with L2
//! regularisation. The output is a softmax over the four cascade stages
//! (`Cache`, `Kv`, `Graph`, `Neural` — in that order).

use std::collections::BTreeMap;

use async_trait::async_trait;
use eoc_core::{Query, Stage};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::router::{ALL_STAGES, LearnedRouter, RouterState, StagePrediction};

const NUM_STAGES: usize = 4;

fn stage_index(s: Stage) -> usize {
    match s {
        Stage::Cache => 0,
        Stage::Kv => 1,
        Stage::Graph => 2,
        Stage::Neural => 3,
    }
}

fn stage_from_index(i: usize) -> Stage {
    ALL_STAGES[i.min(3)]
}

/// L2-regularised multinomial logistic regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRegRouter {
    /// Embedding dimension `d`.
    embedding_dim: usize,
    /// Weights `W ∈ R^{4 × d}`.
    pub weights: DMatrix<f32>,
    /// Per-stage bias `b ∈ R^4`.
    pub bias: DVector<f32>,
    /// L2 regularisation strength `λ`.
    pub regularization: f32,
}

impl LogRegRouter {
    /// Construct an untrained router with random init.
    pub fn new(embedding_dim: usize, regularization: f32, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let init = Normal::new(0.0_f32, 0.05).expect("valid normal");
        let weights =
            DMatrix::<f32>::from_fn(NUM_STAGES, embedding_dim, |_, _| init.sample(&mut rng));
        let bias = DVector::<f32>::zeros(NUM_STAGES);
        Self {
            embedding_dim,
            weights,
            bias,
            regularization,
        }
    }

    /// Embedding dimension `d`.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Forward pass: softmax probabilities over stages.
    pub fn predict_probs(&self, embedding: &[f32]) -> Result<[f32; 4]> {
        if embedding.len() != self.embedding_dim {
            return Err(Error::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }
        let emb = DVector::<f32>::from_row_slice(embedding);
        let logits_vec: DVector<f32> = &self.weights * emb + &self.bias;
        let mut logits = [0.0_f32; 4];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = logits_vec[i];
        }
        Ok(softmax4(&logits))
    }

    /// Argmax over softmax probabilities.
    pub fn predict(&self, embedding: &[f32]) -> Result<(Stage, f32, [f32; 4])> {
        let probs = self.predict_probs(embedding)?;
        let (idx, p) = argmax(&probs);
        Ok((stage_from_index(idx), p, probs))
    }

    /// One SGD step (cross-entropy + L2). Returns the loss.
    pub fn sgd_step(&mut self, embedding: &[f32], stage: Stage, lr: f32) -> Result<f32> {
        if embedding.len() != self.embedding_dim {
            return Err(Error::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }
        let emb = DVector::<f32>::from_row_slice(embedding);
        let probs = self.predict_probs(embedding)?;
        let target = stage_index(stage);

        // ∂L/∂logits_i = probs_i - 1{i == target}
        let mut grad_logits = [0.0_f32; 4];
        for (i, g) in grad_logits.iter_mut().enumerate() {
            *g = probs[i] - if i == target { 1.0 } else { 0.0 };
        }

        // ∂L/∂W[i, :] = grad_logits[i] · emb  + λ · W[i, :]
        // ∂L/∂b[i]    = grad_logits[i]
        let lambda = self.regularization;
        for (i, &g) in grad_logits.iter().enumerate() {
            for j in 0..self.embedding_dim {
                let w = self.weights[(i, j)];
                self.weights[(i, j)] = w - lr * (g * emb[j] + lambda * w);
            }
            self.bias[i] -= lr * g;
        }

        let eps = 1e-7_f32;
        let loss = -(probs[target] + eps).ln();
        Ok(loss)
    }
}

fn softmax4(logits: &[f32; 4]) -> [f32; 4] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = [0.0_f32; 4];
    let mut sum = 0.0_f32;
    for (i, e) in exps.iter_mut().enumerate() {
        *e = (logits[i] - max).exp();
        sum += *e;
    }
    if sum <= 0.0 {
        return [0.25; 4];
    }
    for e in exps.iter_mut() {
        *e /= sum;
    }
    exps
}

fn argmax(xs: &[f32; 4]) -> (usize, f32) {
    let mut best = 0;
    let mut bv = xs[0];
    for (i, v) in xs.iter().enumerate().skip(1) {
        if *v > bv {
            best = i;
            bv = *v;
        }
    }
    (best, bv)
}

#[async_trait]
impl LearnedRouter for LogRegRouter {
    async fn route(&self, query: &Query) -> StagePrediction {
        let Some(emb) = query.embedding.as_ref() else {
            return StagePrediction::new(Stage::Cache, 0.0);
        };
        match self.predict(emb) {
            Ok((stage, conf, _probs)) => StagePrediction::new(stage, conf),
            Err(_) => StagePrediction::new(Stage::Cache, 0.0),
        }
    }

    async fn observe(
        &mut self,
        query: &Query,
        chosen_stage: Stage,
        was_correct: bool,
        _joule_cost: u64,
    ) {
        // Only train on successful resolutions — we want P(stage = correct | x).
        if was_correct && let Some(emb) = query.embedding.as_ref() {
            let _ = self.sgd_step(emb, chosen_stage, 0.01);
        }
    }

    fn export_state(&self) -> RouterState {
        let blob = serde_json::to_vec(self).unwrap_or_default();
        let mut meta = BTreeMap::new();
        meta.insert("embedding_dim".to_string(), self.embedding_dim.to_string());
        meta.insert(
            "regularization".to_string(),
            self.regularization.to_string(),
        );
        RouterState {
            algorithm: "logreg".to_string(),
            weights_blob: blob,
            metadata: meta,
        }
    }

    fn import_state(state: RouterState) -> Result<Self> {
        if state.algorithm != "logreg" {
            return Err(Error::StateImportFailed(format!(
                "expected algorithm=logreg, got {}",
                state.algorithm
            )));
        }
        serde_json::from_slice(&state.weights_blob)
            .map_err(|e| Error::StateImportFailed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax4(&[1.0, 2.0, 3.0, 4.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn predict_probs_dim_check() {
        let r = LogRegRouter::new(8, 1e-3, 1);
        assert!(r.predict_probs(&[0.0; 7]).is_err());
    }
}
