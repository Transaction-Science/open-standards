//! Matrix-factorisation router (RouteLLM, ICLR 2025).
//!
//! Latent-factor model: each *query cluster* `q` and each *stage* `s` get a
//! `k`-dimensional latent vector, plus scalar biases. The probability that
//! stage `s` satisfies a query in cluster `q` is
//!
//! ```text
//!     p(success | q, s) = σ( q · s + b_q + b_s + μ )
//! ```
//!
//! We do not require an explicit cluster id at predict-time — instead we
//! compute the query latent factor from the embedding via a learned linear
//! projection `W ∈ R^{k × d}`, so:
//!
//! ```text
//!     q_latent = W · embedding
//! ```
//!
//! Training minimises binary-cross-entropy over `(embedding, stage, success)`
//! examples by SGD. This is the *embedding-conditioned* MF variant of
//! RouteLLM — it generalises to unseen queries by design.

use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;
use eoc_core::{Query, Stage};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::router::{ALL_STAGES, LearnedRouter, RouterState, StagePrediction};

/// Default latent dimension.
pub const DEFAULT_LATENT_DIM: usize = 32;

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

/// Matrix-factorisation router (embedding-conditioned).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfRouter {
    /// Latent dimension `k`.
    latent_dim: usize,
    /// Input embedding dimension `d`.
    embedding_dim: usize,
    /// Projection `W ∈ R^{k × d}` mapping embeddings to latent space.
    projection: DMatrix<f32>,
    /// Stage latent factors `S ∈ R^{k × 4}`, column-per-stage.
    stage_factors: DMatrix<f32>,
    /// Per-stage bias `b_s ∈ R^4`.
    stage_bias: DVector<f32>,
    /// Global bias `μ`.
    global_bias: f32,
    /// Estimated microjoule cost per stage, learned online via EMA.
    joule_ema: HashMap<Stage, u64>,
}

impl MfRouter {
    /// Construct an untrained router with random init.
    ///
    /// `seed` makes initialisation deterministic — required for reproducible
    /// tests. `latent_dim` defaults to [`DEFAULT_LATENT_DIM`] when zero.
    pub fn new(embedding_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let k = if latent_dim == 0 {
            DEFAULT_LATENT_DIM
        } else {
            latent_dim
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let init = Normal::new(0.0_f32, 0.1).expect("valid normal");
        let projection =
            DMatrix::<f32>::from_fn(k, embedding_dim, |_, _| init.sample(&mut rng));
        let stage_factors = DMatrix::<f32>::from_fn(k, 4, |_, _| init.sample(&mut rng));
        let stage_bias = DVector::<f32>::from_fn(4, |_, _| init.sample(&mut rng));
        Self {
            latent_dim: k,
            embedding_dim,
            projection,
            stage_factors,
            stage_bias,
            global_bias: 0.0,
            joule_ema: default_joule_ema(),
        }
    }

    /// Latent dimension `k`.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Expected embedding dimension `d`.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Forward pass: predict success probability per stage.
    ///
    /// Returns logits *and* softmax-normalised probabilities. The softmax is
    /// taken over per-stage sigmoid outputs so the recommended-stage
    /// probability stays in [0, 1] when the router is uncertain.
    pub fn predict_probs(&self, embedding: &[f32]) -> Result<[f32; 4]> {
        if embedding.len() != self.embedding_dim {
            return Err(Error::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }
        let emb = DVector::<f32>::from_row_slice(embedding);
        let latent = &self.projection * emb;
        let mut probs = [0.0_f32; 4];
        for (s, p) in probs.iter_mut().enumerate() {
            let col = self.stage_factors.column(s);
            let dot: f32 = latent.dot(&col);
            let logit = dot + self.stage_bias[s] + self.global_bias;
            *p = sigmoid(logit);
        }
        Ok(probs)
    }

    /// Pick the recommended stage and its confidence.
    pub fn predict(&self, embedding: &[f32]) -> Result<(Stage, f32, [f32; 4])> {
        let probs = self.predict_probs(embedding)?;
        let (idx, conf) = argmax(&probs);
        Ok((stage_from_index(idx), conf, probs))
    }

    /// One SGD step on a single example.
    ///
    /// Loss is binary cross-entropy on `sigmoid(score)` vs `success`. We
    /// update `W`, `S[:, s]`, `b_s` and `μ`. `l2` is the weight-decay
    /// strength.
    pub fn sgd_step(
        &mut self,
        embedding: &[f32],
        stage: Stage,
        success: bool,
        lr: f32,
        l2: f32,
    ) -> Result<f32> {
        if embedding.len() != self.embedding_dim {
            return Err(Error::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }
        let s_idx = stage_index(stage);
        let target = if success { 1.0_f32 } else { 0.0 };

        let emb = DVector::<f32>::from_row_slice(embedding);
        let latent = &self.projection * &emb;
        let stage_vec = self.stage_factors.column(s_idx).into_owned();
        let logit = latent.dot(&stage_vec) + self.stage_bias[s_idx] + self.global_bias;
        let pred = sigmoid(logit);
        let grad_logit = pred - target;

        // Loss for logging.
        let eps = 1e-7_f32;
        let loss = -(target * (pred + eps).ln() + (1.0 - target) * (1.0 - pred + eps).ln());

        // ∂L/∂S[:, s] = grad_logit · latent  + l2 · S[:, s]
        let stage_update = &latent * grad_logit + &stage_vec * l2;
        // ∂L/∂latent  = grad_logit · S[:, s]
        let latent_grad = &stage_vec * grad_logit;
        // ∂L/∂W = latent_grad · emb^T + l2 · W (decay applied below)
        // We use the outer product latent_grad · emb^T directly.
        let projection_update = &latent_grad * emb.transpose();

        // Apply updates.
        for i in 0..self.latent_dim {
            for j in 0..self.embedding_dim {
                let cur = self.projection[(i, j)];
                self.projection[(i, j)] = cur - lr * (projection_update[(i, j)] + l2 * cur);
            }
        }
        for i in 0..self.latent_dim {
            let cur = self.stage_factors[(i, s_idx)];
            self.stage_factors[(i, s_idx)] = cur - lr * stage_update[i];
        }
        self.stage_bias[s_idx] -= lr * grad_logit;
        self.global_bias -= lr * grad_logit;

        Ok(loss)
    }

    /// Record an observed joule cost — EMA with coefficient 0.1.
    pub fn observe_cost(&mut self, stage: Stage, microjoules: u64) {
        let prev = self.joule_ema.get(&stage).copied().unwrap_or(microjoules) as f64;
        let blended = 0.9 * prev + 0.1 * microjoules as f64;
        self.joule_ema.insert(stage, blended as u64);
    }

    /// Current per-stage joule estimate (EMA).
    pub fn joule_estimates(&self) -> HashMap<Stage, u64> {
        self.joule_ema.clone()
    }
}

fn default_joule_ema() -> HashMap<Stage, u64> {
    let mut m = HashMap::new();
    // Conservative defaults, ordered cheap → expensive.
    m.insert(Stage::Cache, 1_000);
    m.insert(Stage::Kv, 50_000);
    m.insert(Stage::Graph, 500_000);
    m.insert(Stage::Neural, 50_000_000);
    m
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
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
impl LearnedRouter for MfRouter {
    async fn route(&self, query: &Query) -> StagePrediction {
        let Some(emb) = query.embedding.as_ref() else {
            // No embedding → fall back to the cheapest stage with low confidence.
            let mut pred = StagePrediction::new(Stage::Cache, 0.0);
            for (s, j) in &self.joule_ema {
                pred = pred.with_estimate(*s, *j);
            }
            return pred;
        };
        match self.predict(emb) {
            Ok((stage, conf, _probs)) => {
                let mut pred = StagePrediction::new(stage, conf);
                for (s, j) in &self.joule_ema {
                    pred = pred.with_estimate(*s, *j);
                }
                pred
            }
            Err(_) => {
                let mut pred = StagePrediction::new(Stage::Cache, 0.0);
                for (s, j) in &self.joule_ema {
                    pred = pred.with_estimate(*s, *j);
                }
                pred
            }
        }
    }

    async fn observe(
        &mut self,
        query: &Query,
        chosen_stage: Stage,
        was_correct: bool,
        joule_cost: u64,
    ) {
        self.observe_cost(chosen_stage, joule_cost);
        if let Some(emb) = query.embedding.as_ref() {
            // Best-effort online SGD step; ignore dimension mismatch.
            let _ = self.sgd_step(emb, chosen_stage, was_correct, 0.01, 1e-4);
        }
    }

    fn export_state(&self) -> RouterState {
        let blob = serde_json::to_vec(self).unwrap_or_default();
        let mut meta = BTreeMap::new();
        meta.insert("latent_dim".to_string(), self.latent_dim.to_string());
        meta.insert("embedding_dim".to_string(), self.embedding_dim.to_string());
        RouterState {
            algorithm: "mf".to_string(),
            weights_blob: blob,
            metadata: meta,
        }
    }

    fn import_state(state: RouterState) -> Result<Self> {
        if state.algorithm != "mf" {
            return Err(Error::StateImportFailed(format!(
                "expected algorithm=mf, got {}",
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
    fn predict_probs_dim_check() {
        let r = MfRouter::new(8, 4, 42);
        let err = r.predict_probs(&[0.0; 7]).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 8);
                assert_eq!(got, 7);
            }
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn sgd_reduces_loss() {
        let mut r = MfRouter::new(4, 4, 7);
        let emb = vec![0.5, -0.5, 0.25, -0.25];
        let l1 = r.sgd_step(&emb, Stage::Neural, true, 0.1, 0.0).unwrap();
        for _ in 0..100 {
            r.sgd_step(&emb, Stage::Neural, true, 0.1, 0.0).unwrap();
        }
        let l2 = r.sgd_step(&emb, Stage::Neural, true, 0.1, 0.0).unwrap();
        assert!(l2 < l1, "loss should decrease ({l1} -> {l2})");
    }
}
