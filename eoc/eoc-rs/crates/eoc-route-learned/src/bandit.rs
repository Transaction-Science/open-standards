//! Bandit routers — online learning when labels arrive after the decision.
//!
//! Two algorithms ship here:
//!
//! - **LinUCB** (Li et al. 2010) — contextual bandit with linear reward
//!   models. Each arm (stage) keeps `A = X^T X + I` and `b = X^T r`. Pick
//!   the arm that maximises `θ_a · x + α · √(x^T A^{-1} x)`.
//! - **Thompson Sampling** — Beta-Bernoulli on per-arm success rates. Cheap
//!   and surprisingly competitive when contexts are noisy.

use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;
use eoc_core::{Query, Stage};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::router::{ALL_STAGES, LearnedRouter, RouterState, StagePrediction};

/// Per-arm state for [`LinUcbRouter`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinUcbArm {
    /// Context dimension `d`.
    pub dim: usize,
    /// `A = λ I + Σ x x^T`
    pub a: DMatrix<f32>,
    /// `b = Σ r · x`
    pub b: DVector<f32>,
}

impl LinUcbArm {
    /// Fresh arm seeded with `λ · I`.
    pub fn new(dim: usize, ridge: f32) -> Self {
        let mut a = DMatrix::<f32>::zeros(dim, dim);
        for i in 0..dim {
            a[(i, i)] = ridge;
        }
        Self {
            dim,
            a,
            b: DVector::<f32>::zeros(dim),
        }
    }

    fn theta(&self) -> DVector<f32> {
        match self.a.clone().try_inverse() {
            Some(a_inv) => a_inv * &self.b,
            None => DVector::<f32>::zeros(self.dim),
        }
    }

    fn ucb(&self, x: &DVector<f32>, alpha: f32) -> f32 {
        let theta = self.theta();
        let mean = theta.dot(x);
        let a_inv = self
            .a
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::<f32>::identity(self.dim, self.dim));
        let bonus_sq = (x.transpose() * &a_inv * x)[(0, 0)].max(0.0);
        mean + alpha * bonus_sq.sqrt()
    }

    fn update(&mut self, x: &DVector<f32>, reward: f32) {
        self.a += x * x.transpose();
        self.b += x * reward;
    }
}

/// LinUCB contextual bandit over the four cascade stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinUcbRouter {
    /// Context (embedding) dimension.
    pub dim: usize,
    /// Exploration coefficient.
    pub alpha: f32,
    /// Ridge regularisation `λ`.
    pub ridge: f32,
    /// Per-stage arm state.
    pub arms: HashMap<Stage, LinUcbArm>,
}

impl LinUcbRouter {
    /// Create a new router. `alpha` is the exploration coefficient (try 1.0).
    pub fn new(dim: usize, alpha: f32, ridge: f32) -> Self {
        let mut arms = HashMap::new();
        for s in ALL_STAGES {
            arms.insert(s, LinUcbArm::new(dim, ridge));
        }
        Self {
            dim,
            alpha,
            ridge,
            arms,
        }
    }

    /// Argmax UCB across stages.
    pub fn pick(&self, embedding: &[f32]) -> Result<(Stage, f32)> {
        if embedding.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: embedding.len(),
            });
        }
        let x = DVector::<f32>::from_row_slice(embedding);
        let mut best = Stage::Cache;
        let mut best_score = f32::NEG_INFINITY;
        for s in ALL_STAGES {
            let arm = match self.arms.get(&s) {
                Some(a) => a,
                None => continue,
            };
            let u = arm.ucb(&x, self.alpha);
            if u > best_score {
                best_score = u;
                best = s;
            }
        }
        Ok((best, best_score))
    }

    /// Update the chosen arm with the observed reward.
    pub fn update(&mut self, embedding: &[f32], stage: Stage, reward: f32) -> Result<()> {
        if embedding.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: embedding.len(),
            });
        }
        let x = DVector::<f32>::from_row_slice(embedding);
        if let Some(arm) = self.arms.get_mut(&stage) {
            arm.update(&x, reward);
        }
        Ok(())
    }
}

#[async_trait]
impl LearnedRouter for LinUcbRouter {
    async fn route(&self, query: &Query) -> StagePrediction {
        let Some(emb) = query.embedding.as_ref() else {
            return StagePrediction::new(Stage::Cache, 0.0);
        };
        match self.pick(emb) {
            Ok((stage, score)) => {
                // Squash UCB score into [0,1] via logistic for "confidence".
                let conf = 1.0 / (1.0 + (-score).exp());
                StagePrediction::new(stage, conf)
            }
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
        if let Some(emb) = query.embedding.as_ref() {
            let reward = if was_correct { 1.0 } else { 0.0 };
            let _ = self.update(emb, chosen_stage, reward);
        }
    }

    fn export_state(&self) -> RouterState {
        let blob = serde_json::to_vec(self).unwrap_or_default();
        let mut meta = BTreeMap::new();
        meta.insert("dim".to_string(), self.dim.to_string());
        meta.insert("alpha".to_string(), self.alpha.to_string());
        RouterState {
            algorithm: "linucb".to_string(),
            weights_blob: blob,
            metadata: meta,
        }
    }

    fn import_state(state: RouterState) -> Result<Self> {
        if state.algorithm != "linucb" {
            return Err(Error::StateImportFailed(format!(
                "expected algorithm=linucb, got {}",
                state.algorithm
            )));
        }
        serde_json::from_slice(&state.weights_blob)
            .map_err(|e| Error::StateImportFailed(e.to_string()))
    }
}

/// Beta arm for [`ThompsonSamplingRouter`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaArm {
    /// Successes + 1 (Laplace prior).
    pub alpha: f32,
    /// Failures + 1.
    pub beta: f32,
}

impl BetaArm {
    /// Fresh arm with `Beta(1, 1)` prior (uniform).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl Default for BetaArm {
    fn default() -> Self {
        Self::new()
    }
}

/// Beta-Bernoulli Thompson sampling over the four stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThompsonSamplingRouter {
    /// Per-stage Beta posteriors.
    pub arms: HashMap<Stage, BetaArm>,
    /// RNG seed for reproducibility — incremented every sample.
    pub seed: u64,
}

impl ThompsonSamplingRouter {
    /// Construct a fresh sampler.
    pub fn new(seed: u64) -> Self {
        let mut arms = HashMap::new();
        for s in ALL_STAGES {
            arms.insert(s, BetaArm::new());
        }
        Self { arms, seed }
    }

    /// Sample once from each arm; return the argmax. Iteration order is the
    /// fixed cheapest-first stage order, so the result is deterministic for
    /// a given `seed`.
    pub fn pick(&self) -> (Stage, f32) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut best = Stage::Cache;
        let mut best_v = f32::NEG_INFINITY;
        for s in ALL_STAGES {
            let arm = match self.arms.get(&s) {
                Some(a) => a,
                None => continue,
            };
            let dist = Beta::new(arm.alpha as f64, arm.beta as f64)
                .unwrap_or_else(|_| Beta::new(1.0, 1.0).expect("beta(1,1) ok"));
            let v = dist.sample(&mut rng) as f32;
            if v > best_v {
                best_v = v;
                best = s;
            }
        }
        (best, best_v)
    }

    /// Update Beta posterior for `stage`.
    pub fn update(&mut self, stage: Stage, was_correct: bool) {
        if let Some(arm) = self.arms.get_mut(&stage) {
            if was_correct {
                arm.alpha += 1.0;
            } else {
                arm.beta += 1.0;
            }
        }
        self.seed = self.seed.wrapping_add(1);
    }
}

#[async_trait]
impl LearnedRouter for ThompsonSamplingRouter {
    async fn route(&self, _query: &Query) -> StagePrediction {
        let (stage, v) = self.pick();
        StagePrediction::new(stage, v.clamp(0.0, 1.0))
    }

    async fn observe(
        &mut self,
        _query: &Query,
        chosen_stage: Stage,
        was_correct: bool,
        _joule_cost: u64,
    ) {
        self.update(chosen_stage, was_correct);
    }

    fn export_state(&self) -> RouterState {
        let blob = serde_json::to_vec(self).unwrap_or_default();
        let meta = BTreeMap::new();
        RouterState {
            algorithm: "thompson".to_string(),
            weights_blob: blob,
            metadata: meta,
        }
    }

    fn import_state(state: RouterState) -> Result<Self> {
        if state.algorithm != "thompson" {
            return Err(Error::StateImportFailed(format!(
                "expected algorithm=thompson, got {}",
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
    fn arm_init_is_ridge_identity() {
        let arm = LinUcbArm::new(4, 1.0);
        for i in 0..4 {
            assert_eq!(arm.a[(i, i)], 1.0);
        }
        assert_eq!(arm.b.iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn thompson_update_changes_posterior() {
        let mut ts = ThompsonSamplingRouter::new(0);
        ts.update(Stage::Cache, true);
        let arm = ts.arms.get(&Stage::Cache).expect("cache arm exists");
        assert_eq!(arm.alpha, 2.0);
        assert_eq!(arm.beta, 1.0);
    }
}
