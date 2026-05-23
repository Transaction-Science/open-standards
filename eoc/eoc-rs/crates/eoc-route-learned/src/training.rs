//! Training loops + dataset I/O.
//!
//! `Example` is the on-disk shape (JSONL) for `(query, stage, success, cost)`
//! triples. Two batch trainers ship: `train_mf` (matrix factorisation) and
//! `train_logreg` (logistic regression). Both consume `&[Example]` and emit a
//! ready-to-serve router.

use std::path::Path;

use eoc_core::Stage;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::classifier::LogRegRouter;
use crate::error::{Error, Result};
use crate::matrix_factorization::{DEFAULT_LATENT_DIM, MfRouter};

/// One labeled training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Query embedding (must have consistent dim across the dataset).
    pub embedding: Vec<f32>,
    /// Stage that ran (or that we're labelling).
    pub stage: Stage,
    /// Did `stage` satisfy the query?
    pub success: bool,
    /// Observed microjoule cost.
    pub cost_microjoules: u64,
}

impl Example {
    /// Convenience constructor.
    pub fn new(embedding: Vec<f32>, stage: Stage, success: bool, cost_microjoules: u64) -> Self {
        Self {
            embedding,
            stage,
            success,
            cost_microjoules,
        }
    }
}

/// Hyper-parameters for the batch trainers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Mini-batch size (currently SGD step is per-example; reserved for future).
    pub batch_size: usize,
    /// SGD learning rate.
    pub learning_rate: f32,
    /// Number of passes through the dataset.
    pub epochs: usize,
    /// Fraction of `examples` held out for validation (informational).
    pub validation_split: f32,
    /// L2 regularisation strength.
    pub l2: f32,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.05,
            epochs: 20,
            validation_split: 0.1,
            l2: 1e-4,
            seed: 1729,
        }
    }
}

/// Read a JSONL file of `Example` rows.
pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Vec<Example>> {
    let bytes = std::fs::read(path).map_err(|e| Error::StateImportFailed(e.to_string()))?;
    load_jsonl_bytes(&bytes)
}

/// Parse a JSONL byte buffer of `Example` rows.
pub fn load_jsonl_bytes(bytes: &[u8]) -> Result<Vec<Example>> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error::StateImportFailed(e.to_string()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ex: Example = serde_json::from_str(line).map_err(|e| {
            Error::StateImportFailed(format!("line {}: {e}", i + 1))
        })?;
        out.push(ex);
    }
    Ok(out)
}

fn first_embedding_dim(examples: &[Example]) -> Result<usize> {
    examples
        .first()
        .map(|e| e.embedding.len())
        .ok_or(Error::NotTrained)
}

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    idx
}

/// Train an MF router by SGD over `examples`.
pub fn train_mf(examples: &[Example], config: TrainingConfig) -> MfRouter {
    train_mf_with_dim(examples, config, DEFAULT_LATENT_DIM)
}

/// Train an MF router with a custom latent dim.
pub fn train_mf_with_dim(
    examples: &[Example],
    config: TrainingConfig,
    latent_dim: usize,
) -> MfRouter {
    let dim = first_embedding_dim(examples).unwrap_or(1);
    let mut router = MfRouter::new(dim, latent_dim, config.seed);
    if examples.is_empty() {
        return router;
    }
    for epoch in 0..config.epochs {
        let order = shuffled_indices(examples.len(), config.seed.wrapping_add(epoch as u64));
        for i in order {
            let e = &examples[i];
            let _ = router.sgd_step(
                &e.embedding,
                e.stage,
                e.success,
                config.learning_rate,
                config.l2,
            );
            router.observe_cost(e.stage, e.cost_microjoules);
        }
    }
    router
}

/// Train a logistic-regression router. Only successful examples shape the
/// classifier — failed-stage examples are skipped because we want
/// `P(stage = correct | x)`.
pub fn train_logreg(examples: &[Example], config: TrainingConfig) -> LogRegRouter {
    let dim = first_embedding_dim(examples).unwrap_or(1);
    let mut router = LogRegRouter::new(dim, config.l2, config.seed);
    if examples.is_empty() {
        return router;
    }
    let positive: Vec<&Example> = examples.iter().filter(|e| e.success).collect();
    if positive.is_empty() {
        return router;
    }
    for epoch in 0..config.epochs {
        let order = shuffled_indices(positive.len(), config.seed.wrapping_add(epoch as u64));
        for i in order {
            let e = positive[i];
            let _ = router.sgd_step(&e.embedding, e.stage, config.learning_rate);
        }
    }
    router
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_jsonl_bytes_roundtrip() {
        let ex = Example::new(vec![0.1, 0.2], Stage::Kv, true, 1234);
        let line = serde_json::to_string(&ex).unwrap();
        let parsed = load_jsonl_bytes(line.as_bytes()).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].cost_microjoules, 1234);
    }

    #[test]
    fn train_with_empty_dataset_returns_untrained() {
        let r = train_logreg(&[], TrainingConfig::default());
        assert_eq!(r.embedding_dim(), 1);
    }
}
