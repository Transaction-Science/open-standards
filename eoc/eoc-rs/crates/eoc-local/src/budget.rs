//! Pre-inference joule budget check.
//!
//! Local inference is *not* free. Even with a measured counter the
//! marginal cost is the dominant operating expense — the whole point of
//! EOC. Before running a query through a local backend, callers can ask
//! the budget module: given this model, this prompt, this joule
//! ceiling, would running locally exceed the budget? If yes, return a
//! `Punt` decision pointing at a cheaper model or punt up to the
//! vendor-API tier (which may itself be cheaper for short prompts at
//! today's coefficients).
//!
//! Composition: this module slots into `eoc-cascade` as a *pre-stage*
//! decision — a sibling of the cascade itself.

use crate::error::LocalResult;
use crate::model_registry::ModelEntry;

/// Joule budget for a single query.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Maximum joule cost the operator is willing to pay, in micro-joules.
    pub max_microjoules: u64,
    /// Maximum tokens to generate. Helps bound the estimate.
    pub max_tokens: u32,
}

impl Budget {
    /// New budget. `1_000_000_000` µJ = 1 kJ ≈ small-model inference.
    pub fn new(max_microjoules: u64, max_tokens: u32) -> Self {
        Self {
            max_microjoules,
            max_tokens,
        }
    }
}

/// Outcome of a budget check.
#[derive(Debug, Clone)]
pub enum BudgetDecision {
    /// Run the model — the prediction came in under budget.
    Run {
        /// Predicted joule cost in micro-joules.
        predicted_microjoules: u64,
    },
    /// Predicted cost exceeds the budget. Caller should pick a cheaper
    /// path (smaller model, cached answer, or vendor punt).
    Punt {
        /// Predicted cost of running the requested model.
        predicted_microjoules: u64,
        /// Configured budget.
        budget_microjoules: u64,
        /// Human-readable reason.
        reason: String,
    },
}

/// Policy for translating model size + prompt length into a predicted
/// joule cost. Reference values are derived from llama.cpp benchmark
/// runs on Apple M3 Max; production deployments should override.
#[derive(Debug, Clone, Copy)]
pub struct BudgetPolicy {
    /// Joules per *input* token, average across quantizations.
    pub joules_per_input_token: f64,
    /// Joules per *output* token (typically 5-10x input).
    pub joules_per_output_token: f64,
    /// Multiplier applied to both coefficients per GiB of model size
    /// (so a 70B-parameter model pays proportionally more).
    pub size_factor_per_gib: f64,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        // Order-of-magnitude defaults. Calibrate to host hardware.
        Self {
            joules_per_input_token: 0.01,
            joules_per_output_token: 0.05,
            size_factor_per_gib: 0.10,
        }
    }
}

impl BudgetPolicy {
    /// Predict the joule cost (in µJ) of running `prompt_tokens` input
    /// + up to `max_output_tokens` output tokens through `entry`.
    pub fn predict_microjoules(
        &self,
        entry: &ModelEntry,
        prompt_tokens: u32,
        max_output_tokens: u32,
    ) -> u64 {
        let gib = (entry.size_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
        let scale = 1.0 + self.size_factor_per_gib * gib;
        let j = (prompt_tokens as f64) * self.joules_per_input_token * scale
            + (max_output_tokens as f64) * self.joules_per_output_token * scale;
        (j * 1_000_000.0).max(0.0) as u64
    }

    /// Decide whether to run the query within the budget.
    pub fn decide(
        &self,
        entry: &ModelEntry,
        prompt_tokens: u32,
        budget: Budget,
    ) -> LocalResult<BudgetDecision> {
        let predicted =
            self.predict_microjoules(entry, prompt_tokens, budget.max_tokens);
        if predicted <= budget.max_microjoules {
            Ok(BudgetDecision::Run {
                predicted_microjoules: predicted,
            })
        } else {
            Ok(BudgetDecision::Punt {
                predicted_microjoules: predicted,
                budget_microjoules: budget.max_microjoules,
                reason: format!(
                    "predicted {predicted} µJ exceeds budget {} µJ for model {}",
                    budget.max_microjoules, entry.name
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_registry::{BackendKind, Quantization};
    use std::path::PathBuf;

    fn tiny() -> ModelEntry {
        ModelEntry {
            name: "tinyllama".into(),
            backend: BackendKind::LlamaCpp,
            path: PathBuf::from("/tmp/tinyllama.gguf"),
            size_bytes: 600 * 1024 * 1024, // 600 MiB
            quantization: Quantization::Q4KM,
            context_window: Some(2048),
            architecture: Some("llama".into()),
        }
    }

    #[test]
    fn predict_is_monotonic_in_tokens() {
        let policy = BudgetPolicy::default();
        let entry = tiny();
        let a = policy.predict_microjoules(&entry, 100, 100);
        let b = policy.predict_microjoules(&entry, 100, 200);
        let c = policy.predict_microjoules(&entry, 200, 100);
        assert!(b > a);
        assert!(c > a);
    }

    #[test]
    fn decide_runs_under_budget() {
        let policy = BudgetPolicy::default();
        let entry = tiny();
        let budget = Budget::new(10_000_000_000, 256);
        match policy.decide(&entry, 64, budget).unwrap() {
            BudgetDecision::Run { .. } => {}
            other => panic!("expected Run, got {other:?}"),
        }
    }

    #[test]
    fn decide_punts_when_over_budget() {
        let policy = BudgetPolicy::default();
        let entry = tiny();
        let budget = Budget::new(1, 4096); // 1 µJ — absurdly tight
        match policy.decide(&entry, 4096, budget).unwrap() {
            BudgetDecision::Punt { .. } => {}
            other => panic!("expected Punt, got {other:?}"),
        }
    }
}
