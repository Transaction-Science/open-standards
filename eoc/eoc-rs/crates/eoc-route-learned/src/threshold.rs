//! Threshold policy — turn a [`StagePrediction`] into a skip/no-skip decision.
//!
//! The router *suggests* a stage and a confidence; the policy decides whether
//! to act on it. Two budgets gate the skip:
//!
//! - **confidence_threshold** — minimum confidence required to skip cheaper stages.
//! - **joule_budget** — if the cheap-stage estimates exceed this budget, skip
//!   them even at lower confidence.
//! - **latency_budget** — informational; deployments can reject decisions that
//!   exceed it.

use std::time::Duration;

use eoc_core::Stage;
use serde::{Deserialize, Serialize};

use crate::router::StagePrediction;

/// Outcome of applying a [`ThresholdPolicy`] to a [`StagePrediction`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThresholdDecision {
    /// Run the full cascade (cache → kv → graph → neural).
    FullCascade,
    /// Skip cheaper stages and start at `Stage`.
    SkipTo(Stage),
}

/// Policy that turns a confidence into a concrete skip decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdPolicy {
    /// Minimum router confidence required to skip cheaper stages.
    pub confidence_threshold: f32,
    /// Optional total joule budget — if exceeded by predicted cheap-stage
    /// costs, we skip even at lower confidence.
    pub joule_budget: Option<u64>,
    /// Optional latency budget (informational).
    pub latency_budget: Option<Duration>,
}

impl ThresholdPolicy {
    /// Sensible defaults: 0.9 threshold, no budget caps.
    pub fn new(confidence_threshold: f32) -> Self {
        Self {
            confidence_threshold,
            joule_budget: None,
            latency_budget: None,
        }
    }

    /// Cap the joule budget for skipped cheap stages.
    pub fn with_joule_budget(mut self, microjoules: u64) -> Self {
        self.joule_budget = Some(microjoules);
        self
    }

    /// Cap the latency budget.
    pub fn with_latency_budget(mut self, latency: Duration) -> Self {
        self.latency_budget = Some(latency);
        self
    }

    /// Decide whether to skip ahead.
    pub fn decide(&self, prediction: &StagePrediction) -> ThresholdDecision {
        let confident = prediction.confidence >= self.confidence_threshold;

        // If a joule budget is set, also skip when the predicted cheap-stage
        // cost up to (and including) `recommended - 1` would blow the budget.
        let cheap_cost_exceeds_budget = match self.joule_budget {
            Some(budget) => cheap_stage_cost(prediction) > budget,
            None => false,
        };

        if confident || cheap_cost_exceeds_budget {
            ThresholdDecision::SkipTo(prediction.recommended)
        } else {
            ThresholdDecision::FullCascade
        }
    }
}

impl Default for ThresholdPolicy {
    fn default() -> Self {
        Self::new(0.9)
    }
}

fn cheap_stage_cost(prediction: &StagePrediction) -> u64 {
    let order = stage_order(prediction.recommended);
    prediction
        .estimated_joules
        .iter()
        .filter(|(s, _)| stage_order(**s) < order)
        .map(|(_, j)| *j)
        .sum()
}

fn stage_order(s: Stage) -> u8 {
    match s {
        Stage::Cache => 0,
        Stage::Kv => 1,
        Stage::Graph => 2,
        Stage::Neural => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn below_threshold_runs_full_cascade() {
        let p = StagePrediction::new(Stage::Neural, 0.5);
        let policy = ThresholdPolicy::new(0.9);
        assert_eq!(policy.decide(&p), ThresholdDecision::FullCascade);
    }

    #[test]
    fn above_threshold_skips() {
        let p = StagePrediction::new(Stage::Neural, 0.95);
        let policy = ThresholdPolicy::new(0.9);
        assert_eq!(policy.decide(&p), ThresholdDecision::SkipTo(Stage::Neural));
    }

    #[test]
    fn budget_forces_skip() {
        let mut estimates = HashMap::new();
        estimates.insert(Stage::Cache, 5_000);
        estimates.insert(Stage::Kv, 10_000_000);
        estimates.insert(Stage::Graph, 5_000_000);
        let p = StagePrediction {
            recommended: Stage::Neural,
            confidence: 0.4,
            estimated_joules: estimates,
        };
        let policy = ThresholdPolicy::new(0.9).with_joule_budget(1_000_000);
        assert_eq!(policy.decide(&p), ThresholdDecision::SkipTo(Stage::Neural));
    }
}
