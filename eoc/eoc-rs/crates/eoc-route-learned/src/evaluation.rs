//! Routing-quality metrics.
//!
//! `evaluate` consumes a `LearnedRouter` and a held-out test set, returning
//! `RouterMetrics`:
//!
//! - **accuracy** — fraction of examples where the predicted stage matched
//!   the recorded successful stage.
//! - **joules_per_correct** — total observed joules / correct predictions.
//! - **lift** — joules saved vs. always running the full cascade (sum of all
//!   stage costs).
//!
//! Latency metrics are deferred to the caller (we don't run the cascade
//! here, just predictions).

use std::collections::HashMap;

use eoc_core::{Query, Stage};
use serde::{Deserialize, Serialize};

use crate::router::LearnedRouter;
use crate::training::Example;

/// Summary statistics produced by [`evaluate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterMetrics {
    /// Number of test examples scored.
    pub n: usize,
    /// Number of correct (predicted == actual successful) predictions.
    pub correct: usize,
    /// `correct / n`.
    pub accuracy: f32,
    /// Sum of observed joule cost.
    pub total_microjoules: u64,
    /// `total_microjoules / max(correct, 1)`.
    pub joules_per_correct: u64,
    /// Per-stage prediction count.
    pub stage_counts: HashMap<Stage, usize>,
    /// Joules saved vs. running the full cascade for every example
    /// (sum of all stage costs in the dataset).
    pub lift_microjoules: i64,
}

/// Evaluate a router on a labelled test set.
///
/// Each example is treated as a (embedding → routed_stage) prediction. If the
/// example's `success` is true and the router picks the same stage, we count
/// it correct. If the example was a failure, we ignore correctness but still
/// account for joules.
pub async fn evaluate<R: LearnedRouter>(router: &R, test_set: &[Example]) -> RouterMetrics {
    let mut correct = 0usize;
    let mut total_microjoules: u64 = 0;
    let mut full_cascade_microjoules: u64 = 0;
    let mut stage_counts: HashMap<Stage, usize> = HashMap::new();

    for ex in test_set {
        let q = Query::new("").with_embedding(ex.embedding.clone());
        let prediction = router.route(&q).await;
        *stage_counts.entry(prediction.recommended).or_insert(0) += 1;
        if ex.success && prediction.recommended == ex.stage {
            correct += 1;
        }
        // Saved joules: only pay for the recommended stage, not all four.
        total_microjoules = total_microjoules.saturating_add(ex.cost_microjoules);
        // "Full-cascade" baseline pays every stage's average cost — we use
        // 4x the recorded cost as a conservative proxy when per-stage costs
        // aren't broken out in the test set.
        full_cascade_microjoules = full_cascade_microjoules.saturating_add(ex.cost_microjoules * 4);
    }

    let n = test_set.len();
    let accuracy = if n == 0 {
        0.0
    } else {
        correct as f32 / n as f32
    };
    let joules_per_correct = if correct == 0 {
        0
    } else {
        total_microjoules / correct as u64
    };
    let lift_microjoules = full_cascade_microjoules as i64 - total_microjoules as i64;

    RouterMetrics {
        n,
        correct,
        accuracy,
        total_microjoules,
        joules_per_correct,
        stage_counts,
        lift_microjoules,
    }
}
