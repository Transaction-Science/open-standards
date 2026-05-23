//! Public router trait and shared value types.

use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;
use eoc_core::{Query, Stage};
use serde::{Deserialize, Serialize};

/// Prediction returned by a [`LearnedRouter`] for a single query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePrediction {
    /// Stage the router recommends running first.
    pub recommended: Stage,
    /// Confidence in `recommended` ∈ [0, 1].
    pub confidence: f32,
    /// Estimated microjoule cost per candidate stage.
    ///
    /// Keyed by `Stage` (which is `Hash + Eq` in `eoc-core`); use
    /// [`StagePrediction::iter_estimates`] for stable, cheap→expensive
    /// iteration order.
    pub estimated_joules: HashMap<Stage, u64>,
}

impl StagePrediction {
    /// Build a prediction with default (empty) joule estimates.
    pub fn new(recommended: Stage, confidence: f32) -> Self {
        Self {
            recommended,
            confidence: confidence.clamp(0.0, 1.0),
            estimated_joules: HashMap::new(),
        }
    }

    /// Attach a per-stage joule estimate.
    pub fn with_estimate(mut self, stage: Stage, microjoules: u64) -> Self {
        self.estimated_joules.insert(stage, microjoules);
        self
    }

    /// Iterate per-stage joule estimates in cheapest-first order.
    pub fn iter_estimates(&self) -> impl Iterator<Item = (Stage, u64)> + '_ {
        ALL_STAGES
            .iter()
            .filter_map(move |s| self.estimated_joules.get(s).map(|j| (*s, *j)))
    }
}

/// Serializable router state (algorithm tag + opaque blob + metadata).
///
/// Each router family chooses its own blob encoding. The crate uses JSON so
/// state is portable across machines / WASM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterState {
    /// Algorithm tag — e.g. `"mf"`, `"logreg"`, `"linucb"`, `"thompson"`.
    pub algorithm: String,
    /// Opaque serialised weights (algorithm-specific).
    pub weights_blob: Vec<u8>,
    /// Free-form metadata — training timestamp, dataset hash, etc.
    /// `BTreeMap` so the metadata serialises in a stable order.
    pub metadata: BTreeMap<String, String>,
}

/// A learned router predicts the best cascade stage from a query.
#[async_trait]
pub trait LearnedRouter: Send + Sync {
    /// Predict the best stage for `query`.
    async fn route(&self, query: &Query) -> StagePrediction;

    /// Online update — observe what actually happened.
    async fn observe(
        &mut self,
        query: &Query,
        chosen_stage: Stage,
        was_correct: bool,
        joule_cost: u64,
    );

    /// Export router weights for persistence / WASM transport.
    fn export_state(&self) -> RouterState;

    /// Reconstruct a router from previously-exported state.
    fn import_state(state: RouterState) -> crate::Result<Self>
    where
        Self: Sized;
}

/// The four cascade stages, in cheapest-first order, useful for iteration.
pub const ALL_STAGES: [Stage; 4] = [Stage::Cache, Stage::Kv, Stage::Graph, Stage::Neural];
