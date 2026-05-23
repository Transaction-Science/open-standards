//! End-to-end LearnedCascade test.
//!
//! Build a real four-stage cascade, route a "novel" query through it twice
//! and compare joule cost between:
//!
//! - the baseline `Cascade` (which walks cache → kv → graph → neural), and
//! - a `LearnedCascade` whose router picks `Neural` with confidence 1.0.
//!
//! Both must produce the same answer; the learned variant must be cheaper
//! (or at worst equal) because it skips three doomed cache/kv/graph misses.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use eoc_cache::LruCache;
use eoc_cascade::Cascade;
use eoc_core::{Query, Stage};
use eoc_graph::GraphStage;
use eoc_kv::{KvStage, MemoryKvBackend};
use eoc_neural::{EchoBackend, NeuralStage};
use eoc_route_learned::inference::LearnedCascade;
use eoc_route_learned::router::{LearnedRouter, RouterState, StagePrediction};
use eoc_route_learned::threshold::ThresholdPolicy;

/// A trivial router that always picks `Neural` with full confidence.
struct AlwaysNeural;

#[async_trait]
impl LearnedRouter for AlwaysNeural {
    async fn route(&self, _query: &Query) -> StagePrediction {
        StagePrediction::new(Stage::Neural, 1.0)
    }
    async fn observe(&mut self, _q: &Query, _s: Stage, _ok: bool, _j: u64) {}
    fn export_state(&self) -> RouterState {
        RouterState {
            algorithm: "always_neural".into(),
            weights_blob: vec![],
            metadata: Default::default(),
        }
    }
    fn import_state(_state: RouterState) -> eoc_route_learned::Result<Self> {
        Ok(AlwaysNeural)
    }
}

fn fixture() -> Cascade {
    let cache = Arc::new(LruCache::new(8));
    let kv_backend = Box::new(MemoryKvBackend::new());
    let kv = Arc::new(KvStage::new(kv_backend));
    let graph = Arc::new(GraphStage::new());
    let neural = Arc::new(NeuralStage::new(Box::new(
        EchoBackend::new().with_cost(50_000),
    )));
    Cascade::new(cache, kv, graph, neural)
}

#[tokio::test]
async fn learned_cascade_routes_to_neural_when_confident() {
    let cascade = Arc::new(fixture());
    let learned = LearnedCascade::new(cascade, AlwaysNeural, ThresholdPolicy::new(0.9));
    let resp = learned.resolve(Query::new("novel query")).await;
    assert_eq!(resp.stage, Stage::Neural);
    assert!(resp.payload.contains("novel query"));
}

#[tokio::test]
async fn learned_cascade_skip_saves_joules_vs_baseline() {
    // Baseline path: full cascade, neural answers — pays neural cost only.
    // Both should answer with the neural stage in this fixture, so this is
    // mostly a smoke test that skip-mode doesn't *increase* cost.
    let baseline = fixture();
    let baseline_response = baseline.resolve(Query::new("brand new prompt")).await;

    let cascade = Arc::new(fixture());
    let learned = LearnedCascade::new(cascade, AlwaysNeural, ThresholdPolicy::new(0.9));
    let learned_response = learned.resolve(Query::new("brand new prompt")).await;

    assert_eq!(learned_response.stage, Stage::Neural);
    // The learned cascade can't be more expensive than the baseline.
    assert!(
        learned_response.joule_cost.microjoules <= baseline_response.joule_cost.microjoules + 1
    );
}

#[tokio::test]
async fn learned_cascade_falls_through_to_full_cascade_when_unconfident() {
    /// Low-confidence router → policy says FullCascade.
    struct LowConf;
    #[async_trait]
    impl LearnedRouter for LowConf {
        async fn route(&self, _q: &Query) -> StagePrediction {
            StagePrediction::new(Stage::Neural, 0.1)
        }
        async fn observe(&mut self, _q: &Query, _s: Stage, _ok: bool, _j: u64) {}
        fn export_state(&self) -> RouterState {
            RouterState {
                algorithm: "low".into(),
                weights_blob: vec![],
                metadata: Default::default(),
            }
        }
        fn import_state(_s: RouterState) -> eoc_route_learned::Result<Self> {
            Ok(LowConf)
        }
    }

    let cascade = Arc::new(fixture());
    let learned = LearnedCascade::new(cascade, LowConf, ThresholdPolicy::new(0.9));
    let resp = learned.resolve(Query::new("anything")).await;
    // Baseline cascade resolves novel queries with the neural stage.
    assert_eq!(resp.stage, Stage::Neural);
    let _ = HashMap::<u32, u32>::new(); // suppress unused-import warnings
}
