//! EOC cascade — the four-stage memoizing pipeline.
//!
//! `Cascade::resolve` walks `cache → kv → graph → neural` in order. The
//! first stage to return `Some(Response)` wins. Joule cost is the sum of
//! the stage-level cost reported by the stage plus any cascade-level
//! overhead measured by the meter between the start and end of the call.

#![forbid(unsafe_code)]

use std::sync::Arc;

use eoc_cache::{LruCache, Stage};
use eoc_core::{JouleCost, JouleSource, Query, Response};
use eoc_graph::GraphStage;
use eoc_kv::KvStage;
use eoc_meter::{JouleCounter, StubCounter};
use eoc_neural::NeuralStage;

/// The four-stage cascade.
pub struct Cascade {
    cache: Arc<LruCache>,
    kv: Arc<KvStage>,
    graph: Arc<GraphStage>,
    neural: Arc<NeuralStage>,
    meter: Arc<dyn JouleCounter>,
    /// Whether to memoize neural/graph/kv answers back into the cache.
    memoize: bool,
}

impl Cascade {
    /// Construct a cascade. The meter defaults to `StubCounter`.
    pub fn new(
        cache: Arc<LruCache>,
        kv: Arc<KvStage>,
        graph: Arc<GraphStage>,
        neural: Arc<NeuralStage>,
    ) -> Self {
        Self {
            cache,
            kv,
            graph,
            neural,
            meter: Arc::new(StubCounter),
            memoize: true,
        }
    }

    /// Swap in a different joule counter.
    pub fn with_meter(mut self, meter: Arc<dyn JouleCounter>) -> Self {
        self.meter = meter;
        self
    }

    /// Disable memoization (each query re-walks the full cascade).
    pub fn without_memoization(mut self) -> Self {
        self.memoize = false;
        self
    }

    /// Borrow the cache (for warmup / wiring).
    pub fn cache(&self) -> &LruCache {
        &self.cache
    }
    /// Borrow the kv stage.
    pub fn kv(&self) -> &KvStage {
        &self.kv
    }
    /// Borrow the graph stage.
    pub fn graph(&self) -> &GraphStage {
        &self.graph
    }
    /// Borrow the neural stage.
    pub fn neural(&self) -> &NeuralStage {
        &self.neural
    }
    /// Borrow the joule meter.
    pub fn meter(&self) -> &dyn JouleCounter {
        self.meter.as_ref()
    }

    /// Resolve a query through the cascade.
    ///
    /// On a miss at one stage, falls through to the next. The neural
    /// stage always answers, so this function always returns `Some` if
    /// the neural stage's backend is well-behaved. We still return
    /// `Option<Response>` for symmetry with the `Stage` trait.
    pub async fn resolve(&self, q: Query) -> Response {
        let start = self.meter.read_microjoules().ok();

        // Walk stages in order.
        let result = if let Some(r) = self.cache.try_resolve(&q).await {
            r
        } else if let Some(r) = self.kv.try_resolve(&q).await {
            self.maybe_memoize(&q, &r);
            r
        } else if let Some(r) = self.graph.try_resolve(&q).await {
            self.maybe_memoize(&q, &r);
            r
        } else {
            // Neural always answers.
            let r = self
                .neural
                .try_resolve(&q)
                .await
                .expect("neural stage must always return Some");
            self.maybe_memoize(&q, &r);
            r
        };

        // Re-derive the response with the cascade-attributed joule cost
        // (stage-reported + measured overhead, if any).
        attach_cascade_cost(result, start, self.meter.as_ref())
    }

    fn maybe_memoize(&self, _q: &Query, r: &Response) {
        if self.memoize {
            // Memoize as a *cache* entry so future hits report `Stage::Cache`
            // and `JouleCost::zero()`. The original receipt is no longer
            // valid (different stage + cost) — we mint a new one.
            let cached =
                Response::new(r.query_id, r.payload.clone(), eoc_core::Stage::Cache, JouleCost::zero());
            self.cache.insert_response(r.query_id, cached);
        }
    }
}

fn attach_cascade_cost(
    mut response: Response,
    start: Option<u64>,
    meter: &dyn JouleCounter,
) -> Response {
    let end = meter.read_microjoules().ok();
    let overhead = match (start, end) {
        (Some(s), Some(e)) if e >= s => e - s,
        _ => 0,
    };
    let stage_cost = response.joule_cost.microjoules;
    let total = stage_cost.saturating_add(overhead);
    let source = if overhead > 0 {
        JouleSource::Measured
    } else {
        response.joule_cost.source
    };
    let joule_cost = JouleCost {
        microjoules: total,
        source,
    };
    // Rebuild the response so the receipt covers the final joule cost.
    response = Response::new(response.query_id, response.payload, response.stage, joule_cost);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use eoc_cache::LruCache;
    use eoc_core::Stage as StageKind;
    use eoc_graph::{GraphStage, Triple};
    use eoc_kv::{KvBackend, KvStage, MemoryKvBackend};
    use eoc_neural::{EchoBackend, NeuralStage};

    fn fixture() -> Cascade {
        let cache = Arc::new(LruCache::new(64));
        let kv_backend = Box::new(MemoryKvBackend::new());
        kv_backend.put("known key", b"known value".to_vec());
        let kv = Arc::new(KvStage::new(kv_backend));
        let graph = Arc::new(GraphStage::new());
        graph.insert(Triple::new("Mars", "fourth planet from", "the Sun"));
        let neural = Arc::new(NeuralStage::new(Box::new(
            EchoBackend::new().with_cost(50_000_000),
        )));
        Cascade::new(cache, kv, graph, neural)
    }

    #[tokio::test]
    async fn cascade_uses_kv_when_it_can() {
        let cascade = fixture();
        let r = cascade.resolve(Query::new("known key")).await;
        assert_eq!(r.stage, StageKind::Kv);
        assert_eq!(r.payload, "known value");
    }

    #[tokio::test]
    async fn cascade_uses_graph_when_kv_misses() {
        let cascade = fixture();
        let r = cascade
            .resolve(Query::new("Mars is the fourth planet from what?"))
            .await;
        assert_eq!(r.stage, StageKind::Graph);
        assert_eq!(r.payload, "the Sun");
    }

    #[tokio::test]
    async fn cascade_falls_through_to_neural() {
        let cascade = fixture();
        let r = cascade.resolve(Query::new("totally novel question")).await;
        assert_eq!(r.stage, StageKind::Neural);
        assert!(r.payload.contains("totally novel question"));
        assert!(r.joule_cost.microjoules > 0);
    }

    #[tokio::test]
    async fn cascade_memoizes_into_cache() {
        let cascade = fixture();
        let q = Query::new("totally novel question");
        let r1 = cascade.resolve(q.clone()).await;
        assert_eq!(r1.stage, StageKind::Neural);
        // Same query the second time round should hit the cache.
        let r2 = cascade.resolve(q).await;
        assert_eq!(r2.stage, StageKind::Cache);
    }
}
