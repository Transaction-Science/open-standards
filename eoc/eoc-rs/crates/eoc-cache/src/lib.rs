//! EOC stage 1 — cache.
//!
//! The cheapest stage: an LRU keyed by `QueryId` returning a previously
//! computed `Response`. Cache hits are reported with `JouleCost::zero()` —
//! the energy of a hashmap lookup is below the noise floor of any
//! contemporary hardware energy counter.

#![forbid(unsafe_code)]

use std::num::NonZeroUsize;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_core::{JouleCost, Query, QueryId, Response, Stage as StageKind};
use lru::LruCache as InnerLru;

/// The stage trait — implemented by every level of the cascade.
///
/// Returning `Some(Response)` means "this stage answered". `None` means
/// "fall through to the next stage".
#[async_trait]
pub trait Stage: Send + Sync {
    /// Try to resolve `q` at this stage.
    async fn try_resolve(&self, q: &Query) -> Option<Response>;
}

/// In-process LRU cache stage.
pub struct LruCache {
    inner: Mutex<InnerLru<QueryId, Response>>,
    capacity: usize,
}

impl LruCache {
    /// Construct an LRU with a given capacity (entries).
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).expect("max(1) is non-zero");
        Self {
            inner: Mutex::new(InnerLru::new(cap)),
            capacity,
        }
    }

    /// Maximum number of entries.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of entries currently held.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("cache lock poisoned").len()
    }

    /// Is the cache empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert (or refresh) a `Response`. Re-stamps the response as `Cache`
    /// stage with zero joule cost so subsequent hits look like hits.
    pub fn insert(&self, q: &Query, payload: impl Into<String>) -> Response {
        let r = Response::new(q.id, payload.into(), StageKind::Cache, JouleCost::zero());
        self.inner
            .lock()
            .expect("cache lock poisoned")
            .put(q.id, r.clone());
        r
    }

    /// Insert a fully formed response under the given query id (used by the
    /// cascade to memoize results produced by deeper stages).
    pub fn insert_response(&self, query_id: QueryId, response: Response) {
        self.inner
            .lock()
            .expect("cache lock poisoned")
            .put(query_id, response);
    }
}

#[async_trait]
impl Stage for LruCache {
    async fn try_resolve(&self, q: &Query) -> Option<Response> {
        self.inner
            .lock()
            .expect("cache lock poisoned")
            .get(&q.id)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eoc_core::JouleSource;

    #[tokio::test]
    async fn lru_hits_are_nearly_free() {
        let cache = LruCache::new(8);
        let q = Query::new("hello");
        assert!(cache.try_resolve(&q).await.is_none());
        cache.insert(&q, "hi");
        let r = cache.try_resolve(&q).await.expect("hit");
        assert_eq!(r.stage, StageKind::Cache);
        assert_eq!(r.joule_cost.microjoules, 0);
        assert_eq!(r.joule_cost.source, JouleSource::Measured);
    }

    #[tokio::test]
    async fn lru_evicts_at_capacity() {
        let cache = LruCache::new(100);
        // Fill 100 entries.
        for i in 0..100 {
            let q = Query::new(format!("q{i}"));
            cache.insert(&q, format!("a{i}"));
        }
        assert_eq!(cache.len(), 100);
        // Insert 50 more — first 50 should be evicted (LRU order).
        for i in 100..150 {
            let q = Query::new(format!("q{i}"));
            cache.insert(&q, format!("a{i}"));
        }
        assert_eq!(cache.len(), 100);
        // q0 should be gone.
        let q0 = Query::new("q0");
        assert!(cache.try_resolve(&q0).await.is_none());
        // q149 should be present.
        let q149 = Query::new("q149");
        assert!(cache.try_resolve(&q149).await.is_some());
    }
}
