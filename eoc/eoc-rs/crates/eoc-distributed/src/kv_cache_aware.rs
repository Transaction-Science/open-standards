//! KV-cache locality routing.
//!
//! When a worker already holds the prefix KV-cache for a session, sending
//! the *next* turn to the same worker avoids re-prefilling, which is
//! often 10-100× cheaper in joules than placing the work cold. This is
//! the trick behind vLLM's `PrefixCachingScheduler`, SGLang's RadixAttn,
//! and the "sticky session" mode in LiteLLM router.
//!
//! [`KvCacheAwareRouter`] is a thin index: `session_id -> worker_id`, with
//! a fallback to the joule-weighted router when the session is unknown
//! (cold start) or the previously-used worker is no longer alive.

use std::collections::HashMap;

use crate::error::{DistributedError, Result};
use crate::router::{Request, Router, Strategy};
use crate::worker::Worker;

/// Locality-aware request descriptor: like [`Request`] but with a
/// session id.
#[derive(Debug, Clone)]
pub struct LocalRequest<'a> {
    /// Caller-side session id (matches the previous turn's session).
    pub session_id: String,
    /// Model.
    pub model: &'a str,
    /// Expected new tokens for the cold-start fallback's joule maths.
    pub expected_tokens: u32,
}

/// KV-cache locality router.
#[derive(Debug)]
pub struct KvCacheAwareRouter {
    sticky: HashMap<String, String>,
    fallback: Router,
}

impl Default for KvCacheAwareRouter {
    fn default() -> Self {
        Self::new(Strategy::JouleWeighted)
    }
}

impl KvCacheAwareRouter {
    /// Construct with a fallback strategy for cold sessions.
    pub fn new(fallback: Strategy) -> Self {
        Self {
            sticky: HashMap::new(),
            fallback: Router::new(fallback),
        }
    }

    /// Borrow the fallback router.
    pub fn fallback(&self) -> &Router {
        &self.fallback
    }

    /// Pick a worker for `req`. If the session has been seen and the
    /// previously-used worker is still in `workers` and serves the
    /// model, returns it (cache hit). Otherwise falls back to
    /// joule-weighted placement and records the binding.
    pub fn pick<'w>(
        &mut self,
        workers: &'w [&'w dyn Worker],
        req: &LocalRequest<'_>,
    ) -> Result<&'w dyn Worker> {
        if workers.is_empty() {
            return Err(DistributedError::NoWorkers);
        }
        if let Some(prev) = self.sticky.get(&req.session_id) {
            if let Some(w) = workers
                .iter()
                .copied()
                .find(|w| w.id() == prev && w.capability().serves(req.model))
            {
                return Ok(w);
            }
            // Stale binding — drop it.
            self.sticky.remove(&req.session_id);
        }
        let pick = self.fallback.pick(
            workers,
            &Request {
                model: req.model,
                expected_tokens: req.expected_tokens,
            },
        )?;
        self.sticky
            .insert(req.session_id.clone(), pick.id().to_string());
        Ok(pick)
    }

    /// Forget the binding for `session_id`. Caller would invoke this on
    /// session end, KV eviction, or worker failure.
    pub fn forget(&mut self, session_id: &str) {
        self.sticky.remove(session_id);
    }

    /// Number of remembered sessions.
    pub fn len(&self) -> usize {
        self.sticky.len()
    }

    /// Whether there are no remembered sessions.
    pub fn is_empty(&self) -> bool {
        self.sticky.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{Accelerator, Capability, InMemoryWorker, Load};

    fn mk(id: &str, micro_j: u32) -> InMemoryWorker {
        InMemoryWorker::new(
            id,
            Capability {
                models: vec!["m".into()],
                accelerator: Accelerator::Gpu,
                max_concurrency: 8,
                continuous_batching: true,
                paged_kv: true,
                zone: "EU-FR".into(),
            },
            Load {
                micro_joules_per_token: micro_j,
                ..Load::idle()
            },
        )
    }

    #[test]
    fn sticky_after_first_call() {
        let a = mk("a", 200);
        let b = mk("b", 50);
        let pool: Vec<&dyn Worker> = vec![&a, &b];
        let mut r = KvCacheAwareRouter::default();
        let first = r
            .pick(
                &pool,
                &LocalRequest {
                    session_id: "s1".into(),
                    model: "m",
                    expected_tokens: 10,
                },
            )
            .expect("ok")
            .id()
            .to_string();
        // Even if we flip the load so the OTHER worker would now be
        // cheaper, the sticky binding must keep us on `first`.
        let cheaper = mk("c", 1);
        let pool2: Vec<&dyn Worker> = vec![&a, &b, &cheaper];
        let second = r
            .pick(
                &pool2,
                &LocalRequest {
                    session_id: "s1".into(),
                    model: "m",
                    expected_tokens: 10,
                },
            )
            .expect("ok")
            .id()
            .to_string();
        assert_eq!(first, second);
    }
}
