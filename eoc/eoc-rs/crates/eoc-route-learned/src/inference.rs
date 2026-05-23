//! Cascade extension that consults a learned router before dispatching.
//!
//! [`LearnedCascade`] wraps the existing [`eoc_cascade::Cascade`]. For each
//! query:
//!
//! 1. Ask the router for a [`StagePrediction`].
//! 2. Apply the [`ThresholdPolicy`] to get a [`ThresholdDecision`].
//! 3. If `FullCascade`, dispatch the regular cascade.
//! 4. If `SkipTo(stage)`, dispatch only the recommended stage (and its
//!    fallbacks for downstream stages, which keeps correctness when the
//!    router is wrong but pays cheap-stage cost when it's right).
//!
//! Skipping is the joule win: a 95%-confident "neural" prediction lets us
//! avoid the cache+kv+graph walk that would have missed anyway.

use std::sync::Arc;

use eoc_cascade::Cascade;
use eoc_core::{JouleCost, Query, Response, Stage};

// `try_resolve` is on `eoc_cache::Stage`, implemented by every stage crate.
use eoc_cache::Stage as _StageTrait;

use crate::router::LearnedRouter;
use crate::threshold::{ThresholdDecision, ThresholdPolicy};

/// Cascade + learned router + threshold policy.
pub struct LearnedCascade<R: LearnedRouter> {
    /// Underlying four-stage cascade.
    pub inner: Arc<Cascade>,
    /// The learned router.
    pub router: R,
    /// Skip-decision policy.
    pub policy: ThresholdPolicy,
}

impl<R: LearnedRouter> LearnedCascade<R> {
    /// Build a learned cascade.
    pub fn new(inner: Arc<Cascade>, router: R, policy: ThresholdPolicy) -> Self {
        Self {
            inner,
            router,
            policy,
        }
    }

    /// Resolve a query through router → policy → cascade.
    ///
    /// When the policy says `SkipTo(stage)` we dispatch exactly that stage.
    /// If the stage misses (only possible for Cache/Kv/Graph), we fall
    /// through to Neural — never run the cheaper stages we just decided to
    /// skip.
    pub async fn resolve(&self, q: Query) -> Response {
        let prediction = self.router.route(&q).await;
        let decision = self.policy.decide(&prediction);
        match decision {
            ThresholdDecision::FullCascade => self.inner.resolve(q).await,
            ThresholdDecision::SkipTo(stage) => self.dispatch_skipped(q, stage).await,
        }
    }

    async fn dispatch_skipped(&self, q: Query, stage: Stage) -> Response {
        let cascade = self.inner.as_ref();
        match stage {
            Stage::Cache => cascade.resolve(q).await,
            Stage::Kv => match cascade.kv().try_resolve(&q).await {
                Some(r) => r,
                None => self.fallback_to_neural(q).await,
            },
            Stage::Graph => match cascade.graph().try_resolve(&q).await {
                Some(r) => r,
                None => self.fallback_to_neural(q).await,
            },
            Stage::Neural => self
                .inner
                .neural()
                .try_resolve(&q)
                .await
                .unwrap_or_else(|| Response::new(q.id, String::new(), Stage::Neural, JouleCost::zero())),
        }
    }

    async fn fallback_to_neural(&self, q: Query) -> Response {
        self.inner
            .neural()
            .try_resolve(&q)
            .await
            .unwrap_or_else(|| Response::new(q.id, String::new(), Stage::Neural, JouleCost::zero()))
    }
}
