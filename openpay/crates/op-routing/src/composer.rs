//! Composed router: MCC → LCR → retry, in that order.
//!
//! The integration. Given a payment intent and the prior attempts so
//! far, [`ComposedRouter::route`] returns a [`RouteDecision`] that
//! the orchestrator can dispatch.
//!
//! Pipeline:
//!
//! 1. **MCC filter**: drop driver exclusions for the intent's MCC;
//!    move preferred drivers to the front of the pool.
//! 2. **LCR sort**: sort the surviving pool by estimated landed
//!    cost (with optional auth-rate bias).
//! 3. **Retry policy**: if `prior_attempts` is non-empty, classify
//!    the last decline. Hard → terminate. Soft (or first attempt) →
//!    pick the first not-yet-tried route from the sorted pool.
//!
//! The decision carries the chosen route plus the backoff delay the
//! caller must observe before issuing the attempt. On first attempt
//! the delay is `Duration::ZERO`.

use std::time::Duration;

use crate::cost::PaymentIntentRef;
use crate::lcr::LeastCostRouter;
use crate::mcc::{Mcc, MccPolicy};
use crate::retry::{Attempt, DeclineCategory, IntelligentRetry};
use crate::route::Route;

/// The output of [`ComposedRouter::route`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteDecision {
    /// Try this route after waiting `delay`.
    Attempt {
        /// The route to try.
        route: Route,
        /// How long to wait before issuing this attempt. `ZERO` on
        /// first attempt; exponential-plus-jitter on retries.
        delay: Duration,
    },
    /// Stop. No further attempts. `reason` is operator-readable.
    Stop {
        /// Human-readable explanation.
        reason: &'static str,
    },
}

/// The composed router.
pub struct ComposedRouter {
    /// Least-cost-routing engine.
    pub lcr: LeastCostRouter,
    /// MCC policy.
    pub mcc: MccPolicy,
    /// Intelligent retry policy.
    pub retry: IntelligentRetry,
}

impl core::fmt::Debug for ComposedRouter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ComposedRouter")
            .field("lcr", &self.lcr)
            .field("mcc", &self.mcc)
            .field("retry", &self.retry)
            .finish()
    }
}

impl ComposedRouter {
    /// Construct.
    #[must_use]
    pub const fn new(lcr: LeastCostRouter, mcc: MccPolicy, retry: IntelligentRetry) -> Self {
        Self { lcr, mcc, retry }
    }

    /// Compute the next routing decision.
    ///
    /// `prior_attempts` is empty on first call. On retries the
    /// caller appends each completed attempt.
    #[must_use]
    pub fn route(
        &self,
        intent: &PaymentIntentRef<'_>,
        prior_attempts: &[Attempt],
    ) -> RouteDecision {
        // Hard-decline short-circuit before any work.
        if let Some(last) = prior_attempts.last()
            && let Some(code) = &last.decline_code
            && self.retry.classify(code) == DeclineCategory::Hard
        {
            return RouteDecision::Stop {
                reason: "hard_decline",
            };
        }

        // Effective max_attempts: min(retry default, MCC override).
        let mcc = intent.mcc.and_then(Mcc::from_str);
        let effective_max = self.mcc.max_attempts_for(mcc, self.retry.max_attempts);
        let attempts_used =
            u8::try_from(prior_attempts.len()).unwrap_or(u8::MAX);
        if attempts_used >= effective_max {
            return RouteDecision::Stop {
                reason: "max_attempts_reached",
            };
        }

        // 1. LCR-sort the operator's full pool by estimated cost.
        let lcr_sorted = self.lcr.select(intent);

        // 2. MCC-filter / re-prioritize.
        let filtered = self.mcc.filter(mcc, &lcr_sorted);

        // 3. Pick the first not-yet-tried route.
        let tried: std::collections::HashSet<&crate::route::DriverId> =
            prior_attempts.iter().map(|a| &a.route.driver).collect();
        let Some(chosen) = filtered.iter().find(|r| !tried.contains(&r.driver)).cloned() else {
            return RouteDecision::Stop {
                reason: "no_routes_remaining",
            };
        };

        // 4. Compute backoff delay (zero on first attempt).
        let attempt_index = u32::try_from(prior_attempts.len()).unwrap_or(u32::MAX);
        let delay = self.retry.backoff.delay_for(attempt_index);

        RouteDecision::Attempt {
            route: chosen,
            delay,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{Bps, CostModel, InterchangePlusEstimator};
    use crate::mcc::{Mcc, McuPreferences};
    use crate::retry::{BackoffPolicy, DeclineCode};
    use crate::route::DriverId;
    use op_core::{Currency, Money, RailKind};

    fn intent_usd<'a>(minor: i64, mcc: &'a str) -> PaymentIntentRef<'a> {
        PaymentIntentRef::from_amount(Money::from_minor(minor, Currency::USD)).with_mcc(mcc)
    }

    fn route(driver: &str) -> Route {
        Route::new(DriverId::new(driver), RailKind::Card).with_country("US")
    }

    fn dc(s: &str) -> DeclineCode {
        DeclineCode::from_str(s).expect("test decline code")
    }

    fn estimator_at(driver: &str, bps: u32) -> InterchangePlusEstimator {
        InterchangePlusEstimator::new().with_default(
            DriverId::new(driver),
            CostModel::new(
                Bps::new(bps),
                Bps::new(0),
                Bps::new(0),
                Money::from_minor(0, Currency::USD),
            ),
        )
    }

    #[test]
    fn end_to_end_first_attempt_picks_mcc_preferred_lcr_cheap() {
        // Three routes: a (expensive), b (cheapest), c (mid).
        // MCC rule (5411 grocery): prefer "a"; exclude "c".
        // Expected first attempt: "a" (MCC preference wins; c removed).
        let lcr = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(estimator_at("a", 300)))
            .with_estimator(DriverId::new("b"), Box::new(estimator_at("b", 100)))
            .with_estimator(DriverId::new("c"), Box::new(estimator_at("c", 200)))
            .with_routes(vec![route("a"), route("b"), route("c")]);

        let mcc = MccPolicy::new().with_rule(
            Mcc::from_str("5411").expect("test mcc"),
            McuPreferences::default()
                .with_preferred(DriverId::new("a"))
                .with_excluded(DriverId::new("c")),
        );

        let router = ComposedRouter::new(lcr, mcc, IntelligentRetry::new());
        let decision = router.route(&intent_usd(10_000, "5411"), &[]);

        let RouteDecision::Attempt { route, delay } = decision else {
            panic!("expected Attempt, got {decision:?}");
        };
        assert_eq!(route.driver.as_str(), "a");
        assert_eq!(delay, Duration::ZERO, "first attempt has zero delay");
    }

    #[test]
    fn soft_decline_retry_walks_to_next_cheapest_unexcluded() {
        let lcr = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(estimator_at("a", 300)))
            .with_estimator(DriverId::new("b"), Box::new(estimator_at("b", 100)))
            .with_estimator(DriverId::new("c"), Box::new(estimator_at("c", 200)))
            .with_routes(vec![route("a"), route("b"), route("c")]);

        // No MCC preferences — straight LCR. Sorted: b, c, a.
        let mcc = MccPolicy::new();
        let backoff = BackoffPolicy {
            initial: Duration::from_millis(100),
            cap: Duration::from_secs(5),
            jitter_max: Duration::ZERO,
            jitter_seed: 0,
        };
        let retry = IntelligentRetry::new().with_backoff(backoff);
        let router = ComposedRouter::new(lcr, mcc, retry);

        // First attempt went to "b" and got a soft decline (05).
        let prior = vec![Attempt::new(route("b"), Some(dc("05")))];
        let decision = router.route(&intent_usd(10_000, "5411"), &prior);
        let RouteDecision::Attempt { route, delay } = decision else {
            panic!("expected Attempt");
        };
        // Next-cheapest untried route is "c".
        assert_eq!(route.driver.as_str(), "c");
        assert_eq!(delay, Duration::from_millis(100));
    }

    #[test]
    fn hard_decline_stops_immediately() {
        let lcr = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(estimator_at("a", 300)))
            .with_estimator(DriverId::new("b"), Box::new(estimator_at("b", 100)))
            .with_routes(vec![route("a"), route("b")]);
        let router = ComposedRouter::new(lcr, MccPolicy::new(), IntelligentRetry::new());

        let prior = vec![Attempt::new(route("a"), Some(dc("43")))]; // stolen
        let decision = router.route(&intent_usd(10_000, "5411"), &prior);
        let RouteDecision::Stop { reason } = decision else {
            panic!("expected Stop");
        };
        assert_eq!(reason, "hard_decline");
    }

    #[test]
    fn pool_exhausted_returns_stop() {
        let lcr = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(estimator_at("a", 300)))
            .with_routes(vec![route("a")]);
        let router = ComposedRouter::new(lcr, MccPolicy::new(), IntelligentRetry::new());

        let prior = vec![Attempt::new(route("a"), Some(dc("05")))];
        let decision = router.route(&intent_usd(10_000, "5411"), &prior);
        let RouteDecision::Stop { reason } = decision else {
            panic!("expected Stop");
        };
        assert_eq!(reason, "no_routes_remaining");
    }

    #[test]
    fn mcc_max_attempts_caps_below_global_default() {
        // Global default = 4. MCC 7995 (gambling) caps at 2.
        let lcr = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(estimator_at("a", 300)))
            .with_estimator(DriverId::new("b"), Box::new(estimator_at("b", 200)))
            .with_estimator(DriverId::new("c"), Box::new(estimator_at("c", 100)))
            .with_routes(vec![route("a"), route("b"), route("c")]);

        let mcc = MccPolicy::new().with_rule(
            Mcc::from_str("7995").expect("test mcc"),
            McuPreferences::default().with_max_attempts(2),
        );
        let router = ComposedRouter::new(lcr, mcc, IntelligentRetry::new());

        let prior = vec![
            Attempt::new(route("c"), Some(dc("05"))),
            Attempt::new(route("b"), Some(dc("05"))),
        ];
        let decision = router.route(&intent_usd(10_000, "7995"), &prior);
        let RouteDecision::Stop { reason } = decision else {
            panic!("expected Stop");
        };
        assert_eq!(reason, "max_attempts_reached");
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let lcr = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(estimator_at("a", 300)))
            .with_estimator(DriverId::new("b"), Box::new(estimator_at("b", 100)))
            .with_routes(vec![route("a"), route("b")]);
        let router = ComposedRouter::new(lcr, MccPolicy::new(), IntelligentRetry::new());

        let intent = intent_usd(10_000, "5411");
        let d1 = router.route(&intent, &[]);
        let d2 = router.route(&intent, &[]);
        assert_eq!(d1, d2);
    }
}
