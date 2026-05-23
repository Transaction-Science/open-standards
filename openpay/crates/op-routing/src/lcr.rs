//! Least-cost routing.
//!
//! Given a `PaymentIntent` and a pool of candidate `Route`s plus a
//! [`CostEstimator`] per driver, sort the pool ascending by
//! estimated landed cost and return it. Routes with no estimator
//! registered land at the end (estimated cost `i64::MAX`).
//!
//! The router optionally biases higher-auth-rate routes via the
//! `auth_rate_bias_bps` knob. The idea: a route that estimates 10
//! bp more expensive but converts 200 bp better is net cheaper
//! per dollar successfully captured. The bias subtracts
//! `bias_bps * auth_rate_bps / 10_000` from the route's cost
//! score; tuning is operator territory. Set to zero to disable.

use op_core::Money;
use std::collections::HashMap;

use crate::cost::{CostEstimator, PaymentIntentRef};
use crate::route::{DriverId, Route};

/// LCR engine. Holds the per-driver cost estimators and the
/// candidate route pool.
#[derive(Default)]
pub struct LeastCostRouter {
    /// One estimator per driver. A route whose driver isn't in this
    /// map is treated as cost `i64::MAX` and sorts to the end.
    pub estimators: HashMap<DriverId, Box<dyn CostEstimator>>,

    /// Candidate routes. Operators populate this from their config.
    pub routes: Vec<Route>,

    /// Auth-rate bias in basis points. `0` disables the feature;
    /// higher values move higher-auth-rate routes earlier in the
    /// ranking. Tuning is operator-driven.
    pub auth_rate_bias_bps: u32,
}

impl core::fmt::Debug for LeastCostRouter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LeastCostRouter")
            .field("estimator_count", &self.estimators.len())
            .field("route_count", &self.routes.len())
            .field("auth_rate_bias_bps", &self.auth_rate_bias_bps)
            .finish()
    }
}

impl LeastCostRouter {
    /// Empty router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an estimator for a driver.
    #[must_use]
    pub fn with_estimator(mut self, driver: DriverId, est: Box<dyn CostEstimator>) -> Self {
        self.estimators.insert(driver, est);
        self
    }

    /// Set the candidate route pool.
    #[must_use]
    pub fn with_routes(mut self, routes: Vec<Route>) -> Self {
        self.routes = routes;
        self
    }

    /// Set the auth-rate bias in basis points.
    #[must_use]
    pub const fn with_auth_rate_bias_bps(mut self, bps: u32) -> Self {
        self.auth_rate_bias_bps = bps;
        self
    }

    /// Score a single route (cost minus auth-rate bonus). Returned
    /// as `i64` minor units; lower is better.
    fn score(&self, intent: &PaymentIntentRef<'_>, route: &Route) -> i64 {
        let cost = self.estimators.get(&route.driver).map_or_else(
            || Money::from_minor(i64::MAX, intent.amount.currency),
            |e| e.estimate(intent, route),
        );

        if self.auth_rate_bias_bps == 0 || cost.minor_units == i64::MAX {
            return cost.minor_units;
        }

        // bonus = principal_minor * bias_bps * auth_rate_bps / 100_000_000
        // (auth_rate_bps is itself in bp of 10_000, so the divisor
        // is 10_000 * 10_000)
        let auth_bps = u32::from(route.auth_rate_bps.unwrap_or(5_000));
        let bonus_i128 = i128::from(intent.amount.minor_units)
            .saturating_mul(i128::from(self.auth_rate_bias_bps))
            .saturating_mul(i128::from(auth_bps))
            / 100_000_000_i128;
        let bonus = i64::try_from(bonus_i128).unwrap_or(i64::MAX);
        cost.minor_units.saturating_sub(bonus)
    }

    /// Return the candidate pool sorted ascending by score.
    ///
    /// Ties broken by the original order in `self.routes` (stable
    /// sort).
    #[must_use]
    pub fn select(&self, intent: &PaymentIntentRef<'_>) -> Vec<Route> {
        let mut indexed: Vec<(usize, &Route)> = self.routes.iter().enumerate().collect();
        indexed.sort_by(|(ia, a), (ib, b)| {
            let sa = self.score(intent, a);
            let sb = self.score(intent, b);
            sa.cmp(&sb).then(ia.cmp(ib))
        });
        indexed.into_iter().map(|(_, r)| r.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{BlendedRateEstimator, Bps, CostModel, InterchangePlusEstimator};
    use op_core::{Currency, Money, RailKind};

    fn intent(amount_minor: i64) -> PaymentIntentRef<'static> {
        PaymentIntentRef::from_amount(Money::from_minor(amount_minor, Currency::USD))
    }

    fn route(name: &str, auth_bps: u16) -> Route {
        Route::new(DriverId::new(name), RailKind::Card)
            .with_country("US")
            .with_auth_rate_bps(auth_bps)
    }

    #[test]
    fn cheaper_route_wins() {
        let cheap = InterchangePlusEstimator::new().with_default(
            DriverId::new("cheap"),
            CostModel::new(
                Bps::new(100),
                Bps::new(0),
                Bps::new(0),
                Money::from_minor(0, Currency::USD),
            ),
        );
        let pricey = InterchangePlusEstimator::new().with_default(
            DriverId::new("pricey"),
            CostModel::new(
                Bps::new(290),
                Bps::new(0),
                Bps::new(0),
                Money::from_minor(0, Currency::USD),
            ),
        );

        let router = LeastCostRouter::new()
            .with_estimator(DriverId::new("cheap"), Box::new(cheap))
            .with_estimator(DriverId::new("pricey"), Box::new(pricey))
            .with_routes(vec![route("pricey", 9_500), route("cheap", 9_500)]);

        let ordered = router.select(&intent(10_000));
        assert_eq!(ordered[0].driver.as_str(), "cheap");
        assert_eq!(ordered[1].driver.as_str(), "pricey");
    }

    #[test]
    fn driver_with_no_estimator_sorts_to_end() {
        let est = InterchangePlusEstimator::new().with_default(
            DriverId::new("a"),
            CostModel::new(
                Bps::new(200),
                Bps::new(0),
                Bps::new(0),
                Money::from_minor(0, Currency::USD),
            ),
        );
        let router = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(est))
            .with_routes(vec![route("ghost", 9_500), route("a", 9_500)]);

        let ordered = router.select(&intent(10_000));
        assert_eq!(ordered[0].driver.as_str(), "a");
        assert_eq!(ordered[1].driver.as_str(), "ghost");
    }

    #[test]
    fn auth_rate_bias_can_overturn_pure_cost_ranking() {
        // pricey costs 290 bp on $1000 = 2900 minor
        // cheap   costs 100 bp on $1000 = 1000 minor
        // pricey converts 9_900 bp (99.0%); cheap converts 5_000 bp (50.0%).
        // bias = 100 bp.
        // pricey bonus = 100_000 * 100 * 9_900 / 100_000_000 = 990 → score 2900-990 = 1910
        // cheap bonus  = 100_000 * 100 * 5_000 / 100_000_000 = 500 → score 1000-500 = 500
        // cheap still wins by score, just by less margin. Bump pricey's
        // auth rate AND bias dramatically and we can flip the ranking.
        let cheap = BlendedRateEstimator::new().with(
            DriverId::new("cheap"),
            CostModel::new(
                Bps::new(0),
                Bps::new(0),
                Bps::new(290),
                Money::from_minor(0, Currency::USD),
            ),
        );
        let premium = BlendedRateEstimator::new().with(
            DriverId::new("premium"),
            CostModel::new(
                Bps::new(0),
                Bps::new(0),
                Bps::new(300), // basically the same cost
                Money::from_minor(0, Currency::USD),
            ),
        );

        // Without bias, cheap wins by 10bp.
        let no_bias = LeastCostRouter::new()
            .with_estimator(DriverId::new("cheap"), Box::new(cheap.clone()))
            .with_estimator(DriverId::new("premium"), Box::new(premium.clone()))
            .with_routes(vec![route("premium", 9_900), route("cheap", 5_000)]);
        assert_eq!(no_bias.select(&intent(10_000))[0].driver.as_str(), "cheap");

        // With aggressive bias, premium's better auth rate wins.
        let with_bias = LeastCostRouter::new()
            .with_estimator(DriverId::new("cheap"), Box::new(cheap))
            .with_estimator(DriverId::new("premium"), Box::new(premium))
            .with_routes(vec![route("premium", 9_900), route("cheap", 5_000)])
            .with_auth_rate_bias_bps(500);
        assert_eq!(
            with_bias.select(&intent(10_000))[0].driver.as_str(),
            "premium",
        );
    }

    #[test]
    fn stable_under_equal_cost() {
        let est = BlendedRateEstimator::new()
            .with(
                DriverId::new("a"),
                CostModel::new(
                    Bps::new(0),
                    Bps::new(0),
                    Bps::new(100),
                    Money::from_minor(0, Currency::USD),
                ),
            )
            .with(
                DriverId::new("b"),
                CostModel::new(
                    Bps::new(0),
                    Bps::new(0),
                    Bps::new(100),
                    Money::from_minor(0, Currency::USD),
                ),
            );

        let router = LeastCostRouter::new()
            .with_estimator(DriverId::new("a"), Box::new(est.clone()))
            .with_estimator(DriverId::new("b"), Box::new(est))
            .with_routes(vec![route("a", 9_000), route("b", 9_000)]);

        let ordered = router.select(&intent(10_000));
        // Equal cost and equal bonus → original order preserved.
        assert_eq!(ordered[0].driver.as_str(), "a");
        assert_eq!(ordered[1].driver.as_str(), "b");
    }

    #[test]
    fn empty_pool_returns_empty() {
        let router = LeastCostRouter::new();
        assert!(router.select(&intent(10_000)).is_empty());
    }
}
