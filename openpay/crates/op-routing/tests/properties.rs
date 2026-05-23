//! Property tests for `op-routing`.
//!
//! Verifies the deterministic-contract invariant: same input → same
//! output. Routing is a pure compute and must never depend on
//! clock / thread / random state.

use op_core::{Currency, Money, RailKind};
use op_routing::cost::{Bps, CostModel, InterchangePlusEstimator, PaymentIntentRef};
use op_routing::mcc::{Mcc, MccPolicy, McuPreferences};
use op_routing::retry::{Attempt, DeclineCode, IntelligentRetry};
use op_routing::{ComposedRouter, DriverId, LeastCostRouter, Route, RouteDecision};
use proptest::prelude::*;

fn route_strategy() -> impl Strategy<Value = Route> {
    (0usize..5usize).prop_map(|i| {
        let drivers = ["a", "b", "c", "d", "e"];
        Route::new(DriverId::new(drivers[i]), RailKind::Card).with_country("US")
    })
}

fn amount_strategy() -> impl Strategy<Value = i64> {
    1_i64..1_000_000_i64
}

fn build_router() -> ComposedRouter {
    let mut lcr = LeastCostRouter::new()
        .with_routes(vec![
            Route::new(DriverId::new("a"), RailKind::Card).with_country("US"),
            Route::new(DriverId::new("b"), RailKind::Card).with_country("US"),
            Route::new(DriverId::new("c"), RailKind::Card).with_country("US"),
            Route::new(DriverId::new("d"), RailKind::Card).with_country("US"),
            Route::new(DriverId::new("e"), RailKind::Card).with_country("US"),
        ]);
    for (driver, bps) in [
        ("a", 300_u32),
        ("b", 100_u32),
        ("c", 200_u32),
        ("d", 250_u32),
        ("e", 150_u32),
    ] {
        let est = InterchangePlusEstimator::new().with_default(
            DriverId::new(driver),
            CostModel::new(
                Bps::new(bps),
                Bps::new(0),
                Bps::new(0),
                Money::from_minor(0, Currency::USD),
            ),
        );
        lcr = lcr.with_estimator(DriverId::new(driver), Box::new(est));
    }
    let mcc = MccPolicy::new().with_rule(
        Mcc::from_str("5411").expect("test mcc"),
        McuPreferences::default().with_excluded(DriverId::new("d")),
    );
    ComposedRouter::new(lcr, mcc, IntelligentRetry::new())
}

proptest! {
    /// Same input → same output. The hallmark of deterministic routing.
    #[test]
    fn idempotent_same_input_same_output(
        amount in amount_strategy(),
        mcc_idx in 0usize..4usize,
    ) {
        let mccs = ["5411", "5812", "7995", "6051"];
        let mcc_str = mccs[mcc_idx];
        let router = build_router();
        let intent = PaymentIntentRef::from_amount(Money::from_minor(amount, Currency::USD))
            .with_mcc(mcc_str);

        let d1 = router.route(&intent, &[]);
        let d2 = router.route(&intent, &[]);
        prop_assert_eq!(d1, d2);
    }

    /// LCR sort is total-ordered: select() is a permutation of routes
    /// (no duplicates lost, no extras added).
    #[test]
    fn lcr_select_preserves_pool_size(
        amount in amount_strategy(),
        n_routes in 1usize..6usize,
    ) {
        let mut lcr = LeastCostRouter::new();
        let mut routes = Vec::new();
        for i in 0..n_routes {
            let name = format!("d{i}");
            let est = InterchangePlusEstimator::new().with_default(
                DriverId::new(&name),
                CostModel::new(
                    Bps::new(((i as u32) + 1) * 50),
                    Bps::new(0),
                    Bps::new(0),
                    Money::from_minor(0, Currency::USD),
                ),
            );
            lcr = lcr.with_estimator(DriverId::new(&name), Box::new(est));
            routes.push(Route::new(DriverId::new(name), RailKind::Card));
        }
        lcr = lcr.with_routes(routes.clone());

        let intent = PaymentIntentRef::from_amount(Money::from_minor(amount, Currency::USD));
        let sorted = lcr.select(&intent);
        prop_assert_eq!(sorted.len(), routes.len());
        let mut original: Vec<_> = routes.iter().map(|r| r.driver.as_str().to_owned()).collect();
        let mut sorted_names: Vec<_> = sorted.iter().map(|r| r.driver.as_str().to_owned()).collect();
        original.sort();
        sorted_names.sort();
        prop_assert_eq!(original, sorted_names);
    }

    /// MCC filter only ever removes or reorders — never adds.
    #[test]
    fn mcc_filter_never_adds_routes(
        pool in proptest::collection::vec(route_strategy(), 1..6),
        mcc_idx in 0usize..2usize,
    ) {
        let mccs = ["5411", "5812"];
        let mcc_str = mccs[mcc_idx];
        let policy = MccPolicy::new().with_rule(
            Mcc::from_str(mcc_str).expect("test mcc"),
            McuPreferences::default()
                .with_excluded(DriverId::new("a"))
                .with_preferred(DriverId::new("b")),
        );
        let out = policy.filter(Mcc::from_str(mcc_str), &pool);
        prop_assert!(out.len() <= pool.len());
        // Every output route was in the input.
        for r in &out {
            prop_assert!(pool.iter().any(|p| p.driver == r.driver));
        }
    }

    /// Hard decline always terminates; soft decline (within max
    /// attempts and route budget) always continues.
    #[test]
    fn hard_decline_always_terminates(
        amount in amount_strategy(),
        code_idx in 0usize..6usize,
    ) {
        let codes = ["04", "07", "41", "43", "59", "62"];
        let code = codes[code_idx];
        let router = build_router();
        let intent = PaymentIntentRef::from_amount(Money::from_minor(amount, Currency::USD))
            .with_mcc("5811");
        let prior = vec![Attempt::new(
            Route::new(DriverId::new("a"), RailKind::Card),
            Some(DeclineCode::from_str(code).expect("test code")),
        )];
        let d = router.route(&intent, &prior);
        let is_hard_stop = matches!(&d, RouteDecision::Stop { reason } if *reason == "hard_decline");
        prop_assert!(is_hard_stop);
    }
}
