//! Graph-informed routing end-to-end.
//!
//! Closes the loop:
//!   Orchestrator runs attempts
//!     → GraphRailTelemetry records each (rail, driver, outcome)
//!     → next intent's router queries the same graph
//!     → router re-orders the fallback chain to prefer quieter drivers.

use std::sync::Arc;

use op_core::{Currency, Money, PaymentMethod, RailKind, VaultRef};
use op_orchestrator::engine::{AdapterResult, RailAdapter};
use op_orchestrator::{
    Attempt, AttemptOutcome, IdempotencyKey, Orchestrator, PaymentIntent, PolicyRouter,
    RailTelemetry, RoutingSignals,
};

use op_graph::{GraphHandle, GraphRailTelemetry};

const DRIVER_NOISY: &str = "noisy_psp";
const DRIVER_QUIET: &str = "quiet_psp";

/// Adapter that always returns the given outcome — keeps the test
/// fully deterministic so the only variable is the chain order.
#[derive(Clone)]
struct FixedAdapter {
    driver: &'static str,
    outcome: AttemptOutcome,
}

impl RailAdapter for FixedAdapter {
    fn rail(&self) -> RailKind {
        RailKind::Card
    }
    fn driver(&self) -> &str {
        self.driver
    }
    fn attempt(&self, _: &PaymentIntent, _: usize) -> AdapterResult {
        AdapterResult {
            outcome: self.outcome.clone(),
            psp_payment_id: Some(format!("{}_psp_1", self.driver)),
            uetr: None,
        }
    }
}

fn vault_intent(key: &str) -> PaymentIntent {
    PaymentIntent::new(
        IdempotencyKey::new(key),
        Money::from_minor(2_000, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok_v7_test")),
    )
}

#[test]
fn router_reorders_to_prefer_quiet_driver_after_failure_signal_is_recorded() {
    // Single shared graph: telemetry writes here; router reads from
    // here. Same handle = same data path.
    let handle = GraphHandle::new_in_memory();
    let telemetry: Arc<GraphRailTelemetry> =
        Arc::new(GraphRailTelemetry::with_handle(handle.clone()));

    // Router lists noisy_psp FIRST in its static preference order —
    // without signals, it would always try the noisy driver before
    // the quiet one.
    let router = PolicyRouter::new(
        vec![DRIVER_NOISY.to_owned(), DRIVER_QUIET.to_owned()],
        vec![],
    )
    .with_signals(telemetry.clone() as Arc<dyn RoutingSignals>);

    let orch = Orchestrator::new()
        .with_router(Box::new(router))
        .with_telemetry(telemetry.clone() as Arc<dyn RailTelemetry>);

    let noisy_adapter = Arc::new(FixedAdapter {
        driver: DRIVER_NOISY,
        outcome: AttemptOutcome::SoftFailure {
            code: "transport".into(),
        },
    });
    let quiet_adapter = Arc::new(FixedAdapter {
        driver: DRIVER_QUIET,
        outcome: AttemptOutcome::Success,
    });

    let mut orch = orch;
    orch.register_adapter(noisy_adapter);
    orch.register_adapter(quiet_adapter);

    // First run: static order is [noisy, quiet]. Telemetry has no
    // history so failure_score is 0.0 for both — order unchanged.
    // Orchestrator tries noisy (soft-failure), falls back to quiet
    // (success). Two attempts.
    let first = orch.run(&vault_intent("intent-1")).unwrap();
    assert_eq!(first.attempts.len(), 2);
    assert_attempt_order(&first.attempts, &[DRIVER_NOISY, DRIVER_QUIET]);

    // Telemetry should now hold one soft_failure for noisy and one
    // approved for quiet. Score is failures/total: noisy=1/1=1.0,
    // quiet=0/1=0.0.
    let noisy_score = telemetry.failure_score_at(RailKind::Card, DRIVER_NOISY, default_now() + 1);
    let quiet_score = telemetry.failure_score_at(RailKind::Card, DRIVER_QUIET, default_now() + 1);
    assert_eq!(noisy_score, 1.0);
    assert_eq!(quiet_score, 0.0);

    // Second run: signals report noisy as noisier than quiet, so the
    // router re-orders within the Card rail to [quiet, noisy].
    // First attempt hits quiet, which succeeds — orchestration done
    // in ONE attempt, not two.
    let second = orch.run(&vault_intent("intent-2")).unwrap();
    assert_eq!(
        second.attempts.len(),
        1,
        "expected the router to prefer quiet and succeed on first try, got attempts {:?}",
        second.attempts
    );
    assert_eq!(second.attempts[0].driver, DRIVER_QUIET);
}

#[test]
fn empty_history_leaves_static_chain_order_intact() {
    // A router with signals but no recorded attempts behaves the
    // same as a router with no signals — score == 0.0 for everyone
    // and stable sort preserves the operator-configured order.
    let handle = GraphHandle::new_in_memory();
    let telemetry: Arc<GraphRailTelemetry> = Arc::new(GraphRailTelemetry::with_handle(handle));

    let router = PolicyRouter::new(
        vec![DRIVER_NOISY.to_owned(), DRIVER_QUIET.to_owned()],
        vec![],
    )
    .with_signals(telemetry as Arc<dyn RoutingSignals>);

    let decision = op_orchestrator::router::Router::route(&router, &vault_intent("k")).unwrap();
    let drivers: Vec<&str> = decision.chain.iter().map(|c| c.driver.as_str()).collect();
    assert_eq!(drivers, vec![DRIVER_NOISY, DRIVER_QUIET]);
}

#[test]
fn telemetry_history_persists_across_handle_reopen() {
    // Bookkeeping continuity: a daemon restart shouldn't wipe the
    // "this PSP has been flaky" knowledge. Record under one handle,
    // drop, reopen, query — score still reports failure.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rail.graph");
    {
        let h = GraphHandle::new_persistent(&path).unwrap();
        let t = GraphRailTelemetry::with_handle(h);
        t.record_attempt(
            RailKind::Card,
            DRIVER_NOISY,
            op_orchestrator::AttemptResultClass::SoftFailure,
            1_000,
            None,
            None,
        );
        t.record_attempt(
            RailKind::Card,
            DRIVER_NOISY,
            op_orchestrator::AttemptResultClass::SoftFailure,
            1_100,
            None,
            None,
        );
    }
    let h2 = GraphHandle::new_persistent(&path).unwrap();
    let t2 = GraphRailTelemetry::with_handle(h2);
    let score = t2.failure_score_at(RailKind::Card, DRIVER_NOISY, 1_200);
    assert_eq!(score, 1.0);
}

// ============================================================
// Helpers
// ============================================================

fn assert_attempt_order(attempts: &[Attempt], expected_drivers: &[&str]) {
    let got: Vec<&str> = attempts.iter().map(|a| a.driver.as_str()).collect();
    assert_eq!(got, expected_drivers, "attempt order mismatch");
}

fn default_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
