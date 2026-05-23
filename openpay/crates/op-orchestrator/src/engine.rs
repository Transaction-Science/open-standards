//! The orchestrator engine.
//!
//! Wires intent → fraud check → routing → attempt loop → outcome.
//!
//! ## Adapter-per-driver
//!
//! The card and A2A rail traits have very different request shapes
//! (`AuthRequest` carries 3DS preferences and PSP metadata;
//! `CreditTransferReq` carries SEPA / FedNow agent identifiers,
//! debtor/creditor account numbers, etc). Rather than bake all those
//! fields into the generic [`PaymentIntent`], we delegate
//! intent-to-rail-request translation to a per-driver
//! [`RailAdapter`].
//!
//! The orchestrator owns a `HashMap<(rail, driver), Arc<dyn
//! RailAdapter>>` and dispatches by name. Operators register their
//! adapters at startup.
//!
//! This trades a small amount of boilerplate per driver for a
//! coherent orchestrator that doesn't need to know about every rail
//! protocol. New rails (UPI, RTP, etc.) plug in by writing an
//! adapter — no changes to the engine.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use op_core::{PaymentMethod, RailKind};
use op_fraud::features::PaymentDescriptor;
use op_fraud::{
    FeatureVector, FraudDecision, Scorer, ScoringContext, Thresholds, extract_features,
};

use crate::circuit_breaker::{CircuitBreaker, InMemoryCircuitBreaker};
use crate::error::{Error, Result};
use crate::idempotency::{IdempotencyStore, InMemoryIdempotencyStore};
use crate::intent::PaymentIntent;
use crate::outcome::{Attempt, AttemptOutcome, OrchestrationOutcome, TerminalStatus};
use crate::router::{PolicyRouter, Router};

/// Per-driver adapter that maps a generic [`PaymentIntent`] onto its
/// underlying rail's request type, calls the driver, and maps the
/// response onto an [`AttemptOutcome`].
///
/// Implementations live downstream — `op-rails-card::adapter::HyperswitchAdapter`,
/// `op-rails-a2a::adapter::FedNowAdapter`, etc. The orchestrator only
/// sees this trait.
///
/// Returns a tuple `(AttemptOutcome, Option<rail_specific_id>)` so
/// the orchestrator can populate
/// [`OrchestrationOutcome::psp_payment_id`] /
/// [`OrchestrationOutcome::uetr`] on success.
pub trait RailAdapter: Send + Sync {
    /// Driver name. Must match what the router emits in
    /// [`RailChoice::driver`].
    fn driver(&self) -> &str;

    /// Which rail this adapter speaks (Card or A2A).
    fn rail(&self) -> RailKind;

    /// Attempt the payment. The orchestrator passes the intent and
    /// the attempt index (0-based) so adapters that want to vary
    /// their behavior across attempts (e.g. force 3DS on retry) can
    /// see the count.
    fn attempt(&self, intent: &PaymentIntent, attempt_number: usize) -> AdapterResult;

    /// Resume a previously-paused attempt after the customer
    /// completed an out-of-band challenge (3DS, bank app redirect,
    /// OTP). The orchestrator calls this when
    /// [`Orchestrator::resume`] is invoked with a `psp_payment_id`
    /// matching an earlier `RequiresAction` outcome.
    ///
    /// Default impl returns `SoftFailure { code:
    /// "resume_not_supported" }` so adapters that don't have a
    /// challenge flow stay correct without needing to implement
    /// anything.
    fn resume(&self, intent: &PaymentIntent, psp_payment_id: &str) -> AdapterResult {
        let _ = (intent, psp_payment_id);
        AdapterResult {
            outcome: AttemptOutcome::SoftFailure {
                code: "resume_not_supported".to_owned(),
            },
            psp_payment_id: None,
            uetr: None,
        }
    }
}

/// What a [`RailAdapter::attempt`] call returns.
#[derive(Clone, Debug)]
pub struct AdapterResult {
    /// Outcome of this attempt.
    pub outcome: AttemptOutcome,
    /// PSP-issued payment id, if the rail returned one (Card only).
    pub psp_payment_id: Option<String>,
    /// UETR echoed back by the rail, if any (A2A only).
    pub uetr: Option<String>,
}

/// Backoff policy between retries.
#[derive(Copy, Clone, Debug)]
pub struct BackoffPolicy {
    /// Initial delay (caller's notion of seconds — see note below).
    pub initial: u64,

    /// Multiplier applied per attempt.
    pub multiplier: u32,

    /// Cap on the per-attempt delay.
    pub max_delay: u64,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        // 1s, 2s, 4s, 8s, …, capped at 30s.
        Self {
            initial: 1,
            multiplier: 2,
            max_delay: 30,
        }
    }
}

impl BackoffPolicy {
    /// Compute the delay for attempt `n` (0-based).
    pub fn delay_for(&self, n: usize) -> u64 {
        let factor = (self.multiplier as u64).saturating_pow(n as u32);
        self.initial.saturating_mul(factor).min(self.max_delay)
    }
}

/// Engine configuration knobs.
#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    /// Max number of soft-failure retries before declaring exhaustion.
    /// Each rail entry in the routing chain counts toward this cap.
    pub max_attempts: usize,

    /// Backoff policy between attempts.
    pub backoff: BackoffPolicy,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: BackoffPolicy::default(),
        }
    }
}

/// Builder for the per-intent fraud scoring context.
///
/// Operators with real customer-velocity / device-fingerprint data
/// plug in their own. Default produces an empty context (no
/// velocity, no device info, no prior-payment history) — fine for
/// the reference flow but undertrained in production.
pub type ContextBuilder = Box<dyn Fn(&PaymentIntent) -> ScoringContext + Send + Sync>;

/// The orchestrator.
///
/// Construct one per process. Register rail adapters at startup;
/// `run()` an intent to get an [`OrchestrationOutcome`].
pub struct Orchestrator {
    config: OrchestratorConfig,
    router: Box<dyn Router>,
    adapters: HashMap<(RailKind, String), Arc<dyn RailAdapter>>,
    idempotency: Box<dyn IdempotencyStore>,
    breaker: Box<dyn CircuitBreaker>,
    scorer: Option<Box<dyn Scorer>>,
    /// Telemetry sink for completed attempts. Defaults to
    /// [`NoOpRailTelemetry`], so existing callers don't change.
    /// Wire in a real sink (e.g. `GraphRailTelemetry`) to feed the
    /// routing-signals loop.
    telemetry: Arc<dyn crate::signals::RailTelemetry>,
    /// Fraud thresholds (review / decline / freeze). Default per
    /// PCI / industry guidance: 0.50 / 0.80 / 0.95.
    thresholds: Thresholds,
    /// Builder for the per-intent fraud scoring context.
    context_builder: ContextBuilder,
    /// Time source (override in tests). Returns unix epoch seconds.
    now: Box<dyn Fn() -> u64 + Send + Sync>,
}

impl Orchestrator {
    /// Construct with defaults: [`PolicyRouter::default`],
    /// [`InMemoryIdempotencyStore`], [`InMemoryCircuitBreaker`], no
    /// fraud scorer, default thresholds, empty scoring context,
    /// system clock.
    pub fn new() -> Self {
        Self {
            config: OrchestratorConfig::default(),
            router: Box::new(PolicyRouter::default()),
            adapters: HashMap::new(),
            idempotency: Box::new(InMemoryIdempotencyStore::new()),
            breaker: Box::new(InMemoryCircuitBreaker::new()),
            scorer: None,
            telemetry: crate::signals::noop_telemetry(),
            thresholds: Thresholds::default(),
            context_builder: Box::new(|_| ScoringContext::default()),
            now: Box::new(default_now),
        }
    }

    /// Builder: set the fraud thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, t: Thresholds) -> Self {
        self.thresholds = t;
        self
    }

    /// Builder: set the per-intent scoring context builder.
    #[must_use]
    pub fn with_context_builder<F: Fn(&PaymentIntent) -> ScoringContext + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.context_builder = Box::new(f);
        self
    }

    /// Set the configuration.
    #[must_use]
    pub fn with_config(mut self, config: OrchestratorConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the router.
    #[must_use]
    pub fn with_router(mut self, router: Box<dyn Router>) -> Self {
        self.router = router;
        self
    }

    /// Set the idempotency store.
    #[must_use]
    pub fn with_idempotency_store(mut self, store: Box<dyn IdempotencyStore>) -> Self {
        self.idempotency = store;
        self
    }

    /// Set the circuit breaker.
    #[must_use]
    pub fn with_circuit_breaker(mut self, breaker: Box<dyn CircuitBreaker>) -> Self {
        self.breaker = breaker;
        self
    }

    /// Set the fraud scorer.
    #[must_use]
    pub fn with_scorer(mut self, scorer: Box<dyn Scorer>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// Wire in a [`RailTelemetry`](crate::RailTelemetry) sink. After
    /// each `RailAdapter::attempt` completes, the orchestrator
    /// pushes a `(rail, driver, outcome)` record here. Pair this
    /// with a [`RoutingSignals`](crate::RoutingSignals) source on
    /// the router to close the "prefer quiet rails" loop.
    #[must_use]
    pub fn with_telemetry(mut self, telemetry: Arc<dyn crate::signals::RailTelemetry>) -> Self {
        self.telemetry = telemetry;
        self
    }

    /// Set a deterministic clock (for tests).
    #[must_use]
    pub fn with_clock<F: Fn() -> u64 + Send + Sync + 'static>(mut self, now: F) -> Self {
        self.now = Box::new(now);
        self
    }

    /// Register a rail adapter.
    pub fn register_adapter(&mut self, adapter: Arc<dyn RailAdapter>) {
        let key = (adapter.rail(), adapter.driver().to_owned());
        self.adapters.insert(key, adapter);
    }

    /// True iff a `(rail, driver)` adapter has been registered.
    /// Introspection helper for boot-time config tests — production
    /// code routes via [`Self::run`] / [`Router`], not by name lookup.
    #[must_use]
    pub fn has_driver(&self, rail: RailKind, driver: &str) -> bool {
        self.adapters.contains_key(&(rail, driver.to_owned()))
    }

    /// Iterate registered `(rail, driver)` keys. Useful for boot
    /// diagnostics and operator dashboards.
    pub fn registered_drivers(&self) -> impl Iterator<Item = (RailKind, &str)> + '_ {
        self.adapters
            .keys()
            .map(|(rail, driver)| (*rail, driver.as_str()))
    }

    /// Resume an intent that previously returned
    /// [`TerminalStatus::RequiresCustomerAction`]. The caller
    /// provides the same `intent` plus the `(rail, driver)` that
    /// issued the challenge and the `psp_payment_id` returned at
    /// challenge time. The orchestrator calls
    /// [`RailAdapter::resume`] on the matching adapter and
    /// reclassifies the outcome.
    ///
    /// The cached idempotency record is updated with the resumed
    /// outcome so subsequent retries (with the same intent body)
    /// short-circuit to the resolved state.
    ///
    /// # Errors
    /// - [`Error::NoEligibleRail`] if the `(rail, driver)` isn't
    ///   registered.
    /// - Bubbles up adapter / store errors.
    #[tracing::instrument(
        name = "orchestrator.resume",
        skip(self, intent),
        fields(
            idempotency_key = %intent.idempotency_key.as_str(),
            driver = %driver,
            rail = ?rail,
        ),
    )]
    pub fn resume(
        &self,
        intent: &PaymentIntent,
        rail: RailKind,
        driver: &str,
        psp_payment_id: &str,
    ) -> Result<OrchestrationOutcome> {
        let key = (rail, driver.to_owned());
        let adapter = self
            .adapters
            .get(&key)
            .ok_or_else(|| Error::NoEligibleRail {
                reason: format!("no adapter for rail={rail:?} driver={driver}"),
            })?;
        let result = adapter.resume(intent, psp_payment_id);
        let attempt = Attempt {
            rail,
            driver: driver.to_owned(),
            outcome: result.outcome.clone(),
        };
        let terminal = match &result.outcome {
            AttemptOutcome::Success => TerminalStatus::Approved,
            AttemptOutcome::RequiresAction { .. } => TerminalStatus::RequiresCustomerAction,
            AttemptOutcome::HardDecline { .. } | AttemptOutcome::SoftFailure { .. } => {
                TerminalStatus::Declined
            }
        };
        let outcome = OrchestrationOutcome {
            terminal_status: terminal,
            attempts: vec![attempt],
            rail_used: Some(rail),
            psp_payment_id: result.psp_payment_id.clone(),
            uetr: result.uetr.clone(),
        };
        // Update the idempotency cache so a re-run with the same key
        // sees the resolved outcome rather than re-running the
        // challenge.
        self.idempotency.commit(&intent.idempotency_key, &outcome);
        Ok(outcome)
    }

    /// Process an intent. The main entry point.
    #[tracing::instrument(
        name = "orchestrator.run",
        skip(self, intent),
        fields(
            idempotency_key = %intent.idempotency_key.as_str(),
            amount_minor = intent.amount.minor_units,
            currency = intent.amount.currency.code(),
        ),
    )]
    pub fn run(&self, intent: &PaymentIntent) -> Result<OrchestrationOutcome> {
        // 1. Idempotency check — reserve a slot or return cached
        // outcome.
        let body_sig = intent.body_signature();
        if let Some(record) = self.idempotency.reserve(&intent.idempotency_key, &body_sig) {
            if record.body_signature != body_sig {
                return Err(Error::IdempotencyMismatch);
            }
            if let Some(cached) = record.outcome {
                return Ok(cached);
            }
            // In-flight slot — another request is currently
            // processing this key. Treat as a transient: the caller
            // can retry. (A more sophisticated impl would block on
            // a condvar.)
            return Err(Error::AllRailsExhausted { attempt_count: 0 });
        }

        // From here on, we MUST commit or release before returning.
        let result = self.run_inner(intent);

        match &result {
            Ok(outcome) => self.idempotency.commit(&intent.idempotency_key, outcome),
            Err(_) => self.idempotency.release(&intent.idempotency_key),
        }

        result
    }

    fn run_inner(&self, intent: &PaymentIntent) -> Result<OrchestrationOutcome> {
        // 2. Fraud check (if a scorer is configured).
        if let Some(scorer) = &self.scorer {
            let rail = implied_rail(&intent.method);
            let descriptor = PaymentDescriptor {
                amount: intent.amount,
                method: &intent.method,
                rail,
                creditor_account: None,
                creditor_name: None,
                debtor_account: None,
                has_remittance: false,
            };
            let ctx = (self.context_builder)(intent);
            let features: FeatureVector = extract_features(&descriptor, &ctx)?;
            let score = scorer.score(&features)?;
            let decision = self.thresholds.decide(score)?;
            match decision {
                FraudDecision::Approve => { /* proceed */ }
                FraudDecision::Review => {
                    return Err(Error::FraudReviewRequired {
                        reason: format!("score={score:.3} (review threshold)"),
                    });
                }
                FraudDecision::Decline => {
                    return Err(Error::FraudDeclined {
                        reason: format!("score={score:.3} (decline threshold)"),
                    });
                }
                FraudDecision::Freeze => {
                    return Err(Error::FraudDeclined {
                        reason: format!("score={score:.3} (freeze threshold)"),
                    });
                }
            }
        }

        // 3. Routing.
        let decision = self
            .router
            .route(intent)
            .map_err(|reason| Error::NoEligibleRail { reason })?;

        // 4. Attempt loop.
        let mut attempts = Vec::new();
        let mut any_circuit_was_closed = false;

        for (i, choice) in decision.chain.iter().enumerate() {
            if attempts.len() >= self.config.max_attempts {
                break;
            }
            let now = (self.now)();
            if !self.breaker.allow(choice.rail, &choice.driver, now) {
                // Skip rails whose breaker is open; do not count as
                // an attempt against max_attempts.
                continue;
            }
            any_circuit_was_closed = true;

            let Some(adapter) = self.adapters.get(&(choice.rail, choice.driver.clone())) else {
                // Misconfigured: router pointed to a driver we don't
                // have an adapter for. Skip and continue.
                continue;
            };

            let start = std::time::Instant::now();
            let result = adapter.attempt(intent, i);
            let duration_ms = u32::try_from(start.elapsed().as_millis()).unwrap_or(u32::MAX);
            let outcome = result.outcome.clone();
            // Feed the telemetry sink before we mutate `outcome`
            // further; this is what closes the loop with any
            // RoutingSignals source the router consults next turn.
            // The idempotency key is the operator-supplied
            // correlation token; backed stores use it to join this
            // attempt against the resulting ledger transaction and
            // any reconciliation discrepancies that touch it.
            self.telemetry.record_attempt(
                choice.rail,
                &choice.driver,
                crate::signals::AttemptResultClass::classify(&outcome),
                now,
                Some(intent.idempotency_key.as_str()),
                Some(duration_ms),
            );
            attempts.push(Attempt {
                rail: choice.rail,
                driver: choice.driver.clone(),
                outcome: result.outcome,
            });

            match outcome {
                AttemptOutcome::Success => {
                    self.breaker.record_success(choice.rail, &choice.driver);
                    return Ok(OrchestrationOutcome {
                        terminal_status: TerminalStatus::Approved,
                        attempts,
                        rail_used: Some(choice.rail),
                        psp_payment_id: result.psp_payment_id,
                        uetr: result.uetr,
                    });
                }
                AttemptOutcome::RequiresAction { .. } => {
                    // Customer action is "terminal-pending" — we do
                    // not retry; we surface and stop.
                    return Ok(OrchestrationOutcome {
                        terminal_status: TerminalStatus::RequiresCustomerAction,
                        attempts,
                        rail_used: Some(choice.rail),
                        psp_payment_id: result.psp_payment_id,
                        uetr: result.uetr,
                    });
                }
                AttemptOutcome::HardDecline { .. } => {
                    // Customer-side decline. Do NOT fall back to
                    // another rail — the customer must fix it
                    // (insufficient funds, frozen account, expired).
                    self.breaker.record_success(choice.rail, &choice.driver);
                    return Ok(OrchestrationOutcome {
                        terminal_status: TerminalStatus::Declined,
                        attempts,
                        rail_used: Some(choice.rail),
                        psp_payment_id: result.psp_payment_id,
                        uetr: result.uetr,
                    });
                }
                AttemptOutcome::SoftFailure { .. } => {
                    // Transient — record breaker failure and try
                    // the next entry in the chain (which may be a
                    // fallback rail).
                    self.breaker
                        .record_failure(choice.rail, &choice.driver, now);
                    continue;
                }
            }
        }

        if !any_circuit_was_closed {
            return Err(Error::AllCircuitsOpen);
        }

        Err(Error::AllRailsExhausted {
            attempt_count: attempts.len(),
        })
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

fn default_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Map a payment method to the rail it implies for purposes of
/// fraud feature extraction.
///
/// The fraud feature vector has separate `is_rail_card` /
/// `is_rail_a2a` / `is_rail_wallet` / `is_rail_qr` indicators
/// (features 24-27). The fraud check fires BEFORE routing, so we
/// pick the rail the router would most likely choose:
///
/// - `Vault`, `Wallet`, `Emv` → `Card` (PSPs handle these).
/// - `A2a` → `A2a` (FedNow / RTP / PIX / SEPA Inst).
/// - `Qr` → `A2a` (UPI QR, PIX QR all settle on A2A rails).
///
/// If the router later picks a different rail, the score is the
/// same for THIS feature row; richer pipelines re-score per chain
/// entry, but that's an operator extension, not the reference flow.
fn implied_rail(method: &PaymentMethod) -> RailKind {
    match method {
        PaymentMethod::Vault(_) | PaymentMethod::Wallet(_) | PaymentMethod::Emv(_) => {
            RailKind::Card
        }
        PaymentMethod::A2a(_) | PaymentMethod::Qr(_) => RailKind::A2a,
        PaymentMethod::Crypto(_) => RailKind::Crypto,
        // Workspace feature unification surfaces this variant via op-vault.
        // Design contract: raw PAN never leaves the vault — reaching this
        // arm in the orchestrator is a bug, not a runtime condition.
        PaymentMethod::RawPan(_) => RailKind::Card,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money, PaymentMethod, VaultRef};

    use crate::idempotency::IdempotencyKey;

    /// Always-success card adapter for tests.
    struct AlwaysApproveCard;
    impl RailAdapter for AlwaysApproveCard {
        fn driver(&self) -> &str {
            "hyperswitch"
        }
        fn rail(&self) -> RailKind {
            RailKind::Card
        }
        fn attempt(&self, _intent: &PaymentIntent, _n: usize) -> AdapterResult {
            AdapterResult {
                outcome: AttemptOutcome::Success,
                psp_payment_id: Some("psp_approved_1".into()),
                uetr: None,
            }
        }
    }

    /// Always-soft-fail card adapter.
    struct AlwaysSoftFailCard;
    impl RailAdapter for AlwaysSoftFailCard {
        fn driver(&self) -> &str {
            "flaky"
        }
        fn rail(&self) -> RailKind {
            RailKind::Card
        }
        fn attempt(&self, _intent: &PaymentIntent, _n: usize) -> AdapterResult {
            AdapterResult {
                outcome: AttemptOutcome::SoftFailure {
                    code: "timeout".into(),
                },
                psp_payment_id: None,
                uetr: None,
            }
        }
    }

    /// Always-hard-decline card adapter.
    struct AlwaysDeclineCard;
    impl RailAdapter for AlwaysDeclineCard {
        fn driver(&self) -> &str {
            "hyperswitch"
        }
        fn rail(&self) -> RailKind {
            RailKind::Card
        }
        fn attempt(&self, _intent: &PaymentIntent, _n: usize) -> AdapterResult {
            AdapterResult {
                outcome: AttemptOutcome::HardDecline {
                    code: "insufficient_funds".into(),
                },
                psp_payment_id: Some("psp_declined_1".into()),
                uetr: None,
            }
        }
    }

    fn vault_intent(key: &str, amount_minor: i64) -> PaymentIntent {
        PaymentIntent::new(
            IdempotencyKey::new(key),
            Money::from_minor(amount_minor, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_a")),
        )
    }

    #[test]
    fn approve_happy_path() {
        let mut orch = Orchestrator::new();
        orch.register_adapter(Arc::new(AlwaysApproveCard));
        let out = orch.run(&vault_intent("k1", 1000)).unwrap();
        assert!(out.is_approved());
        assert_eq!(out.attempts.len(), 1);
        assert_eq!(out.rail_used, Some(RailKind::Card));
        assert_eq!(out.psp_payment_id.as_deref(), Some("psp_approved_1"));
    }

    #[test]
    fn hard_decline_does_not_retry() {
        let mut orch = Orchestrator::new();
        orch.register_adapter(Arc::new(AlwaysDeclineCard));
        let out = orch.run(&vault_intent("k2", 1000)).unwrap();
        assert!(out.is_declined());
        assert_eq!(out.attempts.len(), 1);
    }

    #[test]
    fn soft_failure_exhausts_chain() {
        // Single driver in the chain that always soft-fails →
        // AllRailsExhausted.
        let router = PolicyRouter::new(vec!["flaky".into()], vec![]);
        let mut orch = Orchestrator::new().with_router(Box::new(router));
        orch.register_adapter(Arc::new(AlwaysSoftFailCard));
        let err = orch.run(&vault_intent("k3", 1000)).unwrap_err();
        assert!(matches!(err, Error::AllRailsExhausted { attempt_count: 1 }));
    }

    #[test]
    fn idempotency_replay_returns_cached_outcome() {
        let mut orch = Orchestrator::new();
        orch.register_adapter(Arc::new(AlwaysApproveCard));
        let intent = vault_intent("kreplay", 1000);

        let first = orch.run(&intent).unwrap();
        let second = orch.run(&intent).unwrap();

        assert!(first.is_approved());
        assert!(second.is_approved());
        // Both should reference the same psp_payment_id — the second
        // call returned a cached outcome, did NOT re-charge.
        assert_eq!(first.psp_payment_id, second.psp_payment_id);
    }

    #[test]
    fn idempotency_mismatch_rejects_different_body() {
        let mut orch = Orchestrator::new();
        orch.register_adapter(Arc::new(AlwaysApproveCard));

        let i1 = vault_intent("kmismatch", 1000);
        let mut i2 = vault_intent("kmismatch", 1000);
        i2.amount = Money::from_minor(9999, Currency::USD);

        orch.run(&i1).unwrap();
        let err = orch.run(&i2).unwrap_err();
        assert!(matches!(err, Error::IdempotencyMismatch));
    }

    #[test]
    fn no_adapter_registered_returns_exhausted() {
        // Router produces a chain but no adapter is registered for
        // the named driver. Engine skips each entry → AllCircuitsOpen
        // (no breaker ever opened, no attempts ever made — the
        // "no attempt was even possible" signal is AllCircuitsOpen
        // by current impl).
        //
        // Actually re-reading: any_circuit_was_closed only flips
        // when we PASSED the breaker check. So with no adapters at
        // all but breakers all closed, we DO pass the breaker check
        // and then the adapter-not-found `continue` runs. attempts
        // stays empty → AllRailsExhausted { attempt_count: 0 }.
        let orch = Orchestrator::new();
        let err = orch.run(&vault_intent("kno", 1000)).unwrap_err();
        assert!(matches!(err, Error::AllRailsExhausted { attempt_count: 0 }));
    }

    #[test]
    fn circuit_breaker_short_circuits_failing_driver() {
        // Configure a low threshold so a few soft failures open the
        // breaker. After it opens, the engine reports AllCircuitsOpen.
        let cb = InMemoryCircuitBreaker::new()
            .with_threshold(2)
            .with_cooldown(3600);
        let router = PolicyRouter::new(vec!["flaky".into()], vec![]);
        let mut orch = Orchestrator::new()
            .with_router(Box::new(router))
            .with_circuit_breaker(Box::new(cb))
            .with_clock(|| 1000);
        orch.register_adapter(Arc::new(AlwaysSoftFailCard));

        // First call: max_attempts=3, but only 1 chain entry. The
        // single driver soft-fails once → breaker records 1 failure
        // → still closed → loop ends because chain is exhausted →
        // AllRailsExhausted.
        let err1 = orch.run(&vault_intent("k-cb-1", 1000)).unwrap_err();
        assert!(matches!(err1, Error::AllRailsExhausted { .. }));

        // Second call: breaker records 2nd failure → opens. But
        // this call ITSELF triggers the failure; the breaker check
        // happens before the call. So second call still attempts,
        // fails, and trips the breaker.
        let err2 = orch.run(&vault_intent("k-cb-2", 1000)).unwrap_err();
        assert!(matches!(err2, Error::AllRailsExhausted { .. }));

        // Third call: breaker is open from the start → no attempts
        // → AllCircuitsOpen.
        let err3 = orch.run(&vault_intent("k-cb-3", 1000)).unwrap_err();
        assert!(matches!(err3, Error::AllCircuitsOpen));
    }

    #[test]
    fn cross_rail_fallback_on_soft_failure() {
        // Chain: flaky-card (soft-fails) → backup-card (approves).
        let router = PolicyRouter::new(vec!["flaky".into(), "hyperswitch".into()], vec![]);
        let mut orch = Orchestrator::new().with_router(Box::new(router));
        orch.register_adapter(Arc::new(AlwaysSoftFailCard));
        orch.register_adapter(Arc::new(AlwaysApproveCard));

        let out = orch.run(&vault_intent("k-fb", 1000)).unwrap();
        assert!(out.is_approved());
        assert_eq!(out.attempts.len(), 2);
        assert_eq!(out.attempts[0].driver, "flaky");
        assert_eq!(out.attempts[1].driver, "hyperswitch");
        assert_eq!(out.rail_used, Some(RailKind::Card));
    }

    #[test]
    fn backoff_policy_grows_geometrically_with_cap() {
        let p = BackoffPolicy::default();
        assert_eq!(p.delay_for(0), 1);
        assert_eq!(p.delay_for(1), 2);
        assert_eq!(p.delay_for(2), 4);
        assert_eq!(p.delay_for(3), 8);
        assert_eq!(p.delay_for(4), 16);
        // 32 capped at 30
        assert_eq!(p.delay_for(5), 30);
        assert_eq!(p.delay_for(99), 30);
    }

    #[test]
    fn max_attempts_caps_loop() {
        // max_attempts=2 with 3 flaky entries → only 2 attempts.
        let router =
            PolicyRouter::new(vec!["flaky".into(), "flaky".into(), "flaky".into()], vec![]);
        // Note: registering "flaky" 3x is the same adapter; the
        // chain has 3 entries pointing at it.
        let mut orch = Orchestrator::new()
            .with_router(Box::new(router))
            .with_config(OrchestratorConfig {
                max_attempts: 2,
                backoff: BackoffPolicy::default(),
            });
        orch.register_adapter(Arc::new(AlwaysSoftFailCard));

        let err = orch.run(&vault_intent("k-cap", 1000)).unwrap_err();
        assert!(matches!(err, Error::AllRailsExhausted { attempt_count: 2 }));
    }
}
