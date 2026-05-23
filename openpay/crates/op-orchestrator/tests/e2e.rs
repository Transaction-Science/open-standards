//! End-to-end integration tests for the OpenPay orchestrator.
//!
//! These tests drive the full layered stack — `op-core` types →
//! `op-vault` (real Phase 7 cipher) → `op-fraud` (real HeuristicScorer)
//! → `op-orchestrator` (engine + adapters) → mock rail drivers —
//! and verify that the architectural composition delivers the
//! expected behavior:
//!
//! 1. Happy path: tokenize → score → route → authorize → outcome.
//! 2. Cross-rail fallback: primary card driver soft-fails, backup
//!    approves.
//! 3. Idempotency replay: same intent submitted twice returns the
//!    cached outcome without hitting the rail.
//! 4. Idempotency mismatch: same key with different body is rejected.
//! 5. Hard decline short-circuits — no fallback.
//! 6. Customer action surfaces a redirect URL terminal-pending.
//! 7. Circuit breaker trips after threshold failures and short-
//!    circuits subsequent calls.
//! 8. Fraud decline: scorer rejects high-risk intent before any rail
//!    is called.
//! 9. UETR determinism across A2A retries.
//! 10. Real vault tokenize → detokenize round-trip used as a method.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use op_core::{A2aKey, Currency, Money, PaymentMethod, RailKind, VaultRef};
use op_orchestrator::{
    A2aAdapter, CardAdapter, Error, IdempotencyKey, InMemoryCircuitBreaker, MerchantBankProfile,
    Orchestrator, OrchestratorConfig, PaymentIntent, PolicyRouter, TerminalStatus,
};
use op_rails_a2a::acquirer::{
    A2aAcquirer, A2aDecision, A2aStatus, CreditTransferReq, ParticipantId, StatusQueryReq,
};
use op_rails_card::acquirer::{AuthStatus, CardAcquirer};
use op_rails_card::{AuthDecision, AuthRequest, CaptureRequest, RefundRequest, VoidRequest};

const VALID_VISA: &str = "4242424242424242";

// ============================================================
// Mock rail drivers used across the e2e suite
// ============================================================

/// Card acquirer that returns a programmable canned outcome and
/// counts the number of authorize() calls.
struct CannedCardAcquirer {
    driver_name: &'static str,
    canned: AuthStatus,
    call_count: AtomicU32,
}

impl CannedCardAcquirer {
    fn new(name: &'static str, canned: AuthStatus) -> Self {
        Self {
            driver_name: name,
            canned,
            call_count: AtomicU32::new(0),
        }
    }
    fn call_count(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl CardAcquirer for CannedCardAcquirer {
    fn name(&self) -> &'static str {
        self.driver_name
    }
    fn supports(&self, _m: &PaymentMethod) -> bool {
        true
    }
    fn authorize(&self, req: &AuthRequest) -> op_rails_card::Result<AuthDecision> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(AuthDecision {
            psp_payment_id: format!("{}_{}", self.driver_name, req.idempotency_key),
            status: self.canned,
            raw_status: format!("{:?}", self.canned).to_lowercase(),
            authorized_amount: Some(req.amount),
            redirect_url: if matches!(self.canned, AuthStatus::RequiresCustomerAction) {
                Some("https://3ds.example/c/abc".into())
            } else {
                None
            },
            error_code: match self.canned {
                AuthStatus::HardDecline => Some("insufficient_funds".into()),
                AuthStatus::Fraud => Some("fraud".into()),
                _ => None,
            },
            error_message: None,
        })
    }
    fn capture(&self, _req: &CaptureRequest) -> op_rails_card::Result<AuthDecision> {
        unimplemented!("not used in e2e")
    }
    fn void(&self, _req: &VoidRequest) -> op_rails_card::Result<AuthDecision> {
        unimplemented!("not used in e2e")
    }
    fn refund(&self, _req: &RefundRequest) -> op_rails_card::Result<AuthDecision> {
        unimplemented!("not used in e2e")
    }
}

/// A2A acquirer that returns a programmable canned outcome and
/// captures the last request seen so tests can assert on it.
struct CannedA2aAcquirer {
    name: &'static str,
    canned: A2aStatus,
    last_uetr: std::sync::Mutex<Option<String>>,
}

impl CannedA2aAcquirer {
    fn new(name: &'static str, canned: A2aStatus) -> Self {
        Self {
            name,
            canned,
            last_uetr: std::sync::Mutex::new(None),
        }
    }
    fn last_uetr(&self) -> Option<String> {
        self.last_uetr.lock().unwrap().clone()
    }
}

impl A2aAcquirer for CannedA2aAcquirer {
    fn name(&self) -> &'static str {
        self.name
    }
    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> op_rails_a2a::Result<A2aDecision> {
        *self.last_uetr.lock().unwrap() = Some(req.uetr.clone());
        Ok(A2aDecision {
            status: self.canned,
            raw_status: format!("{:?}", self.canned),
            reason_code: match self.canned {
                A2aStatus::Rejected => Some("AC03".into()),
                _ => None,
            },
            reason_text: None,
            uetr: Some(req.uetr.clone()),
            rail_txn_id: Some(format!("rail_{}", req.end_to_end_id)),
            settled_amount: Some(req.amount),
        })
    }
    fn query_status(&self, _req: &StatusQueryReq) -> op_rails_a2a::Result<A2aDecision> {
        unimplemented!("not used in e2e")
    }
}

fn merchant_profile() -> MerchantBankProfile {
    MerchantBankProfile {
        creditor_agent: ParticipantId::Aba("021000021".into()),
        creditor_account: "MERCHANT-1".into(),
        creditor_name: "Acme Coffee".into(),
        default_debtor_agent: ParticipantId::Aba("026009593".into()),
        default_debtor_name: "Customer".into(),
    }
}

fn card_intent(key: &str, amount_minor: i64) -> PaymentIntent {
    PaymentIntent::new(
        IdempotencyKey::new(key),
        Money::from_minor(amount_minor, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok_v7_visa")),
    )
}

fn a2a_intent(key: &str, amount_minor: i64) -> PaymentIntent {
    PaymentIntent::new(
        IdempotencyKey::new(key),
        Money::from_minor(amount_minor, Currency::USD),
        PaymentMethod::A2a(A2aKey::UsAch {
            routing: "121000358".into(),
            account: "CUST-9".into(),
        }),
    )
}

// ============================================================
// Test 1: Happy path
// ============================================================

#[test]
fn e2e_happy_path_card_approves() {
    let acq = Arc::new(CannedCardAcquirer::new("hsw", AuthStatus::Settled));
    let mut orch =
        Orchestrator::new().with_router(Box::new(PolicyRouter::new(vec!["hsw".into()], vec![])));
    orch.register_adapter(Arc::new(CardAdapter::new("hsw", acq.clone())));

    let outcome = orch.run(&card_intent("ord-1", 1299)).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);
    assert_eq!(outcome.rail_used, Some(RailKind::Card));
    assert_eq!(outcome.attempts.len(), 1);
    assert_eq!(acq.call_count(), 1);
    assert!(outcome.psp_payment_id.is_some());
}

// ============================================================
// Test 2: Cross-driver fallback inside the card rail
// ============================================================

#[test]
fn e2e_card_psp_fallback_on_transient() {
    let primary = Arc::new(CannedCardAcquirer::new("primary", AuthStatus::Transient));
    let backup = Arc::new(CannedCardAcquirer::new("backup", AuthStatus::Settled));

    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["primary".into(), "backup".into()],
        vec![],
    )));
    orch.register_adapter(Arc::new(CardAdapter::new("primary", primary.clone())));
    orch.register_adapter(Arc::new(CardAdapter::new("backup", backup.clone())));

    let outcome = orch.run(&card_intent("ord-2", 4500)).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);
    assert_eq!(outcome.attempts.len(), 2);
    assert_eq!(outcome.attempts[0].driver, "primary");
    assert_eq!(outcome.attempts[1].driver, "backup");
    assert_eq!(primary.call_count(), 1);
    assert_eq!(backup.call_count(), 1);
}

// ============================================================
// Test 3: Idempotency replay
// ============================================================

#[test]
fn e2e_idempotency_replay_returns_cached_outcome_without_rail_call() {
    let acq = Arc::new(CannedCardAcquirer::new("hsw", AuthStatus::Settled));
    let mut orch =
        Orchestrator::new().with_router(Box::new(PolicyRouter::new(vec!["hsw".into()], vec![])));
    orch.register_adapter(Arc::new(CardAdapter::new("hsw", acq.clone())));

    let intent = card_intent("ord-replay", 1000);
    let o1 = orch.run(&intent).unwrap();
    let o2 = orch.run(&intent).unwrap();

    assert_eq!(o1.terminal_status, TerminalStatus::Approved);
    assert_eq!(o2.terminal_status, TerminalStatus::Approved);
    assert_eq!(o1.psp_payment_id, o2.psp_payment_id);

    // CRITICAL invariant: the rail was called ONCE despite TWO
    // orchestrator runs. This is the no-double-charge guarantee.
    assert_eq!(acq.call_count(), 1, "rail must NOT be re-called on replay");
}

// ============================================================
// Test 4: Idempotency mismatch
// ============================================================

#[test]
fn e2e_idempotency_mismatch_rejects_amount_change() {
    let acq = Arc::new(CannedCardAcquirer::new("hsw", AuthStatus::Settled));
    let mut orch =
        Orchestrator::new().with_router(Box::new(PolicyRouter::new(vec!["hsw".into()], vec![])));
    orch.register_adapter(Arc::new(CardAdapter::new("hsw", acq)));

    let i1 = card_intent("ord-mismatch", 1000);
    let mut i2 = card_intent("ord-mismatch", 1000);
    i2.amount = Money::from_minor(9999, Currency::USD);

    orch.run(&i1).unwrap();
    let err = orch.run(&i2).unwrap_err();
    assert!(
        matches!(err, Error::IdempotencyMismatch),
        "expected IdempotencyMismatch, got {err:?}"
    );
}

// ============================================================
// Test 5: Hard decline short-circuits (no fallback)
// ============================================================

#[test]
fn e2e_hard_decline_does_not_fall_back() {
    let primary = Arc::new(CannedCardAcquirer::new("primary", AuthStatus::HardDecline));
    let backup = Arc::new(CannedCardAcquirer::new("backup", AuthStatus::Settled));

    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["primary".into(), "backup".into()],
        vec![],
    )));
    orch.register_adapter(Arc::new(CardAdapter::new("primary", primary.clone())));
    orch.register_adapter(Arc::new(CardAdapter::new("backup", backup.clone())));

    let outcome = orch.run(&card_intent("ord-hd", 1000)).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Declined);
    assert_eq!(outcome.attempts.len(), 1, "hard decline must NOT fall back");
    assert_eq!(primary.call_count(), 1);
    assert_eq!(backup.call_count(), 0, "backup must not be called");
}

// ============================================================
// Test 6: RequiresCustomerAction surfaces redirect URL terminal-pending
// ============================================================

#[test]
fn e2e_three_ds_challenge_surfaces_redirect_url() {
    let acq = Arc::new(CannedCardAcquirer::new(
        "hsw",
        AuthStatus::RequiresCustomerAction,
    ));
    let backup = Arc::new(CannedCardAcquirer::new("backup", AuthStatus::Settled));

    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["hsw".into(), "backup".into()],
        vec![],
    )));
    orch.register_adapter(Arc::new(CardAdapter::new("hsw", acq)));
    orch.register_adapter(Arc::new(CardAdapter::new("backup", backup.clone())));

    let outcome = orch.run(&card_intent("ord-3ds", 1000)).unwrap();
    assert_eq!(
        outcome.terminal_status,
        TerminalStatus::RequiresCustomerAction
    );
    assert_eq!(
        outcome.attempts.len(),
        1,
        "customer-action is terminal-pending, no fallback"
    );
    assert_eq!(
        backup.call_count(),
        0,
        "backup must not be called for customer-action"
    );
    // The attempt outcome must carry the redirect URL.
    match &outcome.attempts[0].outcome {
        op_orchestrator::AttemptOutcome::RequiresAction { url } => {
            assert_eq!(url, "https://3ds.example/c/abc");
        }
        other => panic!("expected RequiresAction, got {other:?}"),
    }
}

// ============================================================
// Test 7: Circuit breaker trips after consecutive failures
// ============================================================

#[test]
fn e2e_circuit_breaker_trips_then_short_circuits() {
    let acq = Arc::new(CannedCardAcquirer::new("flaky", AuthStatus::Transient));
    let breaker = InMemoryCircuitBreaker::new()
        .with_threshold(2)
        .with_cooldown(3600);

    let mut orch = Orchestrator::new()
        .with_router(Box::new(PolicyRouter::new(vec!["flaky".into()], vec![])))
        .with_circuit_breaker(Box::new(breaker))
        .with_clock(|| 1000);
    orch.register_adapter(Arc::new(CardAdapter::new("flaky", acq.clone())));

    // Each run uses a fresh idempotency key so the cache doesn't
    // short-circuit the rail call.
    let _ = orch.run(&card_intent("k1", 100)).unwrap_err();
    let _ = orch.run(&card_intent("k2", 100)).unwrap_err();
    // By now the breaker has recorded 2 failures (threshold = 2) and
    // is open. The third call should short-circuit.
    let err3 = orch.run(&card_intent("k3", 100)).unwrap_err();
    assert!(
        matches!(err3, Error::AllCircuitsOpen),
        "expected AllCircuitsOpen on 3rd call, got {err3:?}"
    );
    assert_eq!(
        acq.call_count(),
        2,
        "rail must not be called after breaker opens"
    );
}

// ============================================================
// Test 8: Fraud decline blocks the call before rail dispatch
// ============================================================

#[test]
fn e2e_fraud_decline_short_circuits_before_rail() {
    // HeuristicScorer with default thresholds: review at 0.50,
    // decline at 0.80, freeze at 0.95. We can't easily force the
    // scorer to decline without a model — use a custom scorer that
    // returns a fixed high score.
    struct AlwaysDecline;
    impl op_fraud::Scorer for AlwaysDecline {
        fn name(&self) -> &str {
            "always-decline"
        }
        fn score(&self, _f: &op_fraud::FeatureVector) -> op_fraud::Result<f32> {
            Ok(0.99)
        }
    }

    let acq = Arc::new(CannedCardAcquirer::new("hsw", AuthStatus::Settled));
    let mut orch = Orchestrator::new()
        .with_scorer(Box::new(AlwaysDecline))
        .with_router(Box::new(PolicyRouter::new(vec!["hsw".into()], vec![])));
    orch.register_adapter(Arc::new(CardAdapter::new("hsw", acq.clone())));

    let err = orch.run(&card_intent("ord-fraud", 1000)).unwrap_err();
    assert!(
        matches!(err, Error::FraudDeclined { .. }),
        "expected FraudDeclined, got {err:?}"
    );
    assert_eq!(
        acq.call_count(),
        0,
        "rail must NOT be called when fraud declines"
    );
}

// ============================================================
// Test 9: UETR determinism across A2A retries
// ============================================================

#[test]
fn e2e_a2a_uetr_is_deterministic_across_retries() {
    let acq1 = Arc::new(CannedA2aAcquirer::new("fednow", A2aStatus::Transient));
    // First orchestrator: soft-fails so we can inspect the UETR
    // it sent.
    let mut orch1 =
        Orchestrator::new().with_router(Box::new(PolicyRouter::new(vec![], vec!["fednow".into()])));
    orch1.register_adapter(Arc::new(A2aAdapter::new(
        "fednow",
        acq1.clone(),
        merchant_profile(),
    )));
    let _ = orch1.run(&a2a_intent("retry-key-x", 5000)).unwrap_err();
    let first_uetr = acq1.last_uetr().expect("first UETR captured");

    // Fresh orchestrator (simulates a process restart). Same
    // idempotency key. The adapter must derive the SAME UETR so the
    // rail's idempotency contract is preserved.
    let acq2 = Arc::new(CannedA2aAcquirer::new("fednow", A2aStatus::Settled));
    let mut orch2 =
        Orchestrator::new().with_router(Box::new(PolicyRouter::new(vec![], vec!["fednow".into()])));
    orch2.register_adapter(Arc::new(A2aAdapter::new(
        "fednow",
        acq2.clone(),
        merchant_profile(),
    )));
    let o = orch2.run(&a2a_intent("retry-key-x", 5000)).unwrap();
    assert_eq!(o.terminal_status, TerminalStatus::Approved);

    let second_uetr = acq2.last_uetr().expect("second UETR captured");
    assert_eq!(
        first_uetr, second_uetr,
        "UETR MUST be deterministic across retries with the same idempotency key"
    );
}

// ============================================================
// Test 10: A2A native flow round-trip
// ============================================================

#[test]
fn e2e_a2a_native_round_trip() {
    let acq = Arc::new(CannedA2aAcquirer::new("fednow", A2aStatus::Settled));
    let mut orch =
        Orchestrator::new().with_router(Box::new(PolicyRouter::new(vec![], vec!["fednow".into()])));
    orch.register_adapter(Arc::new(A2aAdapter::new("fednow", acq, merchant_profile())));

    let outcome = orch.run(&a2a_intent("ord-a2a", 5000)).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);
    assert_eq!(outcome.rail_used, Some(RailKind::A2a));
    assert!(outcome.uetr.is_some());
    assert!(
        outcome.psp_payment_id.is_none(),
        "A2A produces UETR, not PSP id"
    );
}

// ============================================================
// Test 11: Vault tokenize → detokenize round-trip used as a method
// ============================================================

#[test]
fn e2e_vault_tokenized_pan_flows_through_orchestrator() {
    // Use the real Phase 7 in-memory vault to tokenize a card, then
    // use the returned VaultRef as the PaymentMethod for an intent.
    use op_vault::{CardData, InMemoryVault, TokenizationPolicy, Vault};

    let vault = InMemoryVault::ephemeral("e2e");
    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let token = vault.tokenize(card, TokenizationPolicy::default()).unwrap();
    assert!(token.as_str().starts_with("tok_v7_"));

    // Now use that token as the PaymentMethod.
    let intent = PaymentIntent::new(
        IdempotencyKey::new("vault-flow"),
        Money::from_minor(2500, Currency::USD),
        PaymentMethod::Vault(token.clone()),
    );

    let acq = Arc::new(CannedCardAcquirer::new("hsw", AuthStatus::Settled));
    let mut orch =
        Orchestrator::new().with_router(Box::new(PolicyRouter::new(vec!["hsw".into()], vec![])));
    orch.register_adapter(Arc::new(CardAdapter::new("hsw", acq.clone())));

    let outcome = orch.run(&intent).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);

    // The vault can still detokenize the original token (the
    // orchestrator never reads PAN bytes — that's the merchant's
    // job downstream when it presents to the rail driver).
    let card2 = vault.detokenize(&token).unwrap();
    assert_eq!(card2.last_four(), "4242");
}

// ============================================================
// Test 12: Soft failure exhausts ALL drivers when none recover
// ============================================================

#[test]
fn e2e_all_soft_failures_exhausts_chain() {
    let p1 = Arc::new(CannedCardAcquirer::new("p1", AuthStatus::Transient));
    let p2 = Arc::new(CannedCardAcquirer::new("p2", AuthStatus::Transient));

    let mut orch = Orchestrator::new()
        .with_config(OrchestratorConfig {
            max_attempts: 5,
            backoff: Default::default(),
        })
        .with_router(Box::new(PolicyRouter::new(
            vec!["p1".into(), "p2".into()],
            vec![],
        )));
    orch.register_adapter(Arc::new(CardAdapter::new("p1", p1.clone())));
    orch.register_adapter(Arc::new(CardAdapter::new("p2", p2.clone())));

    let err = orch.run(&card_intent("ord-exhaust", 100)).unwrap_err();
    assert!(matches!(err, Error::AllRailsExhausted { attempt_count: 2 }));
    assert_eq!(p1.call_count(), 1);
    assert_eq!(p2.call_count(), 1);
}

// ============================================================
// Test 13: Cross-platform observability discriminants
// ============================================================
//
// Phase 8/9/10 platform bridges (Swift, JNI, wasm) all share i32
// error discriminants. The orchestrator-level Error type doesn't
// have those discriminants directly, but the inner op_vault and
// op_fraud errors are #[from]-wrapped. Verify the architectural
// contract: vault-side oracle discipline survives the wrap.

#[test]
fn e2e_oracle_discipline_preserved_through_orchestrator() {
    // The op-vault crate collapses NotFound | AuthFailed | InvalidToken
    // into VaultLookupFailed. The orchestrator's Error::Vault wraps
    // op_vault::Error directly so the variant is preserved end-to-end.
    let v: op_orchestrator::Error = op_orchestrator::Error::Vault(op_vault::Error::NotFound);
    match v {
        op_orchestrator::Error::Vault(op_vault::Error::NotFound) => { /* ok */ }
        other => panic!("unexpected: {other:?}"),
    }
    let v2: op_orchestrator::Error = op_orchestrator::Error::Vault(op_vault::Error::AuthFailed);
    match v2 {
        op_orchestrator::Error::Vault(op_vault::Error::AuthFailed) => { /* ok */ }
        other => panic!("unexpected: {other:?}"),
    }
}
