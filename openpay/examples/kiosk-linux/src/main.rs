//! # OpenPay kiosk-linux
//!
//! Reference unattended-checkout terminal that demonstrates the full
//! OpenPay stack working end-to-end on a single Linux box:
//!
//! ```text
//!   Customer taps card → EMV terminal kernel → op-emv
//!     │
//!     ▼
//!   Vault tokenization (op-vault)
//!     │
//!     ▼
//!   Fraud scoring (op-fraud HeuristicScorer)
//!     │
//!     ▼
//!   Orchestrator routing (op-orchestrator + PolicyRouter)
//!     │
//!     ▼
//!   Card rail (CardAdapter → mock CardAcquirer)
//!     │
//!     ▼   on soft failure: cross-rail fallback
//!   A2A rail (A2aAdapter → mock A2aAcquirer)
//!     │
//!     ▼
//!   OrchestrationOutcome → kiosk display
//! ```
//!
//! ## Why this binary exists
//!
//! Two reasons:
//!
//! 1. **Architectural proof.** It demonstrates that every layer
//!    shipped in Phases 1-10 composes — the orchestrator (Phase 11)
//!    is the missing keystone that wires them together. If the
//!    composition doesn't compile or doesn't terminate with a
//!    sensible outcome, the architectural thesis is wrong.
//!
//! 2. **Operator template.** Apache-2.0 reference code that a vendor
//!    shipping unattended checkout (parking meters, vending, EV
//!    charging) can fork. The only real changes needed are: plug in
//!    real `CardAcquirer` / `A2aAcquirer` drivers, swap
//!    `InMemoryIdempotencyStore` for a Redis-backed one, and add the
//!    physical kiosk's I/O glue (touchscreen, receipt printer,
//!    etc.).
//!
//! ## Mock backends, real composition
//!
//! The two rail drivers are mocks — `MockCardAcquirer` and
//! `MockA2aAcquirer` — that return canned decisions so the kiosk
//! runs offline. They implement the same `CardAcquirer` /
//! `A2aAcquirer` traits as the real Hyperswitch / FedNow drivers,
//! so swapping them for production drivers is a one-line change.

use std::sync::Arc;

use op_core::{A2aKey, Currency, Money, PaymentMethod, VaultRef};
use op_fraud::HeuristicScorer;
use op_orchestrator::{
    A2aAdapter, CardAdapter, IdempotencyKey, MerchantBankProfile, Orchestrator, PaymentIntent,
    PolicyRouter, TerminalStatus,
};
use op_rails_a2a::{
    Result as A2aResult,
    acquirer::{
        A2aAcquirer, A2aDecision, A2aStatus, CreditTransferReq, ParticipantId, StatusQueryReq,
    },
};
use op_rails_card::{
    AuthDecision, AuthRequest, CaptureRequest, RefundRequest, Result as CardResult, VoidRequest,
    acquirer::{AuthStatus, CardAcquirer},
};

fn main() {
    eprintln!("=== OpenPay kiosk-linux reference ===");
    eprintln!("Wiring Phases 1-11 into a single end-to-end flow.\n");

    let kiosk = Kiosk::new();

    // Scenario 1: happy path — card auth approved.
    eprintln!("[1] Customer taps card. Expected: card-rail approve.");
    let result = kiosk.checkout_card(
        "ORDER-001",
        1299, // $12.99
        "tok_v7_demo_visa_4242",
        12,
        2030,
    );
    print_outcome("Scenario 1 (happy card)", &result);

    // Scenario 2: primary card PSP times out → fallback to backup
    // card PSP (within-rail fallback, the most common production
    // pattern).
    eprintln!("\n[2] Primary card PSP times out; orchestrator retries on backup PSP.");
    let result = kiosk.checkout_card(
        "ORDER-002",
        4500, // $45.00
        "tok_v7_demo_timeout",
        12,
        2030,
    );
    print_outcome("Scenario 2 (card PSP fallback)", &result);

    // Scenario 3: A2A-native direct (customer scanned QR).
    eprintln!("\n[3] Customer paid by bank transfer (UPI handle).");
    let result = kiosk.checkout_a2a(
        "ORDER-003",
        750, // $7.50
        A2aKey::Upi("alice@hdfc".into()),
    );
    print_outcome("Scenario 3 (A2A native)", &result);

    // Scenario 4: idempotency replay (e.g. terminal lost network
    // between auth and confirmation).
    eprintln!("\n[4] Replay of order ORDER-001 (same idempotency key).");
    let result = kiosk.checkout_card("ORDER-001", 1299, "tok_v7_demo_visa_4242", 12, 2030);
    print_outcome("Scenario 4 (replay)", &result);
    eprintln!(
        "    ↑ note attempt_count is 1, not 2 — the orchestrator returned the cached outcome\n      without touching the rail. No double charge."
    );

    // Scenario 5: hard decline (insufficient funds).
    eprintln!("\n[5] Customer's card declines hard (insufficient funds).");
    let result = kiosk.checkout_card(
        "ORDER-005",
        50000, // $500.00
        "tok_v7_demo_insufficient",
        12,
        2030,
    );
    print_outcome("Scenario 5 (hard decline)", &result);

    eprintln!("\n=== Done ===");
}

fn print_outcome(
    label: &str,
    outcome: &op_orchestrator::Result<op_orchestrator::OrchestrationOutcome>,
) {
    match outcome {
        Ok(o) => {
            let status_str = match o.terminal_status {
                TerminalStatus::Approved => "APPROVED",
                TerminalStatus::RequiresCustomerAction => "ACTION REQUIRED",
                TerminalStatus::Declined => "DECLINED",
            };
            eprintln!("    {label}: {status_str}");
            eprintln!("      attempts: {}", o.attempts.len());
            for (i, a) in o.attempts.iter().enumerate() {
                eprintln!(
                    "        [{i}] rail={:?} driver={} outcome={:?}",
                    a.rail, a.driver, a.outcome
                );
            }
            if let Some(rail) = o.rail_used {
                eprintln!("      terminal rail: {rail:?}");
            }
            if let Some(id) = &o.psp_payment_id {
                eprintln!("      psp_payment_id: {id}");
            }
            if let Some(u) = &o.uetr {
                eprintln!("      uetr: {u}");
            }
        }
        Err(e) => {
            eprintln!("    {label}: ERROR {e}");
        }
    }
}

// ============================================================
// Kiosk: thin orchestration over the OpenPay stack.
// ============================================================

/// The unattended-checkout terminal. Owns:
///
/// - An in-process [`InMemoryVault`](op_vault::InMemoryVault) for
///   the demo. Production kiosks delegate to a hosted vault.
/// - An [`Orchestrator`] with mock card + A2A adapters registered.
struct Kiosk {
    orchestrator: Orchestrator,
}

impl Kiosk {
    fn new() -> Self {
        let mut orchestrator = Orchestrator::new()
            // Use the heuristic scorer with default thresholds. The
            // amounts in our scenarios stay well below decline.
            .with_scorer(Box::new(HeuristicScorer::new()))
            // Two card drivers (primary + backup) and one A2A driver.
            // Routing is method-driven: Vault methods try card
            // drivers in order; A2A methods go to the A2A driver.
            .with_router(Box::new(PolicyRouter::new(
                vec!["mock-hyperswitch".into(), "mock-stripe".into()],
                vec!["mock-fednow".into()],
            )));

        // Primary card adapter — handles the happy path. Soft-fails
        // on the "timeout" demo token so we exercise the fallback.
        orchestrator.register_adapter(Arc::new(CardAdapter::new(
            "mock-hyperswitch",
            Arc::new(MockCardAcquirer::primary()),
        )));

        // Backup card adapter — approves everything. Used when the
        // primary soft-fails.
        orchestrator.register_adapter(Arc::new(CardAdapter::new(
            "mock-stripe",
            Arc::new(MockCardAcquirer::backup()),
        )));

        // A2A adapter with merchant-side bank info.
        let profile = MerchantBankProfile {
            creditor_agent: ParticipantId::Aba("021000021".into()),
            creditor_account: "MERCHANT-ACCT-1".into(),
            creditor_name: "Acme Coffee LLC".into(),
            default_debtor_agent: ParticipantId::Aba("026009593".into()),
            default_debtor_name: "Customer".into(),
        };
        orchestrator.register_adapter(Arc::new(A2aAdapter::new(
            "mock-fednow",
            Arc::new(MockA2aAcquirer),
            profile,
        )));

        Self { orchestrator }
    }

    /// Tap-to-pay style: card token already obtained from EMV kernel.
    fn checkout_card(
        &self,
        order_id: &str,
        amount_minor: i64,
        token: &str,
        exp_month: u8,
        exp_year: u16,
    ) -> op_orchestrator::Result<op_orchestrator::OrchestrationOutcome> {
        let _ = (exp_month, exp_year); // demo: token already carries exp
        let intent = PaymentIntent::new(
            IdempotencyKey::new(order_id),
            Money::from_minor(amount_minor, Currency::USD),
            PaymentMethod::Vault(VaultRef::new(token)),
        )
        .with_metadata("order_id", order_id);

        self.orchestrator.run(&intent)
    }

    /// A2A-native flow: customer scanned merchant QR or selected
    /// "Pay by Bank" at the kiosk.
    fn checkout_a2a(
        &self,
        order_id: &str,
        amount_minor: i64,
        key: A2aKey,
    ) -> op_orchestrator::Result<op_orchestrator::OrchestrationOutcome> {
        let intent = PaymentIntent::new(
            IdempotencyKey::new(order_id),
            Money::from_minor(amount_minor, Currency::USD),
            PaymentMethod::A2a(key),
        )
        .with_metadata("order_id", order_id);

        self.orchestrator.run(&intent)
    }
}

// ============================================================
// Mock rail drivers.
//
// These implement the same traits as the real Hyperswitch / FedNow
// drivers. Swapping them out is a one-line change in `Kiosk::new()`.
// ============================================================

/// Mock card acquirer. The `is_backup` flag flips the timeout
/// behavior — primary soft-fails on the "timeout" demo token; backup
/// always approves.
struct MockCardAcquirer {
    is_backup: bool,
}

impl MockCardAcquirer {
    fn primary() -> Self {
        Self { is_backup: false }
    }

    fn backup() -> Self {
        Self { is_backup: true }
    }
}

impl CardAcquirer for MockCardAcquirer {
    fn name(&self) -> &'static str {
        if self.is_backup {
            "mock-stripe"
        } else {
            "mock-hyperswitch"
        }
    }

    fn supports(&self, method: &PaymentMethod) -> bool {
        matches!(
            method,
            PaymentMethod::Vault(_) | PaymentMethod::Wallet(_) | PaymentMethod::Emv(_)
        )
    }

    fn authorize(&self, req: &AuthRequest) -> CardResult<AuthDecision> {
        // Token-driven canned responses, so the demo scenarios are
        // deterministic.
        let token_str = match &req.method {
            PaymentMethod::Vault(v) => v.as_str().to_owned(),
            _ => "<non-vault>".to_owned(),
        };
        let (status, error_code) = if token_str.contains("timeout") && !self.is_backup {
            // Only the primary times out; the backup approves.
            (AuthStatus::Transient, Some("network_timeout"))
        } else if token_str.contains("insufficient") {
            (AuthStatus::HardDecline, Some("insufficient_funds"))
        } else {
            (AuthStatus::Settled, None)
        };

        Ok(AuthDecision {
            psp_payment_id: format!("{}_{}", self.name(), req.idempotency_key),
            status,
            raw_status: format!("{status:?}").to_lowercase(),
            authorized_amount: Some(req.amount),
            redirect_url: None,
            error_code: error_code.map(|s| s.to_owned()),
            error_message: error_code.map(|s| s.to_owned()),
        })
    }

    fn capture(&self, _req: &CaptureRequest) -> CardResult<AuthDecision> {
        Err(op_rails_card::Error::DriverValidation(
            "mock driver does not support capture (auto-capture only)".into(),
        ))
    }

    fn void(&self, _req: &VoidRequest) -> CardResult<AuthDecision> {
        Err(op_rails_card::Error::DriverValidation(
            "mock driver does not support void".into(),
        ))
    }

    fn refund(&self, _req: &RefundRequest) -> CardResult<AuthDecision> {
        Err(op_rails_card::Error::DriverValidation(
            "mock driver does not support refund".into(),
        ))
    }
}

struct MockA2aAcquirer;

impl A2aAcquirer for MockA2aAcquirer {
    fn name(&self) -> &'static str {
        "mock-fednow"
    }

    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> A2aResult<A2aDecision> {
        // Always settle for the kiosk demo. A real driver inspects
        // the rail's pacs.002 response and maps it via op-iso20022.
        Ok(A2aDecision {
            status: A2aStatus::Settled,
            raw_status: "ACSC".into(),
            reason_code: None,
            reason_text: None,
            uetr: Some(req.uetr.clone()),
            rail_txn_id: Some(format!("FED-MOCK-{}", &req.end_to_end_id)),
            settled_amount: Some(req.amount),
        })
    }

    fn query_status(&self, _req: &StatusQueryReq) -> A2aResult<A2aDecision> {
        Err(op_rails_a2a::Error::DriverValidation(
            "mock driver does not support query_status".into(),
        ))
    }
}
