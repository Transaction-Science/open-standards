//! End-to-end integration: wire `DeterministicCardAcquirer` and
//! `DeterministicA2aGateway` into a real Orchestrator, prove the
//! sample-driver path works without any live PSP.

use std::sync::Arc;

use op_core::RailKind;
use op_core::{CryptoAddress, Currency, Money, PaymentMethod, VaultRef};
use op_driver_sdk::{
    DeterministicA2aGateway, DeterministicCardAcquirer, DeterministicCryptoGateway, conformance,
};
use op_orchestrator::{
    A2aAdapter, CardAdapter, CryptoAdapter, IdempotencyKey, MerchantBankProfile, Orchestrator,
    PaymentIntent, PolicyRouter, TerminalStatus,
};
use op_rails_a2a::acquirer::{A2aStatus, ParticipantId};
use op_rails_card::acquirer::AuthStatus;
use op_rails_crypto::{CryptoStatus, StableToken};

#[test]
fn deterministic_card_routes_through_orchestrator_and_settles() {
    let acquirer = Arc::new(DeterministicCardAcquirer::new());
    let card = Arc::new(CardAdapter::new("det-card", acquirer.clone()));

    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["det-card".to_owned()],
        vec![],
    )));
    orch.register_adapter(card);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("e2e-card-1"),
        Money::from_minor(2500, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok_v7_e2e")),
    );
    let outcome = orch.run(&intent).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);
    assert_eq!(outcome.attempt_count(), 1);
    assert!(outcome.psp_payment_id.is_some());

    // Idempotency key flowed into the acquirer untouched.
    let history = acquirer.auth_history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].idempotency_key, "e2e-card-1");
}

#[test]
fn deterministic_card_with_key_override_declines() {
    let acquirer = Arc::new(DeterministicCardAcquirer::new().with_key_override(
        "e2e-decline",
        AuthStatus::HardDecline,
        Some("nsf".into()),
    ));
    let card = Arc::new(CardAdapter::new("det-card", acquirer));
    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["det-card".to_owned()],
        vec![],
    )));
    orch.register_adapter(card);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("e2e-decline"),
        Money::from_minor(2500, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok_v7_decline")),
    );
    let outcome = orch.run(&intent).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Declined);
}

#[test]
fn deterministic_a2a_routes_through_orchestrator() {
    let gateway = Arc::new(DeterministicA2aGateway::new().with_name("det-a2a"));
    let profile = MerchantBankProfile {
        creditor_agent: ParticipantId::Aba("021000021".into()),
        creditor_account: "1234567890".into(),
        creditor_name: "ACME CORP".into(),
        default_debtor_agent: ParticipantId::Aba("121000248".into()),
        default_debtor_name: "Customer".into(),
    };
    let a2a = Arc::new(A2aAdapter::new("det-a2a", gateway.clone(), profile));

    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec![],
        vec!["det-a2a".to_owned()],
    )));
    orch.register_adapter(a2a);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("e2e-a2a-1"),
        Money::from_minor(50_000, Currency::USD),
        PaymentMethod::A2a(op_core::A2aKey::UsAch {
            routing: "021000021".into(),
            account: "9999".into(),
        }),
    );
    let outcome = orch.run(&intent).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);
    assert!(outcome.uetr.is_some());
    assert_eq!(gateway.transfer_history().len(), 1);
}

#[test]
fn deterministic_a2a_with_amount_rule_routes_to_pending() {
    let gateway = Arc::new(
        DeterministicA2aGateway::new()
            .with_name("det-a2a")
            .with_amount_ge(
                Money::from_minor(1_000_000, Currency::USD),
                A2aStatus::Pending,
                None,
            ),
    );
    let profile = MerchantBankProfile {
        creditor_agent: ParticipantId::Aba("021000021".into()),
        creditor_account: "1234567890".into(),
        creditor_name: "ACME CORP".into(),
        default_debtor_agent: ParticipantId::Aba("121000248".into()),
        default_debtor_name: "Customer".into(),
    };
    let a2a = Arc::new(A2aAdapter::new("det-a2a", gateway, profile));
    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec![],
        vec!["det-a2a".to_owned()],
    )));
    orch.register_adapter(a2a);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("e2e-big"),
        Money::from_minor(2_000_000, Currency::USD),
        PaymentMethod::A2a(op_core::A2aKey::UsAch {
            routing: "021000021".into(),
            account: "9999".into(),
        }),
    );
    // Pending → orchestrator treats as soft failure. With no
    // fallback rail this surfaces as either `Declined` or
    // `Err(AllRailsExhausted)` depending on the engine config;
    // both are correct "not approved" signals.
    match orch.run(&intent) {
        Ok(outcome) => {
            assert_ne!(outcome.terminal_status, TerminalStatus::Approved);
        }
        Err(op_orchestrator::Error::AllRailsExhausted { .. }) => {}
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn deterministic_crypto_routes_through_orchestrator() {
    let gateway = Arc::new(
        DeterministicCryptoGateway::for_token(StableToken::UsdcBase).with_name("usdc-base"),
    );
    let adapter = Arc::new(CryptoAdapter::new("usdc-base", gateway.clone()));
    let router =
        PolicyRouter::new(vec![], vec![]).with_crypto_drivers(vec!["usdc-base".to_owned()]);
    let mut orch = Orchestrator::new().with_router(Box::new(router));
    orch.register_adapter(adapter);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("crypto-e2e-1"),
        Money::from_minor(50_000, Currency::USD),
        PaymentMethod::Crypto(CryptoAddress::new(
            "base",
            "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
        )),
    );
    let outcome = orch.run(&intent).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);
    assert!(outcome.psp_payment_id.is_some(), "tx_hash expected");
    let history = gateway.transfer_history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].idempotency_key, "crypto-e2e-1");
}

#[test]
fn crypto_rejected_status_declines_terminally() {
    let gateway = Arc::new(
        DeterministicCryptoGateway::for_token(StableToken::UsdcSolana)
            .with_name("usdc-solana")
            .with_key_override("rev", CryptoStatus::Rejected, Some("revert".into())),
    );
    let adapter = Arc::new(CryptoAdapter::new("usdc-solana", gateway));
    let router =
        PolicyRouter::new(vec![], vec![]).with_crypto_drivers(vec!["usdc-solana".to_owned()]);
    let mut orch = Orchestrator::new().with_router(Box::new(router));
    orch.register_adapter(adapter);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("rev"),
        Money::from_minor(100, Currency::USD),
        PaymentMethod::Crypto(CryptoAddress::new(
            "solana",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        )),
    );
    let outcome = orch.run(&intent).unwrap();
    assert_eq!(outcome.terminal_status, TerminalStatus::Declined);
}

#[test]
fn three_ds_challenge_then_resume_completes() {
    // Stand up an orchestrator with a deterministic card acquirer
    // that returns RequiresCustomerAction for a specific
    // idempotency key. Then call resume() with the returned
    // psp_payment_id and confirm the final outcome is Approved.
    let acquirer = Arc::new(DeterministicCardAcquirer::new().with_key_override(
        "3ds-pending",
        AuthStatus::RequiresCustomerAction,
        None,
    ));
    let card = Arc::new(CardAdapter::new("det-card", acquirer.clone()));
    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["det-card".to_owned()],
        vec![],
    )));
    orch.register_adapter(card);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("3ds-pending"),
        Money::from_minor(15_000, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok_v7_3ds")),
    );
    let outcome = orch.run(&intent).unwrap();
    assert_eq!(
        outcome.terminal_status,
        TerminalStatus::RequiresCustomerAction
    );
    let psp_id = outcome.psp_payment_id.expect("psp id on challenge");

    // Customer completes the challenge out-of-band; operator
    // calls back into the orchestrator with the same intent +
    // psp_payment_id.
    let resumed = orch
        .resume(&intent, RailKind::Card, "det-card", &psp_id)
        .unwrap();
    assert_eq!(resumed.terminal_status, TerminalStatus::Approved);
    assert_eq!(resumed.psp_payment_id.as_deref(), Some(psp_id.as_str()));
}

#[test]
fn resume_against_unknown_driver_errors() {
    let acquirer = Arc::new(DeterministicCardAcquirer::new());
    let card = Arc::new(CardAdapter::new("det-card", acquirer));
    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["det-card".to_owned()],
        vec![],
    )));
    orch.register_adapter(card);

    let intent = PaymentIntent::new(
        IdempotencyKey::new("missing"),
        Money::from_minor(100, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok")),
    );
    let err = orch
        .resume(&intent, RailKind::Card, "no-such-driver", "psp-x")
        .unwrap_err();
    assert!(matches!(err, op_orchestrator::Error::NoEligibleRail { .. }));
}

#[test]
fn all_three_deterministic_drivers_pass_conformance() {
    let card = DeterministicCardAcquirer::new();
    let card_report = conformance::run_card(&card);
    assert!(
        card_report.is_clean(),
        "card conformance failed: {:?}",
        card_report.failures
    );

    let a2a = DeterministicA2aGateway::new();
    let a2a_report = conformance::run_a2a(&a2a);
    assert!(
        a2a_report.is_clean(),
        "a2a conformance failed: {:?}",
        a2a_report.failures
    );

    let crypto = DeterministicCryptoGateway::new();
    let crypto_report = conformance::run_crypto(&crypto);
    assert!(
        crypto_report.is_clean(),
        "crypto conformance failed: {:?}",
        crypto_report.failures
    );
}
