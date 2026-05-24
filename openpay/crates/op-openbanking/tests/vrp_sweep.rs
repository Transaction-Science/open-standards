//! VRP sweep + non-sweep cap-enforcement integration test.

use op_core::{Currency, Money};
use op_openbanking::aisp::ConsentId;
use op_openbanking::vrp::{
    VrpConsent, VrpControlParameters, VrpExecution, VrpKind, VrpSweep, VrpWindow,
};
use op_openbanking::Error;

fn consent(kind: VrpKind, per: i64, window: i64) -> VrpConsent {
    VrpConsent {
        id: ConsentId("vrp".into()),
        kind,
        debtor_account: "GB29NWBK60161331926819".into(),
        creditor_account: "GB33BUKB20201555555555".into(),
        controls: VrpControlParameters {
            max_individual_amount: Money::from_minor(per, Currency::GBP),
            max_period_amount: Money::from_minor(window, Currency::GBP),
            window: VrpWindow::Month,
            valid_until: None,
        },
    }
}

fn exec(amount: i64) -> VrpExecution {
    VrpExecution {
        amount: Money::from_minor(amount, Currency::GBP),
        end_to_end_id: format!("VRP-{}", amount),
        remittance: None,
        submitted_at: time::OffsetDateTime::UNIX_EPOCH,
    }
}

#[test]
fn sweep_within_caps_passes() {
    let sweep = VrpSweep {
        consent: consent(VrpKind::Sweeping, 10_000, 50_000),
        execution: exec(5_000),
        already_spent_in_window: Money::from_minor(20_000, Currency::GBP),
    };
    sweep.check_caps().expect("ok");
}

#[test]
fn non_sweep_per_payment_cap_fires() {
    let sweep = VrpSweep {
        consent: consent(VrpKind::NonSweeping, 10_000, 50_000),
        execution: exec(15_000),
        already_spent_in_window: Money::zero(Currency::GBP),
    };
    assert!(matches!(
        sweep.check_caps().unwrap_err(),
        Error::VrpLimitExceeded { .. }
    ));
}

#[test]
fn period_cap_fires_on_aggregation() {
    let sweep = VrpSweep {
        consent: consent(VrpKind::Sweeping, 10_000, 50_000),
        execution: exec(8_000),
        already_spent_in_window: Money::from_minor(45_000, Currency::GBP),
    };
    let err = sweep.check_caps().expect_err("period cap");
    let Error::VrpLimitExceeded { reason } = err else {
        panic!("wrong variant");
    };
    assert!(reason.contains("period"));
}

#[test]
fn currency_mismatch_caught_locally() {
    let mut sweep = VrpSweep {
        consent: consent(VrpKind::Sweeping, 10_000, 50_000),
        execution: exec(1_000),
        already_spent_in_window: Money::zero(Currency::GBP),
    };
    sweep.execution.amount = Money::from_minor(1_000, Currency::EUR);
    assert!(matches!(
        sweep.check_caps().unwrap_err(),
        Error::CurrencyMismatch(_)
    ));
}

#[test]
fn expired_consent_rejected() {
    let mut c = consent(VrpKind::Sweeping, 10_000, 50_000);
    c.controls.valid_until = Some(time::OffsetDateTime::UNIX_EPOCH);
    let sweep = VrpSweep {
        consent: c,
        execution: VrpExecution {
            amount: Money::from_minor(1_000, Currency::GBP),
            end_to_end_id: "E".into(),
            remittance: None,
            // Submitted strictly *after* the consent's valid_until.
            submitted_at: time::OffsetDateTime::UNIX_EPOCH + time::Duration::days(1),
        },
        already_spent_in_window: Money::zero(Currency::GBP),
    };
    assert!(matches!(
        sweep.check_caps().unwrap_err(),
        Error::ConsentStateInvalid { .. }
    ));
}
