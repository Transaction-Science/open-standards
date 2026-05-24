//! Integration: build a fee schedule mirroring a Stripe-classic
//! 2.9% + 30c card price plus a 0.13% scheme assessment, and assert
//! both lines accrue at the expected magnitudes.

use op_core::{Currency, Money};
use op_statements::{FeeBucket, FeeRule, FeeSchedule};

#[test]
fn stripe_classic_plus_scheme_accrual() {
    let sched = FeeSchedule::new(Currency::USD)
        .with_rule(
            FeeRule::new(FeeBucket::Acquirer, 290, 30) // 2.90% + 30c
                .with_code("stripe-standard"),
        )
        .with_rule(
            FeeRule::new(FeeBucket::Scheme, 13, 0) // 0.13%
                .with_code("visa-fanf"),
        )
        .with_rule(
            FeeRule::new(FeeBucket::Interchange, 165, 10) // 1.65% + 10c — Visa CPS retail
                .with_code("cps-retail"),
        );

    let fees = sched
        .accrue(Money::from_minor(10_000, Currency::USD), Some("ord-1"))
        .expect("accrue");
    assert_eq!(fees.len(), 3);

    // Acquirer: 10000 * 290 / 10000 + 30 = 290 + 30 = 320
    assert_eq!(fees[0].bucket, FeeBucket::Acquirer);
    assert_eq!(fees[0].amount.minor_units, 320);
    assert_eq!(fees[0].code.as_deref(), Some("stripe-standard"));
    assert_eq!(fees[0].against_external_id.as_deref(), Some("ord-1"));

    // Scheme: 10000 * 13 / 10000 = 13
    assert_eq!(fees[1].bucket, FeeBucket::Scheme);
    assert_eq!(fees[1].amount.minor_units, 13);

    // Interchange: 10000 * 165 / 10000 + 10 = 165 + 10 = 175
    assert_eq!(fees[2].bucket, FeeBucket::Interchange);
    assert_eq!(fees[2].amount.minor_units, 175);

    let grouped = FeeSchedule::group_by_bucket(&fees);
    assert_eq!(grouped.len(), 3);
    let total_minor: i64 = grouped.iter().map(|(_, m)| *m).sum();
    assert_eq!(total_minor, 320 + 13 + 175);
}

#[test]
fn jpy_zero_decimal_accrual() {
    // JPY has 0 decimal places; flat amounts are whole yen.
    let sched = FeeSchedule::new(Currency::JPY).with_rule(FeeRule::new(FeeBucket::Acquirer, 290, 50));
    let fees = sched
        .accrue(Money::from_minor(10_000, Currency::JPY), None)
        .expect("accrue");
    // 10000 * 290 / 10000 + 50 = 290 + 50 = 340 yen
    assert_eq!(fees.len(), 1);
    assert_eq!(fees[0].amount.minor_units, 340);
}
