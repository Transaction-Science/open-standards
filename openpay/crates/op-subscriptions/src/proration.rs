//! Proration math.
//!
//! When a customer switches plans mid-cycle, operators typically
//! credit the unused portion of the current plan against the new
//! plan's first charge. This module exposes a pure-function
//! integer-exact prorater — no floating point, no rounding
//! surprises, every minor unit accounted for.

use op_core::Money;

use crate::error::{Error, Result};
use crate::subscription::Subscription;

/// Credit due back to the customer for the unused portion of
/// `sub`'s current period, computed at `now_unix_secs`.
///
/// Math (all integer): `plan_amount × seconds_remaining /
/// period_length_seconds`, with the integer division biased toward
/// the operator on the remainder (no over-crediting).
///
/// # Errors
/// [`Error::Invalid`] if the period bounds are inverted or
/// `now_unix_secs` is outside the period.
pub fn credit_remaining(sub: &Subscription, now_unix_secs: u64) -> Result<Money> {
    let start = sub.current_period_start_unix_secs;
    let end = sub.current_period_end_unix_secs;
    if end <= start {
        return Err(Error::Invalid(format!(
            "period bounds inverted: start={start} end={end}"
        )));
    }
    if now_unix_secs < start {
        return Err(Error::Invalid(format!(
            "now ({now_unix_secs}) before period start ({start})"
        )));
    }
    if now_unix_secs >= end {
        return Ok(Money::from_minor(0, sub.plan.amount.currency));
    }
    let period_len = end - start;
    let remaining = end - now_unix_secs;
    let amount = i128::from(sub.plan.amount.minor_units);
    let scaled = amount * i128::from(remaining) / i128::from(period_len);
    let credit = i64::try_from(scaled).map_err(|_| Error::Invalid("proration overflow".into()))?;
    Ok(Money::from_minor(credit, sub.plan.amount.currency))
}

/// Charge due for switching mid-cycle: gross new-plan amount
/// minus credit on the old plan. Both must be in the same
/// currency.
///
/// Returns `Money` clamped at zero — never produces a negative
/// charge (operators handle "we owe the customer money" via a
/// refund flow rather than a negative subscription charge).
///
/// # Errors
/// [`Error::CurrencyMismatch`] if new plan and credit currencies
/// disagree.
pub fn switch_charge(new_plan_amount: Money, credit: Money) -> Result<Money> {
    if new_plan_amount.currency != credit.currency {
        return Err(Error::CurrencyMismatch {
            a: new_plan_amount.currency.code().to_owned(),
            b: credit.currency.code().to_owned(),
        });
    }
    let net = new_plan_amount.minor_units - credit.minor_units;
    Ok(Money::from_minor(net.max(0), new_plan_amount.currency))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Interval, Plan};
    use crate::subscription::Subscription;
    use op_core::{Currency, Money, PaymentMethod, VaultRef};

    fn sub_30_day() -> Subscription {
        let p = Plan::new(
            "p",
            Money::from_minor(30_000, Currency::USD), // $300 / 30-day cycle
            Interval::Day,
            30,
        )
        .unwrap();
        Subscription::new(
            "c",
            p,
            PaymentMethod::Vault(VaultRef::new("tok")),
            1_700_000_000,
        )
        .unwrap()
    }

    #[test]
    fn full_period_remaining_credits_full_amount() {
        let s = sub_30_day();
        let c = credit_remaining(&s, s.current_period_start_unix_secs).unwrap();
        // Floor division on the boundary: period_len seconds remain,
        // so credit equals plan amount.
        assert_eq!(c, Money::from_minor(30_000, Currency::USD));
    }

    #[test]
    fn half_period_remaining_credits_half() {
        let s = sub_30_day();
        let halfway = s.current_period_start_unix_secs
            + (s.current_period_end_unix_secs - s.current_period_start_unix_secs) / 2;
        let c = credit_remaining(&s, halfway).unwrap();
        assert_eq!(c, Money::from_minor(15_000, Currency::USD));
    }

    #[test]
    fn period_ended_credits_zero() {
        let s = sub_30_day();
        let after = s.current_period_end_unix_secs + 10;
        let c = credit_remaining(&s, after).unwrap();
        assert_eq!(c, Money::from_minor(0, Currency::USD));
    }

    #[test]
    fn before_period_start_errors() {
        let s = sub_30_day();
        assert!(credit_remaining(&s, s.current_period_start_unix_secs - 1).is_err());
    }

    #[test]
    fn switch_charge_subtracts_credit() {
        let new = Money::from_minor(50_000, Currency::USD);
        let credit = Money::from_minor(15_000, Currency::USD);
        let net = switch_charge(new, credit).unwrap();
        assert_eq!(net, Money::from_minor(35_000, Currency::USD));
    }

    #[test]
    fn switch_charge_clamps_at_zero() {
        let new = Money::from_minor(10_000, Currency::USD);
        let credit = Money::from_minor(30_000, Currency::USD);
        let net = switch_charge(new, credit).unwrap();
        assert_eq!(net, Money::from_minor(0, Currency::USD));
    }

    #[test]
    fn switch_charge_currency_mismatch() {
        let new = Money::from_minor(10_000, Currency::USD);
        let credit = Money::from_minor(10_000, Currency::EUR);
        assert!(matches!(
            switch_charge(new, credit),
            Err(Error::CurrencyMismatch { .. })
        ));
    }
}
