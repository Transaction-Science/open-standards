//! Pure-function conversion of [`Money`] across currencies.
//!
//! Math is integer-exact:
//!
//! ```text
//!   target_minor = round(source_minor × rate_ppm / 1_000_000)
//! ```
//!
//! using a caller-chosen [`RoundingMode`] for the remainder. We
//! use `i128` for the intermediate so even an `i64::MAX` minor
//! amount × a large `rate_ppm` doesn't overflow before we clamp
//! back to `i64`.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::quote::Quote;

/// How to handle the integer remainder when
/// `source_minor × rate_ppm` is not exactly divisible by
/// `1_000_000`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoundingMode {
    /// Banker's rounding (half-to-even). The default for accounting
    /// systems — eliminates the systematic bias of always-up
    /// rounding without statistically advantaging either side.
    #[default]
    HalfEven,
    /// Truncate toward zero. Biased toward the **operator** when
    /// converting *to* an operator-receiving currency; biased
    /// toward the **customer** when converting away.
    Down,
    /// Round away from zero. Biased toward the **customer** in the
    /// same scenarios.
    Up,
}

/// Apply `quote` to `money`, producing the equivalent amount in
/// `quote.target_currency`.
///
/// # Errors
/// - [`Error::CurrencyMismatch`] if `money.currency != quote.source_currency`.
/// - [`Error::SameCurrency`] if source and target match (caller
///   should skip conversion).
/// - [`Error::InvalidRate`] if `quote.rate_ppm == 0`.
/// - [`Error::QuoteExpired`] if `quote` is past its validity.
/// - [`Error::Overflow`] only on pathological inputs.
pub fn convert(
    money: Money,
    quote: &Quote,
    mode: RoundingMode,
    now_unix_secs: u64,
) -> Result<Money> {
    if quote.rate_ppm == 0 {
        return Err(Error::InvalidRate);
    }
    if quote.source_currency == quote.target_currency {
        return Err(Error::SameCurrency(quote.source_currency.code().to_owned()));
    }
    if money.currency != quote.source_currency {
        return Err(Error::CurrencyMismatch {
            money: money.currency.code().to_owned(),
            quote_source: quote.source_currency.code().to_owned(),
        });
    }
    if !quote.is_valid_at(now_unix_secs) {
        return Err(Error::QuoteExpired {
            from_currency: quote.source_currency.code().to_owned(),
            to_currency: quote.target_currency.code().to_owned(),
            valid_until: quote.valid_until_unix_secs,
            now: now_unix_secs,
        });
    }
    let scaled = i128::from(money.minor_units) * i128::from(quote.rate_ppm);
    let divisor: i128 = 1_000_000;
    let quotient = scaled / divisor;
    let remainder = scaled % divisor;
    let adjusted = apply_rounding(quotient, remainder, divisor, mode);
    let minor = i64::try_from(adjusted).map_err(|_| Error::Overflow)?;
    Ok(Money::from_minor(minor, quote.target_currency))
}

fn apply_rounding(quotient: i128, remainder: i128, divisor: i128, mode: RoundingMode) -> i128 {
    if remainder == 0 {
        return quotient;
    }
    let sign: i128 = if (quotient < 0) || (quotient == 0 && remainder < 0) {
        -1
    } else {
        1
    };
    let abs_rem = remainder.abs();
    let half = divisor / 2;
    match mode {
        RoundingMode::Down => quotient,
        RoundingMode::Up => quotient + sign,
        RoundingMode::HalfEven => match abs_rem.cmp(&half) {
            std::cmp::Ordering::Less => quotient,
            std::cmp::Ordering::Greater => quotient + sign,
            std::cmp::Ordering::Equal => {
                // Tie → round to nearest even quotient.
                if quotient % 2 == 0 {
                    quotient
                } else {
                    quotient + sign
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    fn quote(rate_ppm: u64) -> Quote {
        Quote::new(Currency::EUR, Currency::USD, rate_ppm, 0, u64::MAX, "test")
    }

    #[test]
    fn exact_conversion_no_rounding_needed() {
        // €100.00 at 1.000000 = $100.00.
        let m = Money::from_minor(10_000, Currency::EUR);
        let q = quote(1_000_000);
        let r = convert(m, &q, RoundingMode::HalfEven, 0).unwrap();
        assert_eq!(r, Money::from_minor(10_000, Currency::USD));
    }

    #[test]
    fn standard_eur_usd_conversion() {
        // €100.00 at 1.082500 = $108.25 exactly.
        let m = Money::from_minor(10_000, Currency::EUR);
        let q = quote(1_082_500);
        let r = convert(m, &q, RoundingMode::HalfEven, 0).unwrap();
        assert_eq!(r, Money::from_minor(10_825, Currency::USD));
    }

    #[test]
    fn half_even_rounds_to_nearest_even() {
        // 1 cent × 1.5 ppm-equivalent = halfway between 1 and 2 minor units.
        // Setup: 1_500_000 ppm × 1 minor = 1_500_000 / 1_000_000 = 1.5
        // Half-even on quotient=1, rem=500_000, divisor=1_000_000 → tie → nearest even → 2.
        let m = Money::from_minor(1, Currency::EUR);
        let q = quote(1_500_000);
        let r = convert(m, &q, RoundingMode::HalfEven, 0).unwrap();
        assert_eq!(r.minor_units, 2);
        // Quotient=2, rem=500_000 → tie → nearest even → 2.
        let m = Money::from_minor(2, Currency::EUR); // 2 * 1.5 = 3.0 exactly
        let r = convert(m, &q, RoundingMode::HalfEven, 0).unwrap();
        assert_eq!(r.minor_units, 3);
        // 3 × 1.5 = 4.5 → tie → nearest even → 4.
        let m = Money::from_minor(3, Currency::EUR);
        let r = convert(m, &q, RoundingMode::HalfEven, 0).unwrap();
        assert_eq!(r.minor_units, 4);
    }

    #[test]
    fn down_truncates_toward_zero() {
        let m = Money::from_minor(7, Currency::EUR);
        let q = quote(1_500_000); // 7 * 1.5 = 10.5
        let r = convert(m, &q, RoundingMode::Down, 0).unwrap();
        assert_eq!(r.minor_units, 10);
    }

    #[test]
    fn up_rounds_away_from_zero() {
        let m = Money::from_minor(7, Currency::EUR);
        let q = quote(1_500_000); // 7 * 1.5 = 10.5
        let r = convert(m, &q, RoundingMode::Up, 0).unwrap();
        assert_eq!(r.minor_units, 11);
    }

    #[test]
    fn rejects_same_currency() {
        let m = Money::from_minor(100, Currency::EUR);
        let q = Quote::new(Currency::EUR, Currency::EUR, 1_000_000, 0, u64::MAX, "test");
        assert!(matches!(
            convert(m, &q, RoundingMode::HalfEven, 0),
            Err(Error::SameCurrency(_))
        ));
    }

    #[test]
    fn rejects_currency_mismatch() {
        let m = Money::from_minor(100, Currency::USD);
        let q = quote(1_082_500); // EUR/USD
        assert!(matches!(
            convert(m, &q, RoundingMode::HalfEven, 0),
            Err(Error::CurrencyMismatch { .. })
        ));
    }

    #[test]
    fn rejects_zero_rate() {
        let m = Money::from_minor(100, Currency::EUR);
        let q = quote(0);
        assert!(matches!(
            convert(m, &q, RoundingMode::HalfEven, 0),
            Err(Error::InvalidRate)
        ));
    }

    #[test]
    fn rejects_expired_quote() {
        let m = Money::from_minor(100, Currency::EUR);
        let q = Quote::new(Currency::EUR, Currency::USD, 1_082_500, 0, 1_000, "test");
        assert!(matches!(
            convert(m, &q, RoundingMode::HalfEven, 1_001),
            Err(Error::QuoteExpired { .. })
        ));
    }

    #[test]
    fn negative_amount_converts_correctly() {
        // -€100.00 at 1.082500 = -$108.25.
        let m = Money::from_minor(-10_000, Currency::EUR);
        let q = quote(1_082_500);
        let r = convert(m, &q, RoundingMode::HalfEven, 0).unwrap();
        assert_eq!(r, Money::from_minor(-10_825, Currency::USD));
    }
}
