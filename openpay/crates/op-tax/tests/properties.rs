//! Property tests for the native calculator.
//!
//! Three invariants we hold across any randomized rate-table + line:
//!
//! 1. **Boundedness** — total tax is ≥ 0 and ≤ the line amount
//!    (since rates we test are between 0 and 100%).
//! 2. **Determinism** — calling `calculate` twice with the same inputs
//!    produces the same output.
//! 3. **Additivity** — a two-line invoice yields the same total tax as
//!    summing the per-line results of two single-line invoices.

use op_core::{Currency, Money};
use op_tax::{
    Jurisdiction, NativeCalculator, ProductTaxCategory, RateTable, TaxCalculator, TaxContext,
    TaxRate, TaxableLine,
};
use proptest::prelude::*;
use rust_decimal::Decimal;

fn strategy_rate() -> impl Strategy<Value = Decimal> {
    // 0..=50% in basis points.
    (0i64..=5000).prop_map(|bp| Decimal::new(bp, 4))
}

fn ctx() -> TaxContext {
    TaxContext::consumer(chrono::NaiveDate::from_ymd_opt(2026, 6, 15).unwrap())
}

fn run_blocking<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    tokio::runtime::Runtime::new().unwrap().block_on(future)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn total_tax_bounded_by_line_amount(
        rate in strategy_rate(),
        amount_minor in 1i64..1_000_000,
    ) {
        let t = RateTable::empty("prop").with(
            Jurisdiction::region("US", "WA"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(rate),
        );
        let calc = NativeCalculator::new(t);
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(amount_minor, Currency::USD),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "WA"),
        };
        let r = run_blocking(calc.calculate(&[line], &ctx())).unwrap();
        prop_assert!(r.total_tax.minor_units >= 0);
        // Tax must not exceed the line amount when rate <= 100%.
        prop_assert!(r.total_tax.minor_units <= amount_minor);
    }

    #[test]
    fn deterministic_repeated(
        rate in strategy_rate(),
        amount_minor in 1i64..1_000_000,
    ) {
        let t = RateTable::empty("prop").with(
            Jurisdiction::region("US", "WA"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(rate),
        );
        let calc = NativeCalculator::new(t);
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(amount_minor, Currency::USD),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "WA"),
        };
        let r1 = run_blocking(calc.calculate(std::slice::from_ref(&line), &ctx())).unwrap();
        let r2 = run_blocking(calc.calculate(&[line], &ctx())).unwrap();
        prop_assert_eq!(r1.total_tax.minor_units, r2.total_tax.minor_units);
    }

    #[test]
    fn additive_across_lines(
        rate in strategy_rate(),
        a in 1i64..500_000,
        b in 1i64..500_000,
    ) {
        let t = RateTable::empty("prop").with(
            Jurisdiction::region("US", "WA"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(rate),
        );
        let calc = NativeCalculator::new(t);
        let line_a = TaxableLine {
            line_id: "A".into(),
            amount: Money::from_minor(a, Currency::USD),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "WA"),
        };
        let line_b = TaxableLine {
            line_id: "B".into(),
            amount: Money::from_minor(b, Currency::USD),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "WA"),
        };
        let single_a =
            run_blocking(calc.calculate(std::slice::from_ref(&line_a), &ctx())).unwrap();
        let single_b =
            run_blocking(calc.calculate(std::slice::from_ref(&line_b), &ctx())).unwrap();
        let combined = run_blocking(calc.calculate(&[line_a, line_b], &ctx())).unwrap();
        // Additivity holds modulo one minor unit per line of rounding —
        // each line is rounded independently in both worlds, so the
        // difference is bounded by the rounding strategy used per line
        // (round-half-up, max 0.5 minor unit per side, 1 minor unit
        // total when both lines round in opposite directions).
        let sum = single_a.total_tax.minor_units + single_b.total_tax.minor_units;
        let diff = (combined.total_tax.minor_units - sum).abs();
        prop_assert!(diff <= 1, "additivity violated: combined={} single_sum={}", combined.total_tax.minor_units, sum);
    }
}
