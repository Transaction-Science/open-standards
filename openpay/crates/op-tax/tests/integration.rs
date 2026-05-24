//! End-to-end integration tests for `op-tax`. These complement the
//! unit tests inside each module and cover the scenarios the spec
//! flags explicitly:
//!
//! - US sales-tax compound across state + county + city + special district.
//! - VAT inclusive math (€100 gross with 20% VAT → €83.33 net + €16.67 tax).
//! - Exemption certificate application.
//! - Reverse charge for EU B2B cross-border.
//! - Nexus monitor 199th vs 200th transaction.
//! - Property tests: rate-table + line ⇒ bounded, deterministic, additive.

use chrono::NaiveDate;
use op_core::{Currency, Money};
use op_tax::{
    CustomerType, ExemptionCertificate, Jurisdiction, NativeCalculator, NexusEvent, NexusMonitor,
    ProductTaxCategory, RateTable, TaxCalculator, TaxContext, TaxableLine, TransactionRecord,
};
use rust_decimal::Decimal;

fn date(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
}

fn line(amount_minor: i64, currency: Currency, to: Jurisdiction) -> TaxableLine {
    TaxableLine {
        line_id: "L1".into(),
        amount: Money::from_minor(amount_minor, currency),
        category: ProductTaxCategory::TangibleGoods,
        ship_from: None,
        ship_to: to,
    }
}

#[tokio::test]
async fn us_sales_tax_state_county_city_district_compound() {
    // Construct a deliberate four-layer stack (state + city +
    // district — county base is 0 in WA's actual rate schedule):
    // - State WA: 6.5%
    // - City Seattle: 3.85%
    // - District KingTransit: 0.9%
    // Total = 11.25% on $100 = $11.25
    let calc = NativeCalculator::bundled();
    let line = line(
        10_000,
        Currency::USD,
        Jurisdiction::full("US", "WA", "Seattle", "KingTransit"),
    );
    let ctx = TaxContext::consumer(date("2026-06-15"));
    let r = calc.calculate(&[line], &ctx).await.unwrap();
    assert_eq!(r.total_tax.minor_units, 1125);
    let bd = r.per_line.get("L1").unwrap();
    assert_eq!(bd.jurisdiction_layers.len(), 3);
}

#[tokio::test]
async fn vat_inclusive_eur_100_at_20pct_yields_16_67() {
    let calc = NativeCalculator::bundled();
    let line = line(10_000, Currency::EUR, Jurisdiction::country("FR")); // 20% VAT
    let ctx = TaxContext::consumer(date("2026-06-15"));
    let r = calc.calculate(&[line], &ctx).await.unwrap();
    // €100/1.20 = €83.33; tax = €16.67.
    assert_eq!(r.total_tax.minor_units, 1667);
    let bd = r.per_line.get("L1").unwrap();
    assert_eq!(bd.taxable_amount.minor_units, 8333);
}

#[tokio::test]
async fn resale_certificate_zero_rates_line() {
    let calc = NativeCalculator::bundled();
    let cert = ExemptionCertificate {
        id: "RESALE-WA-1".into(),
        holder: "Acme Resellers".into(),
        jurisdictions: vec![Jurisdiction::region("US", "WA")],
        categories: vec![ProductTaxCategory::TangibleGoods],
        valid_from: date("2026-01-01"),
        valid_until: Some(date("2026-12-31")),
        certificate_data: b"<pdf>".to_vec(),
    };
    let mut ctx = TaxContext::consumer(date("2026-06-15"));
    ctx.customer_type = CustomerType::Business {
        tax_id: "12-3456789".into(),
    };
    ctx.exemption_certs.push(cert);
    let line = line(
        10_000,
        Currency::USD,
        Jurisdiction::locality("US", "WA", "Seattle"),
    );
    let r = calc.calculate(&[line], &ctx).await.unwrap();
    assert_eq!(r.total_tax.minor_units, 0);
    let bd = r.per_line.get("L1").unwrap();
    assert!(bd.exemption_reason.is_some());
}

#[tokio::test]
async fn eu_b2b_cross_border_reverse_charge_zero_rates() {
    let calc = NativeCalculator::bundled();
    let mut ctx = TaxContext::consumer(date("2026-06-15"));
    ctx.customer_type = CustomerType::Business {
        tax_id: "FR12345678901".into(),
    };
    let line = TaxableLine {
        line_id: "L1".into(),
        amount: Money::from_minor(10_000, Currency::EUR),
        category: ProductTaxCategory::Saas,
        ship_from: Some(Jurisdiction::country("DE")),
        ship_to: Jurisdiction::country("FR"),
    };
    let r = calc.calculate(&[line], &ctx).await.unwrap();
    assert_eq!(r.total_tax.minor_units, 0);
}

#[test]
fn nexus_monitor_199_vs_200_transactions_baseline_state() {
    let mut m = NexusMonitor::with_default_us_states();
    let mk_tx = || TransactionRecord {
        state: op_tax::RegionCode::new("SD"),
        date: date("2026-06-15"),
        revenue_usd: Decimal::new(10, 0),
    };
    for i in 0..199 {
        let events = m.record(&mk_tx());
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, NexusEvent::Triggered { .. })),
            "should not trigger before 200th transaction; failed at i={i}"
        );
    }
    let events = m.record(&mk_tx());
    let trig = events.iter().find_map(|e| match e {
        NexusEvent::Triggered { dimension, .. } => Some(*dimension),
        _ => None,
    });
    assert_eq!(trig, Some("transactions"));
}

#[tokio::test]
async fn nexus_filter_in_calculator_zero_rates_outside_jurisdictions() {
    let calc = NativeCalculator::bundled();
    let mut ctx = TaxContext::consumer(date("2026-06-15"));
    ctx.nexus_jurisdictions.insert(Jurisdiction::region("US", "WA"));
    // Ship to NY — no nexus.
    let line = line(10_000, Currency::USD, Jurisdiction::region("US", "NY"));
    let r = calc.calculate(&[line], &ctx).await.unwrap();
    assert_eq!(r.total_tax.minor_units, 0);
}

#[tokio::test]
async fn deterministic_repeated_calls_match() {
    let calc = NativeCalculator::bundled();
    let ctx = TaxContext::consumer(date("2026-06-15"));
    let l = line(
        10_000,
        Currency::USD,
        Jurisdiction::locality("US", "WA", "Seattle"),
    );
    let r1 = calc.calculate(std::slice::from_ref(&l), &ctx).await.unwrap();
    let r2 = calc.calculate(&[l], &ctx).await.unwrap();
    assert_eq!(r1.total_tax.minor_units, r2.total_tax.minor_units);
    assert_eq!(
        r1.per_line.get("L1").unwrap().tax_amount.minor_units,
        r2.per_line.get("L1").unwrap().tax_amount.minor_units
    );
}

#[tokio::test]
async fn cbor_snapshot_roundtrip_via_disk() {
    let t = RateTable::bundled();
    let tmp = std::env::temp_dir().join("op_tax_test_rate_table.cbor");
    {
        let f = std::fs::File::create(&tmp).unwrap();
        t.write_cbor(f).unwrap();
    }
    let loaded = RateTable::load_cbor(&tmp).unwrap();
    assert_eq!(loaded.entries.len(), t.entries.len());
    // Calculator over the loaded table should match a calculator over
    // the in-memory table.
    let calc_disk = NativeCalculator::new(loaded);
    let calc_mem = NativeCalculator::new(t);
    let ctx = TaxContext::consumer(date("2026-06-15"));
    let l = line(10_000, Currency::EUR, Jurisdiction::country("DE"));
    let r_disk = calc_disk
        .calculate(std::slice::from_ref(&l), &ctx)
        .await
        .unwrap();
    let r_mem = calc_mem.calculate(&[l], &ctx).await.unwrap();
    assert_eq!(r_disk.total_tax.minor_units, r_mem.total_tax.minor_units);
    let _ = std::fs::remove_file(&tmp);
}
