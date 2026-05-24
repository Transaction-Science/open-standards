//! Worked examples adapted from FASB ASC 606 Implementation Guide
//! (Examples 1, 2, 3, 4 and 5 of section 606-10-55). The numeric
//! amounts are reproduced from the standard; some judgement-only
//! sections (e.g. "the entity concludes that...") are pre-baked into
//! the test inputs.

use chrono::NaiveDate;
use op_core::{Currency, Money};
use op_revrec::{
    Contract, ContractId, Milestone, ObligationId, PerformanceObligation, Presentation,
    RecognitionPattern, TransactionPrice, allocate_transaction_price, generate,
};

fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
}

fn usd(n: i64) -> Money {
    Money::from_minor(n, Currency::USD)
}

/// ASC 606-10-55 Example 1 — Distinct goods or services bundled into a
/// single performance obligation is NOT this example; the example we
/// model here is Example 11: identification of a single performance
/// obligation when an entity sells equipment that requires
/// installation. We model the simpler version: equipment ($1,000) +
/// installation ($200), priced together at $1,100 with a $100 discount.
/// Relative-SSP allocation should split the discount.
#[test]
fn ex11_equipment_plus_installation_allocates_discount() {
    let contract = Contract {
        id: ContractId::new(),
        customer_ref: "acme".into(),
        effective_date: ymd(2026, 1, 1),
        obligations: vec![
            PerformanceObligation {
                id: ObligationId::new("equipment"),
                standalone_selling_price: usd(1_000_00),
                pattern: RecognitionPattern::PointInTime {
                    date: ymd(2026, 1, 15),
                },
                presentation: Presentation::Gross,
            },
            PerformanceObligation {
                id: ObligationId::new("installation"),
                standalone_selling_price: usd(200_00),
                pattern: RecognitionPattern::PointInTime {
                    date: ymd(2026, 1, 20),
                },
                presentation: Presentation::Gross,
            },
        ],
        transaction_price: TransactionPrice::fixed(usd(1_100_00)),
    };
    let alloc = allocate_transaction_price(&contract).expect("alloc");
    // total ssp = 1200; ratios are 1000/1200 and 200/1200; allocations
    // of 1100 = 916.67 and 183.33 — in minor units (cents) and with
    // last-row plug we expect roughly (91666, 18334) summing to 110000.
    let map: std::collections::HashMap<_, _> = alloc.into_iter().collect();
    let eq = map[&ObligationId::new("equipment")];
    let ins = map[&ObligationId::new("installation")];
    assert_eq!(eq + ins, 110_000);
    // Equipment gets ~83.33% of the total: 91_666.
    assert!((eq - 91_666).abs() <= 1);
}

/// ASC 606-10-55 Example 2 — Termination clauses & non-cancellable
/// contracts. We model the recognition pattern: a 12-month subscription
/// at $100/month with no termination right. Straight-line over 12
/// months: each month books $100 with the last entry plugging rounding.
#[test]
fn ex2_twelve_month_subscription_straight_line() {
    let o = PerformanceObligation {
        id: ObligationId::new("sub"),
        standalone_selling_price: usd(1_200_00),
        pattern: RecognitionPattern::StraightLine {
            start: ymd(2026, 1, 1),
            end: ymd(2026, 12, 31),
        },
        presentation: Presentation::Gross,
    };
    let s = generate(&o, 1_200_00, Currency::USD).expect("schedule");
    assert_eq!(s.entries.len(), 12);
    assert_eq!(s.total_minor(), 1_200_00);
}

/// ASC 606-10-55 Example 3 — Implicit price concession. We model the
/// recognition side: a $1,000 receivable where the entity expects a
/// $100 concession. The transaction price is $900 (constrained), not
/// $1,000.
#[test]
fn ex3_implicit_price_concession_uses_constrained_price() {
    use op_revrec::VariableConsideration;
    let mut tp = TransactionPrice::fixed(usd(1_000_00));
    // The $100 concession is a negative variable; constrain to $100.
    tp.variable.push(VariableConsideration::most_likely(
        "implicit_concession",
        -100_00,
        100_00,
    ));
    assert_eq!(tp.total().expect("total"), usd(900_00));
}

/// ASC 606-10-55 Example 4 — Reassessment of variable consideration.
/// We start with an estimate of 90% likelihood of receiving a $200
/// bonus (expected value = $180). Later the entity revises probability
/// to 60% (expected value = $120). The constraint is $200 — i.e. no
/// constraint binds in either case. We assert both estimates produce
/// the expected unconstrained values.
#[test]
fn ex4_variable_consideration_revision() {
    use op_revrec::{EstimationMethod, Outcome, VariableConsideration};
    use rust_decimal::Decimal;
    use std::str::FromStr;

    let v1 = VariableConsideration::expected_value(
        "bonus",
        vec![
            Outcome {
                amount_minor: 20_000,
                probability: Decimal::from_str("0.9").expect("dec"),
            },
            Outcome {
                amount_minor: 0,
                probability: Decimal::from_str("0.1").expect("dec"),
            },
        ],
        20_000,
    );
    assert_eq!(v1.estimate_minor(), 18_000);

    let v2 = VariableConsideration::expected_value(
        "bonus_revised",
        vec![
            Outcome {
                amount_minor: 20_000,
                probability: Decimal::from_str("0.6").expect("dec"),
            },
            Outcome {
                amount_minor: 0,
                probability: Decimal::from_str("0.4").expect("dec"),
            },
        ],
        20_000,
    );
    assert_eq!(v2.estimate_minor(), 12_000);

    // Sanity: confirm the variants we're using actually exist.
    match v2.method {
        EstimationMethod::ExpectedValue { .. } => {}
        EstimationMethod::MostLikelyAmount { .. } => panic!("wrong variant"),
    }
}

/// ASC 606-10-55 Example 5 — The discount allocation method when the
/// observable evidence supports applying the discount to specific
/// obligations only. We model the simpler default: when there's no
/// such evidence, the discount allocates pro-rata, which is what
/// `allocate_transaction_price` does. We assert: a 3-obligation
/// contract with SSPs of $40, $55, $45 (total $140) priced at $100
/// produces allocations 40/140, 55/140, 45/140 of $100.
#[test]
fn ex5_pro_rata_discount_allocation() {
    let contract = Contract {
        id: ContractId::new(),
        customer_ref: "c".into(),
        effective_date: ymd(2026, 1, 1),
        obligations: vec![
            PerformanceObligation {
                id: ObligationId::new("a"),
                standalone_selling_price: usd(40_00),
                pattern: RecognitionPattern::PointInTime { date: ymd(2026, 1, 1) },
                presentation: Presentation::Gross,
            },
            PerformanceObligation {
                id: ObligationId::new("b"),
                standalone_selling_price: usd(55_00),
                pattern: RecognitionPattern::PointInTime { date: ymd(2026, 1, 1) },
                presentation: Presentation::Gross,
            },
            PerformanceObligation {
                id: ObligationId::new("c"),
                standalone_selling_price: usd(45_00),
                pattern: RecognitionPattern::PointInTime { date: ymd(2026, 1, 1) },
                presentation: Presentation::Gross,
            },
        ],
        transaction_price: TransactionPrice::fixed(usd(100_00)),
    };
    let alloc = allocate_transaction_price(&contract).expect("alloc");
    let map: std::collections::HashMap<_, _> = alloc.into_iter().collect();
    // 40/140 * 10000 = 2857 (rounded down)
    // 55/140 * 10000 = 3928
    // 45/140 plug = 10000 - 2857 - 3928 = 3215
    assert_eq!(map[&ObligationId::new("a")], 2857);
    assert_eq!(map[&ObligationId::new("b")], 3928);
    assert_eq!(map[&ObligationId::new("c")], 10_000 - 2857 - 3928);
    let total: i64 = map.values().sum();
    assert_eq!(total, 10_000);
}

/// Output-method milestones (ASC 606-10-55-17). Used by professional-
/// services contracts where customer signs off at each phase.
#[test]
fn output_method_milestones_recognize_at_each_phase() {
    let o = PerformanceObligation {
        id: ObligationId::new("services"),
        standalone_selling_price: usd(100_000_00),
        pattern: RecognitionPattern::OutputMilestones {
            milestones: vec![
                Milestone {
                    label: "design".into(),
                    date: ymd(2026, 2, 28),
                    fraction: "0.20".parse().expect("dec"),
                },
                Milestone {
                    label: "build".into(),
                    date: ymd(2026, 6, 30),
                    fraction: "0.50".parse().expect("dec"),
                },
                Milestone {
                    label: "deploy".into(),
                    date: ymd(2026, 9, 30),
                    fraction: "0.30".parse().expect("dec"),
                },
            ],
        },
        presentation: Presentation::Gross,
    };
    let s = generate(&o, 100_000_00, Currency::USD).expect("schedule");
    assert_eq!(s.entries.len(), 3);
    assert_eq!(s.total_minor(), 100_000_00);
    assert_eq!(s.entries[0].amount_minor, 20_000_00);
    assert_eq!(s.entries[1].amount_minor, 50_000_00);
    assert_eq!(s.entries[2].amount_minor, 30_000_00);
}
