//! Integration: build a multi-line statement and render to CSV.

use op_core::{Currency, Money};
use op_statements::{
    Cadence, Csv, FeeBucket, FeeLine, RenderTarget, Statement, StatementLine, StatementLineKind,
};

#[test]
fn csv_statement_renders_three_lines() {
    let periods = Cadence::Daily.enumerate(0, 0, 86_399).expect("periods");
    let period = periods.into_iter().next().expect("at least one period");

    let mut s = Statement::new("STMT-CSV-1", "MERCHANT-42", period, Currency::USD)
        .expect("statement")
        .with_opening(Money::from_minor(2_500, Currency::USD))
        .expect("opening");

    s.push_line(StatementLine::new(
        "ln-1",
        StatementLineKind::GrossCapture,
        Money::from_minor(10_000, Currency::USD),
        100,
    ))
    .expect("push 1");

    s.push_line(
        StatementLine::new(
            "ln-2",
            StatementLineKind::Fee,
            Money::from_minor(320, Currency::USD),
            101,
        )
        .with_fee(FeeLine::new(
            FeeBucket::Acquirer,
            Money::from_minor(320, Currency::USD),
        ))
        .with_external_id("ord-1"),
    )
    .expect("push 2");

    s.push_line(StatementLine::new(
        "ln-3",
        StatementLineKind::Refund,
        Money::from_minor(500, Currency::USD),
        102,
    ))
    .expect("push 3");

    s.aggregate().expect("aggregate");

    let out = Csv.render(&s).expect("render");
    assert!(out.starts_with("line_id,kind,currency,amount_minor"));
    assert!(out.contains("ln-1,gross_capture,USD,10000,100,"));
    assert!(out.contains("ln-2,fee,USD,320,101,ord-1"));
    assert!(out.contains("ln-3,refund,USD,500,102,"));

    let primary = s.primary_aggregate();
    assert_eq!(primary.opening.minor_units, 2_500);
    assert_eq!(primary.gross_volume.minor_units, 10_000);
    assert_eq!(primary.refunds.minor_units, 500);
    assert_eq!(primary.fees.minor_units, 320);
    // 2500 + 10000 - 500 - 320 = 11680
    assert_eq!(primary.ending.minor_units, 11_680);
}
