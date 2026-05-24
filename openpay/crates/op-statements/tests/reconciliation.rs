//! Integration: end-to-end reconciliation of a statement against a
//! ledger-side view containing one match, one ledger-only, and one
//! statement-only entry.

use op_core::{Currency, Money};
use op_statements::{
    LedgerEntry, Period, Reconciler, Statement, StatementLine, StatementLineKind,
};
use op_statements::reconcile::{LedgerDirection, ReconStatus};

#[test]
fn three_way_reconciliation() {
    let mut s = Statement::new(
        "STMT-RECON-1",
        "MERCHANT-RECON",
        Period::new(0, 86_400).expect("period"),
        Currency::USD,
    )
    .expect("statement");

    // Line 1: matches ledger via external_id.
    s.push_line(
        StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(2_500, Currency::USD),
            1_000,
        )
        .with_external_id("ord-1"),
    )
    .expect("push 1");

    // Line 2: present on statement only — bank double-posted or the
    // ledger never recorded it.
    s.push_line(StatementLine::new(
        "l2",
        StatementLineKind::Fee,
        Money::from_minor(50, Currency::USD),
        2_000,
    ))
    .expect("push 2");

    s.aggregate().expect("aggregate");

    let ledger = vec![
        // Counter-match for l1.
        LedgerEntry {
            id: "tx-1".into(),
            external_id: Some("ord-1".into()),
            amount: Money::from_minor(2_500, Currency::USD),
            direction: LedgerDirection::Credit,
            posted_at_unix_secs: 1_010,
        },
        // Ledger-only entry: posted internally, not yet on the bank
        // statement.
        LedgerEntry {
            id: "tx-2".into(),
            external_id: Some("ord-2".into()),
            amount: Money::from_minor(7_777, Currency::USD),
            direction: LedgerDirection::Credit,
            posted_at_unix_secs: 3_000,
        },
    ];

    let records = Reconciler::with_window(86_400).reconcile(&s, &ledger).expect("reconcile");
    assert_eq!(records.len(), 3);

    // Find each by status.
    let matched = records.iter().filter(|r| r.status == ReconStatus::Matched).count();
    let missing_ledger = records
        .iter()
        .filter(|r| r.status == ReconStatus::MissingInLedger)
        .count();
    let missing_stmt = records
        .iter()
        .filter(|r| r.status == ReconStatus::MissingOnStatement)
        .count();
    assert_eq!(matched, 1);
    assert_eq!(missing_ledger, 1);
    assert_eq!(missing_stmt, 1);

    // The l1 match is zero-delta.
    let m = records
        .iter()
        .find(|r| r.statement_line_id.as_deref() == Some("l1"))
        .expect("l1 record");
    assert_eq!(m.delta_minor_units, 0);

    // The l2 missing-in-ledger carries the line magnitude as delta.
    let ml = records
        .iter()
        .find(|r| r.statement_line_id.as_deref() == Some("l2"))
        .expect("l2 record");
    assert_eq!(ml.delta_minor_units, 50);

    // The tx-2 missing-on-statement carries a negative delta.
    let ms = records
        .iter()
        .find(|r| r.ledger_entry_id.as_deref() == Some("tx-2"))
        .expect("tx-2 record");
    assert_eq!(ms.delta_minor_units, -7_777);
}
