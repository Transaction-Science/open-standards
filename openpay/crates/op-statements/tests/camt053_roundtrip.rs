//! Integration: build a camt.053 XML envelope from a statement and
//! assert the structural elements every conformant parser keys on.

use op_core::{Currency, Money};
use op_statements::{
    Camt053Builder, Period, Statement, StatementLine, StatementLineKind,
};

#[test]
fn camt053_envelope_has_all_required_elements() {
    let mut s = Statement::new(
        "STMT-CAMT-1",
        "ACCT-LOREM",
        Period::new(1_700_000_000, 1_700_086_399).expect("period"),
        Currency::EUR,
    )
    .expect("statement")
    .with_opening(Money::from_minor(50_000, Currency::EUR))
    .expect("opening");

    s.push_line(
        StatementLine::new(
            "ln-1",
            StatementLineKind::GrossCapture,
            Money::from_minor(12_345, Currency::EUR),
            1_700_001_000,
        )
        .with_external_id("E2E-REF-1"),
    )
    .expect("push 1");

    s.push_line(StatementLine::new(
        "ln-2",
        StatementLineKind::Fee,
        Money::from_minor(290, Currency::EUR),
        1_700_001_500,
    ))
    .expect("push 2");

    s.aggregate().expect("aggregate");

    let xml = Camt053Builder.build(&s).expect("build");

    // Envelope
    assert!(xml.contains("<?xml version=\"1.0\""));
    assert!(xml.contains("urn:iso:std:iso:20022:tech:xsd:camt.053.001.13"));
    assert!(xml.contains("<BkToCstmrStmt>"));
    assert!(xml.contains("</BkToCstmrStmt>"));

    // Group header
    assert!(xml.contains("<MsgId>STMT-CAMT-1</MsgId>"));

    // Stmt
    assert!(xml.contains("<Id>STMT-CAMT-1</Id>"));
    assert!(xml.contains("<Othr>"));
    assert!(xml.contains("<Id>ACCT-LOREM</Id>"));
    assert!(xml.contains("<Ccy>EUR</Ccy>"));

    // Balances
    assert!(xml.contains("<Cd>OPBD</Cd>"));
    assert!(xml.contains("<Cd>CLBD</Cd>"));

    // Entries
    assert!(xml.contains("<Ntry>"));
    assert!(xml.contains("<NtryRef>E2E-REF-1</NtryRef>"));
    assert!(xml.contains("<Amt Ccy=\"EUR\">123.45</Amt>"));
    assert!(xml.contains("<CdtDbtInd>CRDT</CdtDbtInd>"));
    assert!(xml.contains("<CdtDbtInd>DBIT</CdtDbtInd>"));
    assert!(xml.contains("<Sts><Cd>BOOK</Cd></Sts>"));
}
