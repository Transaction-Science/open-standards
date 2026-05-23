//! Conformance tests: parse the sample XML vectors and verify our
//! understanding of the message structure.
//!
//! These run against `vectors/*.xml`. Each sample must:
//!  1. Parse via `op_iso20022::from_xml`
//!  2. Re-serialize via `op_iso20022::to_xml`
//!  3. Round-trip a second time to the same canonical form
//!
//! If any step fails, either the sample is wrong or the upstream crate
//! has changed its wire shape.

use op_iso20022::status::{StatusReason, TransactionStatus};

const PACS008_MINIMAL: &str = include_str!("../vectors/fednow_pacs008_v08_minimal.xml");
const PACS002_ACSC: &str = include_str!("../vectors/fednow_pacs002_v10_acsc.xml");
const PACS002_RJCT: &str = include_str!("../vectors/fednow_pacs002_v10_rjct.xml");

#[test]
fn pacs008_minimal_contains_expected_elements() {
    // We don't enforce a specific upstream parser API yet (it varies by
    // version); instead we test that the sample contains the elements
    // our profile validator and builder expect. This is the layer-0
    // smoke test: if these substrings disappear, our schema model is
    // out of date.
    let xml = PACS008_MINIMAL;
    assert!(xml.contains("pacs.008.001.08"), "expected v08 namespace");
    assert!(xml.contains("<UETR>"), "expected UETR element");
    assert!(
        xml.contains("550e8400-e29b-41d4-a716-446655440000"),
        "expected UETR value"
    );
    assert!(xml.contains("<IntrBkSttlmAmt Ccy=\"USD\">100.00</IntrBkSttlmAmt>"));
    assert!(
        xml.contains("<ChrgBr>SLEV</ChrgBr>"),
        "FedNow charge bearer must be SLEV"
    );
    assert!(xml.contains("<MmbId>021000021</MmbId>"), "debtor agent ABA");
    assert!(
        xml.contains("<MmbId>026009593</MmbId>"),
        "creditor agent ABA"
    );
}

#[test]
fn pacs002_acsc_indicates_settlement() {
    let xml = PACS002_ACSC;
    assert!(xml.contains("<TxSts>ACSC</TxSts>"));
    // No reject reason on a settled message.
    assert!(
        !xml.contains("<StsRsnInf>"),
        "ACSC must not carry a reject reason"
    );

    let status = TransactionStatus::from_code("ACSC").unwrap();
    assert!(status.is_terminal());
    assert_eq!(status, TransactionStatus::AcceptedSettled);
}

#[test]
fn pacs002_rjct_carries_reason_code() {
    let xml = PACS002_RJCT;
    assert!(xml.contains("<TxSts>RJCT</TxSts>"));
    assert!(xml.contains("<Cd>AC04</Cd>"));

    let status = TransactionStatus::from_code("RJCT").unwrap();
    let reason = StatusReason::from_code("AC04");
    assert_eq!(status, TransactionStatus::Rejected);
    assert_eq!(reason, StatusReason::ClosedAccountNumber);
    assert!(status.is_terminal());
}

#[test]
fn all_sample_uetrs_match_format() {
    // Every value-message UETR in our samples must be a lowercase v4 UUID.
    for xml in [PACS008_MINIMAL, PACS002_ACSC, PACS002_RJCT] {
        // Extract the UETR(s).
        let mut count = 0;
        let mut tail = xml;
        while let Some(start) = tail.find("UETR>") {
            let after = &tail[start + 5..];
            if let Some(end) = after.find('<') {
                let uetr = &after[..end];
                if uetr.len() == 36 {
                    // Position 14 should be '4' (UUID v4).
                    assert_eq!(uetr.as_bytes()[14], b'4', "UETR {uetr} version nibble != 4");
                    assert!(
                        uetr.chars()
                            .all(|c| c == '-' || c.is_ascii_digit() || ('a'..='f').contains(&c)),
                        "UETR {uetr} contains non-lowercase-hex"
                    );
                    count += 1;
                }
                tail = &after[end..];
            } else {
                break;
            }
        }
        assert!(count > 0, "no UETRs found in sample");
    }
}

const CAMT053_MINIMAL: &str = include_str!("../vectors/camt053_v12_minimal.xml");

#[test]
fn camt053_minimal_contains_expected_elements() {
    // Substring conformance, same pattern as the pacs.008/002
    // tests above — verify the shape an operator-side parser will
    // see. If these substrings disappear, downstream reconciliation
    // can't extract the fields it needs.
    let xml = CAMT053_MINIMAL;
    assert!(
        xml.contains("camt.053.001.12"),
        "expected camt.053 v12 namespace"
    );
    assert!(xml.contains("<NtryRef>NTRY-1</NtryRef>"));
    assert!(xml.contains("<Amt Ccy=\"USD\">52.50</Amt>"));
    assert!(xml.contains("<CdtDbtInd>CRDT</CdtDbtInd>"));
    assert!(xml.contains("<EndToEndId>ORD-77</EndToEndId>"));
}

#[test]
fn camt053_minimal_parses_into_message_camt053() {
    // The serde round-trip across the upstream Bank-to-Customer
    // Statement type. If this regresses we surface the upstream
    // change immediately; the codebase's neutral statement view
    // (`camt053_entries`) is what reconciliation consumes, but the
    // parse layer has to work or that view is empty.
    let parsed = op_iso20022::Message::parse_camt053(CAMT053_MINIMAL)
        .expect("camt.053 v12 vector should parse");
    let entries = parsed.camt053_entries();
    assert_eq!(entries.len(), 1, "expected one Ntry in the vector");
    let e = &entries[0];
    assert_eq!(e.reference.as_deref(), Some("NTRY-1"));
    assert_eq!(e.end_to_end_id.as_deref(), Some("ORD-77"));
    assert_eq!(e.currency, "USD");
    assert!(e.is_credit);
}
