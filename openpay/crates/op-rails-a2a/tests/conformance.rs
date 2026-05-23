//! Cross-driver lifecycle / conformance tests.
//!
//! These tests load the XML conformance vectors from `vectors/` and
//! verify that:
//! 1. The pacs.002 parser extracts the documented fields from each vector.
//! 2. The status mappers (FedNow, PIX, SEPA Instant) all agree on the
//!    interpretation of the shared ISO 20022 status codes.
//!
//! By treating the same `pacs002_*.xml` file as input to three
//! different `status_map::map_transaction_status` functions, we
//! mechanically prove that the rail-specific code agrees on what ACSC
//! and RJCT mean.

// The whole file requires the FedNow XML parser. Sub-tests that
// additionally need PIX or SEPA mappers carry their own cfg gates.
#![cfg(feature = "fednow")]

#[cfg(all(feature = "pix", feature = "sepa-instant"))]
use op_rails_a2a::acquirer::A2aStatus;
use op_rails_a2a::fednow::xml::parse_pacs002;

const ACSC_VECTOR: &str = include_str!("../vectors/pacs002_acsc.xml");
const RJCT_VECTOR: &str = include_str!("../vectors/pacs002_rjct_ac03.xml");

#[test]
fn acsc_vector_parses_all_documented_fields() {
    let parsed = parse_pacs002(ACSC_VECTOR).expect("ACSC vector must parse");
    assert_eq!(parsed.transaction_status, "ACSC");
    assert_eq!(
        parsed.uetr.as_deref(),
        Some("12a345b6-7c89-4d01-23e4-567890abcdef")
    );
    assert_eq!(parsed.original_end_to_end_id.as_deref(), Some("INV4242"));
    assert!(
        parsed.reason_code.is_none(),
        "ACSC should have no reason code"
    );
    assert!(
        parsed.reason_text.is_none(),
        "ACSC should have no reason text"
    );
}

#[test]
fn rjct_vector_parses_reason_code_and_text() {
    let parsed = parse_pacs002(RJCT_VECTOR).expect("RJCT vector must parse");
    assert_eq!(parsed.transaction_status, "RJCT");
    assert_eq!(parsed.reason_code.as_deref(), Some("AC03"));
    assert_eq!(
        parsed.reason_text.as_deref(),
        Some("Invalid creditor account number")
    );
}

#[test]
#[cfg(all(feature = "pix", feature = "sepa-instant"))]
fn all_three_rails_agree_acsc_means_settled() {
    use op_rails_a2a::fednow::status_map as fednow;
    use op_rails_a2a::pix::status_map as pix;
    use op_rails_a2a::sepa_instant::status_map as sepa;

    let parsed = parse_pacs002(ACSC_VECTOR).unwrap();
    assert_eq!(
        fednow::map_transaction_status(&parsed.transaction_status).unwrap(),
        A2aStatus::Settled
    );
    assert_eq!(
        pix::map_transaction_status(&parsed.transaction_status).unwrap(),
        A2aStatus::Settled
    );
    assert_eq!(
        sepa::map_transaction_status(&parsed.transaction_status).unwrap(),
        A2aStatus::Settled
    );
}

#[test]
#[cfg(all(feature = "pix", feature = "sepa-instant"))]
fn all_three_rails_agree_rjct_means_rejected() {
    use op_rails_a2a::fednow::status_map as fednow;
    use op_rails_a2a::pix::status_map as pix;
    use op_rails_a2a::sepa_instant::status_map as sepa;

    let parsed = parse_pacs002(RJCT_VECTOR).unwrap();
    assert_eq!(
        fednow::map_transaction_status(&parsed.transaction_status).unwrap(),
        A2aStatus::Rejected
    );
    assert_eq!(
        pix::map_transaction_status(&parsed.transaction_status).unwrap(),
        A2aStatus::Rejected
    );
    assert_eq!(
        sepa::map_transaction_status(&parsed.transaction_status).unwrap(),
        A2aStatus::Rejected
    );
}

#[test]
fn fednow_outbound_vector_is_well_formed() {
    let outbound = include_str!("../vectors/fednow_pacs008_outbound.xml");
    // FedNow profile sentinel: USABA clearing system
    assert!(outbound.contains("<Cd>USABA</Cd>"), "FedNow uses USABA");
    assert!(outbound.contains(r#"xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08""#));
    assert!(outbound.contains("<SttlmMtd>CLRG</SttlmMtd>"));
    assert!(outbound.contains("<NbOfTxs>1</NbOfTxs>"));
    assert!(outbound.contains("<ChrgBr>SLEV</ChrgBr>"));
    assert!(outbound.contains(r#"<IntrBkSttlmAmt Ccy="USD">"#));
    // Must NOT contain SEPA-specific sentinels.
    assert!(
        !outbound.contains("<LclInstrm>"),
        "FedNow does not use LclInstrm"
    );
    assert!(!outbound.contains("<IBAN>"), "FedNow does not use IBAN");
}

#[test]
#[cfg(feature = "sepa-instant")]
fn sepa_instant_outbound_vector_has_mandatory_inst_code() {
    let outbound = include_str!("../vectors/sepa_instant_pacs008_outbound.xml");
    // Mandatory per EPC SCT Inst IG 2019 v1.0.
    assert!(
        outbound.contains("<Cd>INST</Cd>"),
        "SCT Inst MUST carry LclInstrm.Cd=INST"
    );
    assert!(
        outbound.contains("<SvcLvl><Cd>SEPA</Cd></SvcLvl>"),
        "SCT Inst MUST carry SvcLvl.Cd=SEPA"
    );
    assert!(
        outbound.contains("<NbOfTxs>1</NbOfTxs>"),
        "RT1/TIPS require single tx"
    );
    assert!(
        outbound.contains(r#"<IntrBkSttlmAmt Ccy="EUR">"#),
        "SCT Inst is EUR only"
    );
    assert!(outbound.contains("<IBAN>DE89370400440532013000</IBAN>"));
    assert!(outbound.contains("<BICFI>DEUTDEFFXXX</BICFI>"));
    // Must NOT contain FedNow-specific sentinels.
    assert!(!outbound.contains("USABA"), "SEPA does not use USABA");
}

#[test]
fn outbound_vectors_differ_structurally() {
    let fednow = include_str!("../vectors/fednow_pacs008_outbound.xml");
    let sepa = include_str!("../vectors/sepa_instant_pacs008_outbound.xml");
    assert_ne!(
        fednow, sepa,
        "outbound vectors must be distinguishable per rail"
    );
}
