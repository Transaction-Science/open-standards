//! Integration: SEPA Instant payout produces pacs.008 XML.

use op_core::{Currency, Money};
use op_payouts::sepa::SepaDriver;
use op_payouts::{
    Beneficiary, BeneficiaryAccount, Error, FundingSource, Payout, PayoutMethod, PayoutRequest,
    PayoutStatus,
};

fn driver() -> SepaDriver {
    SepaDriver {
        sender_bic: "AAAADEFFXXX".to_string(),
        sender_name: "Acme GmbH".to_string(),
        sender_iban: "DE89370400440532013000".to_string(),
    }
}

fn req() -> PayoutRequest {
    PayoutRequest {
        idempotency_key: "22222222-2222-4222-8222-222222222222".to_string(),
        method: PayoutMethod::SepaSctInst,
        amount: Money::from_minor(50_000, Currency::EUR), // EUR 500.00
        beneficiary: Beneficiary {
            name: "Jean Dupont".to_string(),
            address: None,
            account: BeneficiaryAccount::Iban("FR1420041010050500013M02606".to_string()),
            kyc_ref: None,
        },
        funding: FundingSource::Prefunded {
            account_ref: "tips-balance".to_string(),
        },
        memo: Some("invoice 42".to_string()),
    }
}

#[test]
fn sepa_inst_produces_pacs008_xml() {
    let res = driver().submit(&req()).expect("offline build");
    assert_eq!(res.status, PayoutStatus::PreparedOffline);
    let xml = String::from_utf8(res.wire_payload.expect("payload")).expect("utf8");
    assert!(xml.contains("urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08"));
    assert!(xml.contains("<IBAN>FR1420041010050500013M02606</IBAN>"));
    assert!(xml.contains("Ccy=\"EUR\""));
    assert!(xml.contains("500.00"));
}

#[test]
fn sepa_inst_rejects_non_eur() {
    let mut r = req();
    r.amount = Money::from_minor(50_000, Currency::USD);
    let err = driver().submit(&r).unwrap_err();
    assert!(matches!(err, Error::LimitViolation { rail: "sepa", .. }));
}

#[test]
fn sepa_inst_enforces_100k_cap() {
    let mut r = req();
    r.amount = Money::from_minor(100_001 * 100, Currency::EUR);
    let err = driver().submit(&r).unwrap_err();
    assert!(matches!(err, Error::LimitViolation { rail: "sepa", .. }));
}
