//! CE3.0 qualifying-transaction rules.
//!
//! Covers the four published constraints:
//!
//! 1. Reason code must be Visa 10.4.
//! 2. >= 2 prior, non-disputed transactions from same cardholder.
//! 3. Each qualifier 120-365 days old at the time of the disputed
//!    transaction.
//! 4. >= 2 linking-data matches between disputed + qualifier.

use op_dispute_evidence::{
    Ce3Qualifier, Network, QualifyingTransaction, ReasonCode, VisaReasonCode,
};
use op_dispute_evidence::ce3::{CE3_MAX_AGE, CE3_MIN_AGE, LinkingData};
use time::{Duration, OffsetDateTime};

fn now() -> OffsetDateTime {
    OffsetDateTime::from_unix_timestamp(1_700_000_000).expect("ok")
}

fn make_qualifier(
    id: &str,
    age: Duration,
    ip: Option<&str>,
    device: Option<&str>,
    shipping: Option<&str>,
    disputed: bool,
) -> QualifyingTransaction {
    QualifyingTransaction {
        id: id.into(),
        authorized_at: now() - age,
        ip: ip.map(str::to_owned),
        device_id: device.map(str::to_owned),
        shipping_address: shipping.map(str::to_owned),
        account_login: Some("user-42".into()),
        was_disputed: disputed,
    }
}

fn disputed_link() -> LinkingData {
    LinkingData {
        ip: Some("203.0.113.7".into()),
        device_id: Some("dev-AAA".into()),
        shipping_address: Some("addr-XYZ".into()),
        account_login: Some("user-42".into()),
    }
}

#[test]
fn ce3_eligible_when_two_qualifiers_with_two_links_each() {
    let qualifiers = [
        make_qualifier(
            "q1",
            Duration::days(150),
            Some("203.0.113.7"),
            Some("dev-AAA"),
            None,
            false,
        ),
        make_qualifier(
            "q2",
            Duration::days(300),
            None,
            Some("dev-AAA"),
            Some("addr-XYZ"),
            false,
        ),
    ];
    let result = Ce3Qualifier::evaluate(
        ReasonCode::Visa(VisaReasonCode::F1040),
        now(),
        &disputed_link(),
        &qualifiers,
    )
    .expect("evaluate");
    assert!(result.eligible, "two qualifiers, each with 2 links, should be eligible");
    assert_eq!(result.matched_qualifiers.len(), 2);
}

#[test]
fn ce3_ineligible_when_too_few_links() {
    // Only IP matches on each qualifier (1 link < required 2).
    let qualifiers = [
        make_qualifier("q1", Duration::days(150), Some("203.0.113.7"), None, None, false),
        make_qualifier("q2", Duration::days(300), Some("203.0.113.7"), None, None, false),
    ];
    let mut link = disputed_link();
    link.device_id = None;
    link.shipping_address = None;
    link.account_login = None;
    let result = Ce3Qualifier::evaluate(
        ReasonCode::Visa(VisaReasonCode::F1040),
        now(),
        &link,
        &qualifiers,
    )
    .expect("evaluate");
    assert!(!result.eligible);
    assert_eq!(result.matched_qualifiers.len(), 0);
}

#[test]
fn ce3_ineligible_when_qualifiers_outside_window() {
    // Both qualifiers are too recent — under the 120-day floor.
    let qualifiers = [
        make_qualifier("q1", Duration::days(30), Some("203.0.113.7"), Some("dev-AAA"), None, false),
        make_qualifier("q2", Duration::days(60), Some("203.0.113.7"), Some("dev-AAA"), None, false),
    ];
    let result = Ce3Qualifier::evaluate(
        ReasonCode::Visa(VisaReasonCode::F1040),
        now(),
        &disputed_link(),
        &qualifiers,
    )
    .expect("evaluate");
    assert!(!result.eligible);
    // Sanity-check the constants are exposed and self-consistent.
    assert!(CE3_MIN_AGE < CE3_MAX_AGE);
}

#[test]
fn ce3_skips_previously_disputed_qualifiers() {
    let qualifiers = [
        make_qualifier(
            "good",
            Duration::days(150),
            Some("203.0.113.7"),
            Some("dev-AAA"),
            None,
            false,
        ),
        make_qualifier(
            "tainted",
            Duration::days(200),
            Some("203.0.113.7"),
            Some("dev-AAA"),
            None,
            true,
        ),
    ];
    let result = Ce3Qualifier::evaluate(
        ReasonCode::Visa(VisaReasonCode::F1040),
        now(),
        &disputed_link(),
        &qualifiers,
    )
    .expect("evaluate");
    // Only one un-disputed qualifier passes — below the 2-min.
    assert!(!result.eligible);
    assert_eq!(result.matched_qualifiers, vec!["good".to_string()]);
}

#[test]
fn ce3_only_applies_to_visa_10_4() {
    let q = make_qualifier(
        "q1",
        Duration::days(150),
        Some("203.0.113.7"),
        Some("dev-AAA"),
        None,
        false,
    );
    let err = Ce3Qualifier::evaluate(
        ReasonCode::Visa(VisaReasonCode::F1030),
        now(),
        &disputed_link(),
        &[q],
    )
    .expect_err("non-10.4 must reject");
    let _ = err; // surface that we got an error
    // Network helper is exercised transitively.
    assert_eq!(ReasonCode::Visa(VisaReasonCode::F1040).network(), Network::Visa);
}
