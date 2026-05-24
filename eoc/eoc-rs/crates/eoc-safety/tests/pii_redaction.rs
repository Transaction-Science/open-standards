//! PII redactor tests.

use eoc_safety::pii::PiiRedactor;

#[test]
fn redacts_email() {
    let r = PiiRedactor::new().expect("build redactor");
    let rep = r.redact("contact me at alice@example.com please");
    assert!(rep.has_pii());
    assert!(rep.redacted.contains("<EMAIL>"));
    assert!(!rep.redacted.contains("alice@example.com"));
}

#[test]
fn redacts_ssn() {
    let r = PiiRedactor::new().expect("build redactor");
    let rep = r.redact("my ssn is 123-45-6789 ok");
    assert!(rep.has_pii());
    assert!(rep.redacted.contains("<SSN>"));
}

#[test]
fn redacts_phone() {
    let r = PiiRedactor::new().expect("build redactor");
    let rep = r.redact("call (415) 555-1234 tomorrow");
    assert!(rep.has_pii());
    assert!(rep.redacted.contains("<PHONE>"));
}

#[test]
fn redacts_luhn_valid_credit_card() {
    let r = PiiRedactor::new().expect("build redactor");
    // Visa test number, Luhn-valid.
    let rep = r.redact("card 4111 1111 1111 1111 expires soon");
    assert!(rep.has_pii());
    assert!(rep.redacted.contains("<CREDIT_CARD>"));
}

#[test]
fn skips_luhn_invalid_number() {
    let r = PiiRedactor::new().expect("build redactor");
    let rep = r.redact("card 1234 5678 9012 3456 expires soon");
    assert!(
        !rep.spans.iter().any(|s| s.category == "CREDIT_CARD"),
        "Luhn-invalid number must not be flagged as a credit card"
    );
}

#[test]
fn redacts_ipv4() {
    let r = PiiRedactor::new().expect("build redactor");
    let rep = r.redact("the server is at 10.10.10.3 today");
    assert!(rep.has_pii());
    assert!(rep.redacted.contains("<IPV4>"));
}

#[test]
fn benign_text_unchanged() {
    let r = PiiRedactor::new().expect("build redactor");
    let s = "the cat sat on the mat";
    let rep = r.redact(s);
    assert!(!rep.has_pii());
    assert_eq!(rep.redacted, s);
}
