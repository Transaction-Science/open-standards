//! Industry test PANs across the major networks should all pass
//! the Luhn check; perturbing the last digit should fail.

use op_bin::luhn;

#[test]
fn industry_test_pans_pass() {
    // Visa
    assert!(luhn::is_valid("4111111111111111"));
    assert!(luhn::is_valid("4012888888881881"));
    // Mastercard
    assert!(luhn::is_valid("5555555555554444"));
    assert!(luhn::is_valid("5105105105105100"));
    // Amex
    assert!(luhn::is_valid("378282246310005"));
    assert!(luhn::is_valid("371449635398431"));
    // Discover
    assert!(luhn::is_valid("6011111111111117"));
    assert!(luhn::is_valid("6011000990139424"));
    // JCB
    assert!(luhn::is_valid("3530111333300000"));
}

#[test]
fn perturbed_last_digit_fails() {
    // Flip the check digit of each — should fail.
    assert!(!luhn::is_valid("4111111111111110"));
    assert!(!luhn::is_valid("5555555555554445"));
    assert!(!luhn::is_valid("378282246310006"));
}

#[test]
fn check_digit_recovers_industry_test_pans() {
    let pairs = [
        ("411111111111111", '1'),
        ("555555555555444", '4'),
        ("37828224631000", '5'),
        ("353011133330000", '0'),
    ];
    for (partial, expected) in pairs {
        let cd = luhn::check_digit(partial).expect("ok");
        assert_eq!(
            char::from_digit(cd as u32, 10),
            Some(expected),
            "for partial {partial}",
        );
    }
}

#[test]
fn non_digit_input_rejected() {
    assert!(!luhn::is_valid("4111-1111-1111-1111"));
    assert!(!luhn::is_valid(" 4111111111111111"));
}
