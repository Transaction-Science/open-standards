//! Every BIN starting with `4` is Visa, regardless of the
//! remaining digits. Confirm against representative samples and
//! boundary digits.

use op_bin::{classify, Bin, CardNetwork};

#[test]
fn all_4_prefixes_classify_as_visa() {
    for tail in ["00000", "11111", "55555", "99999", "12345"] {
        let s = format!("4{tail}");
        let b = Bin::parse(&s).expect("valid BIN");
        assert_eq!(
            classify(&b),
            CardNetwork::Visa,
            "BIN {s} should be Visa",
        );
    }
}

#[test]
fn six_seven_eight_digit_visa_all_classify() {
    let six = Bin::parse("411111").expect("ok");
    let seven = Bin::parse("4111111").expect("ok");
    let eight = Bin::parse("41111111").expect("ok");
    assert_eq!(classify(&six), CardNetwork::Visa);
    assert_eq!(classify(&seven), CardNetwork::Visa);
    assert_eq!(classify(&eight), CardNetwork::Visa);
}

#[test]
fn boundary_3_and_5_are_not_visa() {
    assert_ne!(
        classify(&Bin::parse("399999").expect("ok")),
        CardNetwork::Visa,
    );
    assert_ne!(
        classify(&Bin::parse("500000").expect("ok")),
        CardNetwork::Visa,
    );
}

#[test]
fn classifier_agrees_with_range_tree_for_visa() {
    let tree = op_bin::network_ranges::build_default_tree().expect("disjoint");
    for tail in ["00000", "11111", "55555", "99999"] {
        let s = format!("4{tail}");
        let b = Bin::parse(&s).expect("ok");
        let from_classifier = classify(&b);
        let from_tree = tree.lookup(&b).map(|r| r.network).unwrap_or(CardNetwork::Unknown);
        assert_eq!(from_classifier, CardNetwork::Visa);
        assert_eq!(from_tree, CardNetwork::Visa);
    }
}
