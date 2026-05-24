//! ILP address grammar and prefix-matching tests.

use smart_byte_ilp::{Address, AddressScheme};

#[test]
fn accepts_all_known_schemes() {
    for s in [
        "g.us.bank.alice",
        "private.peer.alice",
        "example.test",
        "peer.alpha",
        "self.local",
        "test.alpha",
        "test1.beta",
        "test2.beta",
        "test3.beta",
        "local.dev",
    ] {
        assert!(Address::parse(s).is_ok(), "should parse: {s}");
    }
}

#[test]
fn rejects_unknown_scheme() {
    assert!(Address::parse("xn--bad").is_err());
    assert!(Address::parse("crypto.foo").is_err());
}

#[test]
fn rejects_empty_segment() {
    assert!(Address::parse("g..bank").is_err());
    assert!(Address::parse(".g.bank").is_err());
}

#[test]
fn rejects_oversized_input() {
    let huge = format!("g.{}", "a".repeat(1024));
    assert!(Address::parse(&huge).is_err());
}

#[test]
fn rejects_oversized_segment() {
    let seg = "x".repeat(129);
    let addr = format!("g.us.{seg}");
    assert!(Address::parse(&addr).is_err());
}

#[test]
fn segment_chars_only() {
    assert!(Address::parse("g.us.with space").is_err());
    assert!(Address::parse("g.us.with/slash").is_err());
}

#[test]
fn prefix_match_is_segment_aligned() {
    let a = Address::parse("g.us.bank.alice").unwrap();
    assert!(a.starts_with_prefix("g"));
    assert!(a.starts_with_prefix("g.us"));
    assert!(a.starts_with_prefix("g.us.bank"));
    assert!(a.starts_with_prefix("g.us.bank.alice"));
    assert!(!a.starts_with_prefix("g.us.bankhq"));
    assert!(!a.starts_with_prefix("g.us.bank.al"));
}

#[test]
fn scheme_lift() {
    assert_eq!(
        Address::parse("g.us").unwrap().scheme(),
        AddressScheme::Global
    );
    assert_eq!(
        Address::parse("test1.foo").unwrap().scheme(),
        AddressScheme::Test
    );
    assert_eq!(
        Address::parse("self.svc").unwrap().scheme(),
        AddressScheme::SelfScheme
    );
}
