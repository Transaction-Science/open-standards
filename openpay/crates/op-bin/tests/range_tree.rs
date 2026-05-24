//! Range-tree integration tests:
//! - the default tree built from `network_ranges` is internally
//!   consistent (no overlaps),
//! - representative BINs across every supported network resolve
//!   to the expected network,
//! - point lookups outside any range return `None`.

use op_bin::network_ranges::build_default_tree;
use op_bin::{Bin, CardNetwork};

#[test]
fn default_tree_constructs_disjoint() {
    let tree = build_default_tree().expect("disjoint by construction");
    assert!(tree.len() >= 25);
}

#[test]
fn known_bins_resolve_per_network() {
    let tree = build_default_tree().expect("ok");
    let cases: &[(&str, CardNetwork)] = &[
        ("411111", CardNetwork::Visa),
        ("510000", CardNetwork::Mastercard),
        ("222100", CardNetwork::Mastercard),
        ("340000", CardNetwork::Amex),
        ("370000", CardNetwork::Amex),
        ("601100", CardNetwork::Discover),
        ("352800", CardNetwork::Jcb),
        ("360000", CardNetwork::DinersClub),
        ("621234", CardNetwork::UnionPay),
        ("810000", CardNetwork::RuPay),
        ("220000", CardNetwork::Mir),
        ("979200", CardNetwork::Troy),
    ];
    for (bin_str, expected) in cases {
        let b = Bin::parse(bin_str).expect("valid");
        let got = tree
            .lookup(&b)
            .unwrap_or_else(|| panic!("no range for {bin_str}"));
        assert_eq!(
            got.network, *expected,
            "BIN {bin_str} expected {expected:?}, got {:?}",
            got.network,
        );
    }
}

#[test]
fn unassigned_bin_returns_none() {
    let tree = build_default_tree().expect("ok");
    // 700000 is unassigned in the published BIN catalog.
    let b = Bin::parse("700000").expect("ok");
    assert!(tree.lookup(&b).is_none());
}

#[test]
fn boundary_half_open() {
    let tree = build_default_tree().expect("ok");
    // Mastercard classic range ends just before 56xxxxxx.
    let last_mc = Bin::parse("559999").expect("ok");
    assert_eq!(
        tree.lookup(&last_mc).map(|r| r.network),
        Some(CardNetwork::Mastercard),
    );
    // 56xxxxxx maps into the Maestro carve-out at 5612 only;
    // 5600xx falls outside any registered range.
    let outside_mc = Bin::parse("560000").expect("ok");
    assert!(tree.lookup(&outside_mc).is_none());
}

#[test]
fn sorted_invariant_holds() {
    let tree = build_default_tree().expect("ok");
    let rs = tree.ranges();
    for win in rs.windows(2) {
        assert!(
            win[0].high <= win[1].low,
            "ranges out of order or overlapping: {:?} then {:?}",
            win[0],
            win[1],
        );
    }
}
