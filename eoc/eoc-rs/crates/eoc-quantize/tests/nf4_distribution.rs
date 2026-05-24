//! NF4 distribution test — the published levels must satisfy the
//! invariants used by QLoRA: 16 distinct points, symmetric extremes at
//! ±1, sorted ascending, and containing exactly one zero.

use eoc_quantize::nf4::NF4_LEVELS;

#[test]
fn has_sixteen_levels() {
    assert_eq!(NF4_LEVELS.len(), 16);
}

#[test]
fn endpoints_are_unit_magnitude() {
    assert!((NF4_LEVELS[0] - (-1.0)).abs() < 1e-6);
    assert!((NF4_LEVELS[15] - 1.0).abs() < 1e-6);
}

#[test]
fn strictly_increasing() {
    for w in NF4_LEVELS.windows(2) {
        assert!(w[1] > w[0], "NF4 not strictly increasing: {w:?}");
    }
}

#[test]
fn contains_zero() {
    assert!(NF4_LEVELS.iter().any(|&v| v == 0.0));
}

#[test]
fn distribution_has_more_resolution_near_zero() {
    // The published NF4 spacing is denser near zero than near the
    // tails (because the standard normal density is denser near zero).
    let near_zero = (NF4_LEVELS[8] - NF4_LEVELS[7]).abs();
    let tail = (NF4_LEVELS[15] - NF4_LEVELS[14]).abs();
    assert!(
        near_zero < tail,
        "expected denser packing near zero, got near_zero={near_zero}, tail={tail}"
    );
}
