//! Predicate proof: prove `age ≥ 18` via Bulletproofs range proof.

use smart_byte_zk::predicates::{
    prove_inequality, prove_range_predicate, verify_inequality, verify_range_predicate,
    InequalityPredicate, RangePredicate,
};

#[test]
fn age_at_least_18_verifies() {
    let predicate = InequalityPredicate {
        threshold: 18,
        bit_length: 8,
    };
    let actual_age: u64 = 21;
    let (proof, commitment) = prove_inequality(actual_age, &predicate).expect("prove");
    assert!(verify_inequality(&predicate, commitment, &proof).expect("verify"));
}

#[test]
fn age_under_18_rejected_at_prove_time() {
    let predicate = InequalityPredicate {
        threshold: 18,
        bit_length: 8,
    };
    let actual_age: u64 = 16;
    let result = prove_inequality(actual_age, &predicate);
    assert!(result.is_err());
}

#[test]
fn age_within_range_18_120_verifies() {
    let predicate = RangePredicate {
        lower: 18,
        upper: 120,
        bit_length: 8,
    };
    let actual_age: u64 = 42;
    let (proof, commitment) = prove_range_predicate(actual_age, &predicate).expect("prove");
    assert!(verify_range_predicate(&predicate, commitment, &proof).expect("verify"));
}
