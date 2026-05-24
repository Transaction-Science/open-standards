use smart_byte_tendermint::validator::{Validator, ValidatorId, ValidatorSet};

fn vid(b: u8) -> ValidatorId {
    ValidatorId::from_bytes([b; 32])
}

#[test]
fn quorum_is_two_thirds_plus_one_for_equal_power() {
    // 4 validators with power 1 each, total = 4
    let vs = ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    assert_eq!(vs.total_voting_power(), 4);
    // floor(2*4/3)+1 = floor(2.66)+1 = 2+1 = 3
    assert_eq!(vs.quorum(), 3);
}

#[test]
fn quorum_handles_weighted_validators() {
    // total = 100 → quorum = 66+1 = 67
    let vs = ValidatorSet::new(vec![
        Validator::new(vid(1), 40),
        Validator::new(vid(2), 30),
        Validator::new(vid(3), 20),
        Validator::new(vid(4), 10),
    ])
    .unwrap();
    assert_eq!(vs.total_voting_power(), 100);
    assert_eq!(vs.quorum(), 67);
    assert_eq!(vs.trust_threshold(), 34);
}

#[test]
fn has_quorum_aggregates_signers_by_power() {
    let vs = ValidatorSet::new(vec![
        Validator::new(vid(1), 40),
        Validator::new(vid(2), 30),
        Validator::new(vid(3), 20),
        Validator::new(vid(4), 10),
    ])
    .unwrap();
    // 40 + 30 = 70 ≥ 67 → quorum
    assert!(vs.has_quorum([&vid(1), &vid(2)].into_iter()));
    // 40 + 20 = 60 < 67 → no quorum
    assert!(!vs.has_quorum([&vid(1), &vid(3)].into_iter()));
    // unknown validator → no quorum
    assert!(!vs.has_quorum([&vid(99)].into_iter()));
}

#[test]
fn overflow_returns_error() {
    let err = ValidatorSet::new(vec![
        Validator::new(vid(1), u64::MAX),
        Validator::new(vid(2), 1),
    ]);
    assert!(err.is_err());
}

#[test]
fn proposer_rotates_by_round() {
    let vs = ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    let p0 = vs.proposer(1, 0).unwrap().id;
    let p1 = vs.proposer(1, 1).unwrap().id;
    let p2 = vs.proposer(1, 2).unwrap().id;
    let p3 = vs.proposer(1, 3).unwrap().id;
    let p4 = vs.proposer(1, 4).unwrap().id;
    // round-robin: 1,2,3,0,1
    assert_ne!(p0, p1);
    assert_ne!(p1, p2);
    assert_ne!(p2, p3);
    assert_eq!(p0, p4);
}
