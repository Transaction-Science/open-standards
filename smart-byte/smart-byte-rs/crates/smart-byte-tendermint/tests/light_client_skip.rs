use smart_byte_tendermint::block::{Commit, CommitSig, Header};
use smart_byte_tendermint::light_client::{
    apply_sequential, verify_sequential, verify_skipping, TrustedState,
};
use smart_byte_tendermint::validator::{Validator, ValidatorId, ValidatorSet};
use smart_byte_tendermint::vote::Signature;

fn vid(b: u8) -> ValidatorId {
    ValidatorId::from_bytes([b; 32])
}

fn empty_sig() -> Signature {
    Signature::from_bytes([0u8; 64])
}

fn make_header(height: u64, last_block_hash: [u8; 32], validators_hash: [u8; 32]) -> Header {
    Header {
        chain_id: "skipchain".into(),
        height,
        time_ms: height as i64 * 1000,
        last_block_hash,
        last_commit_hash: [0; 32],
        data_hash: [0; 32],
        validators_hash,
        next_validators_hash: validators_hash,
        app_hash: [0; 32],
        proposer: vid(0),
    }
}

fn sigs_for<I: IntoIterator<Item = ValidatorId>>(ids: I) -> Vec<CommitSig> {
    ids.into_iter()
        .map(|validator| CommitSig {
            validator,
            signature: empty_sig(),
        })
        .collect()
}

#[test]
fn sequential_verification_succeeds_with_quorum() {
    let vs = ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();

    let h0 = make_header(10, [0; 32], [9; 32]);
    let trusted = TrustedState::new(h0.clone(), vs.clone());

    let h1 = make_header(11, h0.hash(), [9; 32]);
    let commit = Commit {
        height: 11,
        round: 0,
        block_hash: h1.hash(),
        signatures: sigs_for([vid(0), vid(1), vid(2)]),
    };
    verify_sequential(&trusted, &h1, &commit).unwrap();
}

#[test]
fn sequential_verification_rejects_low_power() {
    let vs = ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    let h0 = make_header(10, [0; 32], [9; 32]);
    let trusted = TrustedState::new(h0.clone(), vs);

    let h1 = make_header(11, h0.hash(), [9; 32]);
    let commit = Commit {
        height: 11,
        round: 0,
        block_hash: h1.hash(),
        signatures: sigs_for([vid(0), vid(1)]),
    };
    assert!(verify_sequential(&trusted, &h1, &commit).is_err());
}

#[test]
fn skipping_verification_succeeds_with_one_third_overlap() {
    // Old set: 4 equal validators (ids 0..4). Trust threshold = 1/3*4+1 = 2.
    let old = ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    // New set: 6 validators (ids 0,1,4,5,6,7). Quorum = floor(2*6/3)+1 = 5.
    let new = ValidatorSet::new(
        [0u8, 1, 4, 5, 6, 7]
            .iter()
            .map(|i| Validator::new(vid(*i), 1)),
    )
    .unwrap();

    let h0 = make_header(100, [0; 32], [1; 32]);
    let trusted = TrustedState::new(h0.clone(), old);

    // Skip from height 100 → 200 with signers {0,1,4,5,6}: ids 0,1
    // overlap the OLD set (power 2 ≥ trust_threshold 2) AND all 5
    // satisfy the NEW set's quorum of 5.
    let h_target = make_header(200, [7; 32], [1; 32]);
    let commit = Commit {
        height: 200,
        round: 0,
        block_hash: h_target.hash(),
        signatures: sigs_for([vid(0), vid(1), vid(4), vid(5), vid(6)]),
    };

    verify_skipping(&trusted, &h_target, &new, &commit).unwrap();
}

#[test]
fn skipping_verification_rejects_no_overlap() {
    let old = ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    let new = ValidatorSet::new(
        [10u8, 11, 12, 13]
            .iter()
            .map(|i| Validator::new(vid(*i), 1)),
    )
    .unwrap();

    let h0 = make_header(50, [0; 32], [1; 32]);
    let trusted = TrustedState::new(h0, old);

    let h_target = make_header(75, [7; 32], [1; 32]);
    let commit = Commit {
        height: 75,
        round: 0,
        block_hash: h_target.hash(),
        signatures: sigs_for([vid(10), vid(11), vid(12)]),
    };
    let err = verify_skipping(&trusted, &h_target, &new, &commit).unwrap_err();
    let s = format!("{err}");
    assert!(s.contains("trust"), "{s}");
}

#[test]
fn apply_sequential_advances_trusted_state() {
    let vs = ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    let h0 = make_header(1, [0; 32], [9; 32]);
    let mut trusted = TrustedState::new(h0.clone(), vs.clone());

    let h1 = make_header(2, h0.hash(), [9; 32]);
    let commit = Commit {
        height: 2,
        round: 0,
        block_hash: h1.hash(),
        signatures: sigs_for([vid(0), vid(1), vid(2)]),
    };
    apply_sequential(&mut trusted, h1.clone(), vs, &commit).unwrap();
    assert_eq!(trusted.header.height, 2);
}
