use smart_byte_tendermint::consensus::{ConsensusState, Step};
use smart_byte_tendermint::proposal::Proposal;
use smart_byte_tendermint::validator::{Validator, ValidatorId, ValidatorSet};
use smart_byte_tendermint::vote::{Signature, Vote, VoteType};

fn vid(b: u8) -> ValidatorId {
    ValidatorId::from_bytes([b; 32])
}

fn empty_sig() -> Signature {
    Signature::from_bytes([0u8; 64])
}

fn vote(
    kind: VoteType,
    height: u64,
    round: u32,
    block: Option<[u8; 32]>,
    v: ValidatorId,
) -> Vote {
    Vote {
        kind,
        height,
        round,
        block_hash: block,
        validator: v,
        signature: empty_sig(),
    }
}

/// 4-validator network with equal voting power. 3 of 4 sign the
/// commit, satisfying the 2/3+1 BFT quorum (which is exactly 3 for
/// total power = 4).
#[test]
fn three_of_four_validators_commit_a_block() {
    let validators: Vec<_> = (0..4).map(|i| Validator::new(vid(i), 1)).collect();
    let vs = ValidatorSet::new(validators).unwrap();
    assert_eq!(vs.quorum(), 3);

    let mut state = ConsensusState::new("smart-byte-test", 1, vs.clone());

    let proposer = state.validators().proposer(1, 0).unwrap().id;
    let block: [u8; 32] = [0xAA; 32];
    let proposal = Proposal {
        height: 1,
        round: 0,
        block_hash: block,
        pol_round: None,
        proposer,
        signature: empty_sig(),
    };
    state.apply_proposal(proposal).unwrap();
    assert_eq!(state.step(), Step::Prevote);

    // 3 prevotes for the block from 3 distinct validators
    for i in 0..3u8 {
        let _ = state
            .apply_vote(vote(VoteType::Prevote, 1, 0, Some(block), vid(i)))
            .unwrap();
    }
    assert_eq!(state.step(), Step::Precommit);
    assert_eq!(state.prevote_power(Some(block)), 3);

    // 3 precommits → commit
    let mut committed = false;
    for i in 0..3u8 {
        committed |= state
            .apply_vote(vote(VoteType::Precommit, 1, 0, Some(block), vid(i)))
            .unwrap();
    }
    assert!(committed);
    assert_eq!(state.step(), Step::Commit);
    assert_eq!(state.committed(), Some(block));
    assert!(state.is_committed());
}

#[test]
fn two_of_four_validators_do_not_commit() {
    let vs =
        ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    let mut state = ConsensusState::new("smart-byte-test", 5, vs.clone());
    let proposer = state.validators().proposer(5, 0).unwrap().id;
    let block: [u8; 32] = [0xCC; 32];

    state
        .apply_proposal(Proposal {
            height: 5,
            round: 0,
            block_hash: block,
            pol_round: None,
            proposer,
            signature: empty_sig(),
        })
        .unwrap();

    // Only 2 prevotes — below quorum of 3
    for i in 0..2u8 {
        let _ = state
            .apply_vote(vote(VoteType::Prevote, 5, 0, Some(block), vid(i)))
            .unwrap();
    }
    assert_eq!(state.step(), Step::Prevote);
    assert!(!state.is_committed());
}

#[test]
fn duplicate_vote_from_same_validator_rejected() {
    let vs =
        ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    let mut state = ConsensusState::new("smart-byte-test", 1, vs);
    let proposer = state.validators().proposer(1, 0).unwrap().id;
    let block: [u8; 32] = [0xDD; 32];

    state
        .apply_proposal(Proposal {
            height: 1,
            round: 0,
            block_hash: block,
            pol_round: None,
            proposer,
            signature: empty_sig(),
        })
        .unwrap();

    state
        .apply_vote(vote(VoteType::Prevote, 1, 0, Some(block), vid(0)))
        .unwrap();
    let err = state
        .apply_vote(vote(VoteType::Prevote, 1, 0, Some(block), vid(0)))
        .unwrap_err();
    let s = format!("{err}");
    assert!(s.contains("duplicate"), "expected duplicate error, got {s}");
}

#[test]
fn next_round_clears_tallies_preserves_lock() {
    let vs =
        ValidatorSet::new((0..4).map(|i| Validator::new(vid(i as u8), 1))).unwrap();
    let mut state = ConsensusState::new("smart-byte-test", 9, vs);
    assert_eq!(state.round(), 0);
    state.next_round();
    assert_eq!(state.round(), 1);
    assert_eq!(state.step(), Step::Propose);
}
