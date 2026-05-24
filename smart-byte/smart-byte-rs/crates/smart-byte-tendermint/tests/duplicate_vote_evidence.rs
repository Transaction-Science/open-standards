use smart_byte_tendermint::evidence::{DuplicateVoteEvidence, Evidence, LightClientAttackEvidence};
use smart_byte_tendermint::block::{Commit, CommitSig};
use smart_byte_tendermint::validator::ValidatorId;
use smart_byte_tendermint::vote::{Signature, Vote, VoteType};

fn vid(b: u8) -> ValidatorId {
    ValidatorId::from_bytes([b; 32])
}

fn empty_sig() -> Signature {
    Signature::from_bytes([0u8; 64])
}

fn mk_vote(v: ValidatorId, h: u64, r: u32, block: Option<[u8; 32]>) -> Vote {
    Vote {
        kind: VoteType::Precommit,
        height: h,
        round: r,
        block_hash: block,
        validator: v,
        signature: empty_sig(),
    }
}

#[test]
fn duplicate_vote_detected_when_two_different_block_hashes_at_same_height_round() {
    let ev = DuplicateVoteEvidence {
        vote_a: mk_vote(vid(7), 100, 0, Some([0xAA; 32])),
        vote_b: mk_vote(vid(7), 100, 0, Some([0xBB; 32])),
    };
    ev.verify().unwrap();
    assert_eq!(ev.offender(), vid(7));
}

#[test]
fn matching_votes_not_evidence() {
    let ev = DuplicateVoteEvidence {
        vote_a: mk_vote(vid(7), 100, 0, Some([0xAA; 32])),
        vote_b: mk_vote(vid(7), 100, 0, Some([0xAA; 32])),
    };
    assert!(ev.verify().is_err());
}

#[test]
fn different_validators_not_evidence() {
    let ev = DuplicateVoteEvidence {
        vote_a: mk_vote(vid(7), 100, 0, Some([0xAA; 32])),
        vote_b: mk_vote(vid(8), 100, 0, Some([0xBB; 32])),
    };
    assert!(ev.verify().is_err());
}

#[test]
fn different_heights_not_evidence() {
    let ev = DuplicateVoteEvidence {
        vote_a: mk_vote(vid(7), 100, 0, Some([0xAA; 32])),
        vote_b: mk_vote(vid(7), 101, 0, Some([0xBB; 32])),
    };
    assert!(ev.verify().is_err());
}

#[test]
fn evidence_enum_dispatches_verify() {
    let dv = Evidence::DuplicateVote(DuplicateVoteEvidence {
        vote_a: mk_vote(vid(1), 5, 0, Some([1; 32])),
        vote_b: mk_vote(vid(1), 5, 0, Some([2; 32])),
    });
    dv.verify().unwrap();
}

#[test]
fn light_client_attack_offenders_are_intersection_of_signers() {
    let sigs_a = vec![
        CommitSig { validator: vid(1), signature: empty_sig() },
        CommitSig { validator: vid(2), signature: empty_sig() },
        CommitSig { validator: vid(3), signature: empty_sig() },
    ];
    let sigs_b = vec![
        CommitSig { validator: vid(2), signature: empty_sig() },
        CommitSig { validator: vid(3), signature: empty_sig() },
        CommitSig { validator: vid(4), signature: empty_sig() },
    ];
    let ev = LightClientAttackEvidence {
        canonical: Commit {
            height: 42,
            round: 0,
            block_hash: [0xAA; 32],
            signatures: sigs_a,
        },
        conflicting: Commit {
            height: 42,
            round: 0,
            block_hash: [0xBB; 32],
            signatures: sigs_b,
        },
    };
    ev.verify().unwrap();
    let mut off = ev.offenders();
    off.sort_by_key(|v| v.0[0]);
    assert_eq!(off, vec![vid(2), vid(3)]);
}
