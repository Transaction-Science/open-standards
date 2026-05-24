//! Bulletproofs range-proof integration: prove a balance lies in
//! `[0, 2^32)`.

use smart_byte_zk::bulletproofs::{
    pedersen_commit, prove_range, verify_range, RangeStatement, RangeWitness,
    BulletproofsRangeScheme,
};
use smart_byte_zk::scheme::ZkScheme;

#[test]
fn balance_in_u32_range_verifies() {
    let balance: u64 = 1_234_567_890;
    let (commitment, blinding) = pedersen_commit(balance);
    let witness = RangeWitness {
        value: balance,
        blinding,
    };

    let proof = prove_range(&witness, 32, b"smart-byte-zk/test/balance").expect("prove");
    assert_eq!(proof.bit_length, 32);
    assert_eq!(proof.commitment, commitment.to_bytes());

    let stmt = RangeStatement {
        commitment,
        bit_length: 32,
        label: b"smart-byte-zk/test/balance",
    };
    assert!(verify_range(&stmt, &proof).expect("verify"));
}

#[test]
fn balance_proof_rejects_under_wrong_label() {
    let balance: u64 = 42;
    let (commitment, blinding) = pedersen_commit(balance);
    let witness = RangeWitness {
        value: balance,
        blinding,
    };
    let proof = prove_range(&witness, 32, b"smart-byte-zk/test/balance").expect("prove");

    let stmt = RangeStatement {
        commitment,
        bit_length: 32,
        label: b"smart-byte-zk/test/wrong-label",
    };
    // Wrong transcript label => verification should fail (returns
    // Err from the underlying bulletproofs verifier).
    let result = verify_range(&stmt, &proof);
    assert!(result.is_err() || result.ok() == Some(false));
}

#[test]
fn scheme_trait_round_trip() {
    let scheme = BulletproofsRangeScheme;
    let balance: u64 = 7777;
    let (commitment, blinding) = pedersen_commit(balance);
    let stmt = RangeStatement {
        commitment,
        bit_length: 32,
        label: b"smart-byte-zk/test/scheme",
    };
    let witness = RangeWitness {
        value: balance,
        blinding,
    };
    let (pk, vk) = scheme.keygen(&stmt).expect("keygen");
    let proof = scheme.prove(&pk, &stmt, &witness).expect("prove");
    assert!(scheme.verify(&vk, &stmt, &proof).expect("verify"));
}
