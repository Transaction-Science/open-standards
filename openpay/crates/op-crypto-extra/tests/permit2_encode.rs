//! Integration: Permit2 typed-data struct hashes.

use op_crypto_extra::permit2::{
    PERMIT2_ADDRESS, Permit2BatchTransfer, Permit2SingleTransfer, Permit2TokenPermissions,
};

#[test]
fn canonical_address_is_42_chars() {
    assert_eq!(PERMIT2_ADDRESS.len(), 42);
    assert!(PERMIT2_ADDRESS.starts_with("0x"));
    // CREATE2-deployed everywhere, starts with many zeros.
    assert!(PERMIT2_ADDRESS.starts_with("0x000000"));
}

#[test]
fn single_struct_hash_is_deterministic() {
    let single = Permit2SingleTransfer {
        permitted: Permit2TokenPermissions::new(
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            5_000_000,
        ),
        nonce: 42,
        deadline: 1_900_000_000,
    };
    let h1 = single
        .struct_hash("0x3333333333333333333333333333333333333333")
        .unwrap();
    let h2 = single
        .struct_hash("0x3333333333333333333333333333333333333333")
        .unwrap();
    assert_eq!(h1, h2);
}

#[test]
fn single_struct_hash_changes_with_nonce() {
    let base = Permit2SingleTransfer {
        permitted: Permit2TokenPermissions::new(
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            5_000_000,
        ),
        nonce: 42,
        deadline: 1_900_000_000,
    };
    let mut mutated = base.clone();
    mutated.nonce += 1;
    let h1 = base
        .struct_hash("0x3333333333333333333333333333333333333333")
        .unwrap();
    let h2 = mutated
        .struct_hash("0x3333333333333333333333333333333333333333")
        .unwrap();
    assert_ne!(h1, h2);
}

#[test]
fn batch_with_two_distinct_tokens_hashes() {
    let batch = Permit2BatchTransfer {
        permitted: vec![
            Permit2TokenPermissions::new(
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                1_000_000,
            ),
            Permit2TokenPermissions::new(
                "0xdac17f958d2ee523a2206206994597c13d831ec7",
                2_000_000,
            ),
        ],
        nonce: 0,
        deadline: 1_900_000_000,
    };
    let h = batch
        .struct_hash("0x3333333333333333333333333333333333333333")
        .unwrap();
    assert_ne!(h, [0u8; 32]);
}
