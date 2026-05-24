//! Integration: ERC-4337 v0.7 packed user-op assembly.

use op_crypto_extra::erc4337::{EntryPointVersion, UserOperation, pack_user_op};

#[test]
fn entrypoint_addresses_distinct_across_versions() {
    let a = EntryPointVersion::V0_6.canonical_address();
    let b = EntryPointVersion::V0_7.canonical_address();
    let c = EntryPointVersion::V0_8.canonical_address();
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert!(a.starts_with("0x"));
    assert!(b.starts_with("0x"));
}

#[test]
fn pack_round_trip_preserves_data_fields() {
    let op = UserOperation {
        sender: "0x1111111111111111111111111111111111111111".into(),
        nonce: 7,
        init_code: vec![0xde, 0xad],
        call_data: vec![0xbe, 0xef],
        call_gas_limit: 100_000,
        verification_gas_limit: 60_000,
        pre_verification_gas: 21_000,
        max_fee_per_gas: 10_000_000_000,
        max_priority_fee_per_gas: 2_000_000_000,
        paymaster_and_data: vec![0xab; 20],
        signature: vec![0xcd; 65],
    };
    op.validate_structure().expect("structure ok");
    let packed = pack_user_op(&op);
    assert_eq!(packed.sender, op.sender);
    assert_eq!(packed.nonce, op.nonce);
    assert_eq!(packed.init_code, op.init_code);
    assert_eq!(packed.call_data, op.call_data);
    assert_eq!(packed.pre_verification_gas, op.pre_verification_gas);
    assert_eq!(packed.paymaster_and_data, op.paymaster_and_data);
    assert_eq!(packed.signature, op.signature);

    // account_gas_limits high half = verification_gas_limit
    let high: u128 = u128::from_be_bytes(packed.account_gas_limits[..16].try_into().unwrap());
    let low: u128 = u128::from_be_bytes(packed.account_gas_limits[16..].try_into().unwrap());
    assert_eq!(high, 60_000);
    assert_eq!(low, 100_000);
}

#[test]
fn validate_rejects_paymaster_too_short() {
    let mut op = UserOperation::new("0x1111111111111111111111111111111111111111");
    op.call_gas_limit = 1;
    op.verification_gas_limit = 1;
    op.max_fee_per_gas = 1;
    op.paymaster_and_data = vec![0xab; 5]; // too short
    assert!(op.validate_structure().is_err());
}
