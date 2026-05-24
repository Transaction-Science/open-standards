//! Integration: CCTP v2 burn-message encode / decode.

use op_crypto_extra::cctp::{CctpAttestation, CctpBurnMessage, CctpDomain};

#[test]
fn burn_message_encode_size_is_228_plus_hook() {
    let m = CctpBurnMessage::new([0x11; 32], [0x22; 32], 1, [0x33; 32]);
    let bytes = m.encode();
    assert_eq!(bytes.len(), 228);

    let mut m2 = m.clone();
    m2.hook_data = vec![0x44; 17];
    let bytes2 = m2.encode();
    assert_eq!(bytes2.len(), 228 + 17);
}

#[test]
fn domain_ids_match_circle_table() {
    assert_eq!(CctpDomain::Ethereum.id(), 0);
    assert_eq!(CctpDomain::Avalanche.id(), 1);
    assert_eq!(CctpDomain::Optimism.id(), 2);
    assert_eq!(CctpDomain::Arbitrum.id(), 3);
    assert_eq!(CctpDomain::Solana.id(), 5);
    assert_eq!(CctpDomain::Base.id(), 6);
    assert_eq!(CctpDomain::Polygon.id(), 7);
    assert_eq!(CctpDomain::from_id(42).id(), 42);
}

#[test]
fn encode_decode_round_trip_preserves_all_fields() {
    let mut m = CctpBurnMessage::new(
        [0xaa; 32],
        [0xbb; 32],
        500_000_000_000,
        [0xcc; 32],
    );
    m.max_fee = 12_345;
    m.expiration_block = 9_000_000;
    m.hook_data = b"opaque hook bytes".to_vec();

    let bytes = m.encode();
    let decoded = CctpBurnMessage::decode(&bytes).unwrap();
    assert_eq!(decoded, m);
}

#[test]
fn attestation_layout_check_signature_multiples_of_65() {
    // Three-attestor: 195 bytes.
    let att = CctpAttestation::new(vec![0xaa; 195], vec![1, 2, 3]);
    att.check_layout().expect("3-attestor sig should pass");

    // Bad: 130 bytes (= 2 * 65, OK).
    let two = CctpAttestation::new(vec![0xaa; 130], vec![1, 2, 3]);
    two.check_layout().expect("2-attestor sig should pass");

    // 131 is not a multiple of 65.
    let bad = CctpAttestation::new(vec![0xaa; 131], vec![1]);
    assert!(bad.check_layout().is_err());
}

#[test]
fn cannot_decode_truncated() {
    let res = CctpBurnMessage::decode(&[0u8; 100]);
    assert!(res.is_err());
}
