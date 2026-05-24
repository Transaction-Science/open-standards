//! Status list round-trips: Token Status List (IETF) + Bitstring (W3C).

use smart_byte_oidc4vc::{
    BitstringStatusList, STATUS_INVALID, STATUS_SUSPENDED, STATUS_VALID,
    TokenStatusList, TokenStatusListBytes, check_bitstring_status,
};

#[test]
fn token_status_list_one_bit() {
    let mut l = TokenStatusListBytes::new(1, 8192).expect("list");
    l.set(0, STATUS_INVALID).unwrap();
    l.set(4095, STATUS_INVALID).unwrap();
    l.set(8191, STATUS_INVALID).unwrap();
    let encoded = TokenStatusList::encode(&l).unwrap();
    let decoded = encoded.decode(Some(l.bytes.len())).unwrap();
    assert_eq!(decoded.get(0).unwrap(), STATUS_INVALID);
    assert_eq!(decoded.get(4095).unwrap(), STATUS_INVALID);
    assert_eq!(decoded.get(8191).unwrap(), STATUS_INVALID);
    assert_eq!(decoded.get(1).unwrap(), STATUS_VALID);
    assert_eq!(decoded.get(123).unwrap(), STATUS_VALID);
}

#[test]
fn token_status_list_two_bit_values() {
    let mut l = TokenStatusListBytes::new(2, 64).unwrap();
    l.set(0, STATUS_INVALID).unwrap();
    l.set(1, STATUS_SUSPENDED).unwrap();
    let enc = TokenStatusList::encode(&l).unwrap();
    let back = enc.decode(Some(l.bytes.len())).unwrap();
    assert_eq!(back.get(0).unwrap(), STATUS_INVALID);
    assert_eq!(back.get(1).unwrap(), STATUS_SUSPENDED);
    assert_eq!(back.get(2).unwrap(), STATUS_VALID);
}

#[test]
fn token_status_list_rejects_overflow() {
    let mut l = TokenStatusListBytes::new(1, 8).unwrap();
    assert!(l.set(0, 2).is_err());
}

#[test]
fn bitstring_status_list_passthrough() {
    let mut bs = BitstringStatusList::new(1024);
    bs.set(42, true).unwrap();
    let encoded = bs.to_encoded().unwrap();
    assert!(check_bitstring_status(&encoded, 1024, 42).unwrap());
    assert!(!check_bitstring_status(&encoded, 1024, 41).unwrap());
}
