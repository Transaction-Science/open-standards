//! Integration test: NIP-17 gift-wrapped private DM.

use smart_byte_nostr::nip17::{KIND_GIFT_WRAP, Rumor, unwrap, wrap};
use smart_byte_nostr::NostrSecretKey;

#[test]
fn dm_roundtrip_via_gift_wrap() {
    let alice = NostrSecretKey::generate();
    let bob = NostrSecretKey::generate();
    let rumor = Rumor::new(
        &alice.public_key(),
        &bob.public_key(),
        "private hello",
        1_700_000_000,
    );
    let wrapped = wrap(&alice, &bob.public_key(), &rumor, 1_700_000_000).expect("wrap");
    assert_eq!(wrapped.kind, KIND_GIFT_WRAP);
    // The wrap event is signed by an ephemeral key, NOT Alice's key.
    assert_ne!(wrapped.pubkey, alice.public_key().to_hex());

    let recovered = unwrap(&bob, &wrapped).expect("unwrap");
    assert_eq!(recovered.pubkey, alice.public_key().to_hex());
    assert_eq!(recovered.content, "private hello");
}

#[test]
fn third_party_cannot_unwrap() {
    let alice = NostrSecretKey::generate();
    let bob = NostrSecretKey::generate();
    let eve = NostrSecretKey::generate();
    let rumor = Rumor::new(
        &alice.public_key(),
        &bob.public_key(),
        "for bob only",
        1_700_000_000,
    );
    let wrapped = wrap(&alice, &bob.public_key(), &rumor, 1_700_000_000).expect("wrap");
    assert!(unwrap(&eve, &wrapped).is_err());
}
