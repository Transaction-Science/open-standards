//! Integration test: NIP-44 v2 versioned encryption.

use smart_byte_nostr::nip44;
use smart_byte_nostr::NostrSecretKey;

#[test]
fn nip44_roundtrip_short_and_long() {
    let alice = NostrSecretKey::generate();
    let bob = NostrSecretKey::generate();
    for msg in [
        b"hi".to_vec(),
        b"a slightly longer message".to_vec(),
        vec![0x42u8; 500],
    ] {
        let ct = nip44::encrypt(&alice, &bob.public_key(), &msg).expect("enc");
        let pt = nip44::decrypt(&bob, &alice.public_key(), &ct).expect("dec");
        assert_eq!(pt, msg);
    }
}

#[test]
fn tamper_blob_fails_mac() {
    let alice = NostrSecretKey::generate();
    let bob = NostrSecretKey::generate();
    let ct = nip44::encrypt(&alice, &bob.public_key(), b"hi").expect("enc");
    // Flip a character in the base64 payload (still valid base64, but
    // the underlying bytes will differ) so the MAC check fails.
    let mut bytes = ct.into_bytes();
    let i = bytes.len() / 2;
    bytes[i] = if bytes[i] == b'A' { b'B' } else { b'A' };
    let tampered = String::from_utf8(bytes).expect("utf8");
    assert!(nip44::decrypt(&bob, &alice.public_key(), &tampered).is_err());
}

#[test]
fn wrong_recipient_fails() {
    let alice = NostrSecretKey::generate();
    let bob = NostrSecretKey::generate();
    let eve = NostrSecretKey::generate();
    let ct = nip44::encrypt(&alice, &bob.public_key(), b"hello bob").expect("enc");
    assert!(nip44::decrypt(&eve, &alice.public_key(), &ct).is_err());
}
