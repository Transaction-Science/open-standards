//! Integration test: full event sign + verify with multiple kinds and
//! tag shapes, plus tamper detection.

use smart_byte_nostr::{NostrSecretKey, UnsignedEvent};

#[test]
fn many_kinds_sign_and_verify() {
    for kind in [0u32, 1, 4, 5, 1059, 10002] {
        let sk = NostrSecretKey::generate();
        let pk = sk.public_key();
        let ev = UnsignedEvent::new(pk, kind, format!("payload for {kind}"), 1_700_000_000)
            .with_tag(vec!["p".into(), pk.to_hex()])
            .with_tag(vec!["t".into(), "smart-byte".into()])
            .sign(&sk)
            .expect("sign");
        ev.verify().expect("verify");
        assert_eq!(ev.kind, kind);
    }
}

#[test]
fn tag_reorder_changes_id() {
    let sk = NostrSecretKey::generate();
    let pk = sk.public_key();
    let a = UnsignedEvent::new(pk, 1, "hi", 1_700_000_000)
        .with_tag(vec!["t".into(), "a".into()])
        .with_tag(vec!["t".into(), "b".into()])
        .sign(&sk)
        .expect("sign");
    let b = UnsignedEvent::new(pk, 1, "hi", 1_700_000_000)
        .with_tag(vec!["t".into(), "b".into()])
        .with_tag(vec!["t".into(), "a".into()])
        .sign(&sk)
        .expect("sign");
    assert_ne!(a.id, b.id);
}

#[test]
fn wrong_key_fails_verify() {
    let sk = NostrSecretKey::generate();
    let bad_pk = NostrSecretKey::generate().public_key();
    let mut ev = UnsignedEvent::new(sk.public_key(), 1, "hi", 1_700_000_000)
        .sign(&sk)
        .expect("sign");
    // Re-point the event to a different pubkey but keep the same id/sig.
    ev.pubkey = bad_pk.to_hex();
    assert!(ev.verify().is_err());
}
