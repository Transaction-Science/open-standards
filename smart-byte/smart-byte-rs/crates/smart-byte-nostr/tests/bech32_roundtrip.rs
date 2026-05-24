//! Integration test: NIP-19 bech32 encodings.

use smart_byte_nostr::bech32::{
    NEvent, NProfile, NRelay, decode_nevent, decode_note, decode_nprofile, decode_npub,
    decode_nrelay, decode_nsec, encode_nevent, encode_note, encode_nprofile, encode_npub,
    encode_nrelay, encode_nsec,
};

#[test]
fn npub_nsec_note_roundtrip() {
    let pk = [0x11u8; 32];
    let sk = [0x22u8; 32];
    let note = [0x33u8; 32];
    assert_eq!(decode_npub(&encode_npub(&pk).expect("enc")).expect("dec"), pk);
    assert_eq!(decode_nsec(&encode_nsec(&sk).expect("enc")).expect("dec"), sk);
    assert_eq!(decode_note(&encode_note(&note).expect("enc")).expect("dec"), note);
}

#[test]
fn nprofile_roundtrip_with_relays() {
    let p = NProfile {
        pubkey: [0x44u8; 32],
        relays: vec!["wss://a.relay".into(), "wss://b.relay".into()],
    };
    let s = encode_nprofile(&p).expect("enc");
    let d = decode_nprofile(&s).expect("dec");
    assert_eq!(d, p);
}

#[test]
fn nevent_roundtrip_full() {
    let e = NEvent {
        event_id: [0x55u8; 32],
        relays: vec!["wss://relay.example".into()],
        author: Some([0x66u8; 32]),
        kind: Some(1),
    };
    let s = encode_nevent(&e).expect("enc");
    let d = decode_nevent(&s).expect("dec");
    assert_eq!(d, e);
}

#[test]
fn nrelay_roundtrip() {
    let r = NRelay {
        url: "wss://only.relay".into(),
    };
    let s = encode_nrelay(&r).expect("enc");
    let d = decode_nrelay(&s).expect("dec");
    assert_eq!(d, r);
}

#[test]
fn hrp_mismatch_is_rejected() {
    let pk = [0x77u8; 32];
    let s = encode_npub(&pk).expect("enc");
    // Trying to decode an npub as an nsec should fail.
    assert!(decode_nsec(&s).is_err());
}
