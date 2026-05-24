//! AT URI ↔ AID bridge roundtrips and SAID determinism.

use smart_byte_atproto::{Aid, AtUri};

#[test]
fn at_uri_to_aid_to_at_uri() {
    let s = "at://did:plc:abcd1234/app.bsky.feed.post/3kjp";
    let uri: AtUri = s.parse().unwrap();
    let aid = uri.to_aid();
    let back = aid.to_at_uri();
    assert_eq!(back, uri);
    assert_eq!(back.to_string(), s);
}

#[test]
fn aid_canonical_roundtrip() {
    let aid = Aid {
        authority: "did:plc:abcd1234".into(),
        collection: Some("app.bsky.feed.like".into()),
        rkey: Some("xyz".into()),
    };
    let canonical = aid.canonical();
    let parsed: Aid = canonical.parse().unwrap();
    assert_eq!(parsed, aid);
}

#[test]
fn at_uri_said_is_deterministic() {
    let a: AtUri = "at://did:plc:abc/app.bsky.feed.post/r1".parse().unwrap();
    let b: AtUri = "at://did:plc:abc/app.bsky.feed.post/r1".parse().unwrap();
    assert_eq!(a.to_said(), b.to_said());
    let c: AtUri = "at://did:plc:abc/app.bsky.feed.post/r2".parse().unwrap();
    assert_ne!(a.to_said(), c.to_said());
}

#[test]
fn rejects_empty_collection() {
    let r: Result<AtUri, _> = "at://did:plc:abc//rkey".parse();
    assert!(r.is_err());
}

#[test]
fn handle_authority_is_allowed() {
    let u: AtUri = "at://alice.bsky.social/app.bsky.feed.post/3k"
        .parse()
        .unwrap();
    assert_eq!(u.authority, "alice.bsky.social");
}
