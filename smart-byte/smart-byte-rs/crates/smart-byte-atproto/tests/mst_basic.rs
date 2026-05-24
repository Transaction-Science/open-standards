//! Basic MST insert / delete / lookup invariants.

use smart_byte_atproto::car::Cid;
use smart_byte_atproto::{Mst, MstEntry};

fn cid(b: &[u8]) -> Cid {
    Cid::dag_cbor(b)
}

#[test]
fn insert_lookup_delete() {
    let mut m = Mst::new();
    m.insert("app.bsky.feed.post/aaa", cid(b"A"));
    m.insert("app.bsky.feed.post/bbb", cid(b"B"));
    m.insert("app.bsky.feed.like/ccc", cid(b"C"));
    assert_eq!(m.len(), 3);
    assert_eq!(m.lookup("app.bsky.feed.post/aaa"), Some(&cid(b"A")));
    let prev = m.delete("app.bsky.feed.post/bbb");
    assert_eq!(prev, Some(cid(b"B")));
    assert_eq!(m.len(), 2);
    assert!(m.lookup("app.bsky.feed.post/bbb").is_none());
}

#[test]
fn iter_is_sorted() {
    let mut m = Mst::new();
    m.insert("z/2", cid(b"two"));
    m.insert("a/1", cid(b"one"));
    m.insert("m/3", cid(b"three"));
    let keys: Vec<String> =
        m.iter().map(|e: MstEntry| e.key).collect();
    assert_eq!(keys, vec!["a/1", "m/3", "z/2"]);
}

#[test]
fn root_hash_changes_on_update() {
    let mut m = Mst::new();
    m.insert("k/1", cid(b"v1"));
    let h1 = m.root_hash();
    m.insert("k/1", cid(b"v2"));
    let h2 = m.root_hash();
    assert_ne!(h1, h2);
}

#[test]
fn root_hex_is_64_chars() {
    let mut m = Mst::new();
    m.insert("k/1", cid(b"v"));
    assert_eq!(m.root_hex().len(), 64);
}
