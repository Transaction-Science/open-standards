//! ZCAP-LD capability chain validation: prove delegation narrows
//! permissions, that an unauthorised invocation is rejected, and that an
//! expired root invalidates every descendant.

use chrono::{Duration, Utc};
use smart_byte_edv::error::EdvError;
use smart_byte_edv::zcap::{Capability, Invocation, verify_chain};

#[test]
fn three_link_chain_authorises_intended_action() {
    // alice (controller) -> bob (auditor) -> carol (read-only delegate).
    let root = Capability::root(
        "urn:cap:root",
        "https://vault/example/docs/urn:doc:1",
        "did:example:alice",
        "did:example:bob",
        vec!["read".into(), "write".into(), "delete".into()],
    );
    let mid = root
        .delegate(
            "urn:cap:mid",
            "did:example:bob",
            "did:example:carol",
            vec!["read".into(), "write".into()],
        )
        .expect("delegate mid");
    let leaf = mid
        .delegate(
            "urn:cap:leaf",
            "did:example:carol",
            "did:example:dan",
            vec!["read".into()],
        )
        .expect("delegate leaf");

    let inv = Invocation::new(&leaf, "read");
    verify_chain(
        &[root.clone(), mid.clone(), leaf.clone()],
        "did:example:alice",
        &inv,
    )
    .expect("read should be allowed");

    // dan does not get write — leaf was narrowed to "read".
    let bad = Invocation::new(&leaf, "write");
    let res = verify_chain(&[root, mid, leaf], "did:example:alice", &bad);
    assert!(matches!(res, Err(EdvError::Unauthorized(_, _))));
}

#[test]
fn delegation_cannot_widen_actions() {
    let root = Capability::root(
        "urn:cap:root",
        "https://vault/example",
        "did:example:alice",
        "did:example:bob",
        vec!["read".into()],
    );
    let res = root.delegate(
        "urn:cap:child",
        "did:example:bob",
        "did:example:mallory",
        vec!["read".into(), "delete".into()],
    );
    assert!(matches!(res, Err(EdvError::Capability(_))));
}

#[test]
fn broken_chain_link_is_rejected() {
    let root = Capability::root(
        "urn:cap:root",
        "https://vault/example",
        "did:example:alice",
        "did:example:bob",
        vec!["read".into()],
    );
    let mut child = root
        .delegate(
            "urn:cap:child",
            "did:example:bob",
            "did:example:carol",
            vec!["read".into()],
        )
        .expect("delegate");
    // Snap the chain by pointing to a non-existent parent.
    child.parent_capability = Some("urn:cap:does-not-exist".into());
    let inv = Invocation::new(&child, "read");
    let res = verify_chain(&[root, child], "did:example:alice", &inv);
    assert!(matches!(res, Err(EdvError::Capability(_))));
}

#[test]
fn expired_root_invalidates_chain() {
    let mut root = Capability::root(
        "urn:cap:root",
        "https://vault/example",
        "did:example:alice",
        "did:example:bob",
        vec!["read".into()],
    );
    root.expires = Some(Utc::now() - Duration::minutes(1));
    let child = root
        .delegate(
            "urn:cap:child",
            "did:example:bob",
            "did:example:carol",
            vec!["read".into()],
        )
        .expect("delegate");
    let inv = Invocation::new(&child, "read");
    let res = verify_chain(&[root, child], "did:example:alice", &inv);
    assert!(matches!(res, Err(EdvError::Capability(_))));
}

#[test]
fn wrong_root_controller_rejected() {
    let root = Capability::root(
        "urn:cap:root",
        "https://vault/example",
        "did:example:alice",
        "did:example:bob",
        vec!["read".into()],
    );
    let inv = Invocation::new(&root, "read");
    let res = verify_chain(&[root], "did:example:mallory", &inv);
    assert!(matches!(res, Err(EdvError::Capability(_))));
}
