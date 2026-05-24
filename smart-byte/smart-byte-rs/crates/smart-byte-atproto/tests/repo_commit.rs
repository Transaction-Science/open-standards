//! Build a repo, sign a commit, export to CAR, decode back, verify.

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use smart_byte_atproto::car::{CarFile, Cid};
use smart_byte_atproto::{Repo, SignedCommit};

#[test]
fn sign_then_verify() {
    let mut csprng = OsRng;
    let key = SigningKey::generate(&mut csprng);
    let mut repo = Repo::new("did:plc:repotest0000000000000000");
    repo.put_record("app.bsky.feed.post/aaa", Cid::dag_cbor(b"first"))
        .unwrap();
    repo.put_record("app.bsky.feed.post/bbb", Cid::dag_cbor(b"second"))
        .unwrap();
    let signed = repo.sign_commit(&key).unwrap();
    signed.verify(&key.verifying_key()).unwrap();
    assert!(repo.head.is_some());
}

#[test]
fn wrong_key_fails_verify() {
    let mut csprng = OsRng;
    let key = SigningKey::generate(&mut csprng);
    let other = SigningKey::generate(&mut csprng);
    let mut repo = Repo::new("did:plc:repotest0000000000000000");
    repo.put_record("c/1", Cid::dag_cbor(b"hi")).unwrap();
    let signed = repo.sign_commit(&key).unwrap();
    assert!(signed.verify(&other.verifying_key()).is_err());
}

#[test]
fn repo_exports_to_car_with_commit_root() {
    let mut csprng = OsRng;
    let key = SigningKey::generate(&mut csprng);
    let mut repo = Repo::new("did:plc:repotest0000000000000000");
    let rec_cid = Cid::dag_cbor(b"payload");
    repo.put_record("c/1", rec_cid.clone()).unwrap();
    let signed = repo.sign_commit(&key).unwrap();
    let car = repo
        .to_car(&signed, vec![(rec_cid.clone(), b"payload".to_vec())])
        .unwrap();
    let encoded = car.encode().unwrap();
    let decoded = CarFile::decode(&encoded).unwrap();
    assert_eq!(decoded.roots.len(), 1);
    let commit_cid = signed.cid().unwrap();
    assert_eq!(decoded.roots[0], commit_cid);
    // commit + payload
    assert_eq!(decoded.blocks.len(), 2);
    // Re-deserialise the commit and reverify.
    let commit_block = decoded.get(&commit_cid).unwrap();
    let parsed: SignedCommit =
        serde_cbor::from_slice(&commit_block.data).unwrap();
    parsed.verify(&key.verifying_key()).unwrap();
}
