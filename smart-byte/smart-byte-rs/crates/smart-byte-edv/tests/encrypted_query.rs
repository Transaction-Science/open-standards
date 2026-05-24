//! Insert documents into an in-memory vault and prove that equality
//! queries over HMAC-blinded index tags return the expected matches —
//! without the vault ever seeing plaintext attribute values.

use smart_byte_edv::index::{IndexKey, Query};
use smart_byte_edv::jwe::{KeyPair, Recipient, wrap};
use smart_byte_edv::spec::{Config, EncryptedDocument, KeyDescriptor};
use smart_byte_edv::vault::{InMemoryVault, Vault};

fn cfg() -> Config {
    let kex = KeyDescriptor {
        id: "did:example:alice#kex-1".into(),
        key_type: "JsonWebKey2020".into(),
        controller: None,
    };
    let hmac = KeyDescriptor {
        id: "did:example:alice#hmac-1".into(),
        key_type: "Sha256HmacKey2019".into(),
        controller: None,
    };
    Config::new("urn:uuid:vault", "did:example:alice", kex, hmac)
}

fn encrypted_doc(
    id: &str,
    plaintext: &[u8],
    recipient: &Recipient,
    index: &IndexKey,
    attrs: &[(&str, &str)],
) -> EncryptedDocument {
    let jwe = wrap(plaintext, std::slice::from_ref(recipient)).expect("wrap");
    let entry = index.entry(attrs).expect("entry");
    EncryptedDocument {
        id: id.into(),
        sequence: 0,
        jwe: serde_json::to_value(&jwe).expect("jwe"),
        indexed: vec![entry],
        stream: None,
    }
}

#[tokio::test]
async fn equality_query_returns_only_matching_docs() {
    let vault = InMemoryVault::new(cfg());

    let bob = KeyPair::generate().expect("bob");
    let recipient = Recipient {
        kid: "did:example:bob#kex-1".into(),
        public: bob.public.clone(),
    };
    let index_key = IndexKey {
        kid: "did:example:alice#hmac-1".into(),
        key: [42u8; 32],
    };

    let a = encrypted_doc(
        "urn:doc:a",
        b"note one",
        &recipient,
        &index_key,
        &[("type", "Note"), ("folder", "inbox")],
    );
    let b = encrypted_doc(
        "urn:doc:b",
        b"note two",
        &recipient,
        &index_key,
        &[("type", "Note"), ("folder", "archive")],
    );
    let c = encrypted_doc(
        "urn:doc:c",
        b"photo",
        &recipient,
        &index_key,
        &[("type", "Photo"), ("folder", "inbox")],
    );

    vault.insert(a).await.expect("insert a");
    vault.insert(b).await.expect("insert b");
    vault.insert(c).await.expect("insert c");

    let q = Query::new()
        .equal(&index_key, "type", "Note")
        .expect("q1");
    let hits = vault.query(&q).await.expect("query");
    let ids: Vec<&str> = hits.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids, vec!["urn:doc:a", "urn:doc:b"]);

    let q2 = Query::new()
        .equal(&index_key, "folder", "inbox")
        .expect("q2");
    let hits2 = vault.query(&q2).await.expect("query");
    let ids2: Vec<&str> = hits2.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids2, vec!["urn:doc:a", "urn:doc:c"]);

    let q3 = Query::new()
        .equal(&index_key, "type", "Note")
        .expect("q3a")
        .equal(&index_key, "folder", "archive")
        .expect("q3b");
    let hits3 = vault.query(&q3).await.expect("query");
    let ids3: Vec<&str> = hits3.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids3, vec!["urn:doc:b"]);
}

#[tokio::test]
async fn vault_sees_only_hmac_tags() {
    // Sanity-check that the literal plaintext attribute names and values
    // never appear in the serialised on-disk form.
    let vault = InMemoryVault::new(cfg());
    let bob = KeyPair::generate().expect("bob");
    let recipient = Recipient {
        kid: "did:example:bob#kex-1".into(),
        public: bob.public.clone(),
    };
    let index_key = IndexKey {
        kid: "did:example:alice#hmac-1".into(),
        key: [42u8; 32],
    };
    let doc = encrypted_doc(
        "urn:doc:secret",
        b"top secret",
        &recipient,
        &index_key,
        &[("classification", "TopSecret")],
    );
    vault.insert(doc).await.expect("insert");

    let listed = vault.list().await.expect("list");
    assert_eq!(listed.len(), 1);
    let stored = vault.get("urn:doc:secret").await.expect("get");
    let wire = serde_json::to_string(&stored).expect("serialize");
    assert!(!wire.contains("TopSecret"));
    assert!(!wire.contains("classification"));
    assert!(!wire.contains("top secret"));
}
