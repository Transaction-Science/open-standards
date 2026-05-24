//! End-to-end JWE round-trip through the EncryptedDocument wire format.

use smart_byte_edv::jwe::{KeyPair, PrivateKey, Recipient, unwrap, wrap};
use smart_byte_edv::spec::EncryptedDocument;

#[test]
fn document_jwe_round_trip() {
    let bob = KeyPair::generate().expect("bob");
    let recipient = Recipient {
        kid: "did:example:bob#kex-1".into(),
        public: bob.public.clone(),
    };
    let plaintext = b"the quick brown fox jumps over the lazy dog";
    let jwe = wrap(plaintext, &[recipient]).expect("wrap");

    // Embed the JWE in an EncryptedDocument and round-trip through JSON.
    let doc = EncryptedDocument {
        id: "urn:uuid:11111111-2222-3333-4444-555555555555".into(),
        sequence: 1,
        jwe: serde_json::to_value(&jwe).expect("jwe as value"),
        indexed: Vec::new(),
        stream: None,
    };
    let wire = serde_json::to_string(&doc).expect("doc serialize");
    let restored: EncryptedDocument =
        serde_json::from_str(&wire).expect("doc deserialize");

    // Pull the JWE back out and decrypt.
    let jwe2: smart_byte_edv::jwe::Jwe =
        serde_json::from_value(restored.jwe).expect("jwe from value");
    let pk = PrivateKey {
        kid: "did:example:bob#kex-1".into(),
        secret: bob.secret,
    };
    let recovered = unwrap(&jwe2, &[pk]).expect("unwrap");
    assert_eq!(recovered, plaintext);
}

#[test]
fn multi_recipient_jwe_in_document() {
    let alice = KeyPair::generate().expect("alice");
    let bob = KeyPair::generate().expect("bob");
    let plaintext = b"both can read this";
    let recipients = vec![
        Recipient {
            kid: "did:example:alice#kex-1".into(),
            public: alice.public.clone(),
        },
        Recipient {
            kid: "did:example:bob#kex-1".into(),
            public: bob.public.clone(),
        },
    ];
    let jwe = wrap(plaintext, &recipients).expect("wrap");
    let doc = EncryptedDocument {
        id: "urn:uuid:multi".into(),
        sequence: 1,
        jwe: serde_json::to_value(&jwe).expect("jwe as value"),
        indexed: Vec::new(),
        stream: None,
    };
    let restored = serde_json::to_string(&doc)
        .and_then(|s| serde_json::from_str::<EncryptedDocument>(&s))
        .expect("round trip");
    let jwe2: smart_byte_edv::jwe::Jwe =
        serde_json::from_value(restored.jwe).expect("jwe");
    // Both Alice and Bob should be able to decrypt.
    let pt_a = unwrap(
        &jwe2,
        &[PrivateKey {
            kid: "did:example:alice#kex-1".into(),
            secret: alice.secret,
        }],
    )
    .expect("alice");
    assert_eq!(pt_a, plaintext);
    let pt_b = unwrap(
        &jwe2,
        &[PrivateKey {
            kid: "did:example:bob#kex-1".into(),
            secret: bob.secret,
        }],
    )
    .expect("bob");
    assert_eq!(pt_b, plaintext);
}
