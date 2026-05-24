//! End-to-end HTTP Signature round-trip.

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use smart_byte_activitypub::{
    sign_ed25519, verify_ed25519, Digest, Result, SignatureParams, SigningString,
};

#[test]
fn sign_and_verify_post_inbox() -> Result<()> {
    let mut csprng = OsRng;
    let key = SigningKey::generate(&mut csprng);

    let body = br#"{"type":"Create","actor":"https://a.test/users/alice"}"#;
    let digest = Digest::sha256(body);

    let signing = SigningString::build(
        "POST",
        "/users/bob/inbox",
        &[
            ("Host", "b.test"),
            ("Date", "Tue, 20 May 2025 14:00:00 GMT"),
            ("Digest", digest.header_value()),
        ],
        &["(request-target)", "host", "date", "digest"],
    )?;
    let sig_b64 = sign_ed25519(&key, &signing);

    let params = SignatureParams {
        key_id: "https://a.test/users/alice#main-key".to_string(),
        algorithm: "ed25519".to_string(),
        headers: signing.headers.clone(),
        signature: sig_b64.clone(),
    };
    let header = params.to_header();

    // Verifier reparses the header and rebuilds the signing string.
    let parsed = SignatureParams::parse(&header)?;
    let to_verify = SigningString::build(
        "POST",
        "/users/bob/inbox",
        &[
            ("Host", "b.test"),
            ("Date", "Tue, 20 May 2025 14:00:00 GMT"),
            ("Digest", digest.header_value()),
        ],
        parsed
            .headers
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    verify_ed25519(&key.verifying_key(), &to_verify, &parsed.signature)?;

    // Digest also verifies against the body.
    assert!(digest.verify(body));
    Ok(())
}

#[test]
fn parsing_supports_signature_prefix() -> Result<()> {
    let raw = "Signature keyId=\"https://x.test/k#main\",algorithm=\"ed25519\",headers=\"(request-target) host date\",signature=\"AAAA\"";
    let parsed = SignatureParams::parse(raw)?;
    assert_eq!(parsed.key_id, "https://x.test/k#main");
    assert_eq!(
        parsed.headers,
        vec![
            "(request-target)".to_string(),
            "host".to_string(),
            "date".to_string(),
        ]
    );
    Ok(())
}

#[test]
fn tampered_body_fails_digest() {
    let body = b"the original body";
    let digest = Digest::sha256(body);
    let tampered = b"a different body";
    assert!(!digest.verify(tampered));
}
