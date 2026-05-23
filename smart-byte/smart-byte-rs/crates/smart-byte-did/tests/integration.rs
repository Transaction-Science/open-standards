//! Integration tests covering all four DID methods, the universal
//! resolver, DID URL dereferencing, and negative cases.

use smart_byte_did::methods::jwk::encode_did_jwk;
use smart_byte_did::methods::key::{KeyType, decode_did_key, encode_did_key};
use smart_byte_did::methods::peer::encode_numalgo2;
use smart_byte_did::methods::web::WebResolver;
use smart_byte_did::{
    DereferenceResult, Did, DidError, DidMethod, DidUrl, Jwk, Resolver,
    UniversalResolver, dereference,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ED25519_FIXTURE_DID: &str =
    "did:key:z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp";

#[tokio::test]
async fn did_key_resolves_offline() {
    let resolver = UniversalResolver::new();
    let did: Did = ED25519_FIXTURE_DID.parse().unwrap();
    let result = resolver.resolve(&did).await.unwrap();
    let doc = result.did_document.unwrap();
    assert_eq!(doc.id, did);
    assert_eq!(doc.verification_method.len(), 1);
    assert!(
        doc.verification_method[0]
            .public_key_multibase
            .as_ref()
            .unwrap()
            .starts_with('z')
    );
}

#[tokio::test]
async fn did_key_round_trip_p256() {
    // Deterministic scalar.
    let scalar = [
        0x42u8, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];
    use p256::elliptic_curve::sec1::ToSec1Point;
    let sk = p256::SecretKey::from_slice(&scalar).unwrap();
    let pt = sk.public_key().to_sec1_point(true);
    let raw = pt.as_bytes().to_vec();
    let msid = encode_did_key(KeyType::P256, &raw).unwrap();
    let (kind, decoded) = decode_did_key(&msid).unwrap();
    assert_eq!(kind, KeyType::P256);
    assert_eq!(decoded, raw);
}

#[tokio::test]
async fn did_key_round_trip_secp256k1() {
    let scalar = [
        0x33u8, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
        0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
        0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
    ];
    use k256::elliptic_curve::sec1::ToSec1Point;
    let sk = k256::SecretKey::from_slice(&scalar).unwrap();
    let pt = sk.public_key().to_sec1_point(true);
    let raw = pt.as_bytes().to_vec();
    let msid = encode_did_key(KeyType::Secp256k1, &raw).unwrap();
    let (kind, decoded) = decode_did_key(&msid).unwrap();
    assert_eq!(kind, KeyType::Secp256k1);
    assert_eq!(decoded, raw);
}

#[tokio::test]
async fn did_web_resolves_via_wiremock() {
    let server = MockServer::start().await;
    // Build a fake DID whose method-specific id points at the mock host.
    let host = server.address();
    let host_str = format!(
        "{}%3A{}",
        host.ip(),
        host.port()
    );
    let did_str = format!("did:web:{host_str}");
    let did: Did = did_str.parse().unwrap();
    let doc_json = serde_json::json!({
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": did_str,
        "verificationMethod": [{
            "id": format!("{did_str}#keys-1"),
            "type": "Multikey",
            "controller": did_str,
            "publicKeyMultibase": "z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp",
        }],
        "authentication": [format!("{did_str}#keys-1")],
    });
    Mock::given(method("GET"))
        .and(path("/.well-known/did.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                doc_json.to_string(),
                "application/did+json",
            ),
        )
        .mount(&server)
        .await;
    let resolver = WebResolver::with_scheme("http");
    let result = resolver.resolve(&did).await.unwrap();
    let doc = result.did_document.unwrap();
    assert_eq!(doc.id, did);
    assert_eq!(doc.verification_method.len(), 1);
}

#[tokio::test]
async fn did_web_404_returns_not_found() {
    let server = MockServer::start().await;
    let host = server.address();
    let did_str = format!("did:web:{}%3A{}", host.ip(), host.port());
    let did: Did = did_str.parse().unwrap();
    Mock::given(method("GET"))
        .and(path("/.well-known/did.json"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let resolver = WebResolver::with_scheme("http");
    let err = resolver.resolve(&did).await.unwrap_err();
    assert!(matches!(err, DidError::NotFound(_)));
}

#[tokio::test]
async fn did_web_id_mismatch_rejected() {
    let server = MockServer::start().await;
    let host = server.address();
    let did_str = format!("did:web:{}%3A{}", host.ip(), host.port());
    let did: Did = did_str.parse().unwrap();
    // Serve a document whose id field does NOT match the requested DID.
    let doc_json = serde_json::json!({
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": "did:web:wrong.example.com",
    });
    Mock::given(method("GET"))
        .and(path("/.well-known/did.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(
                doc_json.to_string(),
                "application/did+json",
            ),
        )
        .mount(&server)
        .await;
    let resolver = WebResolver::with_scheme("http");
    let err = resolver.resolve(&did).await.unwrap_err();
    assert!(matches!(err, DidError::InvalidDocument(_)));
}

#[tokio::test]
async fn did_peer_numalgo2_round_trip() {
    // Encode a one-key peer DID using an Ed25519 W3C fixture.
    let mk = "z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp";
    let msid = encode_numalgo2(&[('V', mk.into())], &[]);
    let did_str = format!("did:peer:{msid}");
    let did: Did = did_str.parse().unwrap();
    let resolver = UniversalResolver::new();
    let doc = resolver.resolve(&did).await.unwrap().did_document.unwrap();
    assert_eq!(doc.id, did);
    assert_eq!(doc.verification_method.len(), 1);
    assert_eq!(doc.authentication.len(), 1);
}

#[tokio::test]
async fn did_jwk_round_trip() {
    let jwk = Jwk {
        kty: "OKP".into(),
        crv: Some("Ed25519".into()),
        x: Some(
            "11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo".into(),
        ),
        y: None,
        alg: Some("EdDSA".into()),
        kid: None,
        use_: None,
    };
    let msid = encode_did_jwk(&jwk).unwrap();
    let did_str = format!("did:jwk:{msid}");
    let did: Did = did_str.parse().unwrap();
    let resolver = UniversalResolver::new();
    let doc = resolver.resolve(&did).await.unwrap().did_document.unwrap();
    assert_eq!(doc.id, did);
    assert_eq!(doc.verification_method.len(), 1);
    assert!(
        doc.verification_method[0]
            .public_key_jwk
            .as_ref()
            .unwrap()
            .kty
            == "OKP"
    );
}

#[tokio::test]
async fn dereference_did_url_to_verification_method() {
    let did_str = ED25519_FIXTURE_DID;
    let did: Did = did_str.parse().unwrap();
    let resolver = UniversalResolver::new();
    let doc = resolver.resolve(&did).await.unwrap().did_document.unwrap();
    // Use the VM id that the resolver synthesises.
    let vm_id = &doc.verification_method[0].id;
    // Extract just the fragment portion.
    let fragment = vm_id.split('#').nth(1).unwrap();
    let url_str = format!("{did_str}#{fragment}");
    let url: DidUrl = url_str.parse().unwrap();
    let result = dereference(&resolver, &url).await.unwrap();
    match result {
        DereferenceResult::VerificationMethod(vm) => {
            assert_eq!(vm.controller, did);
        }
        other => panic!("expected verification method, got {other:?}"),
    }
}

#[tokio::test]
async fn dereference_unknown_fragment_not_found() {
    let did: Did = ED25519_FIXTURE_DID.parse().unwrap();
    let url: DidUrl =
        format!("{did}#does-not-exist").parse().unwrap();
    let resolver = UniversalResolver::new();
    let err = dereference(&resolver, &url).await.unwrap_err();
    assert!(matches!(err, DidError::NotFound(_)));
}

#[tokio::test]
async fn malformed_identifier_rejected() {
    let r = UniversalResolver::new();
    // Custom method that isn't registered.
    let did = Did::new(DidMethod::Custom("nonesuch".into()), "abc").unwrap();
    let err = r.resolve(&did).await.unwrap_err();
    assert!(matches!(err, DidError::MethodNotSupported(_)));
}

#[tokio::test]
async fn did_key_rejects_garbage_key_bytes() {
    // 'z' + 'AAAA' → valid base58btc but no valid multicodec / key.
    let bad = "did:key:zAAAA";
    let did: Did = bad.parse().unwrap();
    let r = UniversalResolver::new();
    let err = r.resolve(&did).await.unwrap_err();
    assert!(matches!(
        err,
        DidError::UnsupportedKeyCodec(_)
            | DidError::InvalidKey(_)
            | DidError::InvalidIdentifier(_)
    ));
}

#[tokio::test]
async fn signature_method_mismatch_p256_bytes_under_ed25519_codec() {
    // 33 P-256-shaped bytes encoded under the Ed25519 codec → length mismatch.
    let mut buf = Vec::new();
    smart_byte_did_internal::varint_encode(0xed, &mut buf);
    buf.extend_from_slice(&[2u8; 33]); // wrong length for Ed25519 (expects 32).
    let msid = multibase::encode(multibase::Base::Base58Btc, &buf);
    let did: Did = format!("did:key:{msid}").parse().unwrap();
    let r = UniversalResolver::new();
    let err = r.resolve(&did).await.unwrap_err();
    assert!(matches!(err, DidError::InvalidKey(_)));
}

// Re-export the varint helper into a stable shim for the test above.
mod smart_byte_did_internal {
    pub fn varint_encode(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push(((value & 0x7f) as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }
}
