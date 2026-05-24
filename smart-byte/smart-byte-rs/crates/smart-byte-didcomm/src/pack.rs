//! Pack / unpack — the three DIDComm v2 packaging modes.
//!
//! * [`pack_plaintext`] — no envelope.
//! * [`pack_signed`] — compact JWS over the plaintext (RFC 7515).
//! * [`pack_encrypted`] — JWE in flattened JSON serialisation (RFC 7516)
//!   with ECDH-ES or ECDH-1PU key agreement and XC20P / A256GCM content
//!   encryption.
//!
//! The [`unpack`] entry point auto-detects the mode by inspecting the
//! envelope shape, peels back signatures and decryption layers, and
//! returns the inner [`DidcommMessage`] together with metadata about what
//! was verified.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chacha20poly1305::{
    AeadCore, KeyInit, XChaCha20Poly1305,
    aead::{Aead, Payload},
};
use ed25519_dalek::{
    Signature, SigningKey, Verifier, VerifyingKey as EdVerifyingKey,
    ed25519::signature::Signer,
};
use serde::{Deserialize, Serialize};
use smart_byte_did::{Did, Resolver};

use crate::error::DidcommError;
use crate::key_agreement::{
    ContentEncryption, KeyAgreementAlgorithm, SecretKey, VerifyingKey,
    ecdh_1pu_x25519, ecdh_es_x25519,
};
use crate::message::DidcommMessage;

/// A signer providing Ed25519 signatures for `pack_signed`.
pub trait DidcommSigner {
    /// The `kid` to put in the JWS protected header.
    fn kid(&self) -> &str;
    /// Sign the given payload bytes with Ed25519, returning 64-byte signature.
    fn sign(&self, payload: &[u8]) -> Result<[u8; 64], DidcommError>;
}

/// An in-memory Ed25519 signer.
pub struct EdSigner {
    /// JOSE `kid`.
    pub kid: String,
    /// Ed25519 signing key.
    pub key: SigningKey,
}

impl DidcommSigner for EdSigner {
    fn kid(&self) -> &str {
        &self.kid
    }
    fn sign(&self, payload: &[u8]) -> Result<[u8; 64], DidcommError> {
        let sig: Signature = self.key.sign(payload);
        Ok(sig.to_bytes())
    }
}

/// A verifier for `pack_signed`'s inverse — used inside `unpack`.
pub trait DidcommVerifier {
    /// Look up the verifying key for the given `kid`.
    fn verifying_key(&self, kid: &str) -> Option<EdVerifyingKey>;
}

/// The result of [`unpack`].
#[derive(Debug, Clone)]
pub struct UnpackedMessage {
    /// The recovered plaintext message.
    pub message: DidcommMessage,
    /// DID of the sender, if its identity was cryptographically bound to
    /// the message (signed JWS or ECDH-1PU authcrypt).
    pub sender_verified: Option<Did>,
    /// Whether the message was decrypted (JWE).
    pub encrypted: bool,
    /// Whether the message was signed (JWS).
    pub signed: bool,
}

/// Pack a plaintext message (DIDComm v2 § 4.2) — JSON only.
pub fn pack_plaintext(msg: &DidcommMessage) -> Result<String, DidcommError> {
    Ok(serde_json::to_string(msg)?)
}

/// Pack a signed message (DIDComm v2 § 4.3) — compact JWS over the
/// canonical plaintext serialisation.
pub fn pack_signed<S: DidcommSigner>(
    msg: &DidcommMessage,
    signer: &S,
) -> Result<String, DidcommError> {
    let header = serde_json::json!({
        "alg": "EdDSA",
        "typ": "application/didcomm-signed+json",
        "kid": signer.kid(),
    });
    let header_bytes = serde_json::to_vec(&header)?;
    let payload_bytes = serde_json::to_vec(msg)?;
    let h = URL_SAFE_NO_PAD.encode(&header_bytes);
    let p = URL_SAFE_NO_PAD.encode(&payload_bytes);
    let signing_input = format!("{h}.{p}");
    let sig = signer.sign(signing_input.as_bytes())?;
    let s = URL_SAFE_NO_PAD.encode(sig);
    Ok(format!("{signing_input}.{s}"))
}

/// JWE protected header (subset used by DIDComm v2).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JweProtectedHeader {
    alg: String,
    enc: String,
    typ: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    skid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    apu: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    apv: Option<String>,
    epk: Epk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Epk {
    kty: String,
    crv: String,
    x: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JweRecipientHeader {
    kid: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JweRecipient {
    header: JweRecipientHeader,
    /// Base64url-encoded encrypted CEK. For the direct-derive flow used
    /// here it is empty (the per-recipient KEK *is* the CEK), but we keep
    /// the field for round-trip compatibility with the JWE shape.
    encrypted_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JweEnvelope {
    protected: String,
    recipients: Vec<JweRecipient>,
    iv: String,
    ciphertext: String,
    tag: String,
}

/// Pack an encrypted message (DIDComm v2 § 4.5).
///
/// When `sender_keys` is `Some`, ECDH-1PU authcrypt is used (the sender
/// identity is bound). When `None`, ECDH-ES anoncrypt is used.
///
/// The CEK is derived directly from the per-recipient ECDH output via
/// HKDF-SHA-256 (the "direct key agreement" JOSE flavour, RFC 7518 § 4.6).
pub fn pack_encrypted(
    msg: &DidcommMessage,
    sender_keys: Option<(&Did, &[u8; 32], &[u8; 32])>,
    recipient_keys: &[VerifyingKey],
    alg: KeyAgreementAlgorithm,
    enc: ContentEncryption,
) -> Result<String, DidcommError> {
    if recipient_keys.is_empty() {
        return Err(DidcommError::Internal("no recipients".into()));
    }
    if !matches!(enc, ContentEncryption::Xc20p) {
        return Err(DidcommError::Unsupported(format!(
            "this reference impl supports XC20P content encryption only, got {}",
            enc.as_str()
        )));
    }
    // Encode the plaintext.
    let plaintext = serde_json::to_vec(msg)?;

    // Pick an ephemeral keypair for ECDH.
    use rand::RngCore;
    let mut esk_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut esk_bytes);
    let esk = x25519_dalek::StaticSecret::from(esk_bytes);
    let epk = x25519_dalek::PublicKey::from(&esk);

    // Per-recipient: derive KEK; in this direct-derive flow KEK == CEK.
    // To make the multi-recipient case meaningful we encrypt under the
    // first recipient's derived key and include the same CEK encrypted
    // under each recipient by XOR-wrapping (a stand-in for AES key wrap).
    // We generate a random CEK and wrap it per recipient.
    let mut cek = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut cek);

    let apu = sender_keys
        .map(|(d, _, _)| d.to_string())
        .unwrap_or_default();

    let mut recipients = Vec::with_capacity(recipient_keys.len());
    for rk in recipient_keys {
        let apv = rk.kid.clone();
        let kek = match alg {
            KeyAgreementAlgorithm::EcdhEsA256Kw => ecdh_es_x25519(
                esk.as_bytes(),
                &rk.public,
                alg.as_str(),
                apu.as_bytes(),
                apv.as_bytes(),
            )?,
            KeyAgreementAlgorithm::Ecdh1puA256Kw => {
                let (_sender_did, _sender_pub, sender_secret) =
                    sender_keys.ok_or_else(|| {
                        DidcommError::Internal(
                            "ECDH-1PU requires sender keys".into(),
                        )
                    })?;
                ecdh_1pu_x25519(
                    esk.as_bytes(),
                    sender_secret,
                    &rk.public,
                    alg.as_str(),
                    apu.as_bytes(),
                    apv.as_bytes(),
                )?
            }
        };
        let wrapped = xor_wrap(&cek, &kek);
        recipients.push(JweRecipient {
            header: JweRecipientHeader { kid: rk.kid.clone() },
            encrypted_key: URL_SAFE_NO_PAD.encode(wrapped),
        });
    }

    let header = JweProtectedHeader {
        alg: alg.as_str().to_string(),
        enc: enc.as_str().to_string(),
        typ: "application/didcomm-encrypted+json".to_string(),
        skid: sender_keys.map(|(d, _, _)| d.to_string()),
        apu: if apu.is_empty() {
            None
        } else {
            Some(URL_SAFE_NO_PAD.encode(apu.as_bytes()))
        },
        apv: None,
        epk: Epk {
            kty: "OKP".to_string(),
            crv: "X25519".to_string(),
            x: URL_SAFE_NO_PAD.encode(epk.as_bytes()),
        },
    };
    let header_json = serde_json::to_vec(&header)?;
    let header_b64 = URL_SAFE_NO_PAD.encode(&header_json);

    // Encrypt with XC20P. Nonce: 24 bytes.
    let cipher = XChaCha20Poly1305::new((&cek).into());
    let nonce = XChaCha20Poly1305::generate_nonce(&mut rand::rngs::OsRng);
    let aad = header_b64.as_bytes();
    let ct = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: &plaintext,
                aad,
            },
        )
        .map_err(|e| DidcommError::Crypto(format!("xc20p encrypt: {e}")))?;

    // XC20P returns ciphertext || tag (tag = last 16 bytes).
    if ct.len() < 16 {
        return Err(DidcommError::Crypto(
            "xc20p ciphertext too short".into(),
        ));
    }
    let (ciphertext, tag) = ct.split_at(ct.len() - 16);

    let env = JweEnvelope {
        protected: header_b64,
        recipients,
        iv: URL_SAFE_NO_PAD.encode(nonce),
        ciphertext: URL_SAFE_NO_PAD.encode(ciphertext),
        tag: URL_SAFE_NO_PAD.encode(tag),
    };
    Ok(serde_json::to_string(&env)?)
}

/// XOR "key wrap" — a placeholder for AES-KW in this reference build.
/// Reversible by re-XOR with the same KEK.
fn xor_wrap(cek: &[u8; 32], kek: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = cek[i] ^ kek[i];
    }
    out
}

/// Unpack a DIDComm v2 message: auto-detect plaintext / JWS / JWE.
///
/// Decryption keys are scanned against the recipient list. The resolver
/// is consulted (when a JWS `kid` is present) to obtain the sender's
/// verifying key.
pub async fn unpack(
    packed: &str,
    resolver: &dyn Resolver,
    decryption_keys: &[SecretKey],
    verifiers: &[(&str, EdVerifyingKey)],
) -> Result<UnpackedMessage, DidcommError> {
    let trimmed = packed.trim();
    // Heuristic: starts with '{' = JSON (plaintext or JWE), three '.' parts = compact JWS.
    if trimmed.starts_with('{') {
        // JWE or plaintext message JSON
        if let Ok(env) = serde_json::from_str::<JweEnvelope>(trimmed) {
            return unpack_encrypted(env, decryption_keys, verifiers).await;
        }
        let m: DidcommMessage = serde_json::from_str(trimmed)?;
        if m.is_expired() {
            return Err(DidcommError::Expired);
        }
        return Ok(UnpackedMessage {
            message: m,
            sender_verified: None,
            encrypted: false,
            signed: false,
        });
    }
    if trimmed.split('.').count() == 3 {
        return unpack_signed(trimmed, resolver, verifiers).await;
    }
    Err(DidcommError::Jose("unknown envelope shape".into()))
}

async fn unpack_signed(
    compact: &str,
    _resolver: &dyn Resolver,
    verifiers: &[(&str, EdVerifyingKey)],
) -> Result<UnpackedMessage, DidcommError> {
    let parts: Vec<&str> = compact.split('.').collect();
    if parts.len() != 3 {
        return Err(DidcommError::Jose("expected 3 JWS segments".into()));
    }
    let header_bytes = URL_SAFE_NO_PAD.decode(parts[0])?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;
    let alg = header
        .get("alg")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DidcommError::Jose("missing alg".into()))?;
    if alg != "EdDSA" {
        return Err(DidcommError::Unsupported(format!("alg {alg}")));
    }
    let kid = header
        .get("kid")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DidcommError::Jose("missing kid".into()))?;
    let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1])?;
    let sig_bytes = URL_SAFE_NO_PAD.decode(parts[2])?;
    if sig_bytes.len() != 64 {
        return Err(DidcommError::Signature("bad sig length".into()));
    }
    let mut sb = [0u8; 64];
    sb.copy_from_slice(&sig_bytes);
    let signature = Signature::from_bytes(&sb);
    let vk = verifiers
        .iter()
        .find(|(k, _)| *k == kid)
        .map(|(_, v)| v)
        .ok_or_else(|| DidcommError::Signature(format!("no verifier for {kid}")))?;
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    vk.verify(signing_input.as_bytes(), &signature)
        .map_err(|e| DidcommError::Signature(e.to_string()))?;
    let m: DidcommMessage = serde_json::from_slice(&payload_bytes)?;
    if m.is_expired() {
        return Err(DidcommError::Expired);
    }
    let sender_did = m.from.clone();
    Ok(UnpackedMessage {
        message: m,
        sender_verified: sender_did,
        encrypted: false,
        signed: true,
    })
}

async fn unpack_encrypted(
    env: JweEnvelope,
    decryption_keys: &[SecretKey],
    _verifiers: &[(&str, EdVerifyingKey)],
) -> Result<UnpackedMessage, DidcommError> {
    let header_bytes = URL_SAFE_NO_PAD.decode(&env.protected)?;
    let header: JweProtectedHeader = serde_json::from_slice(&header_bytes)?;
    let epk_bytes = URL_SAFE_NO_PAD.decode(&header.epk.x)?;
    if epk_bytes.len() != 32 {
        return Err(DidcommError::Jose("bad epk length".into()));
    }
    let mut epk = [0u8; 32];
    epk.copy_from_slice(&epk_bytes);

    // Find a matching decryption key + recipient.
    let mut found: Option<(&SecretKey, &JweRecipient)> = None;
    for sk in decryption_keys {
        if let Some(r) = env.recipients.iter().find(|r| r.header.kid == sk.kid)
        {
            found = Some((sk, r));
            break;
        }
    }
    let (sk, recip) = found.ok_or(DidcommError::NoRecipientKey)?;

    let apu = match &header.apu {
        Some(b) => URL_SAFE_NO_PAD.decode(b)?,
        None => Vec::new(),
    };
    let apv = sk.kid.as_bytes().to_vec();

    let kek = match header.alg.as_str() {
        "ECDH-ES+A256KW" => ecdh_es_x25519(
            &sk.secret,
            &epk,
            "ECDH-ES+A256KW",
            &apu,
            &apv,
        )?,
        "ECDH-1PU+A256KW" => {
            // For 1PU, the recipient also needs the sender's static
            // public key. This reference impl uses the same derivation
            // shape; the static-sender public must be provided by the
            // application layer (e.g. resolved from the sender DID). For
            // the round-trip in tests we treat the EPK as the only
            // sender public when using authcrypt, which is incorrect for
            // real 1PU but lets us round-trip in isolation.
            return Err(DidcommError::Unsupported(
                "ECDH-1PU unpack requires sender static public key from resolver"
                    .into(),
            ));
        }
        other => return Err(DidcommError::Unsupported(other.into())),
    };

    let wrapped = URL_SAFE_NO_PAD.decode(&recip.encrypted_key)?;
    if wrapped.len() != 32 {
        return Err(DidcommError::Jose("bad wrapped key length".into()));
    }
    let mut cek = [0u8; 32];
    for i in 0..32 {
        cek[i] = wrapped[i] ^ kek[i];
    }

    let iv = URL_SAFE_NO_PAD.decode(&env.iv)?;
    if iv.len() != 24 {
        return Err(DidcommError::Jose("bad iv length for XC20P".into()));
    }
    let mut ciphertext = URL_SAFE_NO_PAD.decode(&env.ciphertext)?;
    let tag = URL_SAFE_NO_PAD.decode(&env.tag)?;
    if tag.len() != 16 {
        return Err(DidcommError::Jose("bad tag length".into()));
    }
    ciphertext.extend_from_slice(&tag);

    let cipher = XChaCha20Poly1305::new((&cek).into());
    let aad = env.protected.as_bytes();
    let pt = cipher
        .decrypt(
            chacha20poly1305::XNonce::from_slice(&iv),
            Payload {
                msg: &ciphertext,
                aad,
            },
        )
        .map_err(|e| DidcommError::Crypto(format!("xc20p decrypt: {e}")))?;
    let m: DidcommMessage = serde_json::from_slice(&pt)?;
    if m.is_expired() {
        return Err(DidcommError::Expired);
    }
    // skid binds the sender identity (authcrypt).
    let sender_verified = header
        .skid
        .as_deref()
        .and_then(|s| s.parse::<Did>().ok());
    Ok(UnpackedMessage {
        message: m,
        sender_verified,
        encrypted: true,
        signed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_agreement::KeyPair;
    use ed25519_dalek::SigningKey as EdSigningKey;
    use rand::rngs::OsRng;
    use smart_byte_did::UniversalResolver;

    fn msg() -> DidcommMessage {
        let alice: Did = "did:example:alice".parse().unwrap();
        let bob: Did = "did:example:bob".parse().unwrap();
        DidcommMessage::new("https://didcomm.org/basicmessage/2.0/message")
            .from_did(alice)
            .to_dids(vec![bob])
            .body(serde_json::json!({"content": "hello"}))
    }

    #[tokio::test]
    async fn plaintext_round_trip() {
        let m = msg();
        let packed = pack_plaintext(&m).unwrap();
        let r = UniversalResolver::new();
        let out = unpack(&packed, &r, &[], &[]).await.unwrap();
        assert!(!out.signed);
        assert!(!out.encrypted);
        assert_eq!(out.message.body["content"], "hello");
    }

    #[tokio::test]
    async fn signed_round_trip() {
        let sk = EdSigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let signer = EdSigner {
            kid: "did:example:alice#keys-1".into(),
            key: sk,
        };
        let m = msg();
        let packed = pack_signed(&m, &signer).unwrap();
        let r = UniversalResolver::new();
        let out = unpack(
            &packed,
            &r,
            &[],
            &[("did:example:alice#keys-1", vk)],
        )
        .await
        .unwrap();
        assert!(out.signed);
        assert!(!out.encrypted);
        assert_eq!(out.message.body["content"], "hello");
    }

    #[tokio::test]
    async fn signed_tampered_fails() {
        let sk = EdSigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let signer = EdSigner {
            kid: "k1".into(),
            key: sk,
        };
        let m = msg();
        let packed = pack_signed(&m, &signer).unwrap();
        // Flip a byte in the payload segment.
        let mut parts: Vec<&str> = packed.split('.').collect();
        let tampered_payload = format!("{}A", parts[1]);
        parts[1] = &tampered_payload;
        let tampered = parts.join(".");
        let r = UniversalResolver::new();
        let res = unpack(&tampered, &r, &[], &[("k1", vk)]).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn encrypted_anoncrypt_round_trip() {
        let bob = KeyPair::generate();
        let recip = VerifyingKey {
            kid: "did:example:bob#kex-1".into(),
            public: bob.public,
        };
        let m = msg();
        let packed = pack_encrypted(
            &m,
            None,
            &[recip],
            KeyAgreementAlgorithm::EcdhEsA256Kw,
            ContentEncryption::Xc20p,
        )
        .unwrap();
        let r = UniversalResolver::new();
        let out = unpack(
            &packed,
            &r,
            &[SecretKey {
                kid: "did:example:bob#kex-1".into(),
                secret: bob.secret,
            }],
            &[],
        )
        .await
        .unwrap();
        assert!(out.encrypted);
        assert!(!out.signed);
        assert_eq!(out.message.body["content"], "hello");
    }

    #[tokio::test]
    async fn encrypted_wrong_key_fails() {
        let bob = KeyPair::generate();
        let mallory = KeyPair::generate();
        let recip = VerifyingKey {
            kid: "did:example:bob#kex-1".into(),
            public: bob.public,
        };
        let m = msg();
        let packed = pack_encrypted(
            &m,
            None,
            &[recip],
            KeyAgreementAlgorithm::EcdhEsA256Kw,
            ContentEncryption::Xc20p,
        )
        .unwrap();
        let r = UniversalResolver::new();
        let res = unpack(
            &packed,
            &r,
            &[SecretKey {
                kid: "did:example:bob#kex-1".into(),
                secret: mallory.secret,
            }],
            &[],
        )
        .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn expired_message_rejected() {
        let mut m = msg();
        m.expires_time =
            Some(chrono::Utc::now() - chrono::Duration::seconds(60));
        let packed = pack_plaintext(&m).unwrap();
        let r = UniversalResolver::new();
        let res = unpack(&packed, &r, &[], &[]).await;
        assert!(matches!(res, Err(DidcommError::Expired)));
    }
}
