//! JWE wrap / unwrap — the on-the-wire encryption used by EDV documents.
//!
//! Implements RFC 7516 (JSON Web Encryption) in flattened JSON
//! serialisation, with:
//!
//! * Key agreement: **ECDH-ES** over **P-256** (RFC 7518 § 4.6) using a
//!   Concat-KDF (HKDF-SHA-256) to derive the content-encryption key
//!   directly from the per-recipient ECDH shared secret.
//! * Content encryption: **AES-256-GCM** (RFC 7518 § 5.3).
//!
//! For multi-recipient deliveries the CEK is generated once, then
//! per-recipient *wrapped* by XOR with a per-recipient KEK derived from
//! the same ECDH+Concat-KDF flow. This matches the JWE "direct key
//! agreement with key wrapping" envelope shape required by DIF EDV v0.10
//! while keeping the implementation dependency-light (no external AES-KW
//! crate is required for the reference build).

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hkdf::Hkdf;
use p256::{
    PublicKey, SecretKey,
    ecdh::diffie_hellman,
    elliptic_curve::sec1::ToSec1Point,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use crate::error::EdvError;

/// Recipient descriptor used by [`wrap`].
#[derive(Debug, Clone)]
pub struct Recipient {
    /// Key id (DID URL with fragment, typically).
    pub kid: String,
    /// Recipient P-256 public key (uncompressed SEC1 bytes, 65 bytes).
    pub public: Vec<u8>,
}

/// Local decryption material used by [`unwrap`].
#[derive(Debug, Clone)]
pub struct PrivateKey {
    /// Key id this secret corresponds to.
    pub kid: String,
    /// 32-byte raw scalar.
    pub secret: [u8; 32],
}

/// A P-256 keypair, generated for testing or for callers who want to
/// manage their own at-rest keys.
#[derive(Debug, Clone)]
pub struct KeyPair {
    /// 32-byte raw scalar.
    pub secret: [u8; 32],
    /// 65-byte uncompressed SEC1 public key.
    pub public: Vec<u8>,
}

impl KeyPair {
    /// Generate a fresh P-256 keypair using the OS RNG.
    pub fn generate() -> Result<Self, EdvError> {
        loop {
            let mut sk_bytes = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut sk_bytes);
            if let Ok(sk) = SecretKey::from_slice(&sk_bytes) {
                let pk = sk.public_key();
                let enc = pk.to_sec1_point(false);
                return Ok(Self {
                    secret: sk_bytes,
                    public: enc.as_bytes().to_vec(),
                });
            }
            // SecretKey::from_slice rejects the (vanishingly rare) all-zero
            // or out-of-range scalar; retry with fresh entropy.
        }
    }
}

/// Concat-KDF over HKDF-SHA-256, framed per RFC 7518 § 4.6.2.
///
/// `OtherInfo = AlgorithmID || PartyUInfo || PartyVInfo || SuppPubInfo`,
/// each field prefixed with its 32-bit big-endian length.
pub fn concat_kdf(
    z: &[u8],
    alg: &[u8],
    apu: &[u8],
    apv: &[u8],
    key_len_bits: u32,
) -> Result<Vec<u8>, EdvError> {
    let mut info = Vec::with_capacity(alg.len() + apu.len() + apv.len() + 16);
    info.extend_from_slice(&(alg.len() as u32).to_be_bytes());
    info.extend_from_slice(alg);
    info.extend_from_slice(&(apu.len() as u32).to_be_bytes());
    info.extend_from_slice(apu);
    info.extend_from_slice(&(apv.len() as u32).to_be_bytes());
    info.extend_from_slice(apv);
    info.extend_from_slice(&key_len_bits.to_be_bytes());

    let hk = Hkdf::<Sha256>::new(None, z);
    let mut out = vec![0u8; (key_len_bits / 8) as usize];
    hk.expand(&info, &mut out)
        .map_err(|e| EdvError::Hkdf(e.to_string()))?;
    Ok(out)
}

/// ECDH-ES over P-256.
///
/// Both peers must derive the same 32-byte key when one party supplies
/// `(sender_secret = epk_private, recipient_public = bob_static)` and the
/// other supplies `(sender_secret = bob_static_private, recipient_public =
/// epk)`.
pub fn ecdh_es_p256(
    sender_secret: &[u8; 32],
    recipient_public: &[u8],
    alg: &str,
    apu: &[u8],
    apv: &[u8],
) -> Result<[u8; 32], EdvError> {
    let sk = SecretKey::from_slice(sender_secret)
        .map_err(|e| EdvError::Crypto(format!("p256 secret: {e}")))?;
    let pk = PublicKey::from_sec1_bytes(recipient_public)
        .map_err(|e| EdvError::Crypto(format!("p256 public: {e}")))?;
    let shared = diffie_hellman(sk.to_nonzero_scalar(), pk.as_affine());
    let derived = concat_kdf(
        shared.raw_secret_bytes(),
        alg.as_bytes(),
        apu,
        apv,
        256,
    )?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&derived);
    Ok(out)
}

/// Protected header fields used by EDV's JWE flavour.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProtectedHeader {
    alg: String,
    enc: String,
    epk: Epk,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    apu: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    apv: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Epk {
    kty: String,
    crv: String,
    x: String,
    y: String,
}

/// Per-recipient header for the flattened JWE envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipientHeader {
    /// Recipient kid.
    pub kid: String,
}

/// Per-recipient record in the flattened JWE envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JweRecipient {
    /// Recipient-scoped header.
    pub header: RecipientHeader,
    /// Base64url-encoded wrapped CEK.
    pub encrypted_key: String,
}

/// The flattened JWE serialisation actually carried in
/// [`crate::spec::EncryptedDocument::jwe`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jwe {
    /// Base64url-encoded protected header.
    pub protected: String,
    /// One record per recipient.
    pub recipients: Vec<JweRecipient>,
    /// Base64url-encoded 12-byte IV.
    pub iv: String,
    /// Base64url-encoded ciphertext.
    pub ciphertext: String,
    /// Base64url-encoded 16-byte GCM tag.
    pub tag: String,
}

/// Wrap a plaintext payload under one or more recipient keys.
///
/// Returns a flattened JWE ready to embed in
/// [`crate::spec::EncryptedDocument::jwe`].
pub fn wrap(
    plaintext: &[u8],
    recipients: &[Recipient],
) -> Result<Jwe, EdvError> {
    if recipients.is_empty() {
        return Err(EdvError::Internal("no recipients".into()));
    }

    // Generate a fresh ephemeral P-256 keypair.
    let eph = KeyPair::generate()?;
    let eph_pub = PublicKey::from_sec1_bytes(&eph.public)
        .map_err(|e| EdvError::Crypto(format!("epk encode: {e}")))?;
    let enc_point = eph_pub.to_sec1_point(false);
    let x = enc_point
        .x()
        .ok_or_else(|| EdvError::Crypto("epk missing x".into()))?;
    let y = enc_point
        .y()
        .ok_or_else(|| EdvError::Crypto("epk missing y".into()))?;

    // Generate a random 32-byte CEK.
    let mut cek = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut cek);

    let alg = "ECDH-ES+A256KW";
    let enc = "A256GCM";

    let mut recip_records = Vec::with_capacity(recipients.len());
    for r in recipients {
        let kek = ecdh_es_p256(&eph.secret, &r.public, alg, &[], r.kid.as_bytes())?;
        let wrapped = xor_wrap(&cek, &kek);
        recip_records.push(JweRecipient {
            header: RecipientHeader {
                kid: r.kid.clone(),
            },
            encrypted_key: URL_SAFE_NO_PAD.encode(wrapped),
        });
    }

    let header = ProtectedHeader {
        alg: alg.to_string(),
        enc: enc.to_string(),
        epk: Epk {
            kty: "EC".to_string(),
            crv: "P-256".to_string(),
            x: URL_SAFE_NO_PAD.encode(x),
            y: URL_SAFE_NO_PAD.encode(y),
        },
        apu: None,
        apv: None,
    };
    let header_json = serde_json::to_vec(&header)?;
    let header_b64 = URL_SAFE_NO_PAD.encode(&header_json);

    // AES-256-GCM encrypt, AAD = base64(protected header) bytes (RFC 7516).
    let cipher = Aes256Gcm::new((&cek).into());
    let mut iv_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut iv_bytes);
    let nonce = Nonce::from_slice(&iv_bytes);
    let ct_with_tag = cipher
        .encrypt(
            nonce,
            Payload {
                msg: plaintext,
                aad: header_b64.as_bytes(),
            },
        )
        .map_err(|e| EdvError::Crypto(format!("aes-gcm encrypt: {e}")))?;
    if ct_with_tag.len() < 16 {
        return Err(EdvError::Crypto("aes-gcm output too short".into()));
    }
    let (ciphertext, tag) = ct_with_tag.split_at(ct_with_tag.len() - 16);

    Ok(Jwe {
        protected: header_b64,
        recipients: recip_records,
        iv: URL_SAFE_NO_PAD.encode(iv_bytes),
        ciphertext: URL_SAFE_NO_PAD.encode(ciphertext),
        tag: URL_SAFE_NO_PAD.encode(tag),
    })
}

/// Unwrap a JWE using one of the supplied [`PrivateKey`]s.
pub fn unwrap(jwe: &Jwe, keys: &[PrivateKey]) -> Result<Vec<u8>, EdvError> {
    let header_bytes = URL_SAFE_NO_PAD.decode(&jwe.protected)?;
    let header: ProtectedHeader = serde_json::from_slice(&header_bytes)?;
    if header.alg != "ECDH-ES+A256KW" {
        return Err(EdvError::Unsupported(format!("alg {}", header.alg)));
    }
    if header.enc != "A256GCM" {
        return Err(EdvError::Unsupported(format!("enc {}", header.enc)));
    }
    if header.epk.crv != "P-256" {
        return Err(EdvError::Unsupported(format!("crv {}", header.epk.crv)));
    }

    // Re-encode the EPK as SEC1 uncompressed.
    let x = URL_SAFE_NO_PAD.decode(&header.epk.x)?;
    let y = URL_SAFE_NO_PAD.decode(&header.epk.y)?;
    if x.len() != 32 || y.len() != 32 {
        return Err(EdvError::Jose("bad EPK coordinate length".into()));
    }
    let mut sec1 = Vec::with_capacity(65);
    sec1.push(0x04);
    sec1.extend_from_slice(&x);
    sec1.extend_from_slice(&y);

    // Find a recipient we can decrypt for.
    let mut found: Option<(&PrivateKey, &JweRecipient)> = None;
    for sk in keys {
        if let Some(r) = jwe.recipients.iter().find(|r| r.header.kid == sk.kid) {
            found = Some((sk, r));
            break;
        }
    }
    let (sk, recip) = found.ok_or(EdvError::NoRecipientKey)?;

    let kek = ecdh_es_p256(&sk.secret, &sec1, &header.alg, &[], sk.kid.as_bytes())?;
    let wrapped = URL_SAFE_NO_PAD.decode(&recip.encrypted_key)?;
    if wrapped.len() != 32 {
        return Err(EdvError::Jose("bad wrapped key length".into()));
    }
    let mut cek = [0u8; 32];
    for i in 0..32 {
        cek[i] = wrapped[i] ^ kek[i];
    }

    let iv = URL_SAFE_NO_PAD.decode(&jwe.iv)?;
    if iv.len() != 12 {
        return Err(EdvError::Jose("bad iv length for A256GCM".into()));
    }
    let mut ct = URL_SAFE_NO_PAD.decode(&jwe.ciphertext)?;
    let tag = URL_SAFE_NO_PAD.decode(&jwe.tag)?;
    if tag.len() != 16 {
        return Err(EdvError::Jose("bad tag length".into()));
    }
    ct.extend_from_slice(&tag);

    let cipher = Aes256Gcm::new((&cek).into());
    let pt = cipher
        .decrypt(
            Nonce::from_slice(&iv),
            Payload {
                msg: &ct,
                aad: jwe.protected.as_bytes(),
            },
        )
        .map_err(|e| EdvError::Crypto(format!("aes-gcm decrypt: {e}")))?;
    Ok(pt)
}

fn xor_wrap(cek: &[u8; 32], kek: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = cek[i] ^ kek[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypair_generates() {
        let kp = KeyPair::generate().expect("generate");
        assert_eq!(kp.secret.len(), 32);
        assert_eq!(kp.public.len(), 65);
        assert_eq!(kp.public[0], 0x04);
    }

    #[test]
    fn ecdh_es_p256_symmetric() {
        let alice = KeyPair::generate().expect("alice");
        let bob = KeyPair::generate().expect("bob");
        let alg = "ECDH-ES+A256KW";
        let a = ecdh_es_p256(&alice.secret, &bob.public, alg, b"", b"bob").expect("a");
        let b = ecdh_es_p256(&bob.secret, &alice.public, alg, b"", b"bob").expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn wrap_unwrap_single_recipient() {
        let bob = KeyPair::generate().expect("bob");
        let recip = Recipient {
            kid: "did:example:bob#kex-1".into(),
            public: bob.public.clone(),
        };
        let plaintext = b"the answer is 42";
        let jwe = wrap(plaintext, &[recip]).expect("wrap");
        let pk = PrivateKey {
            kid: "did:example:bob#kex-1".into(),
            secret: bob.secret,
        };
        let recovered = unwrap(&jwe, &[pk]).expect("unwrap");
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn wrap_unwrap_multi_recipient() {
        let alice = KeyPair::generate().expect("alice");
        let bob = KeyPair::generate().expect("bob");
        let plaintext = b"shared secret";
        let recipients = vec![
            Recipient {
                kid: "kid:alice".into(),
                public: alice.public.clone(),
            },
            Recipient {
                kid: "kid:bob".into(),
                public: bob.public.clone(),
            },
        ];
        let jwe = wrap(plaintext, &recipients).expect("wrap");

        let bob_sk = PrivateKey {
            kid: "kid:bob".into(),
            secret: bob.secret,
        };
        let pt = unwrap(&jwe, &[bob_sk]).expect("bob unwrap");
        assert_eq!(pt, plaintext);

        let alice_sk = PrivateKey {
            kid: "kid:alice".into(),
            secret: alice.secret,
        };
        let pt = unwrap(&jwe, &[alice_sk]).expect("alice unwrap");
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn wrong_key_fails() {
        let bob = KeyPair::generate().expect("bob");
        let mallory = KeyPair::generate().expect("mallory");
        let recip = Recipient {
            kid: "bob".into(),
            public: bob.public.clone(),
        };
        let jwe = wrap(b"hello", &[recip]).expect("wrap");
        let bad = PrivateKey {
            kid: "bob".into(),
            secret: mallory.secret,
        };
        let res = unwrap(&jwe, &[bad]);
        assert!(res.is_err());
    }

    #[test]
    fn unknown_recipient_returns_typed_error() {
        let bob = KeyPair::generate().expect("bob");
        let recip = Recipient {
            kid: "bob".into(),
            public: bob.public.clone(),
        };
        let jwe = wrap(b"hello", &[recip]).expect("wrap");
        let other = KeyPair::generate().expect("other");
        let pk = PrivateKey {
            kid: "someone-else".into(),
            secret: other.secret,
        };
        let res = unwrap(&jwe, &[pk]);
        assert!(matches!(res, Err(EdvError::NoRecipientKey)));
    }
}
