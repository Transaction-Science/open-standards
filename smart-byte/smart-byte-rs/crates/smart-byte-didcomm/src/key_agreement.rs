//! ECDH-1PU and ECDH-ES key agreement for DIDComm v2 (DIF DIDComm
//! Messaging v2.1 § 4).
//!
//! Both modes derive a content-encryption key via HKDF-SHA256 in the
//! "Concat KDF" framing required by JWE (RFC 7518 § 4.6 for ECDH-ES,
//! draft-madden-jose-ecdh-1pu for ECDH-1PU).

use hkdf::Hkdf;
use sha2::Sha256;

use crate::error::DidcommError;

/// Supported key-agreement algorithms (DIF DIDComm v2 § 4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAgreementAlgorithm {
    /// Authenticated: ECDH-1PU + A256KW (RFC 6643 key wrapping).
    Ecdh1puA256Kw,
    /// Anonymous: ECDH-ES + A256KW.
    EcdhEsA256Kw,
}

impl KeyAgreementAlgorithm {
    /// JOSE `alg` string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ecdh1puA256Kw => "ECDH-1PU+A256KW",
            Self::EcdhEsA256Kw => "ECDH-ES+A256KW",
        }
    }
}

/// Supported content-encryption algorithms (DIF DIDComm v2 § 4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentEncryption {
    /// AES-256 GCM (RFC 7518).
    A256Gcm,
    /// XChaCha20-Poly1305 (24-byte nonce variant).
    Xc20p,
    /// AES-256 CBC + HMAC-SHA-512 (RFC 7518 § 5.3).
    A256CbcHs512,
}

impl ContentEncryption {
    /// JOSE `enc` string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::A256Gcm => "A256GCM",
            Self::Xc20p => "XC20P",
            Self::A256CbcHs512 => "A256CBC-HS512",
        }
    }

    /// Content-encryption key byte length.
    pub fn cek_len(&self) -> usize {
        match self {
            Self::A256Gcm => 32,
            Self::Xc20p => 32,
            Self::A256CbcHs512 => 64,
        }
    }
}

/// A key-agreement key pair (sender's local secret + public).
#[derive(Debug, Clone)]
pub struct KeyPair {
    /// The 32-byte X25519 secret scalar (clamped).
    pub secret: [u8; 32],
    /// The matching X25519 public key.
    pub public: [u8; 32],
}

impl KeyPair {
    /// Generate a fresh X25519 keypair.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut sk = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut sk);
        let secret = x25519_dalek::StaticSecret::from(sk);
        let public = x25519_dalek::PublicKey::from(&secret);
        Self {
            secret: secret.to_bytes(),
            public: *public.as_bytes(),
        }
    }
}

/// Concat-KDF-style HKDF-SHA-256 used by both ECDH-ES and ECDH-1PU.
///
/// Per RFC 7518 § 4.6.2 the OtherInfo is:
///   AlgorithmID || PartyUInfo || PartyVInfo || SuppPubInfo
/// with 32-bit big-endian length prefixes on each field.
pub fn concat_kdf(
    z: &[u8],
    alg: &[u8],
    apu: &[u8],
    apv: &[u8],
    key_len_bits: u32,
) -> Result<Vec<u8>, DidcommError> {
    let mut info = Vec::with_capacity(alg.len() + apu.len() + apv.len() + 24);
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
        .map_err(|e| DidcommError::Crypto(format!("hkdf expand: {e}")))?;
    Ok(out)
}

/// ECDH-ES (anonymous) key agreement over X25519.
///
/// `sender_secret` is an ephemeral secret generated fresh for this
/// message; the corresponding ephemeral public key MUST be transmitted in
/// the JWE header `epk`.
pub fn ecdh_es_x25519(
    sender_secret: &[u8; 32],
    recipient_public: &[u8; 32],
    alg: &str,
    apu: &[u8],
    apv: &[u8],
) -> Result<[u8; 32], DidcommError> {
    let sk = x25519_dalek::StaticSecret::from(*sender_secret);
    let pk = x25519_dalek::PublicKey::from(*recipient_public);
    let z = sk.diffie_hellman(&pk);
    let derived = concat_kdf(z.as_bytes(), alg.as_bytes(), apu, apv, 256)?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&derived);
    Ok(out)
}

/// ECDH-1PU (authenticated) key agreement over X25519.
///
/// Per `draft-madden-jose-ecdh-1pu`, the shared secret is the
/// concatenation of two DH outputs:
///   Ze = ECDH(ephemeral_sender, recipient_static)
///   Zs = ECDH(static_sender, recipient_static)
///   Z  = Ze || Zs
pub fn ecdh_1pu_x25519(
    ephemeral_secret: &[u8; 32],
    static_sender_secret: &[u8; 32],
    recipient_public: &[u8; 32],
    alg: &str,
    apu: &[u8],
    apv: &[u8],
) -> Result<[u8; 32], DidcommError> {
    let pk = x25519_dalek::PublicKey::from(*recipient_public);
    let esk = x25519_dalek::StaticSecret::from(*ephemeral_secret);
    let ssk = x25519_dalek::StaticSecret::from(*static_sender_secret);
    let ze = esk.diffie_hellman(&pk);
    let zs = ssk.diffie_hellman(&pk);
    let mut z = Vec::with_capacity(64);
    z.extend_from_slice(ze.as_bytes());
    z.extend_from_slice(zs.as_bytes());
    let derived = concat_kdf(&z, alg.as_bytes(), apu, apv, 256)?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&derived);
    Ok(out)
}

/// A handle to a recipient verification key, suitable for key agreement.
#[derive(Debug, Clone)]
pub struct VerifyingKey {
    /// Recipient kid (typically a DID URL with key fragment).
    pub kid: String,
    /// 32-byte X25519 public key bytes.
    pub public: [u8; 32],
}

/// A handle to a local decryption secret key.
#[derive(Debug, Clone)]
pub struct SecretKey {
    /// Local kid this secret corresponds to.
    pub kid: String,
    /// 32-byte X25519 secret scalar.
    pub secret: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecdh_es_round_trip_x25519() {
        // Sender ephemeral keypair.
        let eph = KeyPair::generate();
        // Recipient static keypair.
        let recip = KeyPair::generate();
        let alg = "ECDH-ES+A256KW";
        let sender_derived = ecdh_es_x25519(
            &eph.secret,
            &recip.public,
            alg,
            b"",
            b"",
        )
        .unwrap();
        // Recipient derives the same K by ECDH(recipient_static, sender_ephemeral_public).
        let recip_derived = ecdh_es_x25519(
            &recip.secret,
            &eph.public,
            alg,
            b"",
            b"",
        )
        .unwrap();
        assert_eq!(sender_derived, recip_derived);
    }

    #[test]
    fn ecdh_1pu_round_trip_x25519() {
        let alice_static = KeyPair::generate();
        let alice_ephemeral = KeyPair::generate();
        let bob = KeyPair::generate();
        let alg = "ECDH-1PU+A256KW";
        let alice_k = ecdh_1pu_x25519(
            &alice_ephemeral.secret,
            &alice_static.secret,
            &bob.public,
            alg,
            b"alice",
            b"bob",
        )
        .unwrap();
        // Bob receives both alice_static.public and alice_ephemeral.public.
        // Ze = DH(bob_secret, alice_ephemeral_public)
        // Zs = DH(bob_secret, alice_static_public)
        let bob_sk = x25519_dalek::StaticSecret::from(bob.secret);
        let ze = bob_sk
            .diffie_hellman(&x25519_dalek::PublicKey::from(alice_ephemeral.public));
        let zs = bob_sk
            .diffie_hellman(&x25519_dalek::PublicKey::from(alice_static.public));
        let mut z = Vec::with_capacity(64);
        z.extend_from_slice(ze.as_bytes());
        z.extend_from_slice(zs.as_bytes());
        let bob_k = concat_kdf(&z, alg.as_bytes(), b"alice", b"bob", 256).unwrap();
        assert_eq!(alice_k.to_vec(), bob_k);
    }

    #[test]
    fn concat_kdf_deterministic() {
        let z = [7u8; 32];
        let a = concat_kdf(&z, b"ECDH-ES+A256KW", b"alice", b"bob", 256).unwrap();
        let b = concat_kdf(&z, b"ECDH-ES+A256KW", b"alice", b"bob", 256).unwrap();
        assert_eq!(a, b);
        let c = concat_kdf(&z, b"ECDH-ES+A256KW", b"alice", b"carol", 256).unwrap();
        assert_ne!(a, c);
    }
}
