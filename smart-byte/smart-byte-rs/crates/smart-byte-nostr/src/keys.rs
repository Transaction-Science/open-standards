//! Nostr keypair: secp256k1 secret + BIP-340 x-only public key.
//!
//! Nostr identities are x-only secp256k1 public keys (32 bytes), with
//! Schnorr signatures per BIP-340. This module wraps the [`secp256k1`]
//! crate with the small surface the rest of the adapter needs.

use crate::error::NostrError;
use secp256k1::{Keypair, SECP256K1, SecretKey, XOnlyPublicKey};

/// A Nostr secret key (32 bytes).
#[derive(Debug, Clone)]
pub struct NostrSecretKey {
    pub(crate) inner: SecretKey,
}

/// A Nostr x-only public key (32 bytes) — also the Nostr identity hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NostrPublicKey {
    pub(crate) inner: XOnlyPublicKey,
}

impl NostrSecretKey {
    /// Construct a secret key from 32 raw bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, NostrError> {
        let inner = SecretKey::from_byte_array(bytes)
            .map_err(|e| NostrError::InvalidKey(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Generate a random secret key.
    pub fn generate() -> Self {
        let (inner, _pk) = SECP256K1.generate_keypair(&mut rand::thread_rng());
        Self { inner }
    }

    /// Return the 32-byte secret scalar.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.inner.secret_bytes()
    }

    /// Derive the x-only public key associated with this secret.
    pub fn public_key(&self) -> NostrPublicKey {
        let kp = Keypair::from_secret_key(SECP256K1, &self.inner);
        let (xonly, _parity) = kp.x_only_public_key();
        NostrPublicKey { inner: xonly }
    }

    /// Borrow the underlying [`secp256k1::Keypair`].
    pub(crate) fn keypair(&self) -> Keypair {
        Keypair::from_secret_key(SECP256K1, &self.inner)
    }
}

impl NostrPublicKey {
    /// Construct an x-only public key from 32 raw bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, NostrError> {
        let inner = XOnlyPublicKey::from_byte_array(bytes)
            .map_err(|e| NostrError::InvalidKey(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Return the 32-byte x-only public key.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.inner.serialize()
    }

    /// Return the hex string (NIP-01 identity form).
    pub fn to_hex(&self) -> String {
        hex_encode(&self.to_bytes())
    }

    /// Parse from a 64-character hex string.
    pub fn from_hex(s: &str) -> Result<Self, NostrError> {
        let bytes = hex_decode(s)?;
        if bytes.len() != 32 {
            return Err(NostrError::InvalidKey("pubkey must be 32 bytes".into()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Self::from_bytes(&arr)
    }
}

/// Sign a 32-byte message digest under BIP-340 Schnorr.
pub fn schnorr_sign(sk: &NostrSecretKey, msg32: &[u8; 32]) -> Result<[u8; 64], NostrError> {
    let kp = sk.keypair();
    let sig = SECP256K1.sign_schnorr(msg32, &kp);
    Ok(sig.to_byte_array())
}

/// Verify a 64-byte BIP-340 Schnorr signature against an x-only pubkey.
pub fn schnorr_verify(
    pk: &NostrPublicKey,
    msg32: &[u8; 32],
    sig64: &[u8; 64],
) -> Result<(), NostrError> {
    let sig = secp256k1::schnorr::Signature::from_byte_array(*sig64);
    SECP256K1
        .verify_schnorr(&sig, msg32, &pk.inner)
        .map_err(|_| NostrError::BadSignature)
}

/// Lowercase hex encode (no prefix).
pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Strict lowercase / mixed-case hex decode.
pub fn hex_decode(s: &str) -> Result<Vec<u8>, NostrError> {
    if !s.len().is_multiple_of(2) {
        return Err(NostrError::Hex("odd length".into()));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = nib(bytes[i])?;
        let lo = nib(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn nib(c: u8) -> Result<u8, NostrError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(NostrError::Hex(format!("bad hex char {c:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let bytes = [0xde, 0xad, 0xbe, 0xef, 0x00, 0xff];
        let s = hex_encode(&bytes);
        assert_eq!(s, "deadbeef00ff");
        assert_eq!(hex_decode(&s).expect("hex"), bytes);
    }

    #[test]
    fn keygen_and_pubkey() {
        let sk = NostrSecretKey::generate();
        let pk = sk.public_key();
        let hex = pk.to_hex();
        assert_eq!(hex.len(), 64);
        let parsed = NostrPublicKey::from_hex(&hex).expect("parse");
        assert_eq!(parsed, pk);
    }

    #[test]
    fn schnorr_sign_verify() {
        let sk = NostrSecretKey::generate();
        let pk = sk.public_key();
        let msg = [42u8; 32];
        let sig = schnorr_sign(&sk, &msg).expect("sign");
        schnorr_verify(&pk, &msg, &sig).expect("verify");
        let mut bad = msg;
        bad[0] ^= 1;
        assert!(schnorr_verify(&pk, &bad, &sig).is_err());
    }
}
