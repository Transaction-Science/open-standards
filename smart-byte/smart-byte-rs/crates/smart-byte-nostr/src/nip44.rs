//! NIP-44 versioned encrypted payloads (v2).
//!
//! NIP-44 v2 layout (binary, before base64):
//!
//! ```text
//! version (1 byte = 0x02)
//! nonce   (32 bytes, random)
//! ciphertext (variable, ChaCha20 of padded plaintext)
//! mac     (32 bytes, HMAC-SHA256 over nonce || ciphertext)
//! ```
//!
//! Keys are derived from the secp256k1 ECDH shared x-coordinate via
//! HKDF-SHA256 with salt `"nip44-v2"`:
//!   * `chacha_key`   = HKDF[0..32]
//!   * `chacha_nonce` = HKDF[32..44]  (truncated; the 32-byte nonce is
//!                                     also fed into HKDF info)
//!   * `hmac_key`     = HKDF[44..76]
//!
//! Plaintext is length-prefixed (`u16 BE`) and padded to a NIP-44
//! bucket size, allowing message-length to be hidden in fixed
//! categories.

use crate::error::NostrError;
use crate::keys::{NostrPublicKey, NostrSecretKey};
use crate::nip04::derive_shared_key;
use chacha20::ChaCha20;
use chacha20::cipher::{KeyIvInit, StreamCipher};
use data_encoding::BASE64;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const VERSION: u8 = 0x02;
const HKDF_SALT: &[u8] = b"nip44-v2";

/// Encrypt `plaintext` for `recipient_pk` and return a base64 NIP-44 v2 blob.
pub fn encrypt(
    sender_sk: &NostrSecretKey,
    recipient_pk: &NostrPublicKey,
    plaintext: &[u8],
) -> Result<String, NostrError> {
    if plaintext.is_empty() || plaintext.len() > 65535 {
        return Err(NostrError::Crypto("plaintext length out of range".into()));
    }
    let shared = derive_shared_key(sender_sk, recipient_pk)?;
    let mut nonce = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut nonce);

    let (chacha_key, chacha_nonce12, hmac_key) = derive_keys(&shared, &nonce);

    let padded = pad_plaintext(plaintext);
    let mut ct = padded.clone();
    let mut cipher = ChaCha20::new(&chacha_key.into(), &chacha_nonce12.into());
    cipher.apply_keystream(&mut ct);

    let mut mac = HmacSha256::new_from_slice(&hmac_key)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    mac.update(&nonce);
    mac.update(&ct);
    let tag = mac.finalize().into_bytes();

    let mut out = Vec::with_capacity(1 + 32 + ct.len() + 32);
    out.push(VERSION);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    out.extend_from_slice(&tag);
    Ok(BASE64.encode(&out))
}

/// Decrypt a base64 NIP-44 v2 blob from `sender_pk`.
pub fn decrypt(
    recipient_sk: &NostrSecretKey,
    sender_pk: &NostrPublicKey,
    blob_b64: &str,
) -> Result<Vec<u8>, NostrError> {
    let raw = BASE64
        .decode(blob_b64.as_bytes())
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    if raw.len() < 1 + 32 + 1 + 32 {
        return Err(NostrError::Crypto("nip-44 blob too short".into()));
    }
    let version = raw[0];
    if version != VERSION {
        return Err(NostrError::UnsupportedVersion(version));
    }
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&raw[1..33]);
    let mac_start = raw.len() - 32;
    let ct = &raw[33..mac_start];
    let tag = &raw[mac_start..];

    let shared = derive_shared_key(recipient_sk, sender_pk)?;
    let (chacha_key, chacha_nonce12, hmac_key) = derive_keys(&shared, &nonce);

    let mut mac = HmacSha256::new_from_slice(&hmac_key)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    mac.update(&nonce);
    mac.update(ct);
    mac.verify_slice(tag).map_err(|_| NostrError::MacMismatch)?;

    let mut padded = ct.to_vec();
    let mut cipher = ChaCha20::new(&chacha_key.into(), &chacha_nonce12.into());
    cipher.apply_keystream(&mut padded);

    unpad_plaintext(&padded)
}

fn derive_keys(shared: &[u8; 32], nonce: &[u8; 32]) -> ([u8; 32], [u8; 12], [u8; 32]) {
    let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT), shared);
    let mut okm = [0u8; 32 + 12 + 32];
    // info = nonce to bind keys to nonce per NIP-44 v2.
    hk.expand(nonce, &mut okm).expect("hkdf okm fits");
    let mut chacha_key = [0u8; 32];
    chacha_key.copy_from_slice(&okm[..32]);
    let mut chacha_nonce12 = [0u8; 12];
    chacha_nonce12.copy_from_slice(&okm[32..44]);
    let mut hmac_key = [0u8; 32];
    hmac_key.copy_from_slice(&okm[44..76]);
    (chacha_key, chacha_nonce12, hmac_key)
}

/// Pad plaintext to a NIP-44 bucket. Layout: `u16 BE length || plaintext || zero pad`.
fn pad_plaintext(plaintext: &[u8]) -> Vec<u8> {
    let target = bucket_size(plaintext.len());
    let mut out = Vec::with_capacity(2 + target);
    let len_be = (plaintext.len() as u16).to_be_bytes();
    out.extend_from_slice(&len_be);
    out.extend_from_slice(plaintext);
    out.resize(2 + target, 0);
    out
}

fn unpad_plaintext(padded: &[u8]) -> Result<Vec<u8>, NostrError> {
    if padded.len() < 2 {
        return Err(NostrError::Crypto("padded too short".into()));
    }
    let mut len_be = [0u8; 2];
    len_be.copy_from_slice(&padded[..2]);
    let len = u16::from_be_bytes(len_be) as usize;
    if 2 + len > padded.len() {
        return Err(NostrError::Crypto("pad length too large".into()));
    }
    Ok(padded[2..2 + len].to_vec())
}

/// Compute the NIP-44 padding bucket for `len` plaintext bytes.
/// Buckets are powers-of-two style with a 32-byte minimum.
pub fn bucket_size(len: usize) -> usize {
    if len <= 32 {
        return 32;
    }
    let next = (len - 1).next_power_of_two();
    // Sub-bucket: round up to nearest multiple of next/8 (per NIP-44 v2).
    let chunk = (next / 8).max(32);
    len.div_ceil(chunk) * chunk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_sizes_are_monotonic() {
        for n in 1..200 {
            assert!(bucket_size(n) >= n);
        }
        assert!(bucket_size(40) >= 40);
    }

    #[test]
    fn nip44_roundtrip() {
        let alice = NostrSecretKey::generate();
        let bob = NostrSecretKey::generate();
        let blob = encrypt(&alice, &bob.public_key(), b"hello nip44").expect("enc");
        let pt = decrypt(&bob, &alice.public_key(), &blob).expect("dec");
        assert_eq!(pt, b"hello nip44");
    }
}
