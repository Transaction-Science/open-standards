//! NIP-04 legacy direct messages.
//!
//! NIP-04 derives a shared secret as the x-coordinate of `dh = sk * PK`
//! and uses AES-256-CBC with a random 16-byte IV. The wire format is
//! `<base64(ciphertext)>?iv=<base64(iv)>`. NIP-04 is deprecated in favor
//! of NIP-44 + NIP-17; we include it for relay-side compatibility but
//! never use it as the default.

use crate::error::NostrError;
use crate::keys::{NostrPublicKey, NostrSecretKey};
use aes::Aes256;
use aes::cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};
use data_encoding::BASE64;
use rand::RngCore;
use secp256k1::{Parity, PublicKey, SECP256K1, Scalar};

type Aes256CbcEnc = cbc::Encryptor<Aes256>;
type Aes256CbcDec = cbc::Decryptor<Aes256>;

/// Derive the NIP-04 shared key: the 32-byte x coordinate of
/// `sk * PK`. (NIP-04 uses the raw x with no hashing, see spec.)
pub fn derive_shared_key(sk: &NostrSecretKey, their_pk: &NostrPublicKey) -> Result<[u8; 32], NostrError> {
    // Lift the x-only public key to a full secp256k1 point with even Y,
    // then multiply by the secret scalar.
    let full = PublicKey::from_x_only_public_key(their_pk.inner, Parity::Even);
    let scalar = Scalar::from_be_bytes(sk.inner.secret_bytes())
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    let shared = full
        .mul_tweak(SECP256K1, &scalar)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    let serialized = shared.serialize();
    // serialized is 33 bytes (compressed); strip the parity byte and keep x.
    let mut out = [0u8; 32];
    out.copy_from_slice(&serialized[1..33]);
    Ok(out)
}

/// Encrypt `plaintext` to `recipient` and return the NIP-04 wire string.
pub fn encrypt(
    sender_sk: &NostrSecretKey,
    recipient_pk: &NostrPublicKey,
    plaintext: &[u8],
) -> Result<String, NostrError> {
    let key = derive_shared_key(sender_sk, recipient_pk)?;
    let mut iv = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut iv);
    let enc = Aes256CbcEnc::new_from_slices(&key, &iv)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    let ct = enc.encrypt_padded_vec_mut::<Pkcs7>(plaintext);
    Ok(format!("{}?iv={}", BASE64.encode(&ct), BASE64.encode(&iv)))
}

/// Decrypt a NIP-04 wire string.
pub fn decrypt(
    recipient_sk: &NostrSecretKey,
    sender_pk: &NostrPublicKey,
    wire: &str,
) -> Result<Vec<u8>, NostrError> {
    let (ct_b64, iv_b64) = wire
        .split_once("?iv=")
        .ok_or_else(|| NostrError::Crypto("missing ?iv= delimiter".into()))?;
    let ct = BASE64
        .decode(ct_b64.as_bytes())
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    let iv = BASE64
        .decode(iv_b64.as_bytes())
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    if iv.len() != 16 {
        return Err(NostrError::Crypto("iv must be 16 bytes".into()));
    }
    let key = derive_shared_key(recipient_sk, sender_pk)?;
    let dec = Aes256CbcDec::new_from_slices(&key, &iv)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    let pt = dec
        .decrypt_padded_vec_mut::<Pkcs7>(&ct)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    Ok(pt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nip04_roundtrip() {
        let alice = NostrSecretKey::generate();
        let bob = NostrSecretKey::generate();
        let alice_pk = alice.public_key();
        let bob_pk = bob.public_key();
        let pt = b"hello bob, this is alice";
        let wire = encrypt(&alice, &bob_pk, pt).expect("enc");
        let got = decrypt(&bob, &alice_pk, &wire).expect("dec");
        assert_eq!(got, pt);
    }

    #[test]
    fn shared_key_is_symmetric() {
        let a = NostrSecretKey::generate();
        let b = NostrSecretKey::generate();
        let k_ab = derive_shared_key(&a, &b.public_key()).expect("ab");
        let k_ba = derive_shared_key(&b, &a.public_key()).expect("ba");
        assert_eq!(k_ab, k_ba);
    }
}
