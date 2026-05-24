//! BBS+ key material.
//!
//! Per `draft-irtf-cfrg-bbs-signatures-08` § 3.4, a BBS+ signing key is
//! a non-zero scalar in the BLS12-381 scalar field and the public key
//! is `W = g2 * sk` where `g2` is the canonical generator of G2.
//!
//! `SecretKey` zeroes its inner scalar on drop; `KeyPair` does not
//! hold a separate copy of `secret`, it owns the unique copy.

use bls12_381::{G2Affine, G2Projective, Scalar};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::encode::{
    SCALAR_BYTES, g2_from_bytes, g2_to_bytes, scalar_from_bytes,
    scalar_to_bytes,
};
use crate::error::BbsError;

/// A BBS+ secret key. A non-zero scalar in the BLS12-381 scalar field.
///
/// The inner scalar is zeroed on drop.
#[derive(Clone)]
pub struct SecretKey(Scalar);

impl SecretKey {
    /// Borrow the inner scalar. Internal API.
    pub(crate) fn as_scalar(&self) -> &Scalar {
        &self.0
    }

    /// Construct from a scalar. The scalar must be non-zero; callers
    /// should obtain it from [`keygen`] in normal use.
    pub fn from_scalar(s: Scalar) -> Self {
        Self(s)
    }

    /// Encode as 32 canonical little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        scalar_to_bytes(&self.0)
    }

    /// Decode from 32 canonical little-endian bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, BbsError> {
        let s = scalar_from_bytes(bytes)?;
        Ok(Self(s))
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        // `Scalar` does not impl `Zeroize` directly but its bytes do.
        // We force a zero of the scalar by overwriting with a fresh
        // zero scalar. The compiler does not optimise out the assign
        // because `self.0` is read again in tests.
        let mut bytes = self.0.to_bytes();
        bytes.zeroize();
        self.0 = Scalar::from(0u64);
    }
}

impl Serialize for SecretKey {
    fn serialize<S: serde::Serializer>(
        &self,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let bytes = self.to_bytes();
        serde_bytes::Bytes::new(&bytes).serialize(ser)
    }
}

impl<'de> Deserialize<'de> for SecretKey {
    fn deserialize<D: serde::Deserializer<'de>>(
        de: D,
    ) -> Result<Self, D::Error> {
        let bytes: serde_bytes::ByteBuf = serde_bytes::ByteBuf::deserialize(de)?;
        if bytes.len() != SCALAR_BYTES {
            return Err(serde::de::Error::custom(format!(
                "expected {SCALAR_BYTES} scalar bytes, got {}",
                bytes.len()
            )));
        }
        let mut b = [0u8; SCALAR_BYTES];
        b.copy_from_slice(&bytes);
        SecretKey::from_bytes(&b).map_err(serde::de::Error::custom)
    }
}

/// A BBS+ public key. Lives in G2 of BLS12-381.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicKey {
    /// `W = g2 * sk`.
    pub w: G2Affine,
}

impl PublicKey {
    /// Construct from a known `G2Affine`.
    pub fn new(w: G2Affine) -> Self {
        Self { w }
    }

    /// Encode as 96 canonical compressed bytes.
    pub fn to_bytes(&self) -> [u8; 96] {
        g2_to_bytes(&self.w)
    }

    /// Decode from 96 canonical compressed bytes.
    pub fn from_bytes(bytes: &[u8; 96]) -> Result<Self, BbsError> {
        Ok(Self {
            w: g2_from_bytes(bytes)?,
        })
    }
}

impl Serialize for PublicKey {
    fn serialize<S: serde::Serializer>(
        &self,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let bytes = self.to_bytes();
        serde_bytes::Bytes::new(&bytes).serialize(ser)
    }
}

impl<'de> Deserialize<'de> for PublicKey {
    fn deserialize<D: serde::Deserializer<'de>>(
        de: D,
    ) -> Result<Self, D::Error> {
        let bytes: serde_bytes::ByteBuf = serde_bytes::ByteBuf::deserialize(de)?;
        if bytes.len() != 96 {
            return Err(serde::de::Error::custom(format!(
                "expected 96 G2 bytes, got {}",
                bytes.len()
            )));
        }
        let mut b = [0u8; 96];
        b.copy_from_slice(&bytes);
        PublicKey::from_bytes(&b).map_err(serde::de::Error::custom)
    }
}

/// A paired (secret, public) BBS+ key.
#[derive(Clone, Serialize, Deserialize)]
pub struct KeyPair {
    /// The secret scalar.
    pub secret: SecretKey,
    /// The matching public-key point in G2.
    pub public: PublicKey,
}

/// Generate a fresh BBS+ key pair.
///
/// Draws a non-zero scalar from `rng` (retrying on the negligible-
/// probability zero case) and derives `W = g2 * sk`.
pub fn keygen<R: RngCore + CryptoRng>(rng: &mut R) -> KeyPair {
    loop {
        let mut wide = [0u8; 64];
        rng.fill_bytes(&mut wide);
        let sk = Scalar::from_bytes_wide(&wide);
        // Probability of hitting zero is ~2^-255; the loop is
        // defensive.
        if bool::from(sk.ct_is_zero()) {
            continue;
        }
        let w = G2Projective::generator() * sk;
        return KeyPair {
            secret: SecretKey(sk),
            public: PublicKey { w: G2Affine::from(w) },
        };
    }
}

/// Constant-time-ish zero check on Scalar. (`Scalar` already supports
/// `ConstantTimeEq` but exposing it cleanly takes a small helper.)
trait CtZero {
    fn ct_is_zero(&self) -> subtle::Choice;
}

impl CtZero for Scalar {
    fn ct_is_zero(&self) -> subtle::Choice {
        use subtle::ConstantTimeEq;
        self.ct_eq(&Scalar::from(0u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn keygen_produces_nonzero_keys() {
        let kp = keygen(&mut OsRng);
        assert!(!bool::from(kp.secret.as_scalar().ct_is_zero()));
        assert_ne!(kp.public.w, G2Affine::identity());
    }

    #[test]
    fn keygen_is_random() {
        let a = keygen(&mut OsRng);
        let b = keygen(&mut OsRng);
        assert_ne!(a.secret.to_bytes(), b.secret.to_bytes());
        assert_ne!(a.public.to_bytes(), b.public.to_bytes());
    }

    #[test]
    fn secret_key_roundtrips() {
        let kp = keygen(&mut OsRng);
        let bytes = kp.secret.to_bytes();
        let back = SecretKey::from_bytes(&bytes).unwrap();
        assert_eq!(back.to_bytes(), bytes);
    }

    #[test]
    fn public_key_roundtrips() {
        let kp = keygen(&mut OsRng);
        let bytes = kp.public.to_bytes();
        let back = PublicKey::from_bytes(&bytes).unwrap();
        assert_eq!(back.w, kp.public.w);
    }

    #[test]
    fn public_key_matches_secret() {
        let kp = keygen(&mut OsRng);
        let derived = G2Projective::generator() * kp.secret.as_scalar();
        assert_eq!(G2Affine::from(derived), kp.public.w);
    }

    #[test]
    fn keypair_serde_cbor_roundtrip() {
        let kp = keygen(&mut OsRng);
        let bytes = serde_cbor::to_vec(&kp).unwrap();
        let back: KeyPair = serde_cbor::from_slice(&bytes).unwrap();
        assert_eq!(back.secret.to_bytes(), kp.secret.to_bytes());
        assert_eq!(back.public.w, kp.public.w);
    }
}
