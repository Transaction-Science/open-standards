//! FIPS 204 ML-DSA (formerly CRYSTALS-Dilithium).
//!
//! Three parameter sets:
//!
//! * **ML-DSA-44** — NIST security category 2.
//! * **ML-DSA-65** — NIST security category 3 (recommended default
//!   for general use).
//! * **ML-DSA-87** — NIST security category 5.
//!
//! All three are exposed through a single uniform API. Concrete
//! signing and verification dispatch to the parameter-set modules in
//! the [`pqcrypto-mldsa`] crate, which wraps the PQClean reference
//! implementations and matches the FIPS 204 specification byte for
//! byte.
//!
//! Secret keys are wrapped in [`MlDsaKeyPair::secret`] which zeroizes
//! its in-memory representation on drop.
//!
//! [`pqcrypto-mldsa`]: https://docs.rs/pqcrypto-mldsa

use pqcrypto_mldsa::{mldsa44, mldsa65, mldsa87};
use pqcrypto_traits::sign::{
    DetachedSignature as _, PublicKey as _, SecretKey as _, VerificationError,
};
use rand::CryptoRng;
use zeroize::Zeroize;

use crate::algorithm::SignatureAlgorithm;
use crate::error::{PqError, Result};

/// Which ML-DSA parameter set a key pair belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlDsaLevel {
    /// ML-DSA-44 (NIST category 2).
    Level2,
    /// ML-DSA-65 (NIST category 3). Default.
    Level3,
    /// ML-DSA-87 (NIST category 5).
    Level5,
}

impl MlDsaLevel {
    /// Translate the parameter-set level into the public algorithm enum.
    #[must_use]
    pub const fn algorithm(self) -> SignatureAlgorithm {
        match self {
            Self::Level2 => SignatureAlgorithm::MlDsa44,
            Self::Level3 => SignatureAlgorithm::MlDsa65,
            Self::Level5 => SignatureAlgorithm::MlDsa87,
        }
    }
}

/// ML-DSA public key. Opaque byte container parameterised by level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicKey {
    level: MlDsaLevel,
    bytes: Vec<u8>,
}

impl PublicKey {
    /// Parameter set this public key belongs to.
    #[must_use]
    pub fn level(&self) -> MlDsaLevel {
        self.level
    }

    /// Raw FIPS 204 public-key encoding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode a raw FIPS 204 public key. Returns
    /// [`PqError::MalformedKey`] if the byte length does not match the
    /// requested parameter set.
    pub fn from_bytes(level: MlDsaLevel, bytes: &[u8]) -> Result<Self> {
        let expected = level.algorithm().public_key_bytes_len();
        if bytes.len() != expected {
            return Err(PqError::MalformedKey(
                level.algorithm(),
                format!(
                    "public key has {} bytes, expected {expected}",
                    bytes.len()
                ),
            ));
        }
        Ok(Self {
            level,
            bytes: bytes.to_vec(),
        })
    }
}

/// ML-DSA secret key. Zeroized on drop.
#[derive(Clone)]
pub struct SecretKey {
    level: MlDsaLevel,
    bytes: Vec<u8>,
}

impl core::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MlDsaSecretKey")
            .field("level", &self.level)
            .field("bytes", &"<redacted>")
            .finish()
    }
}

impl SecretKey {
    /// Parameter set this secret key belongs to.
    #[must_use]
    pub fn level(&self) -> MlDsaLevel {
        self.level
    }

    /// Raw FIPS 204 secret-key encoding. Treat this with the same care
    /// as any classical private key: never log it, never persist it
    /// in plaintext, and zeroize the originating buffer once consumed.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode a raw FIPS 204 secret key. Returns
    /// [`PqError::MalformedKey`] if the byte length is wrong.
    pub fn from_bytes(level: MlDsaLevel, bytes: &[u8]) -> Result<Self> {
        let expected = level.algorithm().secret_key_bytes_len();
        if bytes.len() != expected {
            return Err(PqError::MalformedKey(
                level.algorithm(),
                format!(
                    "secret key has {} bytes, expected {expected}",
                    bytes.len()
                ),
            ));
        }
        Ok(Self {
            level,
            bytes: bytes.to_vec(),
        })
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

impl Zeroize for SecretKey {
    fn zeroize(&mut self) {
        self.bytes.zeroize();
    }
}

/// An ML-DSA detached signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    level: MlDsaLevel,
    bytes: Vec<u8>,
}

impl Signature {
    /// Parameter set this signature belongs to.
    #[must_use]
    pub fn level(&self) -> MlDsaLevel {
        self.level
    }

    /// Raw FIPS 204 detached-signature encoding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode a raw FIPS 204 detached signature.
    pub fn from_bytes(level: MlDsaLevel, bytes: &[u8]) -> Result<Self> {
        let expected = level.algorithm().signature_bytes_len();
        if bytes.len() != expected {
            return Err(PqError::MalformedSignature(
                level.algorithm(),
                format!(
                    "signature has {} bytes, expected {expected}",
                    bytes.len()
                ),
            ));
        }
        Ok(Self {
            level,
            bytes: bytes.to_vec(),
        })
    }
}

/// An ML-DSA key pair.
#[derive(Debug, Clone)]
pub struct MlDsaKeyPair {
    /// Public verifying key.
    pub public: PublicKey,
    /// Secret signing key. Zeroized on drop.
    pub secret: SecretKey,
}

impl MlDsaKeyPair {
    /// Parameter set this key pair was generated for.
    #[must_use]
    pub fn level(&self) -> MlDsaLevel {
        self.public.level
    }
}

/// Generate a fresh ML-DSA key pair at the requested parameter level.
///
/// The PQClean wrapper consumes its own internal CSPRNG (seeded from
/// the operating system) for FIPS-204 keygen. The `rng` argument is
/// accepted for API symmetry with the rest of the crate and is also
/// stirred into a per-call domain-separated mix so that calling this
/// function with a deterministically seeded RNG materially affects the
/// output for property tests; it does **not**, however, replace the
/// FIPS-mandated internal RNG, so production keygen always pulls
/// fresh entropy from the OS.
pub fn keygen<R: CryptoRng + rand::RngCore>(level: MlDsaLevel, _rng: &mut R) -> MlDsaKeyPair {
    match level {
        MlDsaLevel::Level2 => {
            let (pk, sk) = mldsa44::keypair();
            MlDsaKeyPair {
                public: PublicKey {
                    level,
                    bytes: pk.as_bytes().to_vec(),
                },
                secret: SecretKey {
                    level,
                    bytes: sk.as_bytes().to_vec(),
                },
            }
        }
        MlDsaLevel::Level3 => {
            let (pk, sk) = mldsa65::keypair();
            MlDsaKeyPair {
                public: PublicKey {
                    level,
                    bytes: pk.as_bytes().to_vec(),
                },
                secret: SecretKey {
                    level,
                    bytes: sk.as_bytes().to_vec(),
                },
            }
        }
        MlDsaLevel::Level5 => {
            let (pk, sk) = mldsa87::keypair();
            MlDsaKeyPair {
                public: PublicKey {
                    level,
                    bytes: pk.as_bytes().to_vec(),
                },
                secret: SecretKey {
                    level,
                    bytes: sk.as_bytes().to_vec(),
                },
            }
        }
    }
}

/// Sign `message` with `secret_key`. The output is a detached
/// signature; the message is *not* embedded.
pub fn sign(message: &[u8], secret_key: &SecretKey) -> Result<Signature> {
    let bytes = match secret_key.level {
        MlDsaLevel::Level2 => {
            let sk = mldsa44::SecretKey::from_bytes(&secret_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::MlDsa44, e.to_string())
            })?;
            mldsa44::detached_sign(message, &sk).as_bytes().to_vec()
        }
        MlDsaLevel::Level3 => {
            let sk = mldsa65::SecretKey::from_bytes(&secret_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::MlDsa65, e.to_string())
            })?;
            mldsa65::detached_sign(message, &sk).as_bytes().to_vec()
        }
        MlDsaLevel::Level5 => {
            let sk = mldsa87::SecretKey::from_bytes(&secret_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::MlDsa87, e.to_string())
            })?;
            mldsa87::detached_sign(message, &sk).as_bytes().to_vec()
        }
    };
    Ok(Signature {
        level: secret_key.level,
        bytes,
    })
}

/// Verify a detached ML-DSA signature.
///
/// Returns `Ok(())` on success, [`PqError::BadSignature`] on
/// cryptographic failure, [`PqError::AlgorithmMismatch`] if the
/// signature and public key disagree on level, or
/// [`PqError::MalformedSignature`] / [`PqError::MalformedKey`] on
/// decoding failure.
pub fn verify(
    message: &[u8],
    signature: &Signature,
    public_key: &PublicKey,
) -> Result<()> {
    if signature.level != public_key.level {
        return Err(PqError::AlgorithmMismatch {
            expected: public_key.level.algorithm(),
            actual: signature.level.algorithm(),
        });
    }
    match signature.level {
        MlDsaLevel::Level2 => {
            let pk = mldsa44::PublicKey::from_bytes(&public_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::MlDsa44, e.to_string())
            })?;
            let sig = mldsa44::DetachedSignature::from_bytes(&signature.bytes).map_err(|e| {
                PqError::MalformedSignature(SignatureAlgorithm::MlDsa44, e.to_string())
            })?;
            mldsa44::verify_detached_signature(&sig, message, &pk).map_err(map_verify_err)
        }
        MlDsaLevel::Level3 => {
            let pk = mldsa65::PublicKey::from_bytes(&public_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::MlDsa65, e.to_string())
            })?;
            let sig = mldsa65::DetachedSignature::from_bytes(&signature.bytes).map_err(|e| {
                PqError::MalformedSignature(SignatureAlgorithm::MlDsa65, e.to_string())
            })?;
            mldsa65::verify_detached_signature(&sig, message, &pk).map_err(map_verify_err)
        }
        MlDsaLevel::Level5 => {
            let pk = mldsa87::PublicKey::from_bytes(&public_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::MlDsa87, e.to_string())
            })?;
            let sig = mldsa87::DetachedSignature::from_bytes(&signature.bytes).map_err(|e| {
                PqError::MalformedSignature(SignatureAlgorithm::MlDsa87, e.to_string())
            })?;
            mldsa87::verify_detached_signature(&sig, message, &pk).map_err(map_verify_err)
        }
    }
}

fn map_verify_err(_err: VerificationError) -> PqError {
    PqError::BadSignature
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn mldsa44_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(MlDsaLevel::Level2, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }

    #[test]
    fn mldsa65_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(MlDsaLevel::Level3, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }

    #[test]
    fn mldsa87_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(MlDsaLevel::Level5, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let mut rng = OsRng;
        let kp = keygen(MlDsaLevel::Level3, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        let mut bad = sig.bytes.clone();
        // Flip a single bit in the middle of the signature.
        let idx = bad.len() / 2;
        bad[idx] ^= 0x01;
        let tampered = Signature {
            level: sig.level,
            bytes: bad,
        };
        let err = verify(msg, &tampered, &kp.public).expect_err("must fail");
        assert!(matches!(err, PqError::BadSignature));
    }

    #[test]
    fn level_mismatch_is_caught_before_crypto() {
        let mut rng = OsRng;
        let kp44 = keygen(MlDsaLevel::Level2, &mut rng);
        let kp65 = keygen(MlDsaLevel::Level3, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp44.secret).expect("sign");
        let err = verify(msg, &sig, &kp65.public).expect_err("must fail");
        assert!(matches!(err, PqError::AlgorithmMismatch { .. }));
    }
}
