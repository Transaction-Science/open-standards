//! Draft FIPS 206 FN-DSA (formerly Falcon).
//!
//! FN-DSA is gated behind the `falcon` cargo feature because FIPS 206
//! is still in draft as of mid-2025. The implementation below uses
//! the [`pqcrypto-falcon`] crate, which wraps the PQClean Falcon
//! reference. Once the standard finalises and the wrapper aligns
//! parameter and encoding details to the published FIPS 206 byte
//! layout, the gating can be removed.
//!
//! Two parameter sets are exposed:
//!
//! * **FN-DSA-512** — NIST security category 1.
//! * **FN-DSA-1024** — NIST security category 5.
//!
//! Falcon signatures are variable-length; lengths reported by
//! [`crate::SignatureAlgorithm::signature_bytes_len`] are the maximum
//! sizes used for buffer allocation.
//!
//! [`pqcrypto-falcon`]: https://docs.rs/pqcrypto-falcon

use pqcrypto_falcon::{falcon1024, falcon512};
use pqcrypto_traits::sign::{
    DetachedSignature as _, PublicKey as _, SecretKey as _, VerificationError,
};
use rand::CryptoRng;
use zeroize::Zeroize;

use crate::algorithm::SignatureAlgorithm;
use crate::error::{PqError, Result};

/// FN-DSA parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FnDsaParam {
    /// FN-DSA-512 (NIST category 1).
    N512,
    /// FN-DSA-1024 (NIST category 5).
    N1024,
}

impl FnDsaParam {
    /// Translate to the [`SignatureAlgorithm`] enum.
    #[must_use]
    pub const fn algorithm(self) -> SignatureAlgorithm {
        match self {
            Self::N512 => SignatureAlgorithm::FnDsa512,
            Self::N1024 => SignatureAlgorithm::FnDsa1024,
        }
    }
}

/// FN-DSA public key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicKey {
    param: FnDsaParam,
    bytes: Vec<u8>,
}

impl PublicKey {
    /// Parameter set this key belongs to.
    #[must_use]
    pub fn param(&self) -> FnDsaParam {
        self.param
    }
    /// Raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// FN-DSA secret key. Zeroized on drop.
#[derive(Clone)]
pub struct SecretKey {
    param: FnDsaParam,
    bytes: Vec<u8>,
}

impl core::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FnDsaSecretKey")
            .field("param", &self.param)
            .field("bytes", &"<redacted>")
            .finish()
    }
}

impl SecretKey {
    /// Parameter set this key belongs to.
    #[must_use]
    pub fn param(&self) -> FnDsaParam {
        self.param
    }
    /// Raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
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

/// FN-DSA detached signature. Variable length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    param: FnDsaParam,
    bytes: Vec<u8>,
}

impl Signature {
    /// Parameter set this signature belongs to.
    #[must_use]
    pub fn param(&self) -> FnDsaParam {
        self.param
    }
    /// Raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// FN-DSA key pair.
#[derive(Debug, Clone)]
pub struct FnDsaKeyPair {
    /// Public verifying key.
    pub public: PublicKey,
    /// Secret signing key. Zeroized on drop.
    pub secret: SecretKey,
}

/// Generate a fresh FN-DSA key pair at the requested parameter set.
pub fn keygen<R: CryptoRng + rand::RngCore>(param: FnDsaParam, _rng: &mut R) -> FnDsaKeyPair {
    match param {
        FnDsaParam::N512 => {
            let (pk, sk) = falcon512::keypair();
            FnDsaKeyPair {
                public: PublicKey {
                    param,
                    bytes: pk.as_bytes().to_vec(),
                },
                secret: SecretKey {
                    param,
                    bytes: sk.as_bytes().to_vec(),
                },
            }
        }
        FnDsaParam::N1024 => {
            let (pk, sk) = falcon1024::keypair();
            FnDsaKeyPair {
                public: PublicKey {
                    param,
                    bytes: pk.as_bytes().to_vec(),
                },
                secret: SecretKey {
                    param,
                    bytes: sk.as_bytes().to_vec(),
                },
            }
        }
    }
}

/// Sign `message` with `secret_key`. Detached signature.
pub fn sign(message: &[u8], secret_key: &SecretKey) -> Result<Signature> {
    let bytes = match secret_key.param {
        FnDsaParam::N512 => {
            let sk = falcon512::SecretKey::from_bytes(&secret_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::FnDsa512, e.to_string())
            })?;
            falcon512::detached_sign(message, &sk).as_bytes().to_vec()
        }
        FnDsaParam::N1024 => {
            let sk = falcon1024::SecretKey::from_bytes(&secret_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::FnDsa1024, e.to_string())
            })?;
            falcon1024::detached_sign(message, &sk).as_bytes().to_vec()
        }
    };
    Ok(Signature {
        param: secret_key.param,
        bytes,
    })
}

/// Verify a detached FN-DSA signature.
pub fn verify(
    message: &[u8],
    signature: &Signature,
    public_key: &PublicKey,
) -> Result<()> {
    if signature.param != public_key.param {
        return Err(PqError::AlgorithmMismatch {
            expected: public_key.param.algorithm(),
            actual: signature.param.algorithm(),
        });
    }
    match signature.param {
        FnDsaParam::N512 => {
            let pk = falcon512::PublicKey::from_bytes(&public_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::FnDsa512, e.to_string())
            })?;
            let sig =
                falcon512::DetachedSignature::from_bytes(&signature.bytes).map_err(|e| {
                    PqError::MalformedSignature(SignatureAlgorithm::FnDsa512, e.to_string())
                })?;
            falcon512::verify_detached_signature(&sig, message, &pk).map_err(map_verify_err)
        }
        FnDsaParam::N1024 => {
            let pk = falcon1024::PublicKey::from_bytes(&public_key.bytes).map_err(|e| {
                PqError::MalformedKey(SignatureAlgorithm::FnDsa1024, e.to_string())
            })?;
            let sig = falcon1024::DetachedSignature::from_bytes(&signature.bytes).map_err(
                |e| PqError::MalformedSignature(SignatureAlgorithm::FnDsa1024, e.to_string()),
            )?;
            falcon1024::verify_detached_signature(&sig, message, &pk).map_err(map_verify_err)
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
    fn fndsa512_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(FnDsaParam::N512, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }

    #[test]
    fn fndsa1024_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(FnDsaParam::N1024, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }
}
