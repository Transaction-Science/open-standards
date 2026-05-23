//! FN-DSA stub. The `falcon` cargo feature is disabled, so no concrete
//! implementation is compiled. The public types still exist so callers
//! that match on [`crate::SignatureAlgorithm`] keep compiling; the
//! signer functions return [`crate::error::PqError::UnsupportedAlgorithm`].
//!
//! Enable the `falcon` feature to switch to the real implementation
//! backed by `pqcrypto-falcon`. See `src/fndsa.rs` for the real module.

use rand::CryptoRng;
use zeroize::Zeroize;

use crate::algorithm::SignatureAlgorithm;
use crate::error::{PqError, Result};

/// FN-DSA parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FnDsaParam {
    /// FN-DSA-512.
    N512,
    /// FN-DSA-1024.
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

/// FN-DSA public key (stub — enable `falcon` feature for the real type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicKey {
    /// Parameter set this key is tagged with.
    pub param: FnDsaParam,
}

/// FN-DSA secret key (stub).
#[derive(Clone)]
pub struct SecretKey {
    /// Parameter set this key is tagged with.
    pub param: FnDsaParam,
}

impl core::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FnDsaSecretKey(stub)")
            .field("param", &self.param)
            .finish()
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        // No bytes to scrub in the stub, but match the real impl shape.
    }
}

impl Zeroize for SecretKey {
    fn zeroize(&mut self) {
        // No-op for the stub.
    }
}

/// FN-DSA detached signature (stub).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// Parameter set this signature is tagged with.
    pub param: FnDsaParam,
}

/// FN-DSA key pair (stub).
#[derive(Debug, Clone)]
pub struct FnDsaKeyPair {
    /// Public verifying key.
    pub public: PublicKey,
    /// Secret signing key.
    pub secret: SecretKey,
}

/// Stub keygen. Returns a tagged but bytes-less key pair so callers
/// matching on enum variants compile. Use the `falcon` feature to get
/// real keys.
pub fn keygen<R: CryptoRng + rand::RngCore>(param: FnDsaParam, _rng: &mut R) -> FnDsaKeyPair {
    FnDsaKeyPair {
        public: PublicKey { param },
        secret: SecretKey { param },
    }
}

/// Stub sign. Always returns
/// [`crate::error::PqError::UnsupportedAlgorithm`].
pub fn sign(_message: &[u8], secret_key: &SecretKey) -> Result<Signature> {
    Err(PqError::UnsupportedAlgorithm(secret_key.param.algorithm()))
}

/// Stub verify. Always returns
/// [`crate::error::PqError::UnsupportedAlgorithm`].
pub fn verify(
    _message: &[u8],
    signature: &Signature,
    _public_key: &PublicKey,
) -> Result<()> {
    Err(PqError::UnsupportedAlgorithm(signature.param.algorithm()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn stub_sign_reports_unsupported() {
        let mut rng = OsRng;
        let kp = keygen(FnDsaParam::N512, &mut rng);
        let err = sign(b"hi", &kp.secret).expect_err("must be unsupported");
        assert!(matches!(err, PqError::UnsupportedAlgorithm(_)));
    }
}
