//! FIPS 205 SLH-DSA (formerly SPHINCS+).
//!
//! Twelve parameter sets across two hash families (SHA-2 and SHAKE),
//! three security levels (128, 192, 256-bit), and two size/speed
//! trade-offs (`s` = small signature, slower sign; `f` = fast sign,
//! larger signature).
//!
//! SLH-DSA is purely hash-based, which makes it the most conservative
//! choice in the post-quantum signature suite: its security relies on
//! the second-preimage resistance of its underlying hash, not on any
//! lattice or code-based assumption. The trade-off is large
//! signatures (between 7,856 and 49,856 bytes per the FIPS-205
//! parameter sets) and relatively slow signing (especially the `s`
//! variants).

use pqcrypto_sphincsplus::{
    sphincssha2128fsimple, sphincssha2128ssimple, sphincssha2192fsimple, sphincssha2192ssimple,
    sphincssha2256fsimple, sphincssha2256ssimple, sphincsshake128fsimple, sphincsshake128ssimple,
    sphincsshake192fsimple, sphincsshake192ssimple, sphincsshake256fsimple,
    sphincsshake256ssimple,
};
use pqcrypto_traits::sign::{
    DetachedSignature as _, PublicKey as _, SecretKey as _, VerificationError,
};
use rand::CryptoRng;
use zeroize::Zeroize;

use crate::algorithm::SignatureAlgorithm;
use crate::error::{PqError, Result};

/// An SLH-DSA parameter set. Mirrors the twelve variants standardised
/// in FIPS 205 (simple instantiations).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlhDsaParam {
    /// SLH-DSA-SHA2-128s
    Sha2_128s,
    /// SLH-DSA-SHA2-128f
    Sha2_128f,
    /// SLH-DSA-SHA2-192s
    Sha2_192s,
    /// SLH-DSA-SHA2-192f
    Sha2_192f,
    /// SLH-DSA-SHA2-256s
    Sha2_256s,
    /// SLH-DSA-SHA2-256f
    Sha2_256f,
    /// SLH-DSA-SHAKE-128s
    Shake_128s,
    /// SLH-DSA-SHAKE-128f
    Shake_128f,
    /// SLH-DSA-SHAKE-192s
    Shake_192s,
    /// SLH-DSA-SHAKE-192f
    Shake_192f,
    /// SLH-DSA-SHAKE-256s
    Shake_256s,
    /// SLH-DSA-SHAKE-256f
    Shake_256f,
}

impl SlhDsaParam {
    /// Map to the [`SignatureAlgorithm`] enum.
    #[must_use]
    pub const fn algorithm(self) -> SignatureAlgorithm {
        match self {
            Self::Sha2_128s => SignatureAlgorithm::SlhDsaSha2_128s,
            Self::Sha2_128f => SignatureAlgorithm::SlhDsaSha2_128f,
            Self::Sha2_192s => SignatureAlgorithm::SlhDsaSha2_192s,
            Self::Sha2_192f => SignatureAlgorithm::SlhDsaSha2_192f,
            Self::Sha2_256s => SignatureAlgorithm::SlhDsaSha2_256s,
            Self::Sha2_256f => SignatureAlgorithm::SlhDsaSha2_256f,
            Self::Shake_128s => SignatureAlgorithm::SlhDsaShake_128s,
            Self::Shake_128f => SignatureAlgorithm::SlhDsaShake_128f,
            Self::Shake_192s => SignatureAlgorithm::SlhDsaShake_192s,
            Self::Shake_192f => SignatureAlgorithm::SlhDsaShake_192f,
            Self::Shake_256s => SignatureAlgorithm::SlhDsaShake_256s,
            Self::Shake_256f => SignatureAlgorithm::SlhDsaShake_256f,
        }
    }
}

/// SLH-DSA public key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicKey {
    param: SlhDsaParam,
    bytes: Vec<u8>,
}

impl PublicKey {
    /// Parameter set this key belongs to.
    #[must_use]
    pub fn param(&self) -> SlhDsaParam {
        self.param
    }

    /// Raw FIPS 205 public-key encoding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode a raw FIPS 205 public key. Length-checked.
    pub fn from_bytes(param: SlhDsaParam, bytes: &[u8]) -> Result<Self> {
        let expected = param.algorithm().public_key_bytes_len();
        if bytes.len() != expected {
            return Err(PqError::MalformedKey(
                param.algorithm(),
                format!(
                    "public key has {} bytes, expected {expected}",
                    bytes.len()
                ),
            ));
        }
        Ok(Self {
            param,
            bytes: bytes.to_vec(),
        })
    }
}

/// SLH-DSA secret key. Zeroized on drop.
#[derive(Clone)]
pub struct SecretKey {
    param: SlhDsaParam,
    bytes: Vec<u8>,
}

impl core::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SlhDsaSecretKey")
            .field("param", &self.param)
            .field("bytes", &"<redacted>")
            .finish()
    }
}

impl SecretKey {
    /// Parameter set this key belongs to.
    #[must_use]
    pub fn param(&self) -> SlhDsaParam {
        self.param
    }

    /// Raw FIPS 205 secret-key encoding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode a raw FIPS 205 secret key. Length-checked.
    pub fn from_bytes(param: SlhDsaParam, bytes: &[u8]) -> Result<Self> {
        let expected = param.algorithm().secret_key_bytes_len();
        if bytes.len() != expected {
            return Err(PqError::MalformedKey(
                param.algorithm(),
                format!(
                    "secret key has {} bytes, expected {expected}",
                    bytes.len()
                ),
            ));
        }
        Ok(Self {
            param,
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

/// An SLH-DSA detached signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    param: SlhDsaParam,
    bytes: Vec<u8>,
}

impl Signature {
    /// Parameter set this signature belongs to.
    #[must_use]
    pub fn param(&self) -> SlhDsaParam {
        self.param
    }

    /// Raw FIPS 205 detached-signature encoding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode a raw FIPS 205 detached signature. Length-checked.
    pub fn from_bytes(param: SlhDsaParam, bytes: &[u8]) -> Result<Self> {
        let expected = param.algorithm().signature_bytes_len();
        if bytes.len() != expected {
            return Err(PqError::MalformedSignature(
                param.algorithm(),
                format!(
                    "signature has {} bytes, expected {expected}",
                    bytes.len()
                ),
            ));
        }
        Ok(Self {
            param,
            bytes: bytes.to_vec(),
        })
    }
}

/// An SLH-DSA key pair.
#[derive(Debug, Clone)]
pub struct SlhDsaKeyPair {
    /// Public verifying key.
    pub public: PublicKey,
    /// Secret signing key. Zeroized on drop.
    pub secret: SecretKey,
}

impl SlhDsaKeyPair {
    /// Parameter set this key pair was generated for.
    #[must_use]
    pub fn param(&self) -> SlhDsaParam {
        self.public.param
    }
}

// ---------------------------------------------------------------------------
// Dispatch macros. SLH-DSA has 12 variants and each lives in its own module
// in pqcrypto-sphincsplus with identical APIs but distinct types; a macro
// keeps dispatch one match arm per variant instead of twelve copy-paste blocks.
// ---------------------------------------------------------------------------

macro_rules! slhdsa_dispatch_keygen {
    ($module:ident, $param:expr) => {{
        let (pk, sk) = $module::keypair();
        SlhDsaKeyPair {
            public: PublicKey {
                param: $param,
                bytes: pk.as_bytes().to_vec(),
            },
            secret: SecretKey {
                param: $param,
                bytes: sk.as_bytes().to_vec(),
            },
        }
    }};
}

/// Generate a fresh SLH-DSA key pair at the requested parameter set.
///
/// As with ML-DSA, the underlying PQClean wrapper uses its own
/// system-seeded CSPRNG for FIPS-mandated reasons; the `rng` argument
/// is kept for API parity.
pub fn keygen<R: CryptoRng + rand::RngCore>(param: SlhDsaParam, _rng: &mut R) -> SlhDsaKeyPair {
    match param {
        SlhDsaParam::Sha2_128s => slhdsa_dispatch_keygen!(sphincssha2128ssimple, param),
        SlhDsaParam::Sha2_128f => slhdsa_dispatch_keygen!(sphincssha2128fsimple, param),
        SlhDsaParam::Sha2_192s => slhdsa_dispatch_keygen!(sphincssha2192ssimple, param),
        SlhDsaParam::Sha2_192f => slhdsa_dispatch_keygen!(sphincssha2192fsimple, param),
        SlhDsaParam::Sha2_256s => slhdsa_dispatch_keygen!(sphincssha2256ssimple, param),
        SlhDsaParam::Sha2_256f => slhdsa_dispatch_keygen!(sphincssha2256fsimple, param),
        SlhDsaParam::Shake_128s => slhdsa_dispatch_keygen!(sphincsshake128ssimple, param),
        SlhDsaParam::Shake_128f => slhdsa_dispatch_keygen!(sphincsshake128fsimple, param),
        SlhDsaParam::Shake_192s => slhdsa_dispatch_keygen!(sphincsshake192ssimple, param),
        SlhDsaParam::Shake_192f => slhdsa_dispatch_keygen!(sphincsshake192fsimple, param),
        SlhDsaParam::Shake_256s => slhdsa_dispatch_keygen!(sphincsshake256ssimple, param),
        SlhDsaParam::Shake_256f => slhdsa_dispatch_keygen!(sphincsshake256fsimple, param),
    }
}

macro_rules! slhdsa_dispatch_sign {
    ($module:ident, $alg:expr, $sk_bytes:expr, $msg:expr) => {{
        let sk = $module::SecretKey::from_bytes($sk_bytes)
            .map_err(|e| PqError::MalformedKey($alg, e.to_string()))?;
        $module::detached_sign($msg, &sk).as_bytes().to_vec()
    }};
}

/// Sign `message` with `secret_key`. Detached signature; the message
/// is not embedded.
pub fn sign(message: &[u8], secret_key: &SecretKey) -> Result<Signature> {
    let alg = secret_key.param.algorithm();
    let bytes = match secret_key.param {
        SlhDsaParam::Sha2_128s => {
            slhdsa_dispatch_sign!(sphincssha2128ssimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Sha2_128f => {
            slhdsa_dispatch_sign!(sphincssha2128fsimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Sha2_192s => {
            slhdsa_dispatch_sign!(sphincssha2192ssimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Sha2_192f => {
            slhdsa_dispatch_sign!(sphincssha2192fsimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Sha2_256s => {
            slhdsa_dispatch_sign!(sphincssha2256ssimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Sha2_256f => {
            slhdsa_dispatch_sign!(sphincssha2256fsimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Shake_128s => {
            slhdsa_dispatch_sign!(sphincsshake128ssimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Shake_128f => {
            slhdsa_dispatch_sign!(sphincsshake128fsimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Shake_192s => {
            slhdsa_dispatch_sign!(sphincsshake192ssimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Shake_192f => {
            slhdsa_dispatch_sign!(sphincsshake192fsimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Shake_256s => {
            slhdsa_dispatch_sign!(sphincsshake256ssimple, alg, &secret_key.bytes, message)
        }
        SlhDsaParam::Shake_256f => {
            slhdsa_dispatch_sign!(sphincsshake256fsimple, alg, &secret_key.bytes, message)
        }
    };
    Ok(Signature {
        param: secret_key.param,
        bytes,
    })
}

macro_rules! slhdsa_dispatch_verify {
    ($module:ident, $alg:expr, $pk_bytes:expr, $sig_bytes:expr, $msg:expr) => {{
        let pk = $module::PublicKey::from_bytes($pk_bytes)
            .map_err(|e| PqError::MalformedKey($alg, e.to_string()))?;
        let sig = $module::DetachedSignature::from_bytes($sig_bytes)
            .map_err(|e| PqError::MalformedSignature($alg, e.to_string()))?;
        $module::verify_detached_signature(&sig, $msg, &pk).map_err(map_verify_err)?;
    }};
}

/// Verify a detached SLH-DSA signature.
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
    let alg = signature.param.algorithm();
    match signature.param {
        SlhDsaParam::Sha2_128s => slhdsa_dispatch_verify!(
            sphincssha2128ssimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Sha2_128f => slhdsa_dispatch_verify!(
            sphincssha2128fsimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Sha2_192s => slhdsa_dispatch_verify!(
            sphincssha2192ssimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Sha2_192f => slhdsa_dispatch_verify!(
            sphincssha2192fsimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Sha2_256s => slhdsa_dispatch_verify!(
            sphincssha2256ssimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Sha2_256f => slhdsa_dispatch_verify!(
            sphincssha2256fsimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Shake_128s => slhdsa_dispatch_verify!(
            sphincsshake128ssimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Shake_128f => slhdsa_dispatch_verify!(
            sphincsshake128fsimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Shake_192s => slhdsa_dispatch_verify!(
            sphincsshake192ssimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Shake_192f => slhdsa_dispatch_verify!(
            sphincsshake192fsimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Shake_256s => slhdsa_dispatch_verify!(
            sphincsshake256ssimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
        SlhDsaParam::Shake_256f => slhdsa_dispatch_verify!(
            sphincsshake256fsimple,
            alg,
            &public_key.bytes,
            &signature.bytes,
            message
        ),
    }
    Ok(())
}

fn map_verify_err(_err: VerificationError) -> PqError {
    PqError::BadSignature
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    // SLH-DSA is slow, especially the `s` (small/slow) variants and the
    // `f` variants at 256-bit security. We exercise three carefully
    // chosen parameter sets for positive round-trips, plus one negative
    // (tamper) test. Keep total runtime modest.

    #[test]
    fn sha2_128s_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(SlhDsaParam::Sha2_128s, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }

    #[test]
    fn sha2_192f_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(SlhDsaParam::Sha2_192f, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }

    #[test]
    fn shake_128f_round_trip() {
        let mut rng = OsRng;
        let kp = keygen(SlhDsaParam::Shake_128f, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        verify(msg, &sig, &kp.public).expect("verify");
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let mut rng = OsRng;
        let kp = keygen(SlhDsaParam::Sha2_128f, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign(msg, &kp.secret).expect("sign");
        let mut bad = sig.bytes.clone();
        let idx = bad.len() / 2;
        bad[idx] ^= 0x01;
        let tampered = Signature {
            param: sig.param,
            bytes: bad,
        };
        let err = verify(msg, &tampered, &kp.public).expect_err("must fail");
        assert!(matches!(err, PqError::BadSignature));
    }
}
