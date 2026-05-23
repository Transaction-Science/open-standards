//! Hybrid (classical + post-quantum) signatures.
//!
//! For the transition period, NIST and the IETF
//! (`draft-ietf-pquip-pqt-hybrid-terminology`) recommend hybrid
//! signatures that combine a battle-tested classical scheme
//! (Ed25519, ECDSA-P256) with a post-quantum scheme (typically
//! ML-DSA). The verifier requires **both** component signatures to
//! verify; this is the cautious mode that protects against unknown
//! cryptanalytic surprises in either family during the migration
//! window.
//!
//! This module ships an Ed25519 + ML-DSA hybrid. Ed25519 was chosen
//! over ECDSA-P256 for direct alignment with `smart-byte-core::sign`,
//! which already uses Ed25519. An ECDSA-P256 variant can be added
//! later once a corresponding classical signer plugs into
//! [`crate::Signer`].

use ed25519_dalek::{
    Signature as EdSignature, Signer as EdSigner, SigningKey, Verifier as EdVerifier,
    VerifyingKey,
};
use rand::{CryptoRng, RngCore, rngs::OsRng};
use zeroize::Zeroize;

use crate::algorithm::SignatureAlgorithm;
use crate::error::{PqError, Result};
use crate::mldsa::{self, MlDsaKeyPair, MlDsaLevel};

/// Classical half of a hybrid key pair. Today only Ed25519 is
/// implemented; an ECDSA-P256 case will join when there is a clean
/// signer for it in the workspace.
#[derive(Clone)]
pub struct ClassicalKeyPair {
    /// Classical signing key (Ed25519).
    pub signing: SigningKey,
    /// Classical verifying key (Ed25519).
    pub verifying: VerifyingKey,
}

impl core::fmt::Debug for ClassicalKeyPair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClassicalKeyPair")
            .field("signing", &"<redacted>")
            .field("verifying", &self.verifying)
            .finish()
    }
}

impl ClassicalKeyPair {
    /// Build a key pair from an existing Ed25519 signing key.
    #[must_use]
    pub fn from_signing_key(signing: SigningKey) -> Self {
        let verifying = signing.verifying_key();
        Self { signing, verifying }
    }

    /// Generate a fresh Ed25519 key pair.
    pub fn generate<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        let signing = SigningKey::generate(rng);
        let verifying = signing.verifying_key();
        Self { signing, verifying }
    }
}

impl Drop for ClassicalKeyPair {
    fn drop(&mut self) {
        // ed25519_dalek::SigningKey already zeroizes on drop via the
        // ZeroizeOnDrop derive in the upstream crate; this is left here
        // as documentation of intent.
    }
}

/// Public counterpart to [`ClassicalKeyPair`] / [`HybridKeyPair`].
#[derive(Debug, Clone)]
pub struct HybridPublicKey {
    /// Classical verifying key.
    pub classical: VerifyingKey,
    /// Post-quantum verifying key.
    pub pq: mldsa::PublicKey,
}

/// Hybrid key pair: classical signing material plus a parallel ML-DSA
/// signing key. Each signature produced by [`sign_hybrid`] is a pair
/// of independent signatures over the same message.
#[derive(Clone)]
pub struct HybridKeyPair {
    /// Classical half (Ed25519).
    pub classical: ClassicalKeyPair,
    /// Post-quantum half (ML-DSA).
    pub pq: MlDsaKeyPair,
}

impl core::fmt::Debug for HybridKeyPair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HybridKeyPair")
            .field("classical", &self.classical)
            .field("pq", &self.pq)
            .finish()
    }
}

impl HybridKeyPair {
    /// Generate a fresh hybrid key pair. Picks Ed25519 for the
    /// classical half and the requested ML-DSA level for the PQ half.
    pub fn generate<R: CryptoRng + RngCore>(level: MlDsaLevel, rng: &mut R) -> Self {
        let classical = ClassicalKeyPair::generate(rng);
        let pq = mldsa::keygen(level, rng);
        Self { classical, pq }
    }

    /// Derive the public-only view that verifiers need.
    #[must_use]
    pub fn public_key(&self) -> HybridPublicKey {
        HybridPublicKey {
            classical: self.classical.verifying,
            pq: self.pq.public.clone(),
        }
    }
}

/// A hybrid detached signature: two independent signatures over the
/// same message. Both must verify for the hybrid signature to be
/// accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridSignature {
    /// Classical signature bytes (Ed25519 -> 64 bytes).
    pub classical: Vec<u8>,
    /// Post-quantum detached signature bytes (ML-DSA -> level-dependent).
    pub pq: Vec<u8>,
    /// Level of the PQ component, recorded so that verification can
    /// dispatch on the correct ML-DSA parameter set without consulting
    /// the public key.
    pub pq_level: MlDsaLevel,
}

impl HybridSignature {
    /// Concatenated wire encoding: 64-byte Ed25519 signature followed
    /// by the level-dependent ML-DSA signature, with no length prefix
    /// (lengths are fixed by `pq_level`). Convenient for hand-rolled
    /// CBOR encoders.
    #[must_use]
    pub fn to_concatenated_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.classical.len() + self.pq.len());
        out.extend_from_slice(&self.classical);
        out.extend_from_slice(&self.pq);
        out
    }

    /// Inverse of [`Self::to_concatenated_bytes`]. The caller has to
    /// supply `pq_level` since the encoding itself does not include
    /// it.
    pub fn from_concatenated_bytes(pq_level: MlDsaLevel, bytes: &[u8]) -> Result<Self> {
        let pq_sig_len = pq_level.algorithm().signature_bytes_len();
        let total = 64 + pq_sig_len;
        if bytes.len() != total {
            return Err(PqError::MalformedSignature(
                pq_level.algorithm(),
                format!(
                    "hybrid signature has {} bytes, expected {total} (64 classical + {pq_sig_len} PQ)",
                    bytes.len()
                ),
            ));
        }
        let (classical, pq) = bytes.split_at(64);
        Ok(Self {
            classical: classical.to_vec(),
            pq: pq.to_vec(),
            pq_level,
        })
    }
}

impl Zeroize for HybridSignature {
    fn zeroize(&mut self) {
        self.classical.zeroize();
        self.pq.zeroize();
    }
}

/// Produce a hybrid signature over `message`.
///
/// The signature is the concatenation of an Ed25519 signature and an
/// ML-DSA signature, each computed independently over the same
/// message. There is no cross-binding between the two halves; per
/// `draft-ietf-pquip-pqt-hybrid-terminology`, this is the
/// straightforward "concatenation combiner" mode.
pub fn sign_hybrid(message: &[u8], key: &HybridKeyPair) -> Result<HybridSignature> {
    let classical_sig: EdSignature = key.classical.signing.sign(message);
    let pq_sig = mldsa::sign(message, &key.pq.secret)?;
    Ok(HybridSignature {
        classical: classical_sig.to_bytes().to_vec(),
        pq: pq_sig.as_bytes().to_vec(),
        pq_level: key.pq.level(),
    })
}

/// Verify a hybrid signature. Returns `Ok(())` only if **both**
/// components verify. Returns [`PqError::BadSignature`] if either
/// half fails (so callers can't distinguish which half was bad —
/// intentional, since either failure is grounds to reject).
pub fn verify_hybrid(
    message: &[u8],
    sig: &HybridSignature,
    key: &HybridPublicKey,
) -> Result<()> {
    // Classical first — it's cheap.
    if sig.classical.len() != 64 {
        return Err(PqError::MalformedSignature(
            SignatureAlgorithm::Ed25519,
            format!(
                "Ed25519 component must be 64 bytes, got {}",
                sig.classical.len()
            ),
        ));
    }
    let mut classical_bytes = [0u8; 64];
    classical_bytes.copy_from_slice(&sig.classical);
    let classical_sig = EdSignature::from_bytes(&classical_bytes);
    key.classical
        .verify(message, &classical_sig)
        .map_err(|_| PqError::BadSignature)?;

    // Then PQ.
    if sig.pq_level != key.pq.level() {
        return Err(PqError::AlgorithmMismatch {
            expected: key.pq.level().algorithm(),
            actual: sig.pq_level.algorithm(),
        });
    }
    let pq_sig = mldsa::Signature::from_bytes(sig.pq_level, &sig.pq)?;
    mldsa::verify(message, &pq_sig, &key.pq)?;

    Ok(())
}

/// Convenience: generate a hybrid key pair against the OS RNG.
#[must_use]
pub fn generate_hybrid(level: MlDsaLevel) -> HybridKeyPair {
    let mut rng = OsRng;
    HybridKeyPair::generate(level, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ed25519_mldsa65_round_trip() {
        let mut rng = OsRng;
        let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
        let pub_key = kp.public_key();
        let msg = b"smart byte envelope SAID";
        let sig = sign_hybrid(msg, &kp).expect("sign");
        verify_hybrid(msg, &sig, &pub_key).expect("verify");
    }

    #[test]
    fn hybrid_concatenated_round_trip() {
        let mut rng = OsRng;
        let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
        let msg = b"smart byte envelope SAID";
        let sig = sign_hybrid(msg, &kp).expect("sign");
        let raw = sig.to_concatenated_bytes();
        let parsed =
            HybridSignature::from_concatenated_bytes(MlDsaLevel::Level3, &raw).expect("parse");
        assert_eq!(parsed, sig);
    }

    #[test]
    fn pq_only_valid_is_rejected() {
        // Classical half is bogus; PQ half is valid. Must reject.
        let mut rng = OsRng;
        let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
        let pub_key = kp.public_key();
        let msg = b"smart byte envelope SAID";
        let mut sig = sign_hybrid(msg, &kp).expect("sign");
        sig.classical[0] ^= 0xFF;
        let err = verify_hybrid(msg, &sig, &pub_key).expect_err("must fail");
        assert!(matches!(err, PqError::BadSignature));
    }

    #[test]
    fn classical_only_valid_is_rejected() {
        // Classical half is valid; PQ half is tampered. Must reject.
        let mut rng = OsRng;
        let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
        let pub_key = kp.public_key();
        let msg = b"smart byte envelope SAID";
        let mut sig = sign_hybrid(msg, &kp).expect("sign");
        let idx = sig.pq.len() / 2;
        sig.pq[idx] ^= 0x01;
        let err = verify_hybrid(msg, &sig, &pub_key).expect_err("must fail");
        assert!(matches!(err, PqError::BadSignature));
    }
}
