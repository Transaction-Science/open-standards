//! Polymorphic `Signer` / `Verifier` traits.
//!
//! Smart Byte signs envelope SAIDs (32-byte BLAKE3 commitments). The
//! traits below give the rest of the substrate a single uniform entry
//! point regardless of which algorithm a particular key happens to
//! use; downstream code can iterate over a heterogeneous collection of
//! `Box<dyn Signer>` and sign without branching on enum variants.

use ed25519_dalek::{
    Signature as EdSignature, Signer as EdSignerTrait, SigningKey, Verifier as EdVerifierTrait,
    VerifyingKey,
};

use crate::algorithm::SignatureAlgorithm;
use crate::error::{PqError, Result};
use crate::hybrid::{HybridKeyPair, HybridPublicKey, HybridSignature, sign_hybrid, verify_hybrid};
use crate::mldsa::{self, MlDsaKeyPair, MlDsaLevel};
use crate::slhdsa::{self, SlhDsaKeyPair, SlhDsaParam};

/// Anything that can produce a signature over a message.
pub trait Signer {
    /// Which algorithm this signer uses (matches the byte placed in
    /// the envelope's algorithm-identifier field).
    fn algorithm(&self) -> SignatureAlgorithm;
    /// Produce a detached signature over `message`. Returns the raw
    /// signature bytes. For hybrid signers, the result is the
    /// concatenation produced by
    /// [`crate::hybrid::HybridSignature::to_concatenated_bytes`].
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>>;
    /// Public key bytes, in the encoding the corresponding `Verifier`
    /// expects.
    fn public_key_bytes(&self) -> Vec<u8>;
}

/// Anything that can verify a signature against a message + public key.
pub trait Verifier {
    /// Which algorithm this verifier handles.
    fn algorithm(&self) -> SignatureAlgorithm;
    /// Verify `signature` over `message` using `public_key`. Returns
    /// `Ok(())` only on cryptographic success.
    fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<()>;
}

// --- Ed25519 (delegates to smart-byte-core's primitive choice) -------------
//
// smart-byte-core::sign uses ed25519-dalek directly; the trait
// implementation here is built on the same crate, so the wire
// behavior matches byte-for-byte.

/// Ed25519 signer adapter (classical Smart Byte default).
pub struct Ed25519Signer {
    signing: SigningKey,
    verifying: VerifyingKey,
}

impl Ed25519Signer {
    /// Wrap an Ed25519 signing key.
    #[must_use]
    pub fn new(signing: SigningKey) -> Self {
        let verifying = signing.verifying_key();
        Self { signing, verifying }
    }
}

impl Signer for Ed25519Signer {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        let sig: EdSignature = self.signing.sign(message);
        Ok(sig.to_bytes().to_vec())
    }
    fn public_key_bytes(&self) -> Vec<u8> {
        self.verifying.to_bytes().to_vec()
    }
}

/// Ed25519 verifier adapter.
pub struct Ed25519Verifier;

impl Verifier for Ed25519Verifier {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }
    fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<()> {
        if public_key.len() != 32 {
            return Err(PqError::MalformedKey(
                SignatureAlgorithm::Ed25519,
                format!("expected 32-byte Ed25519 public key, got {}", public_key.len()),
            ));
        }
        if signature.len() != 64 {
            return Err(PqError::MalformedSignature(
                SignatureAlgorithm::Ed25519,
                format!("expected 64-byte Ed25519 signature, got {}", signature.len()),
            ));
        }
        let mut pk_bytes = [0u8; 32];
        pk_bytes.copy_from_slice(public_key);
        let vk = VerifyingKey::from_bytes(&pk_bytes).map_err(|e| {
            PqError::MalformedKey(SignatureAlgorithm::Ed25519, e.to_string())
        })?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(signature);
        let sig = EdSignature::from_bytes(&sig_bytes);
        vk.verify(message, &sig).map_err(|_| PqError::BadSignature)
    }
}

// --- ML-DSA ---------------------------------------------------------------

/// ML-DSA signer adapter.
pub struct MlDsaSigner {
    keypair: MlDsaKeyPair,
}

impl MlDsaSigner {
    /// Wrap a freshly-generated or restored ML-DSA key pair.
    #[must_use]
    pub fn new(keypair: MlDsaKeyPair) -> Self {
        Self { keypair }
    }

    /// Convenience: which parameter level this signer covers.
    #[must_use]
    pub fn level(&self) -> MlDsaLevel {
        self.keypair.level()
    }
}

impl Signer for MlDsaSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.keypair.level().algorithm()
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        Ok(mldsa::sign(message, &self.keypair.secret)?.as_bytes().to_vec())
    }
    fn public_key_bytes(&self) -> Vec<u8> {
        self.keypair.public.as_bytes().to_vec()
    }
}

/// ML-DSA verifier adapter. Constructed with the parameter level so
/// it can decode raw byte slices into the typed [`mldsa::PublicKey`].
pub struct MlDsaVerifier {
    level: MlDsaLevel,
}

impl MlDsaVerifier {
    /// Build a verifier for the given parameter level.
    #[must_use]
    pub fn new(level: MlDsaLevel) -> Self {
        Self { level }
    }
}

impl Verifier for MlDsaVerifier {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.level.algorithm()
    }
    fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<()> {
        let pk = mldsa::PublicKey::from_bytes(self.level, public_key)?;
        let sig = mldsa::Signature::from_bytes(self.level, signature)?;
        mldsa::verify(message, &sig, &pk)
    }
}

// --- SLH-DSA --------------------------------------------------------------

/// SLH-DSA signer adapter.
pub struct SlhDsaSigner {
    keypair: SlhDsaKeyPair,
}

impl SlhDsaSigner {
    /// Wrap an SLH-DSA key pair.
    #[must_use]
    pub fn new(keypair: SlhDsaKeyPair) -> Self {
        Self { keypair }
    }
}

impl Signer for SlhDsaSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.keypair.param().algorithm()
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        Ok(slhdsa::sign(message, &self.keypair.secret)?.as_bytes().to_vec())
    }
    fn public_key_bytes(&self) -> Vec<u8> {
        self.keypair.public.as_bytes().to_vec()
    }
}

/// SLH-DSA verifier adapter.
pub struct SlhDsaVerifier {
    param: SlhDsaParam,
}

impl SlhDsaVerifier {
    /// Build a verifier for the given parameter set.
    #[must_use]
    pub fn new(param: SlhDsaParam) -> Self {
        Self { param }
    }
}

impl Verifier for SlhDsaVerifier {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.param.algorithm()
    }
    fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<()> {
        let pk = slhdsa::PublicKey::from_bytes(self.param, public_key)?;
        let sig = slhdsa::Signature::from_bytes(self.param, signature)?;
        slhdsa::verify(message, &sig, &pk)
    }
}

// --- Hybrid ---------------------------------------------------------------

/// Hybrid (Ed25519 + ML-DSA) signer adapter. The on-the-wire signature
/// is the concatenation produced by
/// [`HybridSignature::to_concatenated_bytes`]; the public-key bytes
/// are `ed25519_pk (32) || mldsa_pk (level-dependent)`.
pub struct HybridSigner {
    keypair: HybridKeyPair,
}

impl HybridSigner {
    /// Wrap a hybrid key pair.
    #[must_use]
    pub fn new(keypair: HybridKeyPair) -> Self {
        Self { keypair }
    }

    /// Convenience: PQ-level information for callers needing to size
    /// signature buffers ahead of time.
    #[must_use]
    pub fn pq_level(&self) -> MlDsaLevel {
        self.keypair.pq.level()
    }
}

impl Signer for HybridSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        // Hybrid signatures carry the PQ algorithm byte in the
        // envelope's identifier field. Verifiers know the classical
        // half is always Ed25519 in v1.
        self.keypair.pq.level().algorithm()
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        let sig = sign_hybrid(message, &self.keypair)?;
        Ok(sig.to_concatenated_bytes())
    }
    fn public_key_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(32 + self.keypair.pq.level().algorithm().public_key_bytes_len());
        out.extend_from_slice(self.keypair.classical.verifying.as_bytes());
        out.extend_from_slice(self.keypair.pq.public.as_bytes());
        out
    }
}

/// Hybrid (Ed25519 + ML-DSA) verifier adapter.
pub struct HybridVerifier {
    pq_level: MlDsaLevel,
}

impl HybridVerifier {
    /// Build a verifier for the given PQ level (classical half is
    /// always Ed25519 in this implementation).
    #[must_use]
    pub fn new(pq_level: MlDsaLevel) -> Self {
        Self { pq_level }
    }
}

impl Verifier for HybridVerifier {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.pq_level.algorithm()
    }
    fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<()> {
        let pq_pk_len = self.pq_level.algorithm().public_key_bytes_len();
        let expected_pk = 32 + pq_pk_len;
        if public_key.len() != expected_pk {
            return Err(PqError::MalformedKey(
                self.pq_level.algorithm(),
                format!(
                    "hybrid public key has {} bytes, expected {expected_pk}",
                    public_key.len()
                ),
            ));
        }
        let (classical_pk_bytes, pq_pk_bytes) = public_key.split_at(32);
        let mut classical_arr = [0u8; 32];
        classical_arr.copy_from_slice(classical_pk_bytes);
        let classical_vk = VerifyingKey::from_bytes(&classical_arr).map_err(|e| {
            PqError::MalformedKey(SignatureAlgorithm::Ed25519, e.to_string())
        })?;
        let pq_pk = mldsa::PublicKey::from_bytes(self.pq_level, pq_pk_bytes)?;

        let hybrid_sig = HybridSignature::from_concatenated_bytes(self.pq_level, signature)?;
        let hybrid_pk = HybridPublicKey {
            classical: classical_vk,
            pq: pq_pk,
        };
        verify_hybrid(message, &hybrid_sig, &hybrid_pk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mldsa::keygen as mldsa_keygen;
    use crate::slhdsa::keygen as slhdsa_keygen;
    use rand::rngs::OsRng;

    #[test]
    fn ed25519_signer_round_trip() {
        let mut rng = OsRng;
        let sk = SigningKey::generate(&mut rng);
        let signer = Ed25519Signer::new(sk);
        let verifier = Ed25519Verifier;
        let msg = b"hello";
        let sig = signer.sign(msg).expect("sign");
        verifier
            .verify(msg, &sig, &signer.public_key_bytes())
            .expect("verify");
        assert_eq!(signer.algorithm(), SignatureAlgorithm::Ed25519);
    }

    #[test]
    fn mldsa_signer_round_trip() {
        let mut rng = OsRng;
        let kp = mldsa_keygen(MlDsaLevel::Level3, &mut rng);
        let signer = MlDsaSigner::new(kp);
        let verifier = MlDsaVerifier::new(MlDsaLevel::Level3);
        let msg = b"hello";
        let sig = signer.sign(msg).expect("sign");
        verifier
            .verify(msg, &sig, &signer.public_key_bytes())
            .expect("verify");
    }

    #[test]
    fn slhdsa_signer_round_trip() {
        let mut rng = OsRng;
        let kp = slhdsa_keygen(SlhDsaParam::Sha2_128f, &mut rng);
        let signer = SlhDsaSigner::new(kp);
        let verifier = SlhDsaVerifier::new(SlhDsaParam::Sha2_128f);
        let msg = b"hello";
        let sig = signer.sign(msg).expect("sign");
        verifier
            .verify(msg, &sig, &signer.public_key_bytes())
            .expect("verify");
    }

    #[test]
    fn hybrid_signer_round_trip() {
        let mut rng = OsRng;
        let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
        let signer = HybridSigner::new(kp);
        let verifier = HybridVerifier::new(MlDsaLevel::Level3);
        let msg = b"hello";
        let sig = signer.sign(msg).expect("sign");
        verifier
            .verify(msg, &sig, &signer.public_key_bytes())
            .expect("verify");
    }
}
