//! Ed25519 sign + verify over an envelope's SAID.
//!
//! Signing an envelope means signing its 32-byte SAID. Because the SAID
//! is itself a BLAKE3 commitment to the canonical CBOR of the envelope
//! (with `id` zeroed), signing the SAID is sufficient to bind the
//! signature to the entire envelope content.

use ed25519_dalek::Signer;
pub use ed25519_dalek::{Signature, SigningKey, Verifier, VerifyingKey};

use crate::envelope::Envelope;

/// Errors returned by verification.
#[derive(Debug, thiserror::Error)]
pub enum SignError {
    #[error("signature verification failed: {0}")]
    Bad(String),
}

/// Sign an envelope's SAID with the given Ed25519 signing key.
pub fn sign(envelope: &Envelope, key: &SigningKey) -> Signature {
    key.sign(envelope.id.as_bytes())
}

/// Verify that `sig` is a valid Ed25519 signature over `envelope.id`
/// produced by the private key matching `key`.
pub fn verify(
    envelope: &Envelope,
    sig: &Signature,
    key: &VerifyingKey,
) -> Result<(), SignError> {
    key.verify(envelope.id.as_bytes(), sig)
        .map_err(|e| SignError::Bad(e.to_string()))
}
