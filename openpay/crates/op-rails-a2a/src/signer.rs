//! `Signer` trait — operator-supplied XML signing.
//!
//! PIX (and SEPA Inst's higher trust modes) require XML Signature on
//! the message body. The signing key MUST live in an HSM — AWS
//! `CloudHSM`, Azure Dedicated HSM, an on-prem nCipher, etc. `OpenPay`
//! never sees the private key.
//!
//! Operators implement [`Signer`] over their HSM client. We provide
//! [`NoOpSigner`] for tests and rails that don't require signing
//! (`FedNow`'s MQ layer handles authentication at the transport tier).

use crate::error::Result;

/// Signs payment-message bytes. Operators supply implementations
/// backed by their HSM.
pub trait Signer: Send + Sync {
    /// Sign `payload` and return the signature bytes. The exact format
    /// (raw DSA signature, XML Signature element, etc.) is rail-defined
    /// and contractually known to the driver that uses the signer.
    fn sign(&self, payload: &[u8]) -> Result<Vec<u8>>;

    /// Returns an identifier for the signing key, included in headers
    /// where the rail expects it (e.g. PIX `x-signature-key-id`).
    fn key_id(&self) -> &str;
}

/// Pass-through signer for tests and unsigned rails. Returns empty bytes.
#[derive(Debug, Default, Clone)]
pub struct NoOpSigner {
    key_id: String,
}

impl NoOpSigner {
    /// Construct with a label that gets returned from `key_id`.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            key_id: label.into(),
        }
    }
}

impl Signer for NoOpSigner {
    fn sign(&self, _payload: &[u8]) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn key_id(&self) -> &str {
        &self.key_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_signer_returns_empty_bytes() {
        let s = NoOpSigner::new("test-key");
        assert_eq!(s.sign(b"anything").unwrap(), Vec::<u8>::new());
        assert_eq!(s.key_id(), "test-key");
    }

    #[test]
    fn signer_is_object_safe() {
        // If Signer isn't dyn-compatible the orchestrator can't hold a
        // Box<dyn Signer>. This compile-time check is the test.
        let _: Box<dyn Signer> = Box::new(NoOpSigner::new("k"));
    }
}
