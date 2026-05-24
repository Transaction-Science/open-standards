//! Confidential transfer sketch.
//!
//! Confidential-token systems hide some subset of a transfer's
//! plaintext (sender, receiver, amount) behind zero-knowledge
//! proofs. Each system makes different cuts; the common envelope is:
//!
//! ```text
//!   (public_fields, ciphertext_envelope, proof_envelope)
//! ```
//!
//! Examples:
//! - **Solana confidential token extension** — uses ElGamal +
//!   range-proofs; sender / receiver are public, amount is hidden,
//!   "auditor" key can decrypt for compliance.
//! - **Aleo** — fully shielded; sender, receiver, amount all hidden
//!   behind a zk-SNARK transition function.
//! - **Aztec** — UTXO note model + PLONK proofs; sender + amount
//!   hidden behind notes; receiver derived from notes' recipient
//!   key.
//!
//! This module sketches the envelope without committing to a
//! specific zk system. Operators integrating against a real chain
//! supply the concrete proof bytes via [`ProofEnvelope`].

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Which confidential system this transfer is from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfidentialSystem {
    /// Solana SPL Confidential Token extension. ElGamal + range
    /// proofs. Sender / receiver public, amount + auditor envelope
    /// hidden.
    SolanaSplConfidential,
    /// Aleo. Zexe-based zk-SNARK; everything shielded.
    Aleo,
    /// Aztec. Note-based UTXO model with PLONK.
    Aztec,
    /// Generic — proof system is operator-defined.
    Generic,
}

impl ConfidentialSystem {
    /// True iff sender + receiver addresses are part of the public
    /// envelope (i.e. visible on-chain). Solana's extension does
    /// this; Aleo / Aztec hide both.
    #[must_use]
    pub const fn has_public_endpoints(self) -> bool {
        matches!(self, Self::SolanaSplConfidential)
    }
}

/// Proof envelope. Opaque bytes — the proof system is the
/// [`ConfidentialTransfer::system`] tag, the bytes are whatever that
/// system's verifier consumes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofEnvelope {
    /// System this proof targets. Must match the enclosing
    /// transfer's `system` (cross-system mixing isn't supported).
    pub system: ConfidentialSystem,
    /// Proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs that the verifier checks the proof against.
    /// Format is system-specific; we just carry the bytes.
    pub public_inputs: Vec<u8>,
}

impl ProofEnvelope {
    /// Construct.
    #[must_use]
    pub fn new(system: ConfidentialSystem, proof: Vec<u8>, public_inputs: Vec<u8>) -> Self {
        Self {
            system,
            proof,
            public_inputs,
        }
    }

    /// Structural validation: proof non-empty and labelled correctly.
    ///
    /// # Errors
    /// Returns [`Error::Integrity`] on layout failure.
    pub fn check(&self) -> Result<()> {
        if self.proof.is_empty() {
            return Err(Error::Integrity("empty confidential-transfer proof".into()));
        }
        Ok(())
    }
}

/// A confidential transfer's structural envelope.
///
/// - `system` selects the proof system.
/// - `public_sender` / `public_receiver` are filled iff the system
///   reveals them.
/// - `ciphertext` is the hidden amount (system-dependent encoding).
/// - `proof` carries the validity proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfidentialTransfer {
    /// Proof system.
    pub system: ConfidentialSystem,
    /// Public sender, when the system reveals it.
    pub public_sender: Option<String>,
    /// Public receiver, when the system reveals it.
    pub public_receiver: Option<String>,
    /// Hidden-amount ciphertext.
    pub ciphertext: Vec<u8>,
    /// Validity proof.
    pub proof: ProofEnvelope,
}

impl ConfidentialTransfer {
    /// Construct.
    #[must_use]
    pub fn new(
        system: ConfidentialSystem,
        public_sender: Option<String>,
        public_receiver: Option<String>,
        ciphertext: Vec<u8>,
        proof: ProofEnvelope,
    ) -> Self {
        Self {
            system,
            public_sender,
            public_receiver,
            ciphertext,
            proof,
        }
    }

    /// Structural validation:
    /// - `proof.system == self.system`.
    /// - Endpoints present iff the system reveals them.
    /// - Ciphertext non-empty.
    ///
    /// # Errors
    /// Returns [`Error::Constraint`] or [`Error::Integrity`].
    pub fn check(&self) -> Result<()> {
        if self.proof.system != self.system {
            return Err(Error::Constraint {
                field: "proof.system",
                reason: "must match transfer system".into(),
            });
        }
        if self.ciphertext.is_empty() {
            return Err(Error::Integrity("empty confidential ciphertext".into()));
        }
        if self.system.has_public_endpoints() {
            if self.public_sender.is_none() {
                return Err(Error::MissingField("public_sender"));
            }
            if self.public_receiver.is_none() {
                return Err(Error::MissingField("public_receiver"));
            }
        } else {
            if self.public_sender.is_some() {
                return Err(Error::Constraint {
                    field: "public_sender",
                    reason: "system fully shields endpoints; field must be None".into(),
                });
            }
            if self.public_receiver.is_some() {
                return Err(Error::Constraint {
                    field: "public_receiver",
                    reason: "system fully shields endpoints; field must be None".into(),
                });
            }
        }
        self.proof.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solana_requires_endpoints() {
        let t = ConfidentialTransfer::new(
            ConfidentialSystem::SolanaSplConfidential,
            None,
            None,
            vec![1, 2, 3],
            ProofEnvelope::new(ConfidentialSystem::SolanaSplConfidential, vec![1], vec![]),
        );
        let err = t.check().unwrap_err();
        assert!(matches!(err, Error::MissingField("public_sender")));
    }

    #[test]
    fn aleo_rejects_endpoints() {
        let t = ConfidentialTransfer::new(
            ConfidentialSystem::Aleo,
            Some("aleo1xxx".into()),
            None,
            vec![1, 2],
            ProofEnvelope::new(ConfidentialSystem::Aleo, vec![1], vec![]),
        );
        let err = t.check().unwrap_err();
        assert!(matches!(err, Error::Constraint { .. }));
    }

    #[test]
    fn proof_system_must_match() {
        let t = ConfidentialTransfer::new(
            ConfidentialSystem::Aztec,
            None,
            None,
            vec![1, 2],
            ProofEnvelope::new(ConfidentialSystem::Aleo, vec![1], vec![]),
        );
        let err = t.check().unwrap_err();
        assert!(matches!(err, Error::Constraint { .. }));
    }

    #[test]
    fn empty_proof_rejected() {
        let envelope = ProofEnvelope::new(ConfidentialSystem::Aztec, vec![], vec![]);
        assert!(envelope.check().is_err());
    }

    #[test]
    fn happy_path_solana() {
        let t = ConfidentialTransfer::new(
            ConfidentialSystem::SolanaSplConfidential,
            Some("sender-pubkey".into()),
            Some("receiver-pubkey".into()),
            vec![0xaa; 64],
            ProofEnvelope::new(
                ConfidentialSystem::SolanaSplConfidential,
                vec![0xbb; 128],
                vec![],
            ),
        );
        t.check().unwrap();
    }

    #[test]
    fn happy_path_aleo() {
        let t = ConfidentialTransfer::new(
            ConfidentialSystem::Aleo,
            None,
            None,
            vec![0xaa; 32],
            ProofEnvelope::new(ConfidentialSystem::Aleo, vec![0xbb; 384], vec![]),
        );
        t.check().unwrap();
    }
}
