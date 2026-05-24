//! Scheme-agnostic ZK trait surface.
//!
//! Every backend (real or stub) implements [`ZkScheme`]. The trait is
//! intentionally minimal — keygen / prove / verify with opaque
//! byte-shaped statements and witnesses — so that the upstream
//! [`crate::presentation::VerifiablePresentation`] builder can mix
//! proofs from multiple schemes inside a single envelope without
//! taking a direct dependency on any one backend.
//!
//! The newtypes ([`ProvingKey`], [`VerifyingKey`], [`Proof`]) are wire
//! shapes. They carry an opaque `Vec<u8>` whose interpretation is
//! defined by the implementing scheme.

use serde::{Deserialize, Serialize};

use crate::error::ZkError;

/// Opaque proving-key wire shape. Interpretation is scheme-defined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvingKey(#[serde(with = "serde_bytes")] pub Vec<u8>);

/// Opaque verifying-key wire shape. Interpretation is scheme-defined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifyingKey(#[serde(with = "serde_bytes")] pub Vec<u8>);

/// Opaque proof wire shape. Interpretation is scheme-defined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof(#[serde(with = "serde_bytes")] pub Vec<u8>);

/// Scheme-agnostic ZK interface.
///
/// Backends may be **real** (e.g. [`crate::bulletproofs`]) or
/// **stub** (e.g. [`crate::groth16`], [`crate::plonk`]). Stubs satisfy
/// the trait so that downstream code can wire against the surface,
/// but their proofs are *not* cryptographically binding.
pub trait ZkScheme {
    /// Statement shape — i.e. what the verifier sees.
    type Statement;
    /// Witness shape — secret input known only to the prover.
    type Witness;

    /// Human-readable scheme tag (`"bulletproofs"`, `"groth16-stub"`,
    /// `"plonk-stub"`). Used as the `scheme` field in
    /// [`crate::presentation::VerifiablePresentation`] entries.
    fn name(&self) -> &'static str;

    /// Generate a (`pk`, `vk`) pair for the given statement template.
    fn keygen(&self, statement: &Self::Statement) -> Result<(ProvingKey, VerifyingKey), ZkError>;

    /// Produce a proof that `witness` satisfies `statement` under
    /// `pk`.
    fn prove(
        &self,
        pk: &ProvingKey,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<Proof, ZkError>;

    /// Verify `proof` against `statement` under `vk`.
    fn verify(
        &self,
        vk: &VerifyingKey,
        statement: &Self::Statement,
        proof: &Proof,
    ) -> Result<bool, ZkError>;
}
