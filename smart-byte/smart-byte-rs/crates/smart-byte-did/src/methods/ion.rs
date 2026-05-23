//! `did:ion` — Sidetree-based DID method (Bitcoin anchor).
//!
//! Real ION resolution requires an operator-side **Sidetree node** (e.g.
//! [`ion-microsoft/ion`][ion]) running against a Bitcoin full node. The
//! Sidetree node owns the IPFS retrieval, the Bitcoin anchor chain
//! traversal, and the operation-replay state machine; the result is a
//! DID document served via HTTP.
//!
//! This crate provides the parsing surface (so callers can carry
//! `did:ion:...` identifiers through the substrate) and a stub resolver
//! that returns [`DidError::Stubbed`] — wire it to your Sidetree node
//! in deployment.
//!
//! [ion]: https://github.com/decentralized-identity/ion

use async_trait::async_trait;

use crate::did::{Did, DidMethod};
use crate::error::DidError;
use crate::resolver::{ResolutionResult, Resolver};

/// Stub `did:ion` resolver.
///
/// A production deployment should replace this with a resolver that
/// proxies to a Sidetree node's resolution endpoint.
pub struct IonResolver;

impl IonResolver {
    /// Construct a new (stub) resolver.
    pub fn new() -> Self {
        IonResolver
    }
}

impl Default for IonResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Resolver for IonResolver {
    async fn resolve(&self, did: &Did) -> Result<ResolutionResult, DidError> {
        if did.method != DidMethod::Ion {
            return Err(DidError::InvalidIdentifier(format!(
                "not a did:ion: {did}"
            )));
        }
        Err(DidError::Stubbed(
            "did:ion resolution requires an operator-side Sidetree node".into(),
        ))
    }
}
