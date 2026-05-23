//! Resolver trait and the [`UniversalResolver`] that dispatches by method.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::did::{Did, DidMethod};
use crate::document::DidDocument;
use crate::error::DidError;
use crate::methods::{jwk::JwkResolver, key::KeyResolver, peer::PeerResolver, web::WebResolver};

#[cfg(feature = "ion")]
use crate::methods::ion::IonResolver;

/// W3C DID Resolution v1 result (DID Resolution § 7).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResolutionResult {
    /// The resolved DID document, or `None` if the document was
    /// deactivated or not found.
    #[serde(rename = "didDocument", default)]
    pub did_document: Option<DidDocument>,
    /// Metadata about the resolution process itself.
    #[serde(rename = "didResolutionMetadata", default)]
    pub did_resolution_metadata: ResolutionMetadata,
    /// Metadata about the document (created, updated, deactivated, etc.).
    #[serde(rename = "didDocumentMetadata", default)]
    pub did_document_metadata: DocumentMetadata,
}

/// Resolution metadata (DID Resolution § 7.1).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResolutionMetadata {
    /// MIME content type of the document representation.
    #[serde(rename = "contentType", skip_serializing_if = "Option::is_none", default)]
    pub content_type: Option<String>,
    /// Resolver-supplied error code (DID Resolution § 7.1.2).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<String>,
}

/// Document metadata (DID Resolution § 7.2).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    /// ISO-8601 creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created: Option<String>,
    /// ISO-8601 last-update timestamp.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub updated: Option<String>,
    /// `true` if the DID has been deactivated.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub deactivated: Option<bool>,
    /// Version id, if the method versions documents.
    #[serde(rename = "versionId", skip_serializing_if = "Option::is_none", default)]
    pub version_id: Option<String>,
}

/// A method-specific resolver. Resolution is async because some methods
/// (notably `did:web`) involve network I/O.
#[async_trait]
pub trait Resolver: Send + Sync {
    /// Resolve `did` to a [`ResolutionResult`].
    async fn resolve(&self, did: &Did) -> Result<ResolutionResult, DidError>;
}

/// Universal resolver that dispatches to method-specific resolvers based
/// on `did.method`.
///
/// By default it knows about [`DidMethod::Key`], [`DidMethod::Web`],
/// [`DidMethod::Peer`] and [`DidMethod::Jwk`] (and [`DidMethod::Ion`]
/// when the `ion` feature is enabled). Custom resolvers can be
/// registered via [`UniversalResolver::register`].
pub struct UniversalResolver {
    resolvers: HashMap<String, Box<dyn Resolver>>,
}

impl Default for UniversalResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl UniversalResolver {
    /// Build a resolver with the built-in method resolvers wired up.
    pub fn new() -> Self {
        let mut resolvers: HashMap<String, Box<dyn Resolver>> = HashMap::new();
        resolvers.insert("key".into(), Box::new(KeyResolver::new()));
        resolvers.insert("web".into(), Box::new(WebResolver::new()));
        resolvers.insert("peer".into(), Box::new(PeerResolver::new()));
        resolvers.insert("jwk".into(), Box::new(JwkResolver::new()));
        #[cfg(feature = "ion")]
        resolvers.insert("ion".into(), Box::new(IonResolver::new()));
        UniversalResolver { resolvers }
    }

    /// Build a resolver with no built-in methods registered.
    pub fn empty() -> Self {
        UniversalResolver {
            resolvers: HashMap::new(),
        }
    }

    /// Register (or replace) a resolver for a specific DID method name.
    pub fn register(
        &mut self,
        method: impl Into<String>,
        resolver: Box<dyn Resolver>,
    ) {
        self.resolvers.insert(method.into(), resolver);
    }

    /// Look up the resolver for a method, if any.
    pub fn resolver_for(&self, method: &DidMethod) -> Option<&dyn Resolver> {
        self.resolvers.get(method.as_str()).map(|r| r.as_ref())
    }
}

#[async_trait]
impl Resolver for UniversalResolver {
    async fn resolve(&self, did: &Did) -> Result<ResolutionResult, DidError> {
        let method_name = did.method.as_str();
        let r = self.resolvers.get(method_name).ok_or_else(|| {
            DidError::MethodNotSupported(method_name.to_string())
        })?;
        r.resolve(did).await
    }
}
