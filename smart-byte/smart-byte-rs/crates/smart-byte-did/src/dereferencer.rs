//! DID URL dereferencing (DID Core 1.0 § 7).
//!
//! Given a DID URL `did:<method>:<id>[/path][?query][#fragment]`,
//! dereference it to a specific [`VerificationMethod`] or [`Service`]
//! inside the DID document. The common case is a fragment that names a
//! verification method id.

use crate::did::DidUrl;
use crate::document::{Service, VerificationMethod, VerificationRelationship};
use crate::error::DidError;
use crate::resolver::Resolver;

/// What a DID URL dereferences to.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum DereferenceResult {
    /// The DID URL named a verification method by fragment.
    VerificationMethod(VerificationMethod),
    /// The DID URL named a service endpoint by fragment.
    Service(Service),
    /// The DID URL had no fragment; the whole document is returned.
    Document(Box<crate::document::DidDocument>),
}

/// Dereference `url` using `resolver`.
pub async fn dereference<R: Resolver + ?Sized>(
    resolver: &R,
    url: &DidUrl,
) -> Result<DereferenceResult, DidError> {
    let result = resolver.resolve(&url.did).await?;
    let doc = result.did_document.ok_or_else(|| {
        DidError::NotFound(url.did.to_string())
    })?;
    let frag = match &url.fragment {
        None => return Ok(DereferenceResult::Document(Box::new(doc))),
        Some(f) => f,
    };
    // Match against `id` either as a full DID URL (`did#frag`) or as a
    // bare fragment (`#frag`).
    let full = format!("{}#{frag}", url.did);
    let bare = format!("#{frag}");
    // Look up among verification methods (including those embedded in
    // verification relationships).
    let mut vms: Vec<VerificationMethod> =
        doc.verification_method.clone();
    for rel in doc
        .authentication
        .iter()
        .chain(doc.assertion_method.iter())
        .chain(doc.key_agreement.iter())
        .chain(doc.capability_invocation.iter())
        .chain(doc.capability_delegation.iter())
    {
        if let VerificationRelationship::Embedded(vm) = rel {
            vms.push(vm.clone());
        }
    }
    if let Some(vm) = vms
        .iter()
        .find(|v| v.id == full || v.id == bare || v.id == *frag)
    {
        return Ok(DereferenceResult::VerificationMethod(vm.clone()));
    }
    if let Some(svc) = doc
        .service
        .iter()
        .find(|s| s.id == full || s.id == bare || s.id == *frag)
    {
        return Ok(DereferenceResult::Service(svc.clone()));
    }
    Err(DidError::NotFound(format!(
        "fragment '{frag}' not found in {}",
        url.did
    )))
}
