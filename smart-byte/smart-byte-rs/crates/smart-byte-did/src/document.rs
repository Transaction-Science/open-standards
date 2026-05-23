//! DID Document data model (W3C DID Core 1.0 § 5).
//!
//! Fields are aligned with the JSON-LD serialisation but stored in a
//! plain Rust shape. The `@context` is preserved on deserialise and
//! emitted on serialise; everything else is strongly typed.

use serde::{Deserialize, Serialize};

use crate::did::Did;

/// Default DID Core v1 JSON-LD context.
pub const DID_CONTEXT_V1: &str = "https://www.w3.org/ns/did/v1";

/// A DID document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DidDocument {
    /// `@context` array. Defaults to `[DID_CONTEXT_V1]` on construction.
    #[serde(rename = "@context", default = "default_context")]
    pub context: serde_json::Value,
    /// The DID this document describes.
    pub id: Did,
    /// Controllers (DID Core § 5.1.2). Empty by default — controller is
    /// implicitly `id` if absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controller: Vec<Did>,
    /// Verification methods (DID Core § 5.2).
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "verificationMethod")]
    pub verification_method: Vec<VerificationMethod>,
    /// `authentication` relationship (DID Core § 5.3.1).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authentication: Vec<VerificationRelationship>,
    /// `assertionMethod` relationship (DID Core § 5.3.2).
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "assertionMethod")]
    pub assertion_method: Vec<VerificationRelationship>,
    /// `keyAgreement` relationship (DID Core § 5.3.3).
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "keyAgreement")]
    pub key_agreement: Vec<VerificationRelationship>,
    /// `capabilityInvocation` relationship (DID Core § 5.3.4).
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "capabilityInvocation")]
    pub capability_invocation: Vec<VerificationRelationship>,
    /// `capabilityDelegation` relationship (DID Core § 5.3.5).
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "capabilityDelegation")]
    pub capability_delegation: Vec<VerificationRelationship>,
    /// Service endpoints (DID Core § 5.4).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub service: Vec<Service>,
}

fn default_context() -> serde_json::Value {
    serde_json::Value::Array(vec![serde_json::Value::String(
        DID_CONTEXT_V1.into(),
    )])
}

impl DidDocument {
    /// Create an empty DID document for `id`, with the default v1 context.
    pub fn new(id: Did) -> Self {
        DidDocument {
            context: default_context(),
            id,
            controller: vec![],
            verification_method: vec![],
            authentication: vec![],
            assertion_method: vec![],
            key_agreement: vec![],
            capability_invocation: vec![],
            capability_delegation: vec![],
            service: vec![],
        }
    }
}

/// A verification method (DID Core § 5.2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationMethod {
    /// The verification method id — typically a DID URL with a fragment.
    pub id: String,
    /// JSON-LD type, e.g. `Multikey`, `JsonWebKey`,
    /// `Ed25519VerificationKey2020`, `EcdsaSecp256k1VerificationKey2019`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Controller of this verification method.
    pub controller: Did,
    /// `publicKeyMultibase` (DID Core § 5.2.3 / Multikey).
    #[serde(rename = "publicKeyMultibase", skip_serializing_if = "Option::is_none", default)]
    pub public_key_multibase: Option<String>,
    /// `publicKeyJwk` (RFC 7517).
    #[serde(rename = "publicKeyJwk", skip_serializing_if = "Option::is_none", default)]
    pub public_key_jwk: Option<Jwk>,
}

/// A verification relationship is either an embedded
/// [`VerificationMethod`] or a reference (DID URL) to one defined in the
/// document's `verificationMethod` array.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum VerificationRelationship {
    /// An inline verification method.
    Embedded(VerificationMethod),
    /// A reference (DID URL string) to a method in `verificationMethod`.
    Reference(String),
}

impl Serialize for VerificationRelationship {
    fn serialize<S: serde::Serializer>(
        &self,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match self {
            VerificationRelationship::Embedded(vm) => vm.serialize(s),
            VerificationRelationship::Reference(r) => s.serialize_str(r),
        }
    }
}

impl<'de> Deserialize<'de> for VerificationRelationship {
    fn deserialize<D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        match v {
            serde_json::Value::String(s) => {
                Ok(VerificationRelationship::Reference(s))
            }
            serde_json::Value::Object(_) => {
                let vm = serde_json::from_value::<VerificationMethod>(v)
                    .map_err(serde::de::Error::custom)?;
                Ok(VerificationRelationship::Embedded(vm))
            }
            _ => Err(serde::de::Error::custom(
                "verification relationship must be a string or object",
            )),
        }
    }
}

/// A service endpoint (DID Core § 5.4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Service {
    /// Service id — typically a DID URL fragment.
    pub id: String,
    /// Service type (e.g. `DIDCommMessaging`, `LinkedDomains`).
    #[serde(rename = "type")]
    pub type_: String,
    /// Service endpoint (URI, map, or set of URIs).
    #[serde(rename = "serviceEndpoint")]
    pub service_endpoint: ServiceEndpoint,
}

/// Service endpoint — DID Core § 5.4 allows a URI string, an array of
/// URIs, or an object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ServiceEndpoint {
    /// A single endpoint URI.
    Uri(String),
    /// Multiple endpoint URIs.
    Set(Vec<String>),
    /// A structured endpoint object.
    Map(serde_json::Map<String, serde_json::Value>),
}

/// A minimal JWK (RFC 7517). Only the fields needed for DID work are
/// strongly typed; unknown members are preserved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Jwk {
    /// Key type, e.g. `OKP`, `EC`, `RSA`.
    pub kty: String,
    /// Curve, e.g. `Ed25519`, `P-256`, `secp256k1`. Present for OKP/EC.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crv: Option<String>,
    /// `x` coordinate (base64url, no padding). Required for OKP/EC.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x: Option<String>,
    /// `y` coordinate (base64url, no padding). Required for EC.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub y: Option<String>,
    /// Algorithm, e.g. `EdDSA`, `ES256`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub alg: Option<String>,
    /// `kid` — key identifier.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kid: Option<String>,
    /// Allowed key operations.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub use_: Option<String>,
}
