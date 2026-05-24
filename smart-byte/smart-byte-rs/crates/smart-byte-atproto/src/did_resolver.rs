//! DID resolution for AT Protocol identities.
//!
//! AT Protocol uses two DID methods:
//!
//! * **`did:web`** — handled directly by [`smart_byte_did::methods::web`].
//! * **`did:plc`** — Bluesky's "placeholder" method backed by a community
//!   directory (default `https://plc.directory`). This crate ships a
//!   thin [`PlcResolver`] that performs the directory lookup; the
//!   document is validated as a standard W3C DID document.
//!
//! The combined [`AtprotoResolver`] dispatches on the DID method.

use async_trait::async_trait;
use reqwest::Client;
use smart_byte_did::methods::web::WebResolver;
use smart_byte_did::resolver::{
    DocumentMetadata, ResolutionMetadata, ResolutionResult, Resolver,
};
use smart_byte_did::{Did, DidDocument, DidError, DidMethod};

/// `did:plc` directory resolver.
///
/// Resolves `did:plc:<id>` by issuing
/// `GET <directory>/did:plc:<id>` and parsing the JSON body as a DID
/// document. The default directory is `https://plc.directory`. Tests can
/// override the directory base URL (typically pointing at a wiremock
/// server) via [`PlcResolver::with_directory`].
pub struct PlcResolver {
    client: Client,
    directory: String,
}

impl PlcResolver {
    /// Construct a resolver pointing at `https://plc.directory`.
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("smart-byte-atproto/0.1")
                .build()
                .unwrap_or_else(|_| Client::new()),
            directory: "https://plc.directory".into(),
        }
    }

    /// Construct a resolver pointing at a custom directory base URL.
    pub fn with_directory(directory: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .user_agent("smart-byte-atproto/0.1")
                .build()
                .unwrap_or_else(|_| Client::new()),
            directory: directory.into(),
        }
    }

    /// Compute the URL the DID document is fetched from.
    pub fn document_url(&self, did: &Did) -> Result<String, DidError> {
        match &did.method {
            DidMethod::Custom(m) if m == "plc" => {
                Ok(format!("{}/{}", self.directory.trim_end_matches('/'), did))
            }
            other => Err(DidError::InvalidIdentifier(format!(
                "expected did:plc, got did:{other}"
            ))),
        }
    }
}

impl Default for PlcResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Resolver for PlcResolver {
    async fn resolve(
        &self,
        did: &Did,
    ) -> Result<ResolutionResult, DidError> {
        let url = self.document_url(did)?;
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| DidError::NetworkError(format!("GET {url}: {e}")))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(DidError::NotFound(did.to_string()));
        }
        if !status.is_success() {
            return Err(DidError::NetworkError(format!(
                "GET {url} → {status}"
            )));
        }
        let body = resp
            .text()
            .await
            .map_err(|e| DidError::NetworkError(format!("body {url}: {e}")))?;
        let doc: DidDocument = serde_json::from_str(&body)?;
        if doc.id != *did {
            return Err(DidError::InvalidDocument(format!(
                "document id {} does not match requested DID {}",
                doc.id, did
            )));
        }
        Ok(ResolutionResult {
            did_document: Some(doc),
            did_resolution_metadata: ResolutionMetadata {
                content_type: Some("application/did+json".into()),
                error: None,
            },
            did_document_metadata: DocumentMetadata::default(),
        })
    }
}

/// Combined AT Protocol DID resolver covering both `did:plc` and `did:web`.
pub struct AtprotoResolver {
    plc: PlcResolver,
    web: WebResolver,
}

impl AtprotoResolver {
    /// Build a resolver with the default `https://plc.directory` PLC
    /// directory and HTTPS for `did:web`.
    pub fn new() -> Self {
        Self {
            plc: PlcResolver::new(),
            web: WebResolver::new(),
        }
    }

    /// Build a resolver with explicit directory + scheme overrides for
    /// tests.
    pub fn with_overrides(
        plc_directory: impl Into<String>,
        web_scheme: impl Into<String>,
    ) -> Self {
        Self {
            plc: PlcResolver::with_directory(plc_directory),
            web: WebResolver::with_scheme(web_scheme),
        }
    }
}

impl Default for AtprotoResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Resolver for AtprotoResolver {
    async fn resolve(
        &self,
        did: &Did,
    ) -> Result<ResolutionResult, DidError> {
        match &did.method {
            DidMethod::Web => self.web.resolve(did).await,
            DidMethod::Custom(m) if m == "plc" => self.plc.resolve(did).await,
            other => Err(DidError::MethodNotSupported(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plc_url_format() {
        let r = PlcResolver::with_directory("http://localhost:1234");
        let did: Did = "did:plc:abcd1234abcd1234abcd1234".parse().unwrap();
        let url = r.document_url(&did).unwrap();
        assert_eq!(
            url,
            "http://localhost:1234/did:plc:abcd1234abcd1234abcd1234"
        );
    }

    #[test]
    fn plc_rejects_non_plc() {
        let r = PlcResolver::new();
        let did: Did = "did:web:example.com".parse().unwrap();
        assert!(r.document_url(&did).is_err());
    }
}
