//! `did:web` — DID document fetched over HTTPS.
//!
//! Per the `did:web` Method Specification, the method-specific id is a
//! percent-encoded host (optionally with port), optionally followed by a
//! colon-delimited path. Examples:
//!
//! * `did:web:example.com` → `https://example.com/.well-known/did.json`
//! * `did:web:example.com:users:alice` → `https://example.com/users/alice/did.json`
//! * `did:web:example.com%3A8443:bob` → `https://example.com:8443/bob/did.json`

use async_trait::async_trait;
use reqwest::Client;

use crate::did::{Did, DidMethod};
use crate::document::DidDocument;
use crate::error::DidError;
use crate::resolver::{DocumentMetadata, ResolutionMetadata, ResolutionResult, Resolver};

/// `did:web` resolver.
pub struct WebResolver {
    client: Client,
    /// Override the URL scheme. Defaults to `https`. Tests can set this
    /// to `http` to point at a [`wiremock`] mock server.
    scheme: String,
}

impl WebResolver {
    /// Construct a new resolver using HTTPS.
    pub fn new() -> Self {
        WebResolver {
            client: Client::builder()
                .user_agent("smart-byte-did/0.1")
                .build()
                .unwrap_or_else(|_| Client::new()),
            scheme: "https".into(),
        }
    }

    /// Construct a resolver with a custom URL scheme (for tests).
    pub fn with_scheme(scheme: impl Into<String>) -> Self {
        WebResolver {
            client: Client::builder()
                .user_agent("smart-byte-did/0.1")
                .build()
                .unwrap_or_else(|_| Client::new()),
            scheme: scheme.into(),
        }
    }

    /// Construct a resolver using a caller-supplied [`reqwest::Client`].
    pub fn with_client(client: Client, scheme: impl Into<String>) -> Self {
        WebResolver {
            client,
            scheme: scheme.into(),
        }
    }

    /// Compute the HTTP URL the DID document should be fetched from.
    pub fn document_url(&self, did: &Did) -> Result<String, DidError> {
        if did.method != DidMethod::Web {
            return Err(DidError::InvalidIdentifier(format!(
                "not a did:web: {did}"
            )));
        }
        let raw = &did.method_specific_id;
        let parts: Vec<&str> = raw.split(':').collect();
        if parts.is_empty() || parts[0].is_empty() {
            return Err(DidError::InvalidIdentifier(
                "empty did:web host".into(),
            ));
        }
        // Percent-decoded host (only `%3A` → `:` is interesting in
        // practice for the host:port form).
        let host = percent_decode(parts[0])?;
        let url = if parts.len() == 1 {
            format!("{}://{}/.well-known/did.json", self.scheme, host)
        } else {
            // Remaining segments form the path; segments are kept
            // percent-encoded as the spec stores them encoded.
            let path = parts[1..]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("/");
            format!("{}://{}/{}/did.json", self.scheme, host, path)
        };
        Ok(url)
    }
}

fn percent_decode(s: &str) -> Result<String, DidError> {
    // Minimal percent-decoder: handles %XX sequences.
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'%' {
            if i + 2 >= bytes.len() {
                return Err(DidError::InvalidIdentifier(
                    "truncated percent escape".into(),
                ));
            }
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3])
                .map_err(|_| {
                    DidError::InvalidIdentifier(
                        "non-utf8 percent escape".into(),
                    )
                })?;
            let v = u8::from_str_radix(hex, 16).map_err(|_| {
                DidError::InvalidIdentifier(
                    "bad percent escape hex".into(),
                )
            })?;
            out.push(v as char);
            i += 3;
        } else {
            out.push(b as char);
            i += 1;
        }
    }
    Ok(out)
}

impl Default for WebResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Resolver for WebResolver {
    async fn resolve(&self, did: &Did) -> Result<ResolutionResult, DidError> {
        let url = self.document_url(did)?;
        let resp = self.client.get(&url).send().await.map_err(|e| {
            DidError::NetworkError(format!("GET {url}: {e}"))
        })?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(DidError::NotFound(did.to_string()));
        }
        if !status.is_success() {
            return Err(DidError::NetworkError(format!(
                "GET {url} → {status}"
            )));
        }
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());
        // The spec allows application/json or application/did+json.
        if let Some(ct) = &content_type {
            let lc = ct.to_ascii_lowercase();
            if !lc.contains("json") {
                return Err(DidError::InvalidDocument(format!(
                    "unexpected content-type: {ct}"
                )));
            }
        }
        let body = resp.text().await.map_err(|e| {
            DidError::NetworkError(format!("body read {url}: {e}"))
        })?;
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
                content_type: content_type.or_else(|| {
                    Some("application/did+json".into())
                }),
                error: None,
            },
            did_document_metadata: DocumentMetadata::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_for_bare_host() {
        let r = WebResolver::new();
        let did: Did = "did:web:example.com".parse().unwrap();
        assert_eq!(
            r.document_url(&did).unwrap(),
            "https://example.com/.well-known/did.json"
        );
    }

    #[test]
    fn url_for_path() {
        let r = WebResolver::new();
        let did: Did = "did:web:example.com:users:alice".parse().unwrap();
        assert_eq!(
            r.document_url(&did).unwrap(),
            "https://example.com/users/alice/did.json"
        );
    }

    #[test]
    fn url_for_port_encoded() {
        let r = WebResolver::new();
        let did: Did = "did:web:example.com%3A8443:bob".parse().unwrap();
        assert_eq!(
            r.document_url(&did).unwrap(),
            "https://example.com:8443/bob/did.json"
        );
    }
}
