//! Vendor authentication header construction.
//!
//! Each variant maps to a single HTTP header pattern. The wrapped string
//! is the raw secret material; it is never written to logs (see
//! [`Auth::redacted`]).

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::error::VendorError;

/// Vendor credential.
#[derive(Debug, Clone)]
pub enum Auth {
    /// `Authorization: Bearer <token>` — used by OpenAI, Mistral, Cohere,
    /// Groq, Together, Fireworks.
    Bearer(String),
    /// `x-api-key: <key>` — used by Anthropic.
    ApiKey(String),
    /// Google's query-parameter style (`?key=<key>`). The key is appended
    /// to the request URL by the [`GoogleBackend`](crate::GoogleBackend).
    GoogleApiKey(String),
}

impl Auth {
    /// Apply this credential to a `HeaderMap`. No-op for
    /// [`Auth::GoogleApiKey`] (Google authenticates via URL query).
    pub fn apply(&self, headers: &mut HeaderMap) -> Result<(), VendorError> {
        match self {
            Auth::Bearer(token) => {
                let v = HeaderValue::from_str(&format!("Bearer {token}"))
                    .map_err(|_| VendorError::InvalidApiKey)?;
                headers.insert(reqwest::header::AUTHORIZATION, v);
            }
            Auth::ApiKey(key) => {
                let name = HeaderName::from_static("x-api-key");
                let v = HeaderValue::from_str(key).map_err(|_| VendorError::InvalidApiKey)?;
                headers.insert(name, v);
            }
            Auth::GoogleApiKey(_) => {}
        }
        Ok(())
    }

    /// Returns the raw secret material — used only by
    /// [`GoogleBackend`](crate::GoogleBackend) to append `?key=...` to
    /// the request URL.
    pub(crate) fn google_key(&self) -> Option<&str> {
        match self {
            Auth::GoogleApiKey(k) => Some(k),
            _ => None,
        }
    }

    /// A constant placeholder safe for structured logs. The actual key
    /// never leaves the struct.
    pub fn redacted(&self) -> &'static str {
        "<redacted>"
    }
}
