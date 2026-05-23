//! Real HTTP transport backed by [`reqwest`]'s blocking client.
//!
//! Gated on the `reqwest-transport` cargo feature so the crate's
//! default dependency footprint stays minimal. Operators who want
//! to ship webhooks over the wire enable the feature and construct
//! [`ReqwestTransport`], which implements the crate's
//! [`HttpTransport`] trait.
//!
//! # Semantics
//!
//! - Network / DNS / TLS / timeout failures map to
//!   [`Error::Transport`].
//! - Non-2xx HTTP responses are NOT errors here — the dispatcher
//!   classifies them. The status code and response body flow
//!   through [`HttpResponse`] unchanged.
//! - Per-request `timeout_secs` is honored when set (>0); otherwise
//!   the client-builder default applies.
//!
//! # Example
//!
//! ```no_run
//! use op_webhook::reqwest_transport::ReqwestTransport;
//! use op_webhook::{HttpRequest, HttpTransport};
//!
//! let transport = ReqwestTransport::new().expect("client built");
//! let request = HttpRequest {
//!     url: "https://merchant.example/webhook".into(),
//!     headers: vec![("content-type".into(), "application/json".into())],
//!     body: br#"{"event":"intent.created"}"#.to_vec(),
//!     timeout_secs: 10,
//! };
//! let _ = transport.send(&request);
//! ```

use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::error::{Error, Result};
use crate::transport::{HttpRequest, HttpResponse, HttpTransport};

/// Default client-builder timeout, in seconds, used when neither the
/// builder nor the per-request override specifies one.
const DEFAULT_TIMEOUT_SECS: u64 = 10;

/// HTTP transport backed by `reqwest::blocking::Client`.
///
/// The inner client is reused across calls so connection pooling,
/// DNS caching, and TLS session reuse apply across deliveries. The
/// `Default` impl panics on client-build failure; use [`Self::new`]
/// for a fallible path.
#[derive(Debug, Clone)]
pub struct ReqwestTransport {
    client: Client,
    /// Per-call default timeout when an [`HttpRequest`] specifies
    /// `timeout_secs == 0`. Set by [`Self::with_timeout_secs`] and
    /// the constructors.
    default_timeout: Duration,
}

impl ReqwestTransport {
    /// Construct with a 10-second builder default timeout.
    ///
    /// # Errors
    /// Returns [`Error::Transport`] if the underlying
    /// `reqwest::blocking::Client` fails to build (e.g., TLS
    /// backend initialization failure).
    pub fn new() -> Result<Self> {
        Self::with_timeout_secs_fallible(DEFAULT_TIMEOUT_SECS as u32)
    }

    /// Builder method: set the default per-call timeout in seconds.
    ///
    /// This replaces the inner client with a freshly-built one. A
    /// value of `0` is treated as "no client-level timeout"; the
    /// per-request override (if any) still applies.
    #[must_use]
    pub fn with_timeout_secs(self, timeout_secs: u32) -> Self {
        // Best-effort: if the rebuild fails, retain the existing
        // client and update only the default_timeout field. The
        // fallible path is `with_timeout_secs_fallible`.
        match Self::with_timeout_secs_fallible(timeout_secs) {
            Ok(s) => s,
            Err(_) => Self {
                client: self.client,
                default_timeout: Duration::from_secs(u64::from(timeout_secs)),
            },
        }
    }

    /// Fallible counterpart to [`Self::with_timeout_secs`].
    fn with_timeout_secs_fallible(timeout_secs: u32) -> Result<Self> {
        let default_timeout = Duration::from_secs(u64::from(timeout_secs));
        let mut builder = Client::builder();
        if timeout_secs > 0 {
            builder = builder.timeout(default_timeout);
        }
        let client = builder
            .build()
            .map_err(|e| Error::Transport(format!("reqwest client build failed: {e}")))?;
        Ok(Self {
            client,
            default_timeout,
        })
    }
}

impl Default for ReqwestTransport {
    /// Default-construct via [`Self::new`].
    ///
    /// # Panics
    /// Panics if client construction fails. Prefer
    /// [`Self::new`] for a graceful error.
    fn default() -> Self {
        Self::new().expect("reqwest::blocking::Client build failed in Default")
    }
}

impl HttpTransport for ReqwestTransport {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse> {
        let mut header_map = HeaderMap::with_capacity(request.headers.len());
        for (name, value) in &request.headers {
            let header_name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|e| Error::Transport(format!("invalid header name {name:?}: {e}")))?;
            let header_value = HeaderValue::from_str(value)
                .map_err(|e| Error::Transport(format!("invalid header value for {name:?}: {e}")))?;
            header_map.append(header_name, header_value);
        }

        let timeout = if request.timeout_secs > 0 {
            Duration::from_secs(u64::from(request.timeout_secs))
        } else {
            self.default_timeout
        };

        let mut builder = self
            .client
            .post(&request.url)
            .headers(header_map)
            .body(request.body.clone());
        if timeout > Duration::ZERO {
            builder = builder.timeout(timeout);
        }

        let response = builder
            .send()
            .map_err(|e| Error::Transport(format!("request failed: {e}")))?;

        let status = response.status().as_u16();
        let body = response
            .bytes()
            .map_err(|e| Error::Transport(format!("response body read failed: {e}")))?
            .to_vec();

        Ok(HttpResponse { status, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_succeeds() {
        let transport = ReqwestTransport::new();
        assert!(transport.is_ok(), "client builder should succeed");
    }

    #[test]
    fn unreachable_url_returns_transport_error() {
        // Port 1 on loopback is reserved (tcpmux) and not listening
        // in any sane test environment; we expect a connection
        // refused / network error, which must surface as
        // Error::Transport.
        let transport = ReqwestTransport::new()
            .expect("client built")
            .with_timeout_secs(2);
        let request = HttpRequest {
            url: "http://127.0.0.1:1/".into(),
            headers: vec![],
            body: b"{}".to_vec(),
            timeout_secs: 2,
        };
        let started = std::time::Instant::now();
        let result = transport.send(&request);
        let elapsed = started.elapsed();
        assert!(
            matches!(result, Err(Error::Transport(_))),
            "expected Err(Error::Transport(_)), got {result:?}",
        );
        // Confirm we didn't blow past the timeout by an order of
        // magnitude. Generous bound for slow CI.
        assert!(
            elapsed < Duration::from_secs(15),
            "transport hung past timeout: {elapsed:?}",
        );
    }
}
