//! API-key authentication middleware.
//!
//! Implements [`ApiKeyAuthLayer`] — a `tower::Layer` that wraps the
//! router and rejects any request whose configured header does not
//! present an accepted key. Operators construct one with the set of
//! valid keys (typically loaded from env or a secret store) and
//! optionally a list of path prefixes that bypass the check (so
//! load-balancer health probes don't need a key).
//!
//! Failures return a `401 Unauthorized` with the same JSON envelope
//! shape as [`crate::error::ApiError`]:
//!
//! ```json
//! { "code": "unauthorized", "message": "...", "details": null }
//! ```
//!
//! ## Format
//!
//! The default header is `Authorization` with a `Bearer <key>`
//! prefix. Operators may override the header name; non-default
//! headers are read verbatim (no `Bearer` prefix stripping). The
//! key set is matched case-sensitively.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::Json;
use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::IntoResponse;
use serde_json::json;
use tower::{Layer, Service};

/// The `Authorization` header name, used when no override is set.
const DEFAULT_HEADER: &str = "authorization";

/// The `Bearer` prefix stripped from default-header values.
const BEARER_PREFIX: &str = "Bearer ";

/// Shared inner state — cheap to clone via [`Arc`].
#[derive(Debug)]
struct Inner {
    keys: HashSet<String>,
    header: String,
    bypass_paths: Vec<String>,
}

/// `tower::Layer` enforcing API-key authentication on every request.
///
/// Construct with [`Self::new`]; tweak the header or bypass list via
/// the builder methods. Clone is cheap — internal state lives behind
/// an [`Arc`].
#[derive(Debug, Clone)]
pub struct ApiKeyAuthLayer {
    inner: Arc<Inner>,
}

impl ApiKeyAuthLayer {
    /// Create a layer that accepts any key in `keys`. Header defaults
    /// to `Authorization` (with `Bearer ` prefix) and no paths bypass.
    #[must_use]
    pub fn new(keys: HashSet<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                keys,
                header: DEFAULT_HEADER.to_owned(),
                bypass_paths: Vec::new(),
            }),
        }
    }

    /// Override the header inspected for the key. Common alternatives:
    /// `X-API-Key`, `X-Auth-Token`. When set to anything other than
    /// `Authorization`, the value is matched verbatim (no `Bearer`
    /// stripping).
    #[must_use]
    pub fn with_header(mut self, header: impl Into<String>) -> Self {
        let header = header.into().to_ascii_lowercase();
        Arc::make_mut(&mut self.inner).header = header;
        self
    }

    /// Configure path prefixes that skip the auth check entirely.
    /// Typical use: `/health`, `/readiness`. Match is prefix-based:
    /// `/health` accepts `/health` and `/health/anything`.
    #[must_use]
    pub fn with_bypass_paths(mut self, paths: Vec<String>) -> Self {
        Arc::make_mut(&mut self.inner).bypass_paths = paths;
        self
    }
}

// `Arc::make_mut` requires `Clone` on the inner type.
impl Clone for Inner {
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            header: self.header.clone(),
            bypass_paths: self.bypass_paths.clone(),
        }
    }
}

impl<S> Layer<S> for ApiKeyAuthLayer {
    type Service = ApiKeyAuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ApiKeyAuthService {
            inner,
            cfg: Arc::clone(&self.inner),
        }
    }
}

/// Wrapped service that performs the per-request key check.
#[derive(Debug, Clone)]
pub struct ApiKeyAuthService<S> {
    inner: S,
    cfg: Arc<Inner>,
}

impl<S> Service<Request<Body>> for ApiKeyAuthService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        // Clone the inner service. `tower` recommends swapping the
        // ready service for a fresh clone so the cloned service that
        // gets driven in the future is the one that was polled ready.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let cfg = Arc::clone(&self.cfg);

        Box::pin(async move {
            let path = req.uri().path();
            // Bypass health / readiness etc.
            let bypass = cfg
                .bypass_paths
                .iter()
                .any(|p| path == p || path.starts_with(&format!("{p}/")));
            if bypass {
                return inner.call(req).await;
            }

            let header_val = req.headers().get(&cfg.header).and_then(|v| v.to_str().ok());

            let Some(candidate) = header_val else {
                return Ok(unauthorized("missing api key").into_response());
            };

            // Strip `Bearer ` only for the default Authorization header.
            let presented = if cfg.header == DEFAULT_HEADER {
                candidate.strip_prefix(BEARER_PREFIX).unwrap_or(candidate)
            } else {
                candidate
            };

            if cfg.keys.contains(presented) {
                inner.call(req).await
            } else {
                Ok(unauthorized("invalid api key").into_response())
            }
        })
    }
}

/// Build a 401 response with the [`crate::error::ApiError`] JSON
/// envelope shape.
fn unauthorized(message: &str) -> Response<Body> {
    let body = json!({
        "code": "unauthorized",
        "message": message,
        "details": serde_json::Value::Null,
    });
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}
