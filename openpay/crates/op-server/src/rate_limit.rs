//! Per-source token-bucket rate limiting.
//!
//! Implements [`RateLimitLayer`] — a `tower::Layer` that throttles
//! requests using a classic token-bucket algorithm. Each source key
//! (the authenticated API key when present, otherwise the
//! `X-Forwarded-For` / `Forwarded` peer IP) gets its own bucket of
//! `capacity` tokens that refill at `refill_per_second` until full.
//!
//! Requests with empty buckets receive a `429 Too Many Requests` plus
//! a `Retry-After` header. The response body matches the
//! [`crate::error::ApiError`] JSON envelope:
//!
//! ```json
//! { "code": "rate_limited", "message": "...", "details": null }
//! ```
//!
//! ## Hand-rolled vs. `governor`
//!
//! We avoid pulling in `governor` / `tower-governor` because (a) the
//! workspace policy is to keep the dependency surface tight and (b)
//! the token-bucket math is ~30 lines once you have a clock.
//!
//! ## Mockable clock
//!
//! [`RateLimitLayer`] accepts an injectable `Arc<dyn Fn() -> u64 + …>`
//! returning unix seconds. Tests advance the clock; production passes
//! `SystemTime::now()`. The clock is sampled at second granularity —
//! sub-second jitter is irrelevant at this layer.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::IntoResponse;
use serde_json::json;
use tower::{Layer, Service};

/// Boxed clock returning unix-seconds. Cheap to clone.
pub type Clock = Arc<dyn Fn() -> u64 + Send + Sync + 'static>;

/// Single token-bucket state. Tokens stored as a float so partial
/// refills survive between calls without rounding to zero.
#[derive(Debug, Clone, Copy)]
struct Bucket {
    tokens: f64,
    last_refill: u64,
}

/// Per-source bucket map plus refill parameters.
struct State {
    buckets: Mutex<HashMap<String, Bucket>>,
    capacity: f64,
    refill_per_second: f64,
    clock: Clock,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("capacity", &self.capacity)
            .field("refill_per_second", &self.refill_per_second)
            .finish_non_exhaustive()
    }
}

/// `tower::Layer` applying a token-bucket throttle keyed by API key
/// or source IP.
#[derive(Debug, Clone)]
pub struct RateLimitLayer {
    state: Arc<State>,
}

impl RateLimitLayer {
    /// Bucket of `capacity` tokens, refilling at `refill_per_second`
    /// tokens / sec. Uses the wall clock by default.
    #[must_use]
    pub fn new(capacity: u32, refill_per_second: f64) -> Self {
        Self {
            state: Arc::new(State {
                buckets: Mutex::new(HashMap::new()),
                capacity: f64::from(capacity),
                refill_per_second,
                clock: Arc::new(default_clock),
            }),
        }
    }

    /// Convenience: a bucket of `capacity` requests / minute. Refill
    /// rate becomes `capacity / 60.0` tokens per second so a steady
    /// stream of one request per (60 / capacity) seconds never
    /// drains the bucket.
    #[must_use]
    pub fn per_minute(capacity: u32) -> Self {
        Self::new(capacity, f64::from(capacity) / 60.0)
    }

    /// Replace the clock with an injectable function. Used by tests
    /// to advance time without sleeping.
    #[must_use]
    pub fn with_clock<F>(mut self, clock: F) -> Self
    where
        F: Fn() -> u64 + Send + Sync + 'static,
    {
        Arc::make_mut(&mut self.state).clock = Arc::new(clock);
        self
    }
}

impl Clone for State {
    fn clone(&self) -> Self {
        Self {
            buckets: Mutex::new(self.buckets.lock().expect("buckets mutex").clone()),
            capacity: self.capacity,
            refill_per_second: self.refill_per_second,
            clock: Arc::clone(&self.clock),
        }
    }
}

fn default_clock() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl<S> Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            state: Arc::clone(&self.state),
        }
    }
}

/// Wrapped service that consults the bucket map on every call.
#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    state: Arc<State>,
}

impl<S> std::fmt::Debug for RateLimitService<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimitService")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl<S> Service<Request<Body>> for RateLimitService<S>
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
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let state = Arc::clone(&self.state);

        Box::pin(async move {
            let key = source_key(&req);
            let outcome = consume_token(&state, &key);
            match outcome {
                BucketOutcome::Allowed => inner.call(req).await,
                BucketOutcome::Limited { retry_after_secs } => {
                    Ok(rate_limited(retry_after_secs).into_response())
                }
            }
        })
    }
}

enum BucketOutcome {
    Allowed,
    Limited { retry_after_secs: u64 },
}

/// Refill then attempt to consume a single token. Returns whether the
/// request is allowed and how many seconds the caller should wait
/// before retrying when limited.
fn consume_token(state: &State, key: &str) -> BucketOutcome {
    let now = (state.clock)();
    let mut buckets = state.buckets.lock().expect("buckets mutex");
    let bucket = buckets.entry(key.to_owned()).or_insert(Bucket {
        tokens: state.capacity,
        last_refill: now,
    });

    // Refill — saturating subtract guards against backwards clocks.
    let elapsed = now.saturating_sub(bucket.last_refill);
    if elapsed > 0 {
        // `as f64` from u64 — the elapsed gap is real-world seconds
        // since the last refill, so even u32::MAX would round-trip
        // through f64 with negligible loss.
        #[allow(clippy::cast_precision_loss)]
        let refill = elapsed as f64 * state.refill_per_second;
        bucket.tokens = (bucket.tokens + refill).min(state.capacity);
        bucket.last_refill = now;
    }

    if bucket.tokens >= 1.0 {
        bucket.tokens -= 1.0;
        BucketOutcome::Allowed
    } else {
        // Compute when one full token will be available again.
        let deficit = 1.0 - bucket.tokens;
        let secs = if state.refill_per_second > 0.0 {
            (deficit / state.refill_per_second).ceil()
        } else {
            1.0
        };
        // `secs` is a small positive number derived from a token
        // deficit (≤ capacity) — fits easily in u64.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let retry_after_secs = secs as u64;
        BucketOutcome::Limited {
            retry_after_secs: retry_after_secs.max(1),
        }
    }
}

/// Extract the bucket key from the request. Priority:
/// 1. `Authorization: Bearer <key>` (or bare value of that header)
/// 2. `X-Forwarded-For` first hop
/// 3. `Forwarded` first `for=`
/// 4. Fallback to the literal string `"anonymous"` so missing-IP
///    deployments still share a single bucket rather than panicking.
fn source_key(req: &Request<Body>) -> String {
    if let Some(auth) = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
    {
        let key = auth.strip_prefix("Bearer ").unwrap_or(auth);
        return format!("key:{key}");
    }
    if let Some(xff) = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        && let Some(first) = xff.split(',').next()
    {
        let ip = first.trim();
        if !ip.is_empty() {
            return format!("ip:{ip}");
        }
    }
    if let Some(fwd) = req.headers().get("forwarded").and_then(|v| v.to_str().ok()) {
        // Take the first "for=..." parameter.
        for part in fwd.split(';') {
            let part = part.trim();
            if let Some(rest) = part.strip_prefix("for=") {
                let ip = rest.trim_matches(|c: char| c == '"' || c.is_whitespace());
                if !ip.is_empty() {
                    return format!("ip:{ip}");
                }
            }
        }
    }
    "anonymous".to_owned()
}

/// Build a 429 response with the [`crate::error::ApiError`] JSON
/// envelope shape and a `Retry-After` header.
fn rate_limited(retry_after_secs: u64) -> Response<Body> {
    let body = json!({
        "code": "rate_limited",
        "message": "rate limit exceeded",
        "details": serde_json::Value::Null,
    });
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response();
    if let Ok(val) = axum::http::HeaderValue::from_str(&retry_after_secs.to_string()) {
        resp.headers_mut().insert("retry-after", val);
    }
    resp
}
