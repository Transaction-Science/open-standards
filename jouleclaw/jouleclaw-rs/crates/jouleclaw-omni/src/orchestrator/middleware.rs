//! Middleware stack for the orchestrator — auth, rate limiting, metrics.

use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// ============================================================================
// API Key Authentication
// ============================================================================

/// Validate the `Authorization: Bearer <key>` header against allowed API keys.
///
/// Skips auth for:
///   - Internal routes (`/internal/v1/*`) — these use cluster_secret
///   - Health endpoints (`/orchestrator/health`, `/health`)
pub async fn auth_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    let path = req.uri().path();

    // Skip auth for internal/health routes
    if path.starts_with("/internal/") || path == "/orchestrator/health" || path == "/health" {
        return Ok(next.run(req).await);
    }

    // Skip auth for metrics endpoint (Prometheus scraper)
    if path == "/orchestrator/metrics" {
        return Ok(next.run(req).await);
    }

    // Check if auth is configured (API_KEYS env var)
    let api_keys = match std::env::var("API_KEYS") {
        Ok(keys) => keys,
        Err(_) => return Ok(next.run(req).await), // No auth configured = open
    };

    let allowed: Vec<&str> = api_keys.split(',').map(|k| k.trim()).collect();
    if allowed.is_empty() || (allowed.len() == 1 && allowed[0].is_empty()) {
        return Ok(next.run(req).await);
    }

    // Extract Bearer token
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => {
            tracing::warn!(path = %path, "Missing or invalid Authorization header");
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    if !allowed.iter().any(|k| *k == token) {
        tracing::warn!(path = %path, "Invalid API key");
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(next.run(req).await)
}

/// Validate the `X-Cluster-Secret` header for internal routes.
pub async fn cluster_auth_middleware(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let path = req.uri().path();

    // Only apply to internal routes
    if !path.starts_with("/internal/") {
        return Ok(next.run(req).await);
    }

    // Check if cluster secret is configured
    let secret = match std::env::var("CLUSTER_SECRET") {
        Ok(s) if !s.is_empty() => s,
        _ => return Ok(next.run(req).await), // No secret = open
    };

    let header_secret = req
        .headers()
        .get("x-cluster-secret")
        .and_then(|v| v.to_str().ok());

    match header_secret {
        Some(s) if s == secret => Ok(next.run(req).await),
        _ => {
            tracing::warn!(path = %path, "Invalid or missing cluster secret");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

// ============================================================================
// Rate Limiting (Token Bucket)
// ============================================================================

/// Per-client rate limiter state.
struct ClientBucket {
    tokens: f64,
    last_refill: Instant,
}

/// Token-bucket rate limiter keyed by client IP.
pub struct RateLimiter {
    buckets: Mutex<HashMap<String, ClientBucket>>,
    /// Requests per second per client
    rate: f64,
    /// Maximum burst size
    burst: f64,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(requests_per_second: f64, burst: usize) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            rate: requests_per_second,
            burst: burst as f64,
        }
    }

    /// Try to consume one token for the given client. Returns true if allowed.
    pub async fn try_acquire(&self, client_id: &str) -> bool {
        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();

        let bucket = buckets.entry(client_id.to_string()).or_insert(ClientBucket {
            tokens: self.burst,
            last_refill: now,
        });

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.rate).min(self.burst);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Prune stale entries older than the given duration.
    pub async fn prune(&self, max_age: Duration) {
        let mut buckets = self.buckets.lock().await;
        let cutoff = Instant::now() - max_age;
        buckets.retain(|_, b| b.last_refill > cutoff);
    }
}

/// Rate limiting middleware.
pub async fn rate_limit_middleware(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip rate limiting for internal routes
    let path = req.uri().path();
    if path.starts_with("/internal/") || path == "/orchestrator/health" || path == "/orchestrator/metrics" {
        return Ok(next.run(req).await);
    }

    // Get or create global rate limiter
    static LIMITER: std::sync::OnceLock<RateLimiter> = std::sync::OnceLock::new();
    let limiter = LIMITER.get_or_init(|| {
        let rps: f64 = std::env::var("RATE_LIMIT_RPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0);
        let burst: usize = std::env::var("RATE_LIMIT_BURST")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        RateLimiter::new(rps, burst)
    });

    // Extract client identifier (prefer X-Forwarded-For, fall back to peer IP)
    let client_id = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or("unknown").trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    if !limiter.try_acquire(&client_id).await {
        tracing::warn!(client = %client_id, path = %path, "Rate limited");
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    Ok(next.run(req).await)
}

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Per-worker circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    /// Normal operation
    Closed,
    /// Too many failures — rejecting requests
    Open,
    /// Testing if the worker has recovered
    HalfOpen,
}

/// Circuit breaker for proxy requests.
pub struct CircuitBreaker {
    states: Mutex<HashMap<String, CircuitBreakerEntry>>,
    /// Number of failures before opening
    failure_threshold: u32,
    /// How long to stay open before trying half-open
    open_duration: Duration,
}

struct CircuitBreakerEntry {
    state: CircuitState,
    failure_count: u32,
    last_failure: Instant,
    last_state_change: Instant,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(failure_threshold: u32, open_duration: Duration) -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
            failure_threshold,
            open_duration,
        }
    }

    /// Check if requests to a worker should be allowed.
    pub async fn allow_request(&self, worker_id: &str) -> bool {
        let mut states = self.states.lock().await;
        let now = Instant::now();

        let entry = states.entry(worker_id.to_string()).or_insert(CircuitBreakerEntry {
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure: now,
            last_state_change: now,
        });

        match entry.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                if now.duration_since(entry.last_state_change) >= self.open_duration {
                    entry.state = CircuitState::HalfOpen;
                    entry.last_state_change = now;
                    tracing::info!(worker_id = %worker_id, "Circuit breaker half-open");
                    true // Allow one test request
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                false // Only one request at a time in half-open
            }
        }
    }

    /// Record a successful request to a worker.
    pub async fn record_success(&self, worker_id: &str) {
        let mut states = self.states.lock().await;
        if let Some(entry) = states.get_mut(worker_id) {
            if entry.state == CircuitState::HalfOpen {
                tracing::info!(worker_id = %worker_id, "Circuit breaker closed (recovered)");
            }
            entry.state = CircuitState::Closed;
            entry.failure_count = 0;
            entry.last_state_change = Instant::now();
        }
    }

    /// Record a failed request to a worker.
    pub async fn record_failure(&self, worker_id: &str) {
        let mut states = self.states.lock().await;
        let now = Instant::now();

        let entry = states.entry(worker_id.to_string()).or_insert(CircuitBreakerEntry {
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure: now,
            last_state_change: now,
        });

        entry.failure_count += 1;
        entry.last_failure = now;

        if entry.state == CircuitState::HalfOpen {
            // Failed during half-open test → reopen
            entry.state = CircuitState::Open;
            entry.last_state_change = now;
            tracing::warn!(worker_id = %worker_id, "Circuit breaker reopened");
        } else if entry.failure_count >= self.failure_threshold {
            entry.state = CircuitState::Open;
            entry.last_state_change = now;
            tracing::warn!(
                worker_id = %worker_id,
                failures = entry.failure_count,
                "Circuit breaker opened"
            );
        }
    }

    /// Get the state for a worker.
    pub async fn get_state(&self, worker_id: &str) -> CircuitState {
        let states = self.states.lock().await;
        states
            .get(worker_id)
            .map(|e| e.state)
            .unwrap_or(CircuitState::Closed)
    }
}

/// Global circuit breaker instance.
pub fn global_circuit_breaker() -> &'static CircuitBreaker {
    static CB: std::sync::OnceLock<CircuitBreaker> = std::sync::OnceLock::new();
    CB.get_or_init(|| CircuitBreaker::new(5, Duration::from_secs(30)))
}

// ============================================================================
// Prometheus Metrics
// ============================================================================

/// Simple Prometheus-compatible metrics collector.
pub struct Metrics {
    /// Total requests processed
    pub requests_total: AtomicU64,
    /// Requests routed to workers (successful proxy)
    pub requests_proxied: AtomicU64,
    /// Requests rejected (no worker, rate limited, auth failed)
    pub requests_rejected: AtomicU64,
    /// Total request duration (microseconds, for computing average)
    pub request_duration_us_total: AtomicU64,
    /// Active in-flight requests
    pub in_flight: AtomicU64,
    /// Circuit breaker trips
    pub circuit_breaker_trips: AtomicU64,
}

impl Metrics {
    /// Create a new metrics instance.
    pub const fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_proxied: AtomicU64::new(0),
            requests_rejected: AtomicU64::new(0),
            request_duration_us_total: AtomicU64::new(0),
            in_flight: AtomicU64::new(0),
            circuit_breaker_trips: AtomicU64::new(0),
        }
    }

    /// Format metrics in Prometheus text exposition format.
    pub fn to_prometheus(&self, worker_count: usize, healthy_count: usize) -> String {
        let total = self.requests_total.load(Ordering::Relaxed);
        let proxied = self.requests_proxied.load(Ordering::Relaxed);
        let rejected = self.requests_rejected.load(Ordering::Relaxed);
        let duration_us = self.request_duration_us_total.load(Ordering::Relaxed);
        let in_flight = self.in_flight.load(Ordering::Relaxed);
        let cb_trips = self.circuit_breaker_trips.load(Ordering::Relaxed);

        let avg_duration_ms = if total > 0 {
            (duration_us as f64 / total as f64) / 1000.0
        } else {
            0.0
        };

        format!(
            "# HELP create_requests_total Total API requests processed.\n\
             # TYPE create_requests_total counter\n\
             create_requests_total {total}\n\
             # HELP create_requests_proxied Requests successfully proxied to workers.\n\
             # TYPE create_requests_proxied counter\n\
             create_requests_proxied {proxied}\n\
             # HELP create_requests_rejected Requests rejected (auth, rate limit, no worker).\n\
             # TYPE create_requests_rejected counter\n\
             create_requests_rejected {rejected}\n\
             # HELP create_request_duration_avg_ms Average request duration in milliseconds.\n\
             # TYPE create_request_duration_avg_ms gauge\n\
             create_request_duration_avg_ms {avg_duration_ms:.2}\n\
             # HELP create_in_flight_requests Currently in-flight requests.\n\
             # TYPE create_in_flight_requests gauge\n\
             create_in_flight_requests {in_flight}\n\
             # HELP create_circuit_breaker_trips Total circuit breaker trips.\n\
             # TYPE create_circuit_breaker_trips counter\n\
             create_circuit_breaker_trips {cb_trips}\n\
             # HELP create_workers_total Total registered workers.\n\
             # TYPE create_workers_total gauge\n\
             create_workers_total {worker_count}\n\
             # HELP create_workers_healthy Healthy workers.\n\
             # TYPE create_workers_healthy gauge\n\
             create_workers_healthy {healthy_count}\n"
        )
    }
}

/// Global metrics instance.
pub fn global_metrics() -> &'static Metrics {
    static M: Metrics = Metrics::new();
    &M
}

/// Metrics-tracking middleware — increments counters and records latency.
pub async fn metrics_middleware(req: Request, next: Next) -> Response {
    let metrics = global_metrics();
    metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    metrics.in_flight.fetch_add(1, Ordering::Relaxed);

    let start = Instant::now();
    let mut response = next.run(req).await;
    let elapsed_us = start.elapsed().as_micros() as u64;

    metrics.request_duration_us_total.fetch_add(elapsed_us, Ordering::Relaxed);
    metrics.in_flight.fetch_sub(1, Ordering::Relaxed);

    let status = response.status().as_u16();
    if status >= 200 && status < 400 {
        metrics.requests_proxied.fetch_add(1, Ordering::Relaxed);
    } else if status >= 400 {
        metrics.requests_rejected.fetch_add(1, Ordering::Relaxed);
    }

    // Add server timing header
    if let Ok(timing) = HeaderValue::from_str(&format!("proxy;dur={:.1}", elapsed_us as f64 / 1000.0)) {
        response.headers_mut().insert("server-timing", timing);
    }

    response
}
