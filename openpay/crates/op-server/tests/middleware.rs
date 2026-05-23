//! Tests for [`op_server::ApiKeyAuthLayer`] and [`op_server::RateLimitLayer`].
//!
//! Same `tower::ServiceExt::oneshot` strategy as `api.rs` — no socket,
//! no listener, just direct router invocation through the middleware
//! stack. The rate-limit tests inject a mock clock so refill behavior
//! is observed without sleeping.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use op_server::{ApiKeyAuthLayer, AppState, RateLimitLayer, router_with_middleware};
use serde_json::Value;
use tower::util::ServiceExt;

async fn body_to_value(body: Body) -> Value {
    let bytes = body.collect().await.expect("collect body").to_bytes();
    if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("body is json")
    }
}

fn req(method: &str, path: &str, headers: &[(&str, &str)]) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(path);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    b.body(Body::empty()).unwrap()
}

fn allow_keys(keys: &[&str]) -> ApiKeyAuthLayer {
    let set: HashSet<String> = keys.iter().map(|s| (*s).to_owned()).collect();
    ApiKeyAuthLayer::new(set).with_bypass_paths(vec!["/health".into(), "/readiness".into()])
}

/// Convenience: a permissive rate limiter (high capacity) — used by
/// auth tests so the rate limiter never trips by accident.
fn permissive_limiter() -> RateLimitLayer {
    RateLimitLayer::new(10_000, 10_000.0)
}

// ====================================================================
// API-key auth
// ====================================================================

#[tokio::test]
async fn auth_missing_header_returns_401() {
    let app = router_with_middleware(
        AppState::new_in_memory(),
        allow_keys(&["sk-good"]),
        permissive_limiter(),
    );
    let res = app
        .oneshot(req(
            "GET",
            "/v1/audit/report?start_tx=0&end_tx=1&generated_at_unix_secs=0",
            &[],
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let body = body_to_value(res.into_body()).await;
    assert_eq!(body["code"], "unauthorized");
    assert!(body["details"].is_null());
}

#[tokio::test]
async fn auth_wrong_key_returns_401() {
    let app = router_with_middleware(
        AppState::new_in_memory(),
        allow_keys(&["sk-good"]),
        permissive_limiter(),
    );
    let res = app
        .oneshot(req(
            "GET",
            "/v1/audit/report?start_tx=0&end_tx=1&generated_at_unix_secs=0",
            &[("authorization", "Bearer sk-bad")],
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_valid_key_passes_through() {
    let app = router_with_middleware(
        AppState::new_in_memory(),
        allow_keys(&["sk-good"]),
        permissive_limiter(),
    );
    let res = app
        .oneshot(req(
            "GET",
            "/v1/audit/report?start_tx=0&end_tx=1&generated_at_unix_secs=0",
            &[("authorization", "Bearer sk-good")],
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_bypass_health_endpoint() {
    let app = router_with_middleware(
        AppState::new_in_memory(),
        allow_keys(&["sk-good"]),
        permissive_limiter(),
    );
    let res = app
        .clone()
        .oneshot(req("GET", "/health", &[]))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let res = app.oneshot(req("GET", "/readiness", &[])).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_custom_header_no_bearer_stripping() {
    let auth = ApiKeyAuthLayer::new({
        let mut s = HashSet::new();
        s.insert("raw-token".to_owned());
        s
    })
    .with_header("x-api-key")
    .with_bypass_paths(vec!["/health".into()]);
    let app = router_with_middleware(AppState::new_in_memory(), auth, permissive_limiter());

    // Missing custom header → 401.
    let res = app
        .clone()
        .oneshot(req(
            "GET",
            "/v1/audit/report?start_tx=0&end_tx=1&generated_at_unix_secs=0",
            &[],
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Present custom header, raw (no Bearer prefix) → 200.
    let res = app
        .oneshot(req(
            "GET",
            "/v1/audit/report?start_tx=0&end_tx=1&generated_at_unix_secs=0",
            &[("x-api-key", "raw-token")],
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

// ====================================================================
// Rate limiting
// ====================================================================

#[tokio::test]
async fn rate_limit_blocks_after_capacity() {
    // capacity=3, refill very slow so the bucket stays empty during
    // the burst.
    let limiter = RateLimitLayer::new(3, 0.01).with_clock(|| 1_000);
    let app = router_with_middleware(AppState::new_in_memory(), allow_keys(&["sk"]), limiter);

    let headers = [("authorization", "Bearer sk")];
    let mut statuses = Vec::new();
    let mut retry_after_headers = Vec::new();
    for _ in 0..5 {
        let res = app
            .clone()
            .oneshot(req("GET", "/health", &headers))
            .await
            .unwrap();
        statuses.push(res.status());
        retry_after_headers.push(
            res.headers()
                .get("retry-after")
                .map(|v| v.to_str().unwrap().to_owned()),
        );
    }

    // First 3 succeed, next 2 are rate limited.
    assert_eq!(statuses[0], StatusCode::OK);
    assert_eq!(statuses[1], StatusCode::OK);
    assert_eq!(statuses[2], StatusCode::OK);
    assert_eq!(statuses[3], StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(statuses[4], StatusCode::TOO_MANY_REQUESTS);

    // The 429 responses must carry a Retry-After header.
    assert!(retry_after_headers[3].is_some());
    assert!(retry_after_headers[4].is_some());
    let retry_secs: u64 = retry_after_headers[3]
        .as_ref()
        .unwrap()
        .parse()
        .expect("Retry-After is a number");
    assert!(retry_secs >= 1);
}

#[tokio::test]
async fn rate_limit_refills_over_time() {
    // capacity=2, refill 1 token/sec. Mock clock starts at 1000 and we
    // advance it manually.
    let clock = Arc::new(AtomicU64::new(1_000));
    let clock_for_layer = Arc::clone(&clock);
    let limiter =
        RateLimitLayer::new(2, 1.0).with_clock(move || clock_for_layer.load(Ordering::SeqCst));
    let app = router_with_middleware(AppState::new_in_memory(), allow_keys(&["sk"]), limiter);

    let headers = [("authorization", "Bearer sk")];

    // Drain bucket: 2 OK, then 429.
    for _ in 0..2 {
        let res = app
            .clone()
            .oneshot(req("GET", "/health", &headers))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
    let res = app
        .clone()
        .oneshot(req("GET", "/health", &headers))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);

    // Advance clock by 2s → bucket refilled to 2 tokens.
    clock.fetch_add(2, Ordering::SeqCst);

    let res = app
        .clone()
        .oneshot(req("GET", "/health", &headers))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = app
        .clone()
        .oneshot(req("GET", "/health", &headers))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = app.oneshot(req("GET", "/health", &headers)).await.unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn rate_limit_buckets_are_per_source_key() {
    // capacity=1, slow refill so buckets only see one request.
    let limiter = RateLimitLayer::new(1, 0.01).with_clock(|| 1_000);
    let app = router_with_middleware(
        AppState::new_in_memory(),
        allow_keys(&["sk-a", "sk-b"]),
        limiter,
    );

    // First key burns its bucket.
    let res = app
        .clone()
        .oneshot(req("GET", "/health", &[("authorization", "Bearer sk-a")]))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = app
        .clone()
        .oneshot(req("GET", "/health", &[("authorization", "Bearer sk-a")]))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);

    // Second key has its own untouched bucket.
    let res = app
        .oneshot(req("GET", "/health", &[("authorization", "Bearer sk-b")]))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}
