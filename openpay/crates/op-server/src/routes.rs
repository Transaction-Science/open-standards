//! Route table.

use axum::Router;
use axum::routing::{get, post};

use crate::auth::ApiKeyAuthLayer;
use crate::handlers;
use crate::rate_limit::RateLimitLayer;
use crate::state::AppState;

/// Build the full HTTP router for the `OpenPay` server. Embedders
/// can nest this under a prefix, layer middleware on it, or run it
/// as-is in the binary.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health::health))
        .route("/readiness", get(handlers::health::readiness))
        // Intents
        .route("/v1/intents", post(handlers::intent::create))
        .route("/v1/intents/resume", post(handlers::intent::resume))
        // Refunds
        .route("/v1/refunds", post(handlers::refund::create))
        .route("/v1/refunds/{id}", get(handlers::refund::get))
        .route("/v1/refunds/{id}/submit", post(handlers::refund::submit))
        .route("/v1/refunds/{id}/approve", post(handlers::refund::approve))
        .route("/v1/refunds/{id}/settle", post(handlers::refund::settle))
        // Disputes
        .route("/v1/disputes", post(handlers::dispute::create))
        .route("/v1/disputes/{id}", get(handlers::dispute::get))
        .route(
            "/v1/disputes/{id}/evidence",
            post(handlers::dispute::attach_evidence),
        )
        // Settlement
        .route("/v1/settlement/batches", post(handlers::settlement::open))
        .route(
            "/v1/settlement/batches/{id}",
            get(handlers::settlement::get),
        )
        .route(
            "/v1/settlement/batches/{id}/entries",
            post(handlers::settlement::add_entry),
        )
        .route(
            "/v1/settlement/batches/{id}/close",
            post(handlers::settlement::close),
        )
        // Audit
        .route("/v1/audit/report", get(handlers::audit::report))
        // Subscriptions
        .route("/v1/subscriptions", post(handlers::subscription::create))
        .route("/v1/subscriptions", get(handlers::subscription::list))
        .route("/v1/subscriptions/{id}", get(handlers::subscription::get))
        .route(
            "/v1/subscriptions/{id}/cancel",
            post(handlers::subscription::cancel),
        )
        .route(
            "/v1/subscriptions/{id}/pause",
            post(handlers::subscription::pause),
        )
        .route(
            "/v1/subscriptions/{id}/resume",
            post(handlers::subscription::resume),
        )
        // FX
        .route("/v1/fx/quote", get(handlers::fx::quote))
        .route("/v1/fx/convert", post(handlers::fx::convert_handler))
        // Webhook endpoint management
        .route("/v1/webhooks/endpoints", post(handlers::webhook::create))
        .route("/v1/webhooks/endpoints/{id}", get(handlers::webhook::get))
        .route(
            "/v1/webhooks/endpoints/{id}/disable",
            post(handlers::webhook::disable),
        )
        .route(
            "/v1/webhooks/endpoints/{id}/enable",
            post(handlers::webhook::enable),
        )
        .with_state(state)
}

/// Like [`router`] but wraps the route stack with the supplied API-key
/// auth and rate-limit layers. Auth runs first so unauthenticated
/// traffic never consumes a per-key bucket entry. Use this when
/// binding to a public IP.
///
/// ```ignore
/// use std::collections::HashSet;
/// use op_server::{router_with_middleware, AppState, ApiKeyAuthLayer, RateLimitLayer};
///
/// let mut keys = HashSet::new();
/// keys.insert("supersecret".to_owned());
/// let auth = ApiKeyAuthLayer::new(keys)
///     .with_bypass_paths(vec!["/health".into(), "/readiness".into()]);
/// let limit = RateLimitLayer::per_minute(600);
/// let app = router_with_middleware(AppState::new_in_memory(), auth, limit);
/// ```
pub fn router_with_middleware(
    state: AppState,
    auth: ApiKeyAuthLayer,
    rate_limit: RateLimitLayer,
) -> Router {
    // `.layer(rate_limit).layer(auth)` makes auth the outermost layer
    // — meaning auth runs first on the way in.
    router(state).layer(rate_limit).layer(auth)
}
