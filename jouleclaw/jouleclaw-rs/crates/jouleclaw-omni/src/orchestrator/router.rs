//! Request router — proxies API requests to the best available worker.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;
use std::sync::Arc;

use super::middleware::global_circuit_breaker;
use super::types::PipelineType;

/// Shared state for the orchestrator server.
#[derive(Clone)]
pub struct OrchestratorState {
    /// Worker registry
    pub registry: Arc<super::registry::WorkerRegistry>,
    /// HTTP client for proxying requests
    pub client: reqwest::Client,
    /// Orchestrator configuration
    pub config: Arc<super::types::OrchestratorConfig>,
}

/// Proxy an incoming API request to the best available worker.
///
/// This is the main catch-all handler. It:
/// 1. Determines the pipeline type from the request path
/// 2. Finds the best worker via the registry
/// 3. Checks the circuit breaker for the worker
/// 4. Forwards the request (including body and headers)
/// 5. Streams the response back (supporting SSE)
/// 6. Records success/failure with the circuit breaker
pub async fn proxy_request(
    State(state): State<OrchestratorState>,
    req: Request,
) -> Result<Response, StatusCode> {
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    let method = req.method().clone();
    let pipeline_type = PipelineType::from_path(&path);

    // Find the best worker for this request
    let (worker_id, worker_endpoint) = state
        .registry
        .find_best_worker(pipeline_type)
        .ok_or_else(|| {
            tracing::warn!(
                path = %path,
                pipeline = pipeline_type.capability_str(),
                "No healthy worker available"
            );
            StatusCode::SERVICE_UNAVAILABLE
        })?;

    // Check circuit breaker
    let cb = global_circuit_breaker();
    if !cb.allow_request(&worker_id).await {
        tracing::warn!(
            worker_id = %worker_id,
            path = %path,
            "Request blocked by circuit breaker"
        );
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let target_url = format!("{worker_endpoint}{path}{query}");
    tracing::debug!(
        worker_id = %worker_id,
        target = %target_url,
        method = %method,
        "Routing request"
    );

    // Extract headers we want to forward
    let content_type = req
        .headers()
        .get("content-type")
        .cloned();
    let accept = req.headers().get("accept").cloned();
    let authorization = req.headers().get("authorization").cloned();

    // Read the request body
    let body_bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Build the proxied request
    let mut builder = state.client.request(
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET),
        &target_url,
    );

    if let Some(ct) = content_type {
        builder = builder.header("content-type", ct.to_str().unwrap_or("application/json"));
    }
    if let Some(acc) = accept {
        builder = builder.header("accept", acc.to_str().unwrap_or("*/*"));
    }
    if let Some(auth) = authorization {
        builder = builder.header("authorization", auth.to_str().unwrap_or(""));
    }

    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes);
    }

    // Send request to worker
    let worker_resp = match builder.send().await {
        Ok(resp) => {
            cb.record_success(&worker_id).await;
            resp
        }
        Err(e) => {
            tracing::error!(
                worker_id = %worker_id,
                error = %e,
                "Failed to proxy request to worker"
            );
            cb.record_failure(&worker_id).await;
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    // Build the response back to the client
    let status = StatusCode::from_u16(worker_resp.status().as_u16()).unwrap_or(StatusCode::OK);

    let mut response_builder = Response::builder().status(status);

    // Forward key response headers
    for (name, value) in worker_resp.headers() {
        if let Ok(header_name) = axum::http::HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(header_value) = HeaderValue::from_bytes(value.as_bytes()) {
                response_builder = response_builder.header(header_name, header_value);
            }
        }
    }

    // Add X-Worker-Id header so the client knows which worker handled the request
    if let Ok(hv) = HeaderValue::from_str(&worker_id) {
        response_builder = response_builder.header("X-Worker-Id", hv);
    }

    // Stream the response body (important for SSE endpoints)
    let body_stream = worker_resp.bytes_stream();
    let body = Body::from_stream(body_stream);

    response_builder.body(body).map_err(|e| {
        tracing::error!(error = %e, "Failed to build proxy response");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}
