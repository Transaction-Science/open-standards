//! Webhook endpoint management.
//!
//! Create / get / disable / re-enable webhook endpoints. Event
//! delivery happens at the emit-site in other handlers; this
//! module only handles the operator-side registration surface.

use axum::Json;
use axum::extract::{Path, State};
use op_webhook::{Endpoint, EndpointId, EndpointStatus, WebhookStore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct CreateEndpointRequest {
    pub url: String,
    /// Shared HMAC secret. Operators rotate out-of-band.
    pub secret: String,
    /// Exact-match event types this endpoint subscribes to.
    /// `["*"]` receives everything.
    pub event_filters: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct EndpointResponse {
    pub id: Uuid,
    pub url: String,
    pub event_filters: Vec<String>,
    pub status: String,
    pub consecutive_failures: u32,
}

fn status_code(s: EndpointStatus) -> &'static str {
    match s {
        EndpointStatus::Active => "active",
        EndpointStatus::Disabled => "disabled",
        EndpointStatus::AutoDisabled => "auto_disabled",
    }
}

fn to_response(ep: &Endpoint) -> EndpointResponse {
    EndpointResponse {
        id: ep.id.0,
        url: ep.url.clone(),
        event_filters: ep.event_filters.clone(),
        status: status_code(ep.status).to_owned(),
        consecutive_failures: ep.consecutive_failures,
    }
}

fn webhook_to_api(e: &op_webhook::Error) -> ApiError {
    use op_webhook::Error::{
        AttemptNotFound, EndpointDisabled, EndpointNotFound, EventNotFound, InvalidInput,
        InvalidUrl,
    };
    match e {
        EndpointNotFound(_) | EventNotFound(_) | AttemptNotFound(_) => {
            ApiError::NotFound(e.to_string())
        }
        InvalidUrl(_) | InvalidInput(_) => ApiError::BadRequest(e.to_string()),
        EndpointDisabled(_) => ApiError::Conflict(e.to_string()),
        _ => ApiError::Internal(e.to_string()),
    }
}

/// `POST /v1/webhooks/endpoints`
pub async fn create(
    State(state): State<AppState>,
    Json(req): Json<CreateEndpointRequest>,
) -> ApiResult<Json<EndpointResponse>> {
    let ep = Endpoint::new(req.url, req.secret.into_bytes(), req.event_filters)
        .map_err(|e| webhook_to_api(&e))?;
    let id = state
        .webhooks
        .put_endpoint(ep)
        .map_err(|e| webhook_to_api(&e))?;
    let stored = state
        .webhooks
        .get_endpoint(id)
        .map_err(|e| webhook_to_api(&e))?;
    Ok(Json(to_response(&stored)))
}

/// `GET /v1/webhooks/endpoints/{id}`
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<EndpointResponse>> {
    let ep = state
        .webhooks
        .get_endpoint(EndpointId(id))
        .map_err(|e| webhook_to_api(&e))?;
    Ok(Json(to_response(&ep)))
}

/// `POST /v1/webhooks/endpoints/{id}/disable`
pub async fn disable(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<EndpointResponse>> {
    state
        .webhooks
        .set_endpoint_status(EndpointId(id), EndpointStatus::Disabled)
        .map_err(|e| webhook_to_api(&e))?;
    let ep = state
        .webhooks
        .get_endpoint(EndpointId(id))
        .map_err(|e| webhook_to_api(&e))?;
    Ok(Json(to_response(&ep)))
}

/// `POST /v1/webhooks/endpoints/{id}/enable`
pub async fn enable(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<EndpointResponse>> {
    state
        .webhooks
        .set_endpoint_status(EndpointId(id), EndpointStatus::Active)
        .map_err(|e| webhook_to_api(&e))?;
    state
        .webhooks
        .set_endpoint_consecutive_failures(EndpointId(id), 0)
        .map_err(|e| webhook_to_api(&e))?;
    let ep = state
        .webhooks
        .get_endpoint(EndpointId(id))
        .map_err(|e| webhook_to_api(&e))?;
    Ok(Json(to_response(&ep)))
}
