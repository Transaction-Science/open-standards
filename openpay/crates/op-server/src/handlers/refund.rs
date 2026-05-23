//! Refund handlers: create, get, transition.

use axum::Json;
use axum::extract::{Path, State};
use op_core::{Currency, Money};
use op_ledger::TransactionId;
use op_refund::{Refund, RefundId, RefundReason, RefundStore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::events::emit;
use crate::state::AppState;

/// POST `/v1/refunds` request body.
#[derive(Debug, Deserialize)]
pub struct CreateRefundRequest {
    /// Original ledger transaction id (UUID).
    pub original_tx_id: Uuid,
    /// Amount in minor units (cents for USD).
    pub amount_minor: i64,
    /// ISO 4217 currency code.
    pub currency: String,
    /// Normalized refund reason (`customer_request`, `duplicate_charge`,
    /// `fraudulent_charge`, `merchant_initiated`, `dispute_resolution`,
    /// `other:<freeform>`).
    pub reason: String,
    /// Caller-supplied idempotency key.
    pub external_id: Option<String>,
    /// Unix epoch seconds when the request was made (caller clock —
    /// keeps the server deterministic for replay / tests).
    pub requested_at_unix_secs: u64,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Vec<(String, String)>,
}

/// POST `/v1/refunds/{id}:submit` body.
#[derive(Debug, Deserialize)]
pub struct SubmitRefundRequest {
    /// PSP-side identifier for the refund operation.
    pub psp_refund_id: String,
}

/// POST `/v1/refunds/{id}:settle` body.
#[derive(Debug, Deserialize)]
pub struct SettleRefundRequest {
    /// Settlement unix epoch seconds (caller clock).
    pub settled_at_unix_secs: u64,
}

/// Response envelope for refund objects.
#[derive(Debug, Serialize)]
pub struct RefundResponse {
    /// Id.
    pub id: Uuid,
    /// Original ledger tx id.
    pub original_tx_id: Uuid,
    /// External id (idempotency key) if set.
    pub external_id: Option<String>,
    /// Amount.
    pub amount_minor: i64,
    /// Currency.
    pub currency: String,
    /// Reason code.
    pub reason: String,
    /// Short status code (`requested`/`submitted`/`approved`/...).
    pub status: String,
    /// Unix epoch seconds the refund was created.
    pub requested_at_unix_secs: u64,
}

impl From<&Refund> for RefundResponse {
    fn from(r: &Refund) -> Self {
        Self {
            id: r.id.as_uuid(),
            original_tx_id: r.original_tx_id.as_uuid(),
            external_id: r.external_id.clone(),
            amount_minor: r.amount.minor_units,
            currency: r.amount.currency.code().to_owned(),
            reason: reason_to_code(&r.reason),
            status: r.status.code().to_owned(),
            requested_at_unix_secs: r.requested_at_unix_secs,
        }
    }
}

fn reason_to_code(r: &RefundReason) -> String {
    match r {
        RefundReason::CustomerRequest => "customer_request".into(),
        RefundReason::DuplicateCharge => "duplicate_charge".into(),
        RefundReason::FraudulentCharge => "fraudulent_charge".into(),
        RefundReason::MerchantInitiated => "merchant_initiated".into(),
        RefundReason::DisputeResolution => "dispute_resolution".into(),
        RefundReason::Other(s) => format!("other:{s}"),
    }
}

fn code_to_reason(s: &str) -> ApiResult<RefundReason> {
    Ok(match s {
        "customer_request" => RefundReason::CustomerRequest,
        "duplicate_charge" => RefundReason::DuplicateCharge,
        "fraudulent_charge" => RefundReason::FraudulentCharge,
        "merchant_initiated" => RefundReason::MerchantInitiated,
        "dispute_resolution" => RefundReason::DisputeResolution,
        other if other.starts_with("other:") => {
            RefundReason::Other(other.trim_start_matches("other:").to_owned())
        }
        bad => return Err(ApiError::BadRequest(format!("unknown reason `{bad}`"))),
    })
}

fn currency_from_code(code: &str) -> ApiResult<Currency> {
    let mut bytes = [b'?'; 3];
    if code.len() != 3 || !code.chars().all(|c| c.is_ascii_uppercase()) {
        return Err(ApiError::BadRequest(format!(
            "currency `{code}` must be 3 ASCII uppercase letters"
        )));
    }
    for (i, b) in code.bytes().enumerate() {
        bytes[i] = b;
    }
    Currency::try_new(bytes, 2).map_err(|e| ApiError::BadRequest(format!("invalid currency: {e}")))
}

/// `POST /v1/refunds`
pub async fn create(
    State(state): State<AppState>,
    Json(req): Json<CreateRefundRequest>,
) -> ApiResult<Json<RefundResponse>> {
    let reason = code_to_reason(&req.reason)?;
    let currency = currency_from_code(&req.currency)?;
    let amount = Money::from_minor(req.amount_minor, currency);
    let mut refund = Refund::new(
        TransactionId::from_uuid(req.original_tx_id),
        amount,
        reason,
        req.requested_at_unix_secs,
    )
    .map_err(ApiError::from)?;
    if let Some(ext) = &req.external_id {
        refund = refund.with_external_id(ext);
    }
    for (k, v) in &req.metadata {
        refund = refund.with_metadata(k, v);
    }
    let id = state.refunds.create_refund(refund.clone())?;
    let stored = state.refunds.get_refund(id)?;
    let response = RefundResponse::from(&stored);
    emit(&state, "refund.created", &response);
    Ok(Json(response))
}

/// `GET /v1/refunds/{id}`
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<RefundResponse>> {
    let refund = state.refunds.get_refund(RefundId::from_uuid(id))?;
    Ok(Json(RefundResponse::from(&refund)))
}

/// `POST /v1/refunds/{id}:submit`
pub async fn submit(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<SubmitRefundRequest>,
) -> ApiResult<Json<RefundResponse>> {
    let refund = state.refunds.update(RefundId::from_uuid(id), |r| {
        r.submit(req.psp_refund_id.clone())
    })?;
    let response = RefundResponse::from(&refund);
    emit(&state, "refund.submitted", &response);
    Ok(Json(response))
}

/// `POST /v1/refunds/{id}:approve`
pub async fn approve(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<RefundResponse>> {
    let refund = state
        .refunds
        .update(RefundId::from_uuid(id), op_refund::Refund::approve)?;
    let response = RefundResponse::from(&refund);
    emit(&state, "refund.approved", &response);
    Ok(Json(response))
}

/// `POST /v1/refunds/{id}:settle`
pub async fn settle(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<SettleRefundRequest>,
) -> ApiResult<Json<RefundResponse>> {
    let refund = state.refunds.update(RefundId::from_uuid(id), |r| {
        r.settle(req.settled_at_unix_secs)
    })?;
    let response = RefundResponse::from(&refund);
    emit(&state, "refund.settled", &response);
    Ok(Json(response))
}
