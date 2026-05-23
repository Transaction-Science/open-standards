//! Dispute handlers.

use axum::Json;
use axum::extract::{Path, State};
use op_core::{Currency, Money};
use op_dispute::{Dispute, DisputeId, DisputeReason, DisputeStore, EvidenceRef};
use op_ledger::TransactionId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::events::emit;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct CreateDisputeRequest {
    pub original_tx_id: Uuid,
    pub amount_minor: i64,
    pub currency: String,
    pub reason: String,
    pub network_reason_code: Option<String>,
    pub external_id: Option<String>,
    pub opened_at_unix_secs: u64,
    pub due_by_unix_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct AttachEvidenceRequest {
    pub kind: String,
    pub url: String,
    pub note: Option<String>,
    pub attached_at_unix_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct DisputeResponse {
    pub id: Uuid,
    pub original_tx_id: Uuid,
    pub external_id: Option<String>,
    pub amount_minor: i64,
    pub currency: String,
    pub reason: String,
    pub network_reason_code: Option<String>,
    pub status: String,
    pub opened_at_unix_secs: u64,
    pub evidence_count: usize,
}

impl From<&Dispute> for DisputeResponse {
    fn from(d: &Dispute) -> Self {
        Self {
            id: d.id.as_uuid(),
            original_tx_id: d.original_tx_id.as_uuid(),
            external_id: d.external_id.clone(),
            amount_minor: d.amount.minor_units,
            currency: d.amount.currency.code().to_owned(),
            reason: reason_to_code(&d.reason),
            network_reason_code: d.network_reason_code.clone(),
            status: d.status.code().to_owned(),
            opened_at_unix_secs: d.opened_at_unix_secs,
            evidence_count: d.evidence.len(),
        }
    }
}

fn reason_to_code(r: &DisputeReason) -> String {
    match r {
        DisputeReason::Fraudulent => "fraudulent".into(),
        DisputeReason::ProductNotReceived => "product_not_received".into(),
        DisputeReason::NotAsDescribed => "not_as_described".into(),
        DisputeReason::Duplicate => "duplicate".into(),
        DisputeReason::CancelledSubscription => "cancelled_subscription".into(),
        DisputeReason::CreditNotProcessed => "credit_not_processed".into(),
        DisputeReason::AuthorizationIssue => "authorization_issue".into(),
        DisputeReason::ProcessingError => "processing_error".into(),
        DisputeReason::Other(s) => format!("other:{s}"),
    }
}

fn code_to_reason(s: &str) -> ApiResult<DisputeReason> {
    Ok(match s {
        "fraudulent" => DisputeReason::Fraudulent,
        "product_not_received" => DisputeReason::ProductNotReceived,
        "not_as_described" => DisputeReason::NotAsDescribed,
        "duplicate" => DisputeReason::Duplicate,
        "cancelled_subscription" => DisputeReason::CancelledSubscription,
        "credit_not_processed" => DisputeReason::CreditNotProcessed,
        "authorization_issue" => DisputeReason::AuthorizationIssue,
        "processing_error" => DisputeReason::ProcessingError,
        other if other.starts_with("other:") => {
            DisputeReason::Other(other.trim_start_matches("other:").to_owned())
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

/// `POST /v1/disputes`
pub async fn create(
    State(state): State<AppState>,
    Json(req): Json<CreateDisputeRequest>,
) -> ApiResult<Json<DisputeResponse>> {
    let reason = code_to_reason(&req.reason)?;
    let currency = currency_from_code(&req.currency)?;
    let amount = Money::from_minor(req.amount_minor, currency);
    let mut d = Dispute::new(
        TransactionId::from_uuid(req.original_tx_id),
        amount,
        reason,
        req.opened_at_unix_secs,
    )
    .map_err(ApiError::from)?;
    if let Some(ext) = &req.external_id {
        d = d.with_external_id(ext);
    }
    if let Some(code) = &req.network_reason_code {
        d = d.with_network_reason_code(code);
    }
    if let Some(due) = req.due_by_unix_secs {
        d = d.with_due_by(due);
    }
    let id = state.disputes.create_dispute(d.clone())?;
    let stored = state.disputes.get_dispute(id)?;
    let response = DisputeResponse::from(&stored);
    emit(&state, "dispute.created", &response);
    Ok(Json(response))
}

/// `GET /v1/disputes/{id}`
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<DisputeResponse>> {
    let d = state.disputes.get_dispute(DisputeId::from_uuid(id))?;
    Ok(Json(DisputeResponse::from(&d)))
}

/// `POST /v1/disputes/{id}/evidence`
pub async fn attach_evidence(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<AttachEvidenceRequest>,
) -> ApiResult<Json<DisputeResponse>> {
    let ev = EvidenceRef {
        kind: req.kind,
        url: req.url,
        note: req.note,
        attached_at_unix_secs: req.attached_at_unix_secs,
    };
    let d = state
        .disputes
        .update(DisputeId::from_uuid(id), |d| d.attach_evidence(ev.clone()))?;
    let response = DisputeResponse::from(&d);
    emit(&state, "dispute.evidence_attached", &response);
    Ok(Json(response))
}
