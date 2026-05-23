//! Settlement batch handlers.

use axum::Json;
use axum::extract::{Path, State};
use op_core::{Currency, Money};
use op_ledger::TransactionId;
use op_settlement::{Batch, BatchId, HoldbackPolicy, PayoutRail, SettlementStore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::events::emit;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct OpenBatchRequest {
    pub currency: String,
    pub rail: String,
    pub external_id: Option<String>,
    pub opened_at_unix_secs: u64,
}

#[derive(Debug, Deserialize)]
pub struct AddEntryRequest {
    pub tx_id: Uuid,
    pub amount_minor: i64,
    pub reference: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CloseBatchRequest {
    pub flat_rate_bps: u16,
    pub max_total_bps: u16,
    pub dispute_adjustment_bps: u16,
    pub closed_at_unix_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub id: Uuid,
    pub currency: String,
    pub rail: String,
    pub external_id: Option<String>,
    pub status: String,
    pub entry_count: usize,
    pub gross_minor: i64,
    pub reserve_minor: Option<i64>,
    pub net_minor: Option<i64>,
    pub opened_at_unix_secs: u64,
}

fn parse_rail(s: &str) -> ApiResult<PayoutRail> {
    Ok(match s {
        "ach_nacha" => PayoutRail::AchNacha,
        "sepa_ct" => PayoutRail::SepaCt,
        "fednow" => PayoutRail::FedNow,
        "rtp" => PayoutRail::Rtp,
        "wire" => PayoutRail::Wire,
        "internal_book_transfer" => PayoutRail::InternalBookTransfer,
        bad => {
            return Err(ApiError::BadRequest(format!("unknown payout rail `{bad}`")));
        }
    })
}

fn rail_to_string(r: PayoutRail) -> &'static str {
    match r {
        PayoutRail::AchNacha => "ach_nacha",
        PayoutRail::SepaCt => "sepa_ct",
        PayoutRail::FedNow => "fednow",
        PayoutRail::Rtp => "rtp",
        PayoutRail::Wire => "wire",
        PayoutRail::InternalBookTransfer => "internal_book_transfer",
    }
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

fn batch_to_response(b: &Batch) -> BatchResponse {
    let gross = b.gross().map_or(0, |m| m.minor_units);
    let reserve = b.holdback.as_ref().map(|h| h.reserve.minor_units);
    let net = b
        .holdback
        .as_ref()
        .and_then(|h| h.net().ok().map(|m| m.minor_units));
    BatchResponse {
        id: b.id.as_uuid(),
        currency: b.currency.code().to_owned(),
        rail: rail_to_string(b.rail).to_owned(),
        external_id: b.external_id.clone(),
        status: b.status.code().to_owned(),
        entry_count: b.entries.len(),
        gross_minor: gross,
        reserve_minor: reserve,
        net_minor: net,
        opened_at_unix_secs: b.opened_at_unix_secs,
    }
}

/// `POST /v1/settlement/batches`
pub async fn open(
    State(state): State<AppState>,
    Json(req): Json<OpenBatchRequest>,
) -> ApiResult<Json<BatchResponse>> {
    let currency = currency_from_code(&req.currency)?;
    let rail = parse_rail(&req.rail)?;
    let mut batch = Batch::open(currency, rail, req.opened_at_unix_secs);
    if let Some(ext) = &req.external_id {
        batch = batch.with_external_id(ext);
    }
    let id = state.settlement.create_batch(batch.clone())?;
    let stored = state.settlement.get_batch(id)?;
    let response = batch_to_response(&stored);
    emit(&state, "settlement.batch_opened", &response);
    Ok(Json(response))
}

/// `GET /v1/settlement/batches/{id}`
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<BatchResponse>> {
    let b = state.settlement.get_batch(BatchId::from_uuid(id))?;
    Ok(Json(batch_to_response(&b)))
}

/// `POST /v1/settlement/batches/{id}/entries`
pub async fn add_entry(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<AddEntryRequest>,
) -> ApiResult<Json<BatchResponse>> {
    let b = state.settlement.update(BatchId::from_uuid(id), |b| {
        let amount = Money::from_minor(req.amount_minor, b.currency);
        b.add_entry(
            TransactionId::from_uuid(req.tx_id),
            amount,
            req.reference.clone(),
        )
    })?;
    Ok(Json(batch_to_response(&b)))
}

/// `POST /v1/settlement/batches/{id}:close`
pub async fn close(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<CloseBatchRequest>,
) -> ApiResult<Json<BatchResponse>> {
    let policy = HoldbackPolicy::flat(req.flat_rate_bps).with_ceiling(req.max_total_bps);
    let b = state.settlement.update(BatchId::from_uuid(id), |b| {
        let gross = b.gross()?;
        let hb = policy.compute(gross, req.dispute_adjustment_bps)?;
        b.close(hb, req.closed_at_unix_secs)
    })?;
    let response = batch_to_response(&b);
    emit(&state, "settlement.batch_closed", &response);
    Ok(Json(response))
}
