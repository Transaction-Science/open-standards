//! Subscription handlers — create, get, cancel, pause, resume,
//! list per customer.

use axum::Json;
use axum::extract::{Path, Query, State};
use op_core::{Currency, Money, PaymentMethod, VaultRef};
use op_subscriptions::{Interval, Plan, Status, Subscription, SubscriptionId, SubscriptionStore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::events::emit;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct CreateSubscriptionRequest {
    pub customer_ref: String,
    pub plan_name: String,
    pub amount_minor: i64,
    pub currency: String,
    pub interval: String, // "day" / "week" / "month" / "year"
    pub interval_count: u32,
    pub trial_days: Option<u32>,
    pub method: MethodPayload,
    pub external_id: Option<String>,
    pub now_unix_secs: u64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MethodPayload {
    Vault { token: String },
}

#[derive(Debug, Deserialize)]
pub struct CancelRequest {
    /// `true` = end-of-period (default), `false` = immediate.
    pub at_period_end: Option<bool>,
    pub now_unix_secs: u64,
}

#[derive(Debug, Deserialize)]
pub struct PauseRequest {
    pub now_unix_secs: u64,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub customer_ref: String,
}

#[derive(Debug, Serialize)]
pub struct SubscriptionResponse {
    pub id: Uuid,
    pub external_id: Option<String>,
    pub customer_ref: String,
    pub plan_name: String,
    pub amount_minor: i64,
    pub currency: String,
    pub interval: String,
    pub interval_count: u32,
    pub status: String,
    pub current_period_start_unix_secs: u64,
    pub current_period_end_unix_secs: u64,
    pub cancel_at_period_end: bool,
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

fn parse_interval(s: &str) -> ApiResult<Interval> {
    Ok(match s {
        "day" => Interval::Day,
        "week" => Interval::Week,
        "month" => Interval::Month,
        "year" => Interval::Year,
        bad => {
            return Err(ApiError::BadRequest(format!(
                "unknown interval `{bad}` (expected day/week/month/year)"
            )));
        }
    })
}

fn method_from_payload(m: MethodPayload) -> PaymentMethod {
    match m {
        MethodPayload::Vault { token } => PaymentMethod::Vault(VaultRef::new(token)),
    }
}

fn to_response(s: &Subscription) -> SubscriptionResponse {
    SubscriptionResponse {
        id: s.id.as_uuid(),
        external_id: s.external_id.clone(),
        customer_ref: s.customer_ref.clone(),
        plan_name: s.plan.name.clone(),
        amount_minor: s.plan.amount.minor_units,
        currency: s.plan.amount.currency.code().to_owned(),
        interval: s.plan.interval.code().to_owned(),
        interval_count: s.plan.interval_count,
        status: status_code(&s.status).to_owned(),
        current_period_start_unix_secs: s.current_period_start_unix_secs,
        current_period_end_unix_secs: s.current_period_end_unix_secs,
        cancel_at_period_end: s.cancel_at_period_end,
    }
}

fn status_code(s: &Status) -> &'static str {
    s.code()
}

/// `POST /v1/subscriptions`
pub async fn create(
    State(state): State<AppState>,
    Json(req): Json<CreateSubscriptionRequest>,
) -> ApiResult<Json<SubscriptionResponse>> {
    let currency = currency_from_code(&req.currency)?;
    let interval = parse_interval(&req.interval)?;
    let amount = Money::from_minor(req.amount_minor, currency);
    let mut plan =
        Plan::new(req.plan_name, amount, interval, req.interval_count).map_err(ApiError::from)?;
    if let Some(days) = req.trial_days
        && days > 0
    {
        plan = plan.with_trial_days(days);
    }
    let mut sub = Subscription::new(
        req.customer_ref,
        plan,
        method_from_payload(req.method),
        req.now_unix_secs,
    )
    .map_err(ApiError::from)?;
    if let Some(ext) = req.external_id {
        sub = sub.with_external_id(ext);
    }
    let id = state.subscriptions.create_subscription(sub)?;
    let stored = state.subscriptions.get_subscription(id)?;
    let response = to_response(&stored);
    emit(&state, "subscription.created", &response);
    Ok(Json(response))
}

/// `GET /v1/subscriptions/{id}`
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<SubscriptionResponse>> {
    let s = state
        .subscriptions
        .get_subscription(SubscriptionId::from_uuid(id))?;
    Ok(Json(to_response(&s)))
}

/// `POST /v1/subscriptions/{id}/cancel`
pub async fn cancel(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<CancelRequest>,
) -> ApiResult<Json<SubscriptionResponse>> {
    let at_period_end = req.at_period_end.unwrap_or(true);
    let s = state
        .subscriptions
        .update(SubscriptionId::from_uuid(id), |sub| {
            if at_period_end {
                sub.schedule_cancel_at_period_end();
            } else {
                sub.cancel_now(req.now_unix_secs);
            }
            Ok(())
        })?;
    let response = to_response(&s);
    emit(&state, "subscription.canceled", &response);
    Ok(Json(response))
}

/// `POST /v1/subscriptions/{id}/pause`
pub async fn pause(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<PauseRequest>,
) -> ApiResult<Json<SubscriptionResponse>> {
    let s = state
        .subscriptions
        .update(SubscriptionId::from_uuid(id), |sub| {
            sub.pause(req.now_unix_secs)
        })?;
    let response = to_response(&s);
    emit(&state, "subscription.paused", &response);
    Ok(Json(response))
}

/// `POST /v1/subscriptions/{id}/resume`
pub async fn resume(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<SubscriptionResponse>> {
    let s = state.subscriptions.update(
        SubscriptionId::from_uuid(id),
        op_subscriptions::Subscription::resume,
    )?;
    let response = to_response(&s);
    emit(&state, "subscription.resumed", &response);
    Ok(Json(response))
}

/// `GET /v1/subscriptions?customer_ref=...`
pub async fn list(
    State(state): State<AppState>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Vec<SubscriptionResponse>>> {
    let subs = state.subscriptions.list_for_customer(&q.customer_ref)?;
    Ok(Json(subs.iter().map(to_response).collect()))
}
