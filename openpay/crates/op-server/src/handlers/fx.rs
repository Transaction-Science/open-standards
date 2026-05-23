//! FX endpoints: quote retrieval and conversion.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Query, State};
use op_core::{Currency, Money};
use op_fx::{RoundingMode, convert};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct QuoteQuery {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Serialize)]
pub struct QuoteResponse {
    pub from: String,
    pub to: String,
    pub rate_ppm: u64,
    pub fetched_at_unix_secs: u64,
    pub valid_until_unix_secs: u64,
    pub source: String,
}

#[derive(Debug, Deserialize)]
pub struct ConvertRequest {
    pub from: String,
    pub to: String,
    pub amount_minor: i64,
    /// Optional: `half_even` (default), `down`, `up`.
    pub rounding: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConvertResponse {
    pub from: String,
    pub to: String,
    pub source_amount_minor: i64,
    pub target_amount_minor: i64,
    pub rate_ppm: u64,
    pub rounding: String,
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

fn parse_rounding(s: Option<&str>) -> ApiResult<RoundingMode> {
    Ok(match s.unwrap_or("half_even") {
        "half_even" => RoundingMode::HalfEven,
        "down" => RoundingMode::Down,
        "up" => RoundingMode::Up,
        bad => {
            return Err(ApiError::BadRequest(format!(
                "unknown rounding `{bad}` (expected half_even/down/up)"
            )));
        }
    })
}

fn rounding_code(m: RoundingMode) -> &'static str {
    match m {
        RoundingMode::HalfEven => "half_even",
        RoundingMode::Down => "down",
        RoundingMode::Up => "up",
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `GET /v1/fx/quote?from=USD&to=EUR`
pub async fn quote(
    State(state): State<AppState>,
    Query(q): Query<QuoteQuery>,
) -> ApiResult<Json<QuoteResponse>> {
    let from = currency_from_code(&q.from)?;
    let to = currency_from_code(&q.to)?;
    let quote = state.fx.get_quote(from, to, now())?;
    Ok(Json(QuoteResponse {
        from: quote.source_currency.code().to_owned(),
        to: quote.target_currency.code().to_owned(),
        rate_ppm: quote.rate_ppm,
        fetched_at_unix_secs: quote.fetched_at_unix_secs,
        valid_until_unix_secs: quote.valid_until_unix_secs,
        source: quote.source_name,
    }))
}

/// `POST /v1/fx/convert`
pub async fn convert_handler(
    State(state): State<AppState>,
    Json(req): Json<ConvertRequest>,
) -> ApiResult<Json<ConvertResponse>> {
    let from = currency_from_code(&req.from)?;
    let to = currency_from_code(&req.to)?;
    let rounding = parse_rounding(req.rounding.as_deref())?;
    let now_secs = now();
    let quote = state.fx.get_quote(from, to, now_secs)?;
    let source_money = Money::from_minor(req.amount_minor, from);
    let target_money = convert(source_money, &quote, rounding, now_secs)?;
    Ok(Json(ConvertResponse {
        from: from.code().to_owned(),
        to: to.code().to_owned(),
        source_amount_minor: req.amount_minor,
        target_amount_minor: target_money.minor_units,
        rate_ppm: quote.rate_ppm,
        rounding: rounding_code(rounding).to_owned(),
    }))
}
