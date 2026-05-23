//! HTTP error envelope.
//!
//! Maps each domain error variant to an HTTP status + a stable JSON
//! envelope:
//!
//! ```json
//! { "code": "not_found", "message": "...", "details": null }
//! ```

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::Value as Json_;

/// Stable JSON error envelope.
#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    /// Short machine-readable kind (`"not_found"`, `"invalid"`, ...).
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// Optional structured details. `null` if absent.
    pub details: Option<Json_>,
}

/// Wrapper that converts each crate's `Error` into an HTTP response.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// Routing / payload validation rejected the request.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// Resource lookup missed.
    #[error("not found: {0}")]
    NotFound(String),
    /// State machine refused the transition.
    #[error("conflict: {0}")]
    Conflict(String),
    /// Idempotency replay with mismatched body.
    #[error("idempotency mismatch on `{0}`")]
    IdempotencyMismatch(String),
    /// Catch-all for unexpected backend failures.
    #[error("internal error: {0}")]
    Internal(String),
}

impl ApiError {
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) | Self::IdempotencyMismatch(_) => StatusCode::CONFLICT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::NotFound(_) => "not_found",
            Self::Conflict(_) => "conflict",
            Self::IdempotencyMismatch(_) => "idempotency_mismatch",
            Self::Internal(_) => "internal",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ApiErrorBody {
            code: self.code().to_owned(),
            message: self.to_string(),
            details: None,
        };
        (self.status(), Json(body)).into_response()
    }
}

/// Shorthand result alias for handler returns.
pub type ApiResult<T> = Result<T, ApiError>;

// ============================================================
// Conversions from each crate's Error type
// ============================================================

impl From<op_refund::Error> for ApiError {
    fn from(e: op_refund::Error) -> Self {
        match e {
            op_refund::Error::NotFound(s) => Self::NotFound(s),
            op_refund::Error::InvalidTransition { .. } => Self::Conflict(e.to_string()),
            op_refund::Error::IdempotencyMismatch(s) => Self::IdempotencyMismatch(s),
            op_refund::Error::Invalid(_)
            | op_refund::Error::AmountExceeded { .. }
            | op_refund::Error::Core(_) => Self::BadRequest(e.to_string()),
        }
    }
}

impl From<op_dispute::Error> for ApiError {
    fn from(e: op_dispute::Error) -> Self {
        match e {
            op_dispute::Error::NotFound(s) => Self::NotFound(s),
            op_dispute::Error::InvalidTransition { .. } => Self::Conflict(e.to_string()),
            op_dispute::Error::IdempotencyMismatch(s) => Self::IdempotencyMismatch(s),
            op_dispute::Error::Invalid(_) | op_dispute::Error::Core(_) => {
                Self::BadRequest(e.to_string())
            }
        }
    }
}

impl From<op_settlement::Error> for ApiError {
    fn from(e: op_settlement::Error) -> Self {
        match e {
            op_settlement::Error::NotFound(s) => Self::NotFound(s),
            op_settlement::Error::InvalidTransition { .. } => Self::Conflict(e.to_string()),
            op_settlement::Error::IdempotencyMismatch(s) => Self::IdempotencyMismatch(s),
            op_settlement::Error::Invalid(_)
            | op_settlement::Error::CurrencyMismatch { .. }
            | op_settlement::Error::EmptyBatch
            | op_settlement::Error::Core(_)
            | op_settlement::Error::Iso20022(_) => Self::BadRequest(e.to_string()),
        }
    }
}

impl From<op_orchestrator::Error> for ApiError {
    fn from(e: op_orchestrator::Error) -> Self {
        Self::Internal(e.to_string())
    }
}

impl From<op_ledger::Error> for ApiError {
    fn from(e: op_ledger::Error) -> Self {
        match e {
            op_ledger::Error::AccountNotFound(s)
            | op_ledger::Error::TransactionNotFound(s)
            | op_ledger::Error::LedgerNotFound(s) => Self::NotFound(s),
            op_ledger::Error::IdempotencyMismatch => Self::IdempotencyMismatch(String::new()),
            op_ledger::Error::TerminalState { .. } => Self::Conflict(e.to_string()),
            _ => Self::BadRequest(e.to_string()),
        }
    }
}

impl From<op_graph::Error> for ApiError {
    fn from(e: op_graph::Error) -> Self {
        Self::Internal(e.to_string())
    }
}

impl From<op_fx::Error> for ApiError {
    fn from(e: op_fx::Error) -> Self {
        match e {
            op_fx::Error::NoQuote { .. } => Self::NotFound(e.to_string()),
            op_fx::Error::QuoteExpired { .. } => Self::Conflict(e.to_string()),
            op_fx::Error::SameCurrency(_)
            | op_fx::Error::CurrencyMismatch { .. }
            | op_fx::Error::InvalidRate
            | op_fx::Error::Core(_) => Self::BadRequest(e.to_string()),
            op_fx::Error::Overflow => Self::Internal(e.to_string()),
        }
    }
}

impl From<op_subscriptions::Error> for ApiError {
    fn from(e: op_subscriptions::Error) -> Self {
        match e {
            op_subscriptions::Error::NotFound(s) => Self::NotFound(s),
            op_subscriptions::Error::InvalidTransition { .. } => Self::Conflict(e.to_string()),
            op_subscriptions::Error::IdempotencyMismatch(s) => Self::IdempotencyMismatch(s),
            op_subscriptions::Error::Invalid(_)
            | op_subscriptions::Error::CurrencyMismatch { .. }
            | op_subscriptions::Error::Core(_) => Self::BadRequest(e.to_string()),
        }
    }
}
