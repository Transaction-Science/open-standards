//! Payment intent endpoint — runs the orchestrator.

use axum::Json;
use axum::extract::State;
use op_core::{Currency, Money, PaymentMethod, VaultRef};
// RailKind import is used in resume() below via the
// fully-qualified path; explicit import keeps the handler tidy.
use op_orchestrator::{IdempotencyKey, PaymentIntent};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::events::emit;
use crate::state::AppState;
use op_core::RailKind;

/// POST `/v1/intents` body.
#[derive(Debug, Deserialize)]
pub struct CreateIntentRequest {
    /// Caller-supplied idempotency key (the orchestrator dedupes
    /// against this; same key + same body returns the cached outcome).
    pub idempotency_key: String,
    /// Amount in minor units.
    pub amount_minor: i64,
    /// ISO 4217 currency.
    pub currency: String,
    /// Payment method. Only `vault` and `qr` are supported on the
    /// reference server surface; richer types live behind operator
    /// auth (we don't accept raw PAN on the public API).
    pub method: MethodPayload,
    /// Free-form metadata round-tripped onto outcome telemetry.
    #[serde(default)]
    pub metadata: Vec<(String, String)>,
}

/// Payment-method payload — small subset for the reference server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MethodPayload {
    /// Vault token — opaque PSP-side ref to a stored card.
    Vault {
        /// Vault token.
        token: String,
    },
    /// QR code payload (UPI / Pix string).
    Qr {
        /// QR string content.
        payload: String,
    },
}

/// POST `/v1/intents` response.
#[derive(Debug, Serialize)]
pub struct IntentResponse {
    /// `approved` / `declined` / `requires_customer_action`.
    pub terminal_status: String,
    /// How many rails were tried.
    pub attempt_count: usize,
    /// Rail used on the final attempt, if any.
    pub rail_used: Option<String>,
    /// PSP-side payment id (card rails) — caller holds this for
    /// capture/refund.
    pub psp_payment_id: Option<String>,
    /// UETR (A2A rails) — match settlement back to this intent.
    pub uetr: Option<String>,
    /// Per-attempt list.
    pub attempts: Vec<AttemptSummary>,
}

/// One entry in [`IntentResponse::attempts`].
#[derive(Debug, Serialize)]
pub struct AttemptSummary {
    /// Rail (`"Card"` / `"A2a"` / ...).
    pub rail: String,
    /// Driver name.
    pub driver: String,
    /// Outcome (`success` / `hard_decline` / `soft_failure` /
    /// `requires_action`).
    pub outcome: String,
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

fn method_from_payload(m: MethodPayload) -> PaymentMethod {
    match m {
        MethodPayload::Vault { token } => PaymentMethod::Vault(VaultRef::new(token)),
        MethodPayload::Qr { payload } => PaymentMethod::Qr(payload),
    }
}

fn outcome_code(o: &op_orchestrator::AttemptOutcome) -> &'static str {
    use op_orchestrator::AttemptOutcome::{HardDecline, RequiresAction, SoftFailure, Success};
    match o {
        Success => "success",
        HardDecline { .. } => "hard_decline",
        SoftFailure { .. } => "soft_failure",
        RequiresAction { .. } => "requires_action",
    }
}

fn terminal_code(t: op_orchestrator::TerminalStatus) -> &'static str {
    use op_orchestrator::TerminalStatus::{Approved, Declined, RequiresCustomerAction};
    match t {
        Approved => "approved",
        RequiresCustomerAction => "requires_customer_action",
        Declined => "declined",
    }
}

/// `POST /v1/intents`
pub async fn create(
    State(state): State<AppState>,
    Json(req): Json<CreateIntentRequest>,
) -> ApiResult<Json<IntentResponse>> {
    let currency = currency_from_code(&req.currency)?;
    let amount = Money::from_minor(req.amount_minor, currency);
    let method = method_from_payload(req.method);
    let mut intent = PaymentIntent::new(IdempotencyKey::new(&req.idempotency_key), amount, method);
    for (k, v) in req.metadata {
        intent = intent.with_metadata(k, v);
    }

    let outcome = state.orchestrator.run(&intent)?;
    let attempts: Vec<AttemptSummary> = outcome
        .attempts
        .iter()
        .map(|a| AttemptSummary {
            rail: format!("{:?}", a.rail),
            driver: a.driver.clone(),
            outcome: outcome_code(&a.outcome).to_owned(),
        })
        .collect();
    let response = IntentResponse {
        terminal_status: terminal_code(outcome.terminal_status).to_owned(),
        attempt_count: outcome.attempts.len(),
        rail_used: outcome.rail_used.map(|r| format!("{r:?}")),
        psp_payment_id: outcome.psp_payment_id.clone(),
        uetr: outcome.uetr.clone(),
        attempts,
    };
    let event_type = match outcome.terminal_status {
        op_orchestrator::TerminalStatus::Approved => "intent.approved",
        op_orchestrator::TerminalStatus::Declined => "intent.declined",
        op_orchestrator::TerminalStatus::RequiresCustomerAction => "intent.requires_action",
    };
    emit(&state, event_type, &response);
    Ok(Json(response))
}

/// `POST /v1/intents/resume` body — caller hands back the
/// `psp_payment_id` they received from a previous `intent.create`
/// that returned `requires_customer_action`, plus the rail/driver
/// pair that issued the challenge.
#[derive(Debug, Deserialize)]
pub struct ResumeIntentRequest {
    /// Original intent body — same fields as `CreateIntentRequest`.
    /// We require the full intent rather than relying on a server-
    /// side cache so the resume primitive is stateless from the
    /// HTTP layer's perspective. Idempotency still applies through
    /// the orchestrator's store.
    pub idempotency_key: String,
    pub amount_minor: i64,
    pub currency: String,
    pub method: MethodPayload,
    #[serde(default)]
    pub metadata: Vec<(String, String)>,
    /// Rail of the adapter that issued the challenge. `"card"` is
    /// the common case.
    pub rail: String,
    /// Driver name (e.g. `"hyperswitch"`, `"stripe"`).
    pub driver: String,
    /// The PSP-issued payment id the original `intent.create`
    /// returned in its response.
    pub psp_payment_id: String,
}

fn parse_rail(s: &str) -> ApiResult<RailKind> {
    Ok(match s {
        "card" => RailKind::Card,
        "a2a" => RailKind::A2a,
        "wallet" => RailKind::Wallet,
        "qr" => RailKind::Qr,
        "crypto" => RailKind::Crypto,
        bad => {
            return Err(ApiError::BadRequest(format!(
                "unknown rail `{bad}` (expected card/a2a/wallet/qr/crypto)"
            )));
        }
    })
}

/// `POST /v1/intents/resume`
pub async fn resume(
    State(state): State<AppState>,
    Json(req): Json<ResumeIntentRequest>,
) -> ApiResult<Json<IntentResponse>> {
    let currency = currency_from_code(&req.currency)?;
    let amount = Money::from_minor(req.amount_minor, currency);
    let method = method_from_payload(req.method);
    let rail = parse_rail(&req.rail)?;
    let mut intent = PaymentIntent::new(
        op_orchestrator::IdempotencyKey::new(&req.idempotency_key),
        amount,
        method,
    );
    for (k, v) in req.metadata {
        intent = intent.with_metadata(k, v);
    }

    let outcome = state
        .orchestrator
        .resume(&intent, rail, &req.driver, &req.psp_payment_id)?;

    let attempts: Vec<AttemptSummary> = outcome
        .attempts
        .iter()
        .map(|a| AttemptSummary {
            rail: format!("{:?}", a.rail),
            driver: a.driver.clone(),
            outcome: outcome_code(&a.outcome).to_owned(),
        })
        .collect();
    let response = IntentResponse {
        terminal_status: terminal_code(outcome.terminal_status).to_owned(),
        attempt_count: outcome.attempts.len(),
        rail_used: outcome.rail_used.map(|r| format!("{r:?}")),
        psp_payment_id: outcome.psp_payment_id.clone(),
        uetr: outcome.uetr.clone(),
        attempts,
    };
    let event_type = match outcome.terminal_status {
        op_orchestrator::TerminalStatus::Approved => "intent.resumed.approved",
        op_orchestrator::TerminalStatus::Declined => "intent.resumed.declined",
        op_orchestrator::TerminalStatus::RequiresCustomerAction => "intent.resumed.requires_action",
    };
    emit(&state, event_type, &response);
    Ok(Json(response))
}
