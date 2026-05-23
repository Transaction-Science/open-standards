//! `/health` and `/readiness` endpoints.

use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};

use crate::state::AppState;

/// Liveness — always 200. Operators use this for load-balancer
/// health checks.
pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// Readiness — verifies the in-memory stores can answer a basic
/// query (refunds and disputes report their length without
/// erroring). For real backends, replace with a backend ping.
pub async fn readiness(State(state): State<AppState>) -> Json<Value> {
    let _ = state.refunds.len();
    let _ = state.disputes.len();
    let _ = state.settlement.len();
    Json(json!({
        "status": "ready",
        "refunds": state.refunds.len(),
        "disputes": state.disputes.len(),
        "batches": state.settlement.len(),
    }))
}
