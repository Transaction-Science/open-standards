//! Audit report endpoint.

use axum::Json;
use axum::extract::{Query, State};
use op_graph::AuditReport;
use serde::Deserialize;

use crate::error::ApiResult;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct AuditQuery {
    pub start_tx: u64,
    pub end_tx: u64,
    pub generated_at_unix_secs: u64,
}

/// `GET /v1/audit/report?start_tx=..&end_tx=..&generated_at_unix_secs=..`
pub async fn report(
    State(state): State<AppState>,
    Query(q): Query<AuditQuery>,
) -> ApiResult<Json<AuditReport>> {
    let report =
        AuditReport::for_window(&state.graph, q.start_tx, q.end_tx, q.generated_at_unix_secs)?;
    Ok(Json(report))
}
