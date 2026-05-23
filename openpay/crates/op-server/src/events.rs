//! Small helpers for emitting domain events at the HTTP handler
//! boundary.
//!
//! Each successful handler call invokes `emit(state, event_type,
//! &body)` after the mutation is persisted. Emission is
//! best-effort and never fails the HTTP response — the user-
//! observable state change already succeeded.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::state::AppState;

/// Serialize `body` as JSON and publish to every endpoint
/// subscribed to `event_type`. Failures are logged via the
/// emitter's underlying tracing; the handler caller never sees an
/// error from this.
pub fn emit<T: Serialize>(state: &AppState, event_type: &str, body: &T) {
    let payload = match serde_json::to_vec(body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                error = %e,
                event_type,
                "event payload serialization failed; skipping emission"
            );
            return;
        }
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    state.events.emit(event_type, payload, now);
}
