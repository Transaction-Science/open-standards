//! [`GraphRailTelemetry`] — graph-backed implementation of both
//! [`RailTelemetry`] and [`RoutingSignals`].
//!
//! Each `record_attempt` call writes a `rail_attempt` vertex with
//! these properties:
//!
//! - `rail` — `"Card"`, `"A2a"`, `"Wallet"`, `"Qr"` (the
//!   [`RailKind`] discriminant).
//! - `driver` — operator-registered driver name.
//! - `outcome` — `"approved"`, `"soft_failure"`, `"hard_failure"`.
//! - `at_unix_secs` — when the attempt happened.
//!
//! `failure_score(rail, driver)` walks every `rail_attempt`
//! vertex whose `at_unix_secs` falls inside the configured
//! sliding window (default 1 hour) and returns
//! `failures / total`. Zero attempts in window → score 0.0 (the
//! "no information" answer the router treats as no preference).
//!
//! ## Persistence
//!
//! When the underlying [`GraphHandle`] is file-backed
//! (`GraphHandle::new_persistent`), the attempt log survives across
//! restarts — operators don't lose their rail-health view because
//! the orchestrator was bounced.

use std::time::{SystemTime, UNIX_EPOCH};

use op_core::RailKind;
use op_orchestrator::{AttemptResultClass, RailTelemetry, RoutingSignals};
use serde_json::Value as Json;
use uuid::Uuid;

use crate::graph::{GraphHandle, etypes, vtypes};

/// Default sliding window for `failure_score`: one hour. Long enough
/// to dampen one-off blips, short enough that a dead PSP gets
/// avoided promptly.
pub const DEFAULT_WINDOW_SECS: u64 = 3600;

/// Default latency SLO. The `latency_score` is `min(1.0, p50_ms / SLO)`,
/// so a driver whose median attempt clears in 2s on a 2s SLO has
/// score 1.0 (push back). Operators tune via [`GraphRailTelemetry::with_latency_slo_ms`].
pub const DEFAULT_LATENCY_SLO_MS: u32 = 2_000;

/// Per-discrepancy-kind severity weight. Multiplied into the raw
/// `tasks/attempts` ratio in `discrepancy_score`. A driver
/// producing a hard `amount_mismatch` is worse than one producing
/// an `status_mismatch` that's likely just a settlement-timing
/// blip. Defaults reflect that asymmetry; operators override via
/// [`GraphRailTelemetry::with_discrepancy_weights`].
#[derive(Clone, Debug)]
pub struct DiscrepancyWeights {
    /// Bank says a payment landed we never booked. Hard signal.
    pub unmatched_statement: f32,
    /// We booked a tx the bank hasn't reported. Often a timing
    /// blip; weight lower than `unmatched_statement`.
    pub unmatched_ledger: f32,
    /// Books and bank disagree on amount. Hard signal.
    pub amount_mismatch: f32,
    /// Books still pending but bank settled (or vice versa). Often
    /// a timing blip.
    pub status_mismatch: f32,
    /// Catch-all for kinds the table doesn't enumerate.
    pub fallback: f32,
}

impl Default for DiscrepancyWeights {
    fn default() -> Self {
        Self {
            unmatched_statement: 1.0,
            unmatched_ledger: 0.7,
            amount_mismatch: 1.0,
            status_mismatch: 0.6,
            fallback: 1.0,
        }
    }
}

impl DiscrepancyWeights {
    /// Look up the weight for a `TaskDescriptor::kind` string.
    #[must_use]
    pub fn for_kind(&self, kind: &str) -> f32 {
        match kind {
            "unmatched_statement" => self.unmatched_statement,
            "unmatched_ledger" => self.unmatched_ledger,
            "amount_mismatch" => self.amount_mismatch,
            "status_mismatch" => self.status_mismatch,
            _ => self.fallback,
        }
    }
}

/// Graph-backed implementation of both [`RailTelemetry`] (writer
/// side) and [`RoutingSignals`] (reader side).
#[derive(Debug, Clone)]
pub struct GraphRailTelemetry {
    handle: GraphHandle,
    window_secs: u64,
    latency_slo_ms: u32,
    weights: DiscrepancyWeights,
}

impl GraphRailTelemetry {
    /// Build with a fresh in-memory graph.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_handle(GraphHandle::new_in_memory())
    }

    /// Build over an existing handle — share it with the ledger /
    /// webhook / reconciliation stores so rail-attempt history sits
    /// alongside the rest of the graph.
    #[must_use]
    pub fn with_handle(handle: GraphHandle) -> Self {
        Self {
            handle,
            window_secs: DEFAULT_WINDOW_SECS,
            latency_slo_ms: DEFAULT_LATENCY_SLO_MS,
            weights: DiscrepancyWeights::default(),
        }
    }

    /// Builder: override the per-kind discrepancy severity weights.
    #[must_use]
    pub fn with_discrepancy_weights(mut self, weights: DiscrepancyWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Builder: override the sliding window for `failure_score`.
    #[must_use]
    pub fn with_window_secs(mut self, window_secs: u64) -> Self {
        self.window_secs = window_secs;
        self
    }

    /// Builder: override the latency SLO that anchors `latency_score`.
    /// The score for a driver is `min(1.0, p50_window_ms / slo_ms)`.
    #[must_use]
    pub fn with_latency_slo_ms(mut self, slo_ms: u32) -> Self {
        self.latency_slo_ms = slo_ms.max(1);
        self
    }

    /// Borrow the underlying handle.
    #[must_use]
    pub fn handle(&self) -> &GraphHandle {
        &self.handle
    }

    /// Read every `rail_attempt` vertex's `(rail, driver, outcome,
    /// at_unix_secs)`. Used by `failure_score` and by tests that
    /// want to inspect the attempt log directly.
    pub fn list_attempts(&self) -> crate::Result<Vec<RailAttemptRecord>> {
        let verts = self.handle.vertices_of_type(vtypes::RAIL_ATTEMPT)?;
        let mut out = Vec::with_capacity(verts.len());
        for v in verts {
            let props = self.handle.get_vertex_properties(v.id)?;
            let rail = match props.get("rail") {
                Some(Json::String(s)) => parse_rail(s),
                _ => continue,
            };
            let driver = match props.get("driver") {
                Some(Json::String(s)) => s.clone(),
                _ => continue,
            };
            let outcome = match props.get("outcome") {
                Some(Json::String(s)) => parse_outcome(s),
                _ => continue,
            };
            let at_unix_secs = match props.get("at_unix_secs") {
                Some(Json::Number(n)) => n.as_u64().unwrap_or(0),
                _ => continue,
            };
            let external_id_hint = match props.get("external_id_hint") {
                Some(Json::String(s)) => Some(s.clone()),
                _ => None,
            };
            let duration_ms = match props.get("duration_ms") {
                Some(Json::Number(n)) => n.as_u64().and_then(|v| u32::try_from(v).ok()),
                _ => None,
            };
            out.push(RailAttemptRecord {
                rail,
                driver,
                outcome,
                at_unix_secs,
                external_id_hint,
                duration_ms,
            });
        }
        Ok(out)
    }

    /// Reference "now" used by `failure_score`'s window cutoff.
    /// Overridden in tests via [`Self::failure_score_at`].
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Same as [`RoutingSignals::failure_score`] but with an
    /// explicit `now_unix_secs` — for deterministic tests.
    #[must_use]
    pub fn failure_score_at(&self, rail: RailKind, driver: &str, now_unix_secs: u64) -> f32 {
        let cutoff = now_unix_secs.saturating_sub(self.window_secs);
        let Ok(records) = self.list_attempts() else {
            return 0.0;
        };
        let mut total: u32 = 0;
        let mut failures: u32 = 0;
        for r in records {
            if r.at_unix_secs < cutoff {
                continue;
            }
            if r.rail != rail || r.driver != driver {
                continue;
            }
            total += 1;
            if r.outcome.is_failure() {
                failures += 1;
            }
        }
        if total == 0 {
            return 0.0;
        }
        f32::from(u16::try_from(failures).unwrap_or(u16::MAX))
            / f32::from(u16::try_from(total).unwrap_or(u16::MAX))
    }
}

impl RailTelemetry for GraphRailTelemetry {
    fn record_attempt(
        &self,
        rail: RailKind,
        driver: &str,
        outcome: AttemptResultClass,
        at_unix_secs: u64,
        external_id_hint: Option<&str>,
        duration_ms: Option<u32>,
    ) {
        // record_attempt is fire-and-forget in the trait contract;
        // a backend failure shouldn't crash a payment flow, so we
        // swallow errors. Future enhancement: add an out-of-band
        // diagnostic sink for operators who want to observe these.
        let id = Uuid::new_v4();
        let _ = self.handle.create_vertex(vtypes::RAIL_ATTEMPT, id);
        let _ =
            self.handle
                .set_vertex_property(id, "rail", Json::String(rail_to_str(rail).to_owned()));
        let _ = self
            .handle
            .set_vertex_property(id, "driver", Json::String(driver.to_owned()));
        let _ = self.handle.set_vertex_property(
            id,
            "outcome",
            Json::String(outcome_to_str(outcome).to_owned()),
        );
        let _ = self.handle.set_vertex_property(
            id,
            "at_unix_secs",
            Json::Number(serde_json::Number::from(at_unix_secs)),
        );
        // The correlation token — typically the intent's
        // idempotency key. Lets `discrepancy_score` join this
        // attempt against the ledger transaction the operator
        // posted under the same id, and from there to any
        // reconciliation tasks that touch it.
        if let Some(hint) = external_id_hint {
            let _ = self.handle.set_vertex_property(
                id,
                "external_id_hint",
                Json::String(hint.to_owned()),
            );
        }
        if let Some(ms) = duration_ms {
            let _ = self.handle.set_vertex_property(
                id,
                "duration_ms",
                Json::Number(serde_json::Number::from(ms)),
            );
        }
    }
}

impl RoutingSignals for GraphRailTelemetry {
    fn failure_score(&self, rail: RailKind, driver: &str) -> f32 {
        self.failure_score_at(rail, driver, Self::now())
    }

    fn discrepancy_score(&self, rail: RailKind, driver: &str) -> f32 {
        self.discrepancy_score_at(rail, driver, Self::now())
    }

    fn latency_score(&self, rail: RailKind, driver: &str) -> f32 {
        self.latency_score_at(rail, driver, Self::now())
    }
}

impl GraphRailTelemetry {
    /// Reconciliation-discrepancy density for `(rail, driver)` over
    /// the sliding window. Joins:
    ///
    /// ```text
    /// rail_attempt(rail, driver, external_id_hint)
    ///   ↳ ledger_tx{external_id == external_id_hint}
    ///       ↳ inbound task_about edges from reconciliation_task vertices
    /// ```
    ///
    /// Score is `tasks_touching_driver_attempts / attempts_in_window`.
    /// Capped at 1.0 (multiple tasks against a single attempt push the
    /// driver back hard but not insanely). Returns 0.0 when there are
    /// no recent attempts for `(rail, driver)`.
    #[must_use]
    pub fn discrepancy_score_at(&self, rail: RailKind, driver: &str, now_unix_secs: u64) -> f32 {
        let cutoff = now_unix_secs.saturating_sub(self.window_secs);
        let Ok(records) = self.list_attempts() else {
            return 0.0;
        };

        // Collect (rail, driver)'s recent attempts and their hints.
        let mut hints: Vec<String> = Vec::new();
        let mut attempts: u32 = 0;
        for r in &records {
            if r.rail != rail || r.driver != driver {
                continue;
            }
            if r.at_unix_secs < cutoff {
                continue;
            }
            attempts += 1;
            if let Some(h) = &r.external_id_hint {
                hints.push(h.clone());
            }
        }
        if attempts == 0 {
            return 0.0;
        }

        // Find ledger_tx vertices whose `external_id` matches one of
        // the hints. For each inbound `task_about` edge, read the
        // referencing reconciliation_task's `kind` property and add
        // the configured severity weight to the tally.
        let Ok(tx_vertices) = self.handle.vertices_of_type(vtypes::LEDGER_TX) else {
            return 0.0;
        };
        let mut weighted_tasks: f32 = 0.0;
        for v in tx_vertices {
            let Ok(props) = self.handle.get_vertex_properties(v.id) else {
                continue;
            };
            let Some(Json::String(ext)) = props.get("external_id") else {
                continue;
            };
            if !hints.iter().any(|h| h == ext) {
                continue;
            }
            let Ok(incoming) = self.handle.in_edges(v.id, etypes::TASK_ABOUT) else {
                continue;
            };
            for edge in incoming {
                // The source of a `task_about` edge is the task vertex.
                let kind = self
                    .handle
                    .get_vertex_properties(edge.from)
                    .ok()
                    .and_then(|p| match p.get("kind") {
                        Some(Json::String(k)) => Some(k.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                weighted_tasks += self.weights.for_kind(&kind);
            }
        }

        // Normalize. Weighted-tasks per attempt; cap at 1.0.
        let raw = weighted_tasks / f32::from(u16::try_from(attempts).unwrap_or(u16::MAX));
        raw.min(1.0)
    }

    /// Latency score for `(rail, driver)` over the sliding window.
    /// Median observed `duration_ms` divided by the configured SLO,
    /// capped at 1.0. Attempts in the window without a recorded
    /// duration are skipped. Returns 0.0 when no timed attempts
    /// exist for the (rail, driver) pair.
    #[must_use]
    pub fn latency_score_at(&self, rail: RailKind, driver: &str, now_unix_secs: u64) -> f32 {
        let cutoff = now_unix_secs.saturating_sub(self.window_secs);
        let Ok(records) = self.list_attempts() else {
            return 0.0;
        };
        let mut samples: Vec<u32> = Vec::new();
        for r in records {
            if r.rail != rail || r.driver != driver || r.at_unix_secs < cutoff {
                continue;
            }
            if let Some(ms) = r.duration_ms {
                samples.push(ms);
            }
        }
        if samples.is_empty() {
            return 0.0;
        }
        // Median (sort then mid). For ML / production this would be
        // p95; for the reference impl median is fine and stable.
        samples.sort_unstable();
        let p50 = samples[samples.len() / 2];
        let ratio = f64::from(p50) / f64::from(self.latency_slo_ms.max(1));
        #[allow(clippy::cast_possible_truncation)]
        let score = ratio.min(1.0) as f32;
        score
    }
}

/// One row of the attempt log, decoded from the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailAttemptRecord {
    /// Rail this attempt went to.
    pub rail: RailKind,
    /// Driver name.
    pub driver: String,
    /// Coarse outcome class.
    pub outcome: AttemptResultClass,
    /// When the attempt happened (unix epoch seconds).
    pub at_unix_secs: u64,
    /// Operator-supplied correlation token (typically the intent's
    /// idempotency key). `None` if the caller didn't supply one.
    pub external_id_hint: Option<String>,
    /// Observed adapter latency for this attempt. `None` if not
    /// measured.
    pub duration_ms: Option<u32>,
}

// ============================================================
// Codec helpers
// ============================================================

const fn rail_to_str(r: RailKind) -> &'static str {
    match r {
        RailKind::Card => "Card",
        RailKind::A2a => "A2a",
        RailKind::Wallet => "Wallet",
        RailKind::Qr => "Qr",
        RailKind::Crypto => "Crypto",
    }
}

fn parse_rail(s: &str) -> RailKind {
    match s {
        "A2a" => RailKind::A2a,
        "Wallet" => RailKind::Wallet,
        "Qr" => RailKind::Qr,
        "Crypto" => RailKind::Crypto,
        // "Card" or anything unrecognized falls back to Card —
        // safer to attribute an unknown rail to the most common
        // class than to drop the record.
        _ => RailKind::Card,
    }
}

const fn outcome_to_str(o: AttemptResultClass) -> &'static str {
    match o {
        AttemptResultClass::Approved => "approved",
        AttemptResultClass::SoftFailure => "soft_failure",
        AttemptResultClass::HardFailure => "hard_failure",
    }
}

fn parse_outcome(s: &str) -> AttemptResultClass {
    match s {
        "soft_failure" => AttemptResultClass::SoftFailure,
        "hard_failure" => AttemptResultClass::HardFailure,
        _ => AttemptResultClass::Approved,
    }
}
