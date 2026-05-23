//! Routing-signal plumbing: how the orchestrator's *writer* side
//! (telemetry) and the router's *reader* side (signals) hand off
//! information about which rails and drivers have been noisy.
//!
//! Both surfaces are traits the orchestrator owns. Concrete impls
//! ship in storage crates (`op-graph` provides one over the
//! reconciliation graph); operators with their own observability
//! pipeline plug in their own. Defaults are no-ops, so the existing
//! routing behaviour is preserved when these aren't wired up.

use std::sync::Arc;

use op_core::RailKind;

use crate::outcome::AttemptOutcome;

/// What an attempt ultimately did, in the form the telemetry layer
/// cares about. Coarser than [`AttemptOutcome`] — the signal layer
/// only needs to know "did it move money?", "did it bounce
/// retryable?", "did it bounce terminal?".
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AttemptResultClass {
    /// Adapter returned a successful authorization / settlement.
    Approved,
    /// Adapter returned a retryable failure (transient transport,
    /// rate limit, 5xx, etc.).
    SoftFailure,
    /// Adapter returned a non-retryable failure (hard decline,
    /// fraud lock, schema rejection).
    HardFailure,
}

impl AttemptResultClass {
    /// Project an [`AttemptOutcome`] onto the coarser class the
    /// signals layer cares about. `RequiresAction` is treated as
    /// `Approved` from the telemetry standpoint — the rail did its
    /// job; the customer side is now in play.
    #[must_use]
    pub fn classify(o: &AttemptOutcome) -> Self {
        match o {
            AttemptOutcome::Success | AttemptOutcome::RequiresAction { .. } => Self::Approved,
            AttemptOutcome::SoftFailure { .. } => Self::SoftFailure,
            AttemptOutcome::HardDecline { .. } => Self::HardFailure,
        }
    }

    /// True for anything that didn't move money.
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::SoftFailure | Self::HardFailure)
    }
}

/// Writer side: the engine pushes an outcome here after every
/// `RailAdapter::attempt` call. Implementors record it however
/// they like (in-memory ring buffer, graph store, Prometheus,
/// log line).
pub trait RailTelemetry: Send + Sync + std::fmt::Debug {
    /// Record one (rail, driver) outcome at `at_unix_secs`.
    ///
    /// `external_id_hint` is the operator-supplied correlation
    /// token — typically the intent's idempotency key, which most
    /// deployments propagate verbatim to the ledger transaction's
    /// `external_id`. Backed stores use it to join attempt history
    /// against downstream ledger / reconciliation state (see
    /// [`RoutingSignals::discrepancy_score`]). Pass `None` for
    /// attempts that don't correspond to a ledger transaction
    /// (e.g. a dry-run probe).
    ///
    /// `duration_ms` is the adapter's observed latency for this
    /// attempt. `None` when timing wasn't measured.
    fn record_attempt(
        &self,
        rail: RailKind,
        driver: &str,
        outcome: AttemptResultClass,
        at_unix_secs: u64,
        external_id_hint: Option<&str>,
        duration_ms: Option<u32>,
    );
}

/// Reader side: the router queries this to learn which (rail,
/// driver) combinations have been noisy recently, and uses the
/// scores to prefer quieter ones in the fallback chain.
///
/// All scores are in `[0.0, 1.0]`:
/// - `0.0` = no recent signal (clean or no data)
/// - `1.0` = every recent attempt fired the signal
///
/// "Recent" is implementation-defined — typically a sliding window
/// of the last few minutes to an hour. The router folds the axes
/// into a single ranking key via [`SignalCombiner`].
pub trait RoutingSignals: Send + Sync + std::fmt::Debug {
    /// Recent failure rate for the given `(rail, driver)` — the
    /// HTTP-level "did the adapter return OK" signal.
    fn failure_score(&self, rail: RailKind, driver: &str) -> f32;

    /// Recent reconciliation-discrepancy density for the given
    /// `(rail, driver)` — the bank-level "did the books match" signal.
    ///
    /// A driver whose HTTP attempts all return OK can still produce
    /// statement-mismatch tasks downstream (PSP fee netting wrong,
    /// settlement amount drift, etc.). The router treats this as an
    /// additional reason to push the driver back, complementary to
    /// [`Self::failure_score`].
    ///
    /// Returns `0.0` by default — implementations that don't have
    /// access to reconciliation state should leave this alone.
    fn discrepancy_score(&self, _rail: RailKind, _driver: &str) -> f32 {
        0.0
    }

    /// Recent latency score for `(rail, driver)` — fast=0.0,
    /// pathological=1.0. The mapping from raw latency (milliseconds)
    /// to a `[0.0, 1.0]` score is implementation-defined; a sensible
    /// default is `min(1.0, p50_ms / SLO_MS)`. A driver that returns
    /// success but takes 30 seconds is still a bad routing choice.
    ///
    /// Returns `0.0` by default — implementations that don't track
    /// latency should leave this alone.
    fn latency_score(&self, _rail: RailKind, _driver: &str) -> f32 {
        0.0
    }
}

/// How [`PolicyRouter`](crate::PolicyRouter) folds the
/// [`RoutingSignals`] axes (`failure`, `discrepancy`, `latency`)
/// into a single ranking score in `[0.0, 1.0]`.
///
/// Default is [`Self::WorstAxis`] — "max across axes." That preserves
/// the Phase 18/19 semantics: any single axis being bad is enough
/// reason to push a driver back.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum SignalCombiner {
    /// `max(failure, discrepancy, latency)`. Worst-news-wins.
    #[default]
    WorstAxis,
    /// Weighted sum, clamped to `[0.0, 1.0]`. Weights need not sum
    /// to 1 — they're a relative importance vector. Useful for
    /// operators who want "two soft signals shouldn't outvote one
    /// hard failure": e.g. `Weighted { failure: 1.0, discrepancy:
    /// 0.5, latency: 0.3 }`.
    Weighted {
        /// Multiplier on `failure_score`.
        failure: f32,
        /// Multiplier on `discrepancy_score`.
        discrepancy: f32,
        /// Multiplier on `latency_score`.
        latency: f32,
    },
    /// Treat a single axis as a hard demote: if it exceeds the
    /// threshold, returns `1.0` regardless of the other axes.
    /// Otherwise falls back to `WorstAxis`. Useful for "fail-fast
    /// on outages" deployments where any soft signal at 0.9+ should
    /// immediately ground a driver.
    HardDemoteAbove {
        /// Threshold a single axis must exceed to trigger the demote.
        threshold: f32,
    },
}

impl SignalCombiner {
    /// Apply this combiner to a given `(rail, driver)`. The result
    /// is the single ranking score the router uses to order drivers
    /// within a rail group (lower = quieter).
    pub fn combine(&self, signals: &dyn RoutingSignals, rail: RailKind, driver: &str) -> f32 {
        let f = signals.failure_score(rail, driver);
        let d = signals.discrepancy_score(rail, driver);
        let l = signals.latency_score(rail, driver);
        match self {
            Self::WorstAxis => f.max(d).max(l),
            Self::Weighted {
                failure,
                discrepancy,
                latency,
            } => {
                let raw = f * *failure + d * *discrepancy + l * *latency;
                raw.clamp(0.0, 1.0)
            }
            Self::HardDemoteAbove { threshold } => {
                if f >= *threshold || d >= *threshold || l >= *threshold {
                    1.0
                } else {
                    f.max(d).max(l)
                }
            }
        }
    }
}

// ============================================================
// No-op defaults
// ============================================================

/// Telemetry sink that discards every record. The default for
/// operators who haven't wired up a real telemetry backend; keeps
/// the orchestrator working without ever requiring one.
#[derive(Clone, Debug, Default)]
pub struct NoOpRailTelemetry;

impl RailTelemetry for NoOpRailTelemetry {
    fn record_attempt(
        &self,
        _rail: RailKind,
        _driver: &str,
        _outcome: AttemptResultClass,
        _at_unix_secs: u64,
        _external_id_hint: Option<&str>,
        _duration_ms: Option<u32>,
    ) {
    }
}

/// Signals source that returns zero for every query — equivalent to
/// "no information," which the router treats as "no preference."
/// The default for routers without a wired-up signals backend.
#[derive(Clone, Debug, Default)]
pub struct NoOpRoutingSignals;

impl RoutingSignals for NoOpRoutingSignals {
    fn failure_score(&self, _rail: RailKind, _driver: &str) -> f32 {
        0.0
    }
}

/// Type-erased default sink — clones cheaply.
#[must_use]
pub fn noop_telemetry() -> Arc<dyn RailTelemetry> {
    Arc::new(NoOpRailTelemetry)
}

/// Type-erased default signals — clones cheaply.
#[must_use]
pub fn noop_signals() -> Arc<dyn RoutingSignals> {
    Arc::new(NoOpRoutingSignals)
}
