//! The `Tier` trait and the `Runtime` that walks tiers in cost order.
//!
//! See `specs/r0.1-query-answer-tier.md` and
//! `specs/r0.2-budget-determinism.md` for the design.

use crate::types::*;
use std::time::{Duration, Instant};

// ============================================================
// Tier trait
// ============================================================

/// Cost prediction for a tier on a specific query, before doing work.
#[derive(Debug, Clone, Copy)]
pub struct TierEstimate {
    /// Predicted joule cost.
    pub joules: f64,
    /// Predicted wall-clock latency.
    pub latency: Duration,
    /// Tier's lower-bound confidence in answering this query class.
    /// Used by the runtime to skip tiers whose floor is below the
    /// query's quality floor.
    pub confidence_floor: f32,
}

/// Every tier implements this trait. The runtime walks tiers, calling
/// `estimate_cost` to decide whether to dispatch and `try_answer` to
/// actually run.
///
/// Contract:
/// - `estimate_cost` is cheap (microseconds, picojoules).
/// - `estimate_cost` returns `None` iff the tier cannot handle this
///   query class at all.
/// - `try_answer` MUST NOT spend more than `budget_remaining` joules.
/// - `try_answer`'s returned `Answer.tier_used` MUST equal `self.id()`.
/// - `try_answer` returns `AnswerOutput::Refused` rather than
///   producing an answer below `q.quality.min_confidence`.
pub trait Tier: Send {
    fn id(&self) -> TierId;
    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate>;
    fn try_answer(
        &mut self,
        q: &Query,
        budget_remaining: f64,
    ) -> Result<Answer, AnswerError>;

    /// The Synthesis coordinate of this tier, if known.
    /// `⟨Z, E, T, P, I, V, R⟩` from the Periodic Stack.
    ///
    /// Default: `None`, meaning the tier has not been coordinated.
    /// New tiers should override this; legacy tiers can be
    /// coordinated incrementally without breaking.
    fn coord(&self) -> Option<crate::coord::Coord> {
        None
    }

    /// Rich cost estimate with prefill/decode split, substrate,
    /// and impedance mismatch. Optional; tiers that haven't been
    /// upgraded return `None` and the runtime falls back to
    /// `estimate_cost`.
    fn cost_estimate(&self, _q: &Query) -> Option<crate::cost::CostEstimate> {
        None
    }
}

// ============================================================
// Cascade — the ordered registry of tiers
// ============================================================

/// The ordered registry of available tiers. The runtime walks tiers in
/// the order they were registered when no router is provided.
pub struct Cascade {
    tiers: Vec<Box<dyn Tier>>,
}

impl Cascade {
    pub fn new() -> Self {
        Self { tiers: Vec::new() }
    }

    /// Register a tier. Tiers should be added in cheapest-first order
    /// for the default cost-ordered walk.
    pub fn register(&mut self, tier: Box<dyn Tier>) -> &mut Self {
        self.tiers.push(tier);
        self
    }

    pub fn tier_ids(&self) -> Vec<TierId> {
        self.tiers.iter().map(|t| t.id()).collect()
    }

    /// Return the Synthesis coordinate of each registered tier
    /// (where the tier reports one).
    pub fn tier_coords(&self) -> Vec<(TierId, Option<crate::coord::Coord>)> {
        self.tiers.iter().map(|t| (t.id(), t.coord())).collect()
    }

    /// Return the cells (discrete cell IDs) occupied by registered
    /// tiers. Useful for asking: "which of the 8,000 cells does this
    /// cascade actually cover?"
    pub fn occupied_cells(&self) -> Vec<u16> {
        let mut cells: Vec<u16> = self.tiers.iter()
            .filter_map(|t| t.coord().map(|c| c.cell_id()))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    /// Count of distinct cells occupied. The "coverage" of this
    /// cascade across the Synthesis discrete subspace.
    pub fn coverage(&self) -> usize {
        self.occupied_cells().len()
    }
}

impl Default for Cascade {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Runtime — the cascade walker
// ============================================================

/// The cascade runtime. `answer(query)` is the entry point.
///
/// The runtime maintains a built-in L0 cache as the first tier. The
/// cache is consulted before any other tier; successful answers from
/// later tiers are written back to it. Constructing a runtime without
/// the auto-cache (e.g., for tests that want to disable L0) is possible
/// via `Runtime::new_without_l0`.
pub struct Runtime {
    /// The L0 cache. Always checked first, always written to on hit
    /// from another tier. `None` only if the constructor was
    /// `new_without_l0`.
    l0: Option<crate::l0_cache::L0Cache>,
    /// Optional pluggable history layer. When present, the runtime
    /// reads from and writes to this *in addition to* the built-in
    /// L0 cache. The L0 stays as the hot in-memory tier; the history
    /// layer is the durable backing.
    history: Option<Box<dyn crate::history::HistoryLayer>>,
    /// Optional router. When present, the runtime consults the router
    /// to pick a tier dispatch order. When absent, tiers are walked
    /// in registration order.
    router: Option<Box<dyn crate::router::Router>>,
    /// Per-tier calibration data. Populated on every tier dispatch.
    /// Used to detect dishonest cost estimates over time.
    calibration: crate::calibration::CalibrationReport,
    /// Verification ledger for delayed-outcome dispatches. When a
    /// tier returns an Answer with `Pending(token)`, the token is
    /// issued here; the caller calls `runtime.resolve(token, outcome)`
    /// when the eventual outcome lands, and calibration is updated
    /// automatically with the post-hoc actual cost.
    ledger: crate::verification::VerificationLedger,
    cascade: Cascade,
}

impl Runtime {
    /// Construct a runtime with an L0 cache prepended to the cascade.
    /// The provided cascade should NOT include an L0Cache; the runtime
    /// owns its own.
    pub fn new(cascade: Cascade) -> Self {
        Self {
            l0: Some(crate::l0_cache::L0Cache::new()),
            history: None,
            router: None,
            calibration: crate::calibration::CalibrationReport::default(),
            ledger: crate::verification::VerificationLedger::new(),
            cascade,
        }
    }

    /// Construct a runtime with both an L0 in-memory cache AND a
    /// durable `HistoryLayer`. On startup, the runtime warms the L0
    /// cache from the history layer's existing entries. On answer,
    /// successful tier results are written to both L0 and the
    /// history layer.
    pub fn new_with_history(
        cascade: Cascade,
        mut history: Box<dyn crate::history::HistoryLayer>,
    ) -> Self {
        let _ = &mut history;
        Self {
            l0: Some(crate::l0_cache::L0Cache::new()),
            history: Some(history),
            router: None,
            calibration: crate::calibration::CalibrationReport::default(),
            ledger: crate::verification::VerificationLedger::new(),
            cascade,
        }
    }

    /// Construct a runtime with a router. The router examines each
    /// query and produces a tier dispatch order. Falls back to
    /// registration order when the router returns an empty plan.
    pub fn new_with_router(
        cascade: Cascade,
        router: Box<dyn crate::router::Router>,
    ) -> Self {
        Self {
            l0: Some(crate::l0_cache::L0Cache::new()),
            history: None,
            router: Some(router),
            calibration: crate::calibration::CalibrationReport::default(),
            ledger: crate::verification::VerificationLedger::new(),
            cascade,
        }
    }

    /// Construct a runtime with both history and router. The router
    /// runs before L0; the runtime uses the router's plan to pick tier
    /// order after L0 misses.
    pub fn new_full(
        cascade: Cascade,
        history: Box<dyn crate::history::HistoryLayer>,
        router: Box<dyn crate::router::Router>,
    ) -> Self {
        Self {
            l0: Some(crate::l0_cache::L0Cache::new()),
            history: Some(history),
            router: Some(router),
            calibration: crate::calibration::CalibrationReport::default(),
            ledger: crate::verification::VerificationLedger::new(),
            cascade,
        }
    }

    /// Construct a runtime without an auto-cache. The provided cascade
    /// is walked as-is. Mainly for tests that want to observe what
    /// happens with no caching.
    pub fn new_without_l0(cascade: Cascade) -> Self {
        Self {
            l0: None, history: None, router: None,
            calibration: crate::calibration::CalibrationReport::default(),
            ledger: crate::verification::VerificationLedger::new(),
            cascade,
        }
    }

    /// Calibration report — how honestly each tier estimated its cost.
    pub fn calibration(&self) -> &crate::calibration::CalibrationReport {
        &self.calibration
    }

    /// The verification ledger — tokens for in-flight delayed
    /// dispatches.
    pub fn ledger(&self) -> &crate::verification::VerificationLedger {
        &self.ledger
    }

    /// Issue a verification token. Tiers that return `Pending` answers
    /// call this to mint the token; the runtime tracks
    /// (tier, coord, estimate, initial_actual) until the outcome is
    /// resolved.
    pub fn issue_token(
        &mut self,
        tier: TierId,
        coord: Option<crate::coord::Coord>,
        estimated: f64,
        initial_actual: f64,
    ) -> crate::verification::VerificationToken {
        self.ledger.issue(tier, coord, estimated, initial_actual)
    }

    /// Resolve a previously-issued token. The post-hoc actual cost is
    /// written into per-cell calibration automatically. Returns the
    /// resolved dispatch metadata, or `None` for unknown tokens
    /// (idempotent for duplicate resolves).
    pub fn resolve(
        &mut self,
        token: crate::verification::VerificationToken,
        outcome: &crate::verification::VerificationOutcome,
    ) -> Option<crate::verification::ResolvedDispatch> {
        let resolved = self.ledger.resolve(token, outcome)?;
        let actual = outcome.joules();
        if actual.is_finite() && resolved.initial_estimate > 0.0 {
            if let Some(ref c) = resolved.coord {
                self.calibration.record_with_coord(
                    resolved.tier_id, c,
                    resolved.initial_estimate, actual,
                );
            } else {
                self.calibration.record(
                    resolved.tier_id,
                    resolved.initial_estimate, actual,
                );
            }
        }
        Some(resolved)
    }

    /// The Synthesis coordinate of every registered tier (where the
    /// tier reports one). Exposed for routers, observability, and
    /// coverage analysis.
    pub fn cascade_coords(&self) -> Vec<(TierId, Option<crate::coord::Coord>)> {
        self.cascade.tier_coords()
    }

    /// Reset calibration data (e.g., after a calibration cycle).
    pub fn reset_calibration(&mut self) {
        self.calibration = crate::calibration::CalibrationReport::default();
    }

    /// Borrow the history layer (if any).
    pub fn history(&self) -> Option<&dyn crate::history::HistoryLayer> {
        self.history.as_deref()
    }

    /// Mutably borrow the history layer (if any).
    pub fn history_mut(&mut self) -> Option<&mut (dyn crate::history::HistoryLayer + 'static)> {
        self.history.as_deref_mut()
    }

    /// Number of entries currently in the L0 cache.
    pub fn l0_len(&self) -> usize {
        self.l0.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// L0 stats (hits/misses/writes).
    pub fn l0_stats(&self) -> crate::l0_cache::L0Stats {
        self.l0.as_ref().map(|c| c.stats().clone())
            .unwrap_or_default()
    }

    pub fn tier_ids(&self) -> Vec<TierId> {
        let mut ids = if self.l0.is_some() { vec![TierId::L0] } else { vec![] };
        ids.extend(self.cascade.tier_ids());
        ids
    }

    /// Answer a query by walking the cascade in registered order.
    ///
    /// Order of operations:
    ///   0. Check the auto-L0 cache. Hit → return immediately.
    ///   1. For each registered tier:
    ///      a. Ask the tier to estimate cost. Skip if inapplicable.
    ///      b. Check the estimate against remaining budget. Skip if
    ///         unaffordable.
    ///      c. Check the tier's confidence floor against the query's
    ///         quality floor. Skip if too low.
    ///      d. Check the wall-clock deadline. Fail if missed.
    ///      e. Dispatch to the tier. If it answers, record the answer
    ///         back to L0 and return. If it refuses, record and
    ///         continue.
    ///
    /// If every tier refuses or is unaffordable, return the
    /// appropriate error with the partial trace.
    pub fn answer(&mut self, q: Query) -> Result<Answer, AnswerError> {
        let start = Instant::now();
        let mut spent = 0.0;
        let mut trace = ExecutionTrace::default();
        let mut refusals: Vec<(TierId, RefusalReason)> = Vec::new();
        let mut attempted: Vec<(TierId, f64)> = Vec::new();

        // L0 first.
        if let Some(l0) = self.l0.as_mut() {
            let estimate = l0.estimate_cost(&q).unwrap();
            let remaining = q.budget.remaining(spent);
            if estimate.joules <= remaining {
                let l0_answer = l0.try_answer(&q, remaining)?;
                match &l0_answer.output {
                    AnswerOutput::Refused(_) => {
                        spent += l0_answer.joules_spent;
                        trace.attempts.push(TraceEntry {
                            tier: TierId::L0,
                            outcome: TraceOutcome::Refused(RefusalReason::Inapplicable),
                            joules: l0_answer.joules_spent,
                        });
                    }
                    _ => {
                        // L0 hit. Merge our walk's trace and return.
                        spent += l0_answer.joules_spent;
                        trace.attempts.push(TraceEntry {
                            tier: TierId::L0,
                            outcome: TraceOutcome::Hit,
                            joules: l0_answer.joules_spent,
                        });
                        return Ok(Answer {
                            joules_spent: spent,
                            trace,
                            ..l0_answer
                        });
                    }
                }
            } else {
                trace.attempts.push(TraceEntry {
                    tier: TierId::L0,
                    outcome: TraceOutcome::SkippedBudget,
                    joules: 0.0,
                });
            }
        }

        // L0 missed (or doesn't exist). Try the durable history layer.
        // A history hit warms L0 for next time, then returns.
        if let Some(history) = self.history.as_mut() {
            let key = crate::history::key_for(&q);
            let history_cost = history.estimate_lookup_cost(&q);
            let remaining = q.budget.remaining(spent);
            if history_cost <= remaining {
                match history.lookup_exact(&key) {
                    Ok(Some(ha)) => {
                        spent += history_cost;
                        let answer = Answer {
                            output: ha.output,
                            tier_used: TierId::L0,   // surfaced as L0 — the cache hit, durable
                            joules_spent: spent,
                            confidence: ha.confidence,
                            trace: {
                                let mut t = trace.clone();
                                t.attempts.push(TraceEntry {
                                    tier: TierId::L0,
                                    outcome: TraceOutcome::Hit,
                                    joules: history_cost,
                                });
                                t
                            },
                            verification: crate::verification::VerificationStatus::Resolved,
                        };
                        // Warm L0 with this answer for next time.
                        if let Some(l0) = self.l0.as_mut() {
                            l0.put(&q, &answer);
                        }
                        return Ok(answer);
                    }
                    Ok(None) => { /* history miss, fall through */ }
                    Err(e) => return Err(AnswerError::TierFailed {
                        tier: TierId::L0,
                        cause: format!("history: {}", e),
                    }),
                }
            }
        }

        // Consult the router (if any) to pick a tier dispatch order.
        // The router's joule cost is counted against the query budget.
        let walk_indices: Vec<usize> = if let Some(router) = self.router.as_ref() {
            // Build the tier-coord list for coordinate-aware routers.
            let tier_coords: Vec<(TierId, crate::coord::Coord)> =
                self.cascade.tiers.iter()
                    .filter_map(|t| t.coord().map(|c| (t.id(), c)))
                    .collect();
            let plan = if tier_coords.is_empty() {
                router.route(&q)
            } else {
                router.route_with_coords(&q, &tier_coords)
            };
            spent += plan.router_joules;
            if !plan.is_fallback() {
                // Map router's TierId list to cascade indices. Tiers
                // not in the cascade are ignored.
                let mut indices = Vec::with_capacity(plan.tier_order.len());
                for tid in &plan.tier_order {
                    if let Some(idx) = self.cascade.tiers.iter().position(|t| t.id() == *tid) {
                        indices.push(idx);
                    }
                }
                indices
            } else {
                (0..self.cascade.tiers.len()).collect()
            }
        } else {
            (0..self.cascade.tiers.len()).collect()
        };

        for &idx in &walk_indices {
            let tier = &mut self.cascade.tiers[idx];
            let tier_id = tier.id();
            let tier_coord = tier.coord();

            // Deadline check.
            if let Some(deadline) = q.deadline {
                let elapsed = start.elapsed();
                if elapsed > deadline {
                    return Err(AnswerError::DeadlineExceeded {
                        elapsed, deadline,
                    });
                }
            }

            // Cost estimate. None → inapplicable.
            let mut estimate = match tier.estimate_cost(&q) {
                Some(e) => e,
                None => {
                    trace.attempts.push(TraceEntry {
                        tier: tier_id,
                        outcome: TraceOutcome::SkippedInapplicable,
                        joules: 0.0,
                    });
                    continue;
                }
            };

            // Closed calibration loop. Keep the *raw* estimate for
            // calibration recording (so learned μ reflects the tier's
            // intrinsic estimate↔actual bias and reaches a stable
            // fixed point), but use a *corrected* estimate for the
            // budget/routing decision. `learned_mu` is 1.0 until ≥3
            // samples accrue, so cold-start is unperturbed.
            let raw_estimate_joules = estimate.joules;
            if let Some(ref c) = tier_coord {
                let mu = self.calibration.learned_mu(c);
                if mu != 1.0 {
                    estimate.joules *= mu;
                }
            }

            // Budget check.
            let remaining = q.budget.remaining(spent);
            if estimate.joules > remaining {
                trace.attempts.push(TraceEntry {
                    tier: tier_id,
                    outcome: TraceOutcome::SkippedBudget,
                    joules: 0.0,
                });
                continue;
            }

            // Quality floor check.
            if estimate.confidence_floor < q.quality.min_confidence {
                trace.attempts.push(TraceEntry {
                    tier: tier_id,
                    outcome: TraceOutcome::SkippedQuality,
                    joules: 0.0,
                });
                continue;
            }

            // Dispatch.
            let dispatch_result = tier.try_answer(&q, remaining);
            // Record calibration data on every dispatch outcome.
            // The `actual` is the tier's self-reported spend; we charge
            // it to the budget regardless of outcome.
            match dispatch_result {
                Ok(answer) => {
                    let actual = answer.joules_spent;
                    if let Some(ref c) = tier_coord {
                        self.calibration.record_with_coord(
                            tier_id, c, raw_estimate_joules, actual);
                    } else {
                        self.calibration.record(tier_id, raw_estimate_joules, actual);
                    }

                    // Budget violation: tier overshot the budget cap it
                    // was given. Treat as a hard error rather than
                    // silently moving on — this catches dishonest
                    // tiers and keeps the runtime's joule contract.
                    if actual > remaining {
                        self.calibration.record_violation(tier_id);
                        spent += actual;
                        attempted.push((tier_id, actual));
                        trace.attempts.push(TraceEntry {
                            tier: tier_id,
                            outcome: TraceOutcome::SkippedBudget,
                            joules: actual,
                        });
                        return Err(AnswerError::BudgetExhausted {
                            spent, limit: q.budget.hard_limit,
                            attempted_tiers: attempted,
                        });
                    }

                    match &answer.output {
                        AnswerOutput::Refused(reason) => {
                            spent += actual;
                            attempted.push((tier_id, actual));
                            refusals.push((tier_id, reason.clone()));
                            trace.attempts.push(TraceEntry {
                                tier: tier_id,
                                outcome: TraceOutcome::Refused(reason.clone()),
                                joules: actual,
                            });
                        }
                        _ => {
                            spent += actual;
                            trace.attempts.push(TraceEntry {
                                tier: tier_id,
                                outcome: TraceOutcome::Hit,
                                joules: actual,
                            });
                            let final_answer = Answer {
                                joules_spent: spent,
                                trace,
                                ..answer
                            };
                            // Write back to L0 for next time.
                            if let Some(l0) = self.l0.as_mut() {
                                l0.put(&q, &final_answer);
                            }
                            // Write to the durable history layer too.
                            if let Some(history) = self.history.as_mut() {
                                let _ = history.record(&q, &final_answer);
                            }
                            return Ok(final_answer);
                        }
                    }
                }
                Err(AnswerError::BudgetExhausted { spent: tier_spent, .. }) => {
                    spent += tier_spent;
                    attempted.push((tier_id, tier_spent));
                    trace.attempts.push(TraceEntry {
                        tier: tier_id,
                        outcome: TraceOutcome::SkippedBudget,
                        joules: tier_spent,
                    });
                }
                Err(e) => return Err(e),
            }
        }

        // Every tier either refused, errored, or was unaffordable.
        if !refusals.is_empty() {
            Err(AnswerError::NoTierSatisfied { refusals })
        } else {
            Err(AnswerError::BudgetExhausted {
                spent,
                limit: q.budget.hard_limit,
                attempted_tiers: attempted,
            })
        }
    }
}
