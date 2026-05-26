//! Delayed verification. Some answers can't be checked at decision
//! time — they have to play out. This module is the mechanism by
//! which the runtime represents and tracks those.
//!
//! Three concepts:
//!
//!   * `VerificationStatus` — attached to every `Answer`. Most answers
//!     are `Resolved` (checked at decision time); some are
//!     `Pending(token)` and resolve later.
//!
//!   * `VerificationToken` — opaque ID the runtime hands out so the
//!     caller can later report the outcome.
//!
//!   * `VerificationOutcome` — what eventually came back. Success,
//!     failure, or timeout. Optionally carries the *actual* joule
//!     cost of the eventual effect (e.g., the model said the
//!     deployment would succeed; the deployment actually took 12s
//!     and cost 8 mJ of cloud compute).
//!
//! The runtime keeps a `VerificationLedger` mapping tokens to
//! pending dispatches. When `resolve(token, outcome)` is called, the
//! ledger looks up the original (tier_id, coord, initial_estimate)
//! and records the verification outcome into calibration.
//!
//! Why this matters for the cascade: a tier whose outputs are
//! verified asynchronously can't be calibrated in real-time. Without
//! this layer, an L4 tier that says "the email is drafted" would be
//! treated as instantly verified — but the *actual* verification
//! ("did the recipient open the email and reply") only resolves
//! hours later. Joule needs to attribute that eventual outcome to
//! the original dispatch.

use crate::coord::Coord;
use crate::types::TierId;
use std::collections::HashMap;

/// Opaque token identifying a pending verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VerificationToken(pub u64);

/// What the verification eventually said.
#[derive(Debug, Clone)]
pub enum VerificationOutcome {
    /// The action succeeded. `actual_joules` is the true post-hoc
    /// cost (which may differ from the tier's estimate).
    Success { actual_joules: f64 },
    /// The action failed. `actual_joules` is what was actually spent.
    Failure { actual_joules: f64, reason: String },
    /// Verification window expired without a result. Penalize as a
    /// soft failure: the answer might have been right, but we can't
    /// know.
    Timeout,
}

impl VerificationOutcome {
    /// Joules to credit to the dispatch, for calibration purposes.
    pub fn joules(&self) -> f64 {
        match self {
            Self::Success { actual_joules } => *actual_joules,
            Self::Failure { actual_joules, .. } => *actual_joules,
            Self::Timeout => f64::NAN,
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }
}

/// Attached to every `Answer`. The verification status tells the
/// caller whether the answer is final or whether its correctness
/// will land later.
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    /// Verification complete at decision time. The answer's
    /// confidence is final.
    Resolved,
    /// Verification pending. The caller may later submit an outcome
    /// via `Runtime::resolve(token, ...)`. Until then, the answer's
    /// reported `joules_spent` is the tier's estimate; the real cost
    /// will land with the outcome.
    Pending(VerificationToken),
}

impl Default for VerificationStatus {
    fn default() -> Self { Self::Resolved }
}

/// Record of a pending dispatch, kept until the verification lands.
#[derive(Debug, Clone)]
struct PendingDispatch {
    tier_id: TierId,
    coord: Option<Coord>,
    initial_estimate: f64,
    initial_actual: f64,    // what the tier reported at dispatch
}

/// The runtime's ledger of pending verifications.
#[derive(Debug, Default)]
pub struct VerificationLedger {
    pending: HashMap<VerificationToken, PendingDispatch>,
    next_token: u64,
    /// Total verifications ever issued, completed, timed out.
    pub issued: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub timed_out: u64,
}

impl VerificationLedger {
    pub fn new() -> Self { Self::default() }

    /// Issue a new pending verification. The runtime keeps the
    /// dispatch metadata; the token is what the caller returns when
    /// the outcome lands.
    pub fn issue(
        &mut self,
        tier_id: TierId,
        coord: Option<Coord>,
        initial_estimate: f64,
        initial_actual: f64,
    ) -> VerificationToken {
        self.next_token += 1;
        let token = VerificationToken(self.next_token);
        self.pending.insert(token, PendingDispatch {
            tier_id, coord, initial_estimate, initial_actual,
        });
        self.issued += 1;
        token
    }

    /// Resolve a pending verification. Returns the dispatch metadata
    /// so the runtime can update calibration. Returns None if the
    /// token is unknown (already resolved, or was never issued).
    pub fn resolve(
        &mut self,
        token: VerificationToken,
        outcome: &VerificationOutcome,
    ) -> Option<ResolvedDispatch> {
        let pending = self.pending.remove(&token)?;
        match outcome {
            VerificationOutcome::Success { .. } => self.succeeded += 1,
            VerificationOutcome::Failure { .. } => self.failed += 1,
            VerificationOutcome::Timeout => self.timed_out += 1,
        }
        Some(ResolvedDispatch {
            tier_id: pending.tier_id,
            coord: pending.coord,
            initial_estimate: pending.initial_estimate,
            initial_actual: pending.initial_actual,
        })
    }

    /// Number of verifications currently pending.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// All currently-pending tokens. Useful for diagnostic snapshots.
    pub fn pending_tokens(&self) -> Vec<VerificationToken> {
        let mut tokens: Vec<_> = self.pending.keys().copied().collect();
        tokens.sort();
        tokens
    }
}

/// What the ledger returns when a verification resolves. The runtime
/// uses this to update calibration with the post-hoc actual cost.
#[derive(Debug, Clone)]
pub struct ResolvedDispatch {
    pub tier_id: TierId,
    pub coord: Option<Coord>,
    pub initial_estimate: f64,
    pub initial_actual: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{L1Primitive, L4ModelId};

    #[test]
    fn issue_resolve_lifecycle() {
        let mut ledger = VerificationLedger::new();
        let token = ledger.issue(
            TierId::L4(L4ModelId(0)), None, 0.5, 0.5);
        assert_eq!(ledger.pending_count(), 1);
        assert_eq!(ledger.issued, 1);

        let resolved = ledger.resolve(
            token, &VerificationOutcome::Success { actual_joules: 0.7 });
        assert!(resolved.is_some());
        assert_eq!(ledger.pending_count(), 0);
        assert_eq!(ledger.succeeded, 1);

        let r = resolved.unwrap();
        assert_eq!(r.initial_estimate, 0.5);
        assert_eq!(r.initial_actual, 0.5);
    }

    #[test]
    fn resolve_unknown_token_returns_none() {
        let mut ledger = VerificationLedger::new();
        let bogus = VerificationToken(999);
        let r = ledger.resolve(bogus,
            &VerificationOutcome::Success { actual_joules: 0.0 });
        assert!(r.is_none());
    }

    #[test]
    fn double_resolve_is_safe() {
        let mut ledger = VerificationLedger::new();
        let token = ledger.issue(
            TierId::L1(L1Primitive::Execute), None, 1e-9, 1e-9);
        let _ = ledger.resolve(token,
            &VerificationOutcome::Success { actual_joules: 1e-9 });
        // Second resolve is a no-op.
        let r2 = ledger.resolve(token,
            &VerificationOutcome::Success { actual_joules: 1e-9 });
        assert!(r2.is_none());
    }

    #[test]
    fn failure_records_actual_joules() {
        let outcome = VerificationOutcome::Failure {
            actual_joules: 0.123,
            reason: "remote service returned 500".into(),
        };
        assert_eq!(outcome.joules(), 0.123);
        assert!(!outcome.is_success());
    }

    #[test]
    fn timeout_outcome_returns_nan_joules() {
        let o = VerificationOutcome::Timeout;
        assert!(o.joules().is_nan());
        assert!(!o.is_success());
    }

    #[test]
    fn ledger_counts_outcomes_separately() {
        let mut ledger = VerificationLedger::new();
        let t1 = ledger.issue(TierId::L0, None, 1e-9, 1e-9);
        let t2 = ledger.issue(TierId::L0, None, 1e-9, 1e-9);
        let t3 = ledger.issue(TierId::L0, None, 1e-9, 1e-9);

        ledger.resolve(t1, &VerificationOutcome::Success { actual_joules: 1e-9 });
        ledger.resolve(t2, &VerificationOutcome::Failure {
            actual_joules: 1e-9, reason: "x".into() });
        ledger.resolve(t3, &VerificationOutcome::Timeout);

        assert_eq!(ledger.succeeded, 1);
        assert_eq!(ledger.failed, 1);
        assert_eq!(ledger.timed_out, 1);
        assert_eq!(ledger.pending_count(), 0);
    }
}
