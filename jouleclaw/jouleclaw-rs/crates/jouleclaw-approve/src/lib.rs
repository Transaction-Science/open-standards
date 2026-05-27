//! Approval gates — pause a cascade dispatch for review before it runs.
//!
//! Trigger.dev lets you "pause your task until a human can approve, reject
//! or give feedback." JouleClaw ingests that, energy-shaped: gating is
//! most valuable in front of the **statistical compartment** (the L3/L4
//! model tiers), and least valuable in front of cheap deterministic work.
//! So the canonical gate, [`JouleThresholdGate`], only escalates to a
//! reviewer when the dispatch's estimated joules exceed a threshold —
//! humans review the expensive (yang) calls, the cheap (yin) ones never
//! ask. Energy discipline made structural.
//!
//! The gate is a trait, so "review" can be anything: a CLI prompt, a
//! Slack round-trip, an automated policy, or a headless [`AutoApprove`].
//! [`GatedTier`] wraps any [`Tier`] so the gate is consulted before the
//! inner tier runs: approve → delegate, reject → refuse (the cascade
//! falls through to a cheaper tier, or fails if none can answer).

#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex};

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;

/// What a gate is being asked to approve.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// Human-readable action, e.g. `"dispatch Model"`.
    pub action: String,
    /// The tier whose dispatch is gated.
    pub tier: TierId,
    /// The dispatch's estimated joule cost (the reason gating matters).
    pub estimated_joules: f64,
}

/// A gate's verdict.
#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    /// Proceed with the dispatch.
    Approve,
    /// Block the dispatch; the cascade treats it as a refusal.
    Reject { reason: String },
}

impl ApprovalDecision {
    pub fn is_approved(&self) -> bool {
        matches!(self, ApprovalDecision::Approve)
    }
}

/// Reviews gated dispatches. Implementations MUST be `Send + Sync` so a
/// gate can be shared across a long-lived runtime.
pub trait ApprovalGate: Send + Sync {
    fn review(&self, request: &ApprovalRequest) -> ApprovalDecision;
    fn name(&self) -> &str {
        "gate"
    }
}

/// Headless default: approve everything (no gating overhead).
pub struct AutoApprove;
impl ApprovalGate for AutoApprove {
    fn review(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Approve
    }
    fn name(&self) -> &str {
        "auto-approve"
    }
}

/// Lockdown: reject everything (e.g. a kill-switch for the compartment).
pub struct DenyAll;
impl ApprovalGate for DenyAll {
    fn review(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Reject {
            reason: "all dispatches denied".into(),
        }
    }
    fn name(&self) -> &str {
        "deny-all"
    }
}

/// Energy-shaped gate: auto-approve any dispatch estimated **at or below**
/// `threshold_joules`; escalate the rest to `inner`. The cheap
/// deterministic floor never asks for approval; only the expensive model
/// calls do.
pub struct JouleThresholdGate {
    threshold_joules: f64,
    inner: Arc<dyn ApprovalGate>,
}

impl JouleThresholdGate {
    pub fn new(threshold_joules: f64, inner: Arc<dyn ApprovalGate>) -> Self {
        Self {
            threshold_joules,
            inner,
        }
    }
}

impl ApprovalGate for JouleThresholdGate {
    fn review(&self, request: &ApprovalRequest) -> ApprovalDecision {
        if request.estimated_joules <= self.threshold_joules {
            ApprovalDecision::Approve
        } else {
            self.inner.review(request)
        }
    }
    fn name(&self) -> &str {
        "joule-threshold"
    }
}

/// Records every request it sees (for audit), delegating the decision to
/// an inner gate.
pub struct RecordingGate {
    inner: Arc<dyn ApprovalGate>,
    log: Mutex<Vec<ApprovalRequest>>,
}

impl RecordingGate {
    pub fn new(inner: Arc<dyn ApprovalGate>) -> Self {
        Self {
            inner,
            log: Mutex::new(Vec::new()),
        }
    }

    /// Snapshot of the requests reviewed so far.
    pub fn log(&self) -> Vec<ApprovalRequest> {
        self.log.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

impl ApprovalGate for RecordingGate {
    fn review(&self, request: &ApprovalRequest) -> ApprovalDecision {
        if let Ok(mut l) = self.log.lock() {
            l.push(request.clone());
        }
        self.inner.review(request)
    }
    fn name(&self) -> &str {
        "recording"
    }
}

/// A swappable, shared gate. The boxed gate can be replaced at runtime
/// (e.g. engage a lockdown) without rebuilding the cascade.
pub type SharedGate = Arc<Mutex<Box<dyn ApprovalGate>>>;

/// Wrap a gate into a [`SharedGate`].
pub fn shared_gate(gate: impl ApprovalGate + 'static) -> SharedGate {
    Arc::new(Mutex::new(Box::new(gate)))
}

/// Swap the active gate inside a [`SharedGate`] in place.
pub fn set_gate(shared: &SharedGate, gate: impl ApprovalGate + 'static) {
    if let Ok(mut g) = shared.lock() {
        *g = Box::new(gate);
    }
}

/// Wraps any [`Tier`] so its dispatch is gated: the gate reviews a
/// request built from the inner tier's own cost estimate, and the inner
/// tier runs only on approval. On rejection the wrapped tier refuses, so
/// the cascade continues to a cheaper tier (or fails if none can answer).
pub struct GatedTier<T: Tier> {
    inner: T,
    gate: SharedGate,
}

impl<T: Tier> GatedTier<T> {
    pub fn new(inner: T, gate: SharedGate) -> Self {
        Self { inner, gate }
    }
}

impl<T: Tier> Tier for GatedTier<T> {
    fn id(&self) -> TierId {
        self.inner.id()
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        self.inner.estimate_cost(q)
    }

    fn try_answer(&mut self, q: &Query, budget_remaining: f64) -> Result<Answer, AnswerError> {
        let id = self.inner.id();
        let estimated_joules = self
            .inner
            .estimate_cost(q)
            .map(|e| e.joules)
            .unwrap_or(0.0);
        let request = ApprovalRequest {
            action: format!("dispatch {}", id.name()),
            tier: id,
            estimated_joules,
        };
        let decision = match self.gate.lock() {
            Ok(gate) => gate.review(&request),
            Err(_) => ApprovalDecision::Reject {
                reason: "approval gate poisoned".into(),
            },
        };
        match decision {
            ApprovalDecision::Approve => self.inner.try_answer(q, budget_remaining),
            ApprovalDecision::Reject { reason } => Ok(Answer {
                output: AnswerOutput::Refused(RefusalReason::TierSpecific(format!(
                    "approval denied: {reason}"
                ))),
                tier_used: id,
                joules_spent: 0.0,
                confidence: 0.0,
                trace: ExecutionTrace::default(),
                verification: VerificationStatus::Resolved,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::tier::TierEstimate;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, L3ModelId, QualityFloor, QueryInput,
    };
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    fn q() -> Query {
        Query {
            input: QueryInput::Text("hi".into()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn req(joules: f64) -> ApprovalRequest {
        ApprovalRequest {
            action: "dispatch Model".into(),
            tier: TierId::L3(L3ModelId(0)),
            estimated_joules: joules,
        }
    }

    /// A tier that counts how many times it actually ran, returning a
    /// fixed answer at a fixed estimated cost.
    struct CountingTier {
        calls: Arc<AtomicU32>,
        joules: f64,
    }
    impl Tier for CountingTier {
        fn id(&self) -> TierId {
            TierId::L3(L3ModelId(0))
        }
        fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
            Some(TierEstimate {
                joules: self.joules,
                latency: Duration::from_millis(1),
                confidence_floor: 0.5,
            })
        }
        fn try_answer(&mut self, _q: &Query, _b: f64) -> Result<Answer, AnswerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Answer {
                output: AnswerOutput::Text("model answer".into()),
                tier_used: TierId::L3(L3ModelId(0)),
                joules_spent: self.joules,
                confidence: 0.8,
                trace: ExecutionTrace::default(),
                verification: VerificationStatus::Resolved,
            })
        }
    }

    #[test]
    fn auto_approve_and_deny_all() {
        assert!(AutoApprove.review(&req(1.0)).is_approved());
        assert!(!DenyAll.review(&req(1.0)).is_approved());
    }

    #[test]
    fn threshold_gates_only_expensive() {
        let gate = JouleThresholdGate::new(1.0, Arc::new(DenyAll));
        // Cheap: auto-approved even though inner is DenyAll.
        assert!(gate.review(&req(0.5)).is_approved());
        assert!(gate.review(&req(1.0)).is_approved()); // at threshold
        // Expensive: escalates to the (denying) inner gate.
        assert!(!gate.review(&req(2.0)).is_approved());
    }

    #[test]
    fn recording_gate_logs_and_delegates() {
        let g = RecordingGate::new(Arc::new(AutoApprove));
        assert!(g.review(&req(5.0)).is_approved());
        assert!(g.review(&req(9.0)).is_approved());
        let log = g.log();
        assert_eq!(log.len(), 2);
        assert!((log[1].estimated_joules - 9.0).abs() < 1e-9);
    }

    #[test]
    fn gated_tier_approve_delegates() {
        let calls = Arc::new(AtomicU32::new(0));
        let mut t = GatedTier::new(
            CountingTier {
                calls: calls.clone(),
                joules: 2.0,
            },
            shared_gate(AutoApprove),
        );
        let ans = t.try_answer(&q(), 100.0).unwrap();
        assert!(matches!(ans.output, AnswerOutput::Text(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 1); // inner ran
    }

    #[test]
    fn gated_tier_reject_refuses_without_running_inner() {
        let calls = Arc::new(AtomicU32::new(0));
        let mut t = GatedTier::new(
            CountingTier {
                calls: calls.clone(),
                joules: 2.0,
            },
            shared_gate(DenyAll),
        );
        let ans = t.try_answer(&q(), 100.0).unwrap();
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::TierSpecific(_))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0); // inner NEVER ran — no joules spent
        assert_eq!(ans.joules_spent, 0.0);
    }

    #[test]
    fn gate_can_be_swapped_at_runtime() {
        let calls = Arc::new(AtomicU32::new(0));
        let shared = shared_gate(AutoApprove);
        let mut t = GatedTier::new(
            CountingTier {
                calls: calls.clone(),
                joules: 2.0,
            },
            shared.clone(),
        );
        assert!(matches!(t.try_answer(&q(), 100.0).unwrap().output, AnswerOutput::Text(_)));
        // Engage lockdown.
        set_gate(&shared, DenyAll);
        assert!(matches!(
            t.try_answer(&q(), 100.0).unwrap().output,
            AnswerOutput::Refused(_)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1); // only the first ran
    }
}
