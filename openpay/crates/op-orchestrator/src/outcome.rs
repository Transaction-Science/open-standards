//! Orchestration outcome.
//!
//! What the orchestrator returns to the caller: a unified outcome
//! that abstracts over which rail handled the request and how many
//! attempts it took. The merchant code doesn't have to care whether
//! the payment was routed through a card network or an instant rail.

use op_core::RailKind;
use serde::{Deserialize, Serialize};

/// Per-attempt outcome inside a multi-attempt orchestration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    /// Which rail was attempted.
    pub rail: RailKind,

    /// Driver name for telemetry (e.g. `"hyperswitch"`, `"fednow"`).
    pub driver: String,

    /// Outcome of this single attempt.
    pub outcome: AttemptOutcome,
}

/// Outcome of a single rail attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptOutcome {
    /// Rail returned a positive terminal status (approved, settled).
    Success,

    /// Rail returned a negative terminal status — customer-side
    /// problem (insufficient funds, expired card, frozen account).
    /// Orchestrator will NOT retry to another rail; the customer
    /// must act.
    HardDecline {
        /// Short normalized code (e.g. `"insufficient_funds"`).
        code: String,
    },

    /// Rail returned a soft / transient failure (network timeout,
    /// PSP 5xx). Orchestrator will retry or fall back.
    SoftFailure {
        /// Short normalized code.
        code: String,
    },

    /// Rail says the customer needs to do something (3DS challenge,
    /// bank app redirect). Orchestrator surfaces this and stops —
    /// the caller resumes once the customer completes the action.
    RequiresAction {
        /// URL or deep-link the customer must navigate to.
        url: String,
    },
}

/// Final terminal state of the orchestration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalStatus {
    /// Funds authorized / captured / settled depending on the rail
    /// and the merchant's auto-capture flag.
    Approved,

    /// Customer action needed (3DS, bank app, OTP). Caller pauses
    /// and resumes once the customer completes.
    RequiresCustomerAction,

    /// All rails declined or exhausted; payment failed terminally.
    Declined,
}

/// The orchestrator's return value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrchestrationOutcome {
    /// Final terminal state.
    pub terminal_status: TerminalStatus,

    /// One entry per rail attempt, in order.
    pub attempts: Vec<Attempt>,

    /// Which rail produced the terminal status (`None` if every rail
    /// failed). Convenient even though `attempts.last().rail` would
    /// be equivalent — explicit field makes downstream telemetry
    /// pipelines simpler.
    pub rail_used: Option<RailKind>,

    /// For card auths: the PSP's payment id. Hold this for
    /// capture/void/refund.
    pub psp_payment_id: Option<String>,

    /// For A2A: the UETR. Match settlement notifications back to
    /// the original payment.
    pub uetr: Option<String>,
}

impl OrchestrationOutcome {
    /// Convenience: was the payment approved?
    pub fn is_approved(&self) -> bool {
        self.terminal_status == TerminalStatus::Approved
    }

    /// Convenience: does the customer have something to do?
    pub fn requires_customer_action(&self) -> bool {
        self.terminal_status == TerminalStatus::RequiresCustomerAction
    }

    /// Convenience: was the payment declined terminally?
    pub fn is_declined(&self) -> bool {
        self.terminal_status == TerminalStatus::Declined
    }

    /// Number of attempts made.
    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approved() -> OrchestrationOutcome {
        OrchestrationOutcome {
            terminal_status: TerminalStatus::Approved,
            attempts: vec![Attempt {
                rail: RailKind::Card,
                driver: "hyperswitch".into(),
                outcome: AttemptOutcome::Success,
            }],
            rail_used: Some(RailKind::Card),
            psp_payment_id: Some("psp_test_1".into()),
            uetr: None,
        }
    }

    #[test]
    fn convenience_predicates_for_approved() {
        let o = approved();
        assert!(o.is_approved());
        assert!(!o.requires_customer_action());
        assert!(!o.is_declined());
        assert_eq!(o.attempt_count(), 1);
    }

    #[test]
    fn convenience_predicates_for_declined() {
        let o = OrchestrationOutcome {
            terminal_status: TerminalStatus::Declined,
            attempts: vec![],
            rail_used: None,
            psp_payment_id: None,
            uetr: None,
        };
        assert!(!o.is_approved());
        assert!(o.is_declined());
        assert_eq!(o.attempt_count(), 0);
    }

    #[test]
    fn convenience_predicates_for_action() {
        let o = OrchestrationOutcome {
            terminal_status: TerminalStatus::RequiresCustomerAction,
            attempts: vec![Attempt {
                rail: RailKind::Card,
                driver: "hyperswitch".into(),
                outcome: AttemptOutcome::RequiresAction {
                    url: "https://3ds.example/c".into(),
                },
            }],
            rail_used: Some(RailKind::Card),
            psp_payment_id: Some("psp_test_2".into()),
            uetr: None,
        };
        assert!(!o.is_approved());
        assert!(o.requires_customer_action());
        assert!(!o.is_declined());
    }

    #[test]
    fn attempt_outcome_equality_for_pattern_matching() {
        // The AttemptOutcome variants need to compare for equality
        // so test code can assert exact attempt sequences.
        assert_eq!(AttemptOutcome::Success, AttemptOutcome::Success);
        assert_ne!(
            AttemptOutcome::Success,
            AttemptOutcome::HardDecline { code: "x".into() },
        );
    }
}
