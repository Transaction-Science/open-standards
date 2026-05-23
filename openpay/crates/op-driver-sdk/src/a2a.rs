//! Deterministic mock A2A gateway.
//!
//! [`DeterministicA2aGateway`] is to [`A2aAcquirer`] what
//! [`crate::DeterministicCardAcquirer`] is to `CardAcquirer`: a
//! programmable side-effect-free reference impl. UETR-keyed
//! overrides, amount thresholds, transport-error mode, request
//! history.

use std::sync::Mutex;

use op_core::Money;
use op_rails_a2a::acquirer::{A2aDecision, A2aStatus, CreditTransferReq, StatusQueryReq};
use op_rails_a2a::{A2aAcquirer, Error, Result};
use serde::{Deserialize, Serialize};

/// Programmable A2A gateway mock.
#[derive(Default)]
pub struct DeterministicA2aGateway {
    name: &'static str,
    policy: Mutex<Policy>,
    history: Mutex<History>,
}

#[derive(Default)]
struct Policy {
    default_status: Option<A2aStatus>,
    uetr_overrides: Vec<(String, A2aStatus, Option<String>)>,
    amount_rules: Vec<AmountRule>,
    transport_error: Option<String>,
    next_rail_seq: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct AmountRule {
    op: Comparator,
    threshold_minor: i64,
    currency: String,
    status: A2aStatus,
    reason_code: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Comparator {
    Ge,
    Gt,
    Le,
    Lt,
    Eq,
}

#[derive(Default)]
struct History {
    transfers: Vec<CreditTransferReq>,
    queries: Vec<StatusQueryReq>,
}

impl DeterministicA2aGateway {
    /// Fresh gateway named `"deterministic"`. Defaults to
    /// [`A2aStatus::Settled`] on every input.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "deterministic",
            policy: Mutex::default(),
            history: Mutex::default(),
        }
    }

    /// Builder: rename the gateway. Useful when running multiple
    /// mocks in one test (each can pretend to be a different rail).
    #[must_use]
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    /// Builder: set the default status.
    #[must_use]
    pub fn with_default_status(self, status: A2aStatus) -> Self {
        self.policy.lock().expect("poisoned").default_status = Some(status);
        self
    }

    /// Builder: force a UETR to return `status`.
    #[must_use]
    pub fn with_uetr_override(
        self,
        uetr: impl Into<String>,
        status: A2aStatus,
        reason_code: Option<String>,
    ) -> Self {
        self.policy.lock().expect("poisoned").uetr_overrides.push((
            uetr.into(),
            status,
            reason_code,
        ));
        self
    }

    /// Builder: amount-based rule (`>= threshold`).
    #[must_use]
    pub fn with_amount_ge(
        self,
        threshold: Money,
        status: A2aStatus,
        reason_code: Option<String>,
    ) -> Self {
        self.policy
            .lock()
            .expect("poisoned")
            .amount_rules
            .push(AmountRule {
                op: Comparator::Ge,
                threshold_minor: threshold.minor_units,
                currency: threshold.currency.code().to_owned(),
                status,
                reason_code,
            });
        self
    }

    /// Builder: amount-based rule (`< threshold`).
    #[must_use]
    pub fn with_amount_lt(
        self,
        threshold: Money,
        status: A2aStatus,
        reason_code: Option<String>,
    ) -> Self {
        self.policy
            .lock()
            .expect("poisoned")
            .amount_rules
            .push(AmountRule {
                op: Comparator::Lt,
                threshold_minor: threshold.minor_units,
                currency: threshold.currency.code().to_owned(),
                status,
                reason_code,
            });
        self
    }

    /// Builder: every call returns `Err(Error::Transport(_))`.
    #[must_use]
    pub fn with_transport_error(self, message: impl Into<String>) -> Self {
        self.policy.lock().expect("poisoned").transport_error = Some(message.into());
        self
    }

    /// All credit transfers submitted since construction.
    #[must_use]
    pub fn transfer_history(&self) -> Vec<CreditTransferReq> {
        self.history.lock().expect("poisoned").transfers.clone()
    }

    /// All status queries seen.
    #[must_use]
    pub fn query_history(&self) -> Vec<StatusQueryReq> {
        self.history.lock().expect("poisoned").queries.clone()
    }

    fn next_rail_id(&self) -> String {
        let mut p = self.policy.lock().expect("poisoned");
        p.next_rail_seq = p.next_rail_seq.saturating_add(1);
        format!("rail_det_{:010}", p.next_rail_seq)
    }

    fn resolve_status(&self, uetr: &str, amount: Money) -> (A2aStatus, Option<String>) {
        let p = self.policy.lock().expect("poisoned");
        for (k, status, code) in &p.uetr_overrides {
            if k == uetr {
                return (*status, code.clone());
            }
        }
        for rule in &p.amount_rules {
            if rule.currency != amount.currency.code() {
                continue;
            }
            let m = amount.minor_units;
            let matched = match rule.op {
                Comparator::Ge => m >= rule.threshold_minor,
                Comparator::Gt => m > rule.threshold_minor,
                Comparator::Le => m <= rule.threshold_minor,
                Comparator::Lt => m < rule.threshold_minor,
                Comparator::Eq => m == rule.threshold_minor,
            };
            if matched {
                return (rule.status, rule.reason_code.clone());
            }
        }
        (p.default_status.unwrap_or(A2aStatus::Settled), None)
    }
}

impl A2aAcquirer for DeterministicA2aGateway {
    fn name(&self) -> &'static str {
        self.name
    }

    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> Result<A2aDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .transfers
            .push(req.clone());
        if let Some(msg) = self
            .policy
            .lock()
            .expect("poisoned")
            .transport_error
            .clone()
        {
            return Err(Error::Transport(msg));
        }
        let (status, code) = self.resolve_status(&req.uetr, req.amount);
        Ok(A2aDecision {
            status,
            raw_status: format!("{status:?}").to_lowercase(),
            reason_code: code,
            reason_text: None,
            uetr: Some(req.uetr.clone()),
            rail_txn_id: Some(self.next_rail_id()),
            settled_amount: matches!(status, A2aStatus::Settled | A2aStatus::Accepted)
                .then_some(req.amount),
        })
    }

    fn query_status(&self, req: &StatusQueryReq) -> Result<A2aDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .queries
            .push(req.clone());
        let (status, code) =
            self.resolve_status(&req.uetr, Money::from_minor(0, op_core::Currency::USD));
        Ok(A2aDecision {
            status,
            raw_status: format!("{status:?}").to_lowercase(),
            reason_code: code,
            reason_text: None,
            uetr: Some(req.uetr.clone()),
            rail_txn_id: None,
            settled_amount: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};
    use op_rails_a2a::acquirer::ParticipantId;

    fn transfer(uetr: &str, amount_minor: i64) -> CreditTransferReq {
        CreditTransferReq {
            uetr: uetr.into(),
            end_to_end_id: "e2e-1".into(),
            amount: Money::from_minor(amount_minor, Currency::USD),
            debtor_agent: ParticipantId::Aba("121000248".into()),
            creditor_agent: ParticipantId::Aba("021000021".into()),
            debtor_account: "11111".into(),
            creditor_account: "22222".into(),
            debtor_name: "ACME".into(),
            creditor_name: "BENE".into(),
            remittance: None,
            idempotency_key: "k".into(),
        }
    }

    #[test]
    fn default_settled() {
        let g = DeterministicA2aGateway::new();
        let d = g.submit_credit_transfer(&transfer("u-1", 1000)).unwrap();
        assert_eq!(d.status, A2aStatus::Settled);
        assert_eq!(d.uetr.as_deref(), Some("u-1"));
    }

    #[test]
    fn uetr_override_rejects() {
        let g = DeterministicA2aGateway::new().with_uetr_override(
            "u-bad",
            A2aStatus::Rejected,
            Some("AC03".into()),
        );
        let d = g.submit_credit_transfer(&transfer("u-bad", 1)).unwrap();
        assert_eq!(d.status, A2aStatus::Rejected);
        assert_eq!(d.reason_code.as_deref(), Some("AC03"));
    }

    #[test]
    fn amount_rule_fires_above_threshold() {
        let g = DeterministicA2aGateway::new().with_amount_ge(
            Money::from_minor(1_000_000, Currency::USD),
            A2aStatus::Pending,
            None,
        );
        let big = g
            .submit_credit_transfer(&transfer("u-big", 2_000_000))
            .unwrap();
        assert_eq!(big.status, A2aStatus::Pending);
    }

    #[test]
    fn transport_error_short_circuits() {
        let g = DeterministicA2aGateway::new().with_transport_error("rst");
        let err = g.submit_credit_transfer(&transfer("u-1", 1)).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn name_can_be_overridden() {
        let g = DeterministicA2aGateway::new().with_name("fednow-mock");
        assert_eq!(g.name(), "fednow-mock");
    }
}
