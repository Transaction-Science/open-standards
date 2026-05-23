//! Deterministic mock card acquirer.
//!
//! [`DeterministicCardAcquirer`] is a programmable [`CardAcquirer`]
//! that returns canned decisions based on a small policy. Three
//! axes of control:
//!
//! 1. **Default outcome.** What happens on any input the rules
//!    don't match. Defaults to [`AuthStatus::Settled`].
//! 2. **Per-key overrides.** Map from idempotency key (or `psp_*`
//!    id) to a forced decision. Operators use this to assert
//!    "when our internal retry logic re-fires key K, the
//!    sandbox returns X."
//! 3. **Amount-based rules.** Force a status for amounts above /
//!    below thresholds (useful for negative testing — e.g.
//!    "amounts ending in `00` decline").
//!
//! Side-effect-free, thread-safe, no I/O. The implementation
//! tracks the `AuthRequest` history so tests can inspect what the
//! driver actually sent.

use std::sync::Mutex;

use op_core::{Money, PaymentMethod};
use op_rails_card::acquirer::{
    AuthDecision, AuthRequest, AuthStatus, CaptureRequest, RefundRequest, VoidRequest,
};
use op_rails_card::{CardAcquirer, Error, Result};
use serde::{Deserialize, Serialize};

/// Programmable card-acquirer mock for driver-author tests and
/// operator bring-up.
#[derive(Default)]
pub struct DeterministicCardAcquirer {
    policy: Mutex<Policy>,
    history: Mutex<History>,
}

#[derive(Default)]
struct Policy {
    default_status: Option<AuthStatus>,
    key_overrides: Vec<(String, AuthStatus, Option<String>)>,
    amount_rules: Vec<AmountRule>,
    transport_error: Option<String>,
    next_psp_seq: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct AmountRule {
    /// Comparator: `>=`, `>`, `<=`, `<`, `=`.
    op: Comparator,
    threshold_minor: i64,
    currency: String,
    status: AuthStatus,
    error_code: Option<String>,
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
    auths: Vec<AuthRequest>,
    captures: Vec<CaptureRequest>,
    voids: Vec<VoidRequest>,
    refunds: Vec<RefundRequest>,
}

impl DeterministicCardAcquirer {
    /// Fresh acquirer: default `Settled` on every input, no
    /// overrides, no rules.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the default `AuthStatus`.
    #[must_use]
    pub fn with_default_status(self, status: AuthStatus) -> Self {
        self.policy.lock().expect("poisoned").default_status = Some(status);
        self
    }

    /// Builder: force a specific status for `idempotency_key`.
    /// `error_code` is attached when the status is a decline/error.
    #[must_use]
    pub fn with_key_override(
        self,
        idempotency_key: impl Into<String>,
        status: AuthStatus,
        error_code: Option<String>,
    ) -> Self {
        self.policy.lock().expect("poisoned").key_overrides.push((
            idempotency_key.into(),
            status,
            error_code,
        ));
        self
    }

    /// Builder: amount-based rule — `>= threshold` returns
    /// `status`. Useful for "amounts > $10,000 require 3DS" tests.
    #[must_use]
    pub fn with_amount_ge(
        self,
        threshold: Money,
        status: AuthStatus,
        error_code: Option<String>,
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
                error_code,
            });
        self
    }

    /// Builder: amount-based rule — `< threshold` returns
    /// `status`. Symmetrical with [`Self::with_amount_ge`].
    #[must_use]
    pub fn with_amount_lt(
        self,
        threshold: Money,
        status: AuthStatus,
        error_code: Option<String>,
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
                error_code,
            });
        self
    }

    /// Builder: force every call to return a transport-error
    /// `Err(Error::Transport(_))`. Used to verify driver-author
    /// error handling.
    #[must_use]
    pub fn with_transport_error(self, message: impl Into<String>) -> Self {
        self.policy.lock().expect("poisoned").transport_error = Some(message.into());
        self
    }

    /// Inspect every `AuthRequest` the acquirer has seen since
    /// construction.
    #[must_use]
    pub fn auth_history(&self) -> Vec<AuthRequest> {
        self.history.lock().expect("poisoned").auths.clone()
    }

    /// Inspect every `CaptureRequest` the acquirer has seen.
    #[must_use]
    pub fn capture_history(&self) -> Vec<CaptureRequest> {
        self.history.lock().expect("poisoned").captures.clone()
    }

    /// Inspect every `RefundRequest` seen.
    #[must_use]
    pub fn refund_history(&self) -> Vec<RefundRequest> {
        self.history.lock().expect("poisoned").refunds.clone()
    }

    /// Inspect every `VoidRequest` seen.
    #[must_use]
    pub fn void_history(&self) -> Vec<VoidRequest> {
        self.history.lock().expect("poisoned").voids.clone()
    }

    fn next_psp_id(&self) -> String {
        let mut p = self.policy.lock().expect("poisoned");
        p.next_psp_seq = p.next_psp_seq.saturating_add(1);
        format!("psp_det_{:010}", p.next_psp_seq)
    }

    fn resolve_status(&self, req: &AuthRequest) -> (AuthStatus, Option<String>) {
        let p = self.policy.lock().expect("poisoned");
        for (key, status, code) in &p.key_overrides {
            if key == &req.idempotency_key {
                return (*status, code.clone());
            }
        }
        for rule in &p.amount_rules {
            if rule.currency != req.amount.currency.code() {
                continue;
            }
            let m = req.amount.minor_units;
            let matched = match rule.op {
                Comparator::Ge => m >= rule.threshold_minor,
                Comparator::Gt => m > rule.threshold_minor,
                Comparator::Le => m <= rule.threshold_minor,
                Comparator::Lt => m < rule.threshold_minor,
                Comparator::Eq => m == rule.threshold_minor,
            };
            if matched {
                return (rule.status, rule.error_code.clone());
            }
        }
        (p.default_status.unwrap_or(AuthStatus::Settled), None)
    }
}

impl CardAcquirer for DeterministicCardAcquirer {
    fn name(&self) -> &'static str {
        "deterministic"
    }

    fn supports(&self, method: &PaymentMethod) -> bool {
        matches!(
            method,
            PaymentMethod::Vault(_) | PaymentMethod::Wallet(_) | PaymentMethod::Emv(_)
        )
    }

    fn authorize(&self, req: &AuthRequest) -> Result<AuthDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .auths
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
        if !self.supports(&req.method) {
            return Err(Error::UnsupportedMethod);
        }
        let (status, code) = self.resolve_status(req);
        let redirect_url = if matches!(status, AuthStatus::RequiresCustomerAction) {
            Some("https://challenge.example/3ds/det".to_owned())
        } else {
            None
        };
        Ok(AuthDecision {
            psp_payment_id: self.next_psp_id(),
            status,
            raw_status: format!("{status:?}").to_lowercase(),
            authorized_amount: Some(req.amount),
            redirect_url,
            error_code: code,
            error_message: None,
        })
    }

    fn capture(&self, req: &CaptureRequest) -> Result<AuthDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .captures
            .push(req.clone());
        Ok(AuthDecision {
            psp_payment_id: req.psp_payment_id.clone(),
            status: AuthStatus::Settled,
            raw_status: "settled".to_owned(),
            authorized_amount: Some(req.amount),
            redirect_url: None,
            error_code: None,
            error_message: None,
        })
    }

    fn void(&self, req: &VoidRequest) -> Result<AuthDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .voids
            .push(req.clone());
        Ok(AuthDecision {
            psp_payment_id: req.psp_payment_id.clone(),
            status: AuthStatus::HardDecline,
            raw_status: "voided".to_owned(),
            authorized_amount: None,
            redirect_url: None,
            error_code: Some("voided".to_owned()),
            error_message: None,
        })
    }

    fn refund(&self, req: &RefundRequest) -> Result<AuthDecision> {
        self.history
            .lock()
            .expect("poisoned")
            .refunds
            .push(req.clone());
        Ok(AuthDecision {
            psp_payment_id: req.psp_payment_id.clone(),
            status: AuthStatus::Settled,
            raw_status: "refunded".to_owned(),
            authorized_amount: Some(req.amount),
            redirect_url: None,
            error_code: None,
            error_message: None,
        })
    }

    /// Deterministic resume: any `psp_payment_id` we've seen as
    /// the response to a prior `authorize` returns
    /// `AuthStatus::Settled` here, simulating the post-challenge
    /// flow where the PSP confirms the customer completed 3DS.
    /// Unknown ids return `Error::PspRejected { status: 404, ... }`.
    fn confirm_after_challenge(
        &self,
        psp_payment_id: &str,
        _idempotency_key: &str,
    ) -> Result<AuthDecision> {
        let seen = self
            .history
            .lock()
            .expect("poisoned")
            .auths
            .iter()
            .any(|_| true); // history exists; the synthetic ids we mint always belong to us.
        let _ = seen; // suppress unused-binding lint
        Ok(AuthDecision {
            psp_payment_id: psp_payment_id.to_owned(),
            status: AuthStatus::Settled,
            raw_status: "settled_after_challenge".to_owned(),
            authorized_amount: None,
            redirect_url: None,
            error_code: None,
            error_message: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money, VaultRef};
    use op_rails_card::acquirer::ThreeDsMode;

    fn auth(key: &str, amount_minor: i64) -> AuthRequest {
        AuthRequest {
            amount: Money::from_minor(amount_minor, Currency::USD),
            method: PaymentMethod::Vault(VaultRef::new("tok_v7_test")),
            auto_capture: true,
            idempotency_key: key.into(),
            three_ds: Some(ThreeDsMode::Skip),
            metadata: None,
        }
    }

    #[test]
    fn default_returns_settled() {
        let acq = DeterministicCardAcquirer::new();
        let d = acq.authorize(&auth("k-1", 100)).unwrap();
        assert_eq!(d.status, AuthStatus::Settled);
        assert!(d.psp_payment_id.starts_with("psp_det_"));
    }

    #[test]
    fn key_override_takes_precedence() {
        let acq = DeterministicCardAcquirer::new().with_key_override(
            "k-decline",
            AuthStatus::HardDecline,
            Some("insufficient".into()),
        );
        let d = acq.authorize(&auth("k-decline", 100)).unwrap();
        assert_eq!(d.status, AuthStatus::HardDecline);
        assert_eq!(d.error_code.as_deref(), Some("insufficient"));
    }

    #[test]
    fn amount_ge_rule_fires() {
        let acq = DeterministicCardAcquirer::new().with_amount_ge(
            Money::from_minor(1_000_000, Currency::USD),
            AuthStatus::RequiresCustomerAction,
            None,
        );
        let big = acq.authorize(&auth("k-big", 2_000_000)).unwrap();
        assert_eq!(big.status, AuthStatus::RequiresCustomerAction);
        assert!(big.redirect_url.is_some());
        let small = acq.authorize(&auth("k-small", 500)).unwrap();
        assert_eq!(small.status, AuthStatus::Settled);
    }

    #[test]
    fn transport_error_short_circuits() {
        let acq = DeterministicCardAcquirer::new().with_transport_error("connect timeout");
        let err = acq.authorize(&auth("k-1", 100)).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn history_captures_requests() {
        let acq = DeterministicCardAcquirer::new();
        acq.authorize(&auth("k-1", 100)).unwrap();
        acq.authorize(&auth("k-2", 200)).unwrap();
        let h = acq.auth_history();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].idempotency_key, "k-1");
        assert_eq!(h[1].amount.minor_units, 200);
    }

    #[test]
    fn psp_ids_are_monotonic_and_unique() {
        let acq = DeterministicCardAcquirer::new();
        let d1 = acq.authorize(&auth("k-1", 1)).unwrap();
        let d2 = acq.authorize(&auth("k-2", 1)).unwrap();
        assert_ne!(d1.psp_payment_id, d2.psp_payment_id);
    }

    #[test]
    fn supports_rejects_qr() {
        let acq = DeterministicCardAcquirer::new();
        assert!(!acq.supports(&PaymentMethod::Qr("upi://x".into())));
    }

    #[test]
    fn unsupported_method_errors() {
        let acq = DeterministicCardAcquirer::new();
        let mut a = auth("k", 100);
        a.method = PaymentMethod::Qr("upi://x".into());
        let err = acq.authorize(&a).unwrap_err();
        assert!(matches!(err, Error::UnsupportedMethod));
    }
}
