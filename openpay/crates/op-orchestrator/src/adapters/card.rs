//! Card-rail adapter.
//!
//! Wraps any [`op_rails_card::CardAcquirer`] into a
//! [`RailAdapter`](crate::RailAdapter) that the orchestrator can
//! call generically.
//!
//! The adapter is intentionally thin — it knows how to:
//!
//! 1. Build an [`AuthRequest`] from a [`PaymentIntent`].
//! 2. Call `acquirer.authorize(...)`.
//! 3. Classify the [`AuthDecision`] into an
//!    [`AttemptOutcome`](crate::AttemptOutcome).
//!
//! What it does NOT do:
//!
//! - **Capture/void/refund.** The orchestrator stops at auth.
//!   Downstream lifecycle is a separate concern handled by the
//!   merchant's payment-state-machine code, which holds the
//!   `psp_payment_id` returned in the outcome.
//! - **3DS challenge resumption.** `RequiresCustomerAction` is a
//!   terminal-pending state from the orchestrator's view; the
//!   caller resumes the flow once the customer completes.

use std::sync::Arc;

use op_core::RailKind;
use op_rails_card::acquirer::{AuthStatus, ThreeDsMode};
use op_rails_card::{AuthDecision, AuthRequest, CardAcquirer};

use crate::engine::{AdapterResult, RailAdapter};
use crate::intent::PaymentIntent;
use crate::outcome::AttemptOutcome;

/// Generic card adapter. Wraps any `CardAcquirer` impl.
///
/// Construct one per (driver, configuration) pair and register with
/// the orchestrator. The `driver_name` field must match the string
/// the router emits in [`RailChoice::driver`](crate::RailChoice::driver).
pub struct CardAdapter {
    driver_name: String,
    acquirer: Arc<dyn CardAcquirer>,
    auto_capture: bool,
}

impl CardAdapter {
    /// Construct.
    pub fn new(driver_name: impl Into<String>, acquirer: Arc<dyn CardAcquirer>) -> Self {
        Self {
            driver_name: driver_name.into(),
            acquirer,
            auto_capture: true,
        }
    }

    /// Builder: opt into manual-capture flows. Default is
    /// `auto_capture = true` (the orchestrator hands back a success
    /// outcome with funds already settled).
    #[must_use]
    pub fn with_manual_capture(mut self) -> Self {
        self.auto_capture = false;
        self
    }
}

impl RailAdapter for CardAdapter {
    fn driver(&self) -> &str {
        &self.driver_name
    }

    fn rail(&self) -> RailKind {
        RailKind::Card
    }

    fn attempt(&self, intent: &PaymentIntent, _attempt_number: usize) -> AdapterResult {
        // Build the AuthRequest. Method is forwarded verbatim — the
        // acquirer is responsible for checking method-compatibility
        // (the trait returns `Error::UnsupportedMethod` if not).
        let req = AuthRequest {
            amount: intent.amount,
            method: intent.method.clone(),
            auto_capture: self.auto_capture,
            // **Critical**: the SAME idempotency key flows across
            // every rail attempt for this intent. PSPs that honor
            // idempotency-key headers (Stripe, Adyen, Hyperswitch)
            // will deduplicate retries.
            idempotency_key: intent.idempotency_key.as_str().to_owned(),
            three_ds: if intent.hints.three_ds_enrolled {
                Some(ThreeDsMode::Required)
            } else {
                None
            },
            metadata: build_metadata(intent),
        };

        match self.acquirer.authorize(&req) {
            Ok(decision) => {
                let psp_payment_id = Some(decision.psp_payment_id.clone());
                let outcome = classify_decision(decision);
                AdapterResult {
                    outcome,
                    psp_payment_id,
                    uetr: None,
                }
            }
            Err(e) => {
                // Acquirer-level error (transport failure, PSP 5xx,
                // unsupported method, etc.). Treated as soft failure
                // so the orchestrator can fall back to another driver
                // — except UnsupportedMethod which is a misconfigured
                // route. We still surface it as soft and let the
                // chain advance; if no driver matches, the engine
                // returns AllRailsExhausted.
                AdapterResult {
                    outcome: AttemptOutcome::SoftFailure {
                        code: classify_error(&e),
                    },
                    psp_payment_id: None,
                    uetr: None,
                }
            }
        }
    }

    fn resume(&self, intent: &PaymentIntent, psp_payment_id: &str) -> AdapterResult {
        match self
            .acquirer
            .confirm_after_challenge(psp_payment_id, intent.idempotency_key.as_str())
        {
            Ok(decision) => {
                let id = Some(decision.psp_payment_id.clone());
                let outcome = classify_decision(decision);
                AdapterResult {
                    outcome,
                    psp_payment_id: id,
                    uetr: None,
                }
            }
            Err(e) => AdapterResult {
                outcome: AttemptOutcome::SoftFailure {
                    code: classify_error(&e),
                },
                psp_payment_id: Some(psp_payment_id.to_owned()),
                uetr: None,
            },
        }
    }
}

/// Map an [`AuthDecision`] into an [`AttemptOutcome`].
///
/// The card AuthStatus taxonomy is richer than ours; we collapse:
/// - Approved / AuthorizedAwaitingCapture / Settled → Success
/// - RequiresCustomerAction → RequiresAction (carries redirect_url)
/// - HardDecline / Fraud → HardDecline (no rail-fallback retry)
/// - SoftDecline / Transient / RequiresMerchantAction → SoftFailure
fn classify_decision(decision: AuthDecision) -> AttemptOutcome {
    let code_or = |default: &str| -> String {
        decision
            .error_code
            .clone()
            .unwrap_or_else(|| default.to_owned())
    };
    match decision.status {
        AuthStatus::Approved | AuthStatus::AuthorizedAwaitingCapture | AuthStatus::Settled => {
            AttemptOutcome::Success
        }
        AuthStatus::RequiresCustomerAction => AttemptOutcome::RequiresAction {
            url: decision
                .redirect_url
                .unwrap_or_else(|| "<missing-url>".to_owned()),
        },
        AuthStatus::HardDecline => AttemptOutcome::HardDecline {
            code: code_or("hard_decline"),
        },
        AuthStatus::Fraud => AttemptOutcome::HardDecline {
            code: code_or("fraud"),
        },
        AuthStatus::SoftDecline => AttemptOutcome::SoftFailure {
            code: code_or("soft_decline"),
        },
        AuthStatus::Transient => AttemptOutcome::SoftFailure {
            code: code_or("transient"),
        },
        AuthStatus::RequiresMerchantAction => AttemptOutcome::SoftFailure {
            code: code_or("requires_merchant_action"),
        },
    }
}

fn classify_error(e: &op_rails_card::Error) -> String {
    // Stable normalized code rather than the inner display text —
    // operators correlate codes across PSPs.
    use op_rails_card::Error::*;
    match e {
        Transport(_) => "transport".to_owned(),
        PspRejected { status, .. } => format!("psp_{status}"),
        MissingField(_) => "missing_field".to_owned(),
        Parse(_) => "parse".to_owned(),
        UnknownStatus(_) => "unknown_status".to_owned(),
        UnsupportedMethod => "unsupported_method".to_owned(),
        Core(_) => "core".to_owned(),
        DriverValidation(_) => "driver_validation".to_owned(),
    }
}

fn build_metadata(intent: &PaymentIntent) -> Option<serde_json::Value> {
    if intent.metadata.is_empty() {
        return None;
    }
    let map: serde_json::Map<String, serde_json::Value> = intent
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    Some(serde_json::Value::Object(map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money, PaymentMethod, VaultRef};
    use op_rails_card::{AuthDecision, CaptureRequest, RefundRequest, VoidRequest};

    use crate::idempotency::IdempotencyKey;

    /// Test double — captures the AuthRequest's idempotency key and
    /// returns a canned AuthDecision (or an Error).
    struct FakeAcquirer {
        result: std::sync::Mutex<op_rails_card::Result<AuthDecision>>,
        captured_idempotency: std::sync::Mutex<Option<String>>,
    }

    impl FakeAcquirer {
        fn approve(decision: AuthDecision) -> Self {
            Self {
                result: std::sync::Mutex::new(Ok(decision)),
                captured_idempotency: std::sync::Mutex::new(None),
            }
        }

        fn err(e: op_rails_card::Error) -> Self {
            Self {
                result: std::sync::Mutex::new(Err(e)),
                captured_idempotency: std::sync::Mutex::new(None),
            }
        }

        fn captured(&self) -> Option<String> {
            self.captured_idempotency.lock().unwrap().clone()
        }
    }

    impl CardAcquirer for FakeAcquirer {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn supports(&self, _m: &op_core::PaymentMethod) -> bool {
            true
        }
        fn authorize(&self, req: &AuthRequest) -> op_rails_card::Result<AuthDecision> {
            *self.captured_idempotency.lock().unwrap() = Some(req.idempotency_key.clone());
            // Clone the stored result so repeated calls are stable.
            match &*self.result.lock().unwrap() {
                Ok(d) => Ok(d.clone()),
                Err(e) => Err(e.clone()),
            }
        }
        fn capture(&self, _req: &CaptureRequest) -> op_rails_card::Result<AuthDecision> {
            unimplemented!("capture not used in adapter tests")
        }
        fn void(&self, _req: &VoidRequest) -> op_rails_card::Result<AuthDecision> {
            unimplemented!("void not used in adapter tests")
        }
        fn refund(&self, _req: &RefundRequest) -> op_rails_card::Result<AuthDecision> {
            unimplemented!("refund not used in adapter tests")
        }
    }

    fn intent() -> PaymentIntent {
        PaymentIntent::new(
            IdempotencyKey::new("idem-test-1"),
            Money::from_minor(1234, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_a")),
        )
    }

    fn approved_decision() -> AuthDecision {
        AuthDecision {
            psp_payment_id: "psp_test_42".into(),
            status: AuthStatus::Settled,
            raw_status: "settled".into(),
            authorized_amount: Some(Money::from_minor(1234, Currency::USD)),
            redirect_url: None,
            error_code: None,
            error_message: None,
        }
    }

    fn make_decision(status: AuthStatus, error_code: Option<&str>) -> AuthDecision {
        AuthDecision {
            psp_payment_id: "psp_test".into(),
            status,
            raw_status: format!("{status:?}").to_lowercase(),
            authorized_amount: None,
            redirect_url: None,
            error_code: error_code.map(|s| s.to_owned()),
            error_message: None,
        }
    }

    #[test]
    fn approved_maps_to_success() {
        let acq = Arc::new(FakeAcquirer::approve(approved_decision()));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        assert_eq!(r.outcome, AttemptOutcome::Success);
        assert_eq!(r.psp_payment_id.as_deref(), Some("psp_test_42"));
        assert!(r.uetr.is_none());
    }

    #[test]
    fn authorized_awaiting_capture_also_maps_to_success() {
        let acq = Arc::new(FakeAcquirer::approve(make_decision(
            AuthStatus::AuthorizedAwaitingCapture,
            None,
        )));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        assert_eq!(r.outcome, AttemptOutcome::Success);
    }

    #[test]
    fn hard_decline_maps_with_code() {
        let acq = Arc::new(FakeAcquirer::approve(make_decision(
            AuthStatus::HardDecline,
            Some("insufficient_funds"),
        )));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        match r.outcome {
            AttemptOutcome::HardDecline { code } => assert_eq!(code, "insufficient_funds"),
            other => panic!("expected HardDecline, got {other:?}"),
        }
    }

    #[test]
    fn fraud_maps_to_hard_decline() {
        let acq = Arc::new(FakeAcquirer::approve(make_decision(
            AuthStatus::Fraud,
            Some("3d_fraud_score"),
        )));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        assert!(matches!(r.outcome, AttemptOutcome::HardDecline { .. }));
    }

    #[test]
    fn soft_decline_maps_to_soft_failure() {
        let acq = Arc::new(FakeAcquirer::approve(make_decision(
            AuthStatus::SoftDecline,
            Some("issuer_unavailable"),
        )));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        assert!(matches!(r.outcome, AttemptOutcome::SoftFailure { .. }));
    }

    #[test]
    fn transient_maps_to_soft_failure() {
        let acq = Arc::new(FakeAcquirer::approve(make_decision(
            AuthStatus::Transient,
            None,
        )));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        match r.outcome {
            AttemptOutcome::SoftFailure { code } => assert_eq!(code, "transient"),
            other => panic!("expected SoftFailure, got {other:?}"),
        }
    }

    #[test]
    fn requires_action_carries_redirect_url() {
        let mut d = make_decision(AuthStatus::RequiresCustomerAction, None);
        d.redirect_url = Some("https://3ds.example/challenge/abc".into());
        let adapter = CardAdapter::new("test", Arc::new(FakeAcquirer::approve(d)));
        let r = adapter.attempt(&intent(), 0);
        match r.outcome {
            AttemptOutcome::RequiresAction { url } => {
                assert_eq!(url, "https://3ds.example/challenge/abc");
            }
            other => panic!("expected RequiresAction, got {other:?}"),
        }
    }

    #[test]
    fn requires_action_with_missing_url_uses_placeholder() {
        // Defensive: PSP returns RequiresCustomerAction but no URL.
        // We surface a placeholder rather than panicking.
        let d = make_decision(AuthStatus::RequiresCustomerAction, None);
        let adapter = CardAdapter::new("test", Arc::new(FakeAcquirer::approve(d)));
        let r = adapter.attempt(&intent(), 0);
        match r.outcome {
            AttemptOutcome::RequiresAction { url } => {
                assert_eq!(url, "<missing-url>");
            }
            other => panic!("expected RequiresAction, got {other:?}"),
        }
    }

    #[test]
    fn idempotency_key_flows_into_auth_request() {
        let acq = Arc::new(FakeAcquirer::approve(approved_decision()));
        let adapter = CardAdapter::new("test", acq.clone());
        adapter.attempt(&intent(), 0);
        assert_eq!(acq.captured().as_deref(), Some("idem-test-1"));
    }

    #[test]
    fn transport_error_maps_to_soft_failure() {
        let acq = Arc::new(FakeAcquirer::err(op_rails_card::Error::Transport(
            "connect timeout".into(),
        )));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        match r.outcome {
            AttemptOutcome::SoftFailure { code } => assert_eq!(code, "transport"),
            other => panic!("expected SoftFailure, got {other:?}"),
        }
    }

    #[test]
    fn psp_rejected_includes_status_code() {
        let acq = Arc::new(FakeAcquirer::err(op_rails_card::Error::PspRejected {
            status: 503,
            code: "upstream_unavailable".into(),
            message: "upstream".into(),
        }));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&intent(), 0);
        match r.outcome {
            AttemptOutcome::SoftFailure { code } => assert_eq!(code, "psp_503"),
            other => panic!("expected SoftFailure, got {other:?}"),
        }
    }

    #[test]
    fn metadata_serializes_when_present() {
        let i = intent()
            .with_metadata("order_id", "ORD-42")
            .with_metadata("ip", "203.0.113.5");
        let acq = Arc::new(FakeAcquirer::approve(approved_decision()));
        let adapter = CardAdapter::new("test", acq);
        adapter.attempt(&i, 0);
        // No way to introspect AuthRequest without a more elaborate
        // capture struct — but the build_metadata function is pure
        // and unit-tested below.
        let json = build_metadata(&i).unwrap();
        assert_eq!(json["order_id"], "ORD-42");
        assert_eq!(json["ip"], "203.0.113.5");
    }

    #[test]
    fn metadata_is_none_for_empty_intent() {
        let i = intent();
        assert!(build_metadata(&i).is_none());
    }

    #[test]
    fn driver_and_rail_accessors() {
        let acq = Arc::new(FakeAcquirer::approve(approved_decision()));
        let adapter = CardAdapter::new("hyperswitch", acq);
        assert_eq!(adapter.driver(), "hyperswitch");
        assert_eq!(adapter.rail(), RailKind::Card);
    }

    #[test]
    fn manual_capture_builder_sets_flag() {
        let acq = Arc::new(FakeAcquirer::approve(approved_decision()));
        let adapter = CardAdapter::new("test", acq).with_manual_capture();
        assert!(!adapter.auto_capture);
    }

    #[test]
    fn three_ds_enrollment_forwards_required() {
        // We can't easily inspect the constructed AuthRequest
        // without instrumenting the fake; but we know intent.hints
        // controls it. Smoke-test the path.
        let mut i = intent();
        i.hints.three_ds_enrolled = true;
        let acq = Arc::new(FakeAcquirer::approve(approved_decision()));
        let adapter = CardAdapter::new("test", acq);
        let r = adapter.attempt(&i, 0);
        // Success either way; the test exists to exercise the
        // ThreeDsMode::Required path through the builder.
        assert_eq!(r.outcome, AttemptOutcome::Success);
    }
}
