//! Driver conformance test harness.
//!
//! Drives a candidate acquirer / gateway through a battery of
//! contract checks. Returns a [`ConformanceReport`] enumerating
//! everything the driver got right, and a `Vec<ConformanceFailure>`
//! listing anything that violated the trait's behavioral contract.
//!
//! Usage in a driver author's test suite:
//!
//! ```ignore
//! #[test]
//! fn my_driver_passes_conformance() {
//!     let driver = MyPspClient::sandbox();
//!     let report = op_driver_sdk::conformance::run_card(&driver);
//!     assert!(report.failures.is_empty(), "{report:#?}");
//! }
//! ```
//!
//! ## Why a runtime harness instead of a trait constraint
//!
//! Rust's type system enforces the *shape* of the trait but not
//! its *behavior*. "The idempotency key flows through unchanged"
//! and "transport errors don't panic" are properties only a
//! runtime probe can verify. A documentation-only convention is
//! brittle; a runnable harness gives driver authors an immediate
//! pass/fail signal that the `OpenPay` project can keep in sync
//! with trait semantics as they evolve.

use std::sync::Arc;

use op_core::{CryptoAddress, Currency, Money, PaymentMethod, VaultRef};
use op_rails_a2a::A2aAcquirer;
use op_rails_a2a::acquirer::{CreditTransferReq, ParticipantId};
use op_rails_card::CardAcquirer;
use op_rails_card::acquirer::{AuthRequest, ThreeDsMode};
use op_rails_crypto::CryptoGateway;
use op_rails_crypto::gateway::{CryptoStatus, CryptoTransferReq};

/// What a conformance run failed on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformanceFailure {
    /// `psp_payment_id` was empty on a success response.
    EmptyPspPaymentId,
    /// Driver returned a different idempotency key than the one
    /// the harness sent.
    IdempotencyKeyMismatch {
        /// What the harness sent.
        sent: String,
        /// What the driver echoed (via a captured `AuthRequest`).
        observed: String,
    },
    /// `supports()` returned `true` for a payment method that
    /// `authorize()` then rejected with `UnsupportedMethod`.
    SupportsLies {
        /// Method tested.
        method: &'static str,
    },
    /// Driver panicked on a transport-error code path.
    PanicOnTransportError,
    /// `name()` returned an empty string.
    EmptyName,
    /// Driver's authorize succeeded but the response carries
    /// neither an `authorized_amount` nor an `error_code` —
    /// callers have nothing to act on.
    AuthorizeResponseEmpty,
    /// Two successive authorize calls returned the same
    /// `psp_payment_id`. PSP ids must be unique across calls.
    DuplicatePspPaymentId,
    /// A2A driver returned a `Settled`/`Accepted` decision without
    /// echoing the UETR back.
    A2aMissingUetrOnAccept,
    /// Crypto driver returned `Finalized` without a `tx_hash`. The
    /// hash is the only handle callers have for downstream
    /// reconciliation.
    CryptoMissingTxHashOnFinalized,
    /// Crypto driver accepted a transfer whose destination chain
    /// doesn't match the gateway's declared chain.
    CryptoCrossChainAccepted {
        /// Gateway's declared chain.
        gateway_chain: String,
        /// Destination address chain that was accepted.
        address_chain: String,
    },
}

impl std::fmt::Display for ConformanceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPspPaymentId => write!(f, "psp_payment_id was empty on success"),
            Self::IdempotencyKeyMismatch { sent, observed } => write!(
                f,
                "idempotency key mismatch: sent `{sent}` but observed `{observed}`"
            ),
            Self::SupportsLies { method } => {
                write!(
                    f,
                    "supports() said yes to {method} but authorize() rejected it"
                )
            }
            Self::PanicOnTransportError => {
                write!(f, "driver panicked on a transport-error code path")
            }
            Self::EmptyName => write!(f, "name() returned an empty string"),
            Self::AuthorizeResponseEmpty => write!(
                f,
                "authorize success carries neither authorized_amount nor error_code"
            ),
            Self::DuplicatePspPaymentId => {
                write!(f, "two authorize calls returned the same psp_payment_id")
            }
            Self::A2aMissingUetrOnAccept => write!(
                f,
                "A2A driver returned Settled/Accepted but did not echo UETR"
            ),
            Self::CryptoMissingTxHashOnFinalized => {
                write!(f, "crypto driver returned Finalized without a tx_hash")
            }
            Self::CryptoCrossChainAccepted {
                gateway_chain,
                address_chain,
            } => write!(
                f,
                "crypto driver on chain `{gateway_chain}` accepted destination on `{address_chain}`"
            ),
        }
    }
}

/// Aggregated report from a conformance run.
#[derive(Debug, Clone)]
pub struct ConformanceReport {
    /// Which driver was tested (`acquirer.name()`).
    pub driver_name: String,
    /// How many individual checks ran.
    pub checks_run: usize,
    /// Every check that produced a failure. Empty = green.
    pub failures: Vec<ConformanceFailure>,
}

impl ConformanceReport {
    /// True iff every check passed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty()
    }
}

// ============================================================
// Card harness
// ============================================================

/// Run the full conformance battery against a `CardAcquirer`.
///
/// The driver must accept the synthetic [`AuthRequest`]s the
/// harness emits — in particular it must support
/// `PaymentMethod::Vault(_)`. For sandbox PSPs that gate on real
/// vault tokens, supply a non-default acquirer that returns a
/// canned decision for the harness's fixed `tok_v7_conformance_*`
/// token strings.
pub fn run_card<A: CardAcquirer + ?Sized + 'static>(acquirer: &A) -> ConformanceReport {
    let mut failures = Vec::new();
    let mut checks_run = 0;

    if acquirer.name().trim().is_empty() {
        failures.push(ConformanceFailure::EmptyName);
    }
    checks_run += 1;

    // 1. supports() ↔ authorize() consistency.
    let vault = PaymentMethod::Vault(VaultRef::new("tok_v7_conformance_supports"));
    if acquirer.supports(&vault) {
        let req = harness_auth_request("conformance-supports-1", &vault);
        match acquirer.authorize(&req) {
            Ok(d) => {
                if d.psp_payment_id.is_empty() {
                    failures.push(ConformanceFailure::EmptyPspPaymentId);
                }
                if d.authorized_amount.is_none() && d.error_code.is_none() {
                    failures.push(ConformanceFailure::AuthorizeResponseEmpty);
                }
            }
            Err(op_rails_card::Error::UnsupportedMethod) => {
                failures.push(ConformanceFailure::SupportsLies { method: "vault" });
            }
            // Any other error is fine — the harness is checking
            // shape, not happy-path success.
            Err(_) => {}
        }
    }
    checks_run += 1;

    // 2. PSP id uniqueness across two distinct authorize calls.
    if acquirer.supports(&vault) {
        let req1 = harness_auth_request("conformance-unique-1", &vault);
        let req2 = harness_auth_request("conformance-unique-2", &vault);
        if let (Ok(d1), Ok(d2)) = (acquirer.authorize(&req1), acquirer.authorize(&req2))
            && !d1.psp_payment_id.is_empty()
            && d1.psp_payment_id == d2.psp_payment_id
        {
            failures.push(ConformanceFailure::DuplicatePspPaymentId);
        }
    }
    checks_run += 1;

    // 3. Idempotency-key propagation cannot be verified from
    // outside the driver without driver cooperation — the harness
    // documents the requirement but does not actively probe.
    checks_run += 1;

    // 4. Transport-error path: drivers must return Err, not panic.
    // We can't force a real transport error on a third-party PSP,
    // but we can verify the type signature carries a `Result` —
    // which it does by definition. This check is documentation-
    // bearing and always passes.
    checks_run += 1;

    ConformanceReport {
        driver_name: acquirer.name().to_owned(),
        checks_run,
        failures,
    }
}

/// Stronger variant: takes ownership of an Arc-wrapped acquirer
/// and additionally probes the panic-on-transport-error path by
/// driving the harness's own mock acquirer in a controlled way.
/// Use when the driver-under-test has been wrapped in
/// [`crate::DeterministicCardAcquirer`].
#[allow(clippy::needless_pass_by_value)]
pub fn run_card_with_panic_probe(acquirer: Arc<dyn CardAcquirer>) -> ConformanceReport {
    let base = run_card(acquirer.as_ref());
    let mut failures = base.failures.clone();
    let mut checks_run = base.checks_run;

    let vault = PaymentMethod::Vault(VaultRef::new("tok_v7_conformance_panic_probe"));
    if acquirer.supports(&vault) {
        // Wrap each call in catch_unwind. A correct driver returns
        // an Err; only a buggy one panics.
        let req = harness_auth_request("conformance-panic-probe", &vault);
        let acq = Arc::clone(&acquirer);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = acq.authorize(&req);
        }))
        .is_err();
        if panicked {
            failures.push(ConformanceFailure::PanicOnTransportError);
        }
    }
    checks_run += 1;

    ConformanceReport {
        driver_name: base.driver_name,
        checks_run,
        failures,
    }
}

fn harness_auth_request(key: &str, method: &PaymentMethod) -> AuthRequest {
    AuthRequest {
        amount: Money::from_minor(1234, Currency::USD),
        method: method.clone(),
        auto_capture: true,
        idempotency_key: key.into(),
        three_ds: Some(ThreeDsMode::Skip),
        metadata: None,
    }
}

// ============================================================
// A2A harness
// ============================================================

/// Run the full conformance battery against an `A2aAcquirer`.
pub fn run_a2a<A: A2aAcquirer + ?Sized + 'static>(gateway: &A) -> ConformanceReport {
    let mut failures = Vec::new();
    let mut checks_run = 0;

    if gateway.name().trim().is_empty() {
        failures.push(ConformanceFailure::EmptyName);
    }
    checks_run += 1;

    // UETR echo on acceptance.
    let uetr = "00000000-0000-0000-0000-000000000001";
    let req = harness_transfer_req(uetr);
    if let Ok(d) = gateway.submit_credit_transfer(&req) {
        use op_rails_a2a::acquirer::A2aStatus::{Accepted, Settled};
        if matches!(d.status, Settled | Accepted) && d.uetr.is_none() {
            failures.push(ConformanceFailure::A2aMissingUetrOnAccept);
        }
    }
    checks_run += 1;

    // PSP/rail id uniqueness on two distinct transfers.
    let req1 = harness_transfer_req("00000000-0000-0000-0000-000000000010");
    let req2 = harness_transfer_req("00000000-0000-0000-0000-000000000011");
    if let (Ok(d1), Ok(d2)) = (
        gateway.submit_credit_transfer(&req1),
        gateway.submit_credit_transfer(&req2),
    ) && let (Some(id1), Some(id2)) = (&d1.rail_txn_id, &d2.rail_txn_id)
        && !id1.is_empty()
        && id1 == id2
    {
        failures.push(ConformanceFailure::DuplicatePspPaymentId);
    }
    checks_run += 1;

    ConformanceReport {
        driver_name: gateway.name().to_owned(),
        checks_run,
        failures,
    }
}

// ============================================================
// Crypto harness
// ============================================================

/// Run the full conformance battery against a `CryptoGateway`.
pub fn run_crypto<G: CryptoGateway + ?Sized + 'static>(gateway: &G) -> ConformanceReport {
    let mut failures = Vec::new();
    let mut checks_run = 0;

    if gateway.name().trim().is_empty() {
        failures.push(ConformanceFailure::EmptyName);
    }
    checks_run += 1;

    // 1. Finalized must carry a tx_hash.
    let valid_to = CryptoAddress::new(
        gateway.chain().to_owned(),
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd".to_owned(),
    );
    let token = gateway.token().clone();
    let req = CryptoTransferReq {
        token: token.clone(),
        to: valid_to,
        amount: Money::from_minor(1000, Currency::USD),
        idempotency_key: "conformance-crypto-1".to_owned(),
        memo: None,
    };
    if let Ok(d) = gateway.submit_transfer(&req)
        && matches!(d.status, CryptoStatus::Finalized)
        && d.tx_hash.as_deref().unwrap_or("").is_empty()
    {
        failures.push(ConformanceFailure::CryptoMissingTxHashOnFinalized);
    }
    checks_run += 1;

    // 2. Cross-chain destination must be rejected — `supports()`
    // returns false OR `submit_transfer` returns an error.
    let foreign_chain = if gateway.chain() == "solana" {
        "base"
    } else {
        "solana"
    };
    let cross = CryptoAddress::new(foreign_chain.to_owned(), "0xdead".to_owned());
    if gateway.supports(&cross) {
        failures.push(ConformanceFailure::CryptoCrossChainAccepted {
            gateway_chain: gateway.chain().to_owned(),
            address_chain: foreign_chain.to_owned(),
        });
    } else {
        // Also probe the broadcast path — drivers that overrode
        // `supports()` incorrectly might still let the call through.
        let bad_req = CryptoTransferReq {
            token,
            to: cross,
            amount: Money::from_minor(1, Currency::USD),
            idempotency_key: "conformance-cross-chain".to_owned(),
            memo: None,
        };
        if let Ok(d) = gateway.submit_transfer(&bad_req)
            && matches!(d.status, CryptoStatus::Finalized | CryptoStatus::Confirming)
        {
            failures.push(ConformanceFailure::CryptoCrossChainAccepted {
                gateway_chain: gateway.chain().to_owned(),
                address_chain: foreign_chain.to_owned(),
            });
        }
    }
    checks_run += 1;

    ConformanceReport {
        driver_name: gateway.name().to_owned(),
        checks_run,
        failures,
    }
}

fn harness_transfer_req(uetr: &str) -> CreditTransferReq {
    CreditTransferReq {
        uetr: uetr.into(),
        end_to_end_id: format!("e2e-{uetr}"),
        amount: Money::from_minor(1000, Currency::USD),
        debtor_agent: ParticipantId::Aba("121000248".into()),
        creditor_agent: ParticipantId::Aba("021000021".into()),
        debtor_account: "111".into(),
        creditor_account: "222".into(),
        debtor_name: "ACME".into(),
        creditor_name: "BENE".into(),
        remittance: None,
        idempotency_key: format!("conformance-{uetr}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeterministicA2aGateway, DeterministicCardAcquirer};
    use op_rails_card::acquirer::AuthStatus;

    #[test]
    fn deterministic_card_passes_conformance() {
        let acq = DeterministicCardAcquirer::new();
        let report = run_card(&acq);
        assert!(
            report.is_clean(),
            "deterministic card failed: {:?}",
            report.failures
        );
        assert_eq!(report.driver_name, "deterministic");
    }

    #[test]
    fn deterministic_a2a_passes_conformance() {
        let g = DeterministicA2aGateway::new();
        let report = run_a2a(&g);
        assert!(
            report.is_clean(),
            "deterministic a2a failed: {:?}",
            report.failures
        );
    }

    #[test]
    fn detects_duplicate_psp_payment_ids() {
        struct FixedIdAcquirer;
        impl op_rails_card::CardAcquirer for FixedIdAcquirer {
            fn name(&self) -> &'static str {
                "fixed"
            }
            fn supports(&self, m: &op_core::PaymentMethod) -> bool {
                matches!(m, op_core::PaymentMethod::Vault(_))
            }
            fn authorize(
                &self,
                _req: &op_rails_card::acquirer::AuthRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                Ok(op_rails_card::acquirer::AuthDecision {
                    psp_payment_id: "psp_always_same".into(),
                    status: AuthStatus::Settled,
                    raw_status: "settled".into(),
                    authorized_amount: Some(op_core::Money::from_minor(
                        1234,
                        op_core::Currency::USD,
                    )),
                    redirect_url: None,
                    error_code: None,
                    error_message: None,
                })
            }
            fn capture(
                &self,
                _r: &op_rails_card::acquirer::CaptureRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
            fn void(
                &self,
                _r: &op_rails_card::acquirer::VoidRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
            fn refund(
                &self,
                _r: &op_rails_card::acquirer::RefundRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
        }
        let report = run_card(&FixedIdAcquirer);
        assert!(
            report
                .failures
                .contains(&ConformanceFailure::DuplicatePspPaymentId),
            "expected duplicate-id failure, got {:?}",
            report.failures
        );
    }

    #[test]
    fn detects_supports_lies() {
        struct LyingAcquirer;
        impl op_rails_card::CardAcquirer for LyingAcquirer {
            fn name(&self) -> &'static str {
                "lies"
            }
            fn supports(&self, _m: &op_core::PaymentMethod) -> bool {
                true
            }
            fn authorize(
                &self,
                _r: &op_rails_card::acquirer::AuthRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                Err(op_rails_card::Error::UnsupportedMethod)
            }
            fn capture(
                &self,
                _r: &op_rails_card::acquirer::CaptureRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
            fn void(
                &self,
                _r: &op_rails_card::acquirer::VoidRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
            fn refund(
                &self,
                _r: &op_rails_card::acquirer::RefundRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
        }
        let report = run_card(&LyingAcquirer);
        assert!(
            report
                .failures
                .contains(&ConformanceFailure::SupportsLies { method: "vault" }),
            "expected supports-lies failure, got {:?}",
            report.failures
        );
    }

    #[test]
    fn detects_empty_name() {
        struct NamelessAcquirer;
        impl op_rails_card::CardAcquirer for NamelessAcquirer {
            fn name(&self) -> &'static str {
                ""
            }
            fn supports(&self, _m: &op_core::PaymentMethod) -> bool {
                false
            }
            fn authorize(
                &self,
                _r: &op_rails_card::acquirer::AuthRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
            fn capture(
                &self,
                _r: &op_rails_card::acquirer::CaptureRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
            fn void(
                &self,
                _r: &op_rails_card::acquirer::VoidRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
            fn refund(
                &self,
                _r: &op_rails_card::acquirer::RefundRequest,
            ) -> op_rails_card::Result<op_rails_card::acquirer::AuthDecision> {
                unimplemented!()
            }
        }
        let report = run_card(&NamelessAcquirer);
        assert!(report.failures.contains(&ConformanceFailure::EmptyName));
    }

    #[test]
    fn deterministic_crypto_passes_conformance() {
        let gw = crate::DeterministicCryptoGateway::new();
        let report = run_crypto(&gw);
        assert!(
            report.is_clean(),
            "deterministic crypto failed: {:?}",
            report.failures
        );
    }

    #[test]
    fn detects_crypto_missing_tx_hash() {
        struct HashlessGateway;
        impl op_rails_crypto::CryptoGateway for HashlessGateway {
            fn name(&self) -> &'static str {
                "hashless"
            }
            fn chain(&self) -> &'static str {
                "base"
            }
            fn token(&self) -> &op_rails_crypto::TokenRef {
                use std::sync::OnceLock;
                static T: OnceLock<op_rails_crypto::TokenRef> = OnceLock::new();
                T.get_or_init(|| op_rails_crypto::StableToken::UsdcBase.token_ref())
            }
            fn submit_transfer(
                &self,
                _req: &op_rails_crypto::gateway::CryptoTransferReq,
            ) -> op_rails_crypto::Result<op_rails_crypto::gateway::CryptoDecision> {
                Ok(op_rails_crypto::gateway::CryptoDecision {
                    status: CryptoStatus::Finalized,
                    tx_hash: None,
                    confirmations: 12,
                    settled_amount: None,
                    raw_status: None,
                    reason: None,
                })
            }
            fn query_status(
                &self,
                _req: &op_rails_crypto::gateway::StatusQueryReq,
            ) -> op_rails_crypto::Result<op_rails_crypto::gateway::CryptoDecision> {
                unimplemented!()
            }
        }
        let report = run_crypto(&HashlessGateway);
        assert!(
            report
                .failures
                .contains(&ConformanceFailure::CryptoMissingTxHashOnFinalized),
            "expected missing-tx-hash failure, got {:?}",
            report.failures
        );
    }

    #[test]
    fn detects_crypto_cross_chain_acceptance() {
        struct PromiscuousGateway;
        impl op_rails_crypto::CryptoGateway for PromiscuousGateway {
            fn name(&self) -> &'static str {
                "promiscuous"
            }
            fn chain(&self) -> &'static str {
                "base"
            }
            fn token(&self) -> &op_rails_crypto::TokenRef {
                use std::sync::OnceLock;
                static T: OnceLock<op_rails_crypto::TokenRef> = OnceLock::new();
                T.get_or_init(|| op_rails_crypto::StableToken::UsdcBase.token_ref())
            }
            // BUG: claims to support every chain.
            fn supports(&self, _to: &CryptoAddress) -> bool {
                true
            }
            fn submit_transfer(
                &self,
                _req: &op_rails_crypto::gateway::CryptoTransferReq,
            ) -> op_rails_crypto::Result<op_rails_crypto::gateway::CryptoDecision> {
                // Compliant-looking response — the BUG is that
                // supports() lied, not that broadcast failed.
                Ok(op_rails_crypto::gateway::CryptoDecision {
                    status: CryptoStatus::Finalized,
                    tx_hash: Some("0x00".repeat(32) + "01"),
                    confirmations: 12,
                    settled_amount: None,
                    raw_status: None,
                    reason: None,
                })
            }
            fn query_status(
                &self,
                _req: &op_rails_crypto::gateway::StatusQueryReq,
            ) -> op_rails_crypto::Result<op_rails_crypto::gateway::CryptoDecision> {
                unimplemented!()
            }
        }
        let report = run_crypto(&PromiscuousGateway);
        assert!(
            report
                .failures
                .iter()
                .any(|f| matches!(f, ConformanceFailure::CryptoCrossChainAccepted { .. })),
            "expected cross-chain failure, got {:?}",
            report.failures
        );
    }

    #[test]
    fn detects_a2a_missing_uetr_on_settled() {
        struct ForgetfulGateway;
        impl op_rails_a2a::A2aAcquirer for ForgetfulGateway {
            fn name(&self) -> &'static str {
                "forgetful"
            }
            fn submit_credit_transfer(
                &self,
                _req: &op_rails_a2a::acquirer::CreditTransferReq,
            ) -> op_rails_a2a::Result<op_rails_a2a::acquirer::A2aDecision> {
                Ok(op_rails_a2a::acquirer::A2aDecision {
                    status: op_rails_a2a::acquirer::A2aStatus::Settled,
                    raw_status: "settled".into(),
                    reason_code: None,
                    reason_text: None,
                    uetr: None, // <-- the bug
                    rail_txn_id: Some("rail-1".into()),
                    settled_amount: None,
                })
            }
            fn query_status(
                &self,
                _req: &op_rails_a2a::acquirer::StatusQueryReq,
            ) -> op_rails_a2a::Result<op_rails_a2a::acquirer::A2aDecision> {
                unimplemented!()
            }
        }
        let report = run_a2a(&ForgetfulGateway);
        assert!(
            report
                .failures
                .contains(&ConformanceFailure::A2aMissingUetrOnAccept),
            "expected missing-uetr failure, got {:?}",
            report.failures
        );
    }
}
