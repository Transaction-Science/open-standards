//! Crypto-rail adapter.
//!
//! Wraps any [`op_rails_crypto::CryptoGateway`] into a
//! [`RailAdapter`](crate::RailAdapter) so the orchestrator can
//! route crypto payments through the same pipeline as card / A2A.
//!
//! ## Status → outcome mapping
//!
//! - `Finalized` → `Success`
//! - `Pending` / `Confirming` → `SoftFailure` (the orchestrator
//!   treats the attempt as "not yet a confirmed success"; operator-
//!   side polling is the right place to wait for finality).
//! - `Rejected` → `HardDecline`
//! - `Transient` → `SoftFailure`
//!
//! ## What this adapter does NOT do
//!
//! - **Poll for confirmation.** A crypto transfer that comes back
//!   `Confirming` is reported as such; the operator drives the
//!   polling loop (via `gateway.query_status(...)`) and decides
//!   when the depth meets their finality threshold.
//! - **Sign transactions.** The gateway already has the operator's
//!   signer wired in.
//! - **Convert between fiat and stablecoin.** A `CryptoTransferReq`
//!   carries a `Money` whose `currency.code()` is the token symbol
//!   (`"USDC"`, `"EURC"`, ...). FX is operator-side.

use std::sync::Arc;

use op_core::{CryptoAddress, Money, PaymentMethod, RailKind};
use op_rails_crypto::{CryptoDecision, CryptoGateway, CryptoStatus, CryptoTransferReq, TokenRef};

use crate::engine::{AdapterResult, RailAdapter};
use crate::intent::PaymentIntent;
use crate::outcome::AttemptOutcome;

/// Wraps a `CryptoGateway` into a `RailAdapter`. One per
/// `(chain, token)` deployment.
pub struct CryptoAdapter {
    driver_name: String,
    gateway: Arc<dyn CryptoGateway>,
    token: TokenRef,
}

impl CryptoAdapter {
    /// Construct.
    pub fn new(driver_name: impl Into<String>, gateway: Arc<dyn CryptoGateway>) -> Self {
        let token = gateway.token().clone();
        Self {
            driver_name: driver_name.into(),
            gateway,
            token,
        }
    }
}

impl RailAdapter for CryptoAdapter {
    fn driver(&self) -> &str {
        &self.driver_name
    }

    fn rail(&self) -> RailKind {
        RailKind::Crypto
    }

    fn attempt(&self, intent: &PaymentIntent, _attempt_number: usize) -> AdapterResult {
        let address = match &intent.method {
            PaymentMethod::Crypto(a) => a.clone(),
            // Non-crypto method routed to a crypto adapter is a
            // misconfiguration. Soft-fail so the orchestrator can
            // fall back to another driver.
            _ => {
                return AdapterResult {
                    outcome: AttemptOutcome::SoftFailure {
                        code: "method_not_crypto".to_owned(),
                    },
                    psp_payment_id: None,
                    uetr: None,
                };
            }
        };

        // Gateway must service the intended chain. A driver
        // misregistered against the wrong chain shows up here as a
        // hard decline so the orchestrator stops retrying it for
        // this intent.
        if !self.gateway.supports(&address) {
            return AdapterResult {
                outcome: AttemptOutcome::HardDecline {
                    code: format!("unsupported_chain:{}", address.chain),
                },
                psp_payment_id: None,
                uetr: None,
            };
        }

        let req = CryptoTransferReq {
            token: self.token.clone(),
            to: CryptoAddress::new(address.chain, address.address),
            amount: convert_amount_for_token(intent.amount, &self.token),
            idempotency_key: intent.idempotency_key.as_str().to_owned(),
            memo: intent
                .metadata
                .iter()
                .find(|(k, _)| k == "memo")
                .map(|(_, v)| v.clone()),
        };

        match self.gateway.submit_transfer(&req) {
            Ok(d) => AdapterResult {
                outcome: classify_status(&d),
                psp_payment_id: d.tx_hash,
                uetr: None,
            },
            Err(e) => AdapterResult {
                outcome: AttemptOutcome::SoftFailure {
                    code: classify_error(&e),
                },
                psp_payment_id: None,
                uetr: None,
            },
        }
    }
}

/// The intent carries amount in whatever currency the merchant
/// quoted; the token has its own decimal precision. For now we
/// pass through unchanged — operators quoting in USD and settling
/// in USDC handle the 1:1 mapping themselves (USDC has 6 decimals,
/// USD has 2; the operator's quote layer scales).
fn convert_amount_for_token(amount: Money, _token: &TokenRef) -> Money {
    amount
}

fn classify_status(d: &CryptoDecision) -> AttemptOutcome {
    match d.status {
        CryptoStatus::Finalized => AttemptOutcome::Success,
        CryptoStatus::Rejected => AttemptOutcome::HardDecline {
            code: d.reason.clone().unwrap_or_else(|| "rejected".to_owned()),
        },
        CryptoStatus::Pending | CryptoStatus::Confirming => AttemptOutcome::SoftFailure {
            code: format!("{:?}", d.status).to_lowercase(),
        },
        CryptoStatus::Transient => AttemptOutcome::SoftFailure {
            code: "transient".to_owned(),
        },
    }
}

fn classify_error(e: &op_rails_crypto::Error) -> String {
    use op_rails_crypto::Error::{
        Core, DriverValidation, InvalidAddress, MissingField, Rejected, Transport,
        UnsupportedChain, UnsupportedToken,
    };
    match e {
        Transport(_) => "transport".to_owned(),
        Rejected { code, .. } => format!("rejected:{code}"),
        InvalidAddress { .. } => "invalid_address".to_owned(),
        UnsupportedChain(_) => "unsupported_chain".to_owned(),
        UnsupportedToken(_) => "unsupported_token".to_owned(),
        MissingField(_) => "missing_field".to_owned(),
        Core(_) => "core".to_owned(),
        DriverValidation(_) => "driver_validation".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money, VaultRef};
    use op_rails_crypto::{
        CryptoDecision, CryptoStatus, CryptoTransferReq, StableToken, StatusQueryReq,
    };
    use std::sync::Mutex;

    use crate::idempotency::IdempotencyKey;

    struct FakeGateway {
        token: TokenRef,
        decision: Mutex<op_rails_crypto::Result<CryptoDecision>>,
        last_req: Mutex<Option<CryptoTransferReq>>,
    }

    impl FakeGateway {
        fn ok(status: CryptoStatus) -> Self {
            let token = StableToken::UsdcBase.token_ref();
            Self {
                token: token.clone(),
                decision: Mutex::new(Ok(CryptoDecision {
                    status,
                    tx_hash: Some(
                        "0xdeadbeef0000000000000000000000000000000000000000000000000000beef"
                            .to_owned(),
                    ),
                    confirmations: if matches!(status, CryptoStatus::Finalized) {
                        12
                    } else {
                        0
                    },
                    settled_amount: None,
                    raw_status: Some(format!("{status:?}").to_lowercase()),
                    reason: None,
                })),
                last_req: Mutex::new(None),
            }
        }
    }

    impl CryptoGateway for FakeGateway {
        fn name(&self) -> &'static str {
            "fake-usdc-base"
        }
        fn chain(&self) -> &str {
            "base"
        }
        fn token(&self) -> &TokenRef {
            &self.token
        }
        fn submit_transfer(
            &self,
            req: &CryptoTransferReq,
        ) -> op_rails_crypto::Result<CryptoDecision> {
            *self.last_req.lock().unwrap() = Some(req.clone());
            match &*self.decision.lock().unwrap() {
                Ok(d) => Ok(d.clone()),
                Err(e) => Err(e.clone()),
            }
        }
        fn query_status(&self, _req: &StatusQueryReq) -> op_rails_crypto::Result<CryptoDecision> {
            unimplemented!("not used in adapter tests")
        }
    }

    fn intent_crypto() -> PaymentIntent {
        PaymentIntent::new(
            IdempotencyKey::new("crypto-1"),
            Money::from_minor(50_000, Currency::USD),
            PaymentMethod::Crypto(CryptoAddress::new(
                "base",
                "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            )),
        )
    }

    #[test]
    fn finalized_maps_to_success() {
        let gw = Arc::new(FakeGateway::ok(CryptoStatus::Finalized));
        let adapter = CryptoAdapter::new("usdc-base", gw);
        let r = adapter.attempt(&intent_crypto(), 0);
        assert_eq!(r.outcome, AttemptOutcome::Success);
        assert!(r.psp_payment_id.is_some());
    }

    #[test]
    fn pending_maps_to_soft_failure() {
        let gw = Arc::new(FakeGateway::ok(CryptoStatus::Pending));
        let adapter = CryptoAdapter::new("usdc-base", gw);
        let r = adapter.attempt(&intent_crypto(), 0);
        match r.outcome {
            AttemptOutcome::SoftFailure { code } => assert_eq!(code, "pending"),
            other => panic!("expected SoftFailure, got {other:?}"),
        }
    }

    #[test]
    fn rejected_maps_to_hard_decline() {
        let gw = Arc::new(FakeGateway::ok(CryptoStatus::Rejected));
        let adapter = CryptoAdapter::new("usdc-base", gw);
        let r = adapter.attempt(&intent_crypto(), 0);
        assert!(matches!(r.outcome, AttemptOutcome::HardDecline { .. }));
    }

    #[test]
    fn wrong_chain_hard_declines() {
        let gw = Arc::new(FakeGateway::ok(CryptoStatus::Finalized));
        let adapter = CryptoAdapter::new("usdc-base", gw);
        let intent = PaymentIntent::new(
            IdempotencyKey::new("c"),
            Money::from_minor(100, Currency::USD),
            PaymentMethod::Crypto(CryptoAddress::new("solana", "abc")),
        );
        let r = adapter.attempt(&intent, 0);
        match r.outcome {
            AttemptOutcome::HardDecline { code } => {
                assert!(code.starts_with("unsupported_chain:"));
            }
            other => panic!("expected HardDecline, got {other:?}"),
        }
    }

    #[test]
    fn non_crypto_method_soft_fails() {
        let gw = Arc::new(FakeGateway::ok(CryptoStatus::Finalized));
        let adapter = CryptoAdapter::new("usdc-base", gw);
        let intent = PaymentIntent::new(
            IdempotencyKey::new("c"),
            Money::from_minor(100, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_x")),
        );
        let r = adapter.attempt(&intent, 0);
        assert!(matches!(r.outcome, AttemptOutcome::SoftFailure { .. }));
    }

    #[test]
    fn idempotency_key_flows_into_request() {
        let gw = Arc::new(FakeGateway::ok(CryptoStatus::Finalized));
        let adapter = CryptoAdapter::new("usdc-base", gw.clone());
        adapter.attempt(&intent_crypto(), 0);
        let captured = gw.last_req.lock().unwrap().clone().unwrap();
        assert_eq!(captured.idempotency_key, "crypto-1");
    }

    #[test]
    fn memo_metadata_flows_through() {
        let gw = Arc::new(FakeGateway::ok(CryptoStatus::Finalized));
        let adapter = CryptoAdapter::new("usdc-base", gw.clone());
        let intent = intent_crypto().with_metadata("memo", "ORD-42");
        adapter.attempt(&intent, 0);
        let captured = gw.last_req.lock().unwrap().clone().unwrap();
        assert_eq!(captured.memo.as_deref(), Some("ORD-42"));
    }
}
