//! A2A-rail adapter.
//!
//! Wraps any [`op_rails_a2a::A2aAcquirer`] into a
//! [`RailAdapter`](crate::RailAdapter).
//!
//! ## Where bank-side data comes from
//!
//! Unlike a card auth (where the PSP keeps the merchant's account
//! details server-side), an A2A credit transfer carries the merchant's
//! creditor agent + creditor account + creditor name on EVERY wire
//! message. The intent's [`PaymentMethod::A2a`](op_core::PaymentMethod::A2a)
//! provides the **customer-side** debtor identifier; the **merchant
//! side** is held in the [`A2aAdapter`] itself, fixed per-adapter at
//! construction time.
//!
//! Operators serving multiple merchants register one [`A2aAdapter`]
//! per merchant tenancy.
//!
//! ## UETR generation
//!
//! ISO 20022 mandates a UUID v4 UETR. The intent's idempotency key
//! is reused for `end_to_end_id` (capped at 35 chars per ISO 20022),
//! and a fresh UUID v4 is generated for the UETR. Crucially: the
//! UETR is generated DETERMINISTICALLY from the idempotency key (via
//! SHA-256 then UUID v4 reformatting) so that retries with the same
//! idempotency key produce the same UETR — which keeps the rail-side
//! idempotency contract intact.

use std::sync::Arc;

use op_core::{A2aKey, PaymentMethod, RailKind};
use op_rails_a2a::{A2aAcquirer, A2aDecision, A2aStatus, CreditTransferReq, ParticipantId};

use crate::engine::{AdapterResult, RailAdapter};
use crate::intent::PaymentIntent;
use crate::outcome::AttemptOutcome;

/// Merchant-side bank details, fixed for an A2A adapter at
/// construction time.
#[derive(Clone, Debug)]
pub struct MerchantBankProfile {
    /// Creditor (receiver = merchant) bank.
    pub creditor_agent: ParticipantId,
    /// Creditor (receiver = merchant) account identifier.
    pub creditor_account: String,
    /// Creditor (receiver = merchant) name.
    pub creditor_name: String,
    /// Debtor (sender = customer) bank. For A2A pull flows where the
    /// customer's bank is known (e.g. enrolled UPI handles, SEPA Inst
    /// with stored IBAN), the merchant supplies it here. Pure-PIX
    /// flows where the debtor agent is inferred from the key set this
    /// to a placeholder.
    pub default_debtor_agent: ParticipantId,
    /// Debtor name to send in pacs.008. Optional — many rails accept
    /// "Customer" as a literal placeholder.
    pub default_debtor_name: String,
}

/// A2A adapter. Wraps any `A2aAcquirer` impl.
pub struct A2aAdapter {
    driver_name: String,
    acquirer: Arc<dyn A2aAcquirer>,
    profile: MerchantBankProfile,
}

impl A2aAdapter {
    /// Construct.
    pub fn new(
        driver_name: impl Into<String>,
        acquirer: Arc<dyn A2aAcquirer>,
        profile: MerchantBankProfile,
    ) -> Self {
        Self {
            driver_name: driver_name.into(),
            acquirer,
            profile,
        }
    }
}

impl RailAdapter for A2aAdapter {
    fn driver(&self) -> &str {
        &self.driver_name
    }

    fn rail(&self) -> RailKind {
        RailKind::A2a
    }

    fn attempt(&self, intent: &PaymentIntent, _attempt_number: usize) -> AdapterResult {
        // Extract debtor identifier from the A2aKey.
        let debtor_account = match &intent.method {
            PaymentMethod::A2a(key) => debtor_account_from_key(key),
            PaymentMethod::Qr(s) => s.clone(), // pass-through
            _ => {
                return AdapterResult {
                    outcome: AttemptOutcome::SoftFailure {
                        code: "method_not_a2a".to_owned(),
                    },
                    psp_payment_id: None,
                    uetr: None,
                };
            }
        };

        // Deterministic UETR derived from the idempotency key so
        // retries with the same key produce the same UETR — required
        // for rail-side idempotency contracts.
        let uetr = derive_uetr_v4(intent.idempotency_key.as_str());

        // end_to_end_id capped at 35 chars per ISO 20022.
        let end_to_end_id = clip_to_35(intent.idempotency_key.as_str());

        let req = CreditTransferReq {
            uetr: uetr.clone(),
            end_to_end_id,
            amount: intent.amount,
            debtor_agent: self.profile.default_debtor_agent.clone(),
            creditor_agent: self.profile.creditor_agent.clone(),
            debtor_account,
            creditor_account: self.profile.creditor_account.clone(),
            debtor_name: self.profile.default_debtor_name.clone(),
            creditor_name: self.profile.creditor_name.clone(),
            remittance: build_remittance(intent),
            idempotency_key: intent.idempotency_key.as_str().to_owned(),
        };

        match self.acquirer.submit_credit_transfer(&req) {
            Ok(decision) => {
                let returned_uetr = decision.uetr.clone().or(Some(uetr));
                let outcome = classify_decision(&decision);
                AdapterResult {
                    outcome,
                    psp_payment_id: None,
                    uetr: returned_uetr,
                }
            }
            Err(e) => AdapterResult {
                outcome: AttemptOutcome::SoftFailure {
                    code: classify_error(&e),
                },
                psp_payment_id: None,
                uetr: Some(uetr),
            },
        }
    }
}

fn debtor_account_from_key(key: &A2aKey) -> String {
    match key {
        A2aKey::Upi(h) => h.clone(),
        A2aKey::Pix(h) => h.clone(),
        A2aKey::Iban(i) => i.clone(),
        A2aKey::UsAch { account, .. } => account.clone(),
    }
}

/// Derive a deterministic UUID v4-shaped UETR from an idempotency
/// key. ISO 20022 mandates v4 (random) format, but the value is just
/// a 128-bit identifier in a specific text layout — what matters is
/// that retries with the same idempotency key produce the same UETR.
///
/// We hash the key with SHA-256 and reformat the first 16 bytes into
/// the v4 canonical layout (version nibble = 4, variant bits = 10).
fn derive_uetr_v4(idempotency_key: &str) -> String {
    use sha2_local::Sha256Like;
    let mut h = Sha256Like::new();
    h.update(idempotency_key.as_bytes());
    let mut bytes = h.finalize_16();
    // Set the version (high nibble of byte 6) to 4.
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    // Set the variant bits (high two bits of byte 8) to 10.
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-\
         {:02x}{:02x}-\
         {:02x}{:02x}-\
         {:02x}{:02x}-\
         {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

/// Local SHA-256 wrapper. We don't pull in the full `sha2` crate
/// because the orchestrator otherwise has no crypto dependency; the
/// transitive dep via `op-fraud` is fine but we keep the surface
/// local for clarity.
mod sha2_local {
    /// Thin wrapper that gives us a fixed-16-byte truncated digest.
    pub(crate) struct Sha256Like {
        // We just need a hash that's available throughout the workspace.
        // `op-fraud` already pulls `sha2`, so the workspace resolver
        // has it; depend through that crate's re-export path.
        inner: sha2::Sha256,
    }

    impl Sha256Like {
        pub fn new() -> Self {
            use sha2::Digest;
            Self {
                inner: sha2::Sha256::new(),
            }
        }

        pub fn update(&mut self, data: &[u8]) {
            use sha2::Digest;
            self.inner.update(data);
        }

        pub fn finalize_16(self) -> [u8; 16] {
            use sha2::Digest;
            let full = self.inner.finalize();
            let mut out = [0u8; 16];
            out.copy_from_slice(&full[..16]);
            out
        }
    }
}

fn clip_to_35(s: &str) -> String {
    if s.len() <= 35 {
        s.to_owned()
    } else {
        s.chars().take(35).collect()
    }
}

fn build_remittance(intent: &PaymentIntent) -> Option<String> {
    // Surface the first metadata "order_id" or "invoice" entry as
    // remittance, capped to rail max (140 chars across FedNow / SEPA
    // / PIX — the driver does the actual clipping).
    intent
        .metadata
        .iter()
        .find(|(k, _)| {
            let k = k.as_str();
            k == "order_id" || k == "invoice" || k == "remittance"
        })
        .map(|(_, v)| v.clone())
}

/// Map an [`A2aDecision`] into an [`AttemptOutcome`].
///
/// - `Settled` / `Accepted` / `InProgress` → Success.
///   (`InProgress` is a valid terminal-pending-from-merchant's-view
///   state on rails like RT1 where settlement notifications arrive
///   asynchronously.)
/// - `Rejected` → HardDecline (rail-side rejection, customer must act).
/// - `Pending` / `Transient` / `OperationalError` → SoftFailure.
fn classify_decision(decision: &A2aDecision) -> AttemptOutcome {
    let code_or = |default: &str| -> String {
        decision
            .reason_code
            .clone()
            .unwrap_or_else(|| default.to_owned())
    };
    match decision.status {
        A2aStatus::Settled | A2aStatus::Accepted | A2aStatus::InProgress => AttemptOutcome::Success,
        A2aStatus::Rejected => AttemptOutcome::HardDecline {
            code: code_or("rjct"),
        },
        A2aStatus::Pending => AttemptOutcome::SoftFailure {
            code: code_or("pdng"),
        },
        A2aStatus::Transient => AttemptOutcome::SoftFailure {
            code: code_or("transient"),
        },
        A2aStatus::OperationalError => AttemptOutcome::SoftFailure {
            code: code_or("operational_error"),
        },
    }
}

fn classify_error(e: &op_rails_a2a::Error) -> String {
    use op_rails_a2a::Error::*;
    match e {
        Transport(_) => "transport".to_owned(),
        RailRejected { code, .. } => format!("rail_{code}"),
        Iso20022(_) => "iso20022".to_owned(),
        UnknownStatus(_) => "unknown_status".to_owned(),
        UnsupportedMethod => "unsupported_method".to_owned(),
        UnsupportedA2aKey { .. } => "unsupported_key".to_owned(),
        CurrencyMismatch { .. } => "currency_mismatch".to_owned(),
        Signing(_) => "signing".to_owned(),
        Core(_) => "core".to_owned(),
        DriverValidation(_) => "driver_validation".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};
    use op_rails_a2a::{A2aDecision, StatusQueryReq};

    use crate::idempotency::IdempotencyKey;

    fn merchant_profile() -> MerchantBankProfile {
        MerchantBankProfile {
            creditor_agent: ParticipantId::Aba("021000021".into()),
            creditor_account: "MERCHANT-ACCT-1".into(),
            creditor_name: "Acme Coffee LLC".into(),
            default_debtor_agent: ParticipantId::Aba("026009593".into()),
            default_debtor_name: "Customer".into(),
        }
    }

    fn intent_ach() -> PaymentIntent {
        PaymentIntent::new(
            IdempotencyKey::new("intent-key-1"),
            Money::from_minor(10000, Currency::USD),
            PaymentMethod::A2a(A2aKey::UsAch {
                routing: "121000358".into(),
                account: "CUST-ACCT-9".into(),
            }),
        )
    }

    /// Test double that captures the CreditTransferReq and returns a
    /// canned A2aDecision (or an Error).
    struct FakeA2aAcquirer {
        result: std::sync::Mutex<op_rails_a2a::Result<A2aDecision>>,
        captured_req: std::sync::Mutex<Option<CreditTransferReq>>,
    }

    impl FakeA2aAcquirer {
        fn accept(decision: A2aDecision) -> Self {
            Self {
                result: std::sync::Mutex::new(Ok(decision)),
                captured_req: std::sync::Mutex::new(None),
            }
        }

        fn err(e: op_rails_a2a::Error) -> Self {
            Self {
                result: std::sync::Mutex::new(Err(e)),
                captured_req: std::sync::Mutex::new(None),
            }
        }
    }

    impl A2aAcquirer for FakeA2aAcquirer {
        fn name(&self) -> &'static str {
            "fake-a2a"
        }
        fn submit_credit_transfer(
            &self,
            req: &CreditTransferReq,
        ) -> op_rails_a2a::Result<A2aDecision> {
            *self.captured_req.lock().unwrap() = Some(req.clone());
            match &*self.result.lock().unwrap() {
                Ok(d) => Ok(d.clone()),
                Err(_) => Err(op_rails_a2a::Error::Transport("fake".into())),
            }
        }
        fn query_status(&self, _req: &StatusQueryReq) -> op_rails_a2a::Result<A2aDecision> {
            unimplemented!("query_status not used in adapter tests")
        }
    }

    fn make_decision(status: A2aStatus) -> A2aDecision {
        A2aDecision {
            status,
            raw_status: format!("{status:?}"),
            reason_code: None,
            reason_text: None,
            uetr: Some("uetr-from-rail".into()),
            rail_txn_id: Some("rail-txn-7".into()),
            settled_amount: Some(Money::from_minor(10000, Currency::USD)),
        }
    }

    #[test]
    fn settled_maps_to_success() {
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Settled)));
        let adapter = A2aAdapter::new("fednow", acq, merchant_profile());
        let r = adapter.attempt(&intent_ach(), 0);
        assert_eq!(r.outcome, AttemptOutcome::Success);
        assert_eq!(r.uetr.as_deref(), Some("uetr-from-rail"));
    }

    #[test]
    fn accepted_maps_to_success() {
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Accepted)));
        let adapter = A2aAdapter::new("rt1", acq, merchant_profile());
        let r = adapter.attempt(&intent_ach(), 0);
        assert_eq!(r.outcome, AttemptOutcome::Success);
    }

    #[test]
    fn rejected_maps_to_hard_decline() {
        let mut d = make_decision(A2aStatus::Rejected);
        d.reason_code = Some("AC03".into()); // invalid creditor account number
        let adapter = A2aAdapter::new(
            "fednow",
            Arc::new(FakeA2aAcquirer::accept(d)),
            merchant_profile(),
        );
        let r = adapter.attempt(&intent_ach(), 0);
        match r.outcome {
            AttemptOutcome::HardDecline { code } => assert_eq!(code, "AC03"),
            other => panic!("expected HardDecline, got {other:?}"),
        }
    }

    #[test]
    fn pending_maps_to_soft_failure() {
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Pending)));
        let adapter = A2aAdapter::new("rt1", acq, merchant_profile());
        let r = adapter.attempt(&intent_ach(), 0);
        match r.outcome {
            AttemptOutcome::SoftFailure { code } => assert_eq!(code, "pdng"),
            other => panic!("expected SoftFailure, got {other:?}"),
        }
    }

    #[test]
    fn transient_maps_to_soft_failure() {
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Transient)));
        let adapter = A2aAdapter::new("fednow", acq, merchant_profile());
        let r = adapter.attempt(&intent_ach(), 0);
        assert!(matches!(r.outcome, AttemptOutcome::SoftFailure { .. }));
    }

    #[test]
    fn uetr_is_deterministic_for_same_idempotency_key() {
        let a = derive_uetr_v4("intent-key-1");
        let b = derive_uetr_v4("intent-key-1");
        assert_eq!(a, b);
        // Differs for a different key.
        assert_ne!(derive_uetr_v4("intent-key-2"), a);
    }

    #[test]
    fn uetr_has_v4_format() {
        let u = derive_uetr_v4("intent-key-1");
        // Length: 8-4-4-4-12 + 4 hyphens = 36 chars.
        assert_eq!(u.len(), 36);
        let bytes = u.as_bytes();
        // Hyphens in the right positions.
        assert_eq!(bytes[8], b'-');
        assert_eq!(bytes[13], b'-');
        assert_eq!(bytes[18], b'-');
        assert_eq!(bytes[23], b'-');
        // Version nibble (position 14) is '4'.
        assert_eq!(bytes[14], b'4');
        // Variant nibble (position 19) is 8, 9, a, or b.
        assert!(matches!(bytes[19], b'8' | b'9' | b'a' | b'b'));
    }

    #[test]
    fn idempotency_key_becomes_end_to_end_id_capped_at_35() {
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Settled)));
        let adapter = A2aAdapter::new("fednow", acq.clone(), merchant_profile());

        // Short key flows through unchanged.
        let i_short = intent_ach();
        adapter.attempt(&i_short, 0);
        let req_short = acq.captured_req.lock().unwrap().clone().unwrap();
        assert_eq!(req_short.end_to_end_id, "intent-key-1");

        // Long key truncates to 35 chars.
        let i_long = PaymentIntent::new(
            IdempotencyKey::new("a".repeat(50)),
            Money::from_minor(10000, Currency::USD),
            PaymentMethod::A2a(A2aKey::UsAch {
                routing: "121000358".into(),
                account: "ACCT".into(),
            }),
        );
        adapter.attempt(&i_long, 0);
        let req_long = acq.captured_req.lock().unwrap().clone().unwrap();
        assert_eq!(req_long.end_to_end_id.len(), 35);
    }

    #[test]
    fn debtor_account_extracted_from_a2a_key_variants() {
        assert_eq!(
            debtor_account_from_key(&A2aKey::Upi("bob@axis".into())),
            "bob@axis"
        );
        assert_eq!(
            debtor_account_from_key(&A2aKey::Pix("11122233344".into())),
            "11122233344"
        );
        assert_eq!(
            debtor_account_from_key(&A2aKey::Iban("DE89370400440532013000".into())),
            "DE89370400440532013000"
        );
        assert_eq!(
            debtor_account_from_key(&A2aKey::UsAch {
                routing: "021000021".into(),
                account: "ACCT-1".into(),
            }),
            "ACCT-1"
        );
    }

    #[test]
    fn metadata_order_id_becomes_remittance() {
        let i = intent_ach().with_metadata("order_id", "ORD-77");
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Settled)));
        let adapter = A2aAdapter::new("fednow", acq.clone(), merchant_profile());
        adapter.attempt(&i, 0);
        let req = acq.captured_req.lock().unwrap().clone().unwrap();
        assert_eq!(req.remittance.as_deref(), Some("ORD-77"));
    }

    #[test]
    fn non_a2a_method_returns_soft_failure() {
        // Defensive: if routing somehow sends a Vault method down
        // an A2A driver, we surface a soft failure with a clear code
        // rather than crashing.
        use op_core::VaultRef;
        let i = PaymentIntent::new(
            IdempotencyKey::new("k"),
            Money::from_minor(100, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_x")),
        );
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Settled)));
        let adapter = A2aAdapter::new("fednow", acq, merchant_profile());
        let r = adapter.attempt(&i, 0);
        match r.outcome {
            AttemptOutcome::SoftFailure { code } => assert_eq!(code, "method_not_a2a"),
            other => panic!("expected SoftFailure(method_not_a2a), got {other:?}"),
        }
    }

    #[test]
    fn transport_error_maps_to_soft_failure() {
        let acq = Arc::new(FakeA2aAcquirer::err(op_rails_a2a::Error::Transport(
            "mTLS reset".into(),
        )));
        let adapter = A2aAdapter::new("fednow", acq, merchant_profile());
        let r = adapter.attempt(&intent_ach(), 0);
        match r.outcome {
            AttemptOutcome::SoftFailure { code } => assert_eq!(code, "transport"),
            other => panic!("expected SoftFailure(transport), got {other:?}"),
        }
        // We still return our derived UETR even on transport failure
        // so observability can match a future settlement notification
        // to this attempt.
        assert!(r.uetr.is_some());
    }

    #[test]
    fn driver_and_rail_accessors() {
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Settled)));
        let adapter = A2aAdapter::new("fednow", acq, merchant_profile());
        assert_eq!(adapter.driver(), "fednow");
        assert_eq!(adapter.rail(), RailKind::A2a);
    }

    #[test]
    fn idempotency_key_forwarded_unchanged() {
        let acq = Arc::new(FakeA2aAcquirer::accept(make_decision(A2aStatus::Settled)));
        let adapter = A2aAdapter::new("fednow", acq.clone(), merchant_profile());
        adapter.attempt(&intent_ach(), 0);
        let req = acq.captured_req.lock().unwrap().clone().unwrap();
        assert_eq!(req.idempotency_key, "intent-key-1");
    }
}
