//! # `pci-zero` — Code-form proof of the PCI-zero topology
//!
//! Single-process Rust binary that demonstrates the same data flow the
//! `compliance/pci-zero-architecture.md` document describes. The
//! components are wired in-process here so the flow is auditable as
//! one program, but the *deployment* topology splits them across hosts
//! and network segments — see the companion document for that.
//!
//! ## What this binary proves
//!
//! 1. The merchant code path (the contents of `merchant::checkout`)
//!    holds nothing but `op_core::VaultRef` opaque tokens, `Money`,
//!    and `Payment<S>` typestate values. There is no `RawPan` import,
//!    no `CardData` value, and no PAN-bearing string anywhere in
//!    merchant-owned code.
//! 2. The vault layer (the only caller of `op_vault::CardData::new`)
//!    is the sole holder of plaintext PAN. The browser-side
//!    `collection_iframe::submit` function is the architectural
//!    stand-in for the iframe served from the vault's domain; in
//!    deployment it runs on the customer's device and submits over
//!    TLS to a vault host on a separate network segment.
//! 3. The rail driver (`mock_card::PciZeroCardAcquirer`) calls
//!    `Vault::detokenize` exactly once at acquirer-submit time, posts
//!    the (mock) acquirer request, and drops the recovered `CardData`
//!    immediately so `ZeroizeOnDrop` clears the PAN bytes.
//!
//! ## What this binary does NOT prove
//!
//! - That AES-256-GCM-SIV is correctly implemented. The reference
//!   `InMemoryVault` is for tests and development only; see
//!   `crates/op-vault/src/in_memory.rs` for the cryptographic
//!   acknowledgements and `compliance/hsm-kms-guidance.md` for the
//!   production HSM / KMS shape.
//! - That the network boundary is enforced. In this binary,
//!   "boundary" is module visibility (`pub(crate)`) — in deployment,
//!   it's a firewall, mTLS, and a different VPC / namespace.
//!
//! ## Running
//!
//! ```bash
//! cargo run -p pci-zero
//! ```
//!
//! Expected output: a trace of the four boundary crossings (browser →
//! vault, app server → orchestrator → card rail, card rail → vault for
//! detokenize, card rail → acquirer) and an `APPROVED` terminal status.

#![forbid(unsafe_code)]

use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────
// Merchant code path. This is the surface that compiles into the merchant
// application server in deployment. It must hold only opaque tokens —
// VaultRef, Money, Payment<S>. No raw PAN.
// ─────────────────────────────────────────────────────────────────────────

mod merchant {
    use op_core::{Currency, Money, PaymentMethod, VaultRef};
    use op_fraud::HeuristicScorer;
    use op_orchestrator::{
        CardAdapter, IdempotencyKey, Orchestrator, OrchestrationOutcome, PaymentIntent,
        PolicyRouter, Result, TerminalStatus,
    };
    use op_rails_card::CardAcquirer;
    use std::sync::Arc;

    /// The merchant application server. Holds an `Orchestrator`
    /// wired with a single card driver. No vault handle here —
    /// the *driver* owns the vault reference, because the driver
    /// is the only place that legitimately calls `detokenize`.
    pub struct Merchant {
        orchestrator: Orchestrator,
    }

    impl Merchant {
        pub fn new(acquirer: Arc<dyn CardAcquirer>) -> Self {
            let mut orchestrator = Orchestrator::new()
                .with_scorer(Box::new(HeuristicScorer::new()))
                .with_router(Box::new(PolicyRouter::new(
                    vec!["pci-zero-card".into()],
                    vec![], // no A2A in this demo
                )));

            orchestrator.register_adapter(Arc::new(CardAdapter::new(
                "pci-zero-card",
                acquirer,
            )));

            Self { orchestrator }
        }

        /// Run a checkout. Inputs are all out-of-scope: an order id,
        /// an amount in minor units, and a `VaultRef`. The merchant
        /// code path *cannot* construct a `RawPan` even if it wanted
        /// to — `op_core::method::pci::RawPan` is gated behind the
        /// `pci-scope` feature and the merchant module deliberately
        /// does not depend on `op-vault`.
        pub fn checkout(
            &self,
            order_id: &str,
            amount_minor: i64,
            token: VaultRef,
        ) -> Result<OrchestrationOutcome> {
            let intent = PaymentIntent::new(
                IdempotencyKey::new(order_id),
                Money::from_minor(amount_minor, Currency::USD),
                PaymentMethod::Vault(token),
            )
            .with_metadata("order_id", order_id);

            self.orchestrator.run(&intent)
        }
    }

    /// Stringify a terminal status for the demo printout.
    pub fn status_label(s: TerminalStatus) -> &'static str {
        match s {
            TerminalStatus::Approved => "APPROVED",
            TerminalStatus::RequiresCustomerAction => "ACTION REQUIRED",
            TerminalStatus::Declined => "DECLINED",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Customer browser. In deployment this is JavaScript in a TLS-isolated
// iframe served from the vault's own domain — the merchant's outer page
// cannot read it (Same-Origin Policy). Here it's a Rust module whose
// `submit` function is the only caller in the program that constructs
// CardData (i.e. the only place that holds raw PAN).
// ─────────────────────────────────────────────────────────────────────────

mod collection_iframe {
    use op_core::VaultRef;
    use op_vault::{CardData, TokenizationPolicy, Vault};

    /// "Browser" side: the customer types a PAN into a TLS-isolated
    /// iframe. The iframe's JS calls the vault directly (TLS 1.3,
    /// cert-pinned) and returns the resulting `VaultRef` to the
    /// outer merchant page via `window.postMessage`.
    ///
    /// The function takes `&dyn Vault` rather than a concrete vault
    /// type to match the deployment shape: the iframe is talking to
    /// the *vault service*, not to an in-process vault.
    pub fn submit(
        vault: &dyn Vault,
        pan: &str,
        exp_month: u8,
        exp_year: u16,
    ) -> Result<VaultRef, Box<dyn std::error::Error>> {
        // CardData construction validates length, Luhn, and
        // expiration. The PAN string lives in `card_data` until
        // `tokenize` consumes it; ZeroizeOnDrop clears the bytes
        // when this function returns.
        let card_data = CardData::new(pan, exp_month, exp_year)?;
        let token = vault.tokenize(card_data, TokenizationPolicy::default())?;
        Ok(token)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Card rail driver. The ONLY component (other than the vault itself)
// that holds CardData, and only for the brief window between
// `detokenize` and the acquirer submit. In deployment this driver lives
// on the merchant app server but holds CardData for milliseconds at most.
// ─────────────────────────────────────────────────────────────────────────

mod mock_card {
    use op_core::PaymentMethod;
    use op_rails_card::{
        AuthDecision, AuthRequest, CaptureRequest, CardAcquirer, Error, RefundRequest, Result,
        VoidRequest,
        acquirer::AuthStatus,
    };
    use op_vault::Vault;
    use std::sync::Arc;

    /// A `CardAcquirer` that detokenizes from a vault and posts to a
    /// (mock) acquirer. Models the real Hyperswitch / Stripe / Adyen
    /// drivers, which do the same thing.
    pub struct PciZeroCardAcquirer {
        vault: Arc<dyn Vault>,
    }

    impl PciZeroCardAcquirer {
        pub fn new(vault: Arc<dyn Vault>) -> Self {
            Self { vault }
        }
    }

    impl CardAcquirer for PciZeroCardAcquirer {
        fn name(&self) -> &'static str {
            "pci-zero-card"
        }

        fn supports(&self, method: &PaymentMethod) -> bool {
            matches!(method, PaymentMethod::Vault(_))
        }

        fn authorize(&self, req: &AuthRequest) -> Result<AuthDecision> {
            // The driver receives only a VaultRef. It performs the
            // single privileged operation in the entire flow:
            // detokenize, post to the acquirer, drop the CardData.
            let token = match &req.method {
                PaymentMethod::Vault(v) => v.clone(),
                _ => return Err(Error::UnsupportedMethod),
            };

            // Detokenize. In deployment this is an mTLS call to the
            // vault service; here it's a direct trait dispatch.
            let card_data = self
                .vault
                .detokenize(&token)
                .map_err(|e| Error::DriverValidation(format!("vault detokenize: {e}")))?;

            // Post to the (mock) acquirer. In a real driver this
            // would be a ureq POST to the PSP's authorize endpoint
            // with the PAN in the JSON body. We extract the
            // last-four for receipt purposes; the full PAN never
            // leaves this scope.
            let last_four = card_data.last_four().to_string();
            let raw_status = format!(
                "mock-acquirer-approved card={}******{}",
                card_data.first_six(),
                last_four
            );
            // `card_data` is dropped at end of scope; ZeroizeOnDrop
            // wipes the PAN bytes before this function returns.
            drop(card_data);

            Ok(AuthDecision {
                psp_payment_id: format!("acq_{}", req.idempotency_key),
                status: AuthStatus::Settled,
                raw_status,
                authorized_amount: Some(req.amount),
                redirect_url: None,
                error_code: None,
                error_message: None,
            })
        }

        fn capture(&self, _req: &CaptureRequest) -> Result<AuthDecision> {
            Err(Error::DriverValidation(
                "mock driver: auto-capture only".into(),
            ))
        }

        fn void(&self, _req: &VoidRequest) -> Result<AuthDecision> {
            Err(Error::DriverValidation("mock driver: no void".into()))
        }

        fn refund(&self, _req: &RefundRequest) -> Result<AuthDecision> {
            Err(Error::DriverValidation("mock driver: no refund".into()))
        }
    }
}

fn main() {
    eprintln!("=== OpenPay pci-zero topology demo ===");
    eprintln!("Boundary crossings in this run:");
    eprintln!("  [B1] browser  -> vault       (tokenize, raw PAN)");
    eprintln!("  [B2] merchant -> orchestrator (VaultRef only)");
    eprintln!("  [B3] driver   -> vault       (detokenize, scoped)");
    eprintln!("  [B4] driver   -> acquirer     (raw PAN, mock)\n");

    // The vault is the CDE. In deployment, this is a different process
    // on a different host in a different VPC. The `Arc<dyn Vault>` is
    // the architectural stand-in for the mTLS endpoint exposed by the
    // vault service.
    let vault: Arc<dyn op_vault::Vault> =
        Arc::new(op_vault::InMemoryVault::ephemeral("demo-vault"));

    // The card rail driver is the only consumer of the vault that
    // legitimately calls `detokenize`. The merchant code path never
    // touches the vault directly.
    let acquirer = Arc::new(mock_card::PciZeroCardAcquirer::new(Arc::clone(&vault)));

    // The merchant application server. Notice the constructor receives
    // a `dyn CardAcquirer` — it doesn't even know there's a vault.
    let merchant = merchant::Merchant::new(acquirer);

    // ─────────────────────────────────────────────────────────────
    // [B1] Browser submits PAN to the vault.
    // ─────────────────────────────────────────────────────────────
    eprintln!("[B1] Customer types card into vault-served iframe.");
    let token = collection_iframe::submit(
        vault.as_ref(),
        "4242424242424242", // Stripe / OpenPay shared test card
        12,
        2030,
    )
    .expect("vault tokenize");
    eprintln!("     vault returned VaultRef = {}", token.as_str());

    // ─────────────────────────────────────────────────────────────
    // [B2] Merchant runs the orchestrator with a VaultRef only.
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n[B2] Merchant runs orchestrator with VaultRef.");
    let outcome = merchant
        .checkout("ORDER-PCI-ZERO-001", 4299, token)
        .expect("checkout");

    // ─────────────────────────────────────────────────────────────
    // Print the result. The orchestrator already drove [B3] (the
    // driver's detokenize call) and [B4] (the driver's acquirer
    // submit) during checkout().
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n[B3/B4] driver detokenized + posted to mock acquirer.");
    eprintln!(
        "       terminal status: {}",
        merchant::status_label(outcome.terminal_status)
    );
    eprintln!("       attempts: {}", outcome.attempts.len());
    if let Some(id) = &outcome.psp_payment_id {
        eprintln!("       psp_payment_id: {id}");
    }
    for (i, a) in outcome.attempts.iter().enumerate() {
        eprintln!(
            "         [{i}] rail={:?} driver={} outcome={:?}",
            a.rail, a.driver, a.outcome
        );
    }

    eprintln!("\n=== Done. Merchant code path held VaultRef only. ===");
}
