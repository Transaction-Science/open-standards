//! End-to-end example: take a payment using `op-core`'s typestate machine
//! driven by the Hyperswitch card-rail adapter.
//!
//! Demonstrates:
//! - `Payment<Created>` constructed from a `Money` and a `PaymentMethod`
//! - `CardAcquirer::authorize` returning a normalized `AuthDecision`
//! - State transitioning to `Payment<Authorized>` on success
//! - Manual capture transitioning to `Payment<Captured>`
//! - Refund returning a `Payment<Refunded>`
//!
//! Run with:
//!   `HYPERSWITCH_API_KEY=sk_test_xxx cargo run -p headless-server`
//!
//! In production this entire flow runs on the merchant device with
//! op-ffi-swift / op-ffi-jni binding it to Swift / Kotlin shells.

use op_core::{Currency, Money, Payment, PaymentMethod, RailKind, VaultRef};
use op_rails_card::{
    CardAcquirer,
    acquirer::{AuthRequest, CaptureRequest, RefundRequest, ThreeDsMode},
    hyperswitch::HyperswitchClient,
};
use uuid::Uuid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("HYPERSWITCH_API_KEY")
        .map_err(|_| "Set HYPERSWITCH_API_KEY to a Hyperswitch sandbox key")?;

    let acquirer = HyperswitchClient::new(HyperswitchClient::SANDBOX, api_key);

    // 1) Build the payment in OpenPay terms.
    let amount = Money::from_minor(10_00, Currency::USD); // $10.00
    let method = PaymentMethod::Vault(VaultRef::new("tok_test_card_visa"));
    let payment = Payment::new(amount, method.clone(), RailKind::Card);
    println!("Created payment {} for {}", payment.id, payment.amount);

    // 2) Authorize via the PSP (manual capture so we can demonstrate the
    //    full typestate flow).
    let auth_req = AuthRequest {
        amount,
        method,
        auto_capture: false,
        idempotency_key: Uuid::now_v7().simple().to_string(),
        three_ds: Some(ThreeDsMode::Skip),
        metadata: None,
    };
    let auth = acquirer.authorize(&auth_req)?;
    println!(
        "PSP auth: status={:?} psp_id={}",
        auth.status, auth.psp_payment_id
    );

    // 3) Move the OpenPay typestate forward. `authorize()` on Payment<Created>
    //    consumes self and yields Payment<Authorized> — the compiler now
    //    blocks any attempt to re-authorize or to refund before capture.
    let authorized = payment.authorize(auth.psp_payment_id.clone());

    // 4) Capture.
    let cap_req = CaptureRequest {
        psp_payment_id: auth.psp_payment_id.clone(),
        amount,
        idempotency_key: Uuid::now_v7().simple().to_string(),
    };
    let cap = acquirer.capture(&cap_req)?;
    println!("PSP capture: status={:?}", cap.status);

    let captured = authorized.capture(amount)?;
    println!("Payment {} captured: {}", captured.id, captured.captured);

    // 5) Refund the full amount.
    let refund_req = RefundRequest {
        psp_payment_id: auth.psp_payment_id.clone(),
        amount,
        reason: Some("Customer changed mind".into()),
        idempotency_key: Uuid::now_v7().simple().to_string(),
    };
    let refund = acquirer.refund(&refund_req)?;
    println!("PSP refund: status={:?}", refund.status);

    let refunded = captured.refund(amount)?;
    println!("Payment {} refunded: {}", refunded.id, refunded.refunded);

    Ok(())
}
