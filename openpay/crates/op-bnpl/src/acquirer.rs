//! The generic [`BnplAcquirer`] trait + initiate/authorize/capture/refund
//! value types.
//!
//! Modelled like `op-rails-card::CardAcquirer` but with the BNPL-specific
//! intermediate states: between `initiate` (session creation) and `capture`
//! sits `authorize` (the consumer-token verification step).

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::intent::BnplIntent;
use crate::lifecycle::{BnplProvider, InstalmentPlan};

/// What the provider returns after `initiate`.
///
/// The merchant-side flow up to this point:
///
/// 1. Merchant calls `initiate(&intent)`.
/// 2. Provider returns an [`InitiatedSession`] with a `provider_ref`
///    (Affirm `checkout_token`, Klarna `session_id`, Afterpay `token`)
///    and a `redirect_url` (or client-token for embedded flows).
/// 3. Merchant redirects browser to `redirect_url`.
/// 4. Consumer accepts; browser returns to merchant with a
///    `consumer_token` (Affirm checkout_token, Klarna authorization_token,
///    Afterpay orderToken in the query string).
/// 5. Merchant calls `authorize(&session, &consumer_token)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitiatedSession {
    /// Which provider issued this session.
    pub provider: BnplProvider,
    /// Opaque provider-issued session/checkout id. The merchant carries
    /// this across the redirect-to-consumer boundary.
    pub provider_ref: String,
    /// URL to redirect the consumer to. For embedded widgets (Klarna
    /// JS SDK, Afterpay Express), the SDK consumes this internally.
    pub redirect_url: Option<String>,
    /// Provider-issued client token for embedded flows (Klarna).
    /// `None` for redirect-based flows.
    pub client_token: Option<String>,
    /// When the session expires. After this, `authorize` will fail.
    pub expires_at: Option<DateTime<Utc>>,
}

/// What the provider returns after `authorize`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthorizedCheckout {
    /// Which provider.
    pub provider: BnplProvider,
    /// Opaque provider-issued id (Affirm `charge_id`, Klarna `order_id`,
    /// Afterpay `id`).
    pub provider_ref: String,
    /// Amount the provider locked in. May be less than requested for
    /// partial-approval flows (Klarna and Affirm both occasionally
    /// offer a smaller line of credit than the cart total).
    pub authorized_amount: Money,
    /// Instalment plan the consumer accepted.
    pub plan: InstalmentPlan,
}

/// What the provider returns after `capture`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapturedCheckout {
    /// Which provider.
    pub provider: BnplProvider,
    /// Opaque provider-issued id. Same value as `AuthorizedCheckout`
    /// for Affirm/Klarna; Afterpay may issue a separate `paymentId`
    /// at capture time.
    pub provider_ref: String,
    /// Amount captured.
    pub amount: Money,
    /// Provider's settlement-batch reference, if known at capture time.
    pub settlement_ref: Option<String>,
}

/// What the provider returns after `refund`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefundedCheckout {
    /// Which provider.
    pub provider: BnplProvider,
    /// Provider-issued id of the original charge.
    pub provider_ref: String,
    /// Provider-issued refund id.
    pub refund_ref: String,
    /// Amount refunded.
    pub amount: Money,
}

/// The generic BNPL acquirer interface.
///
/// Implemented by [`crate::AffirmAcquirer`], [`crate::KlarnaAcquirer`],
/// and [`crate::AfterpayAcquirer`]. The orchestrator holds a
/// `Box<dyn BnplAcquirer>` and routes by inspecting the intent's
/// shipping country + currency + amount band.
#[async_trait]
pub trait BnplAcquirer: Send + Sync {
    /// Which provider this driver speaks to.
    fn provider(&self) -> BnplProvider;

    /// Create a session / checkout token. Step 1 of 4.
    async fn initiate(&self, intent: &BnplIntent) -> Result<InitiatedSession>;

    /// Verify the consumer-completion token and lock in funds. Step 3
    /// of 4. The `consumer_token` is whatever the provider redirects
    /// back to the merchant with (query-string parameter / postMessage
    /// payload / JS SDK callback value).
    async fn authorize(
        &self,
        session: &InitiatedSession,
        consumer_token: &str,
    ) -> Result<AuthorizedCheckout>;

    /// Capture funds. Step 4 of 4. `amount = None` means full
    /// authorized amount; `Some(_)` means partial.
    async fn capture(
        &self,
        auth: &AuthorizedCheckout,
        amount: Option<Money>,
    ) -> Result<CapturedCheckout>;

    /// Refund part or all of a captured checkout.
    async fn refund(
        &self,
        captured: &CapturedCheckout,
        amount: Money,
    ) -> Result<RefundedCheckout>;

    /// Void a pre-capture authorization. Releases the provider-side
    /// reservation; consumer is not charged.
    async fn void(&self, auth: &AuthorizedCheckout) -> Result<()>;
}
