//! The [`CardAcquirer`] trait.
//!
//! Every PSP driver implements this trait. The orchestrator routes a
//! payment to a driver by holding `Box<dyn CardAcquirer>`. Adding a new
//! PSP is a matter of writing one impl block; no other code changes.
//!
//! ## Why not `async fn` in trait
//!
//! Rust 1.95 supports `async fn` in traits, but using it with `dyn`
//! requires the `return_type_notation` machinery that's still landing.
//! For now we keep the trait synchronous; drivers that need to do I/O
//! handle it internally (ureq's sync client is fine on a background
//! thread). When `async fn` in `dyn Trait` stabilizes cleanly we'll
//! migrate.

use op_core::{Money, PaymentMethod};
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Request to authorize (and optionally capture) a payment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthRequest {
    /// Amount and currency.
    pub amount: Money,
    /// How value moves. Must be a `Vault`, `Wallet`, or `Emv` variant.
    /// Other variants are rejected with [`Error::UnsupportedMethod`].
    pub method: PaymentMethod,
    /// True if funds should be auto-captured on a successful auth.
    /// False = manual capture; `OpenPay`'s typestate makes the caller
    /// invoke `capture()` separately.
    pub auto_capture: bool,
    /// Caller-supplied idempotency key. Drivers forward this to the PSP
    /// where supported. UUID v7 is a good default.
    pub idempotency_key: String,
    /// Optional 3DS authentication preference. None = use PSP default.
    pub three_ds: Option<ThreeDsMode>,
    /// Free-form metadata. Forwarded to the PSP as opaque key/value.
    pub metadata: Option<serde_json::Value>,
}

/// 3DS authentication preference.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreeDsMode {
    /// Force 3DS challenge if supported by the issuer.
    Required,
    /// Skip 3DS if the PSP permits.
    Skip,
}

/// Request to capture a previously authorized payment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRequest {
    /// PSP's payment id, as returned from `authorize`.
    pub psp_payment_id: String,
    /// Amount to capture. Must be ≤ originally authorized.
    pub amount: Money,
    /// Idempotency key for this capture call.
    pub idempotency_key: String,
}

/// Request to void (cancel) a pre-capture authorization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoidRequest {
    /// PSP's payment id to void.
    pub psp_payment_id: String,
    /// Reason code passed to the PSP.
    pub reason: VoidReason,
    /// Idempotency key.
    pub idempotency_key: String,
}

/// Why a payment was voided.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoidReason {
    /// Customer requested cancellation.
    RequestedByCustomer,
    /// Duplicate of another payment.
    Duplicate,
    /// Suspected fraud.
    Fraudulent,
    /// Some other reason (passed verbatim to PSP).
    Other,
}

/// Request to refund part or all of a captured payment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefundRequest {
    /// PSP's payment id.
    pub psp_payment_id: String,
    /// Amount to refund. Must be ≤ captured.
    pub amount: Money,
    /// Reason for refund (passed to PSP).
    pub reason: Option<String>,
    /// Idempotency key.
    pub idempotency_key: String,
}

/// The PSP's decision for an authorization request.
///
/// Drivers map their native status enums onto this shape so callers
/// can route uniformly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthDecision {
    /// PSP-issued payment id. Hold onto this for capture/void/refund.
    pub psp_payment_id: String,
    /// Normalized outcome.
    pub status: AuthStatus,
    /// PSP-specific status code, preserved for diagnostics.
    pub raw_status: String,
    /// Amount actually authorized (may be less than requested for
    /// partial-auth flows). None if the PSP doesn't report it.
    pub authorized_amount: Option<Money>,
    /// If `status == RequiresCustomerAction`, this carries the URL to
    /// redirect the customer to (e.g. 3DS challenge page).
    pub redirect_url: Option<String>,
    /// PSP error code, if the auth failed.
    pub error_code: Option<String>,
    /// Human-readable error message, if the auth failed.
    pub error_message: Option<String>,
}

/// Normalized authorization outcome.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthStatus {
    /// Funds authorized (manual capture) or settled (auto-capture).
    Approved,
    /// Funds held; waiting for capture call.
    AuthorizedAwaitingCapture,
    /// Funds captured and settled.
    Settled,
    /// PSP needs the customer to do something (3DS challenge, bank app
    /// redirect, OTP). See `redirect_url`.
    RequiresCustomerAction,
    /// PSP needs additional API calls before it can authorize
    /// (Hyperswitch's `requires_payment_method`/`requires_confirmation`).
    RequiresMerchantAction,
    /// Soft decline. Caller may retry with a different method or after
    /// a short delay.
    SoftDecline,
    /// Hard decline. Do not retry with the same method.
    HardDecline,
    /// Suspected fraud. Freeze and review.
    Fraud,
    /// Transient failure (network, timeout). Retry-safe.
    Transient,
}

impl AuthStatus {
    /// True if this status represents a terminal failure.
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::HardDecline | Self::Fraud)
    }

    /// True if it's safe to retry the same request.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::Transient | Self::SoftDecline)
    }

    /// True if funds have moved (or will move automatically).
    #[must_use]
    pub const fn funds_moved(self) -> bool {
        matches!(self, Self::Approved | Self::Settled)
    }
}

/// The generic card-acquirer interface. Every PSP driver implements this.
///
/// Methods are infallible at the type level (return `Result`) and
/// stateless from the caller's perspective. State lives in the PSP and
/// in the `psp_payment_id` strings that flow back to the caller.
pub trait CardAcquirer: Send + Sync {
    /// Driver name (e.g. `"hyperswitch"`, `"stripe"`).
    fn name(&self) -> &'static str;

    /// True if this driver can handle the given payment method.
    fn supports(&self, method: &PaymentMethod) -> bool;

    /// Authorize a payment. If `req.auto_capture` is true and the PSP
    /// approves, funds are captured immediately.
    fn authorize(&self, req: &AuthRequest) -> Result<AuthDecision>;

    /// Capture funds from a prior authorization.
    fn capture(&self, req: &CaptureRequest) -> Result<AuthDecision>;

    /// Void a pre-capture authorization.
    fn void(&self, req: &VoidRequest) -> Result<AuthDecision>;

    /// Refund a captured payment.
    fn refund(&self, req: &RefundRequest) -> Result<AuthDecision>;

    /// Confirm a payment that came back
    /// [`AuthStatus::RequiresCustomerAction`] after the customer
    /// completed the challenge out-of-band (3DS, bank app
    /// redirect, OTP). The PSP knows the challenge result by the
    /// time this is called; the driver fetches the current status.
    ///
    /// Default impl returns [`Error::UnsupportedMethod`] — PSPs
    /// that don't support post-challenge confirmation (rare in
    /// 2026 but possible for old gateways) keep the default and
    /// callers fall back to capturing on a settled webhook.
    ///
    /// # Errors
    /// See [`crate::Error`].
    fn confirm_after_challenge(
        &self,
        psp_payment_id: &str,
        idempotency_key: &str,
    ) -> Result<AuthDecision> {
        let _ = (psp_payment_id, idempotency_key);
        Err(crate::Error::UnsupportedMethod)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_status_failure_classification() {
        assert!(AuthStatus::HardDecline.is_failure());
        assert!(AuthStatus::Fraud.is_failure());
        assert!(!AuthStatus::SoftDecline.is_failure());
        assert!(!AuthStatus::Approved.is_failure());
        assert!(!AuthStatus::Transient.is_failure());
    }

    #[test]
    fn auth_status_retryable_classification() {
        assert!(AuthStatus::Transient.is_retryable());
        assert!(AuthStatus::SoftDecline.is_retryable());
        assert!(!AuthStatus::HardDecline.is_retryable());
        assert!(!AuthStatus::Fraud.is_retryable());
        assert!(!AuthStatus::Approved.is_retryable());
    }

    #[test]
    fn auth_status_funds_moved_classification() {
        assert!(AuthStatus::Approved.funds_moved());
        assert!(AuthStatus::Settled.funds_moved());
        assert!(!AuthStatus::AuthorizedAwaitingCapture.funds_moved());
        assert!(!AuthStatus::SoftDecline.funds_moved());
    }

    #[test]
    fn auth_status_categories_are_disjoint() {
        // No status should be both a failure and retryable.
        for s in [
            AuthStatus::Approved,
            AuthStatus::AuthorizedAwaitingCapture,
            AuthStatus::Settled,
            AuthStatus::RequiresCustomerAction,
            AuthStatus::RequiresMerchantAction,
            AuthStatus::SoftDecline,
            AuthStatus::HardDecline,
            AuthStatus::Fraud,
            AuthStatus::Transient,
        ] {
            assert!(
                !(s.is_failure() && s.is_retryable()),
                "status {s:?} is both failure and retryable"
            );
        }
    }
}
