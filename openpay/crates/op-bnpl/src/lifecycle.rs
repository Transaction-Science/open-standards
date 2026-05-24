//! BNPL checkout typestate machine.
//!
//! Parallel to [`op_core::Payment<S>`] but with BNPL-specific
//! intermediate states. The provider's view of the world is:
//!
//! ```text
//!  Initiated ──consumer-accepts──▶ Approved ──capture──▶ Captured
//!     │                              │                       │
//!     │                              │                       └─settle─▶ Settled
//!     │                              │                                     │
//!     │                              │                                     └─refund─▶ Refunded
//!     │                              │
//!     │                              └─void──▶ Voided
//!     │
//!     └─expire──▶ Expired
//! ```
//!
//! Each state is a zero-sized marker; `BnplCheckout<S>` shares the
//! same memory layout regardless of `S` and the compiler erases the
//! state at codegen.

use chrono::{DateTime, Utc};
use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::intent::BnplIntent;

/// Which BNPL provider underwrites this checkout.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BnplProvider {
    /// Affirm. US-centric, deep checkout integration.
    Affirm,
    /// Klarna. Strong in EU + growing US presence. Multi-region
    /// API base URLs.
    Klarna,
    /// Afterpay (US/AU/NZ/CA) and Clearpay (UK/EU). Same API surface,
    /// different brand.
    AfterpayClearpay,
}

impl BnplProvider {
    /// Stable lowercase identifier (used in logs, metrics, and
    /// webhook routing).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Affirm => "affirm",
            Self::Klarna => "klarna",
            Self::AfterpayClearpay => "afterpay_clearpay",
        }
    }
}

/// How often an instalment is due.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstalmentInterval {
    /// Two weeks apart — Afterpay's "Pay-in-4" default.
    Biweekly,
    /// One month apart — Klarna's "Pay-in-3" / Affirm's monthly plans.
    Monthly,
    /// One week apart (rare; some EU Klarna products).
    Weekly,
}

/// A concrete instalment plan offered by the provider.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalmentPlan {
    /// Number of instalments (e.g. 4 for Pay-in-4, 3 for Pay-in-3).
    pub num_instalments: u8,
    /// Amount per instalment. (instalment_amount × num_instalments may
    /// differ from total by a few minor units due to rounding; the
    /// provider's last-instalment adjustment is authoritative.)
    pub instalment_amount: Money,
    /// When the first instalment is due (UTC).
    pub first_instalment_due: DateTime<Utc>,
    /// Cadence.
    pub interval: InstalmentInterval,
}

impl InstalmentPlan {
    /// Construct.
    #[must_use]
    pub const fn new(
        num_instalments: u8,
        instalment_amount: Money,
        first_instalment_due: DateTime<Utc>,
        interval: InstalmentInterval,
    ) -> Self {
        Self {
            num_instalments,
            instalment_amount,
            first_instalment_due,
            interval,
        }
    }
}

/// Sealed marker — only states defined in this module satisfy it.
pub trait BnplState: sealed::Sealed {
    /// Diagnostic name.
    const NAME: &'static str;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! state {
    ($name:ident, $literal:expr) => {
        /// Checkout state marker. See module docs for the state diagram.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl BnplState for $name {
            const NAME: &'static str = $literal;
        }
    };
}

state!(Initiated, "Initiated");
state!(Approved, "Approved");
state!(Captured, "Captured");
state!(Settled, "Settled");
state!(Refunded, "Refunded");

/// A BNPL checkout, parameterized by lifecycle state.
///
/// `provider_ref` is the provider-issued id (Affirm: `charge_id`;
/// Klarna: `order_id`; Afterpay: `token` / `order_id`). Carries
/// through every state for refund / webhook correlation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BnplCheckout<S: BnplState> {
    /// Provider type.
    pub provider: BnplProvider,
    /// Provider-issued opaque identifier. `None` in `Initiated` until
    /// the provider returns a session token; populated thereafter.
    pub provider_ref: Option<String>,
    /// The originating intent, retained for refund correlation and
    /// for idempotency-replay payload signatures.
    pub intent: BnplIntent,
    /// Authorized amount (≤ `intent.amount`). Zero in `Initiated`.
    pub authorized: Money,
    /// Captured amount (≤ `authorized`). Zero in `Initiated` /
    /// `Approved`.
    pub captured: Money,
    /// Refunded amount (≤ `captured`). Zero before `Refunded`.
    pub refunded: Money,
    /// Instalment plan offered. `None` until `Approved`.
    pub plan: Option<InstalmentPlan>,
    /// Phantom state.
    #[serde(skip)]
    _state: core::marker::PhantomData<S>,
}

impl BnplCheckout<Initiated> {
    /// Construct a fresh checkout in the `Initiated` state.
    #[must_use]
    pub const fn new(provider: BnplProvider, intent: BnplIntent) -> Self {
        let zero = Money::zero(intent.amount.currency);
        Self {
            provider,
            provider_ref: None,
            intent,
            authorized: zero,
            captured: zero,
            refunded: zero,
            plan: None,
            _state: core::marker::PhantomData,
        }
    }

    /// Record the session token returned by the provider's
    /// `/sessions` (Klarna) or `/checkouts` (Affirm, Afterpay)
    /// endpoint. Stays in `Initiated` — consumer flow has not yet
    /// completed.
    #[must_use]
    pub fn with_provider_ref(mut self, provider_ref: String) -> Self {
        self.provider_ref = Some(provider_ref);
        self
    }

    /// Consumer accepted the provider's offer. Transition to
    /// `Approved`. `authorized` is the amount the provider locked in
    /// (usually equal to `intent.amount`, occasionally a smaller
    /// partial offer).
    #[must_use]
    pub fn approve(
        self,
        provider_ref: String,
        authorized: Money,
        plan: InstalmentPlan,
    ) -> BnplCheckout<Approved> {
        BnplCheckout {
            provider: self.provider,
            provider_ref: Some(provider_ref),
            intent: self.intent,
            authorized,
            captured: self.captured,
            refunded: self.refunded,
            plan: Some(plan),
            _state: core::marker::PhantomData,
        }
    }
}

impl BnplCheckout<Approved> {
    /// Capture up to the authorized amount.
    ///
    /// # Errors
    /// - `CurrencyMismatch` if `amount` is in a different currency.
    /// - `IllegalTransition` (via `op_core::Error`) if amount > authorized.
    pub fn capture(self, amount: Money) -> Result<BnplCheckout<Captured>> {
        let new_captured = self.captured.checked_add(amount)?;
        if new_captured.minor_units > self.authorized.minor_units {
            return Err(crate::Error::Core(op_core::Error::IllegalTransition {
                from: <Approved as BnplState>::NAME,
                to: <Captured as BnplState>::NAME,
            }));
        }
        Ok(BnplCheckout {
            provider: self.provider,
            provider_ref: self.provider_ref,
            intent: self.intent,
            authorized: self.authorized,
            captured: new_captured,
            refunded: self.refunded,
            plan: self.plan,
            _state: core::marker::PhantomData,
        })
    }
}

impl BnplCheckout<Captured> {
    /// Mark the checkout as settled by the provider's settlement run.
    /// No money moves here; this is bookkeeping for the provider's
    /// payout-to-merchant event.
    #[must_use]
    pub fn settle(self) -> BnplCheckout<Settled> {
        BnplCheckout {
            provider: self.provider,
            provider_ref: self.provider_ref,
            intent: self.intent,
            authorized: self.authorized,
            captured: self.captured,
            refunded: self.refunded,
            plan: self.plan,
            _state: core::marker::PhantomData,
        }
    }

    /// Refund part or all of the captured amount before settlement.
    /// (Most providers also accept refund-after-settle; the typestate
    /// allows both Captured→Refunded and Settled→Refunded.)
    ///
    /// # Errors
    /// `IllegalTransition` if refund > captured.
    pub fn refund(self, amount: Money) -> Result<BnplCheckout<Refunded>> {
        refund_internal(
            self.provider,
            self.provider_ref,
            self.intent,
            self.authorized,
            self.captured,
            self.refunded,
            self.plan,
            amount,
        )
    }
}

impl BnplCheckout<Settled> {
    /// Refund a settled checkout. Provider claws back funds from the
    /// merchant on the next settlement run.
    ///
    /// # Errors
    /// `IllegalTransition` if refund > captured.
    pub fn refund(self, amount: Money) -> Result<BnplCheckout<Refunded>> {
        refund_internal(
            self.provider,
            self.provider_ref,
            self.intent,
            self.authorized,
            self.captured,
            self.refunded,
            self.plan,
            amount,
        )
    }
}

/// Internal: refund pathway shared between `Captured` and `Settled`.
#[allow(clippy::too_many_arguments)]
fn refund_internal(
    provider: BnplProvider,
    provider_ref: Option<String>,
    intent: BnplIntent,
    authorized: Money,
    captured: Money,
    refunded: Money,
    plan: Option<InstalmentPlan>,
    amount: Money,
) -> Result<BnplCheckout<Refunded>> {
    let new_refunded = refunded.checked_add(amount)?;
    if new_refunded.minor_units > captured.minor_units {
        return Err(crate::Error::Core(op_core::Error::IllegalTransition {
            from: <Captured as BnplState>::NAME,
            to: <Refunded as BnplState>::NAME,
        }));
    }
    Ok(BnplCheckout {
        provider,
        provider_ref,
        intent,
        authorized,
        captured,
        refunded: new_refunded,
        plan,
        _state: core::marker::PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::{
        BillingInfo, BnplIntent, ConsumerInfo, IdempotencyKey, LineItem, RedirectUrls, ShippingInfo,
    };
    use chrono::TimeZone;
    use op_core::Currency;
    use std::collections::BTreeMap;

    fn intent() -> BnplIntent {
        let usd = Currency::USD;
        BnplIntent {
            amount: Money::from_minor(10_000, usd),
            currency: usd,
            line_items: vec![LineItem {
                name: "x".into(),
                sku: None,
                quantity: 1,
                unit_price: Money::from_minor(10_000, usd),
                total_amount: Money::from_minor(10_000, usd),
            }],
            shipping: ShippingInfo {
                name: "A".into(),
                line1: "1".into(),
                line2: None,
                city: "c".into(),
                region: "r".into(),
                postal_code: "p".into(),
                country: "US".into(),
            },
            billing: BillingInfo {
                name: "A".into(),
                line1: "1".into(),
                line2: None,
                city: "c".into(),
                region: "r".into(),
                postal_code: "p".into(),
                country: "US".into(),
            },
            consumer: ConsumerInfo {
                email: "a@b.com".into(),
                phone: None,
                given_name: None,
                family_name: None,
                date_of_birth: None,
            },
            idempotency_key: IdempotencyKey::from("k1"),
            redirect_urls: RedirectUrls {
                success: "s".into(),
                cancel: "c".into(),
                failure: None,
            },
            metadata: BTreeMap::new(),
        }
    }

    fn plan() -> InstalmentPlan {
        InstalmentPlan::new(
            4,
            Money::from_minor(2_500, Currency::USD),
            Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap(),
            InstalmentInterval::Biweekly,
        )
    }

    #[test]
    fn happy_path_initiated_to_settled() {
        let chk = BnplCheckout::new(BnplProvider::Affirm, intent());
        let chk = chk
            .with_provider_ref("session_1".into())
            .approve(
                "charge_abc".into(),
                Money::from_minor(10_000, Currency::USD),
                plan(),
            )
            .capture(Money::from_minor(10_000, Currency::USD))
            .unwrap()
            .settle();
        assert_eq!(chk.captured.minor_units, 10_000);
        assert_eq!(chk.provider_ref.as_deref(), Some("charge_abc"));
    }

    #[test]
    fn refund_from_captured() {
        let chk = BnplCheckout::new(BnplProvider::Klarna, intent())
            .approve(
                "order_x".into(),
                Money::from_minor(10_000, Currency::USD),
                plan(),
            )
            .capture(Money::from_minor(10_000, Currency::USD))
            .unwrap()
            .refund(Money::from_minor(3_000, Currency::USD))
            .unwrap();
        assert_eq!(chk.refunded.minor_units, 3_000);
    }

    #[test]
    fn refund_from_settled() {
        let chk = BnplCheckout::new(BnplProvider::AfterpayClearpay, intent())
            .approve(
                "token_y".into(),
                Money::from_minor(10_000, Currency::USD),
                plan(),
            )
            .capture(Money::from_minor(10_000, Currency::USD))
            .unwrap()
            .settle()
            .refund(Money::from_minor(5_000, Currency::USD))
            .unwrap();
        assert_eq!(chk.refunded.minor_units, 5_000);
    }

    #[test]
    fn overcapture_rejected() {
        let chk = BnplCheckout::new(BnplProvider::Affirm, intent()).approve(
            "c".into(),
            Money::from_minor(10_000, Currency::USD),
            plan(),
        );
        let r = chk.capture(Money::from_minor(15_000, Currency::USD));
        assert!(matches!(
            r,
            Err(crate::Error::Core(op_core::Error::IllegalTransition { .. }))
        ));
    }

    #[test]
    fn overrefund_rejected() {
        let chk = BnplCheckout::new(BnplProvider::Klarna, intent())
            .approve(
                "x".into(),
                Money::from_minor(10_000, Currency::USD),
                plan(),
            )
            .capture(Money::from_minor(4_000, Currency::USD))
            .unwrap();
        let r = chk.refund(Money::from_minor(5_000, Currency::USD));
        assert!(matches!(
            r,
            Err(crate::Error::Core(op_core::Error::IllegalTransition { .. }))
        ));
    }

    #[test]
    fn provider_string_identifiers_stable() {
        assert_eq!(BnplProvider::Affirm.as_str(), "affirm");
        assert_eq!(BnplProvider::Klarna.as_str(), "klarna");
        assert_eq!(BnplProvider::AfterpayClearpay.as_str(), "afterpay_clearpay");
    }
}
