//! Checkout intent: cart contents + parties + idempotency.
//!
//! All three providers want roughly the same payload at session
//! creation, with provider-specific field name remapping. Modelling the
//! intent once and translating per-provider keeps the orchestrator
//! provider-agnostic.

use std::collections::BTreeMap;

use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

/// Caller-supplied idempotency key.
///
/// Stripe/Adyen convention: a UUID v4 (or similar) the merchant
/// generates per logical request. The acquirer caches the *outcome*
/// keyed by this string so retries don't double-charge the consumer.
///
/// `op-bnpl` defines its own newtype rather than depending on
/// `op-orchestrator::IdempotencyKey` to keep this crate's dependency
/// graph clean — the orchestrator imports `op-bnpl`, not the other
/// way around.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdempotencyKey(String);

impl IdempotencyKey {
    /// Construct from any string.
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<T: Into<String>> From<T> for IdempotencyKey {
    fn from(s: T) -> Self {
        Self::new(s)
    }
}

/// A single line item in the consumer's cart.
///
/// Affirm and Klarna both require line-item granularity (Affirm uses
/// `items`, Klarna uses `order_lines`); Afterpay accepts them as
/// `items[]` too. Modelling once and shape-shifting per-provider is
/// less error-prone than three near-duplicate structs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineItem {
    /// Human-readable display name.
    pub name: String,
    /// Optional merchant SKU.
    pub sku: Option<String>,
    /// Quantity (whole units; BNPL providers don't accept fractional).
    pub quantity: u32,
    /// Unit price.
    pub unit_price: Money,
    /// Quantity × unit_price, redundant by design so the caller's
    /// rounding is authoritative.
    pub total_amount: Money,
}

/// Shipping destination. BNPL providers underwrite partly on shipping
/// address (anti-fraud, jurisdiction).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShippingInfo {
    /// Recipient name.
    pub name: String,
    /// First street line.
    pub line1: String,
    /// Optional second street line.
    pub line2: Option<String>,
    /// City / locality.
    pub city: String,
    /// State / province / region (ISO 3166-2 subdivision code if known).
    pub region: String,
    /// Postal / ZIP code.
    pub postal_code: String,
    /// ISO 3166-1 alpha-2 country code.
    pub country: String,
}

/// Billing address. May differ from shipping (especially for
/// gift purchases).
pub type BillingInfo = ShippingInfo;

/// Consumer-identifying hint passed to the provider for underwriting.
///
/// Providers verify these against their own KYC and credit records.
/// The merchant supplies what it knows; the provider does not require
/// any single field but more data gives a better approval rate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumerInfo {
    /// Email (canonical identifier across all three providers).
    pub email: String,
    /// E.164 phone number.
    pub phone: Option<String>,
    /// Given (first) name.
    pub given_name: Option<String>,
    /// Family (last) name.
    pub family_name: Option<String>,
    /// Date of birth in ISO 8601 (YYYY-MM-DD). Required by Afterpay's
    /// age-gate; optional for Affirm/Klarna.
    pub date_of_birth: Option<String>,
}

/// Merchant-controlled redirect endpoints. After the consumer
/// accepts or cancels the provider's flow, the provider redirects
/// the browser back to one of these.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedirectUrls {
    /// Where the provider sends the browser on consumer acceptance.
    pub success: String,
    /// Where the provider sends the browser on consumer cancel.
    pub cancel: String,
    /// Where the provider sends the browser on hard failure (declined,
    /// timeout). Falls back to `cancel` if absent.
    pub failure: Option<String>,
}

/// A BNPL checkout intent.
///
/// Provider-neutral. The provider-specific acquirer consumes this and
/// emits the right JSON shape over the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BnplIntent {
    /// Total amount (sum of `line_items` + shipping + taxes if any).
    pub amount: Money,
    /// Currency. Redundant with `amount.currency` but explicit for
    /// readability and the rare cross-currency-checkout case.
    pub currency: Currency,
    /// Cart contents.
    pub line_items: Vec<LineItem>,
    /// Shipping destination.
    pub shipping: ShippingInfo,
    /// Billing address.
    pub billing: BillingInfo,
    /// Consumer identity hint.
    pub consumer: ConsumerInfo,
    /// Idempotency key.
    pub idempotency_key: IdempotencyKey,
    /// Browser redirect endpoints.
    pub redirect_urls: RedirectUrls,
    /// Free-form metadata. Each provider forwards a different subset
    /// as opaque key/value pairs.
    pub metadata: BTreeMap<String, String>,
}

impl BnplIntent {
    /// Sanity-check the intent: positive amount, non-empty cart, line
    /// totals in the same currency.
    ///
    /// # Errors
    /// [`crate::Error::InvalidIntent`] on any violation.
    pub fn validate(&self) -> crate::Result<()> {
        if !self.amount.is_positive() {
            return Err(crate::Error::InvalidIntent(
                "amount must be positive".into(),
            ));
        }
        if self.amount.currency != self.currency {
            return Err(crate::Error::InvalidIntent(
                "amount.currency != intent.currency".into(),
            ));
        }
        if self.line_items.is_empty() {
            return Err(crate::Error::InvalidIntent(
                "line_items must not be empty".into(),
            ));
        }
        for li in &self.line_items {
            if li.unit_price.currency != self.currency
                || li.total_amount.currency != self.currency
            {
                return Err(crate::Error::InvalidIntent(format!(
                    "line item {:?} currency mismatch",
                    li.name
                )));
            }
            if li.quantity == 0 {
                return Err(crate::Error::InvalidIntent(format!(
                    "line item {:?} has zero quantity",
                    li.name
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_li(currency: Currency) -> LineItem {
        LineItem {
            name: "Widget".into(),
            sku: Some("W-1".into()),
            quantity: 2,
            unit_price: Money::from_minor(5_000, currency),
            total_amount: Money::from_minor(10_000, currency),
        }
    }

    fn sample_intent() -> BnplIntent {
        BnplIntent {
            amount: Money::from_minor(10_000, Currency::USD),
            currency: Currency::USD,
            line_items: vec![sample_li(Currency::USD)],
            shipping: ShippingInfo {
                name: "Alice".into(),
                line1: "1 Market St".into(),
                line2: None,
                city: "San Francisco".into(),
                region: "CA".into(),
                postal_code: "94105".into(),
                country: "US".into(),
            },
            billing: ShippingInfo {
                name: "Alice".into(),
                line1: "1 Market St".into(),
                line2: None,
                city: "San Francisco".into(),
                region: "CA".into(),
                postal_code: "94105".into(),
                country: "US".into(),
            },
            consumer: ConsumerInfo {
                email: "alice@example.com".into(),
                phone: Some("+14155551212".into()),
                given_name: Some("Alice".into()),
                family_name: Some("Smith".into()),
                date_of_birth: Some("1990-01-01".into()),
            },
            idempotency_key: IdempotencyKey::new("idem-1"),
            redirect_urls: RedirectUrls {
                success: "https://m.example/ok".into(),
                cancel: "https://m.example/cancel".into(),
                failure: None,
            },
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn valid_intent_passes() {
        assert!(sample_intent().validate().is_ok());
    }

    #[test]
    fn zero_amount_rejected() {
        let mut i = sample_intent();
        i.amount = Money::from_minor(0, Currency::USD);
        assert!(i.validate().is_err());
    }

    #[test]
    fn empty_cart_rejected() {
        let mut i = sample_intent();
        i.line_items.clear();
        assert!(i.validate().is_err());
    }

    #[test]
    fn line_item_currency_mismatch_rejected() {
        let mut i = sample_intent();
        i.line_items[0].unit_price = Money::from_minor(5_000, Currency::EUR);
        assert!(i.validate().is_err());
    }

    #[test]
    fn zero_quantity_rejected() {
        let mut i = sample_intent();
        i.line_items[0].quantity = 0;
        assert!(i.validate().is_err());
    }

    #[test]
    fn idempotency_key_round_trips() {
        let k = IdempotencyKey::from("abc-123");
        assert_eq!(k.as_str(), "abc-123");
    }

    #[test]
    fn currency_mismatch_amount_vs_intent_rejected() {
        let mut i = sample_intent();
        i.amount = Money::from_minor(10_000, Currency::EUR);
        assert!(i.validate().is_err());
    }
}
