//! Payment intent — the orchestrator's input.
//!
//! An intent is what the merchant code submits: "charge X for this
//! item with this method." It is **not** a payment yet — the
//! orchestrator turns it into one (or several rail attempts) and
//! returns an [`OrchestrationOutcome`](crate::OrchestrationOutcome).

use op_core::{Money, PaymentMethod};

use crate::idempotency::IdempotencyKey;

/// Hints to the [`Router`](crate::Router) about how to pick a rail.
///
/// All hints are optional. A router free of hints falls back to the
/// method-driven default (`Vault | Wallet | Emv → Card`,
/// `A2a | Qr → A2A`).
#[derive(Debug, Clone, Default)]
pub struct RoutingHints {
    /// ISO-3166-1 alpha-2 country code of the customer, if known.
    /// Used by the [`PolicyRouter`](crate::PolicyRouter) to prefer
    /// in-country instant rails over cross-border card auths
    /// (lower cost, lower latency).
    pub customer_country: Option<String>,

    /// ISO-3166-1 alpha-2 country code of the merchant.
    pub merchant_country: Option<String>,

    /// Three-letter MCC equivalent if the merchant has one (we don't
    /// model the full ISO 18245 MCC list here — operators set their
    /// own taxonomy via this string).
    pub merchant_category: Option<String>,

    /// Opaque BIN bucket (e.g. `"visa-credit-consumer"`,
    /// `"mastercard-debit-business"`). Lets policy routers prefer
    /// rails by issuer characteristics without us re-implementing
    /// BIN lookup here.
    pub bin_bucket: Option<String>,

    /// If the customer is enrolled for 3DS, set this so the router
    /// can prefer card networks that support 3DS frictionless flow.
    pub three_ds_enrolled: bool,

    /// If true, the customer has indicated a preference for paying
    /// directly from their bank account (e.g. selected "Pay by
    /// Bank" at checkout). Causes the router to prefer A2A rails.
    pub prefer_a2a: bool,
}

/// A payment intent — the unit of work the orchestrator processes.
///
/// Idempotency-keyed; two intents with the same key are treated as
/// the same request and the cached outcome is returned for the
/// second one.
#[derive(Debug, Clone)]
pub struct PaymentIntent {
    /// Caller-supplied idempotency key. Must be unique per logical
    /// request; reuse with a different body is rejected as
    /// [`Error::IdempotencyMismatch`](crate::Error::IdempotencyMismatch).
    pub idempotency_key: IdempotencyKey,

    /// Amount to charge.
    pub amount: Money,

    /// How the customer is paying — vault token, EMV terminal data,
    /// A2A bank account reference, etc.
    pub method: PaymentMethod,

    /// Hints to the router.
    pub hints: RoutingHints,

    /// Free-form metadata to round-trip back on the outcome.
    /// Typically merchant order id, line items, customer id.
    pub metadata: Vec<(String, String)>,
}

impl PaymentIntent {
    /// Construct a minimal intent. Hints default; metadata is empty.
    pub fn new(idempotency_key: IdempotencyKey, amount: Money, method: PaymentMethod) -> Self {
        Self {
            idempotency_key,
            amount,
            method,
            hints: RoutingHints::default(),
            metadata: Vec::new(),
        }
    }

    /// Builder: set routing hints.
    #[must_use]
    pub fn with_hints(mut self, hints: RoutingHints) -> Self {
        self.hints = hints;
        self
    }

    /// Builder: append a metadata pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// A deterministic signature of the body fields that must match
    /// across retries with the same idempotency key. Hash-like.
    /// Used by [`IdempotencyStore`](crate::IdempotencyStore) to
    /// detect [`Error::IdempotencyMismatch`](crate::Error::IdempotencyMismatch).
    pub fn body_signature(&self) -> String {
        // Compact stringification — same idempotency key but
        // different amount/method/country/etc. → different signature
        // → IdempotencyMismatch. We deliberately do NOT include
        // metadata (it's free-form and the merchant may legitimately
        // add e.g. a new client-side timestamp on retry).
        format!(
            "amount={}|currency={}|method={}|cc={}|mc={}",
            self.amount.minor_units,
            self.amount.currency.code(),
            method_signature(&self.method),
            self.hints.customer_country.as_deref().unwrap_or(""),
            self.hints.merchant_country.as_deref().unwrap_or(""),
        )
    }
}

fn method_signature(m: &PaymentMethod) -> String {
    // op_core::PaymentMethod doesn't impl Display; we synthesize a
    // signature using the variant tag plus an opaque-but-stable
    // discriminator. Token strings and account-number digests are
    // exactly the kind of thing the merchant might mutate by accident
    // on retry, so we DO include them.
    //
    // Note: A2aKey is an enum (Upi/Pix/Iban/UsAch); we project each
    // variant to a single string for signature purposes. This is not
    // a security boundary — same intent must yield same string, that
    // is all.
    match m {
        PaymentMethod::Vault(v) => format!("Vault:{}", v.as_str()),
        PaymentMethod::Wallet(_) => "Wallet:<opaque>".to_owned(),
        PaymentMethod::Emv(_) => "Emv:<binary>".to_owned(),
        PaymentMethod::A2a(k) => format!("A2a:{}", a2a_key_signature(k)),
        PaymentMethod::Qr(s) => format!("Qr:{s}"),
        PaymentMethod::Crypto(a) => format!("Crypto:{}:{}", a.chain, a.address),
        // RawPan reaching the orchestrator means a vault contract broke.
        // We project to a stable, non-revealing string so idempotency
        // matching is still deterministic; PCI scope already mandates
        // upstream audit logs catch this separately.
        PaymentMethod::RawPan(_) => "RawPan:<pci-scoped>".to_owned(),
    }
}

fn a2a_key_signature(k: &op_core::A2aKey) -> String {
    use op_core::A2aKey::*;
    match k {
        Upi(h) => format!("upi:{h}"),
        Pix(h) => format!("pix:{h}"),
        Iban(i) => format!("iban:{i}"),
        UsAch { routing, account } => format!("usach:{routing}:{account}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, VaultRef};

    fn intent_a() -> PaymentIntent {
        PaymentIntent::new(
            IdempotencyKey::from("key-1"),
            Money::from_minor(1234, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_abc")),
        )
    }

    #[test]
    fn signature_is_stable_for_identical_intents() {
        let a = intent_a();
        let b = intent_a();
        assert_eq!(a.body_signature(), b.body_signature());
    }

    #[test]
    fn signature_changes_when_amount_changes() {
        let a = intent_a();
        let mut b = intent_a();
        b.amount = Money::from_minor(9999, Currency::USD);
        assert_ne!(a.body_signature(), b.body_signature());
    }

    #[test]
    fn signature_changes_when_method_changes() {
        let a = intent_a();
        let b = PaymentIntent::new(
            IdempotencyKey::from("key-1"),
            Money::from_minor(1234, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_OTHER")),
        );
        assert_ne!(a.body_signature(), b.body_signature());
    }

    #[test]
    fn signature_does_not_change_with_metadata() {
        let a = intent_a();
        let b = intent_a().with_metadata("order_id", "ORD-42");
        assert_eq!(a.body_signature(), b.body_signature());
    }

    #[test]
    fn builder_pattern_compiles() {
        let intent = PaymentIntent::new(
            IdempotencyKey::from("k"),
            Money::from_minor(100, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_v7_x")),
        )
        .with_hints(RoutingHints {
            customer_country: Some("US".into()),
            ..Default::default()
        })
        .with_metadata("k1", "v1")
        .with_metadata("k2", "v2");
        assert_eq!(intent.metadata.len(), 2);
        assert_eq!(intent.hints.customer_country.as_deref(), Some("US"));
    }
}
