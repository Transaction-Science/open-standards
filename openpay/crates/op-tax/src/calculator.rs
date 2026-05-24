//! [`TaxCalculator`] trait + the shared request/response types.
//!
//! Every backend — native or commercial — implements `TaxCalculator`.
//! Operators wire the backend they trust; the rest of OpenPay never
//! sees the difference.

use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, Utc};
use op_core::Money;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

use crate::category::ProductTaxCategory;
use crate::error::Result;
use crate::exemption::ExemptionCertificate;
use crate::jurisdiction::Jurisdiction;

/// Customer classification — drives reverse-charge eligibility,
/// exemption applicability, and per-jurisdiction B2B rate logic.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CustomerType {
    /// Consumer (B2C). Typical retail flow; full tax collected at PoS.
    Consumer,
    /// Business with a registered tax identifier (B2B). Eligible for
    /// EU reverse-charge across borders; may present resale or
    /// exemption certificates.
    Business {
        /// VAT / GST / sales-tax registration number, as a string.
        /// Format is jurisdiction-specific (`DE123456789`, `GB123456789`,
        /// `12-3456789` for US EINs); we do not validate.
        tax_id: String,
    },
    /// Non-profit / charity / government — usually fully exempt.
    Exempt,
}

/// A single taxable line item.
///
/// Maps roughly one-to-one with Avalara's `LineItemModel`, Stripe Tax's
/// `line_item`, and Vertex's `LineItem`. The names are deliberately
/// adapter-friendly so the conversion code in each backend module is
/// boring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxableLine {
    /// Caller-supplied identifier — stable across retries so a
    /// `TaxResult.per_line` lookup is unambiguous.
    pub line_id: String,
    /// The net (tax-exclusive) or gross (tax-inclusive — see the
    /// jurisdiction's `TaxBase`) line amount.
    pub amount: Money,
    /// Product / service category. Drives rate overrides.
    pub category: ProductTaxCategory,
    /// Origin jurisdiction. Only required for origin-based US states
    /// (a handful — AZ, IL, MO, NM, OH, PA, TN, TX, UT, VA, plus
    /// California's complicated hybrid) and for cross-border duty
    /// detection. `None` is fine for destination-based states.
    pub ship_from: Option<Jurisdiction>,
    /// Destination jurisdiction. Determines what authority is owed.
    pub ship_to: Jurisdiction,
}

/// Contextual data that applies to every line in a calculate call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxContext {
    /// Date the transaction is deemed to occur. Drives rate selection
    /// for jurisdictions whose rates have changed mid-year.
    pub transaction_date: NaiveDate,
    /// Classification of the buyer.
    pub customer_type: CustomerType,
    /// Resale / exemption certificates the buyer has produced.
    pub exemption_certs: Vec<ExemptionCertificate>,
    /// Jurisdictions in which the seller has economic or physical
    /// nexus. Lines shipping to a jurisdiction NOT in this set are
    /// zero-rated (no obligation to collect).
    pub nexus_jurisdictions: HashSet<Jurisdiction>,
}

impl TaxContext {
    /// Convenience constructor for the common consumer case.
    #[must_use]
    pub fn consumer(transaction_date: NaiveDate) -> Self {
        Self {
            transaction_date,
            customer_type: CustomerType::Consumer,
            exemption_certs: Vec::new(),
            nexus_jurisdictions: HashSet::new(),
        }
    }

    /// Builder: declare nexus in a jurisdiction.
    #[must_use]
    pub fn with_nexus(mut self, j: Jurisdiction) -> Self {
        self.nexus_jurisdictions.insert(j);
        self
    }

    /// Builder: declare nexus everywhere the calculator wants — the
    /// "I'm shipping everywhere" case. The calculator will treat
    /// every line's destination as nexused.
    #[must_use]
    pub const fn with_nexus_everywhere(self) -> Self {
        // Sentinel: `nexus_jurisdictions.is_empty()` is the marker
        // the native calculator interprets as "no nexus filtering".
        // We document but don't store anything explicit here — the
        // empty set means "skip the filter."
        self
    }

    /// Builder: attach an exemption certificate.
    #[must_use]
    pub fn with_exemption(mut self, cert: ExemptionCertificate) -> Self {
        self.exemption_certs.push(cert);
        self
    }
}

/// Per-line tax breakdown — every authority that taxed this line,
/// with its rate and amount.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineTaxBreakdown {
    /// Same identifier the caller supplied on [`TaxableLine`].
    pub line_id: String,
    /// Net amount the rate was applied against.
    pub taxable_amount: Money,
    /// Total tax collected from this line.
    pub tax_amount: Money,
    /// Effective compounded rate — sum of all applicable layers.
    pub effective_rate: Decimal,
    /// Per-jurisdiction breakdown, in iteration order (broadest first).
    pub jurisdiction_layers: Vec<JurisdictionTax>,
    /// If exemption fired, why.
    pub exemption_reason: Option<String>,
}

/// Tax collected for a specific authority.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JurisdictionTax {
    /// Authority that gets the money.
    pub jurisdiction: Jurisdiction,
    /// Rate applied at this layer.
    pub rate: Decimal,
    /// Money this authority collected.
    pub amount: Money,
    /// Tax kind for this layer.
    pub kind: crate::rate_table::RateKind,
}

/// Result of a `calculate` call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxResult {
    /// Sum of every line's `tax_amount`. Currency must match all line
    /// currencies (the calculator returns an error if not).
    pub total_tax: Money,
    /// Per-line detail, keyed by `line_id` for deterministic lookup.
    pub per_line: BTreeMap<String, LineTaxBreakdown>,
    /// Flat list of every `(jurisdiction, amount)` charged, summed
    /// across all lines. Used by the remittance / filing layer.
    pub jurisdictions_charged: Vec<JurisdictionTax>,
    /// Backend that produced this result. e.g. `"native"`, `"avalara"`,
    /// `"stripe_tax"`, `"vertex"`.
    pub calculator: String,
    /// When the calculation ran.
    pub calculated_at: DateTime<Utc>,
}

/// The pluggable tax-calculation interface.
///
/// Async because every commercial backend is an HTTPS RPC; the native
/// calculator is async-by-trait too so callers can swap implementations
/// without touching signatures.
#[async_trait]
pub trait TaxCalculator: Send + Sync {
    /// Compute tax for a batch of lines.
    ///
    /// # Errors
    /// - [`crate::error::Error::NoRate`] (native) when no rate exists.
    /// - [`crate::error::Error::Transport`] / `Vendor` (adapter) for
    ///   network and backend errors.
    /// - [`crate::error::Error::Money`] on currency mismatch across
    ///   the line set.
    async fn calculate(&self, lines: &[TaxableLine], ctx: &TaxContext) -> Result<TaxResult>;

    /// Stable identifier for telemetry and audit (`"native"`,
    /// `"avalara"`, etc.). Shows up in `TaxResult.calculator`.
    fn name(&self) -> &'static str;
}
