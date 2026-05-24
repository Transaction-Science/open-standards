//! # `op-tax` — Tax calculation for `OpenPay`
//!
//! A pluggable tax-engine surface for the OpenPay payment stack.
//! Operators wire any of the supported commercial backends (Avalara,
//! Stripe Tax, CCH SureTax, Vertex, TaxJar) behind a single
//! [`TaxCalculator`] trait, or run a small deployment off the
//! bundled [`NativeCalculator`] with no external dependency.
//!
//! ## Scope
//!
//! - **US sales tax**: 50-state + DC, ~14,000 jurisdiction hierarchy
//!   (state → county → city → special district), additive
//!   compounding, category-specific overrides.
//! - **VAT**: EU-27 standard rates, UK 20%, plus inclusive-base math.
//! - **GST**: CA / IN / SG / NZ / AU, replace-style compounding.
//! - **Excise**: alcohol / tobacco / fuel additive layers on top of
//!   sales or VAT.
//! - **Reverse charge**: EU B2B cross-border zero-rating per the EU
//!   VAT Directive Article 196.
//! - **Exemption certificates**: resale + charity + government
//!   credentials with date / jurisdiction / category scoping.
//! - **Economic nexus**: per-US-state Wayfair threshold tracking with
//!   `Triggered` / `Approaching` events.
//! - **Withholding**: 1099-K thresholds, 1042-S NRA + treaty rates,
//!   EU VAT-MOSS / OSS quarterly summaries — surfaced as events for
//!   `op-connect` to file.
//!
//! ## Out of scope (intentionally)
//!
//! - Live VAT-number validation against EU VIES. Operators wire that
//!   independently.
//! - PDF / IRS-XML form generation. Lives in `op-connect`.
//! - Rolling-window aggregation for nexus. The monitor exposes
//!   lifetime totals; downstream is responsible for the 12-month
//!   roll-off if a state mandates that flavor.
//! - Per-product HSN / CN classification. Categories are coarse on
//!   purpose; product-level granularity is the vendor's job.
//!
//! ## Decimal discipline
//!
//! All rates and intermediate tax amounts are [`rust_decimal::Decimal`].
//! The only `f64` appears on the wire to backends that publish JSON
//! schemas using floats (Avalara, Stripe, TaxJar) — those values are
//! converted to/from `Decimal` at the boundary, and the result we
//! return to callers is always exact `i64` minor units inside
//! [`op_core::Money`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
// The lints below are allowed deliberately. Each is justified:
// - module_name_repetitions: `TaxCalculator` inside `calculator`, etc., is the
//   conventional public-API shape for this crate; matches op-fx / op-core idiom.
// - missing_errors_doc / missing_panics_doc: every error variant is documented
//   on the `Error` enum itself; per-function docs would duplicate that.
// - similar_names: tax math involves rate/range/region/result variables that
//   are deliberately short; pedantic renaming would hurt readability.
// - must_use_candidate: these are getter-style methods on data types; callers
//   generally consume the return value.
// - too_many_lines: the `calc_line` function in native.rs is one cohesive
//   piece of compounding logic; splitting it for line-count alone hurts the
//   readability of the algorithm.
// - match_same_arms: many vendor / treaty mapping tables happen to share return
//   values across distinct semantic categories (UK royalties = 0 = UK interest
//   = 0, but they're not the same concept and shouldn't be merged).
// - doc_markdown: prose includes proper nouns (Wayfair, Vertex, IVA, etc.) that
//   are not Rust items.
// - manual_let_else: kept the explicit `match` form in compute_1042s for
//   parallel structure with surrounding error-mapping code.
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::manual_let_else)]

pub mod avalara;
pub mod calculator;
pub mod category;
pub mod cch_suretax;
pub mod error;
pub mod exemption;
pub mod jurisdiction;
pub mod native;
pub mod nexus;
pub mod rate_table;
pub mod reverse_charge;
pub mod stripe_tax;
pub mod taxjar;
pub mod vertex;
pub mod withholding;

pub use avalara::AvalaraAdapter;
pub use calculator::{
    CustomerType, JurisdictionTax, LineTaxBreakdown, TaxCalculator, TaxContext, TaxResult,
    TaxableLine,
};
pub use category::ProductTaxCategory;
pub use cch_suretax::CchSureTaxAdapter;
pub use error::{Error, Result};
pub use exemption::ExemptionCertificate;
pub use jurisdiction::{CountryCode, DistrictCode, Jurisdiction, LocalityCode, RegionCode};
pub use native::NativeCalculator;
pub use nexus::{NexusEvent, NexusMonitor, NexusThreshold, TransactionRecord};
pub use rate_table::{RateKind, RateTable, TaxBase, TaxRate};
pub use stripe_tax::StripeTaxAdapter;
pub use taxjar::TaxJarAdapter;
pub use vertex::VertexAdapter;
pub use withholding::{
    PayeeYtd, PayoutEvent, WithholdingEvent, WithholdingHook, classify, compute_1042s,
    form_1099k_threshold_minor, statutory_nra_rate, treaty_rate, vat_oss_threshold_eur_minor,
};
