//! Rate table — the data backing the [`NativeCalculator`].
//!
//! A [`RateTable`] is `BTreeMap<(Jurisdiction, ProductTaxCategory), TaxRate>`.
//! Lookup is `O(log N)` and iteration is deterministic — important for
//! reproducible audit trails.
//!
//! ## On-disk format
//!
//! The bundled snapshot ships as `data/rate_table_v1.cbor` — a CBOR
//! encoding of the [`RateTableSnapshot`] type below. CBOR is the same
//! format `op-screening` uses for its sanctions index; it gives us a
//! schema-evolution story (decode unknown fields by ignoring them)
//! without paying JSON's whitespace tax on disk.
//!
//! The bundled file is a representative starter set:
//! - ~5,000 US ZIP+4-style locality combinations across every state
//!   that levies a sales tax (we cover state base rates plus a curated
//!   sample of county / city / district overlays).
//! - All EU-27 standard VAT rates (Hungary 27% down to Luxembourg 17%).
//! - UK 20%, AU 10% GST, CA 5%/13%/15% GST/HST, IN 18% GST (rounded
//!   to the most common slab — production deployments should consume
//!   per-product HSN codes), Singapore 9%, New Zealand 15%.
//! - LATAM sample: BR (17% ICMS), MX (16% IVA), AR (21% IVA),
//!   CL (19% IVA), CO (19% IVA).
//!
//! ## Production discipline
//!
//! Operators must NOT rely on this snapshot for material revenue.
//! Real US sales-tax data is ~14,000 jurisdictions changing weekly
//! (Avalara, Vertex, and TaxJar all push daily); EU VAT changes
//! several times a year. The bundled table is for testing,
//! development, and small / single-jurisdiction deployments. Operators
//! shipping at scale should subscribe to a vendor feed and call
//! [`RateTable::load_cbor`] over their own daily snapshot.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::category::ProductTaxCategory;
use crate::error::{Error, Result};
use crate::jurisdiction::Jurisdiction;

/// Whether a published rate is applied to the gross (tax-inclusive)
/// or net (tax-exclusive) line amount.
///
/// US sales tax is universally **exclusive**: the shelf price is net,
/// the receipt adds tax on top.
///
/// EU VAT consumer prices are conventionally **inclusive**: the
/// €100 on the price tag already contains €16.67 VAT at the 20% rate.
/// The merchant has to back out the tax to compute net.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaxBase {
    /// Rate is applied as if the published amount already includes tax.
    /// `tax = amount * rate / (1 + rate)`.
    Inclusive,
    /// Rate is applied to the published amount as if tax-free.
    /// `tax = amount * rate`.
    Exclusive,
}

/// What kind of tax the rate represents. Used by `NativeCalculator`
/// to decide compounding (`Vat` replaces; `Sales` / `Use` / `Excise`
/// add layers).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum RateKind {
    /// US-style sales tax (additive layering across state / county /
    /// city / special district).
    Sales,
    /// Use tax — equivalent to sales tax, owed by the buyer when the
    /// seller didn't collect. Treated identically by the calculator.
    Use,
    /// Value-Added Tax — replace-style (one country-level rate; the
    /// lower hierarchy layers are ignored).
    Vat,
    /// Goods and Services Tax (CA/IN/AU/NZ/SG). For our purposes this
    /// behaves identically to VAT — country-level, replace-style.
    Gst,
    /// Excise tax — additive on top of sales / VAT (alcohol, fuel,
    /// tobacco).
    Excise,
    /// Import duty / customs. Additive at country level on cross-
    /// border `ship_from != ship_to.country` lines.
    ImportDuty,
}

/// A single rate entry: a decimal rate plus its tax kind and the
/// inclusive/exclusive base it applies to.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaxRate {
    /// The rate, expressed as a decimal (`0.0825` = 8.25%).
    pub rate: Decimal,
    /// What kind of tax this is.
    pub kind: RateKind,
    /// Whether the rate applies to gross (inclusive) or net (exclusive).
    pub base: TaxBase,
}

impl TaxRate {
    /// Convenience constructor for an exclusive US-style sales-tax rate.
    #[must_use]
    pub const fn sales(rate: Decimal) -> Self {
        Self {
            rate,
            kind: RateKind::Sales,
            base: TaxBase::Exclusive,
        }
    }

    /// Convenience constructor for an inclusive EU-style VAT rate.
    #[must_use]
    pub const fn vat(rate: Decimal) -> Self {
        Self {
            rate,
            kind: RateKind::Vat,
            base: TaxBase::Inclusive,
        }
    }

    /// Convenience constructor for an inclusive GST rate.
    #[must_use]
    pub const fn gst(rate: Decimal) -> Self {
        Self {
            rate,
            kind: RateKind::Gst,
            base: TaxBase::Inclusive,
        }
    }

    /// Convenience constructor for an additive excise rate (per dollar).
    #[must_use]
    pub const fn excise(rate: Decimal) -> Self {
        Self {
            rate,
            kind: RateKind::Excise,
            base: TaxBase::Exclusive,
        }
    }
}

/// The full rate table, in-memory.
///
/// Construction is via [`Self::builder`] for tests + small deployments;
/// production deployments load via [`Self::load_cbor`] or the bundled
/// [`Self::bundled`] snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateTable {
    /// Sparse map from `(jurisdiction, category) → rate`.
    ///
    /// Category-less rates are stored under
    /// `ProductTaxCategory::TangibleGoods` as the default; the
    /// calculator falls back to the `TangibleGoods` entry if no
    /// exact match exists for the requested category.
    pub entries: BTreeMap<(Jurisdiction, ProductTaxCategory), TaxRate>,
    /// Free-form provenance string — e.g. `"avalara-2026-05-01"`,
    /// `"native-bundled-v1"`. Surfaced in `TaxResult.calculator` and
    /// in compliance logs.
    pub source: String,
    /// When this snapshot was minted. Operators check this to detect
    /// stale tables before high-value calculations.
    pub as_of: DateTime<Utc>,
}

impl RateTable {
    /// Empty table — useful for tests that build their own rates.
    #[must_use]
    pub fn empty(source: impl Into<String>) -> Self {
        Self {
            entries: BTreeMap::new(),
            source: source.into(),
            as_of: chrono::Utc::now(),
        }
    }

    /// Builder-style mutator. Returns `Self` for chaining.
    #[must_use]
    pub fn with(
        mut self,
        jurisdiction: Jurisdiction,
        category: ProductTaxCategory,
        rate: TaxRate,
    ) -> Self {
        self.entries.insert((jurisdiction, category), rate);
        self
    }

    /// Bundled starter table. Returns an empty table at runtime —
    /// the on-disk CBOR snapshot must be loaded via
    /// [`Self::load_cbor`] for non-test deployments.
    ///
    /// We do NOT eagerly embed the snapshot via `include_bytes!`
    /// because the snapshot file is a deployment artifact (refreshed
    /// daily by operators) rather than a source-controlled blob. The
    /// repo carries a sample snapshot under `data/rate_table_v1.cbor`
    /// for tests + examples to load explicitly.
    #[must_use]
    pub fn bundled() -> Self {
        Self::starter_set()
    }

    /// Hand-coded representative starter set used by tests and as a
    /// fallback when no on-disk snapshot is available.
    ///
    /// Coverage is deliberately small (one state + one EU country +
    /// UK + AU + a CA province + IN + a LATAM sample). The on-disk
    /// snapshot extends this with the full ~5,000 starter entries
    /// described in the module docs.
    fn starter_set() -> Self {
        let mut t = Self::empty("native-bundled-v1");

        // US — Washington State sales-tax stack.
        // WA state base rate: 6.5% (2026).
        t = t.with(
            Jurisdiction::region("US", "WA"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(Decimal::new(65, 3)), // 0.065
        );
        // King County local: 0%.
        // City of Seattle: 3.85% (combined city/county portion 2026).
        t = t.with(
            Jurisdiction::locality("US", "WA", "Seattle"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(Decimal::new(385, 4)), // 0.0385
        );
        // King County Regional Transit Authority: 0.9%.
        t = t.with(
            Jurisdiction::full("US", "WA", "Seattle", "KingTransit"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(Decimal::new(9, 3)), // 0.009
        );

        // US — California base rate (state portion 7.25% incl. mandatory
        // local component) plus LA County district add-on.
        t = t.with(
            Jurisdiction::region("US", "CA"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(Decimal::new(725, 4)),
        );

        // US — New York: clothing under $110 exempt is handled by the
        // calculator's category-override logic, but the base rate here:
        t = t.with(
            Jurisdiction::region("US", "NY"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(Decimal::new(4, 2)), // 4%
        );
        // NYC city: 4.5%.
        t = t.with(
            Jurisdiction::locality("US", "NY", "NewYorkCity"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::sales(Decimal::new(45, 3)),
        );

        // EU VAT — standard rates as of 2026 (source: European
        // Commission "VAT rates applied in the Member States").
        for (cc, rate_bp) in [
            ("AT", 2000), // Austria 20%
            ("BE", 2100),
            ("BG", 2000),
            ("HR", 2500),
            ("CY", 1900),
            ("CZ", 2100),
            ("DK", 2500),
            ("EE", 2200),
            ("FI", 2550), // Finland 25.5% (rate hike effective 2024-09)
            ("FR", 2000),
            ("DE", 1900),
            ("GR", 2400),
            ("HU", 2700), // Hungary — highest in EU
            ("IE", 2300),
            ("IT", 2200),
            ("LV", 2100),
            ("LT", 2100),
            ("LU", 1700), // Luxembourg — lowest in EU
            ("MT", 1800),
            ("NL", 2100),
            ("PL", 2300),
            ("PT", 2300),
            ("RO", 1900),
            ("SK", 2300), // Slovakia 23% (rate hike effective 2025-01)
            ("SI", 2200),
            ("ES", 2100),
            ("SE", 2500),
        ] {
            t = t.with(
                Jurisdiction::country(cc),
                ProductTaxCategory::TangibleGoods,
                TaxRate::vat(Decimal::new(rate_bp, 4)),
            );
        }

        // UK VAT 20%.
        t = t.with(
            Jurisdiction::country("GB"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::vat(Decimal::new(2000, 4)),
        );

        // Australia 10% GST.
        t = t.with(
            Jurisdiction::country("AU"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::gst(Decimal::new(1000, 4)),
        );

        // Canada — federal GST 5% (base). Provincial HST overlays
        // (ON 13%, NS/NB/NL/PE 15%) are modelled at the region level.
        t = t.with(
            Jurisdiction::country("CA"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::gst(Decimal::new(500, 4)),
        );
        t = t.with(
            Jurisdiction::region("CA", "ON"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::gst(Decimal::new(1300, 4)), // ON HST 13% replaces fed GST
        );

        // India 18% GST (the most-common slab; production must use
        // per-HSN-code lookups).
        t = t.with(
            Jurisdiction::country("IN"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::gst(Decimal::new(1800, 4)),
        );

        // Singapore 9% GST (post-2024 hike).
        t = t.with(
            Jurisdiction::country("SG"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::gst(Decimal::new(900, 4)),
        );

        // New Zealand 15% GST.
        t = t.with(
            Jurisdiction::country("NZ"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::gst(Decimal::new(1500, 4)),
        );

        // LATAM sample.
        for (cc, rate_bp) in [
            ("BR", 1700), // ICMS — varies by state; 17% is common
            ("MX", 1600), // IVA
            ("AR", 2100),
            ("CL", 1900),
            ("CO", 1900),
        ] {
            t = t.with(
                Jurisdiction::country(cc),
                ProductTaxCategory::TangibleGoods,
                TaxRate::vat(Decimal::new(rate_bp, 4)),
            );
        }

        // Category override example: NY clothing is exempt under $110.
        // We model the rate-table override here at 0%; the calculator's
        // per-line logic enforces the $110 cap when present (see
        // NativeCalculator::apply_category_overrides).
        t = t.with(
            Jurisdiction::region("US", "NY"),
            ProductTaxCategory::Clothing,
            TaxRate::sales(Decimal::ZERO),
        );

        // Excise — federal motor-fuel tax (per-dollar approximation;
        // real excise is per-gallon, but we model the ad-valorem
        // equivalent for the calculator's sake).
        t = t.with(
            Jurisdiction::country("US"),
            ProductTaxCategory::MotorFuel,
            TaxRate::excise(Decimal::new(184, 4)), // 1.84%-equiv of federal $0.184/gal
        );

        t
    }

    /// Look up the rate that applies to `(jurisdiction, category)`.
    ///
    /// Lookup falls back from the exact category to
    /// `TangibleGoods` if no override exists. Returns `None` if even
    /// the `TangibleGoods` entry is missing for that jurisdiction —
    /// meaning the calculator has no rate to apply, and should signal
    /// `Error::NoRate`.
    #[must_use]
    pub fn lookup(
        &self,
        jurisdiction: &Jurisdiction,
        category: &ProductTaxCategory,
    ) -> Option<&TaxRate> {
        if let Some(r) = self
            .entries
            .get(&(jurisdiction.clone(), category.clone()))
        {
            return Some(r);
        }
        self.entries
            .get(&(jurisdiction.clone(), ProductTaxCategory::TangibleGoods))
    }

    /// Load a CBOR-encoded snapshot from disk.
    ///
    /// # Errors
    /// [`Error::Snapshot`] if the file cannot be opened or decoded.
    pub fn load_cbor(path: impl AsRef<Path>) -> Result<Self> {
        let f = std::fs::File::open(path.as_ref()).map_err(|e| Error::Snapshot(e.to_string()))?;
        let snap: RateTable =
            ciborium::de::from_reader(f).map_err(|e| Error::Snapshot(e.to_string()))?;
        Ok(snap)
    }

    /// Load a CBOR-encoded snapshot from any reader. Used by tests
    /// that round-trip an in-memory buffer.
    ///
    /// # Errors
    /// [`Error::Snapshot`] if the bytes cannot be decoded.
    pub fn from_reader(r: impl Read) -> Result<Self> {
        ciborium::de::from_reader(r).map_err(|e| Error::Snapshot(e.to_string()))
    }

    /// Encode this table as CBOR into the given writer.
    ///
    /// # Errors
    /// [`Error::Snapshot`] if encoding or I/O fails.
    pub fn write_cbor(&self, w: impl std::io::Write) -> Result<()> {
        ciborium::ser::into_writer(self, w).map_err(|e| Error::Snapshot(e.to_string()))?;
        Ok(())
    }

    /// Total number of rate entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table contains any entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_table_has_eu_27() {
        let t = RateTable::bundled();
        // Must contain a standard VAT rate for every EU-27 country.
        for cc in [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        ] {
            let j = Jurisdiction::country(cc);
            let r = t.lookup(&j, &ProductTaxCategory::TangibleGoods);
            assert!(r.is_some(), "missing VAT for {cc}");
            assert_eq!(r.unwrap().kind, RateKind::Vat);
        }
    }

    #[test]
    fn bundled_table_compound_us_layers() {
        let t = RateTable::bundled();
        // WA state + Seattle city + KingTransit = three additive layers.
        assert!(
            t.lookup(
                &Jurisdiction::region("US", "WA"),
                &ProductTaxCategory::TangibleGoods
            )
            .is_some()
        );
        assert!(
            t.lookup(
                &Jurisdiction::locality("US", "WA", "Seattle"),
                &ProductTaxCategory::TangibleGoods
            )
            .is_some()
        );
        assert!(
            t.lookup(
                &Jurisdiction::full("US", "WA", "Seattle", "KingTransit"),
                &ProductTaxCategory::TangibleGoods
            )
            .is_some()
        );
    }

    #[test]
    fn cbor_roundtrip_preserves_table() {
        let t = RateTable::bundled();
        let mut buf = Vec::new();
        t.write_cbor(&mut buf).unwrap();
        let decoded = RateTable::from_reader(&buf[..]).unwrap();
        assert_eq!(decoded.entries.len(), t.entries.len());
        assert_eq!(decoded.source, t.source);
    }

    #[test]
    fn category_falls_back_to_tangible_goods() {
        let t = RateTable::bundled();
        // Saas isn't in the bundled table — fallback should give us
        // the TangibleGoods VAT entry for DE.
        let r = t
            .lookup(&Jurisdiction::country("DE"), &ProductTaxCategory::Saas)
            .unwrap();
        assert_eq!(r.kind, RateKind::Vat);
    }

    #[test]
    fn category_override_takes_precedence() {
        let t = RateTable::bundled();
        // NY clothing entry overrides the base NY rate.
        let r = t
            .lookup(
                &Jurisdiction::region("US", "NY"),
                &ProductTaxCategory::Clothing,
            )
            .unwrap();
        assert_eq!(r.rate, Decimal::ZERO);
    }
}
