//! Product tax categories.
//!
//! Every line item is tagged with a category. Categories drive
//! per-jurisdiction rate overrides — the same dollar of revenue is
//! taxed at different rates depending on whether it is clothing,
//! food, alcohol, or SaaS.
//!
//! Examples of real overrides modelled here:
//! - New York exempts clothing under $110 (NY State + many localities).
//! - Pennsylvania exempts most clothing entirely; charges 6% on
//!   formal-wear and accessories.
//! - Texas, Florida, and several other states tax `Software` but not
//!   `Saas`; New York, Massachusetts, Washington, and an expanding
//!   list of states tax `Saas` as a taxable digital service.
//! - EU VAT applies reduced rates (typically 5%–10%) to `Food`,
//!   `Healthcare`, books, and printed media.
//! - Most jurisdictions impose excise on `Alcohol`, `Tobacco`, and
//!   `MotorFuel` on top of (or sometimes instead of) sales tax.
//!
//! The enum is intentionally not exhaustive in the SemVer sense —
//! callers must include an `_` arm. New categories are minor releases.

use serde::{Deserialize, Serialize};

/// Tax-relevant classification of a sold line item.
///
/// The `Other(String)` escape hatch is for operators with industry-
/// specific categories (e.g. cannabis dispensaries) whose categories
/// don't yet appear in the standard list. The native calculator
/// treats unknown categories as `TangibleGoods` for rate lookup; the
/// commercial backends pass the string straight through.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ProductTaxCategory {
    /// Physical goods shipped to the customer. The default.
    TangibleGoods,

    /// Digital goods delivered electronically: ebooks, MP3 downloads,
    /// game DLC. Taxable in most US states post-Wayfair; subject to
    /// VAT in the EU under the place-of-supply rules for digital
    /// services.
    DigitalGoods,

    /// Boxed / downloaded packaged software with a license key.
    /// Historically distinguished from `Saas` in sales-tax law.
    Software,

    /// Software-as-a-Service. Taxable in NY, MA, WA, TX (partially),
    /// PA, and a growing list of US states. Most EU member states
    /// treat it as a taxable digital service.
    Saas,

    /// Professional services: consulting, legal, accounting. Generally
    /// not taxable in US sales-tax states; taxable as services under
    /// EU VAT.
    ProfessionalService,

    /// Wireline / wireless / VoIP communications. Subject to special
    /// federal and state-level taxes (USF, state E911, etc.) in the US
    /// in addition to base sales tax.
    Telecommunications,

    /// Medical devices, prescription drugs, durable medical equipment.
    /// Exempt in many jurisdictions (prescription drugs especially).
    Healthcare,

    /// Grocery food. Often exempt, partially exempt, or taxed at a
    /// reduced rate (e.g. PA exempts groceries entirely; many EU
    /// states apply a 5–10% reduced VAT rate).
    Food,

    /// Apparel. New York, New Jersey, Pennsylvania, Massachusetts,
    /// and Vermont have notable clothing rules (full exemption,
    /// dollar caps, or carve-outs for formal wear).
    Clothing,

    /// Beer, wine, spirits, RTDs. Carries excise on top of sales tax.
    Alcohol,

    /// Cigarettes, cigars, smokeless. Federal + state excise; some
    /// localities add their own (NYC, Chicago).
    Tobacco,

    /// Gasoline, diesel, propane. Federal excise + state excise +
    /// occasional sales tax.
    MotorFuel,

    /// Hotel / vacation-rental / short-term lodging. Often carries
    /// a separate transient occupancy tax (TOT) in addition to base
    /// sales tax.
    Lodging,

    /// Tickets to concerts, sporting events, theaters, museums.
    /// Taxability varies widely by jurisdiction.
    AdmissionsAndEvents,

    /// Escape hatch for industry-specific categories not yet enumerated.
    /// String is passed straight through to commercial backends; the
    /// native calculator falls back to `TangibleGoods` rates.
    Other(String),
}

impl ProductTaxCategory {
    /// String tag suitable for use as a vendor API parameter and as
    /// a key in rate-table CBOR maps.
    ///
    /// Stable across versions (changing one is a SemVer-major change).
    #[must_use]
    pub fn tag(&self) -> &str {
        match self {
            Self::TangibleGoods => "tangible_goods",
            Self::DigitalGoods => "digital_goods",
            Self::Software => "software",
            Self::Saas => "saas",
            Self::ProfessionalService => "professional_service",
            Self::Telecommunications => "telecommunications",
            Self::Healthcare => "healthcare",
            Self::Food => "food",
            Self::Clothing => "clothing",
            Self::Alcohol => "alcohol",
            Self::Tobacco => "tobacco",
            Self::MotorFuel => "motor_fuel",
            Self::Lodging => "lodging",
            Self::AdmissionsAndEvents => "admissions_events",
            Self::Other(s) => s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_is_stable() {
        // Sentinel: changing any of these strings is a SemVer-major change
        // because they appear in CBOR snapshots and vendor wire payloads.
        assert_eq!(ProductTaxCategory::TangibleGoods.tag(), "tangible_goods");
        assert_eq!(ProductTaxCategory::Saas.tag(), "saas");
        assert_eq!(ProductTaxCategory::MotorFuel.tag(), "motor_fuel");
    }

    #[test]
    fn other_passes_string_through() {
        let c = ProductTaxCategory::Other("cannabis_flower".to_owned());
        assert_eq!(c.tag(), "cannabis_flower");
    }
}
