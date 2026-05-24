//! EU B2B reverse-charge predicate.
//!
//! Under EU Council Directive 2006/112/EC (Articles 44 & 196), a B2B
//! cross-border supply of services or goods between two EU member
//! states shifts the VAT-collection obligation from the seller to
//! the buyer. The seller invoices at zero rate and the buyer self-
//! accounts on its own VAT return.
//!
//! Conditions (all must hold):
//! 1. Seller and buyer are in different EU member states.
//! 2. The buyer is a business with a valid VAT identification number
//!    (we treat "non-empty `tax_id`" as the seller's evidence of #2 —
//!    operators that require live VIES validation should wrap this).
//! 3. The supply is one of the categories the directive covers
//!    (services, most cross-border B2B goods). We default to "yes"
//!    for every category except `MotorFuel` and other excise items,
//!    where domestic excise still applies on cross-border movement.
//!
//! ## What this module does not do
//!
//! - **VIES VAT-number validation.** Operators wire that themselves
//!   against the European Commission's VIES SOAP service before
//!   accepting a buyer's tax-ID for reverse-charge purposes.
//! - **Domestic reverse-charge regimes** (UK construction, certain
//!   gold supplies). Out of scope for v1; add per-jurisdiction when
//!   needed.

use crate::calculator::{CustomerType, TaxContext, TaxableLine};
use crate::category::ProductTaxCategory;

/// Returns true if this line qualifies for EU B2B reverse charge —
/// the seller zero-rates and the buyer self-accounts.
#[must_use]
pub fn applies(line: &TaxableLine, ctx: &TaxContext) -> bool {
    let CustomerType::Business { tax_id } = &ctx.customer_type else {
        return false;
    };
    if tax_id.trim().is_empty() {
        return false;
    }
    let to_eu = line.ship_to.is_eu_member();
    let Some(from) = &line.ship_from else {
        return false;
    };
    let from_eu = from.is_eu_member();
    if !(to_eu && from_eu) {
        return false;
    }
    if from.country == line.ship_to.country {
        return false;
    }
    // Excise-bearing categories are NOT reverse-charge eligible —
    // domestic excise duty still applies at the destination.
    !matches!(
        line.category,
        ProductTaxCategory::MotorFuel | ProductTaxCategory::Alcohol | ProductTaxCategory::Tobacco
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jurisdiction::Jurisdiction;
    use chrono::NaiveDate;
    use op_core::{Currency, Money};
    use std::collections::HashSet;

    fn line(from: Option<Jurisdiction>, to: Jurisdiction, cat: ProductTaxCategory) -> TaxableLine {
        TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::EUR),
            category: cat,
            ship_from: from,
            ship_to: to,
        }
    }

    fn business_ctx() -> TaxContext {
        TaxContext {
            transaction_date: NaiveDate::parse_from_str("2026-06-15", "%Y-%m-%d").unwrap(),
            customer_type: CustomerType::Business {
                tax_id: "FR12345678901".into(),
            },
            exemption_certs: vec![],
            nexus_jurisdictions: HashSet::new(),
        }
    }

    #[test]
    fn cross_border_eu_b2b_triggers_reverse_charge() {
        let l = line(
            Some(Jurisdiction::country("DE")),
            Jurisdiction::country("FR"),
            ProductTaxCategory::Saas,
        );
        assert!(applies(&l, &business_ctx()));
    }

    #[test]
    fn domestic_eu_b2b_does_not_trigger() {
        let l = line(
            Some(Jurisdiction::country("DE")),
            Jurisdiction::country("DE"),
            ProductTaxCategory::Saas,
        );
        assert!(!applies(&l, &business_ctx()));
    }

    #[test]
    fn cross_border_consumer_does_not_trigger() {
        let l = line(
            Some(Jurisdiction::country("DE")),
            Jurisdiction::country("FR"),
            ProductTaxCategory::Saas,
        );
        let mut ctx = business_ctx();
        ctx.customer_type = CustomerType::Consumer;
        assert!(!applies(&l, &ctx));
    }

    #[test]
    fn cross_border_to_non_eu_does_not_trigger() {
        let l = line(
            Some(Jurisdiction::country("DE")),
            Jurisdiction::country("CH"),
            ProductTaxCategory::Saas,
        );
        assert!(!applies(&l, &business_ctx()));
    }

    #[test]
    fn excise_category_does_not_trigger_reverse_charge() {
        let l = line(
            Some(Jurisdiction::country("DE")),
            Jurisdiction::country("FR"),
            ProductTaxCategory::Alcohol,
        );
        assert!(!applies(&l, &business_ctx()));
    }
}
