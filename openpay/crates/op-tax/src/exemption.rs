//! Exemption certificate handling.
//!
//! A certificate is a bearer document — typically a state-issued
//! resale certificate (Streamlined Sales and Use Tax Agreement form,
//! or one of the dozens of state-specific equivalents), an EU VAT
//! exemption (Article 138 / Article 151 forms), or an in-house
//! charity / government / diplomatic credential.
//!
//! We store the certificate as opaque bytes (the original PDF or
//! XML, suitable for audit retention) plus the structured metadata
//! needed to evaluate applicability.
//!
//! Storage: certificates are *not* persisted by this crate. Operators
//! retain them in `op-vault` or their preferred secure store and
//! present them on the [`TaxContext`] for each calculate call.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::calculator::TaxableLine;
use crate::category::ProductTaxCategory;
use crate::jurisdiction::Jurisdiction;

/// A buyer's exemption credential.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExemptionCertificate {
    /// Operator-assigned identifier (used for retrieval + audit
    /// cross-reference).
    pub id: String,
    /// Legal name of the holder (the buyer's registered entity name).
    pub holder: String,
    /// Jurisdictions in which the certificate is valid. An empty
    /// vector means "valid anywhere" — useful for federal-level
    /// charity exemptions; operators using this should be explicit.
    pub jurisdictions: Vec<Jurisdiction>,
    /// Categories the certificate exempts. An empty vector means
    /// "all categories".
    pub categories: Vec<ProductTaxCategory>,
    /// First date the certificate is valid (inclusive).
    pub valid_from: NaiveDate,
    /// Last date the certificate is valid (inclusive). `None` for
    /// perpetual certificates (rare — most states require renewal).
    pub valid_until: Option<NaiveDate>,
    /// Original certificate bytes — PDF, XML, JPEG. Retained for
    /// audit; never inspected by this crate.
    pub certificate_data: Vec<u8>,
}

/// Predicate: does `cert` exempt `line` under `ctx`?
///
/// Truth table:
/// 1. The transaction date must fall within `valid_from..=valid_until`.
/// 2. The certificate's `jurisdictions` must contain the line's
///    `ship_to` *or* an ancestor of it (a state-level certificate
///    covers every city in that state). Empty list = matches anywhere.
/// 3. The certificate's `categories` must contain the line's category
///    *or* be empty.
#[must_use]
pub fn applies(
    cert: &ExemptionCertificate,
    line: &TaxableLine,
    ctx: &crate::calculator::TaxContext,
) -> bool {
    // 1. Date window.
    if ctx.transaction_date < cert.valid_from {
        return false;
    }
    if let Some(end) = cert.valid_until
        && ctx.transaction_date > end
    {
        return false;
    }

    // 2. Jurisdiction match.
    if !cert.jurisdictions.is_empty() {
        let ancestors = line.ship_to.ancestors();
        let jurisdiction_match = cert.jurisdictions.iter().any(|j| ancestors.contains(j));
        if !jurisdiction_match {
            return false;
        }
    }

    // 3. Category match.
    if !cert.categories.is_empty() && !cert.categories.contains(&line.category) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculator::{CustomerType, TaxContext};
    use op_core::{Currency, Money};
    use std::collections::HashSet;

    fn line(ship_to: Jurisdiction, cat: ProductTaxCategory) -> TaxableLine {
        TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::USD),
            category: cat,
            ship_from: None,
            ship_to,
        }
    }

    fn ctx(date_iso: &str) -> TaxContext {
        TaxContext {
            transaction_date: NaiveDate::parse_from_str(date_iso, "%Y-%m-%d").unwrap(),
            customer_type: CustomerType::Business {
                tax_id: "12-3456789".into(),
            },
            exemption_certs: vec![],
            nexus_jurisdictions: HashSet::new(),
        }
    }

    #[test]
    fn resale_cert_applies_to_in_state_tangible_goods() {
        let cert = ExemptionCertificate {
            id: "RESALE-WA-1".into(),
            holder: "Acme Resellers".into(),
            jurisdictions: vec![Jurisdiction::region("US", "WA")],
            categories: vec![ProductTaxCategory::TangibleGoods],
            valid_from: NaiveDate::parse_from_str("2026-01-01", "%Y-%m-%d").unwrap(),
            valid_until: Some(NaiveDate::parse_from_str("2026-12-31", "%Y-%m-%d").unwrap()),
            certificate_data: b"<pdf bytes>".to_vec(),
        };
        let l = line(
            Jurisdiction::locality("US", "WA", "Seattle"),
            ProductTaxCategory::TangibleGoods,
        );
        assert!(applies(&cert, &l, &ctx("2026-06-15")));
    }

    #[test]
    fn cert_expired_does_not_apply() {
        let cert = ExemptionCertificate {
            id: "EXPIRED".into(),
            holder: "X".into(),
            jurisdictions: vec![],
            categories: vec![],
            valid_from: NaiveDate::parse_from_str("2025-01-01", "%Y-%m-%d").unwrap(),
            valid_until: Some(NaiveDate::parse_from_str("2025-12-31", "%Y-%m-%d").unwrap()),
            certificate_data: vec![],
        };
        let l = line(
            Jurisdiction::region("US", "WA"),
            ProductTaxCategory::TangibleGoods,
        );
        assert!(!applies(&cert, &l, &ctx("2026-06-15")));
    }

    #[test]
    fn cert_wrong_jurisdiction_does_not_apply() {
        let cert = ExemptionCertificate {
            id: "CA-CERT".into(),
            holder: "X".into(),
            jurisdictions: vec![Jurisdiction::region("US", "CA")],
            categories: vec![],
            valid_from: NaiveDate::parse_from_str("2026-01-01", "%Y-%m-%d").unwrap(),
            valid_until: None,
            certificate_data: vec![],
        };
        let l = line(
            Jurisdiction::region("US", "WA"),
            ProductTaxCategory::TangibleGoods,
        );
        assert!(!applies(&cert, &l, &ctx("2026-06-15")));
    }

    #[test]
    fn cert_wrong_category_does_not_apply() {
        let cert = ExemptionCertificate {
            id: "SAAS-ONLY".into(),
            holder: "X".into(),
            jurisdictions: vec![],
            categories: vec![ProductTaxCategory::Saas],
            valid_from: NaiveDate::parse_from_str("2026-01-01", "%Y-%m-%d").unwrap(),
            valid_until: None,
            certificate_data: vec![],
        };
        let l = line(
            Jurisdiction::region("US", "WA"),
            ProductTaxCategory::TangibleGoods,
        );
        assert!(!applies(&cert, &l, &ctx("2026-06-15")));
    }

    #[test]
    fn empty_jurisdictions_match_anywhere() {
        let cert = ExemptionCertificate {
            id: "FEDERAL".into(),
            holder: "Charity".into(),
            jurisdictions: vec![],
            categories: vec![],
            valid_from: NaiveDate::parse_from_str("2026-01-01", "%Y-%m-%d").unwrap(),
            valid_until: None,
            certificate_data: vec![],
        };
        let l = line(
            Jurisdiction::locality("US", "NY", "NewYorkCity"),
            ProductTaxCategory::TangibleGoods,
        );
        assert!(applies(&cert, &l, &ctx("2026-06-15")));
    }
}
