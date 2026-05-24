//! US Financial Data Exchange (FDX) v6.x binding.
//!
//! Reference: Financial Data Exchange, *FDX API v6* (current spec
//! mainline as of 2025). FDX is the US industry-standard
//! consumer-permissioned data API — the de-facto replacement for
//! screen-scraping aggregators. It was reinforced by CFPB's Section
//! 1033 Personal Financial Data Rights final rule (October 2024),
//! which obligates financial institutions to expose covered data
//! through standards-aligned interfaces.
//!
//! Unlike UK / EU PSD2, FDX is not statutorily mandated *per se*
//! (the CFPB rule references "qualified industry standards"); FDX
//! v6 is currently the only candidate that meets the rule.

use serde::{Deserialize, Serialize};

/// FDX API version markers. v6 has been the mainline since 2024;
/// v5 deployments persist among smaller institutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FdxVersion {
    /// FDX v5.x — legacy.
    V5,
    /// FDX v6.x — current.
    V6,
}

impl FdxVersion {
    /// URL prefix the spec uses for this version family.
    #[must_use]
    pub const fn url_segment(self) -> &'static str {
        match self {
            Self::V5 => "fdx/v5",
            Self::V6 => "fdx/v6",
        }
    }
}

/// FDX resource family — the major endpoint groups under the version
/// root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FdxResource {
    /// `/accounts` — account list / detail.
    Accounts,
    /// `/transactions` — transaction history.
    Transactions,
    /// `/statements` — statement files (PDFs + structured).
    Statements,
    /// `/customers` — KYC-style customer profile.
    Customers,
    /// `/payments/recurring` — recurring-payment commitments.
    RecurringPayments,
    /// `/tax/forms` — 1099-INT / 1099-DIV / 5498 tax forms.
    TaxForms,
    /// `/investments` — investment-account positions.
    Investments,
    /// `/rewards` — rewards balances and history.
    Rewards,
}

impl FdxResource {
    /// Path segment used under the version root.
    #[must_use]
    pub const fn as_segment(self) -> &'static str {
        match self {
            Self::Accounts => "accounts",
            Self::Transactions => "transactions",
            Self::Statements => "statements",
            Self::Customers => "customers",
            Self::RecurringPayments => "payments/recurring",
            Self::TaxForms => "tax/forms",
            Self::Investments => "investments",
            Self::Rewards => "rewards",
        }
    }
}

/// FDX consent record. FDX consents bind a data recipient + resource
/// scope set + duration. The `durationType` enum on the wire is
/// `ONE_TIME` | `TIME_BOUND` | `INDEFINITE`; we model the same shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdxConsent {
    /// `id` minted by the data provider.
    pub id: String,
    /// Resources the consent grants access to.
    pub resources: Vec<FdxResource>,
    /// Account identifiers the consent covers (FDX consents are
    /// typically per-account).
    pub account_ids: Vec<String>,
    /// `expiresAt` — RFC 3339. `None` for `INDEFINITE` consents
    /// (rare, but supported by the FDX schema since v6.1).
    pub expires_at: Option<time::OffsetDateTime>,
}

/// FDX service handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdxService {
    /// Data provider base URL.
    pub provider_base_url: String,
    /// FDX version this provider implements.
    pub version: FdxVersion,
}

impl FdxService {
    /// Build a resource endpoint URL.
    #[must_use]
    pub fn endpoint(&self, resource: FdxResource) -> String {
        let trimmed_base = self.provider_base_url.trim_end_matches('/');
        format!(
            "{}/{}/{}",
            trimmed_base,
            self.version.url_segment(),
            resource.as_segment()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_segments_match_spec() {
        assert_eq!(FdxVersion::V5.url_segment(), "fdx/v5");
        assert_eq!(FdxVersion::V6.url_segment(), "fdx/v6");
    }

    #[test]
    fn resource_segments_match_spec() {
        assert_eq!(FdxResource::TaxForms.as_segment(), "tax/forms");
        assert_eq!(FdxResource::RecurringPayments.as_segment(), "payments/recurring");
    }

    #[test]
    fn endpoint_builds_for_v6_accounts() {
        let svc = FdxService {
            provider_base_url: "https://api.fi.example".into(),
            version: FdxVersion::V6,
        };
        assert_eq!(
            svc.endpoint(FdxResource::Accounts),
            "https://api.fi.example/fdx/v6/accounts"
        );
    }
}
