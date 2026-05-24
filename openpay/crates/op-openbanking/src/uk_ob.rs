//! UK Open Banking Read/Write API binding (OBIE v3.1.x).
//!
//! Reference: Open Banking Implementation Entity, *Read/Write Data
//! API Specification* v3.1.11. Endpoints rooted at
//! `https://<aspsp>/open-banking/v3.1/{aisp|pisp|cbpii}/`.
//!
//! UK OBIE is the most prescriptive Open Banking regime: it pins
//! FAPI 1.0 Advanced for everything except read-only AISP, mandates
//! the OBIE-issued OBSeal certificates for JWS signing, and runs the
//! OBIE directory as the JWK-registration source of truth.

use serde::{Deserialize, Serialize};

/// UK OBIE version markers. v3.1.10 is the current mainline release;
/// v3.1.11 ships the VRP profile increment. We track the latest two
/// minor versions: ASPSPs negotiate the version in the `x-fapi-...`
/// header set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UkRwVersion {
    /// v3.1.10 — most current widely-deployed version (2024-2025).
    V3_1_10,
    /// v3.1.11 — current spec, adds VRP profile increment.
    V3_1_11,
}

impl UkRwVersion {
    /// Path component used when constructing endpoints, e.g.
    /// `https://aspsp/open-banking/v3.1/aisp/accounts`.
    #[must_use]
    pub const fn url_segment(self) -> &'static str {
        // OBIE keeps the same `v3.1` URL segment across all minors.
        "v3.1"
    }
}

/// UK OBIE permission scopes per the *Permissions* enum in the
/// Accounts API. We list the most-used scopes; the full enum has
/// ~40 values, every one of which is opt-in at consent creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UkScope {
    /// `ReadAccountsBasic`.
    ReadAccountsBasic,
    /// `ReadAccountsDetail`.
    ReadAccountsDetail,
    /// `ReadBalances`.
    ReadBalances,
    /// `ReadTransactionsBasic`.
    ReadTransactionsBasic,
    /// `ReadTransactionsDetail`.
    ReadTransactionsDetail,
    /// `ReadTransactionsCredits`.
    ReadTransactionsCredits,
    /// `ReadTransactionsDebits`.
    ReadTransactionsDebits,
    /// `ReadStandingOrdersBasic` / `Detail`.
    ReadStandingOrders,
    /// `ReadDirectDebits`.
    ReadDirectDebits,
    /// `ReadBeneficiaries`.
    ReadBeneficiaries,
    /// `ReadStatementsBasic` / `Detail`.
    ReadStatements,
    /// `ReadParty`.
    ReadParty,
    /// `ReadOffers`.
    ReadOffers,
    /// `ReadScheduledPaymentsBasic` / `Detail`.
    ReadScheduledPayments,
}

impl UkScope {
    /// Wire-format string of the OBIE permission.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadAccountsBasic => "ReadAccountsBasic",
            Self::ReadAccountsDetail => "ReadAccountsDetail",
            Self::ReadBalances => "ReadBalances",
            Self::ReadTransactionsBasic => "ReadTransactionsBasic",
            Self::ReadTransactionsDetail => "ReadTransactionsDetail",
            Self::ReadTransactionsCredits => "ReadTransactionsCredits",
            Self::ReadTransactionsDebits => "ReadTransactionsDebits",
            Self::ReadStandingOrders => "ReadStandingOrders",
            Self::ReadDirectDebits => "ReadDirectDebits",
            Self::ReadBeneficiaries => "ReadBeneficiaries",
            Self::ReadStatements => "ReadStatements",
            Self::ReadParty => "ReadParty",
            Self::ReadOffers => "ReadOffers",
            Self::ReadScheduledPayments => "ReadScheduledPayments",
        }
    }
}

/// A handle that combines the [`UkRwVersion`] in use with the set of
/// scopes negotiated at consent creation. The vendor-neutral service
/// traits ([`crate::AccountInfoService`], etc.) operate against the
/// scopes here; the binding layer translates that into UK-specific
/// `OBReadConsent1` / `OBWriteDomesticConsent4` payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UkOpenBankingService {
    /// OBIE spec version this service speaks.
    pub version: UkRwVersion,
    /// Scopes granted at consent creation.
    pub scopes: Vec<UkScope>,
    /// ASPSP base URL. We carry it as an opaque string; HTTP is
    /// operator-side.
    pub aspsp_base_url: String,
}

impl UkOpenBankingService {
    /// True iff the named scope is in the granted set.
    #[must_use]
    pub fn has(&self, scope: UkScope) -> bool {
        self.scopes.iter().any(|s| *s == scope)
    }

    /// Construct the full URL for an OBIE endpoint path like
    /// `/aisp/accounts`. Joining is naive (assumes no trailing slash
    /// on the base URL); operators using non-trivial bases override
    /// this on their side.
    #[must_use]
    pub fn endpoint(&self, path: &str) -> String {
        let trimmed_base = self.aspsp_base_url.trim_end_matches('/');
        let trimmed_path = path.trim_start_matches('/');
        format!(
            "{}/open-banking/{}/{}",
            trimmed_base,
            self.version.url_segment(),
            trimmed_path
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_url_segment_stays_v3_1() {
        assert_eq!(UkRwVersion::V3_1_10.url_segment(), "v3.1");
        assert_eq!(UkRwVersion::V3_1_11.url_segment(), "v3.1");
    }

    #[test]
    fn endpoint_joins_correctly() {
        let svc = UkOpenBankingService {
            version: UkRwVersion::V3_1_11,
            scopes: vec![UkScope::ReadAccountsDetail],
            aspsp_base_url: "https://api.aspsp.example".into(),
        };
        assert_eq!(
            svc.endpoint("/aisp/accounts"),
            "https://api.aspsp.example/open-banking/v3.1/aisp/accounts"
        );
    }

    #[test]
    fn has_scope_is_decisive() {
        let svc = UkOpenBankingService {
            version: UkRwVersion::V3_1_11,
            scopes: vec![UkScope::ReadBalances],
            aspsp_base_url: "x".into(),
        };
        assert!(svc.has(UkScope::ReadBalances));
        assert!(!svc.has(UkScope::ReadTransactionsDetail));
    }
}
