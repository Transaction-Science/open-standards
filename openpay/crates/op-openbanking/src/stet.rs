//! STET PSD2 API binding (France).
//!
//! Reference: *STET PSD2 API Documentation Part 1* v1.7. STET is the
//! French PSD2 dialect — adopted by BNP Paribas, Crédit Agricole,
//! Société Générale, BPCE, La Banque Postale. The dialect is
//! structurally similar to Berlin Group but uses different endpoint
//! and field names.
//!
//! STET v1.7 does **not** include a VRP profile. Operators needing
//! recurring debits use SEPA Direct Debit instead.

use serde::{Deserialize, Serialize};

/// STET endpoint families. Mapped to `/v1/{stet-endpoint-segment}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StetEndpointKind {
    /// `accounts` — AISP root.
    Accounts,
    /// `balances` — balance reads for a known account.
    Balances,
    /// `transactions` — transaction history.
    Transactions,
    /// `payment-requests` — PISP (single immediate or future-dated).
    PaymentRequests,
    /// `funds-confirmations` — CBPII / CoF.
    FundsConfirmations,
    /// `trusted-beneficiaries` — read of the PSU's trusted beneficiary
    /// list, used by STET for the SCA-exemption white-list flow.
    TrustedBeneficiaries,
}

impl StetEndpointKind {
    /// Path segment used under `/v1/...`.
    #[must_use]
    pub const fn as_segment(self) -> &'static str {
        match self {
            Self::Accounts => "accounts",
            Self::Balances => "balances",
            Self::Transactions => "transactions",
            Self::PaymentRequests => "payment-requests",
            Self::FundsConfirmations => "funds-confirmations",
            Self::TrustedBeneficiaries => "trusted-beneficiaries",
        }
    }
}

/// STET consent. STET pins consent semantics tightly to the PSU's
/// `psuId` (the customer-bank-issued identifier passed in the
/// `PSU-ID` header). The TPP cannot create a consent without prior
/// PSU authentication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StetConsent {
    /// `psuId` of the customer the consent applies to.
    pub psu_id: String,
    /// IBANs the consent covers (empty for `global` consents on
    /// AISP-only flows where the ASPSP returns the list).
    pub ibans: Vec<String>,
    /// Expiry per STET `validUntil`.
    pub valid_until: time::Date,
}

/// STET service handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StetService {
    /// ASPSP base URL.
    pub aspsp_base_url: String,
}

impl StetService {
    /// Build a `/v1/{segment}` URL.
    #[must_use]
    pub fn endpoint(&self, kind: StetEndpointKind) -> String {
        let trimmed = self.aspsp_base_url.trim_end_matches('/');
        format!("{}/v1/{}", trimmed, kind.as_segment())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_segments_are_kebab_case() {
        assert_eq!(StetEndpointKind::PaymentRequests.as_segment(), "payment-requests");
        assert_eq!(
            StetEndpointKind::FundsConfirmations.as_segment(),
            "funds-confirmations"
        );
    }

    #[test]
    fn endpoint_url_constructs() {
        let svc = StetService {
            aspsp_base_url: "https://api.banque.example".into(),
        };
        assert_eq!(
            svc.endpoint(StetEndpointKind::Accounts),
            "https://api.banque.example/v1/accounts"
        );
    }
}
