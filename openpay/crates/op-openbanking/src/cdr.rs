//! Australia Consumer Data Right (CDR) — banking-tier binding.
//!
//! Reference: Consumer Data Standards Australia v1.31 (Data Standards
//! Body, ACCC). Endpoints rooted at `/cds-au/v1/banking/`.
//!
//! CDR is not strictly PSD2-shaped: it is a regulator-led data-sharing
//! regime governed by the Treasury and enforced by the ACCC. The
//! banking sector was the first tranche; energy and telco followed.
//!
//! Distinctive features:
//!
//! - Arrangements (consents) are governed by the *CDR Consumer Data
//!   Standards* with explicit `sharingExpiresAt` and a maximum
//!   12-month sharing window.
//! - The Accredited Data Recipient model (ADR / OSP / Trusted
//!   Adviser) gates who can register, not just what they can read.
//! - There is no payment-initiation surface in v1.31 (initial
//!   action-initiation rules were paused by Treasury in 2024).

use serde::{Deserialize, Serialize};

/// CDR banking permissions per the consumer-data-standards
/// authorisation scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdrScope {
    /// `bank:accounts.basic:read`.
    AccountsBasicRead,
    /// `bank:accounts.detail:read`.
    AccountsDetailRead,
    /// `bank:transactions:read`.
    TransactionsRead,
    /// `bank:payees:read`.
    PayeesRead,
    /// `bank:regular_payments:read`.
    RegularPaymentsRead,
    /// `common:customer.basic:read`.
    CustomerBasicRead,
    /// `common:customer.detail:read`.
    CustomerDetailRead,
}

impl CdrScope {
    /// Wire-format scope string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AccountsBasicRead => "bank:accounts.basic:read",
            Self::AccountsDetailRead => "bank:accounts.detail:read",
            Self::TransactionsRead => "bank:transactions:read",
            Self::PayeesRead => "bank:payees:read",
            Self::RegularPaymentsRead => "bank:regular_payments:read",
            Self::CustomerBasicRead => "common:customer.basic:read",
            Self::CustomerDetailRead => "common:customer.detail:read",
        }
    }
}

/// CDR data-sharing arrangement (consent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdrArrangement {
    /// `cdrArrangementId` minted by the data holder.
    pub id: String,
    /// Scopes granted in this arrangement.
    pub scopes: Vec<CdrScope>,
    /// Account identifiers shared via this arrangement. Empty when
    /// the arrangement covers "all eligible accounts".
    pub account_ids: Vec<String>,
    /// `sharingExpiresAt` — RFC 3339 timestamp. CDS caps this at
    /// 12 months from creation.
    pub expires_at: time::OffsetDateTime,
}

impl CdrArrangement {
    /// True when this arrangement permits the requested scope.
    #[must_use]
    pub fn has(&self, scope: CdrScope) -> bool {
        self.scopes.iter().any(|s| *s == scope)
    }
}

/// CDR banking service handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdrBankingService {
    /// Data holder base URL.
    pub holder_base_url: String,
    /// CDR x-v API version supported by the data holder
    /// (`x-v` header; current minimum is `2` for banking).
    pub x_v: u8,
}

impl CdrBankingService {
    /// Build a `/cds-au/v1/banking/{path}` URL.
    #[must_use]
    pub fn endpoint(&self, path: &str) -> String {
        let trimmed_base = self.holder_base_url.trim_end_matches('/');
        let trimmed_path = path.trim_start_matches('/');
        format!("{}/cds-au/v1/banking/{}", trimmed_base, trimmed_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_strings_match_register() {
        assert_eq!(
            CdrScope::TransactionsRead.as_str(),
            "bank:transactions:read"
        );
        assert_eq!(
            CdrScope::CustomerDetailRead.as_str(),
            "common:customer.detail:read"
        );
    }

    #[test]
    fn endpoint_root_is_cds_au_v1_banking() {
        let svc = CdrBankingService {
            holder_base_url: "https://api.bank.example".into(),
            x_v: 2,
        };
        assert_eq!(
            svc.endpoint("accounts"),
            "https://api.bank.example/cds-au/v1/banking/accounts"
        );
    }

    #[test]
    fn arrangement_scope_check() {
        let a = CdrArrangement {
            id: "arr-1".into(),
            scopes: vec![CdrScope::AccountsDetailRead, CdrScope::TransactionsRead],
            account_ids: vec![],
            expires_at: time::OffsetDateTime::UNIX_EPOCH,
        };
        assert!(a.has(CdrScope::TransactionsRead));
        assert!(!a.has(CdrScope::PayeesRead));
    }
}
