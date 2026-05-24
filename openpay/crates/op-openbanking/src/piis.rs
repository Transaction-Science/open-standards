//! Payment Instrument Issuer Service (PIIS) — Confirmation of Funds.
//!
//! PSD2 Article 65 (CBPII) and UK Open Banking R/W v3.1 § Funds
//! Confirmations. The CBPII (Card-Based Payment Instrument Issuer)
//! asks the ASPSP "does this account hold at least X today?" without
//! seeing balance or transaction detail.
//!
//! The response is **a single boolean**. Standards explicitly forbid
//! returning the actual balance — that would defeat the privacy
//! promise of CoF.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::aisp::ConsentId;
use crate::error::Result;
use crate::fapi::OAuth2Token;

/// CoF request. The account identifier is binding-specific (sort-code
/// + account-no for UK, IBAN for SEPA, account+routing for FDX).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CofRequest {
    /// ASPSP-scoped account identifier the funds question is asked of.
    pub account_identifier: String,
    /// Threshold amount.
    pub amount: Money,
}

/// CoF response. The boolean is the contract; the timestamp lets
/// operators trace when the ASPSP evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CofResponse {
    /// True iff the account had at least [`CofRequest::amount`] at
    /// [`Self::evaluated_at`].
    pub funds_available: bool,
    /// When the ASPSP evaluated the request.
    pub evaluated_at: time::OffsetDateTime,
}

/// Confirmation-of-funds service trait.
pub trait FundsConfirmationService: Send + Sync {
    /// Create a CoF consent. Long-lived: typically tied to the card
    /// product's lifecycle and revoked when the card is closed.
    fn create_consent(&self, token: &OAuth2Token, request: &CofRequest) -> Result<ConsentId>;

    /// Ask the ASPSP whether funds are available right now.
    fn confirm_funds(
        &self,
        consent: &ConsentId,
        token: &OAuth2Token,
        request: &CofRequest,
    ) -> Result<CofResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    #[test]
    fn cof_response_serde_round_trips() {
        let r = CofResponse {
            funds_available: true,
            evaluated_at: time::OffsetDateTime::UNIX_EPOCH,
        };
        let json = serde_json::to_string(&r).expect("ser");
        assert!(json.contains("true"));
        let back: CofResponse = serde_json::from_str(&json).expect("de");
        assert_eq!(r, back);
    }

    #[test]
    fn cof_request_carries_threshold() {
        let r = CofRequest {
            account_identifier: "GB29NWBK60161331926819".into(),
            amount: Money::from_minor(50_00, Currency::GBP),
        };
        assert_eq!(r.amount.minor_units, 5000);
    }
}
