//! Dispute reason codes.
//!
//! Card networks publish their own taxonomies (Visa Reason Codes,
//! Mastercard, Amex, Discover) and the dispute API typically maps
//! the inbound network-specific code to a normalized class. A2A
//! rails have similar conventions (UK Faster Payments `Confirmation
//! of Payee` disputes, `FedNow` exception messages).
//!
//! We expose a curated cross-network taxonomy. Operators who want
//! the raw network code keep it in `Dispute::network_reason_code`.

use serde::{Deserialize, Serialize};

/// Cross-network normalized dispute class.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisputeReason {
    /// Cardholder claims they didn't authorize the transaction.
    /// Visa code family 10 (fraud), Mastercard 4837/4863.
    Fraudulent,
    /// Cardholder didn't receive the goods or service.
    /// Visa 13.1, Mastercard 4855.
    ProductNotReceived,
    /// Goods/service materially different from description.
    /// Visa 13.3, Mastercard 4853.
    NotAsDescribed,
    /// Duplicate charge — same transaction billed twice.
    /// Visa 12.6.1, Mastercard 4834.
    Duplicate,
    /// Cardholder cancelled the recurring service before the
    /// charge. Visa 13.2, Mastercard 4841.
    CancelledSubscription,
    /// Credit (refund) was promised but never appeared.
    /// Visa 13.6, Mastercard 4860.
    CreditNotProcessed,
    /// Authorization mishap (e.g. PSP captured against expired
    /// auth). Visa 11 family.
    AuthorizationIssue,
    /// Processing mishap that's the merchant's responsibility.
    /// Visa 12 family.
    ProcessingError,
    /// Catch-all for anything not in the table above. Carries the
    /// network's own reason-code string verbatim.
    Other(String),
}

impl DisputeReason {
    /// Stable short code for filtering and reporting.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::Fraudulent => "fraudulent",
            Self::ProductNotReceived => "product_not_received",
            Self::NotAsDescribed => "not_as_described",
            Self::Duplicate => "duplicate",
            Self::CancelledSubscription => "cancelled_subscription",
            Self::CreditNotProcessed => "credit_not_processed",
            Self::AuthorizationIssue => "authorization_issue",
            Self::ProcessingError => "processing_error",
            Self::Other(_) => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable() {
        assert_eq!(DisputeReason::Fraudulent.code(), "fraudulent");
        assert_eq!(DisputeReason::Other("4999".into()).code(), "other");
    }

    #[test]
    fn round_trips_via_json() {
        let r = DisputeReason::Fraudulent;
        let s = serde_json::to_string(&r).unwrap();
        let back: DisputeReason = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
