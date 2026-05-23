//! Refund reason taxonomy.
//!
//! These codes are operator-supplied and don't drive any rail-side
//! logic in this crate — most PSPs accept a free-form reason string
//! anyway. The taxonomy exists so reports and audit trails have a
//! stable categorical field for filtering.

use serde::{Deserialize, Serialize};

/// Why a refund was issued.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefundReason {
    /// Customer asked for it.
    CustomerRequest,
    /// Same payment was charged twice; this one reverses the
    /// duplicate.
    DuplicateCharge,
    /// The charge was fraudulent — issued in response to a
    /// customer's "I didn't authorize this" dispute that didn't
    /// reach the chargeback stage.
    FraudulentCharge,
    /// Merchant decided to refund (subscription cancellation,
    /// service-not-rendered, goodwill).
    MerchantInitiated,
    /// Refund issued after a dispute was lost or settled; ties
    /// back to `op-dispute` via the operator's bookkeeping.
    DisputeResolution,
    /// Anything not covered above. Carries a short free-form
    /// description for the operator's report layer.
    Other(String),
}

impl RefundReason {
    /// Canonical short code suitable for ledger metadata or
    /// reporting columns.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::CustomerRequest => "customer_request",
            Self::DuplicateCharge => "duplicate_charge",
            Self::FraudulentCharge => "fraudulent_charge",
            Self::MerchantInitiated => "merchant_initiated",
            Self::DisputeResolution => "dispute_resolution",
            Self::Other(_) => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_strings() {
        assert_eq!(RefundReason::CustomerRequest.code(), "customer_request");
        assert_eq!(RefundReason::DuplicateCharge.code(), "duplicate_charge");
        assert_eq!(RefundReason::Other("legal-hold".to_owned()).code(), "other");
    }

    #[test]
    fn round_trips_via_json() {
        let r = RefundReason::Other("late-cancellation".to_owned());
        let s = serde_json::to_string(&r).unwrap();
        let back: RefundReason = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
