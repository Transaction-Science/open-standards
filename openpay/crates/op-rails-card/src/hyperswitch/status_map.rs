//! Hyperswitch status → `op-rails-card::AuthStatus` mapping.
//!
//! Verified against the Hyperswitch V1 status enum:
//! <https://api-reference.hyperswitch.io/v1/payments/payments--create>
//!
//! ## Mapping table
//!
//! | Hyperswitch status                          | `OpenPay` `AuthStatus`           |
//! |---------------------------------------------|------------------------------|
//! | `succeeded`                                 | `Settled`                    |
//! | `failed`                                    | `HardDecline`                |
//! | `cancelled`                                 | `HardDecline`                |
//! | `cancelled_post_capture`                    | `HardDecline`                |
//! | `processing`                                | `Transient`                  |
//! | `requires_customer_action`                  | `RequiresCustomerAction`     |
//! | `requires_merchant_action`                  | `RequiresMerchantAction`     |
//! | `requires_payment_method`                   | `RequiresMerchantAction`     |
//! | `requires_confirmation`                     | `RequiresMerchantAction`     |
//! | `requires_capture`                          | `AuthorizedAwaitingCapture`  |
//! | `partially_captured`                        | `Settled` (partial)          |
//! | `partially_captured_and_capturable`         | `AuthorizedAwaitingCapture`  |
//! | `partially_authorized_and_requires_capture` | `AuthorizedAwaitingCapture`  |
//! | `partially_captured_and_processing`         | `Transient`                  |
//! | `conflicted`                                | `RequiresMerchantAction`     |
//! | `expired`                                   | `HardDecline`                |
//! | `review`                                    | `Fraud`                      |
//!
//! Any other string returns `Error::UnknownStatus` rather than guessing.

use crate::acquirer::AuthStatus;
use crate::error::{Error, Result};

/// Map a Hyperswitch status string to an `OpenPay` [`AuthStatus`].
///
/// # Errors
/// Returns `Error::UnknownStatus` if the string is not in the
/// documented enum. Unknown statuses are *never* silently coerced —
/// the caller must handle them explicitly (typically by alerting and
/// treating as `Fraud` or rolling back).
pub fn map(hyperswitch_status: &str) -> Result<AuthStatus> {
    Ok(match hyperswitch_status {
        "succeeded" | "partially_captured" => AuthStatus::Settled,
        "failed" | "cancelled" | "cancelled_post_capture" | "expired" => AuthStatus::HardDecline,
        "processing" | "partially_captured_and_processing" => AuthStatus::Transient,
        "requires_customer_action" => AuthStatus::RequiresCustomerAction,
        "requires_merchant_action"
        | "requires_payment_method"
        | "requires_confirmation"
        | "conflicted" => AuthStatus::RequiresMerchantAction,
        "requires_capture"
        | "partially_captured_and_capturable"
        | "partially_authorized_and_requires_capture" => AuthStatus::AuthorizedAwaitingCapture,
        "review" => AuthStatus::Fraud,
        other => return Err(Error::UnknownStatus(other.to_owned())),
    })
}

/// Refund status mapping. Hyperswitch's refund enum is smaller:
/// `pending` | `succeeded` | `failed` | `manual_review`.
///
/// # Errors
/// `Error::UnknownStatus` on unknown values.
pub fn map_refund(hyperswitch_status: &str) -> Result<AuthStatus> {
    Ok(match hyperswitch_status {
        "pending" => AuthStatus::Transient,
        "succeeded" => AuthStatus::Settled,
        "failed" => AuthStatus::HardDecline,
        "manual_review" => AuthStatus::Fraud,
        other => return Err(Error::UnknownStatus(other.to_owned())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every documented Hyperswitch status must map to exactly one
    /// `AuthStatus`. This is the exhaustive list from the V1 docs as
    /// of 2026-05; if Hyperswitch adds a new status, this test stays
    /// green but the new status falls through to `UnknownStatus` —
    /// which is the intended behavior.
    const ALL_DOCUMENTED_STATUSES: &[&str] = &[
        "succeeded",
        "failed",
        "cancelled",
        "cancelled_post_capture",
        "processing",
        "requires_customer_action",
        "requires_merchant_action",
        "requires_payment_method",
        "requires_confirmation",
        "requires_capture",
        "partially_captured",
        "partially_captured_and_capturable",
        "partially_authorized_and_requires_capture",
        "partially_captured_and_processing",
        "conflicted",
        "expired",
        "review",
    ];

    #[test]
    fn every_documented_status_maps() {
        for s in ALL_DOCUMENTED_STATUSES {
            assert!(map(s).is_ok(), "documented status {s:?} should map cleanly");
        }
    }

    #[test]
    fn unknown_status_errors() {
        assert!(matches!(
            map("nonsense_value"),
            Err(Error::UnknownStatus(_))
        ));
        assert!(matches!(map(""), Err(Error::UnknownStatus(_))));
        assert!(matches!(map("SUCCEEDED"), Err(Error::UnknownStatus(_)))); // case-sensitive
    }

    #[test]
    fn succeeded_is_settled() {
        assert_eq!(map("succeeded").unwrap(), AuthStatus::Settled);
    }

    #[test]
    fn requires_capture_is_authorized_awaiting_capture() {
        assert_eq!(
            map("requires_capture").unwrap(),
            AuthStatus::AuthorizedAwaitingCapture
        );
    }

    #[test]
    fn requires_customer_action_indicates_3ds() {
        assert_eq!(
            map("requires_customer_action").unwrap(),
            AuthStatus::RequiresCustomerAction
        );
    }

    #[test]
    fn failed_is_hard_decline() {
        assert_eq!(map("failed").unwrap(), AuthStatus::HardDecline);
        assert_eq!(map("cancelled").unwrap(), AuthStatus::HardDecline);
        assert_eq!(map("expired").unwrap(), AuthStatus::HardDecline);
    }

    #[test]
    fn review_is_fraud() {
        assert_eq!(map("review").unwrap(), AuthStatus::Fraud);
    }

    #[test]
    fn processing_is_transient() {
        assert_eq!(map("processing").unwrap(), AuthStatus::Transient);
    }

    #[test]
    fn partial_capture_outcomes_distinguished() {
        // Fully partial -> Settled (some funds moved, no more capturable).
        assert_eq!(map("partially_captured").unwrap(), AuthStatus::Settled);
        // Multi-capture in flight -> still authorized, can capture more.
        assert_eq!(
            map("partially_captured_and_capturable").unwrap(),
            AuthStatus::AuthorizedAwaitingCapture
        );
        // Still processing.
        assert_eq!(
            map("partially_captured_and_processing").unwrap(),
            AuthStatus::Transient
        );
    }

    #[test]
    fn refund_mapping_complete() {
        assert_eq!(map_refund("pending").unwrap(), AuthStatus::Transient);
        assert_eq!(map_refund("succeeded").unwrap(), AuthStatus::Settled);
        assert_eq!(map_refund("failed").unwrap(), AuthStatus::HardDecline);
        assert_eq!(map_refund("manual_review").unwrap(), AuthStatus::Fraud);
        assert!(matches!(map_refund("xyz"), Err(Error::UnknownStatus(_))));
    }

    #[test]
    fn no_documented_status_maps_to_internal_states() {
        // None of the public Hyperswitch statuses should map to anything
        // that suggests caller error.
        for s in ALL_DOCUMENTED_STATUSES {
            let mapped = map(s).unwrap();
            // SoftDecline and Approved are reserved for non-Hyperswitch
            // sources (e.g. card-network direct rails). Hyperswitch
            // doesn't surface "soft decline" — failed is failed.
            assert_ne!(mapped, AuthStatus::SoftDecline);
            assert_ne!(mapped, AuthStatus::Approved);
        }
    }
}
