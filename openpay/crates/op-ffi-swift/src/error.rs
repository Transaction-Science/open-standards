//! FFI-safe error type.
//!
//! The bridge surface must not leak Rust idioms (`thiserror::Error`,
//! `std::io::Error`, etc.) to Swift. Errors are collapsed into a
//! flat numeric enum with a stable ABI. Swift sees them as a Swift
//! `enum`, the C ABI sees them as `int32_t`.
//!
//! ## Oracle discipline
//!
//! Same rule as `op_vault::Error`: `NotFound` and `AuthFailed`
//! collapse into a single Swift-facing `vaultLookupFailed` so that
//! a Swift caller can't write a probe loop that distinguishes
//! "this token exists with the wrong key" from "this token never
//! existed."
//!
//! ## ABI stability
//!
//! The discriminant values are explicit and stable. Reordering or
//! reusing values is a breaking change for Swift apps that ship a
//! pre-built `libopenpay.a` and link a newer header.

use thiserror::Error;

/// FFI-safe error.
///
/// `#[repr(i32)]` makes this directly representable across the C ABI;
/// `swift-bridge` will mirror it into a Swift enum. The discriminants
/// are stable and must not be reused.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Error)]
#[repr(i32)]
pub enum FfiError {
    /// Success sentinel for C ABI returns. Never returned by Rust-side
    /// `Result::Err`.
    #[error("ok")]
    Ok = 0,

    /// Input data is malformed (bad card number, bad expiration,
    /// invalid token format).
    #[error("invalid input")]
    InvalidInput = 1,

    /// Vault couldn't resolve a token. Collapses
    /// `NotFound | AuthFailed | InvalidToken` for oracle discipline.
    #[error("vault lookup failed")]
    VaultLookupFailed = 2,

    /// Token has expired per policy.
    #[error("token expired")]
    TokenExpired = 3,

    /// Single-use token was already consumed.
    #[error("token already consumed")]
    TokenAlreadyConsumed = 4,

    /// Fraud scorer rejected the request.
    #[error("fraud declined")]
    FraudDeclined = 5,

    /// Operation requires human review per the fraud decision.
    #[error("fraud review required")]
    FraudReviewRequired = 6,

    /// A backend (vault, rail, scorer) returned an implementation-
    /// specific failure. Translation drops the inner message — Swift
    /// gets only the category.
    #[error("backend error")]
    Backend = 7,

    /// FFI-internal: lock poisoned, allocator failure, etc. Should
    /// never appear in normal operation.
    #[error("internal error")]
    Internal = 8,

    /// Capacity / rate limit exceeded.
    #[error("rate limit or capacity")]
    Capacity = 9,
}

impl FfiError {
    /// Cast back from an `i32` returned across the C ABI. Unknown
    /// values map to `Internal`.
    #[must_use]
    pub const fn from_i32(v: i32) -> Self {
        match v {
            0 => Self::Ok,
            1 => Self::InvalidInput,
            2 => Self::VaultLookupFailed,
            3 => Self::TokenExpired,
            4 => Self::TokenAlreadyConsumed,
            5 => Self::FraudDeclined,
            6 => Self::FraudReviewRequired,
            7 => Self::Backend,
            8 => Self::Internal,
            9 => Self::Capacity,
            _ => Self::Internal,
        }
    }

    /// Discriminant as `i32` for the C ABI.
    #[must_use]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }
}

impl From<op_vault::Error> for FfiError {
    fn from(e: op_vault::Error) -> Self {
        // Collapse vault errors per oracle discipline. NotFound,
        // AuthFailed, and InvalidToken all become VaultLookupFailed so
        // a Swift caller can't probe to distinguish them.
        match e {
            op_vault::Error::NotFound
            | op_vault::Error::AuthFailed
            | op_vault::Error::InvalidToken => Self::VaultLookupFailed,
            op_vault::Error::Expired => Self::TokenExpired,
            op_vault::Error::AlreadyConsumed => Self::TokenAlreadyConsumed,
            op_vault::Error::InvalidCard(_) => Self::InvalidInput,
            op_vault::Error::Capacity => Self::Capacity,
            op_vault::Error::Backend(_) => Self::Backend,
        }
    }
}

impl From<op_fraud::Error> for FfiError {
    fn from(e: op_fraud::Error) -> Self {
        match e {
            op_fraud::Error::Features(_) => Self::InvalidInput,
            op_fraud::Error::ModelLoad(_)
            | op_fraud::Error::ModelOutput(_)
            | op_fraud::Error::Backend(_) => Self::Backend,
            op_fraud::Error::ScoreOutOfRange(_) => Self::Backend,
            op_fraud::Error::Core(_) => Self::Backend,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_i32() {
        for v in [
            FfiError::Ok,
            FfiError::InvalidInput,
            FfiError::VaultLookupFailed,
            FfiError::TokenExpired,
            FfiError::TokenAlreadyConsumed,
            FfiError::FraudDeclined,
            FfiError::FraudReviewRequired,
            FfiError::Backend,
            FfiError::Internal,
            FfiError::Capacity,
        ] {
            assert_eq!(FfiError::from_i32(v.as_i32()), v);
        }
    }

    #[test]
    fn unknown_i32_maps_to_internal() {
        assert_eq!(FfiError::from_i32(999), FfiError::Internal);
        assert_eq!(FfiError::from_i32(-1), FfiError::Internal);
    }

    #[test]
    fn vault_not_found_and_auth_failed_collapse_to_lookup_failed() {
        assert_eq!(
            FfiError::from(op_vault::Error::NotFound),
            FfiError::VaultLookupFailed
        );
        assert_eq!(
            FfiError::from(op_vault::Error::AuthFailed),
            FfiError::VaultLookupFailed
        );
        assert_eq!(
            FfiError::from(op_vault::Error::InvalidToken),
            FfiError::VaultLookupFailed
        );
    }

    #[test]
    fn vault_expired_distinct_from_lookup_failed() {
        // Expired is OK to distinguish — the caller needs to know
        // whether to retry vs re-tokenize. Not an oracle.
        assert_eq!(
            FfiError::from(op_vault::Error::Expired),
            FfiError::TokenExpired
        );
    }

    #[test]
    fn vault_consumed_distinct_from_lookup_failed() {
        assert_eq!(
            FfiError::from(op_vault::Error::AlreadyConsumed),
            FfiError::TokenAlreadyConsumed
        );
    }

    #[test]
    fn vault_invalid_card_maps_to_invalid_input() {
        assert_eq!(
            FfiError::from(op_vault::Error::InvalidCard("luhn".into())),
            FfiError::InvalidInput
        );
    }

    #[test]
    fn vault_backend_does_not_leak_inner_message() {
        // The inner string is dropped on the way through the From impl.
        // Verified by the type signature — FfiError::Backend carries no
        // payload. This test just exercises the path.
        let e = FfiError::from(op_vault::Error::Backend("secret postgres detail".into()));
        assert_eq!(e, FfiError::Backend);
        let dbg = format!("{e:?}");
        assert!(!dbg.contains("postgres"));
        assert!(!dbg.contains("secret"));
    }

    #[test]
    fn fraud_features_maps_to_invalid_input() {
        assert_eq!(
            FfiError::from(op_fraud::Error::Features("nan".into())),
            FfiError::InvalidInput
        );
    }

    #[test]
    fn fraud_backend_maps_to_backend() {
        assert_eq!(
            FfiError::from(op_fraud::Error::ModelLoad("missing".into())),
            FfiError::Backend
        );
        assert_eq!(
            FfiError::from(op_fraud::Error::ModelOutput("nan".into())),
            FfiError::Backend
        );
    }

    #[test]
    fn fraud_score_out_of_range_maps_to_backend() {
        assert_eq!(
            FfiError::from(op_fraud::Error::ScoreOutOfRange(1.5)),
            FfiError::Backend
        );
    }

    #[test]
    fn ok_sentinel_is_zero() {
        // The C ABI uses `0` for success; this is a stable invariant.
        assert_eq!(FfiError::Ok.as_i32(), 0);
    }
}
