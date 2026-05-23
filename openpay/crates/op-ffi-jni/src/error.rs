//! FFI-safe error type for JNI.
//!
//! Identical to the Phase 8 `op-ffi-swift::FfiError` discriminants —
//! we deliberately keep them ABI-compatible so a single OpenPay
//! observability stack can correlate failures across iOS and Android.
//!
//! ## Translation strategy
//!
//! Rather than just returning an `i32` (as the Swift bridge does), the
//! JNI surface translates errors into Java exceptions of typed
//! classes:
//!
//! - `dev.openpay.OpenPayException` — base class
//! - `dev.openpay.OpenPayException$InvalidInput`
//! - `dev.openpay.OpenPayException$VaultLookupFailed`
//! - `dev.openpay.OpenPayException$TokenExpired`
//! - `dev.openpay.OpenPayException$TokenAlreadyConsumed`
//! - `dev.openpay.OpenPayException$FraudDeclined`
//! - `dev.openpay.OpenPayException$FraudReviewRequired`
//! - `dev.openpay.OpenPayException$Backend`
//! - `dev.openpay.OpenPayException$Capacity`
//!
//! Each native method that can fail either returns a default value
//! (0, null, -1) and throws, or signals the discriminant via
//! `op_last_error` for the C ABI surface. Idiomatic Kotlin callers
//! get exceptions; raw NDK consumers get the same i32 codes as Swift.

use thiserror::Error;

/// FFI-safe error code. Same discriminants as `op-ffi-swift::FfiError`
/// so a single observability layer can correlate across platforms.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Error)]
#[repr(i32)]
pub enum FfiError {
    /// Success sentinel.
    #[error("ok")]
    Ok = 0,

    /// Input data is malformed.
    #[error("invalid input")]
    InvalidInput = 1,

    /// Vault couldn't resolve a token. Collapses NotFound | AuthFailed
    /// | InvalidToken for oracle discipline.
    #[error("vault lookup failed")]
    VaultLookupFailed = 2,

    /// Token has expired.
    #[error("token expired")]
    TokenExpired = 3,

    /// Single-use token was already consumed.
    #[error("token already consumed")]
    TokenAlreadyConsumed = 4,

    /// Fraud scorer rejected the request.
    #[error("fraud declined")]
    FraudDeclined = 5,

    /// Fraud scorer flagged for human review.
    #[error("fraud review required")]
    FraudReviewRequired = 6,

    /// Backend (vault / rail / scorer) opaque failure.
    #[error("backend error")]
    Backend = 7,

    /// FFI-internal.
    #[error("internal error")]
    Internal = 8,

    /// Rate-limit or capacity exhaustion.
    #[error("rate limit or capacity")]
    Capacity = 9,
}

impl FfiError {
    /// Cast back from an `i32`. Unknown values map to `Internal`.
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

    /// Discriminant as i32.
    #[must_use]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// JNI class name of the matching Java exception.
    #[must_use]
    pub const fn exception_class(self) -> &'static str {
        match self {
            // Ok / Internal don't have a dedicated subclass; map to base.
            Self::Ok => "dev/openpay/OpenPayException",
            Self::InvalidInput => "dev/openpay/OpenPayException$InvalidInput",
            Self::VaultLookupFailed => "dev/openpay/OpenPayException$VaultLookupFailed",
            Self::TokenExpired => "dev/openpay/OpenPayException$TokenExpired",
            Self::TokenAlreadyConsumed => "dev/openpay/OpenPayException$TokenAlreadyConsumed",
            Self::FraudDeclined => "dev/openpay/OpenPayException$FraudDeclined",
            Self::FraudReviewRequired => "dev/openpay/OpenPayException$FraudReviewRequired",
            Self::Backend => "dev/openpay/OpenPayException$Backend",
            Self::Internal => "dev/openpay/OpenPayException",
            Self::Capacity => "dev/openpay/OpenPayException$Capacity",
        }
    }
}

impl From<op_vault::Error> for FfiError {
    fn from(e: op_vault::Error) -> Self {
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
            | op_fraud::Error::Backend(_)
            | op_fraud::Error::ScoreOutOfRange(_)
            | op_fraud::Error::Core(_) => Self::Backend,
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
    fn vault_oracle_collapse_preserved() {
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
    fn expired_consumed_distinct_from_lookup() {
        assert_eq!(
            FfiError::from(op_vault::Error::Expired),
            FfiError::TokenExpired
        );
        assert_eq!(
            FfiError::from(op_vault::Error::AlreadyConsumed),
            FfiError::TokenAlreadyConsumed
        );
    }

    #[test]
    fn invalid_card_to_invalid_input() {
        assert_eq!(
            FfiError::from(op_vault::Error::InvalidCard("luhn".into())),
            FfiError::InvalidInput
        );
    }

    #[test]
    fn backend_does_not_leak_inner_message() {
        let e = FfiError::from(op_vault::Error::Backend("postgres detail".into()));
        assert_eq!(e, FfiError::Backend);
        let dbg = format!("{e:?}");
        assert!(!dbg.contains("postgres"));
    }

    #[test]
    fn fraud_features_to_invalid_input() {
        assert_eq!(
            FfiError::from(op_fraud::Error::Features("nan".into())),
            FfiError::InvalidInput
        );
    }

    #[test]
    fn fraud_other_errors_to_backend() {
        assert_eq!(
            FfiError::from(op_fraud::Error::ModelLoad("missing".into())),
            FfiError::Backend
        );
        assert_eq!(
            FfiError::from(op_fraud::Error::ScoreOutOfRange(1.5)),
            FfiError::Backend
        );
    }

    #[test]
    fn exception_class_paths_use_slashes_for_jni() {
        // JNI class names use forward slashes, not dots. This is a
        // hard requirement of FindClass / ThrowNew.
        for v in [
            FfiError::InvalidInput,
            FfiError::VaultLookupFailed,
            FfiError::TokenExpired,
            FfiError::TokenAlreadyConsumed,
            FfiError::FraudDeclined,
            FfiError::FraudReviewRequired,
            FfiError::Backend,
            FfiError::Capacity,
        ] {
            let cls = v.exception_class();
            assert!(cls.contains('/'), "JNI class names use slashes: {cls}");
            assert!(!cls.contains('.'), "no dots in JNI names: {cls}");
            assert!(cls.starts_with("dev/openpay/"), "wrong package: {cls}");
        }
    }

    #[test]
    fn exception_class_for_ok_returns_base() {
        // Ok should never be thrown, but the class lookup still needs
        // to return something well-formed. Map to the base.
        assert_eq!(
            FfiError::Ok.exception_class(),
            "dev/openpay/OpenPayException"
        );
        assert_eq!(
            FfiError::Internal.exception_class(),
            "dev/openpay/OpenPayException"
        );
    }

    #[test]
    fn discriminants_match_phase_8_swift_for_cross_platform_correlation() {
        // This invariant is the basis for unified observability:
        // the i32 wire values must be identical between iOS and Android.
        assert_eq!(FfiError::Ok.as_i32(), 0);
        assert_eq!(FfiError::InvalidInput.as_i32(), 1);
        assert_eq!(FfiError::VaultLookupFailed.as_i32(), 2);
        assert_eq!(FfiError::TokenExpired.as_i32(), 3);
        assert_eq!(FfiError::TokenAlreadyConsumed.as_i32(), 4);
        assert_eq!(FfiError::FraudDeclined.as_i32(), 5);
        assert_eq!(FfiError::FraudReviewRequired.as_i32(), 6);
        assert_eq!(FfiError::Backend.as_i32(), 7);
        assert_eq!(FfiError::Internal.as_i32(), 8);
        assert_eq!(FfiError::Capacity.as_i32(), 9);
    }
}
