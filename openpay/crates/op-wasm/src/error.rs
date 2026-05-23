//! FFI-safe error type for the JavaScript bridge.
//!
//! Discriminants are byte-identical to `op-ffi-swift::FfiError` and
//! `op-ffi-jni::FfiError` (Phases 8 and 9). A single observability
//! backend can correlate failures by code across iOS, Android, and
//! the web.
//!
//! ## Surface
//!
//! Rust-side errors are returned as `Result<T, JsValue>`. The
//! `JsValue` is an instance of the JS-visible class
//! [`OpenPayError`], which carries:
//!
//! - `.code` — i32 discriminant (0..=9), matching the other bridges.
//! - `.kind` — string name of the variant (e.g. `"VaultLookupFailed"`).
//! - `.message` — short human-readable description; never includes
//!   sensitive data.
//!
//! Idiomatic JS callers:
//!
//! ```text
//! try {
//!   vault.detokenize(token);
//! } catch (e) {
//!   if (e.code === 2) { ... }              // numeric switch
//!   if (e.kind === "VaultLookupFailed") { ... } // by name
//! }
//! ```

use thiserror::Error;
use wasm_bindgen::prelude::*;

/// Internal error variant, mirroring Phase 8/9.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Error)]
#[repr(i32)]
pub enum FfiError {
    /// Success sentinel.
    #[error("ok")]
    Ok = 0,

    /// Input data is malformed.
    #[error("invalid input")]
    InvalidInput = 1,

    /// Vault couldn't resolve a token. Collapses NotFound | AuthFailed |
    /// InvalidToken for oracle discipline.
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

    /// Short kind name (PascalCase) for the JS-visible `kind` field.
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Ok => "Ok",
            Self::InvalidInput => "InvalidInput",
            Self::VaultLookupFailed => "VaultLookupFailed",
            Self::TokenExpired => "TokenExpired",
            Self::TokenAlreadyConsumed => "TokenAlreadyConsumed",
            Self::FraudDeclined => "FraudDeclined",
            Self::FraudReviewRequired => "FraudReviewRequired",
            Self::Backend => "Backend",
            Self::Internal => "Internal",
            Self::Capacity => "Capacity",
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

/// JS-visible error class. Instances are constructed Rust-side and
/// returned as the `Err` arm of `Result<T, JsValue>`. wasm-bindgen
/// converts the wrapper into a JS object that the bindgen runtime
/// throws on the JS side.
#[wasm_bindgen]
pub struct OpenPayError {
    code: i32,
    kind: String,
    message: String,
}

#[wasm_bindgen]
impl OpenPayError {
    /// i32 discriminant, identical to the codes returned by the iOS
    /// and Android bridges.
    #[wasm_bindgen(getter)]
    pub fn code(&self) -> i32 {
        self.code
    }

    /// Variant name, e.g. `"VaultLookupFailed"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.clone()
    }

    /// Short human-readable description. Does not include sensitive
    /// data (no PAN, no token, no vault internals).
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    /// `toString()`-style summary for JS callers that just want to log.
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "OpenPayError [{}/{}]: {}",
            self.code, self.kind, self.message
        )
    }
}

impl OpenPayError {
    /// Construct from an [`FfiError`] discriminant and a contextual
    /// message. The message is never user-supplied; only crate-local
    /// constants and `FfiError::to_string()` outputs flow in here.
    pub fn from_ffi(e: FfiError, message: impl Into<String>) -> Self {
        Self {
            code: e.as_i32(),
            kind: e.kind().to_owned(),
            message: message.into(),
        }
    }
}

/// Convert an `FfiError` directly into a `JsValue` for return paths.
///
/// Calling sites use this with `?`:
///
/// ```text
/// fn detokenize(...) -> Result<CardData, JsValue> {
///     let vref = ...;
///     let card = vault.detokenize(&vref).map_err(jsify_vault_err)?;
///     Ok(card)
/// }
/// ```
pub(crate) fn jsify_vault_err(e: op_vault::Error) -> JsValue {
    let ffi: FfiError = e.into();
    let err = OpenPayError::from_ffi(ffi, ffi.to_string());
    JsValue::from(err)
}

/// Same for fraud errors.
#[allow(dead_code)] // not used until we expose fraud-failing surfaces
pub(crate) fn jsify_fraud_err(e: op_fraud::Error) -> JsValue {
    let ffi: FfiError = e.into();
    let err = OpenPayError::from_ffi(ffi, ffi.to_string());
    JsValue::from(err)
}

/// Direct `FfiError -> JsValue` for cases where we don't have a
/// concrete inner error (null pointer, bad input shape).
pub(crate) fn jsify_ffi(e: FfiError, message: impl Into<String>) -> JsValue {
    let err = OpenPayError::from_ffi(e, message);
    JsValue::from(err)
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
    fn kind_string_is_pascal_case_and_stable() {
        // The .kind field is part of the public JS API; renaming a
        // variant would break consumer code. Lock it in.
        assert_eq!(FfiError::Ok.kind(), "Ok");
        assert_eq!(FfiError::InvalidInput.kind(), "InvalidInput");
        assert_eq!(FfiError::VaultLookupFailed.kind(), "VaultLookupFailed");
        assert_eq!(FfiError::TokenExpired.kind(), "TokenExpired");
        assert_eq!(
            FfiError::TokenAlreadyConsumed.kind(),
            "TokenAlreadyConsumed"
        );
        assert_eq!(FfiError::FraudDeclined.kind(), "FraudDeclined");
        assert_eq!(FfiError::FraudReviewRequired.kind(), "FraudReviewRequired");
        assert_eq!(FfiError::Backend.kind(), "Backend");
        assert_eq!(FfiError::Internal.kind(), "Internal");
        assert_eq!(FfiError::Capacity.kind(), "Capacity");
    }

    #[test]
    fn discriminants_match_phase_8_swift_and_phase_9_jni() {
        // Invariant: a single observability backend uses the same
        // i32 codes across all three platform bridges. Verify the
        // contract for Phase 10 too.
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

    #[test]
    fn open_pay_error_carries_three_fields() {
        let err = OpenPayError::from_ffi(FfiError::TokenExpired, "token expired");
        assert_eq!(err.code(), 3);
        assert_eq!(err.kind(), "TokenExpired");
        assert_eq!(err.message(), "token expired");
    }

    #[test]
    fn open_pay_error_to_string_format() {
        let err = OpenPayError::from_ffi(FfiError::InvalidInput, "bad pan");
        assert_eq!(err.to_string_js(), "OpenPayError [1/InvalidInput]: bad pan");
    }
}
