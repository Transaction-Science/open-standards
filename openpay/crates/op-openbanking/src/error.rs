//! Error types for `op-openbanking`.
//!
//! One sealed enum per crate convention. Adding a variant is a
//! SemVer-major change.

use thiserror::Error;

/// Result alias for `op-openbanking`.
pub type Result<T> = core::result::Result<T, Error>;

/// All possible failure modes inside `op-openbanking`.
#[derive(Debug, Error)]
pub enum Error {
    /// A consent identifier was referenced but does not exist or has expired.
    #[error("consent not found or expired: {0}")]
    ConsentNotFound(String),

    /// The consent is in a state that does not permit the requested operation.
    ///
    /// E.g. an AISP read against a consent that is still `AwaitingAuthorisation`,
    /// or a PISP execution against a `Rejected` consent.
    #[error("consent state invalid: {reason}")]
    ConsentStateInvalid {
        /// Human-readable reason.
        reason: String,
    },

    /// The requested scope was not granted at consent creation. UK
    /// OBIE and Berlin Group both treat this as a hard 403.
    #[error("scope not authorised: {0}")]
    ScopeNotAuthorised(String),

    /// FAPI-required header is missing (e.g. `x-fapi-interaction-id`,
    /// `x-jws-signature`, `psu-id`).
    #[error("FAPI header missing: {0}")]
    FapiHeaderMissing(&'static str),

    /// JWS request-object signature failed verification.
    ///
    /// We do not say *which* part failed; that's a per-ASPSP audit
    /// concern. Operators receive a structured trace via [`tracing`].
    #[error("JWS signature verification failed")]
    JwsSignatureInvalid,

    /// The signer trait impl returned an upstream error (HSM offline,
    /// KMS-policy denial, eIDAS card removed).
    #[error("JWS signer error: {0}")]
    Signer(String),

    /// mTLS client-certificate binding did not match the OAuth2 token
    /// (RFC 8705 § 3 certificate-bound access tokens).
    #[error("certificate binding mismatch (RFC 8705)")]
    CertificateBindingMismatch,

    /// JWK registration / lookup failed in the operator-supplied
    /// [`crate::fapi::JwkRegistration`] impl.
    #[error("JWK registration error: {0}")]
    JwkRegistration(String),

    /// OAuth 2.0 token rejected: expired, audience mismatched, scope
    /// downgraded, or simply not produced by the issuer we expect.
    #[error("OAuth2 token invalid: {reason}")]
    OAuth2TokenInvalid {
        /// Why the token failed validation.
        reason: String,
    },

    /// A payment-initiation payload failed validation (currency mismatch,
    /// zero amount, missing remittance info on a SEPA SCT, etc.).
    #[error("payment initiation invalid: {reason}")]
    PaymentInitiationInvalid {
        /// Why the initiation was rejected.
        reason: String,
    },

    /// A VRP execution exceeded the consent's [`crate::vrp::VrpControlParameters`]
    /// (per-payment, per-period, or aggregate cap).
    #[error("VRP control parameters exceeded: {reason}")]
    VrpLimitExceeded {
        /// Which limit was breached.
        reason: String,
    },

    /// The ASPSP returned a documented error that does not map to any
    /// of the variants above.
    #[error("ASPSP error ({code}): {reason}")]
    AspspError {
        /// Standard-specific code, copied verbatim.
        code: String,
        /// Human-readable reason from the ASPSP response.
        reason: String,
    },

    /// Currency mismatch between request, account, and (optionally) consent.
    #[error("currency mismatch: {0}")]
    CurrencyMismatch(String),

    /// Arithmetic overflow on a Money sum.
    #[error("arithmetic overflow")]
    Overflow,

    /// The standard binding does not support the requested operation
    /// (e.g. VRP under STET, which has no VRP profile in v1.7).
    #[error("operation not supported by binding: {0}")]
    UnsupportedByBinding(&'static str),

    /// Anything bubbled up from `op-core`.
    #[error("op-core error: {0}")]
    Core(#[from] op_core::Error),
}
