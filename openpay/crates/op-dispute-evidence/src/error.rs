//! Error type for the dispute-evidence crate.

use thiserror::Error;

/// Crate result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failures surfaced by the dispute-evidence crate.
///
/// Variants intentionally model *domain* failures (illegal lifecycle
/// transitions, evidence the network won't accept, CE3 ineligibility)
/// rather than I/O — wire transport lives in the operator's adapter.
#[derive(Debug, Error)]
pub enum Error {
    /// Attempted to transition out of a terminal lifecycle phase.
    #[error("illegal lifecycle transition: {from:?} -> {to:?}")]
    IllegalTransition {
        /// Phase the dispute was in.
        from: crate::lifecycle::Phase,
        /// Phase the caller tried to move it to.
        to: crate::lifecycle::Phase,
    },

    /// Caller tried to submit representment without satisfying the
    /// network's required-evidence list for the given reason code.
    #[error("evidence package missing required item: {0}")]
    MissingEvidence(&'static str),

    /// The provided reason-code string did not match the catalog for
    /// the declared network.
    #[error("unknown reason code {code:?} for network {network:?}")]
    UnknownReasonCode {
        /// Network the lookup was made against.
        network: crate::network::Network,
        /// The unmatched code as the caller spelled it.
        code: String,
    },

    /// CE3.0 was requested but the supplied qualifying-transaction
    /// set does not meet Visa's published criteria.
    #[error("CE3.0 ineligible: {0}")]
    Ce3Ineligible(&'static str),

    /// An invariant on the [`EvidencePackage`](crate::evidence::EvidencePackage)
    /// was violated (e.g., empty package, oversized blob).
    #[error("invalid evidence package: {0}")]
    InvalidEvidence(&'static str),
}
