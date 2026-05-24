//! Typed errors for `eoc-carbon`.

use thiserror::Error;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, CarbonError>;

/// All failures surfaced by carbon ingest / accounting / scheduling.
#[derive(Debug, Error)]
pub enum CarbonError {
    /// HTTP transport error from a live provider (only reachable with the
    /// `http` feature enabled).
    #[error("http error: {0}")]
    Http(String),

    /// Failed to deserialize a provider response.
    #[error("decode error: {0}")]
    Decode(String),

    /// The requested zone or region is not in any catalog.
    #[error("unknown zone: {0}")]
    UnknownZone(String),

    /// The provider returned no data for the request.
    #[error("no data: {0}")]
    NoData(String),

    /// Scheduler was given an empty candidate set.
    #[error("empty candidate set")]
    Empty,

    /// Cargo feature required for this call is not enabled.
    #[error("feature {0:?} is required")]
    FeatureDisabled(&'static str),

    /// Catch-all.
    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for CarbonError {
    fn from(err: serde_json::Error) -> Self {
        CarbonError::Decode(err.to_string())
    }
}
