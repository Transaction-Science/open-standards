//! Issuer-side metadata stamped into every envelope.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::said::Said;

/// Provenance metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The principal SAID that authored this envelope.
    pub issuer: Said,
    /// Wall-clock time of issuance.
    pub issued_at: DateTime<Utc>,
    /// Opaque authorization blob. For USD bytes this typically holds a
    /// regulated-issuer attestation; for joule bytes it is empty.
    pub authorization: Vec<u8>,
}

impl Provenance {
    /// Convenience constructor.
    pub fn new(issuer: Said, issued_at: DateTime<Utc>, authorization: Vec<u8>) -> Self {
        Self {
            issuer,
            issued_at,
            authorization,
        }
    }
}
