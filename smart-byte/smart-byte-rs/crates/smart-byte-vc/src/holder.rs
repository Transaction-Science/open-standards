//! VC Holder. The holder is identified by a DID and may present
//! credentials inside a [`crate::presentation::VerifiablePresentation`].

use serde::{Deserialize, Serialize};

use crate::did::Did;

/// A presentation holder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Holder {
    /// Holder DID.
    pub did: Did,
}

impl Holder {
    /// Construct a holder from a DID.
    pub fn new(did: Did) -> Self {
        Self { did }
    }
}
