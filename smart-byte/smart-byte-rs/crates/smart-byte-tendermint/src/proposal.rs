//! Block proposals.
//!
//! Each round, the validator selected by [`ValidatorSet::proposer`]
//! broadcasts a `Proposal` carrying the candidate block. Other
//! validators validate the proposal against the round, height, and
//! proposer before issuing prevotes.
//!
//! [`ValidatorSet::proposer`]: crate::validator::ValidatorSet::proposer

use serde::{Deserialize, Serialize};

use crate::block::Hash;
use crate::validator::ValidatorId;
use crate::vote::Signature;

/// A signed block proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    /// Height being proposed.
    pub height: u64,
    /// Round at which this proposal was made.
    pub round: u32,
    /// Block hash of the candidate block.
    pub block_hash: Hash,
    /// "Proof of lock round" — `Some(r)` if the proposer is reproposing
    /// a block it locked in an earlier round, else `None`.
    pub pol_round: Option<u32>,
    /// Validator id of the proposer (must match the round's elected proposer).
    pub proposer: ValidatorId,
    /// Signature over the canonical proposal bytes.
    pub signature: Signature,
}

impl Proposal {
    /// Canonical sign-bytes for the proposal.
    pub fn sign_bytes(&self, chain_id: &str) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(chain_id.as_bytes());
        hasher.update(b"proposal");
        hasher.update(&self.height.to_be_bytes());
        hasher.update(&self.round.to_be_bytes());
        hasher.update(&self.block_hash);
        match self.pol_round {
            Some(r) => {
                hasher.update(&[1]);
                hasher.update(&r.to_be_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(self.proposer.as_bytes());
        *hasher.finalize().as_bytes()
    }
}
