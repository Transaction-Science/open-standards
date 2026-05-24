//! Tendermint evidence — proofs of validator misbehavior.
//!
//! Tendermint slashes two classes of misbehavior:
//!
//! * **DuplicateVote** — a validator signed two different votes at the
//!   same `(height, round, kind)`. This is the prototypical
//!   double-signing offense.
//! * **LightClientAttack** — a validator signed a commit on a
//!   conflicting fork that a light client could be fooled into
//!   trusting (lunatic / equivocation / amnesia variants).

use serde::{Deserialize, Serialize};

use crate::block::Commit;
use crate::error::{Error, Result};
use crate::validator::ValidatorId;
use crate::vote::Vote;

/// Proof of a validator double-signing two distinct votes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DuplicateVoteEvidence {
    /// First conflicting vote.
    pub vote_a: Vote,
    /// Second conflicting vote.
    pub vote_b: Vote,
}

impl DuplicateVoteEvidence {
    /// Validate that the two votes are in fact a double-sign.
    ///
    /// They must:
    /// * come from the same validator;
    /// * share `(height, round, kind)`;
    /// * vote for different `block_hash` values.
    pub fn verify(&self) -> Result<()> {
        if self.vote_a.validator != self.vote_b.validator {
            return Err(Error::InvalidEvidence("validator mismatch"));
        }
        if self.vote_a.height != self.vote_b.height {
            return Err(Error::InvalidEvidence("height mismatch"));
        }
        if self.vote_a.round != self.vote_b.round {
            return Err(Error::InvalidEvidence("round mismatch"));
        }
        if self.vote_a.kind != self.vote_b.kind {
            return Err(Error::InvalidEvidence("kind mismatch"));
        }
        if self.vote_a.block_hash == self.vote_b.block_hash {
            return Err(Error::InvalidEvidence("votes agree"));
        }
        Ok(())
    }

    /// The misbehaving validator.
    pub fn offender(&self) -> ValidatorId {
        self.vote_a.validator
    }
}

/// Proof that a conflicting block commit exists at the same height.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LightClientAttackEvidence {
    /// The honest, canonical commit.
    pub canonical: Commit,
    /// The conflicting commit that fooled (or could fool) the light client.
    pub conflicting: Commit,
}

impl LightClientAttackEvidence {
    /// Validate the evidence: same height, distinct block hashes.
    pub fn verify(&self) -> Result<()> {
        if self.canonical.height != self.conflicting.height {
            return Err(Error::InvalidEvidence("height mismatch"));
        }
        if self.canonical.block_hash == self.conflicting.block_hash {
            return Err(Error::InvalidEvidence("commits agree"));
        }
        Ok(())
    }

    /// Compute the intersection of validators that signed both commits —
    /// these are the offending validators.
    pub fn offenders(&self) -> Vec<ValidatorId> {
        let a: std::collections::HashSet<_> =
            self.canonical.signatures.iter().map(|s| s.validator).collect();
        self.conflicting
            .signatures
            .iter()
            .map(|s| s.validator)
            .filter(|id| a.contains(id))
            .collect()
    }
}

/// Discriminated evidence union submitted to the chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Evidence {
    /// A validator double-signed.
    DuplicateVote(DuplicateVoteEvidence),
    /// A light-client attack was detected.
    LightClientAttack(LightClientAttackEvidence),
}

impl Evidence {
    /// Validate the evidence.
    pub fn verify(&self) -> Result<()> {
        match self {
            Self::DuplicateVote(e) => e.verify(),
            Self::LightClientAttack(e) => e.verify(),
        }
    }
}
