//! Validator set and voting-power accounting.
//!
//! Tendermint binds Byzantine fault tolerance to **voting power**, not
//! to validator count. The 2/3+1 threshold (`total > 2/3 · N`) is
//! computed over the sum of `voting_power` of the validators that
//! signed a vote.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Stable identifier for a validator: a 32-byte ed25519 public key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValidatorId(pub [u8; 32]);

impl ValidatorId {
    /// Construct from a raw public-key byte array.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A single validator entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Validator {
    /// Stable validator identity (ed25519 public key).
    pub id: ValidatorId,
    /// Voting power. Larger = more weight when accumulating votes.
    pub voting_power: u64,
}

impl Validator {
    /// Construct a validator with a given id and voting power.
    pub fn new(id: ValidatorId, voting_power: u64) -> Self {
        Self { id, voting_power }
    }
}

/// An ordered set of validators with cached total voting power.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatorSet {
    validators: Vec<Validator>,
    total_voting_power: u64,
}

impl ValidatorSet {
    /// Build a validator set from an iterator of validators.
    ///
    /// Returns [`Error::VotingPowerOverflow`] if the sum of voting
    /// power would overflow `u64`.
    pub fn new<I: IntoIterator<Item = Validator>>(validators: I) -> Result<Self> {
        let validators: Vec<Validator> = validators.into_iter().collect();
        let mut total: u64 = 0;
        for v in &validators {
            total = total
                .checked_add(v.voting_power)
                .ok_or(Error::VotingPowerOverflow)?;
        }
        Ok(Self {
            validators,
            total_voting_power: total,
        })
    }

    /// Number of validators in the set.
    pub fn len(&self) -> usize {
        self.validators.len()
    }

    /// Returns `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.validators.is_empty()
    }

    /// Total voting power across all validators.
    pub fn total_voting_power(&self) -> u64 {
        self.total_voting_power
    }

    /// The 2/3+1 voting-power threshold required for commit.
    ///
    /// Tendermint's BFT bound: a quorum is `floor(2T/3) + 1`.
    pub fn quorum(&self) -> u64 {
        (self.total_voting_power * 2) / 3 + 1
    }

    /// The 1/3+1 voting-power threshold required to trust a light-client skip.
    pub fn trust_threshold(&self) -> u64 {
        self.total_voting_power / 3 + 1
    }

    /// Iterate over validators.
    pub fn iter(&self) -> impl Iterator<Item = &Validator> {
        self.validators.iter()
    }

    /// Look up a validator by id.
    pub fn get(&self, id: &ValidatorId) -> Option<&Validator> {
        self.validators.iter().find(|v| &v.id == id)
    }

    /// Look up voting power for a validator id.
    pub fn voting_power_of(&self, id: &ValidatorId) -> Result<u64> {
        self.get(id).map(|v| v.voting_power).ok_or(Error::UnknownValidator)
    }

    /// Deterministic proposer selection by round-robin on round number.
    ///
    /// Production Tendermint uses an "accum" weighted round-robin; we
    /// implement plain round-robin since the wire protocol does not
    /// require the proposer algorithm to be byte-identical — only the
    /// finality voting math is consensus-critical.
    pub fn proposer(&self, height: u64, round: u32) -> Option<&Validator> {
        if self.validators.is_empty() {
            return None;
        }
        let idx = ((height.wrapping_add(u64::from(round))) as usize) % self.validators.len();
        self.validators.get(idx)
    }

    /// Sum voting power for a subset of ids, error on unknown id.
    pub fn sum_power<'a, I: IntoIterator<Item = &'a ValidatorId>>(&self, ids: I) -> Result<u64> {
        let mut sum: u64 = 0;
        for id in ids {
            sum = sum
                .checked_add(self.voting_power_of(id)?)
                .ok_or(Error::VotingPowerOverflow)?;
        }
        Ok(sum)
    }

    /// Return `true` if `signers` represent a 2/3+1 voting-power quorum.
    pub fn has_quorum<'a, I: IntoIterator<Item = &'a ValidatorId>>(&self, signers: I) -> bool {
        match self.sum_power(signers) {
            Ok(power) => power >= self.quorum(),
            Err(_) => false,
        }
    }
}
