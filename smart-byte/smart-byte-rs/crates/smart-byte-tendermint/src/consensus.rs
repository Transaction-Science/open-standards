//! Tendermint consensus state machine.
//!
//! Implements the four-step round model described in Buchman (2018):
//! `Propose → Prevote → Precommit → Commit`. Each round increments
//! `round` and re-enters the cycle. Voting power is what drives
//! transitions: 2/3+1 prevotes for the same `block_id` "polka"s the
//! round; 2/3+1 precommits commit the block.

use serde::{Deserialize, Serialize};

use crate::block::Hash;
use crate::error::{Error, Result};
use crate::proposal::Proposal;
use crate::validator::{ValidatorId, ValidatorSet};
use crate::vote::{Vote, VoteType};

/// Round counter.
pub type Round = u32;

/// Phases inside a single Tendermint round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Step {
    /// Awaiting a proposal from the round's proposer.
    Propose,
    /// Collecting prevotes.
    Prevote,
    /// Collecting precommits.
    Precommit,
    /// 2/3+1 precommits reached for a real block — block is committed.
    Commit,
}

/// Cumulative tally of votes for one specific `block_hash` (or nil).
#[derive(Debug, Default, Clone)]
struct Tally {
    voters: Vec<ValidatorId>,
    power: u64,
}

impl Tally {
    fn add(&mut self, voter: ValidatorId, power: u64) -> bool {
        if self.voters.contains(&voter) {
            return false;
        }
        self.voters.push(voter);
        self.power = self.power.saturating_add(power);
        true
    }
}

/// State machine for a single height.
#[derive(Debug, Clone)]
pub struct ConsensusState {
    chain_id: String,
    height: u64,
    round: Round,
    step: Step,
    validators: ValidatorSet,

    /// Currently received proposal for `round`, if any.
    proposal: Option<Proposal>,

    /// `locked_block` survives across rounds: a validator that has
    /// precommitted for a block stays locked until it sees 2/3+1
    /// prevotes for a different block at a later round.
    locked_block: Option<Hash>,
    locked_round: Option<Round>,

    /// Tallies for the current round.
    prevotes: std::collections::HashMap<Option<Hash>, Tally>,
    precommits: std::collections::HashMap<Option<Hash>, Tally>,

    /// Set once 2/3+1 precommits arrive for a real block.
    committed: Option<Hash>,
}

impl ConsensusState {
    /// Begin consensus for a given height with the supplied validator set.
    pub fn new(chain_id: impl Into<String>, height: u64, validators: ValidatorSet) -> Self {
        Self {
            chain_id: chain_id.into(),
            height,
            round: 0,
            step: Step::Propose,
            validators,
            proposal: None,
            locked_block: None,
            locked_round: None,
            prevotes: std::collections::HashMap::new(),
            precommits: std::collections::HashMap::new(),
            committed: None,
        }
    }

    /// Chain id this state machine is bound to.
    pub fn chain_id(&self) -> &str {
        &self.chain_id
    }

    /// Current height.
    pub fn height(&self) -> u64 {
        self.height
    }

    /// Current round.
    pub fn round(&self) -> Round {
        self.round
    }

    /// Current step.
    pub fn step(&self) -> Step {
        self.step
    }

    /// Committed block hash, if any.
    pub fn committed(&self) -> Option<Hash> {
        self.committed
    }

    /// Borrow the validator set.
    pub fn validators(&self) -> &ValidatorSet {
        &self.validators
    }

    /// Apply a `Proposal` from the round's proposer.
    pub fn apply_proposal(&mut self, proposal: Proposal) -> Result<()> {
        if proposal.height != self.height {
            return Err(Error::InvalidProposal("wrong height"));
        }
        if proposal.round != self.round {
            return Err(Error::InvalidProposal("wrong round"));
        }
        let expected = self
            .validators
            .proposer(self.height, self.round)
            .ok_or(Error::InvalidProposal("no proposer"))?;
        if expected.id != proposal.proposer {
            return Err(Error::InvalidProposal("wrong proposer"));
        }
        if self.step != Step::Propose {
            return Err(Error::InvalidProposal("step is not propose"));
        }
        self.proposal = Some(proposal);
        self.step = Step::Prevote;
        Ok(())
    }

    /// Apply a `Vote` (prevote or precommit) to the active tallies.
    ///
    /// Returns `true` if the vote advanced the state machine into a
    /// new step (e.g. into commit, or into a new round). Returns an
    /// [`Error::InvalidVote`] for malformed / out-of-band votes.
    pub fn apply_vote(&mut self, vote: Vote) -> Result<bool> {
        if vote.height != self.height {
            return Err(Error::InvalidVote("wrong height"));
        }
        if vote.round != self.round {
            return Err(Error::InvalidVote("wrong round"));
        }

        let power = self.validators.voting_power_of(&vote.validator)?;

        let tallies = match vote.kind {
            VoteType::Prevote => &mut self.prevotes,
            VoteType::Precommit => &mut self.precommits,
        };

        let entry = tallies.entry(vote.block_hash).or_default();
        if !entry.add(vote.validator, power) {
            return Err(Error::InvalidVote("duplicate vote from validator"));
        }

        let quorum = self.validators.quorum();

        let mut advanced = false;
        match vote.kind {
            VoteType::Prevote => {
                if let Some(hash) = vote.block_hash {
                    if let Some(t) = self.prevotes.get(&Some(hash)) {
                        if t.power >= quorum && self.step == Step::Prevote {
                            self.locked_block = Some(hash);
                            self.locked_round = Some(self.round);
                            self.step = Step::Precommit;
                            advanced = true;
                        }
                    }
                }
            }
            VoteType::Precommit => {
                if let Some(hash) = vote.block_hash {
                    if let Some(t) = self.precommits.get(&Some(hash)) {
                        if t.power >= quorum {
                            self.step = Step::Commit;
                            self.committed = Some(hash);
                            advanced = true;
                        }
                    }
                }
            }
        }
        Ok(advanced)
    }

    /// Advance to the next round (because a timeout fired, or 2/3+1
    /// precommits for **nil** were observed without locking).
    ///
    /// Clears the round-scoped tallies but preserves locked-block
    /// state across rounds.
    pub fn next_round(&mut self) {
        self.round = self.round.saturating_add(1);
        self.step = Step::Propose;
        self.proposal = None;
        self.prevotes.clear();
        self.precommits.clear();
    }

    /// Voting power that has prevoted for a given block hash this round.
    pub fn prevote_power(&self, block: Option<Hash>) -> u64 {
        self.prevotes.get(&block).map(|t| t.power).unwrap_or(0)
    }

    /// Voting power that has precommitted for a given block hash this round.
    pub fn precommit_power(&self, block: Option<Hash>) -> u64 {
        self.precommits.get(&block).map(|t| t.power).unwrap_or(0)
    }

    /// Whether the state machine has reached the commit step.
    pub fn is_committed(&self) -> bool {
        matches!(self.step, Step::Commit)
    }
}
