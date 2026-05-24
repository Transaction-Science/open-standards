//! IBC light-client adapter sketch.
//!
//! The Inter-Blockchain Communication protocol (ICS-2 / ICS-7) wraps
//! a Tendermint light client on every connected chain. Each chain
//! stores `ClientState` (verification parameters) and rolling
//! `ConsensusState` snapshots (per height: root + validator hash +
//! timestamp). Packet relays bring an updated header; the receiving
//! chain runs `verify_skipping` against the stored client state to
//! advance the trusted state, then verifies a Merkle proof against
//! the new root.

use serde::{Deserialize, Serialize};

use crate::block::{Commit, Hash, Header};
use crate::error::Result;
use crate::light_client::{verify_skipping, TrustedState};
use crate::validator::ValidatorSet;

/// Parameters that govern light-client verification for a remote chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientState {
    /// Remote chain identifier (e.g. `"cosmoshub-4"`).
    pub chain_id: String,
    /// Numerator of the trust threshold (typically `1`).
    pub trust_level_num: u64,
    /// Denominator of the trust threshold (typically `3`).
    pub trust_level_den: u64,
    /// Duration of the trusting period in seconds.
    pub trusting_period_s: u64,
    /// Maximum permitted unbonding period in seconds.
    pub unbonding_period_s: u64,
    /// Latest verified height.
    pub latest_height: u64,
    /// `true` if the client has been frozen by misbehavior evidence.
    pub frozen: bool,
}

impl ClientState {
    /// Construct a default Tendermint client state.
    pub fn new(chain_id: impl Into<String>) -> Self {
        Self {
            chain_id: chain_id.into(),
            trust_level_num: 1,
            trust_level_den: 3,
            trusting_period_s: 14 * 24 * 3600,
            unbonding_period_s: 21 * 24 * 3600,
            latest_height: 0,
            frozen: false,
        }
    }
}

/// Per-height snapshot stored alongside the client state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsensusState {
    /// Timestamp at the verified height.
    pub time_ms: i64,
    /// App-state root, used to verify membership / non-membership proofs.
    pub root: Hash,
    /// Hash commitment of the next validator set.
    pub next_validators_hash: Hash,
}

/// IBC client wrapping a Tendermint light client.
#[derive(Debug, Clone)]
pub struct IbcLightClient {
    /// Client state for this counterparty chain.
    pub client_state: ClientState,
    /// Currently trusted light-client state.
    pub trusted: TrustedState,
}

impl IbcLightClient {
    /// Construct a new IBC light client.
    pub fn new(client_state: ClientState, trusted: TrustedState) -> Self {
        Self {
            client_state,
            trusted,
        }
    }

    /// Apply a `MsgUpdateClient` carrying a new header + commit.
    ///
    /// Uses the skipping path so the relayer can submit sparse headers.
    pub fn update(
        &mut self,
        new_header: Header,
        new_validators: ValidatorSet,
        commit: &Commit,
    ) -> Result<()> {
        verify_skipping(&self.trusted, &new_header, &new_validators, commit)?;
        self.client_state.latest_height = new_header.height;
        self.trusted.header = new_header;
        self.trusted.validators = new_validators;
        Ok(())
    }

    /// Freeze the client after submitted misbehavior is verified.
    pub fn freeze(&mut self) {
        self.client_state.frozen = true;
    }
}
