//! Error types for the Tendermint adapter.

use thiserror::Error;

/// Errors produced by Tendermint consensus, light-client, and evidence logic.
#[derive(Debug, Error)]
pub enum Error {
    /// A validator id was referenced but not present in the active set.
    #[error("unknown validator")]
    UnknownValidator,

    /// The voting-power total overflowed a `u64`.
    #[error("voting power overflow")]
    VotingPowerOverflow,

    /// A vote could not be applied (wrong height, wrong round, duplicate, etc.).
    #[error("invalid vote: {0}")]
    InvalidVote(&'static str),

    /// A proposal was rejected (wrong proposer, wrong round, etc.).
    #[error("invalid proposal: {0}")]
    InvalidProposal(&'static str),

    /// A block failed structural validation.
    #[error("invalid block: {0}")]
    InvalidBlock(&'static str),

    /// A light-client verification step failed.
    #[error("light client: {0}")]
    LightClient(&'static str),

    /// Evidence could not be validated.
    #[error("invalid evidence: {0}")]
    InvalidEvidence(&'static str),

    /// A signature could not be verified.
    #[error("signature verification failed")]
    BadSignature,

    /// An ABCI application returned an error.
    #[error("abci: {0}")]
    Abci(&'static str),
}

/// Convenience `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;
