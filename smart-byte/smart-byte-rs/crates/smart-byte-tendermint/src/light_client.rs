//! Tendermint light client.
//!
//! A light client maintains a "trusted" header at some height and
//! verifies later headers by one of two paths:
//!
//! * **Sequential** — verify that the next height's commit is signed
//!   by 2/3+1 of the trusted validator set, and the new header's
//!   `last_commit_hash` matches.
//! * **Skipping** — jump multiple heights forward if at least a
//!   1/3+1 *trust threshold* of the current trusted validator set
//!   has also signed the target commit. This is the mechanism used
//!   by IBC clients to bridge sparse update frequencies.

use crate::block::{Commit, Header};
use crate::error::{Error, Result};
use crate::validator::ValidatorSet;

/// Light-client trust state.
#[derive(Debug, Clone)]
pub struct TrustedState {
    /// The most recently trusted header.
    pub header: Header,
    /// Validator set the trusted header was signed under.
    pub validators: ValidatorSet,
}

impl TrustedState {
    /// Construct a fresh trusted state (e.g. from a genesis header).
    pub fn new(header: Header, validators: ValidatorSet) -> Self {
        Self { header, validators }
    }
}

/// Verify a sequential header update.
///
/// `commit` must commit `new_header.hash()` and the precommit signers
/// must hold 2/3+1 of `trusted.validators` voting power.
pub fn verify_sequential(
    trusted: &TrustedState,
    new_header: &Header,
    commit: &Commit,
) -> Result<()> {
    if new_header.height != trusted.header.height + 1 {
        return Err(Error::LightClient("not sequential"));
    }
    if new_header.last_block_hash != trusted.header.hash() {
        return Err(Error::LightClient("last_block_hash mismatch"));
    }
    if commit.height != new_header.height {
        return Err(Error::LightClient("commit height mismatch"));
    }
    if commit.block_hash != new_header.hash() {
        return Err(Error::LightClient("commit block_hash mismatch"));
    }
    let signers: Vec<_> = commit.signatures.iter().map(|s| &s.validator).collect();
    if !trusted.validators.has_quorum(signers) {
        return Err(Error::LightClient("no 2/3+1 quorum"));
    }
    Ok(())
}

/// Verify a skipping header update.
///
/// `commit` must commit `new_header.hash()`. The skip is permitted if
/// signers of `commit` hold at least 1/3+1 of `trusted.validators`
/// voting power *and* hold 2/3+1 of `new_validators` voting power.
pub fn verify_skipping(
    trusted: &TrustedState,
    new_header: &Header,
    new_validators: &ValidatorSet,
    commit: &Commit,
) -> Result<()> {
    if new_header.height <= trusted.header.height {
        return Err(Error::LightClient("not forward"));
    }
    if commit.height != new_header.height {
        return Err(Error::LightClient("commit height mismatch"));
    }
    if commit.block_hash != new_header.hash() {
        return Err(Error::LightClient("commit block_hash mismatch"));
    }
    let signers: Vec<_> = commit.signatures.iter().map(|s| &s.validator).collect();

    // 1/3+1 of TRUSTED set
    let trusted_power = trusted
        .validators
        .iter()
        .filter(|v| signers.iter().any(|s| **s == v.id))
        .map(|v| v.voting_power)
        .sum::<u64>();
    if trusted_power < trusted.validators.trust_threshold() {
        return Err(Error::LightClient("trust threshold not met"));
    }

    // 2/3+1 of NEW set
    if !new_validators.has_quorum(signers) {
        return Err(Error::LightClient("new validator quorum not met"));
    }
    Ok(())
}

/// Update a `TrustedState` after a verified sequential step.
pub fn apply_sequential(
    trusted: &mut TrustedState,
    new_header: Header,
    new_validators: ValidatorSet,
    commit: &Commit,
) -> Result<()> {
    verify_sequential(trusted, &new_header, commit)?;
    trusted.header = new_header;
    trusted.validators = new_validators;
    Ok(())
}

/// Update a `TrustedState` after a verified skipping step.
pub fn apply_skipping(
    trusted: &mut TrustedState,
    new_header: Header,
    new_validators: ValidatorSet,
    commit: &Commit,
) -> Result<()> {
    verify_skipping(trusted, &new_header, &new_validators, commit)?;
    trusted.header = new_header;
    trusted.validators = new_validators;
    Ok(())
}
