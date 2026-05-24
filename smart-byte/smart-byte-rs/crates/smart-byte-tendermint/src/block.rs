//! Block, header, and canonical commit format.
//!
//! Tendermint blocks are `(Header, Data, LastCommit)` triples. The
//! header binds the previous block hash, the validator-set hash, the
//! application-state hash, and the previous-commit hash, producing a
//! chain-of-trust that the light client follows.

use serde::{Deserialize, Serialize};

use crate::validator::ValidatorId;
use crate::vote::Signature;

/// A 32-byte Blake3 digest used as a block / header / commit hash.
pub type Hash = [u8; 32];

/// Tendermint block header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    /// Chain identifier (e.g. `"smart-byte-1"`).
    pub chain_id: String,
    /// Height of this block.
    pub height: u64,
    /// Unix-millis timestamp.
    pub time_ms: i64,
    /// Hash of the previous block, or zero for genesis.
    pub last_block_hash: Hash,
    /// Hash of the previous block's commit, or zero for genesis.
    pub last_commit_hash: Hash,
    /// Hash of the block's `Data` payload.
    pub data_hash: Hash,
    /// Hash of the validator set in effect for this height.
    pub validators_hash: Hash,
    /// Hash of the validator set in effect for the **next** height.
    pub next_validators_hash: Hash,
    /// Hash of the application state after this block.
    pub app_hash: Hash,
    /// Identifier of the proposer for this height/round.
    pub proposer: ValidatorId,
}

impl Header {
    /// Hash the header with Blake3 over its CBOR encoding.
    ///
    /// Real Tendermint uses Merkleized SHA-256 fields; we use a flat
    /// Blake3 over the serialized header for substrate-internal use.
    pub fn hash(&self) -> Hash {
        let bytes = serde_cbor_encode(self);
        *blake3::hash(&bytes).as_bytes()
    }
}

/// Opaque application transaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tx(#[serde(with = "serde_bytes")] pub Vec<u8>);

/// Block body — an ordered list of transactions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Data {
    /// Application transactions.
    pub txs: Vec<Tx>,
}

impl Data {
    /// Hash the data payload with Blake3.
    pub fn hash(&self) -> Hash {
        let bytes = serde_cbor_encode(self);
        *blake3::hash(&bytes).as_bytes()
    }
}

/// A single signed precommit recorded inside a `Commit`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitSig {
    /// Validator who signed.
    pub validator: ValidatorId,
    /// ed25519 signature over the canonical vote bytes.
    pub signature: Signature,
}

/// Aggregated precommits for a height/round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commit {
    /// Height being committed.
    pub height: u64,
    /// Round at which commit was reached.
    pub round: u32,
    /// Block id (header hash) being committed.
    pub block_hash: Hash,
    /// Signatures aggregated from the precommits.
    pub signatures: Vec<CommitSig>,
}

impl Commit {
    /// Hash the commit with Blake3.
    pub fn hash(&self) -> Hash {
        let bytes = serde_cbor_encode(self);
        *blake3::hash(&bytes).as_bytes()
    }
}

/// A complete Tendermint block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    /// Header.
    pub header: Header,
    /// Application data.
    pub data: Data,
    /// Commit from the **previous** height.
    pub last_commit: Commit,
}

impl Block {
    /// Hash of this block is the hash of its header.
    pub fn hash(&self) -> Hash {
        self.header.hash()
    }
}

/// Best-effort CBOR encode; falls back to an empty vector on failure.
///
/// We deliberately avoid `unwrap` outside `cfg(test)`; an empty hash
/// preimage will simply produce a deterministic "empty" digest which
/// downstream callers can still compare consistently.
fn serde_cbor_encode<T: Serialize>(value: &T) -> Vec<u8> {
    serde_cbor::to_vec(value).unwrap_or_default()
}
