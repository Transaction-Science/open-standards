//! Application Blockchain Interface sketch.
//!
//! ABCI lets a Tendermint consensus engine drive an arbitrary
//! application. The full protocol has dozens of methods; we model the
//! load-bearing trio:
//!
//! * [`Application::check_tx`] — mempool admission check.
//! * [`Application::deliver_tx`] — state transition for a tx inside a block.
//! * [`Application::commit`] — finalize the app-state hash.

use serde::{Deserialize, Serialize};

use crate::block::Hash;
use crate::error::Result;

/// Result of `check_tx` — used to admit/reject txs into the mempool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckTxResponse {
    /// Application-level result code (`0` = ok).
    pub code: u32,
    /// Optional human-readable log line.
    pub log: String,
    /// Gas consumed checking this tx.
    pub gas_used: u64,
}

/// Result of `deliver_tx` — applied to state during block execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliverTxResponse {
    /// Application-level result code (`0` = ok).
    pub code: u32,
    /// Application data (e.g. CBOR-encoded receipts).
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    /// Optional human-readable log line.
    pub log: String,
    /// Gas consumed delivering this tx.
    pub gas_used: u64,
}

/// Result of `commit` — finalizes block N and surfaces the app-state hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    /// The new application-state root hash to embed in the next header.
    pub app_hash: Hash,
}

/// ABCI application trait.
///
/// Implement this for any state machine you want Tendermint to drive.
pub trait Application {
    /// Check a tx for mempool admission.
    fn check_tx(&mut self, tx: &[u8]) -> Result<CheckTxResponse>;
    /// Deliver a tx during block execution.
    fn deliver_tx(&mut self, tx: &[u8]) -> Result<DeliverTxResponse>;
    /// Commit accumulated state and return the new app-state hash.
    fn commit(&mut self) -> Result<CommitResponse>;
}

/// Minimal in-memory application used in tests: hashes the running
/// concatenation of all delivered transactions.
#[derive(Debug, Default)]
pub struct EchoApp {
    state: Vec<u8>,
}

impl Application for EchoApp {
    fn check_tx(&mut self, _tx: &[u8]) -> Result<CheckTxResponse> {
        Ok(CheckTxResponse {
            code: 0,
            log: String::new(),
            gas_used: 1,
        })
    }

    fn deliver_tx(&mut self, tx: &[u8]) -> Result<DeliverTxResponse> {
        self.state.extend_from_slice(tx);
        Ok(DeliverTxResponse {
            code: 0,
            data: tx.to_vec(),
            log: String::new(),
            gas_used: 1,
        })
    }

    fn commit(&mut self) -> Result<CommitResponse> {
        Ok(CommitResponse {
            app_hash: *blake3::hash(&self.state).as_bytes(),
        })
    }
}
