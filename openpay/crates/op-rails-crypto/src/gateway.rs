//! The [`CryptoGateway`] trait and its request / response types.
//!
//! A gateway is the operator's bridge between `OpenPay`'s abstract
//! "send N tokens to this address" request and a concrete chain
//! client (Solana `RpcClient`, EVM `Provider`, Fireblocks API, ...).
//! Drivers implement [`CryptoGateway`]; the orchestrator routes via
//! the wrapping `CryptoAdapter` in `op-orchestrator`.

use op_core::{CryptoAddress, Money};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::token::TokenRef;

/// Request to send a stablecoin transfer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CryptoTransferReq {
    /// Token to send (chain + contract + decimals + symbol).
    pub token: TokenRef,
    /// Destination address. Must be on `token.chain`.
    pub to: CryptoAddress,
    /// Amount in the token's smallest unit (e.g. `1_000_000` =
    /// `1.000000 USDC` since USDC has 6 decimals). Carried in a
    /// `Money` so accounting downstream sees a single currency
    /// system (with `token.symbol` as the currency code).
    pub amount: Money,
    /// Operator-supplied idempotency key. Drivers forward to their
    /// chain client where supported (Solana's `recent_blockhash` +
    /// signed message dedup is implicit; EVM drivers track keys in
    /// their own DB).
    pub idempotency_key: String,
    /// Operator-supplied memo / reference. Solana accepts the
    /// memo program; EVM has no standard memo (drivers may emit a
    /// `transfer + log` or skip).
    pub memo: Option<String>,
}

/// Request to query the status of a previously broadcast transfer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusQueryReq {
    /// Chain identifier (`"solana"`, `"base"`, ...). Drivers ignore
    /// queries on chains they don't service (return
    /// [`crate::Error::UnsupportedChain`]).
    pub chain: String,
    /// Chain transaction hash / signature.
    pub tx_hash: String,
    /// Operator-side reference — drivers may use it to look up
    /// state in their own DB if the chain query is slow.
    pub idempotency_key: Option<String>,
}

/// What the gateway returns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CryptoDecision {
    /// Normalized status.
    pub status: CryptoStatus,
    /// Chain transaction hash / signature (Solana: 88-char base58
    /// signature; EVM: `0x` + 64-char hex). `None` only on
    /// transient failures *before* broadcast.
    pub tx_hash: Option<String>,
    /// Confirmation depth at decision time. For chains with
    /// instant finality (Solana's "finalized" commitment) this is
    /// 1; for probabilistic-finality chains the driver reports the
    /// observed depth.
    pub confirmations: u32,
    /// Settled amount, if the gateway could read it back from the
    /// receipt. Usually equal to requested.
    pub settled_amount: Option<Money>,
    /// Raw chain-side status code, preserved for diagnostics.
    pub raw_status: Option<String>,
    /// Reason text for rejections.
    pub reason: Option<String>,
}

/// Normalized crypto-transfer status. Tracks the chain-lifecycle
/// stages from broadcast to operator-acceptable finality.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CryptoStatus {
    /// Submitted but not yet observed in a block.
    Pending,
    /// In a block but below operator confirmation threshold.
    Confirming,
    /// Confirmation depth meets the operator's finality threshold.
    /// Funds are settled.
    Finalized,
    /// Chain definitively rejected (revert, simulation failure,
    /// invalid signature). Funds did not move.
    Rejected,
    /// Transient network / RPC issue. The transfer may or may not
    /// have been broadcast; caller queries status with the same
    /// idempotency key to find out.
    Transient,
}

impl CryptoStatus {
    /// True iff settlement is confirmed.
    #[must_use]
    pub const fn funds_moved(self) -> bool {
        matches!(self, Self::Finalized)
    }

    /// True iff a retry of the same request is safe.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::Transient)
    }

    /// True iff the chain definitively refused.
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::Rejected)
    }

    /// True iff the caller should poll status.
    #[must_use]
    pub const fn needs_polling(self) -> bool {
        matches!(self, Self::Pending | Self::Confirming)
    }
}

/// The crypto-rail gateway interface. Each driver implements this
/// for a specific `(chain, token)` deployment.
pub trait CryptoGateway: Send + Sync {
    /// Driver name (`"usdc-solana"`, `"usdc-base"`, ...). The
    /// orchestrator's `PolicyRouter` keys off this.
    fn name(&self) -> &'static str;

    /// The chain this gateway services.
    fn chain(&self) -> &str;

    /// The token this gateway services.
    fn token(&self) -> &TokenRef;

    /// True iff this gateway can settle to `to`. A correct impl
    /// returns `false` for any address whose chain doesn't match
    /// the gateway's [`Self::chain`].
    fn supports(&self, to: &CryptoAddress) -> bool {
        to.chain == self.chain()
    }

    /// Broadcast a transfer.
    ///
    /// # Errors
    /// See [`crate::Error`].
    fn submit_transfer(&self, req: &CryptoTransferReq) -> Result<CryptoDecision>;

    /// Query the status of a previously broadcast transfer.
    ///
    /// # Errors
    /// See [`crate::Error`].
    fn query_status(&self, req: &StatusQueryReq) -> Result<CryptoDecision>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_predicates_disjoint() {
        for s in [
            CryptoStatus::Pending,
            CryptoStatus::Confirming,
            CryptoStatus::Finalized,
            CryptoStatus::Rejected,
            CryptoStatus::Transient,
        ] {
            assert!(!(s.funds_moved() && s.is_failure()));
            assert!(!(s.funds_moved() && s.is_retryable()));
            assert!(!(s.is_failure() && s.is_retryable()));
        }
    }

    #[test]
    fn only_finalized_moves_funds() {
        assert!(CryptoStatus::Finalized.funds_moved());
        for s in [
            CryptoStatus::Pending,
            CryptoStatus::Confirming,
            CryptoStatus::Rejected,
            CryptoStatus::Transient,
        ] {
            assert!(!s.funds_moved());
        }
    }

    #[test]
    fn only_transient_is_retryable() {
        assert!(CryptoStatus::Transient.is_retryable());
        for s in [
            CryptoStatus::Pending,
            CryptoStatus::Confirming,
            CryptoStatus::Finalized,
            CryptoStatus::Rejected,
        ] {
            assert!(!s.is_retryable());
        }
    }

    #[test]
    fn polling_only_for_pending_states() {
        assert!(CryptoStatus::Pending.needs_polling());
        assert!(CryptoStatus::Confirming.needs_polling());
        for s in [
            CryptoStatus::Finalized,
            CryptoStatus::Rejected,
            CryptoStatus::Transient,
        ] {
            assert!(!s.needs_polling());
        }
    }
}
