//! Operator-provided EVM transaction signer abstraction.
//!
//! The reference stack deliberately does NOT bundle key management —
//! see the crate-level docs for the rationale. Operators wire their
//! own Fireblocks / AWS KMS / HSM / hot-wallet signer behind the
//! [`EvmSigner`] trait. The gateway builds an [`UnsignedTx`] (nonce,
//! gas, calldata, to, value, chain id) and hands it off; the signer
//! is responsible for ECDSA signing, RLP encoding, and calling
//! `eth_sendRawTransaction` against whatever endpoint it uses.
//!
//! This split keeps the rail-level code chain-pure: the gateway
//! knows nothing about private keys, and the signer knows nothing
//! about idempotency keys or token references.

use crate::error::Result;

/// 32-byte EVM transaction hash, hex-encoded with a `0x` prefix.
///
/// The driver returns this verbatim in [`crate::CryptoDecision::tx_hash`].
pub type TxHash = String;

/// All the chain-side fields the gateway can populate without
/// access to the operator's private key.
///
/// The signer is expected to:
/// 1. Choose a transaction type (legacy / EIP-1559 / EIP-2930). The
///    reference gateway populates the legacy `gas_price` field; a
///    1559-only signer can read `gas_price` as a hint for the
///    max-fee-per-gas it wants to set.
/// 2. RLP-encode + ECDSA-sign the resulting transaction.
/// 3. Broadcast via `eth_sendRawTransaction` and return the chain's
///    transaction hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsignedTx {
    /// Chain id (EIP-155). 1 = Ethereum mainnet, 137 = Polygon,
    /// 8453 = Base, 42161 = Arbitrum One.
    pub chain_id: u64,
    /// Sender's next-available nonce, fetched from
    /// `eth_getTransactionCount(from, "pending")`.
    pub nonce: u64,
    /// Gas price in wei (legacy transactions). For EIP-1559 signers,
    /// treat this as the maximum the operator is willing to pay.
    pub gas_price: u128,
    /// Gas limit. The gateway populates this via `eth_estimateGas`
    /// plus a small safety margin.
    pub gas_limit: u64,
    /// Recipient — for an ERC-20 transfer, this is the token
    /// contract, NOT the end recipient (the recipient is encoded
    /// in `data`). Lowercase hex `0x` prefix, 42 chars total.
    pub to: String,
    /// Native-token value in wei. Always `0` for an ERC-20 transfer
    /// (the value moves inside the contract's storage, not the
    /// EVM-level value field).
    pub value: u128,
    /// Pre-encoded calldata. For an ERC-20 transfer this is exactly
    /// 68 bytes: 4-byte selector + 32-byte recipient + 32-byte
    /// amount, hex-encoded with `0x` prefix.
    pub data: String,
    /// Operator-supplied "from" address. Some signers (Fireblocks,
    /// MPC) bind the signing key to a known address; passing it
    /// through lets them assert nonce / balance invariants without
    /// a second RPC call. Lowercase hex `0x` prefix.
    pub from: String,
}

/// Operator-supplied signing + broadcast trait.
///
/// Implementations are expected to be thread-safe (`Send + Sync`)
/// because the gateway is used from synchronous orchestrator code
/// that may be called from any thread of the operator's runtime.
pub trait EvmSigner: Send + Sync {
    /// Sign the supplied unsigned transaction and broadcast it.
    ///
    /// Implementations should:
    /// 1. RLP-encode the unsigned transaction.
    /// 2. Sign with ECDSA over secp256k1, EIP-155 chain-id-aware.
    /// 3. Submit via `eth_sendRawTransaction` and return the chain's
    ///    transaction hash (`0x` + 64 hex chars).
    ///
    /// # Errors
    /// Return [`crate::Error::Transport`] for network failures,
    /// [`crate::Error::Rejected`] when the chain refuses (revert,
    /// invalid signature, insufficient funds).
    fn sign_and_broadcast(&self, unsigned: UnsignedTx) -> Result<TxHash>;
}
