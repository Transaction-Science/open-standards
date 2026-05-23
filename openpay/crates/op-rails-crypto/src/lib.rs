//! # `op-rails-crypto` — stablecoin settlement rail
//!
//! `OpenPay`'s crypto rail. The whole point of including this in the
//! reference stack is the fee profile: a card auth costs the
//! merchant **100–300 basis points**; a USDC transfer on Base or
//! Solana costs **single-cents or fractions of a cent**. For a
//! vendor doing $1M/month, the difference is $30k vs $5.
//!
//! ## Scope
//!
//! This crate is rail-agnostic at the chain level: drivers
//! implement [`CryptoGateway`] and the orchestrator routes
//! generically. The reference stack does NOT ship chain SDKs
//! (`solana-sdk`, `ethers-rs`, etc.) — those are operator
//! choice. Three reasons:
//!
//! 1. **Footprint.** `ethers-rs` alone is ~250 crates.
//! 2. **Signing security.** Operators sign with HSMs / KMS /
//!    multisig / Fireblocks. Baking in a software keystore would
//!    be a footgun.
//! 3. **Chain evolution.** New chains and L2s appear quarterly;
//!    the reference stack stays neutral by exposing only the
//!    trait surface.
//!
//! Operators wire their preferred client (a `RpcClient` for
//! Solana, an `EthersProvider` for EVM, Fireblocks for custody)
//! behind a [`CryptoGateway`] impl.
//!
//! ## Domain model
//!
//! ```text
//!   Token  = (chain, contract, decimals)        ← USDC@Base etc.
//!   Wallet = chain-specific address string
//!   Transfer request → Gateway.submit_transfer(...) → Decision
//!   Decision.status ∈ { Pending Confirmations Finalized Rejected Transient }
//! ```
//!
//! Confirmations are chain-specific (1 on Solana, 12 on Ethereum
//! mainnet, varies on L2s). The driver decides what counts as
//! "finalized" and reports `Status::Finalized` only when the chain
//! has reached operator-acceptable depth.
//!
//! ## What this crate does NOT do
//!
//! - **No transaction signing.** Drivers receive an already-formed
//!   transfer intent and broadcast it via the operator's signer.
//! - **No price feeds.** Stablecoin = 1:1 by definition. Operators
//!   wanting non-stablecoin support compose with a separate FX layer.
//! - **No bridging.** Cross-chain transfers are out of scope —
//!   operators handle multi-chain by registering one driver per
//!   `(chain, token)` pair and routing on the customer's wallet
//!   address chain.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod gateway;
pub mod token;

#[cfg(feature = "evm")]
pub mod evm;
#[cfg(feature = "evm")]
pub mod local_signer;
#[cfg(feature = "evm")]
pub mod signer;

pub use error::{Error, Result};
pub use gateway::{CryptoDecision, CryptoGateway, CryptoStatus, CryptoTransferReq, StatusQueryReq};
pub use token::{StableToken, TokenRef};

#[cfg(feature = "evm")]
pub use evm::{EvmJsonRpcGateway, encode_erc20_transfer, erc20_transfer_selector};
#[cfg(feature = "evm")]
pub use local_signer::LocalKeyEvmSigner;
#[cfg(feature = "evm")]
pub use signer::{EvmSigner, TxHash, UnsignedTx};
