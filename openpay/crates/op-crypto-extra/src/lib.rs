//! # `op-crypto-extra` — extended crypto rails
//!
//! `op-rails-crypto` is the minimum-viable production stablecoin
//! rail: USDC / EURC / PYUSD on a handful of chains, ERC-20
//! `transfer`, JSON-RPC broadcast, hot-wallet signer. This crate
//! covers the *rest* of the modern crypto-payments surface:
//!
//! - **Stablecoin catalog.** A canonical [`stable::Stablecoin`] enum
//!   covering USDC, USDT, PYUSD, DAI, RLUSD, EURC, FDUSD and their
//!   well-known deployments across Ethereum L1, the dominant
//!   L2 stack (Optimism, Arbitrum, Base, zkSync Era, Polygon zkEVM,
//!   Linea, Scroll, Starknet), Polygon PoS, and Solana.
//! - **L2 topology.** [`chains`] enumerates each L2 with its
//!   EIP-155 chain id, parent settlement layer, finality model, and
//!   a default public RPC. Used by operators that want to register
//!   one EVM-rail gateway per chain without hard-coding the list.
//! - **Account abstraction.** [`erc4337`] models the ERC-4337
//!   `UserOperation` (v0.7 packed layout) and EntryPoint hand-off;
//!   [`eip7702`] models the EIP-7702 "set-code" authorization that
//!   lets an EOA temporarily execute as a smart account.
//! - **Permits.** [`permit2`] encodes Uniswap's Permit2 transfer
//!   intents (single + batch + `PermitWitnessTransferFrom`).
//!   [`eip2612`] encodes the older ERC-2612 `permit(...)` typed
//!   data for token approvals without on-chain `approve`.
//! - **Cross-chain.** [`cctp`] models Circle's CCTP v2 burn / mint
//!   message (`depositForBurn` + Attestation + `receiveMessage`)
//!   for native cross-chain USDC.
//! - **Lightning.** [`lightning`] parses BOLT-11 invoices (bech32
//!   sanity-check + tagged-field walk) and validates LNURL
//!   bech32-encoded URLs.
//! - **Bitcoin.** [`psbt`] is a thin PSBT v2 sketch (BIP-370
//!   field layout — global / input / output map keys) suited for
//!   round-tripping unsigned Bitcoin transactions through a
//!   third-party signer.
//! - **Atomic swap.** [`htlc`] models the hash-time-lock contract
//!   primitive used in cross-chain swaps (single-secret HTLC with
//!   chain-agnostic field layout).
//! - **Confidential transfer.** [`confidential`] sketches the
//!   structural surface of confidential-token transfers (Solana
//!   confidential token / Aleo / Aztec) — the public field, the
//!   ciphertext envelope, and the proof envelope — without
//!   committing to a specific zk system.
//!
//! ## Scope
//!
//! Like `op-rails-crypto`, this crate is **rail-agnostic at the
//! chain level**. It does not bundle any chain SDK or signer. The
//! types model the *wire layout* of each primitive precisely enough
//! that an operator's signer / broadcaster can produce real
//! transactions; the broadcast itself happens elsewhere (operator's
//! KMS, Fireblocks, Lightning daemon, Bitcoin Core, etc.).
//!
//! ## What this crate does NOT do
//!
//! - **No signing.** Every primitive ends in a "build me the bytes
//!   to be signed" boundary.
//! - **No bridging.** CCTP message construction is included
//!   because it's a Circle-issued message format; live attestation
//!   fetching against Circle's API is the operator's job.
//! - **No price feeds.** Stablecoins are 1:1 by design.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod cctp;
pub mod chains;
pub mod confidential;
pub mod eip2612;
pub mod eip7702;
pub mod erc4337;
pub mod error;
pub mod htlc;
pub mod lightning;
pub mod permit2;
pub mod psbt;
pub mod stable;

pub use cctp::{CctpAttestation, CctpBurnMessage, CctpDomain, CctpMintReceipt};
pub use chains::{ChainId, ChainInfo, ChainKind, FinalityModel, L2Catalog, SettlementLayer};
pub use confidential::{ConfidentialSystem, ConfidentialTransfer, ProofEnvelope};
pub use eip2612::{Eip2612Permit, eip2612_digest};
pub use eip7702::{Eip7702Authorization, AuthorizationList};
pub use erc4337::{EntryPointVersion, PackedUserOperation, UserOperation, pack_user_op};
pub use error::{Error, Result};
pub use htlc::{HtlcContract, HtlcState, HtlcPreimage};
pub use lightning::{Bolt11Invoice, Bolt11Network, LnurlKind, lnurl_decode, parse_bolt11};
pub use permit2::{
    Permit2BatchTransfer, Permit2SingleTransfer, Permit2TokenPermissions, Permit2Witness,
};
pub use psbt::{PsbtV2, PsbtV2Global, PsbtV2Input, PsbtV2Output};
pub use stable::{ChainAddress, Stablecoin, StablecoinDeployment};
