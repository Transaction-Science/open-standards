//! Tendermint / CometBFT BFT consensus adapter for Smart Byte.
//!
//! Tendermint is the BFT consensus algorithm and gossip layer that
//! underpins the Cosmos Hub, the Inter-Blockchain Communication
//! protocol (IBC), and dozens of production application-specific
//! chains. It is described in Ethan Buchman's *The latest gossip on
//! BFT consensus* (2018) and standardized as CometBFT.
//!
//! This crate ingests the Tendermint specification into the Smart
//! Byte substrate so other crates can model BFT consensus, light
//! clients, evidence handling, and IBC light-client updates without
//! taking a runtime dependency on a specific CometBFT implementation.
//!
//! ## Layout
//!
//! * [`consensus`] — round state machine (`Propose → Prevote →
//!   Precommit → Commit`) with voting-power tallies.
//! * [`validator`] — `ValidatorSet`, voting power, quorum +
//!   trust-threshold math.
//! * [`vote`] — `Prevote` / `Precommit` and ed25519 signature wire format.
//! * [`block`] — `Block`, `Header`, `Commit`, `Data`.
//! * [`proposal`] — round proposals.
//! * [`light_client`] — sequential + skipping verification.
//! * [`abci`] — `check_tx` / `deliver_tx` / `commit` application trait.
//! * [`evidence`] — `DuplicateVoteEvidence` + `LightClientAttackEvidence`.
//! * [`ibc`] — IBC light-client adapter sketch.
//! * [`error`] — error and result types.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod abci;
pub mod block;
pub mod consensus;
pub mod error;
pub mod evidence;
pub mod ibc;
pub mod light_client;
pub mod proposal;
pub mod validator;
pub mod vote;

pub use error::{Error, Result};
