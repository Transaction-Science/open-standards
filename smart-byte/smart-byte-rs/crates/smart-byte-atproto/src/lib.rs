//! AT Protocol (Bluesky federation) adapter for Smart Byte.
//!
//! This crate ingests the deployed footprint of the AT Protocol — the
//! federation layer behind Bluesky — into the Smart Byte substrate. It is
//! a *pragmatic* adapter: enough of the protocol to talk to a Personal
//! Data Server (PDS), read and write IPLD CAR-encoded repositories, walk
//! Merkle Search Trees (MSTs), sign repo commits, and bridge AT URIs to
//! Smart Byte [`Aid`][bridge::Aid] / [`Said`][smart_byte_core::Said]
//! identifiers.
//!
//! ## What's covered
//!
//! * **Identity** — [`did_resolver`] delegates DID resolution to
//!   [`smart_byte_did`]. `did:web` is fully resolved; `did:plc` resolution
//!   is delegated to the operator-configured PLC directory.
//! * **Lexicon** — [`lexicon`] models the typed-JSON-RPC surface of AT
//!   Protocol: records with `$type` discriminators, XRPC queries and
//!   procedures, and a curated subset of the well-known schemas
//!   (`com.atproto.{server,sync,repo}` and `app.bsky.feed.{post,like}`).
//! * **CAR + IPLD** — [`car`] reads and writes AT-style CARv1 files
//!   carrying DAG-CBOR blocks with CIDv1/SHA-256.
//! * **MST** — [`mst`] sketches an in-memory Merkle Search Tree with
//!   insert / delete / lookup and a stable root hash.
//! * **Repo + commits** — [`repo`] assembles a signed repo commit
//!   (`did`, `version`, `data`, `prev`, `sig`) using an Ed25519 key.
//! * **PDS client** — [`pds_client`] is a thin XRPC client suitable for
//!   `com.atproto.server.createSession` and friends.
//! * **Firehose** — [`firehose`] consumes `com.atproto.sync.subscribeRepos`
//!   events from a mockable transport.
//! * **Bridge** — [`bridge`] maps `at://did:plc:.../collection/rkey`
//!   to Smart Byte [`Aid`][bridge::Aid] and back.
//!
//! ## What's intentionally scoped out
//!
//! Full WebSocket firehose plumbing, full Lexicon catalogue, JetStream
//! relay support and on-chain `did:plc` operation auditing are not in
//! this crate. The surfaces are designed so an operator can layer them
//! in without re-shaping the public types.

#![forbid(unsafe_code)]

pub mod bridge;
pub mod car;
pub mod did_resolver;
pub mod error;
pub mod firehose;
pub mod lexicon;
pub mod mst;
pub mod pds_client;
pub mod repo;

pub use bridge::{Aid, AtUri};
pub use car::{CarBlock, CarFile, Cid};
pub use did_resolver::{AtprotoResolver, PlcResolver};
pub use error::AtprotoError;
pub use firehose::{FirehoseClient, FirehoseEvent};
pub use lexicon::{LexiconRecord, XrpcRequest, XrpcResponse};
pub use mst::{Mst, MstEntry};
pub use pds_client::PdsClient;
pub use repo::{Repo, SignedCommit};
