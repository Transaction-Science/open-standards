//! The [`NetworkTokenProvider`] trait.
//!
//! Every card-network tokenization service (VTS, MDES, Amex, Discover)
//! implements this trait. The orchestrator holds a
//! `Box<dyn NetworkTokenProvider>` per network and routes provisioning
//! by [`op_core::CardNetwork`].
//!
//! ## Why three methods and not more
//!
//! Tokenization services are large APIs, but `OpenPay` only needs to
//! cover the operations that gate liability shift and PCI scope
//! reduction:
//!
//! 1. **`provision`** — turn a vaulted PAN reference into a network
//!    token. This is the boundary crossing from "vaulted credential"
//!    to "tokenized credential".
//! 2. **`lifecycle`** — listen for credential state changes (issuer
//!    reissue, suspend, delete). The orchestrator needs these to keep
//!    its routing table fresh.
//! 3. **`fetch_cryptogram`** — get a fresh per-transaction cryptogram
//!    bound to (token_ref, amount). Without this, the acquirer can't
//!    claim the liability shift.
//!
//! Token search / lookup / batch operations live in adapter-specific
//! extension traits and are not part of the conformance contract.

use op_core::{CardNetwork, Money, NetworkToken, NetworkTokenLifecycleEvent, VaultRef};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::Result;
use crate::network_token::cryptogram::Cryptogram;

/// A lifecycle event with provenance — what changed, on which token,
/// when, and (when supplied by the network) why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleEvent {
    /// Which token changed state.
    pub token_ref: String,
    /// Normalized event kind.
    pub event: NetworkTokenLifecycleEvent,
    /// When the network emitted the event.
    pub at: OffsetDateTime,
    /// Network-supplied reason code (free-form). Surfaced verbatim
    /// for telemetry and webhook fanout.
    pub reason: Option<String>,
}

/// A synchronous lifecycle stream. Adapters return an iterator the
/// caller can drain. Real adapters back this by a long-poll or
/// webhook consumer; the stub returns a finite canned sequence.
///
/// We use `Box<dyn Iterator>` rather than `futures::Stream` because
/// the rest of the workspace is sync (see the note in
/// `acquirer.rs`). When async-in-trait stabilizes for the dyn path
/// we will migrate.
pub type LifecycleStream = Box<dyn Iterator<Item = LifecycleEvent> + Send>;

/// Provider configuration — shared shape across adapters.
///
/// Adapters that need additional fields (API URL, mTLS bundle, token
/// requestor ID) extend this via composition in their own
/// `XyzProvider::new(ProviderConfig, XyzExtras)` constructor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Which network this configuration targets. Adapters validate
    /// that the requestor's enrolled network matches this field.
    pub network: CardNetwork,
    /// Network-issued token requestor identifier. Opaque string.
    pub token_requestor_id: String,
    /// True if cryptograms should be requested with each provision
    /// call (wallet flow). False for card-on-file flows where the
    /// merchant fetches cryptograms on demand.
    pub eager_cryptograms: bool,
}

impl ProviderConfig {
    /// Construct with `eager_cryptograms` defaulted to `false` — the
    /// safer card-on-file default.
    #[must_use]
    pub fn new(network: CardNetwork, token_requestor_id: impl Into<String>) -> Self {
        Self {
            network,
            token_requestor_id: token_requestor_id.into(),
            eager_cryptograms: false,
        }
    }
}

/// The card-network tokenization provider trait.
///
/// Implementations must be `Send + Sync` so the orchestrator can hold
/// a `Box<dyn NetworkTokenProvider>` across worker threads.
pub trait NetworkTokenProvider: Send + Sync {
    /// Driver name (e.g. `"vts"`, `"mdes"`).
    fn name(&self) -> &'static str;

    /// Which network this provider issues tokens on.
    fn network(&self) -> CardNetwork;

    /// Provision a network token from a vault-resident PAN reference.
    ///
    /// On success, the returned [`NetworkToken`] is the handle that
    /// downstream rails carry instead of the PAN.
    fn provision(&self, card_ref: &VaultRef) -> Result<NetworkToken>;

    /// Subscribe to the lifecycle event stream for a given token.
    ///
    /// Real adapters back this with a webhook consumer or long-poll.
    /// Stubs return a finite canned sequence so the conformance
    /// harness can drain and assert.
    fn lifecycle(&self, token_ref: &str) -> Result<LifecycleStream>;

    /// Fetch a per-transaction cryptogram bound to `(token_ref,
    /// amount)`.
    ///
    /// Stub implementations return a deterministic value: the same
    /// `(token_ref, amount)` pair must produce the same `Cryptogram`
    /// so that the conformance harness can verify reproducibility.
    fn fetch_cryptogram(&self, token_ref: &str, amount: Money) -> Result<Cryptogram>;
}
