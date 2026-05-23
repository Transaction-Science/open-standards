//! Application state.
//!
//! Bundles every store/handle the HTTP handlers need into one
//! `Clone`-able value (cheap — everything is `Arc`-wrapped). Axum
//! pulls this via `State<AppState>` extractors.
//!
//! ## Persistence model
//!
//! `AppState` is rooted on a single [`GraphHandle`]. **All** stores
//! — refund, dispute, settlement, ledger, reconciliation,
//! webhooks — write to that one graph. Pointing the handle at a
//! `.graph` file via [`AppState::with_graph_path`] persists the
//! entire deployment to a single embedded database. No separate
//! server, no network DB, no schema migrations: one file, full
//! bi-temporal history, time-travel queries via
//! `op_ledger::LedgerHistory`.

use std::path::Path;
use std::sync::Arc;

use op_fx::{QuoteProvider, StaticQuoteProvider};
use op_graph::{
    GraphDisputeStore, GraphHandle, GraphIdempotencyStore, GraphLedgerStore, GraphRailTelemetry,
    GraphReconciliationStore, GraphRefundStore, GraphSettlementStore, GraphSubscriptionStore,
    GraphWebhookStore,
};
use op_orchestrator::Orchestrator;
use op_webhook::{
    EventEmitter, ExponentialBackoffPolicy, NoOpEmitter, WebhookDispatcher, WebhookEmitter,
};

/// Bundle of stores + orchestrator + graph handle shared across
/// HTTP handlers. Cloning is `Arc`-cheap.
#[derive(Clone)]
pub struct AppState {
    /// Shared payment orchestrator. Fully configured before being
    /// dropped into state (adapters registered, scorer wired, ...).
    pub orchestrator: Arc<Orchestrator>,
    /// Refund domain store, graph-backed.
    pub refunds: Arc<GraphRefundStore>,
    /// Dispute domain store, graph-backed.
    pub disputes: Arc<GraphDisputeStore>,
    /// Settlement batch store, graph-backed.
    pub settlement: Arc<GraphSettlementStore>,
    /// Ledger store rooted in the shared graph.
    pub ledger: Arc<GraphLedgerStore>,
    /// Reconciliation store rooted in the shared graph.
    pub reconciliation: Arc<GraphReconciliationStore>,
    /// Rail-attempt telemetry.
    pub telemetry: Arc<GraphRailTelemetry>,
    /// Idempotency cache, graph-backed (key → cached outcome).
    pub idempotency: Arc<GraphIdempotencyStore>,
    /// Subscriptions store, graph-backed.
    pub subscriptions: Arc<GraphSubscriptionStore>,
    /// Webhook persistence (events, endpoints, attempts) on the
    /// same `.graph` file as everything else.
    pub webhooks: Arc<GraphWebhookStore>,
    /// Domain-event emitter. Defaults to [`NoOpEmitter`]; replace
    /// via [`AppState::with_event_emitter`] to wire real outbound
    /// webhook delivery.
    pub events: Arc<dyn EventEmitter>,
    /// FX quote provider. Defaults to an empty
    /// [`StaticQuoteProvider`]; replace via
    /// [`AppState::with_fx_provider`] to install a live feed.
    pub fx: Arc<dyn QuoteProvider>,
    /// The shared graph handle. Audit report queries it directly.
    pub graph: GraphHandle,
}

impl AppState {
    /// Build state on a fresh in-memory graph. Suitable for tests
    /// and demos — drops state on restart.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::from_handle(GraphHandle::new_in_memory())
    }

    /// Build state on a single-file `.graph` database at `path`.
    /// Opens existing or creates new. All stores share this one
    /// substrate; reopening the same path recovers every stored
    /// fact.
    ///
    /// # Errors
    /// Bubbles up `op_graph::Error::Backend` if Minigraf can't open
    /// the path.
    pub fn with_graph_path(path: impl AsRef<Path>) -> op_graph::Result<Self> {
        let handle = GraphHandle::new_persistent(path)?;
        Ok(Self::from_handle(handle))
    }

    /// Build state from an existing graph handle. Useful when an
    /// embedder needs to share the same Minigraf across multiple
    /// `AppState` mounts (multi-tenant scenarios), or wants to
    /// preconfigure the orchestrator separately.
    #[must_use]
    pub fn from_handle(graph: GraphHandle) -> Self {
        let telemetry = Arc::new(GraphRailTelemetry::with_handle(graph.clone()));
        let ledger = Arc::new(GraphLedgerStore::with_handle(graph.clone()));
        let reconciliation = Arc::new(GraphReconciliationStore::with_handle(graph.clone()));
        let refunds = Arc::new(GraphRefundStore::with_handle(graph.clone()));
        let disputes = Arc::new(GraphDisputeStore::with_handle(graph.clone()));
        let settlement = Arc::new(GraphSettlementStore::with_handle(graph.clone()));
        let idempotency = Arc::new(GraphIdempotencyStore::with_handle(graph.clone()));
        let subscriptions = Arc::new(GraphSubscriptionStore::with_handle(graph.clone()));
        let webhooks = Arc::new(GraphWebhookStore::with_handle(graph.clone()));
        Self {
            orchestrator: Arc::new(Orchestrator::new()),
            refunds,
            disputes,
            settlement,
            ledger,
            reconciliation,
            telemetry,
            idempotency,
            subscriptions,
            webhooks,
            events: Arc::new(NoOpEmitter::new()),
            fx: Arc::new(StaticQuoteProvider::new()),
            graph,
        }
    }

    /// Builder: install a fully-configured orchestrator (with
    /// adapters registered, fraud scorer wired, etc.).
    #[must_use]
    pub fn with_orchestrator(mut self, orchestrator: Orchestrator) -> Self {
        self.orchestrator = Arc::new(orchestrator);
        self
    }

    /// Builder: install a custom [`EventEmitter`]. Most operators
    /// will pass a [`WebhookEmitter`] wrapping a
    /// [`WebhookDispatcher`] constructed over this `AppState`'s
    /// `webhooks` store.
    #[must_use]
    pub fn with_event_emitter(mut self, emitter: Arc<dyn EventEmitter>) -> Self {
        self.events = emitter;
        self
    }

    /// Convenience: install the default webhook emitter wired to
    /// this state's webhook store, with a Stripe-like retry
    /// policy. Operators supply the HTTP transport (no real client
    /// ships in the workspace — `reqwest` / `ureq` / `hyper` /
    /// Fireblocks etc. are operator choice).
    #[must_use]
    pub fn with_webhook_transport(mut self, transport: Arc<dyn op_webhook::HttpTransport>) -> Self {
        let store = self.webhooks.clone() as Arc<dyn op_webhook::WebhookStore>;
        let retry: Arc<dyn op_webhook::RetryPolicy> =
            Arc::new(ExponentialBackoffPolicy::stripe_like());
        let dispatcher = Arc::new(WebhookDispatcher::new(store, transport, retry));
        self.events = Arc::new(WebhookEmitter::new(dispatcher));
        self
    }

    /// Builder: install a custom FX quote provider (Wise / OER /
    /// internal hedged feed / whatever).
    #[must_use]
    pub fn with_fx_provider(mut self, provider: Arc<dyn QuoteProvider>) -> Self {
        self.fx = provider;
        self
    }
}
