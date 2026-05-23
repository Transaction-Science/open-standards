# Phase 23 — HTTP API server: vendors can actually deploy this

**Status**: Draft v0.23
**Date**: 2026-05-22

## Why

Until Phase 22 OpenPay was a workspace of well-tested Rust crates
with no HTTP surface. To bring margin back to vendors, operators
have to be able to *run* the stack — and that means a deployable
binary that speaks REST. Phase 23 adds `op-server`: an
[`axum`]-based HTTP server that exposes the workspace's surfaces
(intents, refunds, disputes, settlement, audit) as a clean REST
API.

Second of three sequenced phases (22 → 23 → 24). Next phase wires
concrete driver SDKs and reference rail integrations.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `op-server` library + binary crate (`axum` 0.8, `tokio` 1.x, `tower-http`) | `crates/op-server/` |
| 2 | `AppState` — `Clone`-able bundle of `Arc<store>`s + `Arc<Orchestrator>` + shared `GraphHandle` | `state.rs` |
| 3 | `ApiError` — HTTP error envelope with stable JSON shape + per-crate `From` conversions mapping each domain error to the right status code | `error.rs` |
| 4 | Intent endpoint — `POST /v1/intents` runs the orchestrator, returns per-attempt outcome list | `handlers/intent.rs` |
| 5 | Refund endpoints — create / get / submit / approve / settle | `handlers/refund.rs` |
| 6 | Dispute endpoints — create / get / attach evidence | `handlers/dispute.rs` |
| 7 | Settlement endpoints — open / get / add entry / close | `handlers/settlement.rs` |
| 8 | Audit endpoint — `GET /v1/audit/report?start_tx=..&end_tx=..&generated_at_unix_secs=..` | `handlers/audit.rs` |
| 9 | Health + readiness — `GET /health` always-200, `GET /readiness` reports store counts | `handlers/health.rs` |
| 10 | Binary entry point — TCP bind + graceful Ctrl-C / SIGTERM shutdown + `tracing-subscriber` setup | `main.rs` |
| 11 | Integration tests via `tower::ServiceExt::oneshot` — no socket, no listener, 9 e2e tests | `tests/api.rs` |

Workspace at the end of Phase 23:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **906 passing, 0 failing** (+9 vs Phase 22) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

## Endpoint surface

```
GET   /health                                  → 200 always
GET   /readiness                               → store counts
POST  /v1/intents                              → run orchestrator
POST  /v1/refunds                              → create refund
GET   /v1/refunds/{id}                         → fetch refund
POST  /v1/refunds/{id}/submit                  → Requested→Submitted
POST  /v1/refunds/{id}/approve                 → Submitted→Approved
POST  /v1/refunds/{id}/settle                  → Approved→Settled
POST  /v1/disputes                             → create dispute
GET   /v1/disputes/{id}                        → fetch dispute
POST  /v1/disputes/{id}/evidence               → attach evidence
POST  /v1/settlement/batches                   → open batch
GET   /v1/settlement/batches/{id}              → fetch batch
POST  /v1/settlement/batches/{id}/entries      → add tx to batch
POST  /v1/settlement/batches/{id}/close        → apply holdback + close
GET   /v1/audit/report                         → multi-store audit join
```

All POST endpoints take JSON. All responses are JSON; errors carry
the stable shape:

```json
{ "code": "not_found", "message": "refund not found: 0123...", "details": null }
```

## State shape

```rust
pub struct AppState {
    pub orchestrator: Arc<Orchestrator>,
    pub refunds: Arc<InMemoryRefundStore>,
    pub disputes: Arc<InMemoryDisputeStore>,
    pub settlement: Arc<InMemorySettlementStore>,
    pub ledger: Arc<GraphLedgerStore>,
    pub reconciliation: Arc<GraphReconciliationStore>,
    pub telemetry: Arc<GraphRailTelemetry>,
    pub graph: GraphHandle,
}
```

`Clone` is cheap (everything is `Arc` or the graph handle's own
`Arc`). axum extracts `State<AppState>` per request. Operators
replace `InMemory*` with their own `*Store` impls (Postgres,
TigerBeetle, etc.) by constructing a non-default state and
calling `router(state)`.

## Error mapping

Each domain crate's `Error` enum has a `From<Error> for ApiError`
implementation that picks the right HTTP class:

| Domain variant | HTTP |
|---|---|
| `*::NotFound(_)`, `op_ledger::*NotFound(_)` | 404 |
| `*::InvalidTransition { .. }`, `op_ledger::TerminalState` | 409 |
| `*::IdempotencyMismatch(_)` | 409 |
| `*::Invalid`, `*::AmountExceeded`, currency / parse failures | 400 |
| `op_orchestrator::*`, `op_graph::*` (catch-all) | 500 |

The body always uses the stable `{code, message, details}`
envelope, regardless of which crate raised the error.

## Why `&impl SettlementStore` matters here

The settlement engine's `update<F>(... f: F)` trait-method makes
`SettlementStore` non-dyn-compatible. The HTTP handler resolves
this by holding `Arc<InMemorySettlementStore>` directly (concrete
type) — exactly the integration shape the engine expects. Same
pattern for the refund and dispute stores.

## Testing approach

`tower::ServiceExt::oneshot` invokes the router with a synthetic
`Request<Body>` and returns the `Response<Body>` directly — no
TCP socket, no listener, no port assignment. Each test in
`tests/api.rs` builds an `AppState::new_in_memory()`, calls
`oneshot`, asserts on status + JSON body. Fast (≈10ms per test),
deterministic, no port conflicts.

## Honest concerns (carry-forward)

- **No auth.** Operators apply a `tower::Layer` (API-key
  middleware, JWT validator, mTLS). The reference binary ships
  unauthenticated — running it on a public IP is the operator's
  problem.
- **No TLS.** Terminate upstream (Caddy / nginx / load balancer).
  The binary listens plain HTTP.
- **No request body size limits.** Default axum limits apply.
  Operators wanting harder bounds add `RequestBodyLimitLayer`.
- **Reference handlers only.** Refund-approve doesn't take a body;
  dispute lifecycle (escalate/represent/resolve) is currently
  represented only via the in-memory store API, not HTTP. Adding
  those endpoints is straightforward — followed the same pattern
  as the shipped ones.
- **No pagination.** Endpoints that *would* benefit from pagination
  (audit window, list-open batches) currently return everything in
  the window. Real deployments add `?limit=&cursor=` once the
  query layer in `op-graph` supports it.
- **No webhook delivery from the server.** `op-webhook` exists in
  the workspace but its dispatcher isn't wired through the HTTP
  layer — operators run the dispatcher in a separate worker. The
  server publishes events to the store; the dispatcher polls /
  consumes them.
- **Synchronous orchestrator on an async handler.** `Orchestrator::run`
  is sync; the axum handler awaits a `tokio::task::spawn_blocking`
  in production, but the reference path runs it inline on the
  async runtime. Fine for the in-memory ref adapter, painful for a
  real PSP-calling adapter.

## Operator boot sequence

```bash
OP_BIND_ADDR=0.0.0.0:8080 \
RUST_LOG=info,op_server=debug \
cargo run --release -p op-server
```

The binary uses `tracing-subscriber` to emit structured logs;
operators install their own subscriber (OpenTelemetry exporter,
JSON-to-Datadog, etc.) by replacing the init block in `main.rs`
or by linking against the library and calling `op_server::router`
from their own composition root.

## Test totals

```
op-server      9 integration tests
                  health 1, readiness 1, refund create/get/idem 3,
                  refund 404 1, dispute 1, settlement 1, audit 1,
                  bad-currency 1
                                                              ----
                                                              +9 net
```

`cargo test --workspace`: **906 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**
