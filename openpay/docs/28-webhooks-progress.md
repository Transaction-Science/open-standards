# Phase 28 — Webhooks wired end-to-end

**Status**: Draft v0.28
**Date**: 2026-05-22

## Why

`op-webhook` has shipped since Phase 13: dispatcher,
Stripe-compatible signing, exponential backoff with jitter,
endpoint auto-disable. But no internal event in the workspace
actually called it. Refunds settled, disputes escalated,
subscriptions canceled — and nothing fired outbound. That made the
crate dead weight from an operator's perspective.

Phase 28 closes that loop: add an `EventEmitter` seam, wire it
into every domain-mutation handler in op-server, expose endpoint
registration over HTTP, and prove with an integration test that a
refund creation lands on a subscribed endpoint.

First of three sequenced phases (28 → 29 → 30). Next: multi-
currency / FX, then 3DS/SCA resume.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `EventEmitter` trait — single-method `emit(event_type, payload, now_unix_secs)` | `op-webhook/src/emitter.rs` |
| 2 | `WebhookEmitter` adapter — wraps a `WebhookDispatcher` and fans out events to subscribed endpoints | `op-webhook/src/emitter.rs` |
| 3 | `NoOpEmitter` — drops every event on the floor; default in `AppState` so operators who haven't wired transports stay green | `op-webhook/src/emitter.rs` |
| 4 | `AppState.events: Arc<dyn EventEmitter>` + `with_webhook_transport(transport)` builder | `op-server/src/state.rs` |
| 5 | `AppState.webhooks: Arc<GraphWebhookStore>` — webhooks now persist to the same `.graph` file as everything else | `op-server/src/state.rs` |
| 6 | `crate::events::emit(state, event_type, body)` helper — wall-clock timestamp + serde JSON encoding + tracing-logged failure | `op-server/src/events.rs` |
| 7 | Emissions wired into 11 handler success points across refund / dispute / settlement / subscription / intent | `op-server/src/handlers/*.rs` |
| 8 | HTTP endpoint CRUD: `POST /v1/webhooks/endpoints`, `GET /v1/webhooks/endpoints/{id}`, `POST .../disable`, `POST .../enable` | `op-server/src/handlers/webhook.rs`, `routes.rs` |
| 9 | Integration tests — register endpoint + create refund → verify outbound delivery; disable + create refund → verify zero deliveries | `op-server/tests/api.rs` |

Workspace at the end of Phase 28:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **1027 passing, 0 failing** (+4 vs Phase 27) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

## Emitted events

| Event type | Emitted by |
|---|---|
| `refund.created` | `POST /v1/refunds` |
| `refund.submitted` | `POST /v1/refunds/{id}/submit` |
| `refund.approved` | `POST /v1/refunds/{id}/approve` |
| `refund.settled` | `POST /v1/refunds/{id}/settle` |
| `dispute.created` | `POST /v1/disputes` |
| `dispute.evidence_attached` | `POST /v1/disputes/{id}/evidence` |
| `settlement.batch_opened` | `POST /v1/settlement/batches` |
| `settlement.batch_closed` | `POST /v1/settlement/batches/{id}/close` |
| `subscription.created` | `POST /v1/subscriptions` |
| `subscription.canceled` | `POST /v1/subscriptions/{id}/cancel` |
| `subscription.paused` | `POST /v1/subscriptions/{id}/pause` |
| `subscription.resumed` | `POST /v1/subscriptions/{id}/resume` |
| `intent.approved` / `intent.declined` / `intent.requires_action` | `POST /v1/intents` |

Payloads are the same JSON the HTTP handler returned to the
caller — operators implementing webhook consumers can deserialize
into the same structs they'd see from a direct API call.

## Design choice: emit at the handler, not the store

Two ways to wire emission:

1. **Store-trait callbacks** — each `RefundStore` impl emits after
   a successful mutation. Pro: every caller, including direct
   library users, gets events. Con: invasive — adds an
   `Arc<dyn EventEmitter>` parameter to every store constructor
   and every trait method, breaking the existing surfaces.
2. **Handler-boundary emit** — the HTTP layer emits right after
   the store's mutation persists. Pro: zero changes to the
   existing store traits. Con: callers using the library directly
   (CLI, embedded SDK) don't automatically fire events.

We chose (2). The reference deployment is the HTTP server, so
the emit-site is the HTTP boundary. Library-direct callers either
wire their own emitter at their own call sites or upgrade to a
v2 store trait if/when that becomes worth doing.

## Why emission is fire-and-forget

A webhook delivery failure must **never** roll back the user's
state change. The customer was refunded; we recorded it; the fact
that our HTTP POST to the merchant's endpoint failed is a
delivery problem, not a domain problem. The dispatcher's retry
loop handles delivery — the caller's transaction is already
committed.

Concretely:

```rust
let id = state.refunds.create_refund(refund.clone())?;       // persists
let stored = state.refunds.get_refund(id)?;
let response = RefundResponse::from(&stored);
emit(&state, "refund.created", &response);                    // best-effort
Ok(Json(response))                                            // user sees 200
```

`emit` swallows serde / dispatch errors with a `tracing::warn!`.
The HTTP 200 went out before the operator's webhook consumer
ever received its POST. That asymmetry is correct.

## HTTP surface

```
POST /v1/webhooks/endpoints                # register
GET  /v1/webhooks/endpoints/{id}
POST /v1/webhooks/endpoints/{id}/disable   # operator-driven kill switch
POST /v1/webhooks/endpoints/{id}/enable    # reset consecutive_failures
```

Body for create:

```json
{
  "url": "https://merchant.example/webhooks/openpay",
  "secret": "merchant-shared-secret",
  "event_filters": ["refund.created", "subscription.canceled"]
}
```

Filters are exact-match strings; `"*"` matches everything. Same
contract as `op-webhook::Endpoint::matches`.

## Persistence

Webhook endpoints, events, and delivery attempts now live in the
same `.graph` file as refunds, disputes, settlements, etc. (Phase
26's `GraphWebhookStore` was already in the workspace; this phase
wires it into `AppState` as the default). One file, one substrate,
everything queryable from the audit report.

## Operator wiring

The reference binary defaults to `NoOpEmitter`. Operators turn on
real delivery by handing in a transport:

```rust
let transport: Arc<dyn HttpTransport> = Arc::new(MyReqwestTransport::new());
let state = AppState::with_graph_path("/var/lib/openpay/data.graph")?
    .with_webhook_transport(transport);
```

We don't ship a real `reqwest` / `ureq` / `hyper` adapter in the
workspace by design — Phase 13's HttpTransport trait is the seam,
and operators bring their own. A reference `MockTransport`
already exists for tests.

## Honest concerns (carry-forward)

- **No HTTP transport in the workspace.** Real delivery still
  requires the operator to plug in `reqwest` or similar. This is
  intentional — the workspace stays free of HTTP client
  dependencies — but it's a step operators have to take. A
  future phase could ship a thin `op-webhook-reqwest` crate.
- **`process_due_retries` not called automatically.** The
  dispatcher knows how to retry failed deliveries; the
  background loop that calls it on a schedule isn't wired into
  `op-server` (it would need a `tokio::spawn` plus a tick
  interval). Operators run a sidecar that calls
  `dispatcher.process_due_retries()` periodically.
- **No emission on store-direct (non-HTTP) callers.** A binary
  that calls `state.refunds.create_refund(...)` directly bypasses
  the handler-layer emit. Library users who care must call
  `state.events.emit(...)` themselves or wait for a v2 store
  trait that bakes emission in.
- **No event versioning.** Payloads are the current shape of
  `RefundResponse` / etc.; if those change, existing webhook
  consumers break. Operators should treat them as evolving (add
  fields safely; never remove or rename) until a versioning
  scheme is added.
- **No replay UI.** `WebhookDispatcher::replay` exists but no
  HTTP endpoint surfaces it yet. Operators using the in-process
  API can call it directly.

## Test totals

```
op-webhook         +2 tests (emitter dispatches / noop is silent)
op-server          +2 integration tests (refund webhook delivery,
                       disabled endpoint skip)
                                                              ----
                                                              +4 net
```

`cargo test --workspace`: **1027 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**
