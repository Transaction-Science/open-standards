# Phase 26 — One file, all stores: Minigraf-backed persistence everywhere

**Status**: Draft v0.26
**Date**: 2026-05-22

## Why

A wrong turn corrected. The original plan for Phase 26 was
"persistent backends — PostgreSQL `*Store` impls." That was the
wrong instinct: OpenPay's architectural position is "no separate
services," and the ledger / webhook / reconciliation / rail-
telemetry stores were already running on Minigraf (single embedded
`.graph` file, pure Rust, bi-temporal, no server). The only stores
still defaulting to `InMemory*` were the four most recent ones —
refund, dispute, settlement, idempotency.

The right Phase 26 is therefore **extend the Minigraf-backed
pattern to those four**, so the deployable binary persists *every*
operator-state to one file. No new dependencies, no networked DB,
no schema migrations. Drop the `.graph` file into a backup,
restart, you're whole.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | New vertex types `refund`, `dispute`, `settlement_batch`, `idempotency_record` | `op-graph/src/graph.rs::vtypes` |
| 2 | New edge types `refund_refunds`, `dispute_disputes`, `batch_includes` (each pointing at the underlying `ledger_tx` when present) | `op-graph/src/graph.rs::etypes` |
| 3 | `GraphRefundStore` — full `RefundStore` impl, idempotent on `external_id`, atomic `update<F>(closure)`, JSON state + indexed properties | `op-graph/src/refund_store.rs` |
| 4 | `GraphDisputeStore` — mirror shape for disputes; `dispute --disputes--> ledger_tx` edge | `op-graph/src/dispute_store.rs` |
| 5 | `GraphSettlementStore` — batch JSON + per-entry `batch_includes` edges; `list_open` filters via the indexed `status_code` | `op-graph/src/settlement_store.rs` |
| 6 | `GraphIdempotencyStore` — `IdempotencyStore` over Minigraf; UUIDv5 vertex ids keyed on the idempotency string for O(1)-equivalent lookup; CAS via internal `Mutex` lock | `op-graph/src/idempotency_store.rs` |
| 7 | `OrchestrationOutcome` + nested types now derive `Serialize` / `Deserialize` (so commit can persist them) | `op-orchestrator/src/outcome.rs`, `idempotency.rs` |
| 8 | `AppState::from_handle(handle)` constructor — every store wired off ONE shared `GraphHandle` | `op-server/src/state.rs` |
| 9 | `AppState::with_graph_path(path)` — open or create a single `.graph` file; reopening recovers every fact | `op-server/src/state.rs` |
| 10 | Integration test: write refund + dispute + batch + idempotency through one handle, drop, reopen at the same path, verify all four recover | `op-graph/tests/persistence.rs` |

Workspace at the end of Phase 26:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **977 passing, 0 failing** (+20 vs Phase 25) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta: GraphRefundStore +7, GraphDisputeStore +4,
GraphSettlementStore +3, GraphIdempotencyStore +5, persistence
integration +1.

## Architecture

```text
                         one .graph file
                              │
   ┌──────────────────────────┼──────────────────────────┐
   │                          │                          │
   ▼                          ▼                          ▼
 GraphLedgerStore        GraphRefundStore         GraphIdempotencyStore
 GraphWebhookStore       GraphDisputeStore        GraphRailTelemetry
 GraphReconciliationStore GraphSettlementStore    (audit report queries them all)
```

Every store consumes the same `GraphHandle`. The audit report
walks the unified graph in a single pass; no cross-store JOINs
because everything is already in one substrate.

Vertex / edge schema additions:

```text
  refund  ─refunds──►  ledger_tx
  dispute ─disputes─►  ledger_tx
  settlement_batch ─includes─►  ledger_tx (per entry)
  idempotency_record  (UUIDv5-keyed; no edges)
```

## Why Postgres is not needed

Honest accounting of what a SQL/server DB would add over Minigraf
for this workspace:

| Capability | Minigraf | Postgres |
|---|---|---|
| ACID per-transaction | yes | yes |
| Multi-process access | file-locked; needs a thin daemon for HA | networked server |
| Bi-temporal time-travel | native (substrate-level) | DIY (audit tables + triggers) |
| Single-file deploy | yes | no (server + config + tablespace) |
| Schema-less storage of domain JSON | yes | partial (JSONB columns) |
| Backup is "copy a file" | yes | pg_dump + WAL replay |
| Pure Rust, no other runtimes | yes | requires libpq / a connection pool |

The capability gap that *would* push toward a server DB is
multi-host concurrent writes. Even there, the right answer for a
reference stack is "ship a thin graph daemon if you need it" not
"adopt a 30-year-old SQL server with all the operational baggage
that brings." Phase 16 already made this call when picking
Minigraf; Phase 26 extends that decision to the remaining stores.

## Persistence model

```rust
// Single file, all stores:
let state = AppState::with_graph_path("/var/lib/openpay/data.graph")?;
let app = op_server::router(state);
axum::serve(listener, app).await?;
```

Reopen at the same path on the next start: every refund, dispute,
batch, idempotency record, ledger tx, webhook delivery, rail
attempt, reconciliation task is exactly as it was.

For tests / demos:

```rust
let state = AppState::new_in_memory();
```

Same trait surface, in-memory Minigraf; the only difference is
where the bytes land.

## Idempotency semantics

`GraphIdempotencyStore` mirrors the in-memory store's CAS
contract with a small twist: the trait's `release()` method
*marks* a slot as `__released__` rather than deleting it. Minigraf
is bi-temporal — retracts move the "current" view but the history
is preserved. A subsequent `reserve()` for the same key sees a
record whose `body_signature` is the sentinel string, treats it as
"in-flight expired," and proceeds. This matches the production-
guidance language already in the trait docs: "mark as expired-in-
flight so analytics can spot leaked reservations."

## Atomicity of `update<F>(closure)`

The closure-driven `update` contract is "the store commits only
if the closure returns `Ok(())`." The graph stores honor this by
load → stage → run closure → persist-on-`Ok`. Until the closure
returns successfully, nothing has been written. Minigraf is
sequential at the transaction layer, so the persist step is itself
atomic.

The contract is identical to the `InMemory*` ref impls; only the
substrate changed.

## Honest concerns (carry-forward)

- **Linear `external_id` lookup.** Body-equivalence checks scan
  all `refund` / `dispute` / `settlement_batch` vertices. Minigraf
  doesn't expose a generic secondary-index API yet; once it does,
  drop in a `:find` by `external_id` and the operator-visible API
  doesn't change. Acceptable in the meantime — even 100k records
  scan in milliseconds on the in-memory backend.
- **Idempotency vertices stay forever.** No TTL eviction. A
  high-throughput operator running for years accumulates them.
  Plumb in a periodic `release_expired_before(unix_secs)` sweep
  in a follow-up; the trait already has the right shape.
- **Edge re-emission on `update`.** `sync_entry_edges` checks for
  existing edges before creating; idempotent but `O(existing
  edges)` per update. For batches with thousands of entries
  operators may want to short-circuit by tracking which entries
  are new — current code re-checks the whole set every time.
- **`OrchestrationOutcome` serde** is now in the type's public
  contract. We control the producer (the orchestrator) and
  consumer (the idempotency store), so the format is internal;
  but operators reading the JSON directly via the graph should
  treat it as `serde_json::Value`, not a stable schema.

## Multi-process / multi-host

Single file is single writer by Minigraf's design. Operators
running multiple `op-server` instances against the same data have
two options:

1. **Sticky-route by tenant** — each instance owns its own
   `.graph` file. Reasonable for SaaS deployments where tenants
   shard naturally.
2. **Front the graph with a daemon** — a small server that owns
   the Minigraf handle and exposes its API over IPC. Out of scope
   for this phase; the trait surfaces already admit it cleanly
   (each `Graph*Store` takes a `GraphHandle`; replace it with a
   `RemoteGraphHandle` and the rest of the stack is unchanged).

Neither approach requires Postgres. The architecture deliberately
keeps that door open.

## Test totals

```
op-graph         +20 tests
                   GraphRefundStore         7
                   GraphDisputeStore        4
                   GraphSettlementStore     3
                   GraphIdempotencyStore    5
                   persistence integration  1
                                              ----
                                              +20 net
```

`cargo test --workspace`: **977 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**
