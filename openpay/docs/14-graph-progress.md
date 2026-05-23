# Phase 14 — `op-graph` graph-backed stores

**Status**: Draft v0.14
**Date**: 2026-05-18

## Why a graph database, not Postgres

OpenPay's data is graph-shaped. A relational schema flattens
that shape into joins; a graph schema preserves it. Two examples:

1. **A ledger transaction is a hyperedge connecting 2+ accounts.**
   "Which accounts did this transaction touch?" in relational SQL
   is `SELECT a.* FROM entries e JOIN accounts a ON e.account_id =
   a.id WHERE e.transaction_id = $1`. In a graph it's one
   traversal of out-edges. The graph version is also faster on
   wide ledgers because the edges sit next to the transaction
   vertex on disk.

2. **Reversal chains are subgraphs.** Tracing `tx → reverses → tx
   → reverses → tx` in relational SQL is a recursive CTE. In a
   graph it's a four-line loop. Same with fraud-graph queries
   ("which accounts share a device fingerprint via attempted
   chargebacks"), which become first-class operations.

We chose **IndraDB** because:

- Pure Rust, embedded `MemoryDatastore` available out of the box,
- TAO-inspired (Facebook's graph datastore) so query semantics
  scale to the size of a real ledger,
- Mature (~10 years, 2.4k stars), latest release 5.0.0
  (2025-08-16),
- Pluggable datastores — operators can swap in RocksDB for
  persistence later without touching our code.

## License compatibility

This is the section I want to be honest about up front: **IndraDB
is MPL-2.0**, and OpenPay is **Apache-2.0**. They are compatible
in the way that matters for OpenPay operators: a downstream
deployment that **links** against IndraDB from an Apache-2.0
binary does not have to relicense the rest of its code. The
MPL-2.0 obligation is at the **file** level: any modifications to
IndraDB's own source files must remain MPL-2.0. We never modify
IndraDB; we only link.

This is the same composition model that ships in Firefox-embedded
applications across the industry. Operators who want a single-
license deployment can substitute a different `LedgerStore` /
`WebhookStore` implementation — the trait surfaces in `op-ledger`
and `op-webhook` exist precisely for that.

The license note is reproduced in `crates/op-graph/src/lib.rs`
and called out explicitly in this doc so no operator is surprised
later.

## What shipped

A new crate `crates/op-graph` plus an integration test suite that
exercises a single shared `GraphHandle` across both ledger and
webhook stores. **Total: 74 tests (64 unit + 10 integration),
~3,696 LOC.** Cumulative across all 14 phases: ~870 tests,
~35,640 LOC.

| Module | Unit tests | LOC |
|---|---|---|
| `lib.rs` | — | 117 |
| `error.rs` | — | 105 |
| `graph.rs` (typed facade over IndraDB) | 13 | 453 |
| `ledger_store.rs` (`GraphLedgerStore`) | 20 | 1,135 |
| `webhook_store.rs` (`GraphWebhookStore`) | 20 | 1,056 |
| `queries.rs` (read-side helpers) | 11 | 380 |
| `tests/integration.rs` | 10 | 450 |
| **Phase 14 total** | **74** | **3,696** |

## Architecture

```
crates/op-graph/
├── Cargo.toml         — deps: op-core, op-ledger, op-webhook,
│                       indradb-lib = "=5.0.0" pinned tight
├── src/
│   ├── lib.rs         — module roots + schema doc + license note
│   ├── error.rs       — 9-variant Error with From<indradb::Error>,
│   │                   From<indradb::ValidationError>,
│   │                   From<serde_json::Error>, plus bridges to
│   │                   op_ledger::Error and op_webhook::Error
│   ├── graph.rs       — GraphHandle: typed facade over
│   │                   Database<MemoryDatastore>. vtypes / etypes
│   │                   string constant modules.
│   ├── ledger_store.rs— GraphLedgerStore: implements
│   │                   op_ledger::LedgerStore. JSON property
│   │                   codecs; HashMap<String, TransactionId>
│   │                   external_id cache for idempotency.
│   ├── webhook_store.rs— GraphWebhookStore: implements
│   │                   op_webhook::WebhookStore. Base64 codec for
│   │                   binary payload + secret. CSV codec for
│   │                   event_filters. Per-event-type endpoint
│   │                   index.
│   └── queries.rs     — 4 read-side helpers
└── tests/
    └── integration.rs — 10 cross-domain tests
```

### Vertex / edge schema

**Vertex types**: `ledger_account`, `ledger_tx`, `ledger_ledger`,
`webhook_event`, `webhook_endpoint`, `webhook_attempt`.

**Edge types**:
- `ledger_debit`: `ledger_tx` → `ledger_account` (`amount_minor`,
  `currency_code`, `currency_exponent`)
- `ledger_credit`: `ledger_tx` → `ledger_account` (same props)
- `ledger_in_ledger`: `ledger_tx` or `ledger_account` →
  `ledger_ledger`
- `ledger_reverses`: `ledger_tx` → `ledger_tx` (the original)
- `webhook_delivers`: `webhook_event` → `webhook_attempt`
- `webhook_to`: `webhook_attempt` → `webhook_endpoint`

### Why edge-per-entry instead of entries-as-properties

Two options for storing transaction entries:

1. **Property blob**: serialize the entire `Vec<Entry>` as JSON
   and stash on the tx vertex.
2. **Edge per entry**: each entry becomes a typed edge from the
   tx vertex to the account vertex.

Option 1 is simpler but makes "which accounts did this tx touch?"
an O(N) scan of every tx in the database. Option 2 makes it a
single out-edge traversal — which is the whole reason we picked
a graph database. So edge-per-entry it is.

Cost: a tx with k entries is k+1 vertices' worth of edges (plus
the `ledger_in_ledger` edge to the parent ledger). Acceptable.

## Key design decisions

### 1. `GraphHandle` is the seam

The crate exposes a thin facade rather than letting callers pass
raw `Database<MemoryDatastore>` references around. Rationale:
the OpenPay schema (vtypes, etypes, property names) lives in one
place, and operators who want to swap to a different graph
database (e.g. via a custom datastore implementation) can do so
by replacing one file.

`GraphHandle::raw()` is the escape hatch for advanced queries
the facade doesn't model. Calling `raw()` opts out of stability.

### 2. Shared `GraphHandle` across stores

`GraphLedgerStore::with_handle(handle.clone())` and
`GraphWebhookStore::with_handle(handle.clone())` let both stores
write to **one** graph. This unlocks the cross-domain queries:
"this webhook event was emitted because this ledger transaction
was posted — show me both in one read." Demonstrated in
integration test #9.

### 3. Idempotency cache mirrors `InMemoryLedgerStore`

`GraphLedgerStore` keeps a `Mutex<HashMap<String, TransactionId>>`
seeded lazily on write. It's not authoritative — the graph is —
but it makes `find_by_external_id` O(1) instead of an
all-vertices scan. Production stores plugging in a persistent
datastore would replace this with an indexed property query
(IndraDB supports `with_property_equal_to` plus property
indexing); the in-memory facade doesn't bother because the cache
is fast enough.

### 4. JSON property codecs are explicit

We don't `serde_json::to_value(&account)?` and stash the whole
struct as a blob. Every field is encoded explicitly with a known
property name and known type (`currency_code` as string,
`amount_minor` as i64, `effective_at_unix_secs` as u64). Reasons:

- **Future schema evolution**: when we add a property, we don't
  have to deserialize old blobs.
- **Query-friendly**: `VertexWithPropertyValueQuery` works against
  individual properties; a blob property is opaque.
- **Audit-friendly**: an operator browsing the graph (via the
  indradb-server CLI) sees readable property names.

The cost is more code in the codec helpers, but they're trivial
and test-covered.

### 5. Local base64 + filter-CSV codecs, not extra deps

The webhook payload and endpoint secret are `Vec<u8>`. JSON
can't carry raw bytes, so we encode. We wrote a ~60-line
base64 codec rather than adding the `base64` crate — same
dependency hygiene as the local hex codec in op-webhook. RFC 4648
test vectors verified in `base64_known_vectors`.

The event-filter list is encoded as a CSV inside one property
(escaping `,` as `\,`). It could've been a JSON array, but
keeping it textual lets IndraDB indexing work on the full string
(future optimization) and avoids the JSON-property-of-arrays
edge cases.

### 6. Sealed `Error::Invariant`

When the graph is in an unexpected state (e.g. a tx vertex with
no debit/credit edges, or a property of the wrong JSON shape),
we raise `Error::Invariant(String)`. Callers shouldn't normally
see these — they'd indicate either schema drift (someone wrote
through the raw IndraDB API bypassing our facade) or a bug in
our codec. The variant exists so we can be explicit instead of
panicking.

### 7. License-friendly module boundary

`op-graph` depends on `indradb-lib`, but `op-ledger` and
`op-webhook` do NOT — the trait surfaces they expose are
storage-agnostic. Operators who want to ship under a strict
Apache-2.0 hygiene policy can use `op-ledger` and `op-webhook`
with their own (Apache-2.0 or MIT) `LedgerStore` /
`WebhookStore` implementation, simply not depending on
`op-graph`.

## What this crate does NOT do

- **No graph query language.** No Cypher / GQL / Gremlin.
  IndraDB itself doesn't expose those; we expose typed Rust
  helpers (the four functions in `queries.rs`).
- **No persistence by default.** `MemoryDatastore` only. IndraDB
  supports persistent backends via its `rocksdb-datastore`
  feature (we don't enable it; operators choosing persistence
  opt in by depending on `indradb-lib` with that feature and
  passing their own `Database<RocksdbDatastore>` to a custom
  store).
- **No cross-store transactionality.** Writing a ledger
  transaction and a webhook event is two store calls. They're
  both backed by IndraDB so they share the underlying mutex, but
  there's no rollback if the second one fails.
- **No streaming subscriptions.** Operators poll or run periodic
  queries.
- **No fraud-graph queries yet.** The schema supports them
  (everything is in one graph), but the typed helpers in
  `queries.rs` cover only the four ledger + webhook traversals.
  Fraud queries land in Phase 16 or later.

## Bugs caught during construction

1. **MPL-2.0 vs Apache-2.0 compatibility surprise.** I initially
   assumed IndraDB was Apache-2.0 or MIT. Searching turned up
   MPL-2.0. License analysis added to the architectural doc to
   set operator expectations correctly.

2. **IndraDB API drift between major versions.** Earlier docs
   showed `outbound(limit: u32) -> PipeEdgeQuery`. The 5.0.0
   surface is `outbound() -> ValidationResult<PipeQuery>` with
   `.limit()` and `.t()` as builder chains. Pinned to `=5.0.0`
   to avoid this surprise upstream.

3. **Orphan vertex leak on put_attempt.** First draft of
   `GraphWebhookStore::put_attempt` created the attempt vertex
   *before* verifying the event and endpoint vertices existed.
   If validation failed, an orphan vertex was left behind. Fixed
   to validate first, create after.

4. **Status property is not the tx's lifecycle state.** When
   `mark_posted` is called, we have to update the **status
   property** on the tx vertex, not just call `Transaction::post`
   on the in-memory copy. First draft updated the local struct
   but didn't write back; balance() then read the old status and
   miscounted. Fixed by writing the new status property after
   the state-machine method succeeds.

5. **`Currency` constants vs arbitrary codes.** My
   `currency_from_props(code, exponent)` first tries the seven
   curated constants (USD/EUR/BRL/INR/GBP/JPY/CNY) for the
   fast-path, then falls through to `Currency::try_new` for
   arbitrary codes. Without the fast-path the constants
   reconstructed via `try_new` would be `!=` to the original
   despite encoding the same currency, because the comparison
   sees them as semantically equivalent but structurally
   distinct values. (Actually they would be equal — `Currency`
   derives `PartialEq` on its raw byte fields. But the curated
   path is faster and avoids the validation work, so we keep
   it.)

6. **Direction-as-property vs direction-as-edge-type.** I
   initially considered storing `direction` as a property on a
   single `ledger_entry` edge type. Decided against: making
   debit and credit separate edge types lets us filter by
   direction in a single query (`out_edges(tx, "ledger_debit")`)
   rather than out-edges + property filter. Trade-off: two
   edge types instead of one. Worth it.

7. **`SpecificVertexQuery::single(uuid).outbound()?.t(ident)`
   composition.** `outbound()` returns `ValidationResult<PipeQuery>`;
   `t()` consumes `PipeQuery` and returns `PipeQuery`. Took two
   tries to get the ordering right — the chain looks naturally
   `?.t(...)` but I initially wrote it as `.t(ident)?` (calling
   `t` on a Result, which doesn't compile).

## Honest concerns going into Phase 15+

- **`balance()` is O(edges-on-account)**, not O(1). For
  high-throughput accounts (e.g. a corporate cash account with
  millions of transactions) this is slow. A real production
  deployment would cache the running balance and update it on
  every entry. Out of scope for the reference impl.

- **`list_due_retries` does a full-vertex scan.** IndraDB
  doesn't (yet) give us a typed-vertex iterator at the embedded
  API level — we have to walk `AllVertexQuery` and filter by
  type and property. For a webhook system with tens of
  thousands of attempts this is fine; for millions it isn't.
  Fix: when IndraDB exposes typed scans, or when we plug in
  RocksDB and use property indexing.

- **No cycle protection in `reversal_chain` beyond a `HashSet`.**
  If an operator manually wires `t1 → reverses → t2` and `t2 →
  reverses → t1` (which shouldn't happen but the graph doesn't
  forbid), we detect the cycle and return what we've found so
  far. We don't error. Documented in the function comment.

- **The license note is operator-facing.** Anyone considering
  redistributing OpenPay needs to read it. We surface it in
  three places (lib.rs doc comment, Cargo.toml description,
  this progress doc) so no one is surprised.

## Cumulative state

| Phase | Tests | LOC |
|---|---|---|
| 1 op-core | 19 | ~600 |
| 2 op-iso20022 | 43 | ~1,400 |
| 3 op-emv | 50 | ~1,800 |
| 4 op-rails-card | 46 | ~2,100 |
| 5 op-rails-a2a | 73 | ~3,200 |
| 6 op-fraud | 65 | ~2,400 |
| 7 op-vault | 51 | ~2,600 |
| 8 op-ffi-swift | 44 | ~2,700 |
| 9 op-ffi-jni | 69 | ~2,950 |
| 10 op-wasm | 71 | ~2,200 |
| 11 op-orchestrator + kiosk + e2e | 90 | ~4,150 |
| 12 op-ledger | 69 | ~2,540 |
| 13 op-webhook | 106 | ~3,304 |
| **14 op-graph** | **74** | **~3,696** |
| **Total** | **~870** | **~35,640** |

## What's next (Phase 15+ candidates)

- **`op-reconciliation`** — diff ledger vs PSP / bank statements;
  produces per-discrepancy resolution tasks. Natural follow-on
  now that we have a graph-backed ledger to query.
- **Persistent IndraDB backend** — wire up `rocksdb-datastore`
  via a `GraphHandle::new_persistent(path)` constructor. Mostly
  one file's worth of work.
- **Fraud-graph queries** — `accounts_linked_via_chargeback`,
  `endpoints_sharing_secret_prefix`, `attempts_with_shared_ip`.
  Two-hop traversals on the existing schema; no new edge types
  needed.
- **`op-router` integration** — when the orchestrator decides
  which rail to use, it can consult the graph for
  recently-failed endpoints and avoid them.
- **OpenTelemetry trace propagation** — one trace id flowing
  intent → ledger → webhook → consumer.
- **GraphQL adapter** — expose the typed `queries.rs` helpers as
  a GraphQL schema so operator dashboards can poll without
  building Rust.

The thesis stands: pure-Rust, Apache-2.0 reference stack with a
license-compatible MPL-2.0 graph backend chosen because OpenPay's
data is graph-shaped.
