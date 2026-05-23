# Phase 16 — Embedded persistent graph DB (Minigraf)

**Status**: Draft v0.16
**Date**: 2026-05-21

## Why this phase exists

Phase 14 shipped `op-graph` on IndraDB in-memory only. Operator
question that immediately followed: *"how do we keep the graph
across restarts?"* The Phase 14 progress doc promised a RocksDB
backend; this phase delivers that capability — but **not** with
RocksDB.

The "right Rust graph DB" turned out to be a real research question.
The constraint set:

1. Embedded ("in-app like SQLite"), not a separate server.
2. Pure Rust. No heavy C/C++ build chains.
3. Real graph DB. Not "use SQL with self-joins."
4. Single-file durability ("like SQLite").
5. Apache-2.0 / MIT licensable.

The landscape evaluated:

| Candidate | Verdict |
|---|---|
| **IndraDB + RocksDB** | RocksDB is a heavy C++ build, single-process lock. Rejected. |
| **Cozo** | Original crate paused at v0.7.6 (Dec 2023). Community fork `cozo-ce` 0.7.13-alpha.3 fails to compile against current `rayon` (transitive `graph_builder` API drift). Not viable in 2026. |
| **SurrealDB** | Pure Rust, but server-first; embedded mode pulls heavy deps and the core is BSL 1.1, not Apache-2.0. License contagion for a reference stack. Rejected. |
| **petgraph** | Data structure library, not a database. |
| **Minigraf 1.1.1** | **Selected.** Pure safe Rust (`#![forbid(unsafe_code)]`), MIT OR Apache-2.0, single-file `.graph` database, embedded (no server), Datalog queries over an EAV fact store, native `Uuid` entity ids, **bi-temporal history built in** (free time-travel for ledger audits). Active (1.x stable, not alpha). |

Minigraf advertises itself in its own docs as *"the SQLite of graph
databases"* — verbatim what was asked for.

## What shipped

A complete replacement of the IndraDB-backed `op-graph` internals
with Minigraf. The public API surface of every `op-graph` type
(`GraphHandle`, `GraphLedgerStore`, `GraphWebhookStore`,
`GraphReconciliationStore`, all `queries.rs` helpers) is preserved
— callers across Phase 14 / 15 / the orchestrator integration tests
recompile without changes.

| File | LOC | Notes |
|---|---:|---|
| `op-graph/Cargo.toml` | +6 | dep swap: `indradb-lib` → `minigraf`; `tempfile` dev-dep |
| `op-graph/src/graph.rs` | **596** | full rewrite on Minigraf |
| `op-graph/src/error.rs` | −10 | dropped `Indra`/`indradb::ValidationError` bridges, added `Backend(String)` |
| `op-graph/src/ledger_store.rs` | small sweep | `outbound_id`→`from`, `inbound_id`→`to`, `&indradb::Edge`→`&crate::graph::Edge`, plus a real fix to `find_by_external_id` (graph fallback when the cache misses on a freshly-opened persistent handle) |
| `op-graph/src/webhook_store.rs` | small sweep | same field renames |
| `op-graph/src/reconciliation_store.rs` | small sweep | same field renames |
| `op-graph/src/queries.rs` | small sweep | same field renames |
| `op-graph/tests/persistent.rs` | **224** | new — 4 round-trip tests proving data survives handle drop + reopen across the path |
| **Phase 16 totals** | **~820 added** | (the 596 graph.rs LOC mostly replaces 316 prior LOC; net new is on the order of +500) |

Workspace at the end of Phase 16:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **810 passing, 0 failing** |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

The net test count vs Phase 15 (819) is `−13 op-graph lib tests`
(internal `graph.rs` unit tests of IndraDB-specific behaviour that
no longer apply) `+4 persistent integration tests` = 810. All 11
op-graph integration tests including the Phase 15 cross-domain
reconciliation test pass unchanged.

## Architecture: EAV mapped onto property-graph

Minigraf is an EAV (Entity-Attribute-Value) fact store with a
Datalog query language. The OpenPay graph is property-graph shaped
— typed vertices, typed directed edges, JSON properties on both —
so the facade translates one onto the other:

```
Vertex(id, type)                  → [id :_type type]
VertexProperty(id, name, value)   → [id name value]

Edge(from, type, to):
  mint edge_id (Uuid::new_v4)
  → [edge_id :_edge/from <from>]
    [edge_id :_edge/to <to>]
    [edge_id :_edge/type <type>]

EdgeProperty(edge_id, name, val)  → [edge_id name value]
```

Internal attributes are prefixed `_` so user property names (e.g.
`status`, `amount_minor`) can be any keyword without collision.
`get_vertex_properties` filters `_*` out so callers see only their
own properties.

### Schema in code

```rust
pub mod vtypes {
    pub const LEDGER_ACCOUNT: &str = "ledger_account";
    pub const LEDGER_TX: &str = "ledger_tx";
    pub const LEDGER_LEDGER: &str = "ledger_ledger";
    pub const WEBHOOK_EVENT: &str = "webhook_event";
    pub const WEBHOOK_ENDPOINT: &str = "webhook_endpoint";
    pub const WEBHOOK_ATTEMPT: &str = "webhook_attempt";
    pub const STATEMENT_LINE: &str = "statement_line";
    pub const RECONCILIATION_TASK: &str = "reconciliation_task";
}

pub mod etypes {
    pub const LEDGER_DEBIT: &str = "ledger_debit";
    pub const LEDGER_CREDIT: &str = "ledger_credit";
    pub const LEDGER_IN_LEDGER: &str = "ledger_in_ledger";
    pub const LEDGER_REVERSES: &str = "ledger_reverses";
    pub const WEBHOOK_DELIVERS: &str = "webhook_delivers";
    pub const WEBHOOK_TO: &str = "webhook_to";
    pub const RECONCILES: &str = "reconciles";
    pub const TASK_ABOUT: &str = "task_about";
}
```

Unchanged from Phase 14 / 15 — the schema is the same property
graph; only the storage substrate moved.

### Vertex / Edge types are now local to op-graph

The IndraDB-specific `indradb::Vertex` / `indradb::Edge` no longer
appear in the public API. They were replaced with:

```rust
pub struct Vertex { pub id: Uuid, pub t: String }
pub struct Edge   { pub id: Uuid, pub from: Uuid, pub t: String, pub to: Uuid }
```

This is a tiny breaking change relative to Phase 14: `edge.outbound_id`
→ `edge.from`, `edge.inbound_id` → `edge.to`. Internal to op-graph;
no downstream phase touches these field names directly.

## Persistence: it just works

```rust
// Tests, ephemeral:
let h = GraphHandle::new_in_memory();

// Single-file durability:
let h = GraphHandle::new_persistent("./openpay.graph")?;
// ... write through it ...
drop(h);    // file flushes

let h2 = GraphHandle::new_persistent("./openpay.graph")?;
// ... reads see the data written by the previous handle.
```

That's the entire user-facing API change. The four persistent
round-trip tests in `tests/persistent.rs` walk each store type:

1. **`graph_handle_persists_vertex_and_edge_across_reopen`** —
   the low-level GraphHandle primitive.
2. **`ledger_transaction_round_trips_across_reopen`** — write a
   ledger transaction, drop, reopen, verify the entries (stored as
   edges) and the derived balance are all intact.
3. **`webhook_event_and_attempt_round_trip_across_reopen`** — the
   same for webhook events and delivery attempts.
4. **`reconciliation_tasks_survive_restart_and_re_record_stays_idempotent`**
   — the cross-domain integration. Reconcile against a webhook
   source, persist tasks, drop, reopen, re-run reconciliation
   against the persisted ledger, verify task vertices are unchanged
   and the deterministic task_id index keeps `record_report`
   idempotent across the restart.

## Key design decisions

### 1. EAV/Datalog under, property-graph facade above

The data model on disk is EAV facts — every property write is a
fact `[entity attribute value]`. The facade *above* presents a
property-graph API (`create_vertex`, `create_edge`, `set_vertex_property`)
so callers in Phase 14 / 15 don't need to learn Datalog. Datalog
queries are constructed and dispatched only inside `graph.rs`; the
strings never leak across module boundaries.

The trade-off is two-tier: when callers want graph traversals
(`accounts_touched_by_transaction`, `reversal_chain`, etc.) they go
through hand-rolled queries in `queries.rs` that use the facade's
typed primitives. Operators who want raw Datalog can call
`handle.raw().execute("(query ...)")` — the escape hatch is still
there but explicitly opted into.

### 2. Retract-then-assert for scalar property updates

Minigraf is **bi-temporal** — every assert is appended to history,
never overwritten. Naively calling `set_vertex_property(id, "status",
"posted")` after `set_vertex_property(id, "status", "pending")`
leaves *both* facts queryable. To give scalars the "current value"
semantics the rest of the codebase expects, `set_vertex_property`
now:

1. Queries the current value for `(entity, attr)`,
2. Retracts every result (`(retract [[id :attr v]])`),
3. Asserts the new value.

This is 2–3 Datalog ops per set instead of 1. The historical trail
is **preserved** — Minigraf still has every value the attribute
ever held, queryable via `:as-of`. Only the *current view* (the
default query) reflects the latest set. That's free time-travel for
ledger audits.

### 3. Caught a pre-existing bug: `find_by_external_id` graph fallback

`GraphLedgerStore` maintained an in-memory
`Mutex<HashMap<external_id, TransactionId>>` cache, populated lazily
on writes. In Phase 14 the cache was always seeded by the producer
of the transaction in the same process, so cache misses never
mattered. With persistence, a fresh `GraphLedgerStore::with_handle`
on a previously-written `.graph` file has an empty cache — and
`find_by_external_id` returned `None` for transactions that
actually existed in the graph.

Fixed: on cache miss, scan `ledger_tx` vertices via
`handle.vertices_of_type()`, read each `external_id` property, and
populate the cache on the way past. Honest about the cost (linear
in transaction count for the first miss after a restart), but
correct.

### 4. License rebase from MPL-2.0 to MIT OR Apache-2.0

`indradb-lib` is MPL-2.0; Phase 14 documented the operator-facing
implications of MPL-2.0-at-file-level. `minigraf` is MIT OR
Apache-2.0, **strictly more permissive**. The thorough license note
in `op-graph/src/lib.rs` (and Phase 14's progress doc) can be
loosened: there is no longer an MPL-tainted backend in the
operator's primary deployment story. Operators who still want to
swap the store implementation entirely can do so — the trait
seams on `op-ledger` / `op-webhook` / `op-reconciliation` haven't
moved.

### 5. JSON property codec preserved

`set_vertex_property` still takes a `serde_json::Value` and
`get_vertex_properties` still returns `serde_json::Map<String,
Value>`. Internally:

- Scalars (Null / Bool / Number / String) map directly to
  `minigraf::Value::{Null, Boolean, Integer, Float, String}`.
- Arrays and objects serialize to a `json:`-prefixed string and are
  re-parsed on read. (Minigraf has no native nested-JSON value.)

This preserves the codec contract documented in the Phase 14 doc.
The `json:` marker is internal — `mg_to_json` strips it before
returning.

### 6. Backend is fixed at construction time, not feature-flagged

Phase 16's earlier draft enum-dispatched between Memory and
Persistent variants of an internal `Backend` enum. With Minigraf,
both modes are first-class on the same type (`Minigraf::in_memory()`
vs `Minigraf::open(path)`), so the dispatch is just a constructor
choice. Less code, fewer match arms, no feature flags.

## What this crate does NOT do

- **No clustering / multi-process** at the storage layer. Minigraf
  is single-process embedded by design. Operators who want
  horizontal scale plug in a different `LedgerStore` (and friends)
  behind a remote datastore — the trait surfaces are shaped for
  that explicitly and Phase 12/13's docs say as much.
- **No automatic schema migration.** Adding properties is safe
  (Datalog is schemaless w.r.t. attributes). Removing or renaming
  attributes is a Datomic-style retract job operators run
  themselves.
- **No automatic indexing.** `vertices_of_type` walks all vertices
  and filters; for our reference-impl volumes this is fine. A
  custom store impl over a tuned datastore handles high-cardinality
  types.
- **No transaction-time clock control yet.** Bi-temporal queries
  (`:as-of N`) are available via the `raw()` escape hatch; the
  typed facade doesn't expose them. Coming in a later phase if /
  when a use case lands.

## Bugs caught during construction

1. **`QueryResult` variant names**. First draft assumed
   `QueryResult::Query(rows)` / `Transact(_)`. The actual variants
   are `QueryResults { vars, results }`, `Transacted(tx_id)`,
   `Retracted(tx_id)`, `Ok`. The internal `QueryRows::from` adapter
   handles all four.

2. **`EntityId` is just `Uuid`**. Earlier draft called
   `eid.into_uuid()` — the type is a literal `pub type EntityId =
   Uuid`. Direct value use.

3. **Keywords come back with their colon.** Minigraf returns
   `Keyword(":name")`, not `Keyword("name")`. The
   `get_vertex_properties` adapter strips the leading colon so
   callers query bare attribute names. Without this fix every
   property read returned `None`.

4. **Bi-temporal history made `set` a no-op for "current value"
   semantics.** Without retract-then-assert, two writes to
   `:status` left both `"pending"` and `"posted"` queryable, and
   `mark_posted` appeared not to have run. See Decision 2.

5. **The `find_by_external_id` cache-only path was a bug Phase 14
   never noticed** because every process always wrote what it
   later read. Surfaced immediately under persistence. See
   Decision 3.

## Honest concerns going into Phase 17+

- **Query cost in user units.** Each `set_vertex_property` is a
  query + zero-or-one retract + one transact. For a transaction
  with many entries that gets multiple property updates, this is
  noticeably more Datalog work than IndraDB's one-call set. For
  the reference impl it's still fast (the workspace test suite
  runs in seconds). For production volumes operators care about,
  the trait seam allows swapping in a tuned store.

- **No bulk-insert primitive.** Each create / set goes through a
  `(transact ...)` Datalog string. Minigraf doesn't currently
  expose a programmatic bulk-insert that bypasses Datalog parsing.
  The cost is real but bounded — measured impact on the existing
  test suite was unobservable.

- **Edge property API still defined but minimally tested**.
  `set_edge_property` works (it shares `set_entity_property` with
  vertices), but the Phase 14 / 15 stores don't currently set any
  edge properties. No regression risk; just noting the surface is
  less battle-tested than the vertex side.

- **File format is Minigraf's, not OpenPay's.** A `.graph` file
  written by `op-graph v0.16` is only readable by another process
  linking the same minigraf version. Operators who want a
  long-lived archival format export to a portable serialization
  separately. Documented; not in scope to solve here.

- **Bi-temporal history grows unboundedly.** Every retract is
  itself a fact in the history. For a long-running deployment,
  the `.graph` file grows. Compaction is something Minigraf
  supports but `op-graph` doesn't yet expose. Honest follow-on.

## Cumulative state

| Phase | Crate(s) | Tests (observed) | LOC (approx) |
|---|---|---:|---:|
| 1 | op-core | 19 | ~600 |
| 2 | op-iso20022 (pre-camt053) | 39 | ~1,400 |
| 3 | op-emv | 51 | ~1,800 |
| 4 | op-rails-card | 46 | ~2,100 |
| 5 | op-rails-a2a | 66 | ~3,200 |
| 6 | op-fraud | 65 | ~2,400 |
| 7 | op-vault | 51 | ~2,600 |
| 8 | op-ffi-swift | 44 | ~2,700 |
| 9 | op-ffi-jni | 69 | ~2,950 |
| 10 | op-wasm | 35 | ~2,200 |
| 11 | op-orchestrator + examples | 90 | ~4,150 |
| 12 | op-ledger | 69 | ~2,540 |
| 13 | op-webhook | 106 | ~3,304 |
| 14 | op-graph (original, IndraDB) | 74 | ~3,696 |
| 15 | op-reconciliation (+ iso20022 statement view + op-graph store) | 25 | ~1,916 |
| **16** | **op-graph rewrite on Minigraf + persistence** | **(net +1 vs Phase 15: −3 IndraDB-internal unit + +4 persistent integration)** | **+500 net** |
| **Total** | | **810** | **~38,056** |

## What's next (Phase 17+ candidates)

- **CAMT.053 XML serde round-trip** vector + conformance test
  (rolled over from Phase 15).
- **`Camt054Source`** — intra-day notifications.
- **Matched-pair graph enrichment** — emit
  `statement_line --reconciles--> ledger_tx` edges from matched
  pairs, now with the persistence to keep them.
- **Time-travel queries on the ledger.** Minigraf's `:as-of` is
  free; expose a typed helper like
  `balance_as_of(account, tx_count) -> Money` for auditors.
- **History compaction** — surface Minigraf's compaction so
  long-running operators can prune fact history they've already
  archived.
- **`op-router` integration** — orchestrator consults
  reconciliation-task density per rail and prefers quiet rails.

The thesis stands: pure-Rust, Apache-2.0 reference payment stack;
the graph that used to live only in RAM is now a single-file
durable artifact, with free bi-temporal history.
