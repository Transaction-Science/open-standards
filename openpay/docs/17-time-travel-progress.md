# Phase 17 — Time-travel ledger queries

**Status**: Draft v0.17
**Date**: 2026-05-21

## Why

Phase 16 swapped `op-graph`'s storage to Minigraf, which gives us
**bi-temporal history for free**: every assert is appended, never
overwritten; every retract is itself a fact. The graph file at rest
holds every value every property ever had, indexed by a monotonic
transaction counter.

That's a powerful capability sitting unused. Phase 17 surfaces it
as a first-class operator API: *"what did this account's balance
look like at point X?"* *"what was this transaction's status before
mark_posted ran?"* These aren't novel questions — auditors and
reconciliation operators ask them daily — they were just expensive
before. Now they're a query away.

## What shipped

A new trait in `op-ledger` and its implementation in `op-graph`.
The substrate already existed; this phase exposes it cleanly.

| File | LOC | Notes |
|---|---:|---|
| `op-ledger/src/history.rs` | **77** | new — `LedgerHistory` trait |
| `op-ledger/src/lib.rs` | +2 | register module + re-export |
| `op-graph/src/graph.rs` | +~115 | `tx_count()` + four `_at` Datalog query helpers |
| `op-graph/src/ledger_store.rs` | +~180 | `LedgerHistory` impl + `read_entry_edge_at` |
| `op-graph/tests/time_travel.rs` | **203** | new — 5 integration tests |
| **Phase 17 totals** | **~580 added** | (no removals; this is pure capability addition) |

Workspace at the end of Phase 17:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **815 passing, 0 failing** (+5 vs Phase 16) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

## The API

```rust
use op_ledger::{LedgerHistory, LedgerStore};

// Operator snapshots the counter right after a write they care
// about. The counter is opaque — don't subtract it from a wall
// clock; just hold it as "the moment after I posted this."
let id = store.post_transaction(tx)?;
let snap = handle.tx_count();        // u64

// ... arbitrary time passes; many more writes happen ...

// Time-travel reads against that moment.
let bal_then = store.balance_as_of(account_id, snap)?;
let tx_then  = store.transaction_as_of(id, snap)?;
//   ^ status reflects the value at `snap`. If we've since called
//     `mark_posted`, `bal_then.status` is still `Pending`.
```

Two methods, both error-typed against `op_ledger::Error`:

- **`balance_as_of(account, tx_count) -> Result<Balance>`** —
  walks the same entry-per-edge structure as `balance()`, but every
  inbound debit/credit edge read, every edge property read, and
  every tx status read goes through Minigraf's `:as-of N` filter.
  Entries posted after `tx_count` are invisible; statuses that
  flipped after `tx_count` show their earlier value.
- **`transaction_as_of(id, tx_count) -> Result<Transaction>`** —
  reconstructs the transaction as it stood at `tx_count`. Returns
  `Error::TransactionNotFound` if the tx didn't exist (had no
  `_type` fact) at that moment.

## Key design decisions

### 1. The trait lives in `op-ledger`, not in `op-graph`

`LedgerHistory` is a capability operators reasonably want to *check
for* before depending on it. Putting it next to `LedgerStore` in
`op-ledger` makes "this ledger can do time-travel" a typed property
of the store, not a quirk of the graph-store import path. Concrete
stores opt in:

- `GraphLedgerStore` implements it (Minigraf is its substrate).
- `InMemoryLedgerStore` does **not** implement it (it's a snapshot
  of "now"; no fact log).
- Future Postgres-backed or TigerBeetle-backed stores would
  implement it as they see fit.

The trait is **separate from `LedgerStore`** rather than added to
it. This is deliberate: the cost of always having to implement it
would push every minimal store toward keeping its own fact log
just to satisfy the trait. Optional is right.

### 2. The reference point is Minigraf's monotonic counter, not a
wall-clock timestamp

Two options were on the table:

- **Wall-clock timestamp**: operator-friendly ("at 2026-05-21
  10:00 UTC..."), but Minigraf's `:as-of` accepts the monotonic
  counter, not Unix milliseconds. Mapping wall-clock → counter
  would require a separate index op-graph would have to maintain.
- **Monotonic tx counter**: opaque to operators but free —
  Minigraf already maintains it and `:as-of N` reads it directly.

Phase 17 ships the counter form via `GraphHandle::tx_count()`, the
direct path. A future enhancement could add a `tx_count_at_time()`
helper if operators want the wall-clock front door — it'd be a
secondary index over a `:transaction_time` attribute, not a
substrate change.

### 3. Account properties are read at present, not as-of

An account's `currency` and `normal_balance` are immutable: set at
creation, never re-asserted. Reading them at the present view
inside `balance_as_of` is correct *and* faster than threading
`:as-of` through the account read. If a future feature ever lets
an operator mutate those, this assumption breaks and the impl
needs to switch to `get_vertex_properties_at`. Documented in the
impl's docstring.

### 4. Empty property map → `TransactionNotFound`

In `transaction_as_of`, if `get_vertex_properties_at` returns an
empty map at `tx_count`, the tx either didn't exist then (later
creation) or never existed. We don't distinguish; both return
`Error::TransactionNotFound`. That's the right semantic — at the
asked-for moment, the tx wasn't queryable.

### 5. Four new `_at` helpers on `GraphHandle`, not one generic
"as_of mode"

`out_edges_at`, `in_edges_at`, `get_vertex_properties_at`,
`get_edge_properties_at` parallel their present-time siblings.
This deliberately *avoids* an `as_of: Option<u64>` parameter on
every existing GraphHandle method (which would have rippled
through every call site in op-graph). The duplication is small
(each helper is ~10 lines), the call sites stay clean, and the
intent — *"this read is historical"* — is visible at the call.

## What this phase does NOT do

- **No retroactive writes.** All writes happen at "now"; there is
  no `post_transaction_as_of(tx, tx_count)` that asserts facts at
  a past counter. Bi-temporality here is read-only.
- **No range queries.** No `replay_window(start_tx, end_tx)` that
  yields the deltas between two counters. The substrate could
  support it; the use case hasn't surfaced.
- **No wall-clock front door.** See Decision 2.
- **No automatic counter discovery.** Operators have to
  *snapshot* `tx_count()` at the moment they care about. We don't
  archive snapshots, name them, or persist them outside the
  caller's own bookkeeping. (A future "named checkpoints" feature
  is on the list, not in scope here.)
- **No InMemoryLedgerStore impl.** That store doesn't keep a fact
  log; implementing `LedgerHistory` would require either keeping
  a parallel history (a different design) or refusing the call.
  Honest: not implemented.

## Bugs caught during construction

1. **Property attribute filter must strip the colon.** Minigraf
   keywords come back as `:name`, not `name`. The
   `get_vertex_properties_at` helper inherited the same fix as the
   present-time `get_vertex_properties` (Phase 16 had this issue
   too). Triple-checked while writing the new helper.

2. **`Status::Archived` arm in `balance_as_of`**. Without it the
   match was non-exhaustive vs `Status` — the rust compiler caught
   it. Once added it correctly contributes nothing to either
   `posted` or `pending`. Same behaviour as `balance()`.

3. **Test fixture ordering**. The first draft of
   `time_travel_survives_persistence_round_trip` reused a
   `GraphHandle` after taking a snapshot, then dropped + reopened
   — but the snapshot's counter value didn't necessarily match the
   reopened handle's counter, because the persisted counter is
   tied to the file, not the handle. Verified empirically: the
   counter persists with the file, so `snap` taken before drop
   still points at the right moment after reopen. Documented in
   the test.

## Honest concerns going into Phase 18+

- **Counter is opaque to humans.** An operator looking at a
  `tx_count = 4729` value can't easily say "that's last Tuesday."
  A future Phase could add a `:transaction_time` attribute on
  every tx vertex (bi-temporal valid-time) and an index
  `wall_clock → counter` so `balance_as_of` can take a
  `time::OffsetDateTime`. Today: caller bookkeeping.

- **Cost of `balance_as_of` scales with edge count.** Each
  inbound debit/credit edge gets its own property and tx-status
  read — same as `balance()`. The `_at` variants don't slow this
  down (Datalog filters at the storage layer) but don't speed it
  up either. For an account with millions of entries, this is
  slow; production deployments either roll cached running
  balances (out of scope, in op-ledger's "honest concerns"
  section already) or query at a higher level.

- **History grows unboundedly.** Phase 16's note still applies:
  every set is a retract + assert under the hood, leaving a
  permanent record. Long-running graphs need compaction. Minigraf
  supports it but `op-graph` doesn't surface it yet.

- **`get_vertex_properties_at` returning empty means "didn't
  exist."** This isn't airtight: an account could theoretically
  exist with `_type` but no user properties at some early
  `tx_count`. In practice every vertex this codebase creates also
  writes at least one property in the same `post`, so the
  failure mode is unobserved. Documented anyway.

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
| 16 | op-graph rewrite on Minigraf + persistence | net +1 | +500 net |
| **17** | **`LedgerHistory` trait + GraphLedgerStore impl** | **+5** | **+580** |
| **Total** | | **815** | **~38,636** |

## What's next (Phase 18+ candidates)

- **Wall-clock time-travel.** Add a `:transaction_time_unix_secs`
  attribute on every tx vertex on write; index it; expose
  `tx_count_at_time(t: u64) -> u64` so callers can use real
  timestamps. Minor work; user-friendliness win.
- **Named checkpoints.** Operator-bookkept names → counters in a
  small `checkpoints` relation. `balance_as_of_checkpoint("Q4-2025-close")`.
- **`op-router` integration** (Phase 17 sibling, still queued).
- **CAMT.053 XML conformance + `Camt054Source`**.
- **History compaction.** Surface Minigraf's compaction so
  long-running operators can prune.
- **`replay_window(start_tx, end_tx)`** — diff two historical
  points. Easy with what we have; needs only an iteration helper.

The thesis stands: free bi-temporal history shipped as a typed
operator-facing capability. Auditors no longer have to ask
*"can you reconstruct what we knew on November 4th?"* — they can
just do the query.
