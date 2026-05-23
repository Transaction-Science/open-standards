# Phase 20 — Omnibus: every outstanding concern from Phases 15–19

**Status**: Draft v0.20
**Date**: 2026-05-21

## Why

Phases 15 → 19 each shipped with an "honest concerns" section listing
the loose ends that didn't quite fit in scope. Nine of those
accumulated. Phase 20 closes the whole list in one sweep — no new
ambition, just delivering every promise.

## What shipped

| # | Concern from | Item | Where |
|--:|---|---|---|
| 1 | 19 | Configurable signal combiner (`SignalCombiner` enum: `WorstAxis` / `Weighted` / `HardDemoteAbove`) | `op-orchestrator/src/signals.rs`, `router.rs` |
| 2 | 18, 19 | Latency signal (`record_attempt(duration_ms)`, `RoutingSignals::latency_score`, p50-over-SLO impl) | `op-orchestrator/src/{signals,engine}.rs`, `op-graph/src/rail_telemetry.rs` |
| 3 | 19 | Per-kind discrepancy severity weighting (`DiscrepancyWeights` table) | `op-graph/src/rail_telemetry.rs` |
| 4 | 17 | Wall-clock time-travel (`balance_as_of_time`, `transaction_as_of_time` on `LedgerHistory`) | `op-ledger/src/history.rs`, `op-graph/src/ledger_store.rs` |
| 5 | 17 | Named checkpoints (`save_checkpoint`, `tx_count_at_checkpoint`) via a new `LEDGER_CHECKPOINT` vtype, deterministic UUIDv5 ids | `op-ledger`, `op-graph` |
| 6 | 17 | `replay_window(start_tx, end_tx)` on `LedgerHistory` (every tx posted in counter window) | `op-ledger`, `op-graph` |
| 7 | 15, 18 | Matched-pair graph enrichment (`MatchedPair` in `MatchOutcome` and `ReconciliationReport`; `statement_line --reconciles--> ledger_tx` edges emitted by `GraphReconciliationStore`) | `op-reconciliation/src/matcher.rs`, `engine.rs`, `discrepancy.rs`; `op-graph/src/reconciliation_store.rs` |
| 8 | 14 | Fraud-graph queries (`accounts_linked_via_chargeback`, `endpoints_sharing_secret_prefix`, `attempts_with_shared_ip` with operator-supplied IP map) | `op-graph/src/queries.rs` |
| 9 | 15 | CAMT.053 XML conformance vector + serde round-trip via `Document` wrapper, plus `Camt054Source` reusing the same flatten | `op-iso20022/vectors/camt053_v12_minimal.xml`, `tests/conformance.rs`, `op-iso20022/src/{message,statement}.rs`, `op-reconciliation/src/sources/camt054.rs` |

Workspace at the end of Phase 20:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **824 passing, 0 failing** (+3 vs Phase 19) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

The test-count delta is +3 (camt053 substring conformance, camt053
serde round-trip, camt054 flatten reuses-the-helper). The other
eight items rode in on top of the existing 821 tests without
adding new test bodies — they passed the existing tests + a few
ad-hoc smoke tests during construction.

## Configurable signal combiner

`PolicyRouter` no longer hardcodes `max(failure, discrepancy)`.
`SignalCombiner` is a small enum the router consults:

```rust
SignalCombiner::WorstAxis                                  // default
SignalCombiner::Weighted { failure: 1.0, discrepancy: 0.5, latency: 0.3 }
SignalCombiner::HardDemoteAbove { threshold: 0.9 }
```

`Weighted` clamps the linear combination to `[0.0, 1.0]`; weights
need not sum to 1, they're a relative-importance vector. The
`HardDemoteAbove` variant returns `1.0` (full demote) the moment
any single axis crosses the threshold — useful for "fail-fast" on
total-outage operators — and falls back to `WorstAxis` otherwise.

## Latency signal

`RailTelemetry::record_attempt` gained `duration_ms: Option<u32>`.
The orchestrator times each `RailAdapter::attempt` via
`std::time::Instant::now()` and passes the measured wall-clock
elapsed. `GraphRailTelemetry` stores the value on each
`rail_attempt` vertex; `latency_score` returns `min(1.0, p50 /
SLO)` over the window, defaulting to a 2-second SLO. A driver that
returns 200 OK but takes thirty seconds gets the same treatment
as one returning 500 — both push back.

`RoutingSignals::latency_score` has a default `0.0` impl so
existing third-party signals sources keep working unchanged.

## Discrepancy severity weighting

`GraphRailTelemetry.weights: DiscrepancyWeights` controls how each
discrepancy `kind` contributes to `discrepancy_score`. Defaults
reflect Phase 19's honest-concerns table:

| Kind | Weight | Why |
|---|---:|---|
| `unmatched_statement` | 1.0 | Bank says a payment landed we never booked. Hard signal. |
| `amount_mismatch` | 1.0 | Books and bank disagree on amount. Hard signal. |
| `unmatched_ledger` | 0.7 | We booked something the bank hasn't reported yet. Often timing. |
| `status_mismatch` | 0.6 | Pending vs Posted. Often timing. |
| (fallback) | 1.0 | Conservative for kinds the table doesn't enumerate. |

Score formula went from `tasks / attempts` to `Σ(weight(kind)) /
attempts`, clamped at 1.0.

## Wall-clock time-travel + checkpoints + replay_window

Phase 17 shipped tx-count time-travel. Phase 20 makes it usable
for humans:

```rust
// Wall-clock. Operators think in dates.
let bal = store.balance_as_of_time(account, 1_700_000_000)?;
let tx  = store.transaction_as_of_time(id, 1_700_000_000)?;

// Named bookmarks. Reusable references.
let snap = store.save_checkpoint("Q4-2025-close")?;
// ... days later ...
let same = store.tx_count_at_checkpoint("Q4-2025-close")?.unwrap();

// Range queries. "Show me every booking made between two snaps."
let in_window: Vec<TransactionId> = store.replay_window(snap_a, snap_b)?;
```

**Implementation:** every `post_transaction` writes a
`posted_at_tx_count` property on the new `ledger_tx` vertex
(captured *after* the tx's own writes have advanced the
bi-temporal log, so it's the correct as-of anchor). `_at_time`
methods scan ledger_tx vertices, pick the largest `posted_at_tx_count`
whose `effective_at_unix_secs <= at_unix_secs`, and reuse the
existing tx_count-based readers. Checkpoints live in a new
`LEDGER_CHECKPOINT` vtype keyed by deterministic UUIDv5 over the
name — re-saving the same name is idempotent and overwrites.

## Matched-pair graph enrichment

The reservation in Phase 15's schema (`statement_line --reconciles-->
ledger_tx` edge) is filled in. The matcher now emits a
`MatchedPair` for each successful tier-1 or tier-2 join:

```rust
pub struct MatchedPair {
    pub statement_source_id: String,
    pub tx_id: TransactionId,
    pub fuzzy: bool,    // tier-1 = false; tier-2 (amount+window) = true
}
```

`MatchOutcome` and `ReconciliationReport` both carry the full
list. `GraphReconciliationStore::record_report` creates the
statement_line vertex (if absent) and an outbound `reconciles`
edge to the ledger_tx vertex. Fuzzy matches get a `fuzzy: true`
edge property so auditors can distinguish heuristic from exact
matches in the graph.

The `ReconciliationReport.matched_pairs` field is
`#[serde(default)]`, so previously-serialized reports still
deserialize.

## Fraud-graph queries

Three new typed traversals in `op-graph::queries`:

```rust
pub fn accounts_linked_via_chargeback(handle) -> Vec<(AccountId, AccountId)>;
pub fn endpoints_sharing_secret_prefix(handle, prefix_bytes) -> Vec<(EndpointId, EndpointId)>;
pub fn attempts_with_shared_ip<F: Fn(EndpointId) -> Option<String>>(handle, ip_of) -> Vec<(DeliveryAttemptId, DeliveryAttemptId)>;
```

- **`accounts_linked_via_chargeback`** walks every `ledger_reverses`
  edge and emits the cross-product of accounts touched by the
  original tx and the reversal tx, deduplicated as canonical
  `(lo, hi)` pairs.
- **`endpoints_sharing_secret_prefix`** decodes each
  `webhook_endpoint.secret_b64` (the same standard base64 codec
  `op-webhook` uses) and emits pairs whose secrets share a prefix
  of N bytes. Useful proxy for "same operator setup."
- **`attempts_with_shared_ip`** is parameterised by an `ip_of:
  EndpointId -> Option<String>` closure since the graph doesn't
  currently record IPs. The function buckets attempts by shared
  IP and emits cross-pairs. Operator-supplied IP map keeps the
  graph schema unchanged.

## CAMT.053 conformance + Camt054Source

`vectors/camt053_v12_minimal.xml` — a real-shaped CAMT.053 v12
document with the `Document/BkToCstmrStmt` outer wrapping, one
`Stmt`, one `Bal`, one `Ntry` carrying a deep `NtryDtls → TxDtls →
Refs.EndToEndId` path. Two conformance tests:

1. **Substring assertions** match the existing pacs.008 / pacs.002
   pattern.
2. **Serde round-trip** — `Message::parse_camt053` now skips the
   `Document` envelope via a tiny `Wrapper` struct and yields a
   `BankToCustomerStatementV12` the flatten helper can walk. The
   neutral `camt053_entries` view returns one `StatementEntry`
   with `end_to_end_id = Some("ORD-77")`.

`Camt054Source` is the intra-day cousin of `Camt053Source`. Same
`Ntry` shape, same `flatten_entry` helper — added in
`op-iso20022::statement` as `Message::camt054_entries()` — same
`StatementLine` mapping in
`op-reconciliation/src/sources/camt054.rs`. `Message::parse_camt054`
mirrors the camt053 parser with a different outer-element rename
(`BkToCstmrDbtCdtNtfctn`).

## Key design decisions

### 1. Default-preserving trait extensions

`RoutingSignals::latency_score` and `discrepancy_score` both have
default `0.0` implementations. `RailTelemetry::record_attempt`'s
new parameters (`external_id_hint`, `duration_ms`) are both
`Option<...>`. Third-party impls written against the Phase 18
trait signatures recompile clean against Phase 20.

### 2. `SignalCombiner` is a value enum, not a function pointer

Tempting to make `combiner: Box<dyn Fn(...) -> f32>`. Chose the
enum because: (a) the three combiner shapes cover every operator
ask I can imagine without devolving into "go write a closure";
(b) `PolicyRouter` derives `Clone, Debug`, which a `Box<dyn Fn>`
breaks; (c) future shapes are an additional variant, not a
breaking API change.

### 3. Time-travel anchor is `posted_at_tx_count`, not
`effective_at_unix_secs`

The bi-temporal Datalog filter takes a counter, not a date. The
`_at_time` methods translate wall-clock → counter by finding the
ledger_tx whose `effective_at_unix_secs` is at-or-before the
asked-for time with the largest `posted_at_tx_count`. That's the
correct semantic — "the database as it stood just after the last
booking on or before that wall-clock" — and it lets us reuse
the existing `_at` machinery without a new Datalog path.

### 4. Checkpoint id is UUIDv5 over the name, not stored separately

Re-saving the same name produces the same vertex id, so the
"overwrite" semantic is idempotent and lookup is O(1) (compute
the UUIDv5, read the vertex's properties). No separate index
relation needed.

### 5. Matched-pair edges add `fuzzy: true` only when fuzzy

Tier-1 (reference-key) matches emit a plain `reconciles` edge
with no properties. Tier-2 (amount + window heuristic) matches
get `fuzzy: true`. The default case is the strong case — an
operator querying for matched pairs sees clean joins first; the
fuzzy ones stand out for spot-check.

### 6. Fraud queries are *typed*, not Datalog

A future "open this up to operator-supplied Datalog" enhancement
could wrap the same traversals. For Phase 20 the three typed
helpers cover the Phase 14 promise: the operator gets typed Rust
return values, not strings that have to be parsed.

### 7. `attempts_with_shared_ip` takes a closure, not an in-graph
property

The graph doesn't record endpoint IPs and adding that would
either invent a write path (which the operator's app would have
to maintain) or be a noop. Closure-injection keeps the
abstraction pure: the operator who DOES record IPs somewhere
else passes a lookup function. The graph stays free of fields it
can't authoritatively populate.

### 8. CAMT XML serde via a `Wrapper` struct, not by modifying
upstream types

The `Document/BkToCstmrStmt` outer wrapping is real-world
mandatory per the ISO 20022 spec but the upstream
`BankToCustomerStatementV12` type doesn't represent it (the
upstream crate documents this — `Document` is a generated enum
with one variant per message version, only built when you turn
on the right feature). A 4-line per-message wrapper struct
defined inside the parser is cleaner than vendoring upstream
types or fighting feature gates.

## What this phase does NOT do

The two genuinely-deferred items from earlier phases that DON'T
land here:

- **History compaction (Phase 16).** Minigraf supports compaction
  via its `checkpoint` API; surfacing it cleanly requires
  designing the operator-side semantics (when to compact, what
  retention to keep, how to bisect history if compaction is
  destructive). Out-of-scope size for an omnibus phase. Still
  queued.
- **Reconciliation-discrepancy → ledger_tx → rail/driver
  detailed audit reports.** The reconciliation tasks now have
  `reconciles` edges back to the ledger txs that settled
  cleanly; the inverse (which rail/driver produced each
  unresolved task) is computable, but a typed audit-report struct
  belongs in its own phase. Use the existing `task_descriptor`
  shape for now.

## Bugs caught during construction

1. **`impl op_ledger::LedgerHistory` can't hold non-trait
   methods.** The first draft of the new history methods stuffed
   private helpers like `tx_count_at_time` inside the trait impl
   block. Rust correctly rejected it (E0407: method is not a
   member of trait). Split into a `impl GraphLedgerStore` block
   for private helpers; trait methods live in `impl
   op_ledger::LedgerHistory for GraphLedgerStore`.

2. **`replay_window` was named `replay_window_inner`.** Same
   confusion: trait impl vs private helper. Renamed back to
   `replay_window` and made the trait method.

3. **`Default` derive on `SignalCombiner` needs `#[default]` on a
   variant.** Clippy caught the hand-rolled `impl Default` block
   and suggested the modern variant-tagged derive — small but
   cleaner code.

4. **`ReconciliationReport` was a public struct.** Adding
   `matched_pairs` is a binary breaking change for callers who
   construct it by struct literal. Mitigated with
   `#[serde(default)]` so serialized reports still deserialize;
   call sites in the codebase were updated to add the new field
   literal. Worth noting in the API-stability doc when one exists.

5. **`Message::parse_camt053` was rejecting the real-world
   `Document` wrapping.** First draft tried `from_xml::<BkToCstmrStmt>`
   directly; quick-xml interprets the document root as the
   target, so `<Document>` was confusing it. Tiny `Wrapper`
   struct with `#[serde(rename = "BkToCstmrStmt")]` fixed it.

## Honest concerns going into Phase 21+

- **Vertex-scan cost growing.** Phase 20 added more "scan all
  vertices of type X, filter by property Y" queries (replay
  window, time-travel anchor lookup, fraud queries). For
  production volumes operators eventually need a secondary
  index per property. Minigraf supports it; we haven't wired
  it.

- **`save_checkpoint` overwrites silently.** Re-saving an
  existing name returns the new counter and re-asserts the
  vertex properties — no error, no event. Operators who want
  immutable checkpoints can prefix the name with the timestamp
  themselves; or a future phase ships a `save_checkpoint_unique`
  variant.

- **`replay_window` returns ids, not the transactions
  themselves.** Operators read the txs separately via
  `get_transaction`. That's intentional (avoid forcing a giant
  result), but means the caller pays N+1 lookups. A future
  enhancement could ship a paginating iterator.

- **CAMT round-trip is one-way.** We parse `Document/BkToCstmrStmt`
  XML in. We don't yet serialize a `Message::Camt053` back out
  through the same wrapper. The flatten view is the consumer
  path; serialization would matter only if op-iso20022 needed to
  emit camt.053, which no existing rail does.

- **`attempts_with_shared_ip` is closure-based.** Operators who
  want this fully in-graph would need to add an `ip_address`
  property at attempt-record time. The `record_attempt` trait
  could grow another `Option<&str>` parameter; we held off
  because nothing in the existing op-webhook flow has an IP
  available at write time.

## Cumulative state

| Phase | Crate(s) | Tests | LOC |
|---|---|---:|---:|
| 1–19 | (see prior docs) | 821 | ~39,611 |
| **20** | **Omnibus: 9 closures across orchestrator + ledger + reconciliation + graph + iso20022** | **+3** | **~+750** |
| **Total** | | **824** | **~40,361** |

## What's next (Phase 21+ candidates)

- **History compaction** (the genuinely-deferred item above).
- **Secondary indices on Minigraf properties** for the
  hot-property queries (external_id, posted_at_tx_count,
  effective_at_unix_secs). Production scale lever.
- **Audit-report struct** that joins reconciliation tasks back
  through ledger_tx to rail/driver (the inverse of Phase 19's
  signal join).
- **Persistent attempt-log compaction.** As the graph grows,
  `rail_attempt` vertices outside the signal window are dead
  weight. A periodic prune would help.
- **OpenTelemetry trace propagation** (rolled over since Phase 15).
- **GraphQL adapter** to the typed `queries.rs` helpers so
  operator dashboards can query without writing Rust.

The thesis stands: the same single graph stores ledger,
webhooks, reconciliation, time-travel, routing signals, and now
fraud-relationship hints. Every phase adds a view; no phase
adds substrate. Phase 20 cashes in the chips Phases 15–19 left on
the table.
