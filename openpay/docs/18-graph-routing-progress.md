# Phase 18 — Graph-informed routing

**Status**: Draft v0.18
**Date**: 2026-05-21

## Why

The Phase 11 orchestrator routes statically. A `PolicyRouter`
walks a fixed list of card and A2A drivers — every intent gets the
same chain, regardless of which drivers have been crashing for the
last hour. Real merchant deployments observe this is wrong:
- PSPs go down for minutes at a time and the orchestrator keeps
  trying them anyway, eating retry budget.
- A new reconciliation discrepancy spike from one driver is a
  leading indicator we should be using before it becomes a
  customer complaint.

Phase 18 closes the loop: **the orchestrator records every
attempt; the router consults the same history before choosing.**
The substrate for both is the graph we built in Phases 14–17.

## What shipped

Two new traits on `op-orchestrator` plus a single implementation
in `op-graph` that satisfies both. Existing callers don't change:
the defaults are no-ops.

| File | LOC | Notes |
|---|---:|---|
| `op-orchestrator/src/signals.rs` | **127** | new — `RailTelemetry`, `RoutingSignals`, `AttemptResultClass`, NoOp defaults |
| `op-orchestrator/src/router.rs` | +~35 | `PolicyRouter.signals: Option<Arc<dyn RoutingSignals>>` + `with_signals(...)` builder; stable-sort drivers within each rail group by failure score |
| `op-orchestrator/src/engine.rs` | +~20 | `Orchestrator.telemetry: Arc<dyn RailTelemetry>` + `with_telemetry(...)`; call `record_attempt` after each `RailAdapter::attempt` |
| `op-orchestrator/src/lib.rs` | +6 | module + re-exports |
| `op-graph/src/graph.rs` | +5 | new vtype `RAIL_ATTEMPT` + manual `Debug` impl on `GraphHandle` |
| `op-graph/src/rail_telemetry.rs` | **235** | new — `GraphRailTelemetry` impls both traits |
| `op-graph/src/lib.rs` | +3 | module + re-exports |
| `op-graph/Cargo.toml` | +1 | `op-orchestrator` dep (downward only — no cycle) |
| `op-graph/tests/graph_routing.rs` | **185** | new — 3 cross-domain tests |
| **Phase 18 totals** | **~620** | net new |

Workspace at the end of Phase 18:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **818 passing, 0 failing** (+3 vs Phase 17) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

## The API

```rust
use std::sync::Arc;
use op_orchestrator::{Orchestrator, PolicyRouter, RailTelemetry, RoutingSignals};
use op_graph::{GraphHandle, GraphRailTelemetry};

// Shared graph: telemetry writes here, router reads from here.
let handle = GraphHandle::new_persistent("./openpay.graph")?;
let telemetry: Arc<GraphRailTelemetry> =
    Arc::new(GraphRailTelemetry::with_handle(handle));

// Router consults telemetry to re-order its driver lists.
let router = PolicyRouter::new(card_drivers, a2a_drivers)
    .with_signals(telemetry.clone() as Arc<dyn RoutingSignals>);

// Orchestrator pushes each (rail, driver, outcome) into telemetry.
let orch = Orchestrator::new()
    .with_router(Box::new(router))
    .with_telemetry(telemetry.clone() as Arc<dyn RailTelemetry>);
```

The trait surfaces are intentionally small:

```rust
pub trait RailTelemetry: Send + Sync + Debug {
    fn record_attempt(&self, rail: RailKind, driver: &str,
                      outcome: AttemptResultClass, at_unix_secs: u64);
}

pub trait RoutingSignals: Send + Sync + Debug {
    fn failure_score(&self, rail: RailKind, driver: &str) -> f32;  // [0.0, 1.0]
}
```

`AttemptResultClass` is the coarser projection of
`AttemptOutcome` the signals layer cares about: `Approved`
(rail did its job, including `RequiresAction`), `SoftFailure`
(retryable), `HardFailure` (terminal decline).

## Key design decisions

### 1. Two traits, not one

A single trait that "knows about telemetry and signals" would
force every implementor to do both — but the router doesn't write
and the orchestrator doesn't read. Splitting the surfaces lets
each crate depend on exactly the half it needs (`Router` only
sees `RoutingSignals`; `Orchestrator` only sees `RailTelemetry`).
Implementors of both pass the same `Arc` into both positions —
that's how `GraphRailTelemetry` closes the loop.

### 2. Default `NoOp` impls

`Orchestrator::new()` returns an orchestrator with
`NoOpRailTelemetry`; existing Phase 11 callers keep working
unchanged. Operators opt in by calling `.with_telemetry(...)` and
`.with_signals(...)`. This is the same default-then-opt-in
pattern the existing `IdempotencyStore` / `CircuitBreaker` /
`Scorer` builders follow.

### 3. Stable sort within rail groups; rail order is policy

The router still chooses Card-vs-A2A order by **policy**
(customer hint, amount threshold, country). Signals only re-arrange
*within* each rail's driver list. So an operator who configured
`a2a_above_minor_units` still gets A2A first for large intents
even if every A2A driver has a high failure score — the policy
choice is preserved; the *driver* choice within that rail is
informed.

Within a rail group, the sort is by `failure_score` ascending,
ties broken by operator-configured index. Without signals, the
sort is a no-op (every score equals 0.0 from `NoOpRoutingSignals`).

### 4. Storage as the substrate; not a global state

`GraphRailTelemetry` writes one `rail_attempt` vertex per
attempt and computes scores by scanning recent vertices. That
means:

- **No in-memory state**. Multiple `GraphRailTelemetry` instances
  pointing at the same `GraphHandle` see the same data.
- **Persistence across restarts**. Open a persistent handle and a
  daemon restart doesn't wipe the "this PSP has been flaky for an
  hour" signal. The third integration test exercises this.
- **Free historical inspection.** Operators can query the
  `rail_attempt` vertices directly to audit which rails were
  preferred when.

Trade-off: every `record_attempt` is a write through Minigraf
(four `set_vertex_property` calls = four retract-then-assert
ops). At present scales (single-digit attempts per intent) this
is negligible; for high-throughput deployments operators ship a
buffered impl or a sampling impl.

### 5. Sliding window, configurable

Default window is one hour (`DEFAULT_WINDOW_SECS = 3_600`). Long
enough that one bad attempt doesn't push a driver to the back
permanently; short enough that a recovered PSP gets back in
rotation quickly. `GraphRailTelemetry::with_window_secs(...)`
overrides for tests and for operators with different tolerance.

### 6. `failure_score = 0.0` for empty window means "no signal,
no preference"

The router treats a zero score as "no information," not "this
driver is great." That's the semantically correct fallback: a
new driver with no history should slot into its operator-
configured position, not jump to the front because nothing has
gone wrong yet. The stable sort on tied scores enforces this.

### 7. `record_attempt` is fire-and-forget

The trait returns nothing. `GraphRailTelemetry::record_attempt`
swallows any backend errors silently — a telemetry write failure
must not crash a payment flow. A future enhancement could add an
out-of-band diagnostic channel; today, the cost of error
visibility is the cost of writing wrong code in the payment
path, and we err on payment-stays-up.

## Cross-domain integration test

`crates/op-graph/tests/graph_routing.rs` proves the loop closes
end-to-end. Three tests:

1. **`router_reorders_to_prefer_quiet_driver_after_failure_signal_is_recorded`**
   - Static config: `card_drivers = [noisy_psp, quiet_psp]`.
   - First intent: chain is `[noisy, quiet]` (no history, scores
     0/0, stable order). Noisy returns `SoftFailure`, fallback to
     quiet which `Success`es. **Two attempts.**
   - Signals now report `noisy = 1.0`, `quiet = 0.0`.
   - Second intent: chain is re-ordered to `[quiet, noisy]`.
     Quiet succeeds immediately. **One attempt.**

2. **`empty_history_leaves_static_chain_order_intact`** —
   ensures the stable-sort tie-break preserves the operator's
   configured preference when there's no signal data.

3. **`telemetry_history_persists_across_handle_reopen`** —
   records two soft failures, drops the handle, reopens at the
   same path, queries — the score is still `1.0`. Operators
   don't lose rail-health knowledge across daemon restarts.

## What this phase does NOT do

- **No exponential decay** on the failure score. An attempt 59
  minutes ago counts the same as one 30 seconds ago, then drops
  off the cliff at 60 minutes. A future enhancement adds a
  weighted score.
- **No per-rail / per-amount feature engineering.** Scores are
  scalar by `(rail, driver)`, not per intent characteristics
  (amount tier, country, customer cohort). Operators wanting
  that ship their own `RoutingSignals` impl that consults
  whatever they care about.
- **No write-through to the existing circuit breaker.** The
  `CircuitBreaker` interface tracks consecutive failures per
  rail/driver and short-circuits when the trip threshold is hit;
  this phase's telemetry / signals is a *complementary* mechanism
  (soft preference re-ordering, not hard blocking). The two work
  together: signals re-rank quiet drivers above noisy ones, and
  the breaker still hard-stops if a driver crosses the trip
  threshold.
- **No reconciliation-task density signal.** The trait could feed
  off Phase 15's reconciliation tasks (high task count → high
  signal score) but doesn't yet. The `GraphRailTelemetry` impl
  only consults `rail_attempt` vertices. Joining
  `reconciliation_task` → ledger tx → rail/driver is a Phase 19+
  enhancement.
- **No in-memory `RailTelemetry` impl.** Tests can use
  `GraphRailTelemetry::new_in_memory()`; operators with no graph
  dep ship their own. We don't ship a tiny ring-buffer impl
  because the natural place for it is alongside the graph store
  anyway.

## Bugs caught during construction

1. **`AttemptOutcome::Approved` doesn't exist.** First draft of
   `AttemptResultClass::classify` assumed `Approved` was a
   variant. The real variants are `Success`, `HardDecline`,
   `SoftFailure`, `RequiresAction`. Mapped both `Success` and
   `RequiresAction` to `Approved` from the signals standpoint —
   `RequiresAction` means the rail did its job and now the
   customer is in play, which from the rail-health view is a
   non-event.

2. **Trait objects need `Debug` for derived `Debug` on holders.**
   `PolicyRouter` derives `Debug`, so its
   `Option<Arc<dyn RoutingSignals>>` field required
   `RoutingSignals: Debug` as a supertrait. Added to both new
   traits.

3. **`GraphHandle` didn't derive `Debug`.** Earlier phases never
   needed it because no struct that held a `GraphHandle` was
   `Debug`. `GraphRailTelemetry` derives Debug, breaking the
   chain. Added a manual `Debug` impl on `GraphHandle` that
   prints the tx counter (informative without a query cost).

4. **`RailAdapter` doesn't have a `supports` method.** First
   draft of the test adapter implemented `supports` per a faulty
   recollection. The actual trait surface is `driver()`, `rail()`,
   `attempt()`. Removed.

## Honest concerns going into Phase 19+

- **Score-only signal is one-dimensional.** Two drivers with the
  same failure rate look identical to the router, even if one
  fails *fast* (transport timeout, recoverable) and the other
  fails *slow* (PSP keeps the request open for 60s). A future
  signal could include latency.

- **Sliding window on `rail_attempt` vertex scan.** Per
  `failure_score`, we walk every `rail_attempt` vertex in the
  graph and filter by timestamp. For a long-running operator
  this is O(historical_attempts), not O(window). Index by
  timestamp via a secondary attribute when scale demands it; for
  now, history compaction (Phase 16 "honest concerns") would
  also help.

- **Telemetry vs circuit breaker semantics.** Soft preference
  vs hard block. The circuit breaker already handles
  consecutive-failure trip. The signals re-order driver
  preference *before* the breaker would trip — a smoother
  experience. But operators have to reason about both
  mechanisms; documentation could explain the layering
  better. (Maybe phase doc is the right place; noted.)

- **No retroactive signals from reconciliation discrepancies.**
  The most production-valuable signal — "this rail has unmatched
  statement lines, the rail is bad" — needs a join from
  reconciliation tasks back through ledger txs to the rail used.
  Phase 18 doesn't build that join; the substrate to do it is
  in place (matched-pair edges in op-graph are *reserved* in
  the schema, see Phase 15 honest concerns).

## Cumulative state

| Phase | Crate(s) | Tests (observed) | LOC (approx) |
|---|---|---:|---:|
| 1–17 | (see prior docs) | 815 | ~38,636 |
| **18** | **op-orchestrator signals + op-graph rail_telemetry** | **+3** | **+620** |
| **Total** | | **818** | **~39,256** |

## What's next (Phase 19+ candidates)

- **Reconciliation-density signal.** Wire `RoutingSignals` to
  consult `reconciliation_task` vertex density per rail/driver,
  not just `rail_attempt` outcomes. The biggest production
  payoff once it exists.
- **Wall-clock time-travel** (still queued from Phase 17).
- **Latency signal.** Records the attempt duration alongside
  outcome; signals score becomes a weighted combination of
  failure rate and latency.
- **History compaction** (rolled over from Phase 16).
- **CAMT.053 XML conformance + `Camt054Source`** (still queued
  from Phase 15).

The thesis stands: the same graph stores the ledger, the
webhooks, the reconciliation tasks, the time-travel history, and
now the rail-health signal — one substrate, many views,
operator-driven routing decisions informed by the system's own
observed reality.
