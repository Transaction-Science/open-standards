# Phase 19 — Reconciliation-density routing signal

**Status**: Draft v0.19
**Date**: 2026-05-21

## Why

Phase 18's `failure_score` only catches drivers that fail at the
HTTP edge: timeouts, 5xx, transport errors. A worse class of
failure is **silent**: the adapter happily returns 200, the
operator books the ledger transaction, and then — days later — the
bank statement disagrees. PSP fee netting wrong, settlement amount
drift, double-credit on a refund. These are the discrepancies
Phase 15 already classifies into typed `reconciliation_task`
vertices.

Phase 19 makes the router *use* those tasks. The signal:

```text
recent rail_attempts(rail, driver)
  ↳ ledger_tx whose external_id matches an attempt's idempotency key
      ↳ inbound task_about edges = how many discrepancies touch this work
```

A driver whose transactions accumulate discrepancies gets pushed
back in the fallback chain — even though every one of its HTTP
attempts returned OK.

## What shipped

A new method on `RoutingSignals`, an extended `RailTelemetry`
signature, and the join logic in `GraphRailTelemetry`. The router
combines both scores; the orchestrator threads the idempotency
key through as a correlation token.

| File | LOC | Notes |
|---|---:|---|
| `op-orchestrator/src/signals.rs` | +25 | `RoutingSignals::discrepancy_score` (default 0.0); `RailTelemetry::record_attempt` gains `external_id_hint: Option<&str>` |
| `op-orchestrator/src/router.rs` | +20 | `combined_score = max(failure, discrepancy)`; PolicyRouter sort uses it |
| `op-orchestrator/src/engine.rs` | +3 | pass `intent.idempotency_key.as_str()` to `record_attempt` |
| `op-graph/src/rail_telemetry.rs` | +90 | store `external_id_hint` property; new `discrepancy_score` + `discrepancy_score_at` impls; `RailAttemptRecord.external_id_hint` field |
| `op-graph/tests/graph_routing.rs` | +2 | update Phase 18 test to new `record_attempt` signature |
| `op-graph/tests/discrepancy_routing.rs` | **213** | new — 3 end-to-end tests |
| **Phase 19 totals** | **~355** | net new |

Workspace at the end of Phase 19:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **821 passing, 0 failing** (+3 vs Phase 18) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

## The graph join

```text
            ┌────────────────────────────────────────────────────┐
            │           GraphRailTelemetry::discrepancy_score    │
            └────────────────────────────────────────────────────┘
                                    │
                                    ▼
        rail_attempt vertices (window: now - window_secs)
          • filter by (rail, driver)
          • collect external_id_hint values
                                    │
                                    ▼
        ledger_tx vertices whose external_id ∈ collected hints
          • the "what work did this driver actually do?"
                                    │
                                    ▼
        inbound TASK_ABOUT edges on those ledger_tx vertices
          • each one is a reconciliation_task pointing at this tx
          • the discrepancy "count" for this driver
                                    │
                                    ▼
        score = min(1.0, tasks / attempts)
```

`external_id_hint` is the join key. In production it's the intent's
idempotency key — most deployments propagate that verbatim to the
ledger transaction's `external_id`. We don't enforce that
propagation; we just make it work when operators follow that
convention (which the existing `op-orchestrator` /
`op-ledger` examples do).

## The router math

`combined_score = max(failure_score, discrepancy_score)`. Both
sit in `[0.0, 1.0]`; the worst-news-in-either-dimension is the
ranking key. A clean-HTTP-but-noisy-reconciliation driver still
gets pushed back; a failing-HTTP-but-no-recon-history driver
still gets pushed back. The two dimensions aren't additive — a
broken driver isn't *doubly* broken — but either alone is enough
to demote it.

Stable sort on tied scores still preserves operator-configured
preference, same as Phase 18.

## Key design decisions

### 1. Default `discrepancy_score = 0.0`

The trait method has a default body returning `0.0`, so
`NoOpRoutingSignals` and any third-party impl that doesn't track
reconciliation state needs zero work. Implementations that DO
have access can override. Same pattern as the
`set_endpoint_metadata` default-empty trait methods elsewhere in
the codebase.

### 2. `external_id_hint` is `Option<&str>`, not required

A few attempt sites legitimately have no ledger correlation —
a dry-run probe, a health check, a manual replay that doesn't
post. The trait accepts `None` and `GraphRailTelemetry` simply
omits the property. Those attempts contribute to `failure_score`
but never to `discrepancy_score`, which is the correct
semantic.

### 3. Cap the score at 1.0

A particularly broken driver might have 5 reconciliation tasks
against 2 attempts (e.g. several days of statement-line
mismatches against the same buggy charge). The raw `tasks /
attempts` could exceed 1.0; we clamp. The router only needs an
ordering anyway, not a probability — but the docs commit to
`[0.0, 1.0]` and we keep that promise. Anything north of 1.0
just means "as bad as it gets, push back."

### 4. Walk-the-vertices, not a stored index

The join scans `rail_attempt` and `ledger_tx` vertices on every
`discrepancy_score` call. No secondary index. This is the same
operational profile as `failure_score`, and at reference-impl
scale (hundreds or low thousands of recent attempts in the
window) it's plenty fast. A production deployment with millions
of attempts would either (a) shrink the window, (b) shrink
history via compaction, or (c) ship a custom `RoutingSignals`
that consults an external index. The trait shapes for all three.

### 5. Idempotency-key propagation is the *only* contract

The new join works because operators conventionally set their
ledger transaction's `external_id` to the same idempotency key
the intent carried. We don't enforce that — the docs do. If an
operator's ledger writes use a different correlation token, they
ship their own `RailTelemetry` impl that records that token
instead. The orchestrator just forwards what's on the intent.

### 6. The discrepancy signal lives **alongside**, not instead of,
the failure signal

Phase 18's `failure_score` is unchanged. Phase 19 adds a
*second* axis. The router takes the max. This composition keeps
each signal independently meaningful (and independently
debuggable in operator dashboards) — operators can ask "is this
driver bad because it's failing HTTP or because its books are
wrong?" and the two scores answer separately.

## What the end-to-end test demonstrates

`crates/op-graph/tests/discrepancy_routing.rs`:

1. Static config: `[dirty_psp, clean_psp]` in card_drivers.
2. Both adapters always return `Success` — Phase 18's signal is
   silent.
3. First intent (`"intent-1"`): chain is `[dirty, clean]`, dirty
   tried first, succeeds. One attempt recorded against dirty
   with `external_id_hint = "intent-1"`.
4. Operator posts a ledger transaction with `external_id =
   "intent-1"`. (Test simulates the production pattern.)
5. Reconciliation records a `UnmatchedLedger` task pointing at
   that ledger_tx.
6. `discrepancy_score(card, dirty) = 1.0` — one task per one
   attempt.
7. Second intent: combined score reorders chain to `[clean,
   dirty]`. Clean tried first, succeeds. **One attempt, dirty
   never touched.**

Plus two negative tests: empty reconciliation state yields
score 0.0; a driver with no attempts in window yields 0.0.

## What this phase does NOT do

- **No tagging ledger_tx with the rail/driver directly.** That
  would shorten the join — `reconciliation_task → ledger_tx →
  rail/driver` instead of three hops — but it would mean the
  ledger writer (which today is operator app code) has to know
  the rail/driver. The orchestrator doesn't post ledger txs; the
  operator's app does, often based on the orchestrator's
  outcome. Phase 19 deliberately stays out of that boundary; the
  join goes through `external_id` because that's the operator-
  controlled coupling that already exists.

- **No per-task severity weighting.** An `UnmatchedLedger` and
  an `AmountMismatch` count the same. Operators who care can
  ship a custom `RoutingSignals` that walks
  `reconciliation_task.kind` and weights.

- **No discrepancy-task aging.** A reconciliation task from
  three weeks ago counts the same as one from today, as long as
  the rail_attempt that produced it is in the current window.
  Window-scoping is on `rail_attempt`, not the task. If
  operators care about task-age decay, that's a custom impl.

- **No automatic discrepancy resolution.** Phase 19 reads tasks
  to route around them; resolving them is still operator work
  (mark them resolved, retract the vertex, etc.).

## Bugs caught during construction

1. **`record_attempt` signature drift** broke Phase 18's
   persistence test (`graph_routing.rs`). Two callsites needed
   `None` for the new `external_id_hint` parameter. cargo
   pointed straight at them; mechanical fix.

2. **`etypes` not imported** in `rail_telemetry.rs` after the
   new `in_edges(.., etypes::TASK_ABOUT)` call. Added to the
   import line. The same `use crate::graph::{etypes, vtypes,
   GraphHandle}` pattern that every other store uses.

3. **`u32::try_from(...).unwrap_or(...)`** for the task count —
   could overflow `u16` and the score would be `inf`. The
   `min(1.0)` cap catches it but the cast was still wrong;
   replaced with the same safe-cast idiom Phase 18 used.

## Honest concerns going into Phase 20+

- **Task age is unbounded by the window.** As noted above, a
  task from a year ago still bumps the score if the rail_attempt
  that produced it is in the window — unlikely, but possible
  if the attempt log is huge and the window is huge.

- **The join is O(rail_attempts × ledger_tx + tasks per
  candidate tx)**. For reference-impl volumes this is invisible;
  for production it's the same scale concern Phase 18 already
  documented.

- **`external_id` is a property, not an indexed attribute** in
  the current Minigraf usage. We scan all `ledger_tx` vertices
  and string-compare. A secondary EAV index keyed by
  `external_id` value (Minigraf supports it) would speed this
  to O(hints) lookups. Not pulled in yet; trivial to add when
  performance forces the issue.

- **`combined_score = max` is a *choice*, not a derivation.**
  Operators might want weighted (`0.7 failure + 0.3 discrepancy`)
  or threshold (`failure if > 0.5 else discrepancy`). The
  current router is hardcoded to max; future enhancement could
  expose a `combined: fn(f32, f32) -> f32` knob.

## Cumulative state

| Phase | Crate(s) | Tests | LOC |
|---|---|---:|---:|
| 1–18 | (see prior docs) | 818 | ~39,256 |
| **19** | **discrepancy_score on RoutingSignals + GraphRailTelemetry join** | **+3** | **+355** |
| **Total** | | **821** | **~39,611** |

## What's next (Phase 20+ candidates)

- **Wall-clock time-travel + named checkpoints** (still queued
  from Phase 17).
- **Latency signal.** Add `duration_ms` to `record_attempt`,
  surface it as a third `RoutingSignals` axis.
- **CAMT.053 XML conformance + `Camt054Source`** (still queued
  from Phase 15).
- **Weighted / configurable signal combiner.** `combined_score`
  becomes a function the operator can swap (max, weighted,
  threshold-based).
- **History compaction** (rolled over from Phase 16). Now even
  more relevant: the discrepancy join walks the `rail_attempt`
  log alongside the ledger / reconciliation graphs.
- **Reconciliation-task severity weighting.** Per-`kind` weights
  in the discrepancy_score join.

The thesis stands: the same single-file graph stores attempts,
ledger txs, reconciliation tasks, and time-travel history; each
phase pulls a new view out of the same substrate. Phase 19's
view is the one that catches the silent failures — the ones
where the rail says yes and the books say no — and routes
around them automatically.
