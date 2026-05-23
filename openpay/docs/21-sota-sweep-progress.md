# Phase 21 — SOTA sweep: refunds, disputes, observability, audit report

**Status**: Draft v0.21
**Date**: 2026-05-22

## Why

OpenPay's mission is to bring margin back to vendors by removing the
3–5% tax stack. To do that, the reference stack has to cover the
full payment-acceptance surface — not just the happy path. After
Phase 20 the books, the rails, the reconciler, and the routing
brain were all SOTA, but three operator-facing surfaces were still
missing: **refund** workflows, **dispute / chargeback** workflows,
and the **trace + audit-report** plumbing that auditors and SREs
ask for. Phase 21 closes those gaps.

No new external dependencies. Every new crate sits on the same
typed-store / graph-substrate / pluggable-backend pattern the rest
of the workspace already establishes.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `op-refund` crate — full/partial refunds, state machine, idempotency on `external_id`, pluggable `RefundStore` + in-memory ref impl | `crates/op-refund/` |
| 2 | `op-dispute` crate — chargeback / inquiry / representment workflow, normalized cross-network reason taxonomy, evidence refs, pluggable `DisputeStore` | `crates/op-dispute/` |
| 3 | OpenTelemetry-compatible trace propagation — `tracing` workspace dep, `#[tracing::instrument]` on the hot paths in orchestrator / ledger / webhook / reconciliation | `Cargo.toml`, `op-{orchestrator,ledger,webhook,reconciliation}` |
| 4 | History compaction — `GraphHandle::compact()` surfacing Minigraf's checkpoint API | `op-graph/src/graph.rs` |
| 5 | Audit report builder — `AuditReport::for_window(handle, start, end, now)` joining `ledger_tx` × `rail_attempt` × `reconciliation_task` in one graph pass | `op-graph/src/audit.rs`, `tests/audit_report.rs` |

Workspace at the end of Phase 21:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **861 passing, 0 failing** (+37 vs Phase 20) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta breakdown: op-refund +17, op-dispute +16, op-graph
audit_report integration suite +4.

## op-refund

A refund is its own lifecycle, not just a "negative transaction."
Operators need to track *intent* (the customer or merchant asked
for the money back), submit to the PSP, wait for approval, and
then post the ledger reversal when settlement is confirmed —
sometimes days later. Folding all of that into the ledger as raw
debits/credits loses the workflow.

### Domain model

`Refund` carries: a deterministic UUIDv7 `RefundId`, the
`original_tx_id` it reverses, the `Money` amount, a
`RefundReason`, optional operator `external_id` for idempotency,
optional `psp_refund_id` once submitted, an ordered `metadata`
list, request/decision timestamps, and a `Status`.

```rust
pub enum Status {
    Requested,
    Submitted { psp_refund_id: String },
    Approved,
    Settled { settled_at_unix_secs: u64 },
    Declined { code: String, message: String },
    Failed { code: String, message: String },
}
```

State transitions live as methods on `Refund` that enforce the
graph:

- `submit(psp_refund_id)` — `Requested → Submitted`
- `approve()` — `Submitted → Approved`
- `settle(at)` — `Approved → Settled`
- `decline(code, msg)` — `Requested|Submitted → Declined`
- `fail_after_approval(code, msg)` — `Approved → Failed`

Any other transition returns `Error::InvalidTransition`. Amounts
larger than the original tx (operators are supposed to enforce
this with `list_for_tx` + sum) return `Error::AmountExceeded`
when the caller opts in to validation.

### Idempotency

`RefundStore::create_refund` is idempotent on `external_id`. The
in-memory ref impl uses a body-equivalence helper
(`bodies_equivalent`) that does an order-insensitive comparison
of the `metadata` bag — operators retrying with the same logical
payload but a re-ordered metadata HashMap iteration get the
existing id back, not a mismatch. A genuine body change (amount,
reason, tx id, external id, metadata content) returns
`Error::IdempotencyMismatch(external_id)`.

### Storage trait

`RefundStore` mirrors `LedgerStore` / `WebhookStore`: a small
sync trait with explicit atomicity contract on the closure-driven
`update`. The closure receives `&mut Refund`, runs the state
transition, and the store commits *only* if the closure returns
`Ok(())`. A partial mutation must never be persisted. The
in-memory impl clones-stage-commit; a future Postgres impl would
wrap the closure in a transaction.

## op-dispute

Disputes are a different beast from refunds. A refund is operator-
or customer-initiated; a dispute is *issuer-* or *bank-*initiated
and carries evidence, deadlines, and a possible representment
loop. The cross-network taxonomy varies wildly — Visa Reason Codes
vs Mastercard vs Faster Payments "Confirmation of Payee" — so
the crate normalizes to a curated class while letting operators
keep the raw `network_reason_code` on the `Dispute`.

### Domain model

`Dispute` carries id, `tx_id`, `DisputeReason`, optional
`network_reason_code`, a `Vec<EvidenceRef>` (each carrying
`kind`, `url`, optional `note`, and `attached_at_unix_secs`),
optional `external_id`, deadlines, timestamps, and a `Status`:

```rust
pub enum Status {
    Inquiry,        // pre-chargeback; issuer asking for info
    Chargeback,     // money already pulled
    Representment,  // we're contesting the chargeback
    Won,            // representment succeeded
    Lost,           // representment failed
    Accepted,       // we accepted the chargeback (didn't represent)
}
```

Transitions:

- `attach_evidence(EvidenceRef)` — at any pre-resolution status
- `escalate_to_chargeback()` — `Inquiry → Chargeback`
- `represent()` — `Chargeback → Representment`
- `resolve_won()`/`resolve_lost()` — from `Representment`
- `accept()` — `Inquiry|Chargeback → Accepted`

### Storage trait

`DisputeStore` is the same shape as `RefundStore` (deliberate
parallelism — operators implementing the Postgres backend wire
both at once). Idempotency on `external_id`, closure-driven
`update` with atomicity contract, `list_for_tx` for fan-out.

## OpenTelemetry trace propagation

`tracing` is now a workspace dep, and the four crates an operator
typically wants to see in a flamegraph carry
`#[tracing::instrument]` on their hot path:

| Span | Where |
|---|---|
| `orchestrator.run` | `Orchestrator::run` — outer-most boundary; carries `intent_key`, `amount_minor`, `currency` |
| `ledger.post_transaction` | `InMemoryLedgerStore::post_transaction` — `tx_id`, `entry_count` |
| `webhook.dispatch` | `Dispatcher::dispatch` — `event_id`, `endpoint_count` |
| `reconciliation.reconcile` | `Reconciler::reconcile` — `window_start`, `window_end`, `tx_count` |

These are just span boundaries — no opinionated exporter, no
forced OTLP wiring. Operators install whatever
`tracing-subscriber` layer they prefer (Jaeger, Honeycomb,
stdout, `tracing-opentelemetry`). The crates themselves stay
exporter-agnostic.

## History compaction

Minigraf's `Database::checkpoint` collapses the bi-temporal log
once retracted assertions stop mattering. Phase 17 wired the
retract-then-assert pattern in but never surfaced compaction.
`GraphHandle::compact()` is the operator-facing thin wrapper:

```rust
handle.compact()?; // calls minigraf::Database::checkpoint internally
```

For the in-memory backend it returns `Ok(())` (no-op). For a
file-backed Minigraf store it does what you'd expect — fold the
journal, free the dead records. Test
`compact_returns_ok_on_in_memory_handle` in
`tests/audit_report.rs` exercises the no-op path.

## Audit report

Auditors and finance teams ask the same question every quarter:
*"for every ledger transaction in this window, show me the rail
that processed it, the reconciliation tasks that touched it, and
the status."* That join was already implicit across the typed
stores — Phase 21 surfaces it as a structured report on the
shared graph substrate.

```rust
let report = AuditReport::for_window(
    &graph_handle,
    start_tx_count,
    end_tx_count,
    generated_at_unix_secs, // operator-supplied for determinism
)?;
for entry in &report.entries {
    println!(
        "{} status={} rail={:?} driver={:?} tasks={:?}",
        entry.tx_id, entry.status,
        entry.rail, entry.driver,
        entry.reconciliation_task_ids,
    );
}
```

Each `AuditEntry` carries:

- `tx_id` and `external_id` (the idempotency-key join axis)
- `status` (lifecycle: pending / posted / archived)
- `posted_at_tx_count` and `effective_at_unix_secs`
- `settled_amount_minor` + `currency_code` (debit sum, single-
  currency only; multi-currency txs surface as `None`)
- `rail` + `driver` (joined via `rail_attempt.external_id_hint`)
- `reconciliation_task_ids` (inbound `task_about` edges)

The report sorts ascending by `posted_at_tx_count` so it reads
like an audit log. `generated_at_unix_secs` is operator-supplied
so replays / test runs are byte-identical.

### Implementation note

`for_window` is a single-pass walk: build the
`external_id → (rail, driver)` index up front from all
`rail_attempt` vertices, then iterate `ledger_tx` vertices,
filter by `posted_at_tx_count` window, join. One graph, one
report, no cross-store mutex coordination.

## Honest concerns (carry-forward)

- **Refund → ledger reversal posting is the operator's job.** The
  refund crate models the workflow but doesn't auto-post a
  reversal entry when `Status::Settled` lands. That coupling
  belongs in operator glue (which ledger? which reversal
  account?). A future phase could ship a `RefundLedgerLink`
  helper for the common single-ledger case.
- **Dispute → ledger reservation is also operator-side.** Many
  networks debit the merchant the moment the chargeback opens.
  We track the workflow; we don't auto-post the hold.
- **Audit report is read-only.** No "rebuild from journal"
  capability yet — operators with a Minigraf file store get
  bi-temporal time-travel via the existing `LedgerHistory`
  APIs, but the audit report itself is a snapshot.
- **Trace context propagation between crates is not enforced.**
  The instrumented spans nest correctly when callers come from
  the orchestrator, but an operator wiring a standalone
  reconciler against a wholly different orchestrator will see
  detached span trees unless they set up an external context
  propagator.

## Test totals

```
op-refund      17 tests (state machine 11, idempotency 3, store CRUD 3)
op-dispute     16 tests (state machine 10, evidence 2, store CRUD 4)
op-graph        4 tests (audit_report.rs — end-to-end join, empty,
                inverted window, compact no-op)
                                                              ----
                                                              +37 net
```

`cargo test --workspace`: **861 passing, 0 failing**.
`cargo build --workspace --all-targets`: **0 warnings**.
`cargo clippy --workspace --all-targets`: **0 warnings**.
