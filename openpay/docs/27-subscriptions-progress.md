# Phase 27 — Subscriptions: recurring billing

**Status**: Draft v0.27
**Date**: 2026-05-22

## Why

The most-requested feature for an operator-deployable payments
stack that the workspace couldn't model until now. Vendors with
SaaS, memberships, content subscriptions, or any other "charge the
same customer on a schedule" pattern need plans, billing cycles,
trials, dunning, and proration.

Final phase of the sequenced trio (25 → 26 → 27): crypto (Phase
25), then single-file persistence (Phase 26), now subscriptions.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `op-subscriptions` crate — sealed error, plan, subscription, scheduler, dunning, proration, store | `crates/op-subscriptions/` |
| 2 | `Plan` (id, name, amount, interval, interval_count, trial_days) with snapshot-at-creation semantics | `plan.rs` |
| 3 | `Interval` enum: `Day` / `Week` / `Month` / `Year` (Month/Year calendar-aware) | `plan.rs` |
| 4 | `Subscription` + `Status` state machine: `Trialing → Active → PastDue/Paused/Canceled` with transitions | `subscription.rs` |
| 5 | `BillingScheduler` — pure-function period math: `classify(sub, now) → DueState`, `tick(sub, now) → mutate` | `scheduler.rs` |
| 6 | Calendar-aware `add_months` — Jan 31 + 1 month = Feb 28/29 (no off-by-one bugs) | `scheduler.rs` |
| 7 | `DunningPolicy` — pluggable retry schedule with `decide(retry_count, failed_at, now) → DunningOutcome` | `dunning.rs` |
| 8 | `proration::credit_remaining` + `proration::switch_charge` — integer-exact proration math for mid-cycle plan changes | `proration.rs` |
| 9 | `SubscriptionStore` trait + `InMemorySubscriptionStore` ref impl | `store.rs` |
| 10 | `GraphSubscriptionStore` in op-graph — same single-file persistence as the other domain stores | `op-graph/src/subscription_store.rs` |
| 11 | `AppState.subscriptions: Arc<GraphSubscriptionStore>` wired in | `op-server/src/state.rs` |
| 12 | HTTP endpoints: `POST /v1/subscriptions`, `GET /v1/subscriptions/{id}`, `GET /v1/subscriptions?customer_ref=`, `POST /v1/subscriptions/{id}/cancel`, `POST /v1/subscriptions/{id}/pause`, `POST /v1/subscriptions/{id}/resume` | `op-server/src/handlers/subscription.rs`, `routes.rs` |
| 13 | `From<op_subscriptions::Error> for ApiError` mapping (404 / 409 / 400) | `op-server/src/error.rs` |

Workspace at the end of Phase 27:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **1023 passing, 0 failing** (+46 vs Phase 26) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta breakdown: op-subscriptions +41 (plan 3, dunning
5, proration 6, scheduler 9, store 6, subscription 12),
GraphSubscriptionStore +4, op-server subscription HTTP +1.

## Domain model

```
       Plan (snapshot per-subscription)
         │
         ▼
       Subscription
         ├── customer_ref (operator's customer id)
         ├── method        (PaymentMethod — vault token / crypto / A2A / ...)
         ├── status        (Trialing/Active/PastDue/Paused/Canceled)
         ├── period         [start, end)
         └── cancel_at_period_end (bool flag)
```

`Plan` snapshots into `Subscription` at creation. Editing a plan
in the catalog does NOT re-price existing subscribers — that's the
universal SaaS convention and the right default for predictable
billing.

## State machine

```text
   ─────►  Trialing  ──promote_from_trial()──►  Active
              │                                   │
              │                                   ├──pause()────► Paused
              │                                   │                  │
              │                                   │                  └─resume()──►
              │                                   │
              ▼                                   ▼
            (pause/cancel paths)        record_billing_failure()
                                                    │
                                                    ▼
                                                PastDue
                                                    │
                                          ┌─record_billing_recovered()
                                          │
                                          ▼
                                        Active

   Any state ──schedule_cancel_at_period_end()──► flag set;
       scheduler tick at period end ──► Canceled (terminal)
   Any state ──cancel_now()────────────► Canceled (terminal)
```

Illegal transitions return `Error::InvalidTransition`. Terminal
states refuse all further transitions.

## Calendar-aware period math

```rust
advance(jan_31_unix, Interval::Month, 1)   // → feb_29_unix (2024 leap)
advance(jan_15_unix, Interval::Year, 1)    // → jan_15_next_year_unix
```

`add_months` uses the `time` crate to do day-clamping correctly
(Jan 31 → Feb 28/29, not Mar 3). Day / Week math is exact
arithmetic on unix seconds.

## Scheduler

```rust
match BillingScheduler::classify(&sub, now_unix_secs) {
    DueState::NotDue           => /* keep waiting */,
    DueState::TrialEnded       => /* operator promotes via tick() */,
    DueState::PeriodRollover   => /* charge the new period */,
    DueState::CancelAtPeriodEnd => /* sub flipped its flag; finalize */,
}

// Or do the mutation in one shot:
let state = BillingScheduler::tick(&mut sub, now_unix_secs)?;
```

`tick` is the action verb: it transitions the subscription and
returns the resulting `DueState` so the caller knows what to do
externally (typically: `PeriodRollover` → enqueue a charge via the
orchestrator).

## Dunning

```rust
let policy = DunningPolicy::default();              // 1d → 3d → 5d → 7d → cancel
let policy = DunningPolicy::aggressive_daily();     // 7×1d → cancel
let policy = DunningPolicy::conservative();         // 1d → 7d → 30d → cancel

match policy.decide(retry_count, failed_at, now) {
    DunningOutcome::RetryNow                       => /* charge again */,
    DunningOutcome::Wait { next_attempt_at }       => /* enqueue for that time */,
    DunningOutcome::Cancel                         => /* subscription dies */,
}
```

The policy is data — schedules are `Vec<u32>` of per-retry delays
in days. Operators serialize/deserialize them; no need to re-deploy
to change retry behavior.

## Proration

```rust
// Customer upgrades mid-cycle. Credit the unused portion of the
// old plan against the first charge on the new plan.
let credit = proration::credit_remaining(&sub, now_unix_secs)?;
let charge = proration::switch_charge(new_plan.amount, credit)?;
// charge clamps at zero — we don't model "we owe the customer money"
// via subscriptions (refund flow handles that).
```

Math is integer-exact: `amount × seconds_remaining /
period_length`, with the integer floor biased toward the operator.
No floating point, no rounding surprises.

## Single-file persistence

Same model as Phase 26: `GraphSubscriptionStore` stores one
`subscription` vertex per subscription with the full state JSON
plus indexed properties (`external_id`, `customer_ref`,
`status_code`, `current_period_end`). Reopening the `.graph` file
recovers every subscription record alongside refunds / disputes /
batches / idempotency / ledger / webhooks. No separate database
server.

## HTTP surface

```
POST  /v1/subscriptions
GET   /v1/subscriptions/{id}
GET   /v1/subscriptions?customer_ref=...
POST  /v1/subscriptions/{id}/cancel        # body: {at_period_end?, now_unix_secs}
POST  /v1/subscriptions/{id}/pause         # body: {now_unix_secs}
POST  /v1/subscriptions/{id}/resume
```

Same `{code, message, details}` error envelope as the rest of the
server. `404` on unknown id, `409` on illegal transitions, `409`
on `external_id` mismatch, `400` on bad input.

## Honest concerns (carry-forward)

- **No actual charging.** The scheduler tells you a subscription
  is due; the operator wires the charge through `op-orchestrator`
  (or a direct adapter call) and feeds the result back via
  `record_billing_failure` / `record_billing_recovered`. The
  glue layer is operator-side.
- **`Subscription` doesn't implement `PartialEq`.** Its `method:
  PaymentMethod` field carries `Token` bytes that intentionally
  don't have equality semantics. Compare on `id` when testing.
- **No usage-based billing.** Plans are fixed-amount; metered /
  per-unit-of-X pricing is a separate model that doesn't fit
  cleanly into `Plan::amount`. Future phase if operators need it.
- **No add-ons / coupons / discounts.** Operators model these
  via `metadata` and adjust the orchestrator's charge call.
  Building them in would push opinion onto every operator.
- **No customer-side cancellation portal.** That's UI, not
  backend. The HTTP endpoints support it; the portal is the
  operator's frontend job.
- **`list_for_customer` / `list_due_at` scan all vertices.** Same
  limitation as the other graph-backed stores: Minigraf doesn't
  yet expose secondary indexes. For 100k subscriptions this
  scans in milliseconds; at 10M operators want a backend with
  proper indexing (the trait surface admits one drop-in).
- **No webhook events on subscription transitions.** Operators
  wanting "send email on charge-failed" wire that themselves via
  the existing `op-webhook` dispatcher.

## Test totals

```
op-subscriptions     41 tests
                       plan          3
                       dunning       5
                       proration     6
                       scheduler     9
                       store         6
                       subscription 12
op-graph              +4 tests (GraphSubscriptionStore)
op-server             +1 test (subscription_create_pause_resume_cancel)
                                                              ----
                                                              +46 net
```

`cargo test --workspace`: **1023 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**

## State of the project after Phase 27

OpenPay now covers, end-to-end, what a vendor needs to bring
margin back from the card-network tax stack:

| Capability | Phase |
|---|---|
| Double-entry ledger | 12 |
| Graph-backed audit / fraud queries | 14, 18, 19, 21 |
| Bi-temporal time-travel | 17 |
| Cross-rail orchestrator with routing signals | 11, 18, 19, 20 |
| Reconciliation (CAMT.053/054, NACHA-aware) | 15, 20 |
| Refunds + disputes | 21 |
| Settlement + payout files | 22 |
| HTTP API server | 23 |
| Driver SDK + conformance harness | 24 |
| Crypto rail (USDC, EURC, PYUSD) | 25 |
| Single-file persistence across all stores | 26 |
| Subscriptions / recurring billing | **27** |

A vendor can write a driver against `op-driver-sdk`, plug it into
the orchestrator, deploy `op-server` pointed at one `.graph` file,
and run a production payments stack — card / A2A / crypto rails,
refunds, disputes, settlement, subscriptions, audit — without any
external database, queue, or auxiliary service.
