# Phase 22 — Settlement & payout: when does the vendor actually see the money?

**Status**: Draft v0.22
**Date**: 2026-05-22

## Why

Posted ledger transactions are an accounting fact, not a bank
deposit. A "payment stack" that accepts money but doesn't tell the
vendor when funds land in their bank account isn't actually
delivering the SOTA payment-acceptance experience that brings
margin back to vendors — it's just a half of one.

Phase 22 closes that loop. `op-settlement` is the new layer
between posted-ledger and the payout rail: it groups transactions
into **batches**, applies a **holdback** policy (operator reserve
+ dispute risk adjustment), and produces the **payout file** the
rail expects.

This is the first of three sequenced phases (22 → 23 → 24) closing
out the operator-facing surface. Next up: HTTP API server (Phase
23), then driver SDK + reference integrations (Phase 24).

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `op-settlement` crate root + domain modules (`error`, `batch`, `cutoff`, `holdback`, `payout`, `store`, `engine`, `nacha`) | `crates/op-settlement/src/` |
| 2 | `Batch` type — id (UUIDv7), currency, rail, entries, holdback, lifecycle state machine | `batch.rs` |
| 3 | `Cutoff` schedule — `Daily{hour_utc}`, `MultiDaily{hours_utc}`, `Manual` — with `should_close(last, now)` semantics | `cutoff.rs` |
| 4 | `HoldbackPolicy` — flat-rate basis points + dispute adjustment + ceiling; clamps to gross | `holdback.rs` |
| 5 | `PayoutRail` — `AchNacha` / `SepaCt` / `FedNow` / `Rtp` / `Wire` / `InternalBookTransfer`; helpers `is_nacha`, `is_iso20022_pacs008` | `payout.rs` |
| 6 | `Pacs008EntryContext` — per-entry struct mapping a batch line to the data points an [`op_iso20022::CreditTransferBuilder`] needs | `payout.rs` |
| 7 | `SettlementStore` trait + `InMemorySettlementStore` ref impl with idempotency-on-`external_id` and closure-driven `update` | `store.rs` |
| 8 | `SettlementEngine` — opens / closes / submits / settles / fails batches; cutoff-driven `tick(last, now)` | `engine.rs` |
| 9 | `nacha_file(batch, profile, credits)` — full NACHA file generator: file header / batch header / entry detail / batch control / file control + 9-filler block padding | `nacha.rs` |
| 10 | End-to-end integration test: ledger → batch → cutoff → close → NACHA → submit → settled | `tests/end_to_end.rs` |
| 11 | Tracing instrumentation: `settlement.open_batch`, `settlement.add_entry`, `settlement.close_batch` | `engine.rs` |

Workspace at the end of Phase 22:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **897 passing, 0 failing** (+36 vs Phase 21) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta: op-settlement +36 (cutoff 7, holdback 5, batch 6,
store 4, engine 6, nacha 4, end-to-end 2, +2 nacha tests).

## Architectural placement

```
   op-ledger ──posted tx──► op-settlement ──batch──► op-iso20022
                                                       (pacs.008)
                                  │
                                  └─► nacha_file(...) → NACHA writer
```

`op-settlement` is downstream of `op-ledger` (it consumes posted
transactions) and upstream of rail payout (NACHA / `pacs.008`).
It deliberately does NOT talk to banks — that's the operator's
payout adapter. The output is a file (or in the `pacs.008` case,
a structured per-entry context the operator feeds to the existing
`CreditTransferBuilder`). The "last mile" — SFTP to the ODFI,
POST to the FedNow service — sits outside this crate.

## Batch lifecycle

```
   Open  ──close(holdback)──►  Closed  ──pay(reference)──►  Paying
                                                              │
                                                ┌──settled(at)
                                                │
                                                ▼
                                              Paid
                                                │
                                                └──fail(code, msg)──►  Failed
```

- `Open` accepts new posted-tx entries (`Batch::add_entry`).
- `Closed` is frozen — entry list and holdback set in stone.
- `Paying` is the in-flight state, carrying the rail's external
  reference (NACHA trace, pacs.008 msgId, wire ref).
- `Paid` and `Failed` are terminal. Reaching them is the
  operator's responsibility — we don't poll the rail.

Each transition is enforced by `Batch::*` methods that return
`Error::InvalidTransition` on illegal moves.

## Cutoff math

```rust
Cutoff::daily(7)                // 07:00 UTC daily (US nightly ~02:00 ET)
Cutoff::multi_daily(vec![7, 19])// 07:00 + 19:00 UTC
Cutoff::Manual                  // operator drives close
```

`should_close(last_tick, now)` returns `true` iff a scheduled
cutoff hour falls in `(last_tick, now]`. The implementation walks
day-aligned UTC midnight + `hour:00:00` candidates — bounded
iteration, no calendar library needed.

The engine's `tick(store, last, now)` uses this to close exactly
*one* open batch matching the engine's `(currency, rail)` filter.
Multiple open batches return `Error::Invalid` — operators
disambiguate before retrying (typically a misconfigured engine).

## Holdback policy

```rust
HoldbackPolicy::flat(50)                  // 0.50% reserve
HoldbackPolicy::flat(100).with_ceiling(2_000)  // 1% flat, capped at 20%
HoldbackPolicy::none()                    // pass-through
```

`compute(gross, dispute_adjustment_bps)` returns a `Holdback`
carrying the gross, reserve, and the two basis-point inputs.
Combined-reserve clamps at `max_total_bps`, and reserve clamps at
the gross — we never withhold more than the operator earned.

The dispute adjustment is operator-supplied — `op-settlement`
deliberately doesn't read the dispute store. Operators compute the
adjustment from their own risk model (number of open disputes /
recent chargeback rate / etc.) and pass it in. Keeps the crate
boundaries clean.

## NACHA generator

A full NACHA credit file (PPD/CCD) at `nacha::nacha_file`. Five
record types, fixed-width 94-char ASCII, block-aligned to 10
records with `9` filler. Reference: NACHA 2024 Operating Rules,
Appendix Three.

```rust
let file_str: String = nacha_file(&batch, &profile, &credits)?;
// 7 functional records padded to 10, all exactly 94 chars.
```

Scope: this is a **reference** generator covering the common
credit case. Returns, NOC, addenda records, IAT/CTX/web-debit
flavors are explicitly out of scope — operators wanting full
NACHA support extend the generator or wire their own.

## pacs.008 path

The crate doesn't auto-generate `pacs.008` files because
`op-iso20022::CreditTransferBuilder` already builds *one*
credit-transfer message per entry and the bundling format varies
by rail (FedNow takes them one-at-a-time, SEPA Direct File bundles
many in a `pain.001`/`pacs.008` envelope).

`Pacs008EntryContext` is the bridging struct: operators iterate
`batch.entries`, look up debtor/creditor party identifications
from their merchant directory, build a `Pacs008EntryContext`, and
hand it to the builder.

## Engine API surface

```rust
let engine = SettlementEngine::new(
    Currency::USD,
    PayoutRail::AchNacha,
    Cutoff::daily(7)?,
    HoldbackPolicy::flat(50),
);
let batch_id = engine.open_batch(&store, now_unix_secs)?;
engine.add_entry(&store, batch_id, tx_id, amount, Some("o-1".into()))?;
// ... time passes ...
engine.tick(&store, last_tick, now)?;                  // closes if cutoff fired
engine.submit_for_payout(&store, batch_id, "trace-1", now)?;
engine.mark_settled(&store, batch_id, now)?;
```

The engine takes `&impl SettlementStore` (rather than
`&dyn SettlementStore`) because the trait carries a generic
`update<F>` method for closure-driven atomicity, which would
make the trait non-dyn-compatible. Operators with multiple
store backends just monomorphize the engine per backend.

## Honest concerns (carry-forward)

- **No ledger linkage on payout-confirmation.** When a batch
  transitions to `Paid`, we don't auto-post the settlement entry
  (debit the cash-in-transit account, credit the operator's bank).
  That coupling is operator-side and ledger-shape-specific. A
  helper `BatchLedgerLink::post_settlement(batch, &ledger_store)`
  would close this in a follow-up.
- **Single-batch-per-file NACHA.** Real-world NACHA files often
  pack multiple batches per file (different SEC codes, different
  effective dates). The current generator is one-batch-per-file —
  operators bundle externally if needed.
- **`pacs.008` envelope generation is operator-driven.** SEPA
  Direct Debit File / FedNow Service Bus envelope formats vary by
  rail provider. We surface the per-entry context; the envelope is
  outside scope.
- **Total-reserve query is `O(open batches)` on the ref store.**
  `total_reserve_held` filters via `list_open`, which excludes
  closed/paid batches. For real-money reserves operators want a
  proper index on the store backend — surfaceable but not added
  in this phase since it's a backend-specific concern.
- **No payout retry logic.** A `Failed` batch is terminal. The
  pattern is to open a fresh batch and re-enrol the transactions.
  Auto-retry would couple this layer to rail-specific quirks
  (NACHA returns codes vs `pacs.002` status reason codes); the
  operator's payout adapter is better positioned.

## Test totals

```
op-settlement   36 tests
                  cutoff       7  (validation, fire-when-crossed, idempotent windows)
                  holdback     5  (none, flat, dispute add, ceiling, clamp)
                  batch        6  (open / lifecycle / mismatch / transitions)
                  store        4  (CRUD, idempotency, list_open)
                  engine       6  (open, add, close, submit-settle, tick variants)
                  nacha        4  (rail check, empty, block alignment, validation)
                  end_to_end   2  (full ledger→NACHA pipeline, empty-batch path)
                  + 2 nacha record-content assertions
                                                              ----
                                                              +36 net
```

`cargo test --workspace`: **897 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**
