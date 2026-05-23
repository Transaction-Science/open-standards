# Phase 12 — `op-ledger` double-entry append-only ledger

**Status**: Draft v0.12
**Date**: 2026-05-17

## What shipped

A new crate `crates/op-ledger` plus an integration test suite that
demonstrates composition with the orchestrator. **Total: 69 tests
(59 unit + 10 integration), ~2,540 LOC.**

The ledger is the missing **source of truth** for money movement.
Phases 1-11 produce events (a card auth approved, a FedNow transfer
settled); the orchestrator routes them; the ledger **records them**
in a form that:

- balances correctly under concurrency,
- can be audited by a regulator,
- can be reconciled against bank/PSP statements as a set difference,
- survives software bugs (an unbalanced transaction is rejected at
  construction time, not silently committed),
- gracefully models corrections (reversal transactions, not edits).

## Verified ground truth

Researched live (May 2026 sources) before implementation:

| Claim | Source |
|---|---|
| Double-entry: debits == credits *per currency* within a transaction | Modern Treasury docs, pgledger (Paul Gross 2025), Trio fintech architecture guide |
| Append-only / immutable: posted transactions can never be edited, only reversed | Modern Treasury docs, Uber Gulfstream design, ISO 4217 accounting standards |
| Account currency fixed at creation | Modern Treasury docs (`currency` is mandatory; `balances.currency` always matches) |
| Account has a `normal_balance` (debit/credit) — Asset/Expense debit-normal; Liability/Equity/Revenue credit-normal | Standard double-entry accounting; Modern Treasury docs use this exact vocabulary |
| Three balance views: posted, pending (= posted + pending), available | Modern Treasury "How to Think About Ledger Balances" |
| Idempotency by `external_id`: matching body returns existing transaction, mismatched body returns 409-style error | Modern Treasury docs ("immutable, so your ledger records cannot be retroactively altered") + general API idempotency convention (Phase 11) |
| Reversal pattern: original is left immutable; a separate "reversal" transaction with flipped entry directions is posted | Modern Treasury docs, FinLego ledger architecture |
| `uuid::Uuid` implements `Ord`/`PartialOrd` natively | docs.rs/uuid |

## Architecture

```
crates/op-ledger/
├── Cargo.toml          — deps on op-core + serde + thiserror + uuid
├── src/
│   ├── lib.rs          — module roots; pub re-exports
│   ├── error.rs        — 10-variant Error enum + #[from] op-core
│   ├── ledger.rs       — LedgerId (UUID v4), Ledger {id, name, description}
│   ├── account.rs      — AccountId, NormalBalance, AccountClass, Account
│   ├── entry.rs        — Direction, Entry (account_id, direction, amount)
│   ├── transaction.rs  — TransactionId, Status (Pending/Posted/Archived), Transaction with double-entry validation per currency
│   ├── balance.rs      — Balance {currency, posted, pending}
│   └── store.rs        — LedgerStore trait + InMemoryLedgerStore
└── tests/
    └── integration.rs  — 10 tests proving composition with orchestrator
```

### Data model

```
Ledger (1) ──< Account (N: same ledger)
                  │
                  │ (referenced by)
                  │
                  ▼
Ledger (1) ──< Transaction (N: same ledger)
                  │
                  │ has-many
                  ▼
              Entry (account_id, direction, amount)
```

### Key invariants enforced by code (not by convention)

| # | Invariant | Where it's enforced |
|---|---|---|
| 1 | Double-entry per currency: ∀ currency c, Σ debits(c) = Σ credits(c) | `Transaction::validate_balanced()` |
| 2 | Minimum 2 entries per transaction | `Transaction::construct()` |
| 3 | Account currency matches entry currency | `InMemoryLedgerStore::post_transaction()` |
| 4 | Account ledger matches transaction ledger | `InMemoryLedgerStore::post_transaction()` |
| 5 | Account exists | `InMemoryLedgerStore::post_transaction()` |
| 6 | Idempotency by `external_id`: same body → same id; different body → error | `InMemoryLedgerStore::post_transaction()` |
| 7 | Terminal states are immutable: posted/archived cannot be re-transitioned | `Transaction::post() / archive()` |
| 8 | Balance computation is overflow-safe | `InMemoryLedgerStore::balance()` (checked_add/sub everywhere) |

## Key design decisions

### 1. Status lifecycle is two terminal states, not three

```
   Pending ──post()────► Posted   (terminal — settled)
      │
      └────archive()──► Archived  (terminal — voided)
```

No "Reversed" status. Reversals are **separate** transactions that
themselves go through the Pending → Posted lifecycle. The original
posted transaction stays posted forever. This is the Modern
Treasury convention and the auditable convention.

### 2. Reversal preserves an audit chain

`Transaction::reversal_of(&original, effective_at)`:

1. Flips every entry's direction.
2. Generates a new `external_id` of `{original_external_id}:reversal`
   (or `{original_id}:reversal` if no external_id) so the
   idempotency store doesn't collide.
3. Adds a `reverses: {original_id}` metadata entry.
4. Returns a **pending** transaction (caller decides when to post).

Auditors can grep for `:reversal` in `external_id` or `reverses`
in metadata to enumerate all corrections.

### 3. Multi-currency invariant is per-currency

A USD-only transaction needs USD debits == USD credits.

A USD + EUR transaction (e.g. FX) needs:
- USD debits == USD credits
- EUR debits == EUR credits

The crate does **not** enforce any cross-currency relationship
(rates are metadata). Test: `multi_currency_balanced_per_currency`
and `cross_currency_no_invariant_across_currencies`.

### 4. Balance is derived, never stored

Three views computed by walking entries:

- **Posted**: sum of entries from Posted transactions, signed by
  the account's normal balance.
- **Pending**: sum of entries from Posted + Pending transactions,
  signed the same way.
- **Available**: not implemented as a separate view in this phase.
  Equates to posted; left as a future extension for hold-aware
  models.

Derivation respects the account's `NormalBalance`:

- Debit-normal account (assets, expenses): balance = debits − credits
- Credit-normal account (liabilities, equity, revenue): balance = credits − debits

This means a healthy account always shows a positive balance, which
is the natural reporting convention.

### 5. Pluggable storage via `LedgerStore` trait

The 10-method trait separates the **what** from the **how**:

```rust
pub trait LedgerStore: Send + Sync {
    fn create_ledger(&self, ledger: Ledger) -> Result<LedgerId>;
    fn get_ledger(&self, id: LedgerId) -> Result<Ledger>;
    fn create_account(&self, account: Account) -> Result<AccountId>;
    fn get_account(&self, id: AccountId) -> Result<Account>;
    fn post_transaction(&self, t: Transaction) -> Result<TransactionId>;
    fn get_transaction(&self, id: TransactionId) -> Result<Transaction>;
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Transaction>>;
    fn mark_posted(&self, id: TransactionId) -> Result<()>;
    fn mark_archived(&self, id: TransactionId) -> Result<()>;
    fn balance(&self, id: AccountId) -> Result<Balance>;
}
```

The `InMemoryLedgerStore` impl is honest about being for tests and
single-process kiosks: a coarse `Mutex<Inner>` serializes all
operations. Production deployments will write a Postgres-backed
store (each method maps to a SQL transaction) or a TigerBeetle-
backed one (each method maps to a TigerBeetle batch).

### 6. Oracle discipline preserved

`Error::Core(#[from] op_core::Error)` wraps overflow errors verbatim.
Phases 8-10 platform bridges collapse error discriminants for FFI
exposure; the ledger crate exposes the rich inner variant for
server-side telemetry, consistent with op-orchestrator (Phase 11).

### 7. Idempotency at construction is structural, not behavioral

The `same_body()` check in `InMemoryLedgerStore` compares:

1. Same `ledger_id`,
2. Same number of entries,
3. Same `(account_id, direction, amount.minor_units)` triple after
   sorting (order-independent equality).

Notably we do NOT compare `effective_at_unix_secs`, `description`,
or `metadata`. Two re-tries with the same logical body but different
descriptions (e.g. one was annotated by an operator) are still
treated as the same transaction. This is the Modern Treasury
convention; it's documented and tested
(`idempotency_returns_existing_id_for_matching_body`).

## Bugs caught during this phase

1. **`Uuid::Ord` not assumed.** Initial draft of `sort_by_key` in
   `same_body` was written before verifying `Uuid: Ord`. Confirmed
   via docs.rs/uuid that it has been since v1.0; then added
   `PartialOrd, Ord` to derives of `AccountId`, `LedgerId`,
   `TransactionId`, `Direction` so they're sortable too.

2. **Overflow in pending-sum.** First draft computed
   `posted_debits + pending_debits` without `checked_add`. Fixed to
   propagate `op_core::Error::Overflow`.

3. **`sign_by_normal` infallible.** First draft returned `i64`
   directly; `debits - credits` could overflow. Refactored to
   `fn sign_by_normal(...) -> Result<i64>` using `checked_sub`.

4. **Test smell.** `setup()` returns `(store, lid, _cash, _rev)`
   with leading underscores; one test pattern was
   `let real = _cash;`. Code review caught it — fixed to use a
   non-underscore-prefixed binding throughout.

5. **`metadata` as `Vec<(String, String)>` vs `HashMap`.**
   Initial sketch used `HashMap` for metadata. Switched to `Vec` so
   we preserve insertion order and accept duplicate keys (a real
   need — operators may emit multiple `reverses` entries for
   chained reversals). Trade-off: lookup is O(n), but metadata
   sizes are tiny (< 10 entries typical).

6. **`find_by_external_id` returns `Option`, not `Result<Option>`.**
   First sketch returned `Option<Transaction>` directly, but a
   poisoned mutex would have panicked. Wrapped in `Result` for
   consistency with the rest of the trait (though the in-memory
   impl never actually returns a non-Ok value here).

## What this crate does NOT do

- **No persistence.** `InMemoryLedgerStore` loses state on process
  exit. Production needs a Postgres / TigerBeetle backend.
- **No cross-instance consistency.** A single `InMemoryLedgerStore`
  is correct for one process; multi-process deployments need a
  shared backing store.
- **No FX rate management.** Multi-currency transactions record
  both sides as separate entries; the rate is the operator's
  metadata.
- **No GAAP / IFRS classification.** `AccountClass` is
  informational. We don't produce trial balances or financial
  statements.
- **No hierarchical accounts (Categories).** Modern Treasury models
  "Ledger Account Categories" as hierarchical aggregations of
  accounts. Listed as Phase 13+ future work.
- **No balance locking.** Modern Treasury's "balance locking"
  feature (used for credit limits) is not modeled. The
  pluggable-store interface admits a backend that implements it.
- **No async wrapper.** All operations are synchronous. Async
  callers wrap with `tokio::task::spawn_blocking`.

## Composition with the orchestrator

The 10-test integration suite at `tests/integration.rs` shows the
**operator-level pattern**: an orchestration outcome maps to a
ledger transaction.

```text
┌──────────────────────┐         ┌────────────────────────────┐
│  Orchestrator        │ Approved│  Ledger                    │
│  run(intent) ────────┼─────────► post pending tx            │
│                      │         │     external_id = idem_key │
│                      │         │     debit recv             │
│                      │         │     credit revenue         │
└──────────────────────┘         └────────────────────────────┘

      ... settlement notification arrives ...

                                 ┌────────────────────────────┐
                                 │  Ledger                    │
                                 │  mark_posted(tx_id)        │
                                 └────────────────────────────┘
```

The same `idempotency_key` flows from the orchestrator's intent
into the ledger transaction's `external_id`, so duplicate
orchestration calls produce duplicate ledger posts — both of which
correctly return the **existing** transaction id. No double charge,
no double bookkeeping. Test: `replay_with_same_external_id_does_not_double_count`.

Critical: this composition does **not** require any code in
`op-ledger` that knows about the orchestrator. The crates are
decoupled; the integration is at the operator's layer.

## Test count

| Module | Unit tests |
|---|---|
| `account.rs` | 7 |
| `balance.rs` | 3 |
| `entry.rs` | 7 |
| `ledger.rs` | 5 |
| `store.rs` | 18 |
| `transaction.rs` | 19 |
| `tests/integration.rs` | 10 |
| **Phase 12 total** | **69** |

Each integration test:

1. `approved_auth_creates_pending_transaction` — Modern Treasury pattern: card auth → pending tx with pending balance set, posted balance zero.
2. `settlement_marks_posted_and_balances_settle` — mark_posted moves pending → posted; both balances now reflect the amount.
3. `psp_fee_recorded_separately` — fees go in their own transaction so reconciliation can compare gross vs net.
4. `replay_with_same_external_id_does_not_double_count` — the no-double-charge guarantee at the ledger layer.
5. `refund_reverses_balance_to_zero` — refund modeled as reversal; balance returns to zero.
6. `many_small_orders_sum_correctly` — 10 orders accumulate to the expected sum (arithmetic sanity).
7. `archived_transactions_are_invisible_to_balance` — voided auths don't appear in any balance view.
8. `multi_currency_ledger_supports_per_currency_accounts` — USD and EUR accounts in the same ledger; their balances don't leak.
9. `kiosk_day_simulation_balances_remain_consistent` — 5 orders × $10 + 5 × $0.30 fees + 1 refund; all four account balances independently verified.
10. `ledgers_are_isolated_balance_views` — two merchants in different ledgers; movement in one is invisible to the other.

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
| **12 op-ledger** | **69** | **~2,540** |
| **Total** | **~690** | **~28,640** |

## What's next

Phase 13+ candidates (not committed):

- **`op-webhook`** — async outbound webhook fanout for posting ledger
  events to operator-specified endpoints. Retry, signing,
  replay-protection.
- **Postgres-backed `LedgerStore`** — production-ready
  persistent backend.
- **TigerBeetle-backed `LedgerStore`** — high-throughput backend for
  high-volume merchants.
- **`op-reconciliation`** — diff a ledger against bank/PSP
  statements; surface discrepancies as a stream.
- **OpenTelemetry trace propagation** — single trace ID flows from
  the orchestrator's intent into the ledger transaction's metadata,
  observable across the whole stack.
- **Balance locking** for credit-limit / overdraft-prevention flows.
- **Ledger account categories** for hierarchical aggregations.
