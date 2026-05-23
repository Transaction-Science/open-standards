# Phase 24 — Driver SDK: operators can author their own rails

**Status**: Draft v0.24
**Date**: 2026-05-22

## Why

OpenPay's whole architecture is built around pluggable rail
adapters (`RailAdapter`, `CardAcquirer`, `A2aAcquirer`). What it
*didn't* have until Phase 24 was the **author-side scaffolding** —
the deterministic mocks operators use to bring up a deployment
without live credentials, and the conformance harness that lets
driver authors verify their adapter respects the trait's
behavioral contract beyond what the type system enforces.

Final phase of the sequenced trio (22 → 23 → 24) that closes out
the operator-facing surface: settlement (Phase 22), HTTP API
(Phase 23), driver SDK (Phase 24). After this an operator with
real PSP credentials can write a driver, run conformance, register
it with the orchestrator, and deploy.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `op-driver-sdk` crate — lib + author guide module docs | `crates/op-driver-sdk/src/lib.rs` |
| 2 | `DeterministicCardAcquirer` — programmable card acquirer: per-key overrides, amount-threshold rules, transport-error mode, request history | `card.rs` |
| 3 | `DeterministicA2aGateway` — symmetric programmable A2A gateway with UETR overrides | `a2a.rs` |
| 4 | `ConformanceFailure` taxonomy + `ConformanceReport` aggregate | `conformance.rs` |
| 5 | `conformance::run_card` — battery of contract checks for any `CardAcquirer` | `conformance.rs` |
| 6 | `conformance::run_a2a` — equivalent for `A2aAcquirer` | `conformance.rs` |
| 7 | `conformance::run_card_with_panic_probe` — stronger variant via `catch_unwind` | `conformance.rs` |
| 8 | Orchestrator integration tests — deterministic drivers flowing through a real `Orchestrator` via `CardAdapter` / `A2aAdapter` | `tests/orchestrator_integration.rs` |
| 9 | Unit tests proving the harness catches malformed drivers (lying `supports()`, duplicate PSP ids, empty name, missing UETR) | `conformance.rs::tests` |

Workspace at the end of Phase 24:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **930 passing, 0 failing** (+24 vs Phase 23) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta: 19 unit (card 8, a2a 5, conformance 6) + 5 e2e
integration = 24.

## The driver-author flow

```text
1. impl CardAcquirer for MyPspClient { ... }
2. Write at least one happy-path test against your real PSP
   sandbox.
3. Run op_driver_sdk::conformance::run_card(&my_client)?
   to catch structural bugs.
4. Wrap your acquirer in op_orchestrator::CardAdapter and register
   with the orchestrator.
```

In code:

```rust
let driver = MyPspClient::sandbox(api_key);
let report = op_driver_sdk::conformance::run_card(&driver);
assert!(report.is_clean(), "driver failed: {:?}", report.failures);
let card = Arc::new(CardAdapter::new("my-psp", Arc::new(driver)));
orchestrator.register_adapter(card);
```

## What the conformance harness covers

| Check | Failure | What it catches |
|---|---|---|
| `name()` non-empty | `EmptyName` | Telemetry / routing keys off `name()`; empty breaks operator dashboards |
| `supports()` ↔ `authorize()` consistency | `SupportsLies` | A misclassifying driver causes the orchestrator to route to a rail that then 4xx's every request |
| `psp_payment_id` non-empty on success | `EmptyPspPaymentId` | Capture / refund / void all require it |
| `psp_payment_id` unique across calls | `DuplicatePspPaymentId` | Reused ids corrupt later capture / refund routing |
| `authorized_amount` or `error_code` present on success | `AuthorizeResponseEmpty` | Callers have nothing to act on otherwise |
| A2A `Settled`/`Accepted` echoes UETR | `A2aMissingUetrOnAccept` | Settlement notifications match back to UETR; missing UETR breaks reconciliation |
| `catch_unwind` on transport-error paths | `PanicOnTransportError` | A panicking driver brings down the orchestrator thread |

## DeterministicCardAcquirer surface

```rust
DeterministicCardAcquirer::new()
    .with_default_status(AuthStatus::Settled)
    .with_key_override("k-decline", AuthStatus::HardDecline, Some("nsf".into()))
    .with_amount_ge(
        Money::from_minor(1_000_000, Currency::USD),
        AuthStatus::RequiresCustomerAction,
        None,
    )
    .with_transport_error("connect timeout") // exclusive — short-circuits everything
;

// After running:
acq.auth_history();    // every AuthRequest the driver saw
acq.capture_history(); // every CaptureRequest
acq.refund_history();
acq.void_history();
```

Same shape on `DeterministicA2aGateway` — UETR overrides, amount
rules, transport-error mode, `transfer_history()` /
`query_history()` for assertions.

## Why a runtime harness instead of a trait constraint

Rust's type system enforces the shape of the trait but not its
behavior. "The idempotency key flows through unchanged" and
"transport errors don't panic" are properties only a runtime probe
can verify. A doc-only convention is brittle; the runnable harness
gives driver authors an immediate pass/fail signal that the OpenPay
project can keep in sync with the trait semantics as they evolve.

## End-to-end integration

The integration test suite at
[`tests/orchestrator_integration.rs`](crates/op-driver-sdk/tests/orchestrator_integration.rs)
proves the deterministic drivers flow through a real `Orchestrator`
via `CardAdapter` / `A2aAdapter` with `PolicyRouter`. Operators
copy these tests as the starting point for their own driver
acceptance suite.

## Honest concerns (carry-forward)

- **Idempotency-key propagation isn't probed.** The harness
  documents the requirement (the key the caller sends must equal
  the key the PSP sees) but can't actively check it without
  driver cooperation — there's no way to inspect a third-party
  PSP's view of the request. Driver authors should write a
  driver-specific test asserting this.
- **No async harness.** All conformance probes are sync, matching
  the current sync acquirer traits. When the traits go async we
  add the same checks with `tokio::test`.
- **No fuzz / property-based checks.** The harness is a fixed
  battery. Adding proptest-driven invariants ("for any input the
  driver responds in finite time / never panics") would be a
  natural follow-up.
- **`run_card` runs against the real driver.** Calling it from CI
  *will* hit the PSP sandbox if the driver does network I/O.
  Operators wrap the driver in their own throttle if they want to
  run conformance on every PR.
- **No HTTP request recording.** Drivers that talk over HTTP would
  benefit from a record/replay layer (capture sandbox responses,
  replay deterministically in tests). That's a separate crate.
  Most operators handle this with `wiremock` or `mockito`.

## Test totals

```
op-driver-sdk  24 tests
                  card mock                8  (default / overrides / amount / transport / history / supports)
                  a2a mock                 5  (default / override / amount / transport / name)
                  conformance unit         6  (harness self-test + failure detection)
                  orchestrator integration 5  (card happy, card decline, a2a happy, a2a amount, conformance both)
                                                              ----
                                                              +24 net
```

`cargo test --workspace`: **930 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**

## What's next

With Phases 22 → 23 → 24 complete, OpenPay now has:

- A double-entry ledger with bi-temporal time-travel (Phases 12, 17)
- A graph-backed audit / fraud query layer (Phases 14, 18, 19, 21)
- A pluggable orchestrator with routing signals (Phases 11, 18, 19, 20)
- Reconciliation with ISO 20022 sources (Phases 15, 20)
- Refund + dispute workflows (Phase 21)
- Settlement + payout file generation (Phase 22)
- A deployable HTTP API server (Phase 23)
- A driver-author SDK with conformance harness (Phase 24)

The remaining gap to "ship and run in production" is operator-
side: a real merchant directory, a real fraud scorer model, real
PSP integrations using the SDK. None of those belong in the
reference stack — they're per-operator concerns the architecture
deliberately leaves pluggable. **OpenPay is now SOTA-deployable.**
