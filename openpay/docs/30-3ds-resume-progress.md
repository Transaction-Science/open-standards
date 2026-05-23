# Phase 30 — 3DS / SCA resume primitive

**Status**: Draft v0.30
**Date**: 2026-05-22

## Why

Until Phase 30, when a card auth returned
`AttemptOutcome::RequiresAction { url }`, the flow stopped. The HTTP
intent endpoint returned the URL and the `psp_payment_id`, but
there was no way to *resume* the intent once the customer
completed the 3DS challenge. That made the card rail
half-functional for any SCA-regulated region: Europe (PSD2),
India, parts of LATAM.

Phase 30 closes that loop. It adds:

1. A `CardAcquirer::confirm_after_challenge` method (default
   `UnsupportedMethod` so old gateways stay correct).
2. A `RailAdapter::resume` method (default soft-failure so
   adapters without a challenge flow stay correct).
3. An `Orchestrator::resume(intent, rail, driver, psp_payment_id)`
   entry point that reclassifies the resumed outcome and updates
   the idempotency cache.
4. An HTTP `POST /v1/intents/resume` endpoint.
5. `DeterministicCardAcquirer` support for challenge-then-confirm
   so operators can integration-test the flow without a live PSP.

Last phase of the sequenced 28 → 29 → 30 trio. Webhooks, FX,
3DS — the three highest-impact operator-facing gaps from the
post-Phase-27 stack — all closed.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `CardAcquirer::confirm_after_challenge(psp_payment_id, idempotency_key)` trait method with default `Err(UnsupportedMethod)` | `op-rails-card/src/acquirer.rs` |
| 2 | `RailAdapter::resume(intent, psp_payment_id) -> AdapterResult` with default `SoftFailure { code: "resume_not_supported" }` | `op-orchestrator/src/engine.rs` |
| 3 | `CardAdapter::resume` override — calls `confirm_after_challenge` and reuses `classify_decision` to map back to `AttemptOutcome` | `op-orchestrator/src/adapters/card.rs` |
| 4 | `Orchestrator::resume(intent, rail, driver, psp_payment_id) -> Result<OrchestrationOutcome>` — looks up the adapter, runs resume, commits the resolved outcome to the idempotency cache | `op-orchestrator/src/engine.rs` |
| 5 | `DeterministicCardAcquirer::confirm_after_challenge` — synthetic post-challenge settle for integration tests | `op-driver-sdk/src/card.rs` |
| 6 | `POST /v1/intents/resume` HTTP endpoint — takes the same intent body + `rail` + `driver` + `psp_payment_id` | `op-server/src/handlers/intent.rs`, `routes.rs` |
| 7 | Tracing span `orchestrator.resume` with `idempotency_key`, `driver`, `rail` fields | `op-orchestrator/src/engine.rs` |
| 8 | Emitted webhook events: `intent.resumed.approved`, `intent.resumed.declined`, `intent.resumed.requires_action` | `op-server/src/handlers/intent.rs` |
| 9 | Integration tests: orchestrator-level (challenge then resume → approved; resume against unknown driver → error) + HTTP-level (`POST /v1/intents` → `POST /v1/intents/resume`) | `op-driver-sdk/tests/`, `op-server/tests/` |

Workspace at the end of Phase 30:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **1050 passing, 0 failing** (+3 vs Phase 29) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

## Flow

```text
   1. POST /v1/intents
        intent → orchestrator.run() → adapter.attempt()
                                       → AuthStatus::RequiresCustomerAction
        ─► outcome.terminal_status = RequiresCustomerAction
        ─► response carries (psp_payment_id, redirect_url)

   2. Customer completes the 3DS challenge out-of-band.
      PSP knows. OpenPay doesn't (yet).

   3. POST /v1/intents/resume
        body: original intent + rail + driver + psp_payment_id
        → orchestrator.resume()
            → adapter.resume(intent, psp_payment_id)
                → acquirer.confirm_after_challenge(psp_payment_id, idem_key)
                → AuthStatus::Settled (or whatever the PSP reports)
            → classify_decision → AttemptOutcome::Success
        ─► outcome.terminal_status = Approved
        ─► idempotency cache committed
```

The resume primitive is **stateless on the server side**: the
caller hands back the same intent body. We chose this over
server-side intent persistence because:

- The orchestrator already has the idempotency cache; that's the
  source-of-truth for "this intent was previously run."
- Persisting the intent body would duplicate the idempotency
  store with a new "pending intents" table that has the same
  contents.
- Operators with stronger "we don't trust the client to remember"
  requirements layer in their own pending-intents store on top —
  small wedge, opinionated, kept out of the reference path.

## Backward compatibility

Both new trait methods have default implementations:

- `CardAcquirer::confirm_after_challenge` defaults to
  `Err(UnsupportedMethod)`. Existing PSP drivers compile
  unchanged; calling resume against them surfaces a clean error.
- `RailAdapter::resume` defaults to `SoftFailure { code:
  "resume_not_supported" }`. A2A and Crypto adapters that don't
  have a challenge concept inherit this and the resume call
  returns a soft-failure decline rather than panicking.

No existing tests broke. The 1047 tests from Phase 29 still
pass; the 3 new tests are pure additions.

## Idempotency semantics

`Orchestrator::resume` calls `self.idempotency.commit(&intent_key,
&outcome)` after a successful resume. That means:

- A duplicate `POST /v1/intents` with the same idempotency key
  (after resume completed) returns the cached **resumed**
  outcome, not the original `RequiresCustomerAction`.
- A retry of `POST /v1/intents/resume` with the same intent +
  psp_payment_id calls the PSP again. PSPs that honor the
  idempotency-key header (Stripe, Adyen, Hyperswitch) return the
  cached decision; PSPs that don't get a new call but should
  return the same status since the challenge is one-shot.

The idempotency cache is committed only after a successful
classify; if the resume itself transport-fails (`SoftFailure`),
the cache is updated with the soft-failure outcome — re-running
the same intent body will get the cached soft-fail rather than
re-attempting authorization. Operators with stricter "always
re-attempt on soft fail" requirements call `release()` on the
idempotency store before re-running.

## HTTP surface

```
POST /v1/intents/resume
{
  "idempotency_key": "intent-3ds-1",
  "amount_minor": 1500,
  "currency": "USD",
  "method": { "type": "vault", "token": "tok_v7_3ds" },
  "rail": "card",
  "driver": "hyperswitch",
  "psp_payment_id": "psp_42abcd..."
}
```

Response shape is identical to `POST /v1/intents` — same
`IntentResponse` JSON. Operators can write a single handler for
both endpoints on the consumer side.

Emitted events use distinct types (`intent.resumed.approved`
etc.) so webhook consumers can distinguish first-attempt vs
post-challenge approval if they care.

## Honest concerns (carry-forward)

- **No server-side pending-intent store.** The caller carries
  the intent body across the challenge window. Operators wanting
  server-tracked challenges add a thin `PendingIntent` store on
  top (one table keyed on intent idempotency key, expires after
  72 hours).
- **A2A and Crypto adapters have no resume.** Their challenge
  models are different — A2A "challenges" are typically bank-app
  approvals that complete asynchronously via webhook; crypto
  flows don't have a customer-side challenge in the same sense.
  The default `SoftFailure { code: "resume_not_supported" }`
  surfaces the situation cleanly when operators try.
- **No challenge-window enforcement.** A `psp_payment_id` that's
  been pending for six weeks will still be accepted by the
  endpoint; the PSP returns the appropriate "expired" error and
  we surface it as a `SoftFailure`. Operators with strict
  windows reject at the HTTP layer (a tower middleware that
  checks elapsed time before forwarding).
- **No 3DS-on-retry escalator.** Phase 28's option mentioned a
  per-attempt 3DS preference escalator (force 3DS on the second
  attempt of a soft-declined intent). That's a smaller adjacent
  wedge — the `attempt_number` parameter is already on
  `RailAdapter::attempt`, operators can read it in their adapter
  and escalate. Shipping a built-in policy would push opinion
  onto every operator.
- **No customer-facing UI.** That's frontend. The HTTP endpoints
  support whatever flow the operator builds.

## Test totals

```
op-driver-sdk    +2  (orchestrator resume happy path, unknown driver error)
op-server        +1  (HTTP create → resume end-to-end)
                                                              ----
                                                              +3 net
```

`cargo test --workspace`: **1050 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**

## Status of the project after Phase 30

The 28 → 29 → 30 trio closed the last large operator-facing gaps:

| Capability | Phase |
|---|---|
| Webhook delivery, end-to-end | 28 |
| Multi-currency / FX primitives | 29 |
| 3DS / SCA resume | **30** |

Combined with everything before (Phases 11–27), OpenPay is now a
complete reference payment-acceptance stack with:

- Card + A2A + Crypto rails
- Refunds, disputes, settlement, subscriptions
- Single-file Minigraf persistence
- HTTP API server, driver SDK with conformance harness
- FX conversion, webhook delivery, 3DS resume
- Bi-temporal time-travel, graph-backed audit reports

A vendor can write their own driver, point `op-server` at a
single `.graph` file, wire their HTTP transport + FX feed, and
serve a SaaS-shaped payment stack — no external database, no
queue, no auxiliary service. Margin stays with the vendor.
