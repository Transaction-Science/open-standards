# Phase 13 — `op-webhook` outbound delivery

**Status**: Draft v0.13
**Date**: 2026-05-18

## What shipped

A new crate `crates/op-webhook` plus an integration test suite that
demonstrates a ledger-event-driven fanout against a mock HTTP
transport. **Total: 106 tests (96 unit + 10 integration), ~3,304
LOC.**

The crate is the **fanout edge** of the stack: ledger and
orchestrator events emerge here as signed HTTPS POSTs to merchant
endpoints, with retry, dead-letter, replay, and auto-disable
handling. The wire format is byte-for-byte compatible with
Stripe's so any merchant who has already integrated a Stripe
webhook can verify ours with the same code.

## Verified ground truth

Researched live (May 2026 sources) before implementation:

| Claim | Source |
|---|---|
| Exponential backoff with **full jitter** is the industry consensus for webhook retry | AWS "Exponential Backoff and Jitter" (2015); Stripe, GitHub, Shopify "built-in exponential backoff with jitter" (Hooklistener 2026) |
| **72-hour total retry window** | Stripe ("retries failures with exponential backoff for up to 72 hours") |
| **24-hour window then auto-disable** | Razorpay ("retry the delivery in exponential backoff policy for 24 hours") |
| **Auto-disable after ~10 consecutive failures** | Hookdeck Outpost, dev.to webhook-system guide |
| **Stripe header format**: `Stripe-Signature: t={unix_secs},v1={hex_hmac_sha256(secret, "{ts}.{body}")}` | Stripe docs, Hooklistener 2026, multiple independent implementations confirm |
| **5-minute default timestamp tolerance window** | Stripe SDKs, Hooklistener 2026 |
| **Constant-time comparison** (`subtle::ConstantTimeEq`) required to prevent timing attacks | Stripe security guide, OWASP, multiple references |
| **Retry on**: 5xx, 408, 425, 429, transport failures | Hookdeck, integrate.io 2026 |
| **Don't retry on**: 2xx, 3xx, and 4xx other than 408/425/429 | Hookdeck, docsfordevs |
| `hmac` crate API: `Hmac::<Sha256>::new_from_slice(secret)`, `mac.update(bytes)`, `mac.finalize().into_bytes()` | docs.rs/hmac |

## Architecture

```
crates/op-webhook/
├── Cargo.toml          — deps: op-core, serde, thiserror, uuid, subtle, sha2, hmac
├── src/
│   ├── lib.rs          — module roots + pub re-exports
│   ├── error.rs        — 11-variant Error enum
│   ├── hexutil.rs      — tiny in-crate hex encoder (no `hex` dep)
│   ├── signing.rs      — Stripe-compatible HMAC-SHA256 + verify with constant-time compare
│   ├── retry.rs        — RetryPolicy trait, ExponentialBackoffPolicy, jitter source seam
│   ├── event.rs        — WebhookEvent, DeliveryAttempt, DeliveryStatus
│   ├── endpoint.rs     — Endpoint, EndpointId, EndpointStatus, filter matching
│   ├── transport.rs    — HttpTransport trait, MockTransport
│   ├── store.rs        — WebhookStore trait + InMemoryWebhookStore
│   └── dispatcher.rs   — WebhookDispatcher: dispatch / replay / process_due_retries
└── tests/
    └── integration.rs  — 10 tests demonstrating ledger-event-driven fanout
```

### Data flow

```
                  ┌──────────────────────────┐
                  │       WebhookEvent       │
                  │  (opaque Vec<u8> body)   │
                  └────────────┬─────────────┘
                               │
              ┌────────────────┴──────────────────┐
              │       WebhookDispatcher           │
              │                                   │
              │   for each matching endpoint:     │
              │     1. compute HMAC over          │
              │        "{now}.{body}"             │
              │     2. build OpenPay-Signature    │
              │        header                     │
              │     3. transport.send()           │
              │     4. classify response:         │
              │        2xx → Delivered            │
              │        5xx/408/425/429 → Retry    │
              │        other 4xx → DeadLetter     │
              │        transport err → Retry      │
              │     5. record DeliveryAttempt     │
              │     6. update endpoint            │
              │        consecutive_failures       │
              │     7. auto-disable if threshold  │
              │        crossed                    │
              └───────────────────────────────────┘
```

## Key design decisions

### 1. Stripe-compatible signing scheme

```
OpenPay-Signature: t=1700000000,v1=4f4c...
```

The signed payload is `"{ts}.{body}"`; the HMAC key is the
endpoint's secret. This means an existing Stripe-compatible
verifier (`stripe.Webhook.constructEvent`, or any community
implementation) verifies our payloads with only the header NAME
changed. Operators get a 30-second integration.

We accept multiple `v1=` entries in a single header, which is
Stripe's mechanism for key rotation. We don't currently emit
multiple — that's a future extension.

### 2. Exponential backoff with full jitter, NOT equal jitter

Three jitter strategies are documented in the literature:

- **Full**: `delay = rand(0, base * 2^n)`. Most spread, best at
  avoiding thundering herd. **Our default.**
- **Equal**: `delay = base * 2^(n-1) + rand(0, base * 2^(n-1))`.
  Guarantees some minimum delay.
- **Decorrelated**: `delay = rand(base, prev * 3)`. Adaptive.

Full jitter matches AWS's 2015 recommendation and what
Stripe/GitHub/Shopify appear to do based on observed behavior. The
[`RetryPolicy`] trait lets operators substitute a different
strategy without touching the dispatcher.

### 3. Pluggable HTTP transport, no real client in this crate

We deliberately don't depend on `reqwest`, `hyper`, or `ureq`.
Operators inject one through the [`HttpTransport`] trait. This
keeps the dep footprint minimal and decouples the async/sync story
from the dispatcher.

The reference [`MockTransport`] is the test fixture: programmable
per-call responses, request capture for assertions.

### 4. Pluggable jitter RNG for deterministic tests

`RetryPolicy::next_delay_secs` calls a [`JitterRng`] trait
([`SystemJitter`] in production, [`FixedJitter`] in tests). This
keeps tests reproducible without mocking the system clock for the
jitter alone.

### 5. Auto-disable after N consecutive failures

Per the Razorpay / dev.to community patterns. After
`disable_after_consecutive_failures` consecutive failures across
ANY events for one endpoint, the endpoint's status flips to
[`EndpointStatus::AutoDisabled`] and `list_active_endpoints_for`
filters it out. Operators must explicitly re-enable. Any
successful delivery resets the counter to 0.

### 6. Explicit replay, never silent resurrection

After auto-disable or a 4xx dead-letter, events stay in the store
indefinitely. Operators trigger
[`WebhookDispatcher::replay(event_id, endpoint_id)`] manually after
fixing the root cause. The dispatcher never silently retries a
dead-lettered attempt.

**Defensive default**: even `replay()` honors `EndpointStatus`'s
`is_blocking()`. Operators wanting to replay against a disabled
endpoint must first re-enable it. This avoids "I clicked replay
and immediately tripped the auto-disable I was trying to recover
from" footguns.

### 7. The dispatcher is sync; async is operator-level

Matches the rest of the stack. Operators run N dispatcher threads
(one per worker pool) or wrap in `tokio::task::spawn_blocking`.

### 8. Tiny local hex encoder, not the `hex` crate

The `hex` crate is 8KB compiled but it's another supply-chain
surface to audit. We need only encode/decode; 50 lines of code,
7 unit tests, done.

### 9. Constant-time signature comparison

`subtle::ConstantTimeEq::ct_eq` is used in `verify_signature`. A
naive `==` on hex strings would short-circuit on the first
mismatching byte; an attacker who can measure verification latency
could extract the correct signature one byte at a time. `ct_eq`
always processes the full input.

### 10. Idempotency at the consumer, NOT at the dispatcher

Webhook delivery is *at-least-once* by design. Consumers must
dedupe by `WebhookEvent.id` (sent as the `OpenPay-Event-Id`
header). The dispatcher doesn't try to enforce
exactly-once-delivery (an impossible property over an unreliable
network).

## What this crate does NOT do

- **No real HTTP client.** Operators inject one.
- **No async runtime.** Sync trait surfaces; operators wrap.
- **No durable queue.** `InMemoryWebhookStore` is single-process.
  Production deployments swap in Postgres/Redis-backed stores.
- **No CloudEvents formatting.** The payload is opaque; operators
  encode however they like.
- **No key rotation flow.** A single secret per endpoint. The
  signature scheme already supports multiple `v1=` entries during
  rotation — extension is documented as future work.
- **No PII redaction.** Operators redact upstream.
- **No bulk signing.** One event, one HMAC computation per
  endpoint. Bulk operations are out of scope.

## Composition with the rest of the stack

The 10-test integration suite at `tests/integration.rs` shows the
operator pattern: a synthetic `ledger.transaction.posted` payload
flows through the dispatcher to a merchant endpoint, with the
merchant's `verify_signature` call (acting as receiver) succeeding
when the payload is intact and failing when tampered.

```
┌──────────────────┐         ┌───────────────────┐         ┌──────────────────┐
│   op-ledger      │ posted  │   op-webhook      │ HTTPS   │  merchant ERP    │
│  transaction ────┼─event───►   dispatcher      ────POST──►   verify_sig()   │
│                  │         │                   │         │   process()      │
└──────────────────┘         └───────────────────┘         └──────────────────┘
```

Notably **`op-webhook` has no compile-time dependency on
`op-ledger`**. The payload type is `Vec<u8>`; the event type is a
free-form string. Decoupled, by design.

## Bugs caught during construction

1. **Function names starting with digits.** Initial draft had
   `fn 5xx_response_schedules_retry`, `fn 4xx_response_dead_letters_immediately`,
   `fn 429_response_schedules_retry` — all illegal Rust
   identifiers. Renamed to `five_hundred_three_response_schedules_retry`,
   `four_oh_four_response_dead_letters_immediately`,
   `four_twenty_nine_response_schedules_retry`.

2. **Filter glob semantics.** Initial test used filter `"payment.*"`
   expecting it to match `"payment.authorized"`, but the filter
   implementation is **literal compare** (with `*` as the
   "everything" wildcard, not a glob meta-character). Either:
   (a) implement true glob matching, (b) change the test to use
   the documented literal/wildcard semantics. Picked (b) because
   adding glob matching brings either a new dep (`glob` crate) or
   ~80 lines of pattern code with its own test surface, and the
   intended OpenPay event taxonomy is short enough (low single
   digits) that operators can list filters explicitly. Documented
   in the [`Endpoint::matches`] doc comment.

3. **Workspace dep alignment.** `subtle` was already in
   `[workspace.dependencies]` at version 2.6; my crate manifest
   originally pinned its own `subtle = "2"`. Aligned to
   `subtle = { workspace = true }` for consistency.

4. **Tiny hex crate temptation.** First draft was going to add the
   `hex` crate for `hex::encode`/`hex::decode`. Caught and replaced
   with a 50-line local implementation (`hexutil.rs`) plus 7 unit
   tests. Saves a supply-chain audit and a transitive
   `serde + serde_derive` chain on some `hex` versions.

5. **Audit trail in `process_due_retries`.** First draft only
   created NEW attempts on retry without marking the old one
   terminal. `list_due_retries` would have re-picked the OLD
   attempt forever. Fixed by marking the prior attempt as
   `DeliveryStatus::Failed` (the "superseded" sentinel) before
   creating the new attempt.

6. **Replay vs disabled endpoint.** First draft of `replay()` did
   NOT check `EndpointStatus::is_blocking()`, so operators could
   accidentally re-trigger an auto-disable storm by replaying
   against a still-disabled endpoint. Fixed to honor the blocking
   check; operators must re-enable first. Documented + tested.

7. **Excerpt length on response bodies.** Bodies returned by a
   misconfigured receiver could be MB-sized HTML pages. The
   dispatcher truncates to 512 bytes for the audit record
   (`RESPONSE_BODY_EXCERPT_BYTES`). Tested in
   `excerpt_truncates_large_body`.

## Test count

| Module | Unit tests |
|---|---|
| `dispatcher.rs` | 18 |
| `endpoint.rs` | 11 |
| `event.rs` | 6 |
| `hexutil.rs` | 7 |
| `retry.rs` | 17 |
| `signing.rs` | 22 |
| `store.rs` | 10 |
| `transport.rs` | 5 |
| **Unit total** | **96** |
| `tests/integration.rs` | **10** |
| **Phase 13 total** | **106** |

### Integration test summary

1. `fanout_happy_path_signature_verifies_on_receiver_side` — full
   round-trip: dispatcher signs, mock transport captures, receiver
   verifies with the same secret.
2. `modified_body_fails_signature_verification` — tampering check.
3. `old_timestamp_outside_tolerance_is_rejected` — replay
   protection.
4. `endpoint_auto_disables_after_threshold_failures` —
   AutoDisabled flips and subsequent dispatches are skipped.
5. `fanout_multiple_endpoints_each_get_signed_independently` —
   per-endpoint secrets are isolated; one endpoint's secret can
   never verify another's signature.
6. `retry_succeeds_via_process_due_retries` — 5xx → retry → 200.
7. `replay_works_after_operator_reenables_endpoint` — recovery
   workflow.
8. `four_oh_four_dead_letters_without_retry` — 4xx fails fast.
9. `signature_header_is_parseable_by_stripe_style_consumers` —
   on-the-wire compatibility check.
10. `audit_trail_records_every_attempt` — every attempt persists
    for forensics.

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
| 12 op-ledger | 69 | ~2,540 |
| **13 op-webhook** | **106** | **~3,304** |
| **Total** | **~796** | **~31,944** |

## What's next

Phase 14+ candidates (not committed):

- **Postgres-backed `WebhookStore` and `LedgerStore`** —
  production-ready persistent backends; would graduate the
  in-memory references from "fine for kiosks" to "fine for
  Series-A merchants".
- **`op-reconciliation`** — diff ledger vs. PSP / bank statements;
  the webhook receiver above is the natural ingestion path for the
  reconciliation pipeline.
- **TigerBeetle-backed `LedgerStore`** — high-throughput backend.
- **OpenTelemetry trace propagation** — single trace id flowing
  from orchestrator intent → ledger metadata → webhook attempt →
  consumer.
- **CloudEvents v1.0 envelope** — optional output adapter so
  events are CNCF-compatible by default.
- **Key rotation flow** — emit both old and new `v1=` signatures
  during a configurable cutover window.
- **Endpoint health metrics** — p50/p95/p99 latency,
  delivery-success rate, drained-DLQ counts; pluggable
  `WebhookObserver` trait.
- **Async transport adapter** — `tokio::task::spawn_blocking`
  wrapper that lets tokio-native callers integrate without
  blocking the runtime.

The thesis stands: pure-Rust, Apache-2.0, no platform lock-in, every
event signed with a key the merchant controls and verifiable with
the same code they already use for Stripe.
