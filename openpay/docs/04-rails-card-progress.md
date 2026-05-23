# Phase 4 — `op-rails-card` Complete

**Status**: Draft v0.4
**Date**: 2026-05-17

## What shipped

`op-rails-card`: a pluggable PSP abstraction plus a complete Hyperswitch
driver. Every PSP — Hyperswitch, Stripe, Adyen, Finix, Moov, future ones —
implements one trait. The orchestrator holds `Box<dyn CardAcquirer>` and
never knows which one it has.

Plus a working end-to-end example (`headless-server`) that takes a payment
through the full `Created → Authorized → Captured → Refunded` flow using
both the op-core typestate machine and the Hyperswitch driver.

## Modules

| File | Responsibility |
|---|---|
| `acquirer.rs` | `CardAcquirer` trait, `AuthRequest`/`AuthDecision` types, `AuthStatus` (9-variant normalized outcome). |
| `error.rs` | Sealed `Error` enum: Transport, PspRejected, MissingField, Parse, UnknownStatus, UnsupportedMethod, Core, DriverValidation. |
| `hyperswitch/wire.rs` | JSON request/response types matching Hyperswitch V1 API. Eight serde-verified shapes against canonical docs examples. |
| `hyperswitch/status_map.rs` | Maps the 17-variant Hyperswitch status enum to OpenPay's 9-variant `AuthStatus`. |
| `hyperswitch/client.rs` | HTTP client using `ureq` (sync, rustls TLS, no tokio). Implements `CardAcquirer`. |
| `tests/hyperswitch.rs` | 13 integration tests using `httpmock` — full lifecycle without network. |

## Verified ground truth

All endpoint paths, request shapes, and response status enums were
extracted directly from the Hyperswitch V1 API reference fetched live
during construction:

| Claim | Verification |
|---|---|
| `POST /payments` request shape `{amount, currency}` minimal | Verbatim from `api-reference.hyperswitch.io/v1/payments/payments--create` |
| `amount` is int64 minor units (e.g. `6540` = $65.40) | Same source |
| `capture_method` ∈ {automatic, manual, manual_multiple} | Same |
| `authentication_type` ∈ {three_ds, no_three_ds} | Same |
| Idempotency via merchant-supplied `payment_id`, 30 chars | Same |
| Auth header `api-key` only, server-side | "Make sure to never share your API key with your client application" — Hyperswitch docs |
| `POST /payments/{id}/capture` body `{amount_to_capture}` | Manual Capture quickstart, hyperswitch docs |
| `POST /payments/{id}/cancel` body `{cancellation_reason}` | Verified via Hyperswitch hex-docs and the Cancel API reference |
| `POST /refunds` body `{payment_id, amount, reason, merchant_refund_id}` | Refund API reference |
| 17 documented status enum values | Status enum on `payments--create` response schema |
| `requires_capture` follows manual auth | Manual Capture quickstart: "On successful authorization, the payment would transition to 'requires_capture' status" |
| `succeeded` after capture | Same |
| `partially_captured` / `partially_captured_and_capturable` distinction | "When capture method is manual & multiple captures are supported - partially_captured_and_capturable" — Manual Capture docs |
| Error envelope `{error: {type, message, code}}` | Hyperswitch error_codes reference |
| `next_action.redirect_to_url` for 3DS | Confirm API docs: "transition to a requires_customer_action status with a next_action block" |

## Architecture decisions

1. **Sync HTTP, not async.** We use `ureq` for the network layer. Async-fn-in-traits stabilized in Rust 1.95 but doesn't work cleanly through `dyn Trait` without `return_type_notation`. Sync keeps the trait simple and lets the FFI layer call drivers from any thread. When async + dyn stabilizes cleanly we'll migrate.

2. **`Box<dyn CardAcquirer>` from day one.** The orchestrator never knows which PSP it's calling. Adding Stripe/Adyen/Finix is a new file, not a refactor.

3. **Unknown statuses error rather than guess.** Hyperswitch adds new status values periodically. If we see one we don't recognize, we return `Error::UnknownStatus(s)` instead of falling through to a default — silently treating an unknown status as "approved" could move funds incorrectly.

4. **EMV TLV forwarded via `connector_metadata.emv_tlv_hex`.** Hyperswitch V1 doesn't have a first-class Tap-to-Pay payload field yet. We hex-encode the TLV blob and put it in `connector_metadata` where connectors that support card-present mode can pick it up. Tested: the test `emv_payload_forwarded_in_connector_metadata` verifies the wire shape.

5. **No raw PAN ever.** Driver's `supports()` accepts only `Vault`, `Wallet`, `Emv` — the three opaque-reference variants from `op-core`. Raw PAN can't even reach the driver without the `pci-scope` feature flag in `op-core`.

## Test coverage

| File | Tests | Notes |
|---|---|---|
| `acquirer.rs` | 4 | `AuthStatus` classification invariants |
| `hyperswitch/wire.rs` | 10 | JSON round-trip for every request/response type |
| `hyperswitch/status_map.rs` | 11 | All 17 documented statuses mapped; unknown errors |
| `hyperswitch/client.rs` | 8 | URL construction, supports(), currency fallbacks |
| `tests/hyperswitch.rs` | 13 | Full lifecycle via httpmock: authorize, capture, void, refund, error paths, EMV forwarding |
| **Phase 4 total** | **46** | |
| **Cumulative (Phases 1–4)** | **158** | |

## What was independently verified by Python before being asserted in Rust

- `currency_from_code("zzz")` fallback path: lowercase rejected by `try_new`, returns `unwrap_or(USD)` → USD. Confirmed by tracing the byte-validation logic.
- EMV blob hex encoding `[0x9F, 0x02, 0x06, 0×5, 0x01, 0x00] → "9f0206000000000100"`. Confirmed.
- Canonical Hyperswitch response body parses with all expected fields present.

## What's NOT yet implemented (deferred)

- **Live sandbox tests.** Feature-gated under `live-sandbox`. They need a real `HYPERSWITCH_API_KEY` and run against `https://sandbox.hyperswitch.io`. The `headless-server` example is the entry point.
- **Async migration.** Once `return_type_notation` stabilizes, drivers move to `async fn` in trait.
- **Stripe, Adyen, Finix, Moov drivers.** The trait is the abstraction; each new driver is a new module with its own wire types and status map. Will be added in Phase 4.1+ as separate sub-phases.
- **Retry/backoff layer.** Live PSPs occasionally return 5xx. A wrapper that retries `Transient` statuses with exponential backoff sits one level above the driver, not inside it.
- **Webhook ingestion.** When a customer completes a 3DS challenge, Hyperswitch fires a webhook. The receiver lives in `op-orchestrator` (later phase) and uses our `status_map::map` directly.

## Next: Phase 5 — `op-rails-a2a`

The non-card rails: FedNow direct (the prize, lowest fees), then RTP, PIX,
UPI, SEPA Instant. Each driver builds an ISO 20022 pacs.008 via the
verified `op-iso20022` builder from Phase 2 and sends it over the
appropriate transport (mTLS for PIX, FedLine for FedNow, RT1/TIPS for SEPA).
