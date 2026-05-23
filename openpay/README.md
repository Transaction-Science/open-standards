# OpenPay

**An open, reference payment-acceptance stack in Rust.**

A single safety-critical core compiled to every platform — iOS, Android, browser, Linux — that runs the full payment lifecycle: tap-to-pay, vault, on-device fraud, multi-rail routing, double-entry ledger, reconciliation, settlement, refunds, disputes. Apache-2.0. No per-transaction fee to OpenPay.

OpenPay is for merchants, banks, fintechs, and public-sector deployers who want to accept payments without paying a tax on every transaction to a hosted SaaS, and without ceding control of the routing, fraud, and ledger logic that decides whether their business runs.

---

## Why this exists

The modern payment stack is built around assumptions that no longer hold:

- **Card networks aren't the only rail anymore.** FedNow, PIX, UPI, SEPA Instant clear in seconds, at near-zero cost, with no chargeback window. A merchant that can route across rails — card to A2A, primary PSP to backup PSP — captures revenue that a card-only stack drops on the floor.
- **Fraud has to be decided *before* the rail.** A2A settlements are irrevocable in <20s. There is no "refund the fraud later" with FedNow or PIX. The fraud decision has to run on the merchant device, pre-routing, in milliseconds — not in a cloud-hosted post-hoc batch job.
- **The ledger is the system of record.** Most stacks treat the ledger as a derived view of PSP webhooks. OpenPay treats it as a first-class append-only graph with bi-temporal history, so you can ask "what was this account's balance on May 4th, as we knew it on May 5th?" as a single query, not a six-week archaeology project.
- **Software shouldn't tax money movement.** Stripe Terminal, Adyen, Square — all charge per-transaction fees on top of interchange. OpenPay is Apache-2.0 software. You pay the rails. You don't pay OpenPay.

---

## What's in the box

OpenPay is a Rust workspace of 26 crates + 3 examples organized into five tiers. Each tier depends only on the tiers below it.

```
┌──────────────────────────────────────────────────────────────────────┐
│  TIER 5 — FFI + CLI                                                  │
│  op-ffi-swift   op-ffi-jni   op-wasm   op-cli                        │
│  iOS / macOS    Android      browser   operator command line         │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 4 — Deployment                                                 │
│  op-server  op-refund  op-dispute  op-settlement  op-subscriptions   │
│  HTTP API   refunds    disputes    batch payouts  recurring billing  │
│  op-driver-sdk  op-fx                                                │
│  driver authors FX conversion + quote providers                      │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 3 — Orchestration & ledger                                     │
│  op-orchestrator   op-ledger   op-reconciliation   op-graph          │
│  routing + retry   double-      ledger ↔ bank      Minigraf-backed   │
│  + idempotency     entry        statement diff     stores (Datalog,  │
│  + 3DS resume      append-only                     bi-temporal)      │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 2 — Rails & risk                                               │
│  op-rails-card   op-rails-a2a   op-rails-crypto   op-fraud   op-webhook │
│  Hyperswitch     FedNow / PIX   USDC on Base /    on-device  Stripe-shaped │
│  driver          / SEPA Instant Ethereum / etc.   scoring    outbound │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 1 — Foundation                                                 │
│  op-core   op-iso20022   op-emv   op-vault                           │
│  typed     PACS/PAIN/    EMV TLV  token-only                         │
│  money,    CAMT          parser   storage                            │
│  typestate                                                           │
│  payments                                                            │
└──────────────────────────────────────────────────────────────────────┘
```

| Crate | Responsibility |
|---|---|
| `op-core` | Domain types: `Money`, `Currency`, `Payment<S>` typestate machine, `PaymentMethod`, `RailKind`. No `f64` money. PAN never appears in default builds. |
| `op-iso20022` | Thin facade over the `open-payments-iso20022` crates. PACS/PAIN/CAMT message helpers and `Camt053Source` for statement parsing. |
| `op-emv` | BER-TLV codec for parsing EMV tap blobs from iOS Secure Enclave / Android StrongBox. |
| `op-vault` | Token-only storage. `Vault` trait + reference `InMemoryVault` (AES-256-GCM-SIV). |
| `op-rails-card` | `CardAcquirer` trait + Hyperswitch driver + Stripe-compatible patterns. |
| `op-rails-a2a` | `A2aAcquirer` trait + drivers for FedNow, PIX (Bacen SPI), SEPA Instant (RT1/TIPS). Direct ISO 20022, no PSP intermediary. |
| `op-rails-crypto` | Stablecoin rail. `CryptoGateway` trait + `EvmJsonRpcGateway` (feature `evm`) for USDC on Base / Ethereum / Polygon / Arbitrum, with a hand-rolled ERC-20 calldata encoder. `LocalKeyEvmSigner` (k256 + RLP, EIP-155) ships behind the same feature for hot-wallet operators; production deployments wire Fireblocks / KMS / multisig behind the `EvmSigner` trait. |
| `op-fraud` | On-device scoring. `Scorer` trait with three backends: heuristic, ONNX (via `ort`), and pure-Rust Burn. |
| `op-webhook` | Outbound event delivery with Stripe-compatible signing + auto-disable on chronic failure. Real HTTPS delivery via `ReqwestTransport` (feature `reqwest-transport`); `MockTransport` for tests. |
| `op-orchestrator` | Routing across rails, idempotency, soft-failure fallback, 3DS / SCA resume. The "main" library API. |
| `op-ledger` | Double-entry append-only ledger. `Transaction`, `Entry`, `Account`, `Balance`. |
| `op-reconciliation` | Ledger vs. bank-statement diffing. Deterministic UUID v5 matching. |
| `op-graph` | Minigraf-backed implementations of every domain store (`LedgerStore`, `WebhookStore`, `ReconciliationStore`, `RefundStore`, `DisputeStore`, `SettlementStore`, `SubscriptionStore`, `IdempotencyStore`). Bi-temporal history, Datalog queries, audit report builder — all persisted to one `.graph` file. |
| `op-refund` | Refund state machine: `Requested → Submitted → Approved → Settled`. |
| `op-dispute` | Chargeback / dispute workflow with evidence attachment. |
| `op-settlement` | Batch settlement, holdback computation, and NACHA payout-file generation. |
| `op-subscriptions` | Recurring billing: plans, intervals (calendar-aware), trial periods, dunning policy, proration math. |
| `op-fx` | Foreign-exchange primitives: `Quote`, `QuoteProvider` trait, integer-exact `convert` with banker's rounding, `StaticQuoteProvider` + `CachedQuoteProvider` ref impls. |
| `op-server` | Axum HTTP server with env-driven main. `POST /v1/intents` + `/v1/intents/resume` + `/v1/refunds` + `/v1/disputes` + `/v1/settlement/batches` + `/v1/subscriptions` + `/v1/fx/quote` + `/v1/audit/report`. API-key auth + token-bucket rate-limit middleware. Auto-registers a `usdc-base` rail when `OP_BASE_RPC_URL` + `OP_USDC_BASE_PRIVATE_KEY` are set. |
| `op-cli` | `op` command for operator ergonomics. 14 subcommands across health, refund, dispute, batch, subscription, FX, webhooks, audit. |
| `op-driver-sdk` | For driver authors. `DeterministicCardAcquirer` / `DeterministicA2aGateway` / `DeterministicCryptoGateway` mocks + `conformance::run_card / run_a2a / run_crypto` harnesses that catch contract violations in custom drivers. |
| `op-ffi-swift` | Swift bridge via `swift-bridge` + C ABI. Generates `OpenPay.swift` + C header. |
| `op-ffi-jni` | Android bridge via JNI. Opaque handles in Kotlin; PAN never crosses the boundary. |
| `op-wasm` | Browser / Node.js bridge via `wasm-bindgen`. |

---

## Status

**v0.1.0.** This is a reference stack: the architecture is complete and the code compiles, runs, and tests cleanly across all platforms, but no part of it has been certified by a card scheme, audited by a PCI QSA, or run in regulated production. Use it as a starting point, not as one.

Workspace phases 0 through 31 are implemented end-to-end — foundation, rails (card / A2A / crypto), orchestrator with 3DS resume, ledger with bi-temporal time-travel, settlement + payout file generation, subscriptions, FX, webhook delivery, HTTP API, driver SDK with conformance harness, single-file Minigraf persistence across every domain store, and an operator bring-up sprint that ships a real EVM signer (`LocalKeyEvmSigner` via k256), env-driven config, an `op` CLI, and a `demo-merchant` example you can run on a laptop against Base mainnet.

`cargo test --workspace` is **1124 passing, 0 failing**; feature-gated paths add 28 EVM + 111 reqwest-transport tests. `cargo clippy --workspace --all-targets` is **zero warnings**.

See [`docs/`](docs/) for the per-phase progress documents that capture design intent and rationale.

---

## Quick start

### Library: take a card payment

```rust
use op_core::{Currency, Money, Payment, PaymentMethod, RailKind, VaultRef};
use op_rails_card::{
    acquirer::{AuthRequest, ThreeDsMode},
    hyperswitch::HyperswitchClient,
    CardAcquirer,
};
use uuid::Uuid;

let acquirer = HyperswitchClient::new(HyperswitchClient::SANDBOX, api_key);

let amount = Money::from_minor(10_00, Currency::USD);            // $10.00
let method = PaymentMethod::Vault(VaultRef::new("tok_test_visa"));
let payment = Payment::new(amount, method.clone(), RailKind::Card);

let auth = acquirer.authorize(&AuthRequest {
    amount,
    method,
    auto_capture: false,
    idempotency_key: Uuid::now_v7().simple().to_string(),
    three_ds: Some(ThreeDsMode::Skip),
    metadata: None,
})?;

// The typestate machine enforces lifecycle correctness at compile time.
// You cannot refund a Payment<Created>; you must Capture first.
let authorized = payment.authorize(auth.psp_payment_id.clone());
let captured   = authorized.capture(amount)?;
let refunded   = captured.refund(amount)?;
```

Full source: [`examples/headless-server/src/main.rs`](examples/headless-server/src/main.rs).

### Library: route across rails with the orchestrator

```rust
use op_fraud::HeuristicScorer;
use op_orchestrator::{Orchestrator, PolicyRouter, PaymentIntent, IdempotencyKey};
use op_core::{Currency, Money, PaymentMethod, VaultRef};

let orchestrator = Orchestrator::new()
    .with_scorer(Box::new(HeuristicScorer::new()))
    .with_router(Box::new(PolicyRouter::new(
        vec!["hyperswitch".into(), "stripe".into()],   // card driver priority
        vec!["fednow".into()],                          // A2A driver priority
    )));

// ... register CardAdapters and A2aAdapters ...

let intent = PaymentIntent::new(
    IdempotencyKey::new("ORDER-001"),
    Money::from_minor(12_99, Currency::USD),
    PaymentMethod::Vault(VaultRef::new("tok_visa_4242")),
);

let outcome = orchestrator.run(&intent)?;
//  outcome.terminal_status       → Approved / RequiresCustomerAction / Declined
//  outcome.attempts              → full retry / fallback trail
//  outcome.psp_payment_id, .uetr → settlement references
```

A replay with the same `IdempotencyKey` returns the cached outcome without touching the rail — no double charges. A soft failure on the primary PSP transparently falls back to the backup; a hard decline does not. Full source with five end-to-end scenarios: [`examples/kiosk-linux/src/main.rs`](examples/kiosk-linux/src/main.rs).

### Server

```bash
cargo run -p op-server
# POST /v1/intents     { amount_minor, currency, method, ... }
# POST /v1/refunds     { original_payment_id, amount, ... }
# POST /v1/disputes    { original_payment_id, reason, ... }
# POST /v1/settlement/batches
# GET  /v1/audit/report?tx_count=100
```

`op-server` ships with in-memory stores so it runs out of the box. Operators replace the stores with their persistent backends (e.g. `op-graph` for Minigraf, or their own `LedgerStore` impl) by wiring a different `AppState` into the exported `router(AppState)` function.

### Mobile and web

The FFI crates produce native bindings:

- **iOS / macOS:** `cargo build -p op-ffi-swift` → generates `OpenPay.swift` + C header.
- **Android:** `cargo build -p op-ffi-jni --target aarch64-linux-android` → loadable `.so` with `Java_dev_openpay_*` symbols.
- **Browser:** `wasm-pack build crates/op-wasm` → JS module.

Card data is held behind opaque handles on every FFI boundary. The PAN never crosses into Swift, Kotlin, or JavaScript.

---

## Project layout

```
openpay/
├── Cargo.toml          # workspace: 26 member crates + 3 examples
├── crates/             # the OpenPay stack
├── examples/
│   ├── headless-server/   # single-payment lifecycle via op-rails-card
│   ├── kiosk-linux/       # full orchestrator with multi-rail fallback
│   └── demo-merchant/     # generates a fresh wallet, listens for USDC on Base
├── docs/               # per-phase progress documents (NN-name-progress.md)
│   └── deploy/         # Caddyfile, systemd unit, env sample, launch.sh
└── tests/              # workspace-level integration tests
```

---

## Design principles

These aren't conventions; they're enforced by the type system or by the build.

- **No `f64` money.** `Money` is integer minor units paired with `Currency`. There is no floating-point money type in `op-core`, and you can't add `Money(USD)` to `Money(EUR)` — the operation does not compile.
- **Typestate payment lifecycle.** `Payment<Created>`, `Payment<Authorized>`, `Payment<Captured>`, `Payment<Refunded>` are distinct types. Refunding before capture is a compile error, not a runtime check.
- **No PCI scope in default builds.** The `PaymentMethod` enum has no raw-PAN variant unless you turn on the `pci-scope` feature. Orchestrators, fraud scorers, and FFI bridges only ever see `VaultRef` (opaque token) or `Emv(SecureBlob)`.
- **Append-only ledger.** Entries are never updated or deleted. Corrections are themselves entries. Bi-temporal history (via `op-graph` over Minigraf) is free, not bolted on.
- **One core, every platform.** `op-core`, `op-fraud`, `op-vault` are `no_std`-friendly and compile to iOS, Android, WASM, and Linux from the same source. The HTTP server lives in its own crate so the device builds never pull in `tokio` or `axum`.
- **Drivers are external and verifiable.** Operators write their own `CardAcquirer` / `A2aAcquirer` implementations and run `op_driver_sdk::conformance::run_card(&driver)?` to catch contract violations before deployment.

---

## Toolchain

- Rust **1.95** (edition 2024). The MSRV is pinned in `Cargo.toml` and enforced by CI.
- The release profile uses `lto = "fat"`, `codegen-units = 1`, and symbol stripping. The device-side builds are designed to be small.

---

## Documentation

The `docs/` directory holds a `NN-name-progress.md` file for each implementation phase. They are not user manuals — they capture the design choices, the alternatives considered, and the rationale that the code can't express on its own. If you want to know *why* the ledger is graph-backed, why the orchestrator owns idempotency, or why we swapped IndraDB for Minigraf mid-build, start there.

---

## Contributing

OpenPay is early. The contributions that help most right now:

- **Driver implementations** for additional PSPs and A2A rails. The `op-driver-sdk` conformance harness will tell you whether your driver behaves.
- **Backend implementations** of `LedgerStore`, `WebhookStore`, `ReconciliationStore` against your persistence layer of choice.
- **Platform glue** for the FFI bridges — Swift package, Android AAR, NPM package.
- **Test vectors** for ISO 20022 messages, EMV tags, and reconciliation scenarios.

Discussions and PRs welcome.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).
