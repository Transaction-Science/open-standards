# Ingest — Scope of the Open Standards Ecosystem

*Dated 2026-05-23.*

The three open standards in this repository are self-contained. They do not depend on running any external service. They interoperate where interoperation is useful, but they are complete on their own.

The implementation principle is direct: the best ideas from open-source and private efforts in each niche are ingested and built in Rust under permissive licences, so that each protocol is an ever-present, fully complete, free alternative.

This document records the scope of each protocol — the capabilities included and the capabilities being added — so a contributor can see at a glance what is in the ecosystem and what remains to land.

---

## Rust 1.95.0

Every Rust workspace in this repository targets **Rust 1.95.0 (edition 2024)**. The pin is at the repository root in [`rust-toolchain.toml`](rust-toolchain.toml). Per-workspace MSRV is declared in each `Cargo.toml` (`rust-version = "1.95"`). CI enforces both.

---

## openpay/

**Scope:** a complete payment-acceptance stack. Card, account-to-account, and stablecoin rails behind one orchestrator, with a typestate-enforced lifecycle and an append-only double-entry ledger.

### Included in v0.1.0

- Typed money (integer minor units, no `f64`).
- Typestate payment lifecycle (`Payment<Created>` → `Authorized` → `Captured` → `Refunded`, distinct types).
- Card rail with `CardAcquirer` trait + Hyperswitch driver.
- Account-to-account rail with `A2aAcquirer` trait + FedNow / PIX (Bacen SPI) / SEPA Instant (RT1/TIPS) drivers — direct ISO 20022 with no PSP intermediary.
- Stablecoin rail with `CryptoGateway` trait + `EvmJsonRpcGateway` (USDC on Base / Ethereum / Polygon / Arbitrum) + a hand-rolled ERC-20 calldata encoder.
- Token-only vault (`Vault` trait + AES-256-GCM-SIV reference).
- On-device fraud scoring (heuristic, ONNX, pure-Rust Burn backends).
- Outbound webhook delivery with Stripe-compatible signing + auto-disable on chronic failure.
- Multi-rail orchestrator with policy routing, idempotency, soft-failure fallback, and 3DS / SCA resume.
- Append-only double-entry ledger with bi-temporal history.
- Ledger ↔ bank-statement reconciliation via deterministic UUID v5 matching.
- Minigraf-backed implementations of every domain store (`LedgerStore`, `WebhookStore`, `ReconciliationStore`, `RefundStore`, `DisputeStore`, `SettlementStore`, `SubscriptionStore`, `IdempotencyStore`).
- Refunds, disputes, batch settlement (with NACHA payout file generation), recurring subscriptions (calendar-aware intervals, trials, dunning, proration), FX with banker's rounding.
- HTTP server (`op-server`) and operator CLI (`op-cli`) with 14 subcommands.
- Driver-SDK conformance harness (`op-driver-sdk`).
- FFI to iOS / macOS (`op-ffi-swift`), Android (`op-ffi-jni`), and browser/Node (`op-wasm`). PAN never crosses any FFI boundary.
- ISO 20022 message helpers (`op-iso20022`) and an EMV TLV codec (`op-emv`).

### Adding

Tracked in [issues labelled `openpay`](https://github.com/Transaction-Science/open-standards/issues?q=is%3Aissue+label%3Aopenpay):

- PCI-DSS scope-management materials and zero-scope reference deployment ([#1](https://github.com/Transaction-Science/open-standards/issues/1)).
- Network token provisioning (VTS + MDES) behind a `Tokenized<Card>` typestate ([#2](https://github.com/Transaction-Science/open-standards/issues/2)).
- Bi-temporal ledger lens: CLI `as_of` queries + Grafana dashboard + OpenTelemetry traces per state transition ([#3](https://github.com/Transaction-Science/open-standards/issues/3)).
- Dispute auto-evidence packager.
- MCC-aware / least-cost routing.
- Sub-merchant onboarding and KYB.
- Multi-currency settlement orchestration.
- Webhook delivery SLA + retry-policy semantics.

---

## smart-byte/

**Scope:** a complete content-addressed value-envelope substrate. Signed envelopes that carry arbitrary cargo with provenance and energy cost intrinsic, replicated by deterministic lockstep BFT across federated 8–32-node clusters.

### Included in v1 spec

- Envelope schema: identity (content-addressed hash), provenance (issuer-signed birth certificate), ownership chain (signature-bound transitions), cargo (opaque application-layer payload), joule cost (measured + estimated).
- Deterministic lockstep simulation — every node runs the identical state machine, one frame at a time.
- Byzantine supermajority commit (>2/3 cluster agreement on post-frame state hash).
- Federation by bounded clusters connected by a gossip overlay (the pattern that has run the internet).
- Per-byte content-addressed history.
- Security engineering documented in `SECURITY.md` — proven primitives, no exotic constructions.
- The Treatise (Parts I–III in full; IV–VII previewed) and the strategic-context document.

### Adding

Tracked in [issues labelled `smart-byte`](https://github.com/Transaction-Science/open-standards/issues?q=is%3Aissue+label%3Asmart-byte):

- Rust reference implementation `smart-byte-rs` — envelope + sign + lockstep gossip MVP ([#4](https://github.com/Transaction-Science/open-standards/issues/4)).
- KERI Self-Addressing IDentifiers + key-event log format ([#5](https://github.com/Transaction-Science/open-standards/issues/5)).
- Conformance test-vector pack published at `byte.transaction.science/conformance` ([#6](https://github.com/Transaction-Science/open-standards/issues/6)).
- Revocation registry.
- Privacy primitives (selective disclosure).
- Schema discovery format.
- Cross-cluster gateway protocol with formal dispute semantics.

---

## eoc/

**Scope:** a complete energy-optimised compute substrate. Every query resolves through a four-stage memoising cascade — cache → key-value → graph → neural — and a neural model is invoked only when nothing cheaper can answer.

### Included in v1 spec

- Four-stage cascade with explicit stage boundaries.
- Joules as the unit of accounting; energy is a protocol-level field, not telemetry.
- Browser-runnable / commodity-CPU posture.
- Federated, no-off-switch positioning (CC-BY-4.0 spec; no single entity runs it).
- Specification suite across `spec/` covering wire format, decay semantics, registry shape, and worked evaluation instances.

### Adding

Tracked in [issues labelled `eoc`](https://github.com/Transaction-Science/open-standards/issues?q=is%3Aissue+label%3Aeoc):

- Rust + WASM reference implementation `eoc-rs` with hardware joule counters (RAPL / NVML / `powermetrics`) ([#7](https://github.com/Transaction-Science/open-standards/issues/7)).
- Benchmark harness `eoc-bench` and the joules-per-MT-Bench-point metric ([#8](https://github.com/Transaction-Science/open-standards/issues/8)).
- GSF SCI-for-AI submission + HuggingFace cookbook recipe ([#9](https://github.com/Transaction-Science/open-standards/issues/9)).
- Cache-key canonicalisation specification.
- Learned router stage (matrix-factorisation or comparable).
- Semantic-similarity threshold calibration methodology.
- Registered MIME / JSON-Schema for the protocol envelope.

---

## How this document evolves

This file is the living scope manifest. Adding a capability begins by filing an issue with the relevant `openpay` / `smart-byte` / `eoc` label, then appearing in the **Adding** list above. Shipping a capability moves it to the **Included** list.

A new standard joins this repository by:

1. Landing in a new top-level subdirectory with its own README and licence, per [`CHARTER.md`](CHARTER.md).
2. Getting a paragraph in the root [`README.md`](README.md).
3. Adding its own section here.
