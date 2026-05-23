# OpenPay — Foundation Document

**Status**: Draft v0.1
**Date**: 2026-05-17
**Toolchain**: Rust 1.95.0 (stable, released 2026-04-16), edition 2024
**License**: Apache-2.0 (matches Hyperswitch, Fineract, ISO 20022 Rust crates)

---

## 1. Mission

A free, open-source, universal payment-acceptance stack that runs on any modern phone,
tablet, or x86/ARM box, integrates every major payment rail (cards, A2A/instant, wallets,
QR), and ships a single safety-critical Rust core compiled to every platform via cxx,
jni, and wasm-pack.

Not a startup. A reference implementation that any merchant, bank, fintech, or
government can deploy with no per-transaction tax to OpenPay itself.

---

## 2. The Tax Stack We Eliminate (Quantified)

For a merchant doing $1M/yr in card volume on the current US stack:

| Layer | Typical cost | Source |
|---|---|---|
| Interchange (issuer) | 1.5–3.0% | Aeropay 2026 cost analysis |
| Network assessment (Visa/MC) | ~0.13% | Aeropay 2026 cost analysis |
| Processor markup | 0.3–0.5% (often 2.9%+$0.30 blended) | Aeropay 2026 |
| POS SaaS (Toast/Square/Clover) | $69–$300/mo × 12 = $828–$3,600/yr | Vendor pricing pages |
| Hardware lease | $50–$150/mo per terminal | Vendor pricing pages |
| Chargebacks | $15–$100+ per dispute | Aeropay 2026 |
| **Total effective rate** | **3.0–5.0%** | |
| **On $1M volume** | **$30K–$50K/yr** | |

A2A / pay-by-bank rails (FedNow, RTP, PIX, UPI, SEPA Instant) charge cents per
transaction, not percentages. FedNow's 2026 fee schedule increased the per-transaction
limit from $1M to $10M; participation has grown to 1,500+ US financial institutions.
A merchant routing 50% of volume to A2A saves ~$25K/yr on a $1M business before any
SaaS savings.

---

## 3. Hard Constraints (Non-Negotiable Physics)

### 3.1 Tap to Pay on iPhone requires a PSP

Per Apple's developer documentation: "App developers who want to offer Tap to Pay on
iPhone to merchants will first need to integrate with a supported payment service
provider (PSP) who processes the payments read by Tap to Pay on iPhone and provides
the certified terminal configurations that are loaded on the merchant's device."

**Implication**: OpenPay cannot remove the PSP for in-person iPhone card acceptance.
It can (a) treat PSPs as pluggable, swappable drivers and (b) route to non-card rails
wherever the customer permits.

### 3.2 PCI DSS scope is unavoidable for card data

Any code path that touches raw PAN (Primary Account Number) puts the operator in PCI
scope. OpenPay's design must **never see the PAN** in the application layer — only
tokens issued by the PSP or by the device secure element (Apple Secure Enclave,
Android StrongBox, host card emulation).

### 3.3 ISO 20022 is the global messaging substrate

FedNow, PIX, SEPA Instant, India's NPCI (post-IMPS migration), and SWIFT all use
ISO 20022 (PACS for clearing/settlement, PAIN for initiation, CAMT for cash
management). The Rust crate family `open-payments-iso20022-*` v1.0.1 ships full
type-safe serde parsers for every message type. We adopt it directly; we don't
rebuild it.

### 3.4 We are not a money transmitter

OpenPay the project ships software, not regulated money movement. Anyone who deploys
OpenPay to clear funds becomes a money transmitter / payment institution /
authorized payment provider under their own jurisdiction's law and is responsible
for licensing, AML/KYC, and capital requirements. The codebase enforces this with a
clear separation between the `op-core` orchestration layer and `op-rails-*` adapter
crates that must be configured with the operator's own credentials.

---

## 4. Prior Art We Build On (Not Reinvent)

| Project | Role | License | Why we use it |
|---|---|---|---|
| **Hyperswitch** (Juspay) | Payment orchestration server | Apache 2.0 | Rust, PCI-cert, 175M tx/day claimed, 100+ PSP connectors |
| **open-payments-iso20022** | ISO 20022 + FedNow parsers | Apache/MIT | Type-safe pacs/pain/camt in Rust |
| **Apache Fineract** | Core banking / ledger | Apache 2.0 | Account management; we plug in via REST for ledger ops |
| **Mojaloop** | Interoperability switch | Apache 2.0 | Inter-DFSP transfers, Gates Foundation reference |
| **NPCI open source** | UPI reference stack | Apache 2.0 | Distributed payment processing platform |
| **Burn** (tracel-ai) | ML framework | Apache 2.0 / MIT | Cross-platform inference: native, WASM, no_std for embedded |
| **ort** (pykeio) | ONNX Runtime bindings | Apache 2.0 / MIT | Fast inference for production fraud models |

OpenPay's net new contribution is the **edge integration**: a single safety-critical
Rust core that runs on the merchant device (iPhone/Android/web/Linux kiosk), speaks
to Hyperswitch or any PSP for cards, speaks directly ISO 20022 to A2A rails, and
runs fraud inference on-device before any network hop.

---

## 5. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MERCHANT DEVICE (any platform)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │  iOS shell   │  │ Android      │  │ Web PWA      │  │ Kiosk   │  │
│  │  (Swift +    │  │ shell        │  │ (wasm-bindgen│  │ (Linux  │  │
│  │   cxx FFI)   │  │ (Kotlin +    │  │  + JS/TS UI) │  │  + Rust │  │
│  │              │  │  jni)        │  │              │  │  native)│  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────┬────┘  │
│         └─────────────────┴──────┬──────────┴───────────────┘       │
│                                  │                                  │
│                  ┌───────────────▼────────────────┐                 │
│                  │  op-core  (Rust 1.95, no_std-  │                 │
│                  │  compatible where possible)    │                 │
│                  │                                │                 │
│                  │  ┌──────────────────────────┐  │                 │
│                  │  │ Payment state machine    │  │                 │
│                  │  │ (typestate pattern)      │  │                 │
│                  │  └──────────────────────────┘  │                 │
│                  │  ┌──────────────────────────┐  │                 │
│                  │  │ op-emv: EMV TLV decode   │  │                 │
│                  │  │ op-iso20022: ISO parsers │  │                 │
│                  │  │ op-vault: tokens only,   │  │                 │
│                  │  │   never raw PAN          │  │                 │
│                  │  │ op-fraud: on-device      │  │                 │
│                  │  │   inference (Burn/ort)   │  │                 │
│                  │  └──────────────────────────┘  │                 │
│                  └────────┬───────────────────────┘                 │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│ op-rails-card  │  │ op-rails-a2a   │  │ op-rails-wallet│
│                │  │                │  │                │
│ adapter trait: │  │ adapter trait: │  │ Apple Pay,     │
│ → Hyperswitch  │  │ → FedNow       │  │ Google Pay,    │
│ → Stripe       │  │ → RTP (TCH)    │  │ host card      │
│ → Adyen        │  │ → PIX/SPI      │  │ emulation      │
│ → Finix/Moov   │  │ → UPI/NPCI     │  │                │
│ (PSP swappable)│  │ → SEPA Inst.   │  │                │
│                │  │ (rail-direct)  │  │                │
└────────────────┘  └────────────────┘  └────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────▼────────────┐
                │ op-ledger              │
                │ (optional; Fineract    │
                │  adapter or sled       │
                │  embedded DB for       │
                │  edge-only deploys)    │
                └────────────────────────┘
```

---

## 6. Crate Layout (Cargo Workspace)

```
openpay/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── op-core/                # state machines, domain types, traits
│   ├── op-iso20022/            # re-export + helpers over open-payments-iso20022
│   ├── op-emv/                 # EMV TLV parsing, BER-TLV codec
│   ├── op-vault/               # token-only secure storage abstraction
│   ├── op-fraud/               # on-device inference (Burn primary, ort backend)
│   ├── op-rails-card/          # PSP adapter trait + Hyperswitch driver
│   ├── op-rails-a2a/           # FedNow, PIX, UPI, SEPA Instant drivers
│   ├── op-rails-wallet/        # Apple Pay / Google Pay token handling
│   ├── op-ffi-swift/           # cxx bridge for iOS / macOS
│   ├── op-ffi-jni/             # jni bridge for Android / JVM
│   └── op-wasm/                # wasm-bindgen bridge for web / PWA
├── examples/
│   ├── kiosk-linux/            # standalone binary
│   └── headless-server/        # for testing
└── tests/
    ├── conformance/            # ISO 20022 message round-trips
    ├── emv/                    # EMV test vectors
    └── property/               # proptest invariants on state machine
```

---

## 7. Core Principles (Enforced by the Type System)

1. **No raw PAN in application memory.** The `PaymentMethod` enum carries
   `Token(VaultRef)` or `EmvTag(SecureBlob)` variants only. Constructing a
   `PaymentMethod::RawPan` requires the `pci-scope` cargo feature and is gated
   behind `#[cfg(feature = "pci-scope")]`.

2. **Typestate payment machine.** `Payment<S>` where `S: PaymentState`.
   Transitions are functions, not method calls on a mutable struct.
   Illegal transitions don't compile.

3. **Money is never f64.** All amounts are `op_core::Money { minor_units: i64, currency: Iso4217 }`.

4. **Errors are sealed enums, never `Box<dyn Error>`.** Each crate exposes
   exactly one `Error` type with `thiserror`. Callers exhaustive-match.

5. **No `unsafe` outside `op-ffi-*` and `op-emv` (raw byte parsing).**
   Enforced by `#![forbid(unsafe_code)]` at every other crate root.

6. **Every public function is `#[must_use]` when it returns a non-`()` value
   that represents a state transition or side-effect.**

7. **Cross-platform via Rust 1.95's `cfg_select!`** for clean target-specific
   bodies instead of nested `cfg_if!`.

---

## 8. What Comes Next (Sequenced)

Each phase produces verifiable artifacts; we do not move on until the prior
phase's tests pass.

- **Phase 1** (this commit): foundation doc, workspace skeleton, `op-core`
  domain types (`Money`, `Currency`, `PaymentMethod`, typestate machine
  scaffolding).
- **Phase 2**: `op-iso20022` wrapper + conformance tests against PACS.008 /
  PAIN.001 / CAMT.054 test vectors.
- **Phase 3**: `op-emv` BER-TLV codec with EMVCo test vectors.
- **Phase 4**: `op-rails-card` trait + Hyperswitch driver, mocked against
  Hyperswitch's sandbox.
- **Phase 5**: `op-rails-a2a` FedNow driver against FRB test harness.
- **Phase 6**: `op-fraud` Burn model with synthetic transaction stream.
- **Phase 7**: FFI bridges (`op-ffi-swift`, `op-ffi-jni`, `op-wasm`).
- **Phase 8**: example kiosk + headless server + end-to-end test harness.
```

