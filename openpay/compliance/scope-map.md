# PCI-DSS v4.0.1 Scope Map

This document declares, for each of the 26 crates in the OpenPay workspace, whether that crate is **in scope**, **connected-to-scope**, or **out-of-scope** for PCI-DSS v4.0.1, and why. The categorisation follows the PCI SSC *Information Supplement: Guidance for PCI DSS Scoping and Network Segmentation*.

## How the `pci-scope` cargo feature works

`op-core` declares an off-by-default cargo feature named `pci-scope`. With the feature **off**, the `PaymentMethod` enum exposes exactly four variants — `Vault(VaultRef)`, `Wallet(WalletRef)`, `Emv(SecureBlob)`, `A2a(A2aKey)` — none of which carry raw PAN. With the feature **on**, a fifth variant `PaymentMethod::RawPan(RawPan)` becomes visible and the `op_core::method::pci::RawPan` type becomes constructible.

Only `op-vault` enables `pci-scope` unconditionally. Through cargo's workspace feature unification, any crate that transitively depends on `op-vault` (notably `op-orchestrator`, which depends on `op-vault` directly) also sees the `RawPan` variant at compile time, but the orchestrator's match arm for `RawPan` returns `Error::DesignViolation` rather than ever passing the variant down the rail. The variant exists in the type but cannot enter through the orchestrator's public API.

The cardholder-data environment (CDE) in code is therefore the set of:

1. `op-vault` and its in-process implementations
2. Any platform vault adapter the operator implements (iOS Keychain, Android Keystore, KMS-encrypted blob store, HSM-backed remote vault)
3. The rail driver code path that calls `Vault::detokenize` immediately before posting to the acquirer

Everything else in the workspace handles only `VaultRef`, which under PCI SSC *Tokenization Product Security Guidelines* §3.3 carries no value for PAN recovery and is therefore out of scope.

## Per-crate scope

| Crate | PCI-DSS scope | Handles CHD / SAD? | Notes |
|---|---|---|---|
| `op-core` | conditionally in scope | only with `pci-scope` feature ON | Defines `Money`, `Payment<S>`, `RailKind`, and the gated `RawPan` type. Default build is out of scope; turning on `pci-scope` brings the binary into scope and triggers Requirements 3, 4, 6, 8, 10. |
| `op-iso20022` | out of scope | no | Facade over the `open-payments-iso20022` ISO 20022 message library. PACS/PAIN/CAMT messages carry account numbers, not PANs; A2A IBANs are payment-account identifiers, which PCI DSS does not regulate. |
| `op-emv` | in scope | yes — BER-TLV blobs from EMV kernels may contain Track 2 equivalent data (SAD per Req. 3.3) | EMV tap blobs from Secure Enclave / StrongBox are encrypted at the chip level. The blob crossing this crate is opaque ciphertext until decrypted in the HSM-backed acquirer path, but the *handling* of the blob is in scope because it represents a card-present transaction. |
| `op-vault` | **in scope, CDE** | yes — `CardData` holds raw PAN inside the process | The CDE in code. `CardData` is constructible only inside a `pci-scope`-enabled caller, has `Zeroize` + `ZeroizeOnDrop`, and exposes only `first_six` / `last_four` / `exp_*` to non-crate callers. The reference `InMemoryVault` is for tests; production deployments wire an HSM- or KMS-backed implementation. |
| `op-fraud` | out of scope | no — receives features extracted from `VaultRef` + `Money` + device signals | Pure scoring logic. The `Scorer` trait inputs are typed risk features; PAN never reaches the scorer. |
| `op-rails-card` | connected to scope | yes — calls `Vault::detokenize` on the submit path | The driver crate is the only consumer of `CardData` outside `op-vault`. Drivers serialise PAN into the acquirer payload immediately and the local `CardData` value drops + zeroises. Requirements 4.2.1 (TLS to acquirer) and 8.6 (acquirer creds) attach here. |
| `op-rails-a2a` | out of scope | no — account numbers, not PANs | FedNow / PIX / SEPA Instant credit-transfer drivers. ISO 20022 `pacs.008` payloads carry IBANs / routing+account pairs, which fall under bank regulation (NACHA, BSA, AML) but not PCI DSS. |
| `op-rails-crypto` | out of scope | no — public chain addresses + ERC-20 calldata | Stablecoin gateway. EVM addresses are public; the `EvmSigner` trait protects private keys, not cardholder data. Outside PCI DSS scope; FinCEN MSB rules apply instead. |
| `op-ffi-swift` | out of scope | no — opaque handles only | Swift bridge. The PAN never crosses the C ABI; only `VaultRef` strings and `Payment<S>` handles cross into Swift. |
| `op-ffi-jni` | out of scope | no — opaque handles only | Android JNI bridge. Same discipline as the Swift bridge. |
| `op-wasm` | out of scope | no — opaque handles only | Browser bridge. The browser handles a one-time-token returned by the vault's collection iframe; the JS side never sees a PAN. |
| `op-orchestrator` | connected to scope | no, but sees `RawPan` variant in its match arm | The match arm for `PaymentMethod::RawPan` returns `Error::DesignViolation` rather than touching the value. Connected to scope per the *PCI SSC Scoping Guidance* §2.3 (segmentation-connected systems) because it directs traffic to systems that are in scope (the vault and the rail driver). Requirements 1, 2, 6, 8, 10 attach. |
| `op-ledger` | out of scope | no — `VaultRef` only | Double-entry ledger. Account-level book entries do not include PAN; the masked first-six / last-four are sufficient for reconciliation. |
| `op-webhook` | out of scope | no — event payloads do not include PAN | Stripe-shaped outbound delivery with HMAC signing. Webhook payloads include `psp_payment_id`, `VaultRef`, `Money`, masked card metadata only. |
| `op-graph` | out of scope | no — `VaultRef` only | Minigraf-backed persistence for ledger / webhook / reconciliation / refund / dispute / settlement / subscription / idempotency stores. No PAN is stored. |
| `op-reconciliation` | out of scope | no — masked card metadata only | Diffs ledger entries against CAMT.053 bank statements via deterministic UUID v5 keys derived from masked metadata. |
| `op-refund` | out of scope | no — references prior `Payment<Captured>` by id | Refund state machine. Original payment id and `Money` are sufficient. |
| `op-dispute` | out of scope | no — chargeback evidence is the merchant's, not the cardholder's | Dispute workflow. Evidence attachments are operator-supplied PDFs / images; the crate does not introspect them. |
| `op-settlement` | out of scope | no — NACHA payout files carry merchant bank account info | Batch settlement and NACHA file generation. Merchant ACH details are not PCI scope; they are GLBA scope. |
| `op-server` | connected to scope | no, but is the network ingress | Axum HTTP server. Requirements 1.4 (boundary), 2 (config hardening), 4.2.1 (TLS), 6.4 (web-app vulnerabilities), 8 (auth), 10 (logging) attach. The server itself never holds CHD: card-collection traffic terminates at the vault, not at this server. |
| `op-driver-sdk` | out of scope | no — deterministic mocks | Driver-conformance harness with `DeterministicCardAcquirer` / `DeterministicA2aGateway` / `DeterministicCryptoGateway`. Used in CI; not deployed. |
| `op-subscriptions` | out of scope | no — `VaultRef` for card-on-file | Recurring billing. Re-uses long-lived `VaultRef` produced by the vault under `TokenizationPolicy::card_on_file()`. |
| `op-fx` | out of scope | no — `Quote`, `QuoteProvider` only | Foreign-exchange conversion. |
| `op-cli` | connected to scope | no, but operator credentials cross it | `op` operator CLI. Requirement 8 (auth) attaches via the API-key environment variable used to call `op-server`. |

## Build profiles

The PCI-DSS-relevant cargo features are:

- `op-vault/in-memory` — pulls in `aes-gcm-siv` and `rand_core`. Enables `InMemoryVault`. **Not for production.** Use only for tests, the `kiosk-linux` demo, and the `pci-zero` example.
- `op-core/pci-scope` — exposes the `RawPan` variant and `op_core::method::pci::RawPan`. Enabled transitively by `op-vault`. Any production CDE component must build with this feature; everything else builds without it.

The recommended deployment build:

```bash
# CDE: vault service binary
cargo build --release -p <your-vault-service> --features op-vault/<your-backend>

# Connected systems: orchestrator + rail drivers + server
cargo build --release -p op-server -p op-orchestrator -p op-rails-card -p op-rails-a2a

# Out-of-scope systems: ledger / reconciliation / settlement / subscriptions
cargo build --release -p op-ledger -p op-reconciliation -p op-settlement -p op-subscriptions
```

In a PCI-zero topology (see `pci-zero-architecture.md`), the merchant application server compiles only out-of-scope and connected-to-scope crates; the in-scope vault binary is a separate process, on a separate host, in a separate network segment.
