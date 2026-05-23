# `pci-zero` — Code-form demonstration of the PCI-zero topology

A runnable counterpart to `compliance/pci-zero-architecture.md`. Wires the same components — vault, orchestrator, card rail — in a single Rust process so the data-flow is auditable as one program.

## What this demonstrates

The binary exercises four boundary crossings:

1. **Browser → vault.** `collection_iframe::submit` constructs a `CardData` from a test PAN and calls `Vault::tokenize`. The PAN exists in the Rust process for the duration of this call and no longer; `ZeroizeOnDrop` wipes the bytes when `submit` returns.
2. **Merchant → orchestrator.** `merchant::Merchant::checkout` accepts a `VaultRef` and a `Money` amount. The merchant module deliberately does not import `op-vault`; it cannot construct a `CardData` or a `RawPan`. The orchestrator routes the intent to the registered card rail.
3. **Driver → vault.** `mock_card::PciZeroCardAcquirer::authorize` is the only place in the entire program (other than `collection_iframe`) that holds plaintext PAN. It calls `Vault::detokenize`, posts to the (mock) acquirer, and immediately `drop`s the `CardData`.
4. **Driver → acquirer.** A mock acquirer response — Settled, with the masked card number for the receipt.

The merchant code path (the `merchant` module) holds only `VaultRef`, `Money`, and `Payment<S>` typestate values. There is no `RawPan` import in the merchant module, no `CardData` value, no PAN-bearing string.

## What this does NOT demonstrate

- **A real network boundary.** "Boundary" here is Rust module visibility. In deployment the boundary is mTLS between two processes on different hosts.
- **A production cryptographic backend.** The example uses `op_vault::InMemoryVault`, the AES-256-GCM-SIV reference implementation. Production deployments use a KMS- or HSM-backed vault per `compliance/hsm-kms-guidance.md`.
- **Card-on-file flow.** The example tokenizes with `TokenizationPolicy::default()` (random, reusable, no TTL). Card-on-file uses `TokenizationPolicy::card_on_file()`; 3DS challenge flow uses `TokenizationPolicy::single_use(120)`.

## How to run

```bash
cd /path/to/open-standards/openpay
cargo run -p pci-zero
```

Expected output:

```
=== OpenPay pci-zero topology demo ===
Boundary crossings in this run:
  [B1] browser  -> vault       (tokenize, raw PAN)
  [B2] merchant -> orchestrator (VaultRef only)
  [B3] driver   -> vault       (detokenize, scoped)
  [B4] driver   -> acquirer     (raw PAN, mock)

[B1] Customer types card into vault-served iframe.
     vault returned VaultRef = tok_v7_...

[B2] Merchant runs orchestrator with VaultRef.

[B3/B4] driver detokenized + posted to mock acquirer.
       terminal status: APPROVED
       attempts: 1
       psp_payment_id: pci-zero-card_acq_ORDER-PCI-ZERO-001
         [0] rail=Card driver=pci-zero-card outcome=Settled

=== Done. Merchant code path held VaultRef only. ===
```

## Source layout

```
examples/pci-zero/
├── Cargo.toml          workspace member; depends on op-core,
│                       op-vault (in-memory), op-orchestrator,
│                       op-rails-card, op-fraud
├── README.md           this file
└── src/main.rs
    ├── mod merchant          out-of-scope code path
    ├── mod collection_iframe vault-served browser stand-in
    ├── mod mock_card         CardAcquirer that detokenizes
    └── fn main               wires the three modules together
```

## Mapping back to the deployment topology

| In-process construct | Deployment counterpart |
|---|---|
| `Arc<dyn Vault>` | mTLS HTTP API exposed by the vault service in the CDE segment |
| `InMemoryVault::ephemeral` | KMS- or HSM-backed vault implementation (see `compliance/hsm-kms-guidance.md`) |
| `mod merchant` | merchant application server in the connected-to-scope segment |
| `mod collection_iframe` | TLS-isolated iframe served from the vault's own domain |
| `mod mock_card` | real PSP driver (e.g. `op-rails-card` Hyperswitch impl) talking to the acquirer over TLS 1.2+ |
