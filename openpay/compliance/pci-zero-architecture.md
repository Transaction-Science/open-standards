# PCI-Zero Architecture

A deployment topology where merchant application servers never see a primary account number (PAN) and therefore qualify for SAQ-A or SAQ-A-EP rather than SAQ-D. The cardholder-data environment (CDE) collapses to a single tier — the vault — and every other OpenPay component runs outside it.

The runnable counterpart of this document is `examples/pci-zero`. That binary wires the same components in-process so the data-flow is auditable as a single Rust program; the *deployment* topology described here splits those components across hosts and networks.

## Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Customer browser / device                       │
│                                                                        │
│   • collection-iframe.js  (served from vault.example.com domain)       │
│   • TLS 1.3 to vault host (cert-pinned)                                │
│   • PAN entry happens here; merchant JS bundle CANNOT read the iframe  │
│     (Same-Origin Policy isolates the iframe document)                  │
└────────────────────────────────────────────────────────────────────────┘
            │                                       │
            │ raw PAN (TLS 1.3)                     │ window.postMessage(VaultRef)
            ▼                                       ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│ CDE network segment              │    │ Merchant application network     │
│ ──────────────────────────────── │    │ ──────────────────────────────── │
│                                  │    │                                  │
│  ┌────────────────────────────┐  │    │  ┌────────────────────────────┐  │
│  │ Vault service              │  │    │  │ Merchant app server        │  │
│  │ • op-vault impl            │  │    │  │ • op-server                │  │
│  │ • mTLS-fronted HTTP API    │  │    │  │ • op-orchestrator          │  │
│  │ • detokenize/tokenize/     │  │    │  │ • op-rails-card driver     │  │
│  │   delete/exists            │  │    │  │ • op-ledger / op-graph     │  │
│  │                            │◄─┼────┼──┤   stores                   │  │
│  │ Holds: encrypted PAN +     │  │mTLS│  │                            │  │
│  │ pci-scope build            │  │    │  │ Holds: VaultRef tokens     │  │
│  └──────────────┬─────────────┘  │    │  │ NEVER holds: raw PAN       │  │
│                 │                │    │  │ Build: pci-scope = OFF     │  │
│                 ▼ unwrap DEK     │    │  └──────────────────────────────┘ │
│  ┌────────────────────────────┐  │    └──────────────────────────────────┘
│  │ KMS / HSM                  │  │                  │
│  │ • AWS KMS  /  GCP KMS  /   │  │                  │ Vault::detokenize
│  │   HashiCorp Vault Transit  │  │                  │ via mTLS, only at
│  │   / Fortanix DSM           │  │                  │ acquirer submit time
│  │ • FIPS 140-2 L3 or L4      │  │                  ▼
│  │ • Holds: KEKs only         │  │    ┌──────────────────────────────────┐
│  └────────────────────────────┘  │    │ Acquirer (Hyperswitch / direct)  │
│                                  │    │ • PCI-DSS Level 1 service        │
└──────────────────────────────────┘    │   provider; their scope, not     │
            ▲                            │   ours, once PAN is posted      │
            │                            └──────────────────────────────────┘
            │ Vault::detokenize (mTLS)
            │ initiated by op-rails-card during authorize()
            └──────────────────────────────────────────────────────────────
```

### Trust boundaries

1. **Browser ↔ Vault.** The collection iframe runs from a domain controlled by the vault, not the merchant. The merchant's outer page receives a `VaultRef` via `window.postMessage`; the merchant's JavaScript bundle cannot read the iframe document and so cannot read the PAN. This is the same pattern Stripe Elements, Adyen Components, and Braintree Drop-in use to keep merchants on SAQ-A.

2. **Merchant app server ↔ Vault service.** Only at acquirer submit time, the merchant's `op-rails-card` driver code calls `Vault::detokenize(VaultRef)` over a mutually authenticated TLS connection to the vault service. The detokenized `CardData` is serialised straight into the acquirer payload and the local Rust value drops + `Zeroize`s before the function returns.

3. **Vault service ↔ KMS / HSM.** The vault never holds raw key material in memory long-term. The data-encryption key (DEK) used to encrypt each tokenized PAN is wrapped under a key-encryption key (KEK) held by the KMS / HSM. On `detokenize`, the vault calls the KMS to unwrap the DEK, decrypts the ciphertext, and immediately zeroises the DEK. Pattern matches AWS KMS *envelope encryption* and HashiCorp Vault *Transit* secret engine.

4. **Vault service ↔ acquirer.** The vault may post directly to the acquirer ("network token" pattern with Visa Token Service / Mastercard MDES) or hand `CardData` back to the merchant app server for a single submit call. The example uses the latter; both designs are valid, and the second still keeps the merchant on the connected-to-scope side because the merchant process holds `CardData` for the duration of the acquirer HTTP round-trip and then drops it.

## PCI-DSS v4.0.1 requirements satisfied by this topology

| Req. | Title | How this topology satisfies |
|---|---|---|
| 1.4 | Boundary security between trusted and untrusted networks | CDE network segment terminates at the vault host. Inbound rules: mTLS from app servers, TLS 1.3 from browsers (cert-pinned). No other ports. |
| 2.2.1 | Configuration standards | Vault host runs a single binary built with `--features op-vault/<backend>` plus `--features op-core/pci-scope`. No general-purpose OS services. |
| 3.2.1 | Account data is not stored after authorization unless there is a legitimate business need | The merchant app server does not store PAN. Card-on-file goes into the vault under `TokenizationPolicy::card_on_file()`; everything downstream sees only `VaultRef`. |
| 3.3.1 | SAD is not stored after authorization | EMV Track 2 equivalent never persists past the `op-emv` parse step; the parsed blob is forwarded to the acquirer encrypted and not retained. |
| 3.5.1 | PAN is rendered unreadable wherever it is stored | Vault stores AES-256-GCM-SIV ciphertext (RFC 8452, misuse-resistant). The DEK is wrapped by KMS-held KEK. |
| 3.6.1 | Cryptographic keys are managed throughout their lifecycle | KMS / HSM (AWS KMS, GCP KMS, HashiCorp Vault, Fortanix DSM) provides rotation, access control, and audit logging on the KEK. See `hsm-kms-guidance.md`. |
| 3.7.1 | Key-management policies and procedures are documented | Operators document KEK rotation schedule (typical: 365 days for KEKs, 90 days for DEKs), separation of duties (KMS admin ≠ vault admin), and emergency-rotation runbook. |
| 4.2.1 | Strong cryptography during transmission over open / public networks | TLS 1.3 between browser and vault; mTLS between merchant app server and vault; TLS 1.2+ between vault and acquirer (acquirer's requirement). |
| 6.2.4 | Engineering practices to prevent or mitigate common software attacks | `Money` is integer minor units; no `f64` floats. `Payment<S>` is a typestate machine; refund-before-capture is a compile error. `CardData` has `Zeroize` + `ZeroizeOnDrop`. `Vault` trait does not distinguish "not found" from "auth failed" (no oracle). |
| 8.3.1 | Strong authentication for all access into the CDE | Vault service requires mTLS for app-server callers (workload identity) and operator MFA for human admins. |
| 10.2.1 | Audit logs capture all individual user accesses to cardholder data | Vault emits a structured `tracing` event on every `tokenize` / `detokenize` / `delete`, with the calling workload identity, timestamp, `VaultRef`, and outcome. Logs ship to a write-once audit store. |
| 11.4.1 | External and internal penetration testing | Performed against the CDE segment (the vault) and the connected-to-scope boundary (the app server). Out-of-scope crates do not require dedicated pen-testing. |

## What this topology does **not** cover

- **EMV tap.** If the device is a Linux kiosk with a tap reader (see `examples/kiosk-linux`), the EMV blob from the reader's secure element is its own SAD-handling concern. Use a P2PE-validated reader so the EMV path stays SAD-free between the chip and the acquirer.
- **Stored card-on-file.** Card-on-file (recurring billing) requires the vault to persist long-lived `VaultRef → ciphertext` mappings. The vault becomes a long-running database; treat as Req. 3.5 storage scope.
- **Refunds.** A refund references the original `psp_payment_id`, not the PAN, so refunds run entirely outside the CDE.

## Pointer to runnable example

`examples/pci-zero/` ships a single-process Rust binary that demonstrates the same data-flow in code: a fake browser submits PAN to an in-process vault, the vault returns a `VaultRef`, the orchestrator runs the intent through a card rail, and the rail driver calls `Vault::detokenize` once at acquirer submit time. The example asserts via type-level evidence (`#![forbid(unsafe_code)]`, no `RawPan` import outside `op_vault`, no `CardData` value crossing process boundaries) that the merchant code path never holds raw PAN.
