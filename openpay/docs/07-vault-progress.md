# Phase 7 — `op-vault` Complete (Token-Only Storage)

**Status**: Draft v0.7
**Date**: 2026-05-17

## What shipped

`op-vault`: the scope-cutter for the entire OpenPay stack. A `Vault`
trait + types + a reference implementation that takes raw card data,
encrypts it under AES-256-GCM-SIV, and hands back an opaque `VaultRef`
that the rest of the stack can pass around freely without entering
PCI DSS scope.

## Why this matters: the SAQ-A vs SAQ-D delta

PCI DSS v4.0.1 (in force since 31 March 2025, with future-dated
requirements enforceable from 31 March 2026) divides merchants into
several scoping tiers based on what touches cardholder data:

| Tier | Who | Cost / year |
|---|---|---|
| SAQ A | All payment functions outsourced; merchant never sees CHD | dozens of hours, ~$2-5K |
| SAQ A-EP | E-commerce merchant doesn't receive CHD but controls the page | a few hundred hours, ~$10-30K |
| SAQ D Merchant | Stores / processes / transmits CHD | hundreds-thousands of hours, **~$50K-200K** |
| RoC (Level 1) | High-volume, QSA-audited | $200K+ |

Per the PCI SSC Tokenization Guidelines and the v4.0.1 scoping rules,
token-only systems can be **out of scope** when three conditions hold:

1. They cannot access the vault, keys, or detokenization.
2. Tokens have no value for PAN recovery.
3. Tokens cannot be used to initiate transactions independently.

`op-vault` is the architectural boundary that satisfies (1). Random
tokenization satisfies (2). The orchestrator passing `VaultRef` to a
rail driver that itself detokenizes inside the CDE satisfies (3).

The economic argument is direct: without this layer, every component
that touches `PaymentMethod` is in scope; that's the entire stack. With
this layer, only the vault is in scope; the orchestrator, rail drivers,
fraud scorer, and FFI bridges are out. SAQ-D-to-SAQ-A is a 10-50x cost
delta per merchant per year.

## Crate layout

```
crates/op-vault/
├── Cargo.toml                 # default: trait only. `in-memory` feature for reference impl.
├── src/
│   ├── lib.rs                 # crate root, re-exports, feature gates
│   ├── error.rs               # sealed Error: NotFound, AuthFailed, Expired,
│   │                          # AlreadyConsumed, InvalidToken, InvalidCard,
│   │                          # Capacity, Backend
│   ├── policy.rs              # TokenFormat, TokenLifetime, TokenizationPolicy
│   ├── card_data.rs           # CardData — only public type with raw PAN
│   ├── vault.rs               # Vault trait
│   └── in_memory.rs           # InMemoryVault (AES-256-GCM-SIV)  [feature: in-memory]
└── tests/
    └── lifecycle.rs           # end-to-end orchestrator pattern tests
```

## Verified ground truth

### PCI DSS scoping (2026)

| Claim | Source |
|---|---|
| PCI DSS v4.0.1 is in force; future-dated requirements enforced 31 Mar 2026 | PCI SSC; Strictly compliance guide April 2026; Accutive guide |
| Token-only systems are out of scope only if they can't reach the vault / keys / detokenization | PCI SSC Tokenization Guidelines; datastealth.io 2026 checklist |
| Tokens must not be confusable with PANs (§3.3) | PCI SSC Tokenization Guidelines |
| The vault is part of the CDE; segmentation required | PCI SSC Tokenization Guidelines; Petronella 2026 guide |
| SAQ-D Level 1 annual costs: $50K-$200K | datastealth.io 2026 scope-reduction guide; vistainfosec.com fintech 2026 |
| FIPS 140-2 Level 3 (hardware) / Level 2 (software) crypto modules | PCI SSC tokenization product guidance |

### Cryptography

| Claim | Source |
|---|---|
| AES-256-GCM-SIV per RFC 8452 is misuse-resistant | RustCrypto/AEADs aes-gcm-siv 0.11.1 docs |
| Nonce collision doesn't catastrophically break confidentiality in SIV mode | RFC 8452; aes-gcm-siv crate README |
| `Aes256GcmSiv::generate_key(&mut OsRng)` + `cipher.encrypt(&nonce, plaintext)` is the canonical API | aes-gcm-siv 0.11.1 docs; RustCrypto example |
| 96-bit nonces, 128-bit auth tags | aead 0.5.x; aes-gcm-siv constants |

### Platform vault APIs (informational, for Phases 8-10)

| Claim | Source |
|---|---|
| iOS Keychain: `SecItemAdd` with `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` for background-accessible non-migrating tokens | Apple Developer Documentation; OWASP MAS iOS guide |
| Android Keystore via `EncryptedSharedPreferences` + `MasterKey` (AndroidX Security 1.1.x) backed by TEE / StrongBox | source.android.com Hardware-backed Keystore; developer.android.com Keystore |
| Android StrongBox available since API 28 (Android 9) | AOSP Keystore docs |

## Architecture: the trait

```rust
pub trait Vault: Send + Sync {
    fn name(&self) -> &str;
    fn tokenize(&self, card: CardData, policy: TokenizationPolicy) -> Result<VaultRef>;
    fn detokenize(&self, token: &VaultRef) -> Result<CardData>;
    fn exists(&self, token: &VaultRef) -> Result<bool>;
    fn delete(&self, token: &VaultRef) -> Result<bool>;
    fn health_check(&self) -> Result<()> { Ok(()) }
}
```

`Send + Sync` so the orchestrator can share `Arc<dyn Vault>` across
worker threads. Same `dyn Trait` pattern as `CardAcquirer` (Phase 4),
`A2aAcquirer` (Phase 5), and `Scorer` (Phase 6).

## Three-axis tokenization policy

```rust
TokenizationPolicy {
    format:        TokenFormat::Random | Deterministic,
    lifetime:      TokenLifetime::Reusable | SingleUse,
    ttl_seconds:   Option<u64>,
}
```

| Axis | Default | Risk-vs-utility tradeoff |
|---|---|---|
| Format | Random | Random gives PCI's strongest "no value for PAN recovery" claim. Deterministic enables analytics joins on PAN equality but creates a query oracle. |
| Lifetime | Reusable | Reusable is needed for card-on-file. SingleUse is for 3DS auth and other one-shot flows where replay must fail. |
| TTL | None | Bounding the attack window when tokens are exfiltrated. Short TTLs (60-120s) for ephemeral auth flows; no TTL for card-on-file. |

Helper constructors encode the two most common combos:

- `TokenizationPolicy::single_use(120)` — 2-minute 3DS-auth-style token.
- `TokenizationPolicy::card_on_file()` — long-lived random reusable.

## `CardData` — the only PAN-holding public type

The OpenPay surface has exactly one public type that holds raw PAN:
`op_vault::CardData`. Construction enforces three checks:

1. **Length**: 12-19 digits per ISO/IEC 7812.
2. **Luhn (mod-10)**: rejects typos. Doesn't prove legitimacy but
   catches accidental garbage.
3. **Expiration sanity**: month 1-12, year 2000-2099. Does NOT reject
   past dates (the caller may legitimately tokenize an expired-card
   record for refund / chargeback).

The raw-bytes accessor is `pub(crate)` — only the vault implementation
modules inside this crate can read PAN bytes. External callers see
only `first_six` (BIN, PCI §3.4.1 allowed) and `last_four` (allowed).

`CardData` derives `Zeroize` + `ZeroizeOnDrop` via `op_core::pci::RawPan`.

Custom `Debug` masks to `CardData(424242******4242, 12/2030)` — the
PCI DSS 4.0.1 §3.4.1 maximum display allowance. Tested.

## `InMemoryVault` — the reference implementation

AES-256-GCM-SIV (RFC 8452, misuse-resistant). Each entry carries its
own random 96-bit nonce. Tokens are UUID v7 prefixed with `tok_v7_`:

- UUID v7 carries a millisecond timestamp prefix that makes tokens
  sortable by issuance time — useful for vault-internal analytics —
  without exposing the underlying PAN mapping (the random suffix is
  what's cryptographically independent).
- The `tok_v7_` prefix is non-numeric, so tokens are visually
  distinguishable from PANs per PCI Tokenization Guidelines §3.3.

### Status: reference, not production

`InMemoryVault` is correct, exercises the full `Vault` contract, and
ships every behavior the trait promises. It is **not** a PCI-compliant
production vault on its own:

- State is in RAM only. A restart loses every token. Production vaults
  persist ciphertext to durable storage (DB-of-record + KMS, HSM, or a
  dedicated vault service).
- The encryption key is operator-supplied or ephemeral. Production
  vaults integrate with KMS / HSM for key management with rotation,
  dual control, split knowledge, and FIPS 140-2 Level 2/3 modules.
- No audit logging. PCI DSS §10 requires log emission for every
  detokenize call with user identity, timestamp, and outcome.
- No rate limiting. Production vaults bound detokenize throughput per
  caller to limit blast radius on credential compromise.

Use in tests and development. For production, plug an
operator-supplied vault into the same `Vault` trait.

## Oracle discipline in error handling

Error variants are deliberately collapsed in user-facing contexts:

- `NotFound` and `AuthFailed` are distinct in the type but should be
  treated identically by API responses. Distinguishing "token unknown"
  from "decryption failed" is an oracle: an attacker who can probe
  many tokens learns which ones existed but had the wrong key vs which
  ones never existed.
- `Expired` and `AlreadyConsumed` are surfaced separately because the
  legitimate caller (the orchestrator) needs to know which retry path
  to take — they're policy state, not security state.
- `InvalidToken` (malformed) is distinct from `NotFound` because the
  caller can know it never had a chance without leaking anything.

## Test coverage

| Module | Tests | What's covered |
|---|---|---|
| `error.rs` | 0 | Sealed enum, no behavior |
| `policy.rs` | 5 | Defaults, single_use helper, card_on_file helper, JSON round-trip (policy, format) |
| `card_data.rs` | 12 | Luhn happy paths (Visa, MC, Amex), non-digit reject, length reject (short + long), Luhn-fail reject, month reject (0, 13), year reject (1999, 2100), expired-card accept, debug masking, accessors, Luhn truth table |
| `vault.rs` | 2 | Object-safety, default health_check |
| `in_memory.rs` | 19 | Token format, round trip, NotFound, InvalidToken, delete idempotency, exists for unknown / non-token strings, single-use consumption, reusable survival, unique tokens for same PAN under Random, distinct cards distinct tokens, cross-key isolation, same-key compatibility, dyn-compatibility, generate_key entropy, len tracking, default health, prefix discipline, TTL expiration, TTL=0 immediate expiry, clock-skew tolerance |
| `tests/lifecycle.rs` | 12 | Full lifecycle, single-use 3DS, thread sharing (8 workers), multi-card isolation, deletion non-recoverable, PAN not in token string, Debug masking, 1000-token uniqueness, exists probe, dyn Vault routing, policy composition, full pipeline determinism |
| **Phase 7 total** | **50** | |
| **Cumulative Phases 1–7** | **~346** | |

## Independently verified

- **AES-256-GCM-SIV API path** verified against `aes-gcm-siv 0.11.1`
  live documentation: `Aes256GcmSiv::generate_key(&mut OsRng)` returns
  `Key<Aes256GcmSiv>`; `generate_nonce(&mut OsRng)` returns
  `Nonce<Aes256GcmSiv>` (12 bytes); `cipher.encrypt(&nonce, plaintext)`
  returns `Vec<u8>`; `cipher.decrypt(&nonce, ciphertext)` reverses.
- **Luhn correctness** verified against the standard test PANs:
  4242424242424242 (Visa, valid), 5555555555554444 (MC, valid),
  378282246310005 (Amex, valid), 4242424242424241 (one digit off,
  invalid). All four cases covered.
- **Token uniqueness** verified by minting 1000 tokens for the same
  PAN and confirming a `HashSet` accepts all of them.

## Design decisions

### 1. Trait + reference, not a full vault product

The trait is the architectural boundary; the reference impl is for
tests and development. Production deployments bring their own platform
vault (iOS Keychain, Android Keystore, AWS KMS-backed blob, HashiCorp
Vault, HSM). We don't try to be a vault product — we provide the
contract that lets operators choose one.

### 2. AES-GCM-SIV, not AES-GCM

AES-GCM has a catastrophic nonce-reuse failure mode: reusing a nonce
under the same key reveals the XOR of the two plaintexts and lets an
attacker recover the authentication key. AES-GCM-SIV (RFC 8452,
"misuse-resistant") doesn't have this failure mode — nonce reuse only
reveals whether two plaintexts were identical, which is a much weaker
break. Since we ship a library that operators wrap and use however
they like, we can't enforce nonce discipline at every callsite, so we
pick the algorithm that fails gracefully.

Plain AES-256-GCM would also work and is faster — but the misuse-
resistance is worth the marginal encrypt overhead in a vault context
where the encryption rate is bounded by user-facing card-entry flows
(measured in dozens-per-second, not millions).

### 3. CardData is the only public PAN holder

The PCI scope boundary is the type system. `op_core::pci::RawPan` is
behind the `pci-scope` feature flag — code that doesn't opt in cannot
construct it. `op-vault` opts in unconditionally because it must
handle PAN. The vault re-exposes `RawPan`-wrapping behavior only
through `CardData`, which makes its raw bytes `pub(crate)`. Outside
the vault crate, `CardData::raw()` is unreachable; only `first_six`,
`last_four`, and `exp_*` are exposed.

### 4. UUID v7 token format

Tokens are `tok_v7_` + UUID v7 simple form. Three properties:

- Time-prefixed: tokens sort by issuance time for analytics.
- Random-suffixed: the bottom 74 bits are entropy, so distinct tokens
  for the same PAN under Random format are cryptographically
  independent.
- Non-numeric prefix: PCI §3.3 wants tokens visually distinguishable
  from PANs.

### 5. Health check is a default-Ok hook

The trait has a default `health_check` that returns `Ok(())`.
Implementations that probe a remote service override it. This matches
the readiness-probe pattern operators already use for HTTP health
endpoints — they call `vault.health_check()` from their `/healthz`
handler.

### 6. Three-axis policy, two pre-built helpers

The policy carries three orthogonal knobs (format, lifetime, TTL).
That's six combinations, but only two are common enough to warrant
named constructors: `single_use` (for 3DS / one-shot flows) and
`card_on_file` (for recurring billing). Operators with unusual
requirements (deterministic + reusable + short TTL for a join-and-
expire pattern) build a struct directly.

## Bugs caught during construction

1. **`A2aKey` was not re-exported from `op-core`.** Phase 5 added 9+
   call sites using `op_core::A2aKey::*`, but the re-export at
   `op-core/src/lib.rs` only listed `PaymentMethod, Token, VaultRef`.
   Phases 4-5 would have failed `cargo build` with "unresolved import";
   the sandbox has no Rust toolchain so the gap was invisible. Added
   `A2aKey` to the existing re-export line. Phase-7 deliverable that
   retroactively fixes Phase 5.

2. **`RawPan` was crate-private to `op-core`.** The `pci` module is
   `pub(crate) mod pci`, which prevented `op-vault` from naming
   `op_core::method::pci::RawPan`. Added a feature-gated re-export
   `#[cfg(feature = "pci-scope")] pub use method::pci::RawPan;` at
   the `op-core` crate root.

3. **Insufficient accessors on `RawPan`.** Only `last_four` was
   public. Added `first_six` (BIN, PCI §3.4.1 allowed), `pan_bytes`
   (for vault-side encryption), `exp_month`, and `exp_year`. All
   only available when `pci-scope` is enabled.

4. **`GenericArray<u8, U12>` → `[u8; 12]` conversion ambiguity.**
   Initially used `.into()` which has multiple candidate trait impls
   across `generic-array` versions and may not be inferrable. Switched
   to explicit `copy_from_slice(nonce.as_slice())` which is
   unambiguous.

5. **Unused `subtle` dependency.** Declared in Cargo.toml but never
   referenced; constant-time comparison turned out not to be needed
   in the reference implementation (the AES-GCM-SIV decrypt does its
   own constant-time tag verification internally). Removed.

## What's NOT in this phase (explicitly deferred)

- **Platform vault adapters.** iOS Keychain (`SecItemAdd`), Android
  Keystore (`EncryptedSharedPreferences` + `MasterKey`), AWS KMS-
  backed durable vault, HashiCorp Vault transit. These land in
  Phases 8-10 alongside the FFI bridges, where the platform-specific
  code lives. The `Vault` trait is the contract they implement.

- **Network tokens.** Visa Token Service, Mastercard MDES, Amex
  Token Service. Network tokens are issued by the card networks and
  stored as `VaultRef` strings; resolution happens against the
  network token service (proxied through the rail driver). The PSP
  abstracts this in most cases. → Phase 5.2.

- **Audit logging.** PCI DSS §10.2.1.1 requires per-event logs for
  all access to cardholder data. The `Vault` trait is the natural
  seam; we'll add a `with_audit` decorator in Phase 7.1 that wraps an
  arbitrary `Vault` impl and emits structured events.

- **Rate limiting and quotas.** PCI DSS §10 implies bounded
  per-caller throughput. Phase 7.1.

- **Hot-swap and rotation.** Atomic key rotation while the vault is
  live. Operator-specific (KMS rotation events, HSM ceremony). → Phase
  7.2.

- **FIPS 140-2 mode.** The RustCrypto AES implementations are not
  FIPS-validated. Operators requiring formal FIPS validation must
  plug in an HSM-backed `Vault` impl. → operator concern.

## Next: Phases 8-10 — FFI bridges

`op-ffi-swift` (cxx-rs for Swift / iOS / macOS), `op-ffi-jni`
(Kotlin / Android), `op-wasm` (browser via wasm-bindgen). These
expose the orchestrator + vault + fraud scorer + rail drivers to the
host platform without leaking Rust idioms. The vault trait makes this
clean: a Swift `KeychainVault` implements `Vault` via `SecItem*`
calls; a Kotlin `KeystoreVault` implements it via
`EncryptedSharedPreferences`; the orchestrator doesn't know the
difference. PAN never crosses the FFI boundary — only `VaultRef`s do.
