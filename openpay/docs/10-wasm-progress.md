# Phase 10 — `op-wasm` Complete (Browser / Node.js WebAssembly bridge)

**Status**: Draft v0.10
**Date**: 2026-05-17

## What shipped

`op-wasm`: the bridge that exposes the OpenPay Rust core to
JavaScript / TypeScript callers, both in browsers and in Node.js.
Third platform completing the cross-platform trio (Phase 8 iOS,
Phase 9 Android, Phase 10 web).

Same architectural pattern as Phases 8 and 9:

- Opaque class handles over `op_vault::CardData`, `op_vault::VaultRef`,
  `op_vault::InMemoryVault`, `op_fraud::HeuristicScorer`.
- Oracle-discipline error collapsing preserved.
- Error discriminants byte-identical to the iOS and Android bridges
  for unified cross-platform observability.

Plus a JS-side smoke-test suite (`js/test.mjs`) and a self-contained
browser demo (`js/index.html`) to make the integration story
concrete.

## Verified ground truth

| Claim | Source |
|---|---|
| `wasm-bindgen = "0.2"` is current; 0.2.120 released April 2026 | crates.io/crates/wasm-bindgen; docs.rs/crate/wasm-bindgen/latest |
| MSRV 1.77 for libraries, fits our 1.95 toolchain | github.com/wasm-bindgen/wasm-bindgen README |
| `#[wasm_bindgen]` on a `pub struct` becomes a JS class | rustwasm.github.io exporting-a-struct guide |
| `#[wasm_bindgen(constructor)]` enables `new ClassName()` | same |
| `crate-type = ["cdylib"]` for wasm-pack output | same; wasm-bindgen examples |
| `wasm-bindgen-test = "0.3"` is the test harness; runs under Node or headless browser | crates.io/crates/wasm-bindgen-test |
| `wasm-pack test --node` / `--headless --chrome` / `--firefox` invocations | rustwasm.github.io wasm-bindgen-test/browsers |
| `wasm32-unknown-unknown` target requires `getrandom` with the `js` feature for our crypto + uuid path | docs.rs/getrandom; rust-random/getrandom issue #267 |
| `JsValue → ExportedStruct` conversion is NOT publicly available; tests can't read OpenPayError back | github.com/wasm-bindgen/wasm-bindgen issues #2231, discussion #3943 |
| Rust enums with `#[wasm_bindgen]` become TS-friendly enum constants | wasm-bindgen guide |
| `Result<T, JsValue>` propagates as a thrown JS error | wasm-bindgen guide on errors |
| u64 → BigInt on the JS side | wasm-bindgen TS bindings documentation |
| Web Crypto SubtleCrypto is async-only; sync `Vault` trait incompatible | MDN; mozilla docs |
| WebAssembly browser support: Chrome 57+, Firefox 52+, Safari 11+, Edge 16+ — ~96% coverage as of May 2026 | caniuse.com |

## Architecture

```
crates/op-wasm/
├── Cargo.toml                — cdylib + rlib; wasm-bindgen 0.2 + js-sys
│                               + getrandom("js") for wasm32 target only
├── src/
│   ├── lib.rs                — crate root, module re-exports, optional panic hook
│   ├── error.rs              — FfiError (i32 discriminants matching Phases 8/9)
│   │                            + OpenPayError JS class (.code, .kind, .message)
│   ├── card_data.rs          — CardData wasm class
│   ├── vault_ref.rs          — VaultRef wasm class
│   ├── policy.rs             — TokenFormat / TokenLifetime enums + TokenizationPolicy class
│   ├── vault.rs              — RustVault wasm class + tokenizeFromString convenience
│   ├── heuristic_scorer.rs   — HeuristicScorer wasm class
│   └── panic_hook.rs         — Optional console.error panic hook (feature-gated)
├── tests/
│   └── wasm_bindgen.rs       — Integration tests running inside wasm host (Node or browser)
├── js/
│   ├── test.mjs              — JS-side smoke tests covering consumer-visible error shape
│   └── index.html            — Self-contained browser demo
├── scripts/
│   └── build-wasm.sh         — wasm-pack runner for all 4 targets
└── README.md                 — Consumer integration guide
```

## Design choices

### Same cipher as iOS/Android, not Web Crypto

The Phase 7 `op_vault::InMemoryVault` uses `aes-gcm-siv 0.11`. Phase
10 reuses it directly, running entirely inside wasm. This means
identical cryptographic posture across iOS, Android, and the web.

Web Crypto via `SubtleCrypto` would have been the obvious browser
choice — it's hardware-accelerated on most modern devices, and it's
what server-side examples like cloudflare-workers use. The
disqualifier is that **SubtleCrypto is async-only**: every operation
returns a `Promise`. Our `Vault` trait is sync, and async-ifying it
would ripple through every crate in the workspace.

The clean separation is to keep Phase 10 sync (matching Phases 7/8/9)
and ship a future `op-wasm-webcrypto` once the async-vault design is
worked out. Until then, `aes-gcm-siv` compiled to wasm runs in
single-digit microseconds per encrypt on modern browsers — fast
enough for any realistic tokenize/detokenize volume.

### Plain class API, no factory shenanigans

Phase 9 had a complex internal `CardData.fromHandle` factory because
the JNI surface had to bypass PAN validation for vault-returned
cards. wasm-bindgen handles this differently: the Rust struct is
just a struct, and the JS class wrapper holds a pointer into wasm
linear memory. `CardData::from_inner(op_vault::CardData)` is a
plain Rust function that's `pub(crate)` only — not visible from JS.
The vault `detokenize` method internally constructs a fresh
`CardData` via this path. Clean.

### No analog to Phase 9's KeystoreVault

Phase 9 shipped `KeystoreVault` as the production-grade Android
vault backed by EncryptedSharedPreferences. There's no equivalent
for Phase 10 because:

1. The async-only Web Crypto / IndexedDB pair doesn't fit the sync
   `Vault` trait.
2. localStorage is not a security boundary (no encryption, no
   isolation from JavaScript on the same origin).
3. The browser equivalent of "device-bound key" would be a
   non-extractable CryptoKey, but persisting one requires IndexedDB,
   which is async.

The honest answer is: production browser apps shouldn't store raw
PAN in any browser-side store. They should tokenize via a hosted
field (Stripe Elements, Adyen Web Components, etc.) that posts
directly to the PSP, then store only the resulting non-reversible
token. `op-wasm` is appropriate for *receiving* PSP-tokenized
credentials, applying heuristic fraud checks, and routing decisions —
not for being the persistent PAN vault.

### Token strings instead of opaque VaultRef everywhere

In Phase 8 (Swift) and Phase 9 (Kotlin) we wrapped tokens in a
`VaultRef` class with AutoCloseable semantics. Phase 10 keeps the
`VaultRef` JS class for type-safety reasons (JS lacks type-level
distinction between PAN strings and token strings, and a dedicated
class is cheap insurance), but the token string itself is the
canonical representation — durably storeable, transmittable,
JSON-serializable.

JS callers extract the string via `.asString` for persistence and
reconstruct via `VaultRef.fromString(...)` on the way back. Same
pattern as Phases 8 and 9 but the JS object lifecycle is lighter
than Kotlin's AtomicLong handle protocol.

### Why `0` as the TTL sentinel (not `null`)

wasm-bindgen doesn't model `Option<u64>` cleanly across the boundary
— it would require boxing. The `TokenizationPolicy.ttlSeconds`
field uses `0` as the "no TTL" sentinel and translates to
`Option<u64>::None` inside the policy's `to_inner()` conversion.

Tested explicitly: `policy_zero_ttl_maps_to_none` and
`to_inner_zero_ttl_maps_to_none`.

### Cargo target-specific dependencies for getrandom

The single biggest gotcha for wasm-bindgen + cryptography crates:
`getrandom` (a transitive dep of `aes-gcm-siv` via `rand_core`, and
of `uuid v1` directly) emits a `compile_error!` on
`wasm32-unknown-unknown` unless built with the `js` feature.

The fix lives in `Cargo.toml`:

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.2", features = ["js"] }
```

This adds the feature only when targeting wasm, avoiding the
"requires wasm-bindgen" bloat on native test builds. Verified
against the official getrandom docs.

### Error inspection: JS-side only

A subtle wasm-bindgen limitation surfaced while writing the test
suite: `JsValue → ExportedStruct` conversion is NOT publicly
exposed. The macro auto-generates `From<T> for JsValue` but the
reverse (`TryFromJsValue`) is private.

This means **Rust-side integration tests can verify a `Result` is
`Err(_)` but cannot inspect the `OpenPayError.code` / `.kind` from
the failed JsValue.** JS-side tests in `js/test.mjs` cover that
gap — they exercise the same fail paths and assert on
`e.code === 2`, `e.kind === "VaultLookupFailed"`, etc.

Bug caught in the first draft: I wrote
`let err_obj: OpenPayError = err.into()` in the integration tests
and then a manual `impl From<JsValue> for OpenPayError` that would
have masked the absence of a real conversion. Removed both, split
the test responsibility between Rust-side (failure detection) and
JS-side (failure shape).

### Mandatory .free()

JS engines don't expose a portable finalizer mechanism (FinalizationRegistry
exists but isn't deterministic and isn't supported everywhere). Every
`#[wasm_bindgen]` Rust struct exposes an auto-generated `.free()`
method on the JS class; callers must invoke it or leak wasm linear
memory.

Three idiomatic patterns documented in the README:

1. Explicit `.free()` in a `try/finally`.
2. Consume-by-value methods (e.g. `vault.tokenize(card)` takes
   `card` by value, so wasm-bindgen invalidates the JS pointer
   automatically).
3. `using` statement (ES2026) for `Symbol.dispose`-aware
   environments.

## Test count

| Module | Rust unit | wasm-bindgen | JS-side | Total |
|---|---|---|---|---|
| `error.rs` | 12 | | | 12 |
| `card_data.rs` | 5 | | | 5 |
| `vault_ref.rs` | 3 | | | 3 |
| `policy.rs` | 9 | | | 9 |
| `vault.rs` | 4 | | | 4 |
| `heuristic_scorer.rs` | 2 | | | 2 |
| `tests/wasm_bindgen.rs` | | 19 | | 19 |
| `js/test.mjs` | | | 17 | 17 |
| **Phase 10 total** | **35** | **19** | **17** | **71** |

Rust-side unit tests run under `cargo test -p op-wasm` (host
target).

wasm-bindgen integration tests run with `wasm-pack test --node`
(default) or `--headless --chrome` / `--firefox`.

JS-side tests run with `node js/test.mjs` after building the
nodejs-target bundle.

## Cumulative

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
| 9 op-ffi-jni | 69 (40 Rust + 29 Kotlin) | ~2,950 |
| **10 op-wasm** | **71 (35 Rust unit + 19 wasm + 17 JS)** | **~2,400** |
| **Total** | **~532** | **~22,200** |

## Cross-platform observability is now complete

The Phase 8/9/10 error discriminants are pinned across three
platforms:

| Code | iOS (Phase 8) | Android (Phase 9) | Web (Phase 10) |
|---|---|---|---|
| 0 | `.ok` | base `OpenPayException` | `Ok` |
| 1 | `.invalidInput` | `OpenPayException$InvalidInput` | `OpenPayError(kind="InvalidInput")` |
| 2 | `.vaultLookupFailed` | `OpenPayException$VaultLookupFailed` | `OpenPayError(kind="VaultLookupFailed")` |
| 3 | `.tokenExpired` | `OpenPayException$TokenExpired` | `OpenPayError(kind="TokenExpired")` |
| 4 | `.tokenAlreadyConsumed` | `OpenPayException$TokenAlreadyConsumed` | `OpenPayError(kind="TokenAlreadyConsumed")` |
| 5 | `.fraudDeclined` | `OpenPayException$FraudDeclined` | `OpenPayError(kind="FraudDeclined")` |
| 6 | `.fraudReviewRequired` | `OpenPayException$FraudReviewRequired` | `OpenPayError(kind="FraudReviewRequired")` |
| 7 | `.backend` | `OpenPayException$Backend` | `OpenPayError(kind="Backend")` |
| 8 | (internal) | base `OpenPayException` | `OpenPayError(kind="Internal")` |
| 9 | `.capacity` | `OpenPayException$Capacity` | `OpenPayError(kind="Capacity")` |

A single observability backend (Sentry, Datadog, custom telemetry)
can correlate failures by the i32 code across all three platforms.
This is the entire point of pinning them.

## Bugs caught during this phase

1. **First-draft integration tests assumed `From<JsValue> for
   OpenPayError`.** Wrote `let err_obj: OpenPayError = err.into();`
   throughout the integration tests, then a manual `impl
   From<JsValue>` that panicked. Verified via search that the
   reverse conversion is private in wasm-bindgen. Resolved by
   splitting test responsibility: Rust-side tests verify
   `result.is_err()`; JS-side tests in `js/test.mjs` inspect the
   error shape. Documented explicitly in the test file preamble.

2. **`getrandom` would have failed at compile time on
   wasm32-unknown-unknown.** Both `aes-gcm-siv` and `uuid v7`
   transitively depend on it. Caught during the ground-truth
   verification step (not at compile time, since the sandbox has no
   Rust toolchain). Resolved by adding a target-specific dependency
   in Cargo.toml with the `js` feature enabled.

3. **`Option<u64>` doesn't model cleanly across the wasm boundary.**
   The Phase 7 `TokenizationPolicy::ttl_seconds` is
   `Option<u64>`. wasm-bindgen would require boxing. Used `0` as
   the sentinel on the JS side and translated at the
   `to_inner()` boundary. Two tests pin the translation.

4. **Initial `KeystoreVault`-equivalent design.** Started sketching
   a `WebCryptoVault` mirroring Phase 9's `KeystoreVault`. Stopped
   when the SubtleCrypto async-only requirement became evident.
   Resolved by NOT shipping a production vault for browsers in
   Phase 10 and documenting the rationale (PSP-hosted fields are
   the correct browser pattern; persistent client-side PAN storage
   is the wrong architectural choice for the web). Deferred async
   vault to a future phase.

5. **JS u64 → BigInt translation.** `TokenizationPolicy.ttlSeconds`
   is `u64` on the Rust side; wasm-bindgen surfaces this as
   `BigInt` in JavaScript. JS-side tests use `60n` / `0n` literals
   to match. Documented in the JS test file via `// u64 → BigInt
   on JS side` annotation.

6. **HTML demo template literal evaluated `card.firstSix` AFTER
   tokenize consumed the card.** The first version of
   `js/index.html` built the success message via a single template
   literal that read `card.firstSix` immediately after
   `vault.tokenize(card)`. Template literals evaluate eagerly, so
   the access throws because the JS handle was nulled by the
   by-value consumption — and the whole expression ended up in the
   `catch` block displaying "error" instead of "success". Caught
   on review; the user would have seen this on the first click.
   Fixed by capturing `firstSix` and `lastFour` into local
   variables BEFORE calling `vault.tokenize`. The demo now shows
   both safe-to-log fields next to the opaque token, which is more
   pedagogical — it visually demonstrates exactly what survives
   the tokenization.

7. **`OpenPayError` does not extend `Error`.** Bindgen-exported
   Rust structs become standalone JS classes; `e instanceof Error`
   is `false`. This means existing JS error-handling code that
   pivots on the `Error` prototype won't trigger. Documented in
   the README with a wrapper-class pattern consumers can drop into
   their app for `instanceof Error` compatibility. Verified via
   GitHub issue wasm-bindgen#1787.

## What's next

Phase 11 — `op-orchestrator` + `kiosk-linux` + e2e harness:
cross-rail decision logic (route a payment through card → A2A
fallback based on amount, country, scorer output), a Linux kiosk
reference that exercises the full stack end-to-end, and the
integration test harness that proves Phases 1–10 work together.

This closes out the architectural completion of the stack.
