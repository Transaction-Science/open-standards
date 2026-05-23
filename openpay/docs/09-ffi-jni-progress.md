# Phase 9 — `op-ffi-jni` Complete (Kotlin / Java / Android JNI bridge)

**Status**: Draft v0.9
**Date**: 2026-05-17

## What shipped

`op-ffi-jni`: the bridge that exposes the OpenPay Rust core to
Kotlin / Java / Android callers. Same dual-surface architecture as
Phase 8, different platform mechanics:

1. **JNI bridge** — `#[no_mangle] pub extern "system" fn` functions
   named `Java_dev_openpay_<class>_<method>` that the JVM dispatches
   to from `external fun` declarations in matching Kotlin classes.
   Default surface; consumers see idiomatic Kotlin with typed
   exceptions.
2. **Plain C ABI** — identical surface to Phase 8's `c_api`. Same
   function names so a single hand-rolled C++ wrapper can drive both
   iOS and Android builds — important for shops with a unified
   cross-platform C++ core.

Plus a **`KeystoreVault`** Kotlin class — the real production vault
backed by EncryptedSharedPreferences + the Android Keystore. This is
what merchant apps actually deploy; `RustVault` is for tests.

## Why dual surfaces (again)

Same reasoning as Phase 8: JNI is the idiomatic and best-tested path
for Kotlin/Java consumers and gives them typed exceptions and
auto-loaded `.so` files. The C ABI is universally consumable and lets
cross-platform C++ codebases share a single wrapper layer.

## Why a real `KeystoreVault` and not just `RustVault`

`RustVault` is an in-memory reference vault. Its tokens vanish on app
restart, which is exactly what most merchant apps need least.
Production Android apps want persistent encrypted card-on-file
storage backed by hardware-resident keys. That's what the Android
Keystore provides.

`KeystoreVault` writes the validated PAN into
EncryptedSharedPreferences, which encrypts under a master key
managed by `MasterKey.Builder(...).setKeyScheme(AES256_GCM).build()`.
On devices with TEE / StrongBox, the master key never leaves secure
hardware. This is the same pattern Stripe-android, Adyen-android,
and Braintree-android use internally.

## Verified ground truth

| Claim | Source |
|---|---|
| `jni = "0.21"` is the stable widely-deployed version | crates.io/crates/jni; Medium "Rust in Android" May 2026 |
| `jni 0.22` exists but has breaking changes (`EnvUnowned`, mandatory `&mut self`); not yet settled | jni-rs/jni-rs releases page; CHANGELOG.md |
| JNI native signature: `pub extern "system" fn Java_<pkg>_<class>_<method>(mut env: JNIEnv, _class: JClass, ...) -> jstring` | docs.rs/jni; google.github.io/comprehensive-rust |
| `env.get_string(&jstr)?.into()` for JString→Rust; `env.new_string(s)?.into_raw()` for Rust→jstring | Vegapit + Tweede golf + Medium guides 2025-2026 |
| `env.throw_new(class, msg)` for typed exceptions; class names use `/` not `.` | Oracle JNI design spec; jni-rs docs |
| `cargo install cargo-ndk` + `rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android` | Android Developers / NDK guides |
| `cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -t x86 build --release` | cargo-ndk README; Medium Rust-Android guides |
| Output `.so` goes under `app/src/main/jniLibs/<ABI>/` | Android NDK docs `developer.android.com/ndk/guides/abis` |
| `androidx.security:security-crypto` deprecated April 2025 at 1.1.0-alpha07 | github.com/ed-george/encrypted-shared-preferences README; AndroidX deprecation notices |
| `MasterKey.Builder(ctx).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build()` is the current Kotlin API | Android Developers reference; GeeksforGeeks Dec 2025 |
| `EncryptedSharedPreferences.create(ctx, file, key, AES256_SIV, AES256_GCM)` | same |
| `dev.spght:encryptedprefs-ktx` is the maintained community fork preserving the same API | ed-george/encrypted-shared-preferences GitHub README |
| Tink 1.18+ (pulled by security-crypto 1.1.x) requires min SDK 23 | same |
| Encrypted prefs MUST be excluded from Auto Backup because Keystore key is device-bound | same; AndroidX docs on EncryptedSharedPreferences |
| AGP 9.1.0 is current stable (Feb 2026); built-in Kotlin support since 9.0; requires Gradle 9.1+ JDK 17+ | developer.android.com/build/releases/agp-9-1-0-release-notes |
| Compile SDK 36 supported by AGP 9.1.1 | same release notes |

## Architecture

```
crates/op-ffi-jni/
├── Cargo.toml                — cdylib + rlib; jni 0.21 + op-vault + op-fraud
├── src/
│   ├── lib.rs                — #![deny(unsafe_code)] crate root; jni_bridge/c_api with per-module allowance
│   ├── error.rs              — FfiError #[repr(i32)] mirroring Phase 8; exception_class() returns JNI slash-form names
│   ├── jni_bridge.rs         — Java_dev_openpay_* native functions; handle helpers; throw_ffi_error()
│   ├── c_api.rs              — Plain C ABI identical to Phase 8 (same function names: op_*)
│   └── tests.rs              — Cross-surface integration tests
├── scripts/
│   └── build-android.sh      — cargo-ndk runner for all 4 ABIs; stages .so into jniLibs/
└── kotlin/
    ├── settings.gradle.kts   — AGP 9.1.0 + Gradle 9.1+ + JDK 17
    ├── build.gradle.kts      — root project
    ├── gradle.properties     — useAndroidX, parallel, caching, config-cache
    └── openpay/
        ├── build.gradle.kts  — library module; compileSdk 36, minSdk 23
        ├── consumer-rules.pro — -keepclasseswithmembernames for native methods
        └── src/
            ├── main/
            │   ├── AndroidManifest.xml
            │   ├── jniLibs/<ABI>/   — empty until build-android.sh stages .so
            │   ├── res/xml/backup_rules.xml — example Auto Backup exclusion
            │   └── java/dev/openpay/
            │       ├── OpenPayException.kt      — sealed class hierarchy
            │       ├── CardData.kt              — AutoCloseable + AtomicLong handle
            │       ├── VaultRef.kt              — AutoCloseable + AtomicLong handle
            │       ├── TokenizationPolicy.kt    — data class + enums + helpers
            │       ├── Vault.kt                 — interface + RustVault implementation
            │       ├── HeuristicScorer.kt       — AutoCloseable wrapper
            │       └── KeystoreVault.kt         — production vault (does NOT impl Vault)
            ├── test/java/dev/openpay/
            │   └── CoreTest.kt                  — JVM unit tests (~15 tests)
            └── androidTest/java/dev/openpay/
                └── KeystoreVaultInstrumentedTest.kt — device/emulator tests (~12 tests)
```

## JNI handle protocol

The cross-boundary state is held in `jlong` handles. Each Kotlin
wrapper class owns one `AtomicLong` field:

```
Kotlin                      JNI                       Rust
────────                    ─────────                 ──────────────
new CardData(pan, m, y) ──► nativeNew ──► Box::into_raw(Box::new(CardData::new(...))) as jlong
                                          ◄────────────── jlong
card.firstSix ─────────────► nativeFirstSix(h) ─► &*(h as *const CardData) ──► card.first_six()
                                                       ◄──── &str
                                          ◄─ env.new_string(s).into_raw() ─ jstring
card.close()    ───────────► nativeFree(h) ──► Box::from_raw(h as *mut CardData) [drop]
```

Three helpers carry this:

- `box_to_handle<T>(b: Box<T>) -> jlong` — wrap a box for return.
- `unsafe handle_as_ref<'a, T>(h: jlong) -> Option<&'a T>` — borrow.
  Returns `None` for the zero handle (Kotlin's "uninitialized" sentinel),
  saving the caller from having to throw before calling JNI.
- `unsafe handle_drop<T>(h: jlong)` — free via `Box::from_raw`.
- `unsafe handle_take<T>(h: jlong) -> Option<T>` — consume; used by
  `tokenize` because the CardData must move into the vault and the
  Kotlin handle must be invalidated.

`AtomicLong` on the Kotlin side guards against the rare double-close
race. `getAndSet(0L)` is the only mutation; subsequent reads see
zero and throw `InvalidInput`.

## Throwing exceptions correctly

Per JNI: throwing an exception via `env.throw_new(class, msg)` does
not abort the native function. The function must return a sensible
default value (0, null, JNI_FALSE) **after** throwing; the JVM
materializes the exception once the function returns. Subsequent JNI
calls in the same native body would fail because of the pending
exception — you'd need `env.exception_clear()` first.

`throw_ffi_error(env, FfiError, msg)`:

1. Look up the class name from `FfiError::exception_class()` (uses
   `/` not `.`).
2. Call `env.throw_new(...)`. On success, the JVM has a pending
   exception ready to fire on return.
3. On `Err` (class not found — e.g., running outside an Android app
   with stripped resources, or under a host JVM without our Kotlin
   classes), call `env.exception_clear()` then fall back to
   `java/lang/RuntimeException`. We never leave the JVM in an
   inconsistent state.

## `KeystoreVault` design notes

The architecturally honest design decision: **`KeystoreVault` does
not implement the `Vault` interface.**

The interface requires `tokenize(card: CardData)`, but `CardData`
deliberately hides PAN bytes from Kotlin — they live inside the Rust
heap behind a `pub(crate)` accessor. That's the PCI scope boundary
by design. `KeystoreVault` therefore offers
`tokenizeFromString(pan, ...)`, which:

1. Validates the PAN by constructing a transient `CardData` and
   immediately closing it.
2. Generates a token id (`tok_v7_` + UUID v4).
3. Writes a JSON record (`pan`, `exp_month`, `exp_year`, `format`,
   `lifetime`, `ttl_seconds`, `issued_at`, `consumed`) into
   `EncryptedSharedPreferences`.
4. Returns a `VaultRef`.

Detokenize reads the JSON back, enforces TTL + single-use, and
reconstructs a fresh `CardData` (which re-runs Luhn — defense in
depth against tampering with the stored record).

This matches what real PSP SDKs do on Android. The interface fiction
that the same `Vault` shape works on every platform would require
either bundling all crypto inside Rust (forfeiting hardware-backed
keys) or exposing PAN bytes to Kotlin (busting the PCI boundary).
Neither tradeoff is worth it.

## Observability invariant: error codes match Phase 8

`FfiError` discriminants are byte-identical to
`op-ffi-swift::FfiError`:

| Discriminant | Variant | iOS exception/code | Android exception |
|---|---|---|---|
| 0 | Ok | (n/a) | (n/a) |
| 1 | InvalidInput | swift `.invalidInput` | `OpenPayException$InvalidInput` |
| 2 | VaultLookupFailed | `.vaultLookupFailed` | `OpenPayException$VaultLookupFailed` |
| 3 | TokenExpired | `.tokenExpired` | `OpenPayException$TokenExpired` |
| 4 | TokenAlreadyConsumed | `.tokenAlreadyConsumed` | `OpenPayException$TokenAlreadyConsumed` |
| 5 | FraudDeclined | `.fraudDeclined` | `OpenPayException$FraudDeclined` |
| 6 | FraudReviewRequired | `.fraudReviewRequired` | `OpenPayException$FraudReviewRequired` |
| 7 | Backend | `.backend` | `OpenPayException$Backend` |
| 8 | Internal | (n/a) | base `OpenPayException` |
| 9 | Capacity | `.capacity` | `OpenPayException$Capacity` |

A single observability backend can correlate failures across
platforms by matching the i32 code, which is the whole point of
Phase 8 + Phase 9 having identical discriminants.

## Oracle discipline preserved across boundary

Phase 7's `op-vault` collapses `NotFound | AuthFailed | InvalidToken`
into a single externally-visible discriminant. Phase 8's Swift bridge
preserves it. Phase 9's JNI bridge preserves it too — all three Rust
errors throw `OpenPayException.VaultLookupFailed`. Tested explicitly
in `KeystoreVaultInstrumentedTest.keystoreVault_malformedToken_collapsesToLookupFailed`
and in `CoreTest.rustVault_malformedToken_alsoCollapsesToLookupFailed`.

## Build matrix

`scripts/build-android.sh` runs `cargo ndk` with four ABI targets:

| Android ABI | Rust target | Notes |
|---|---|---|
| `arm64-v8a` | `aarch64-linux-android` | All current Android devices. Primary. |
| `armeabi-v7a` | `armv7-linux-androideabi` | Pre-2019 devices. Optional but typical. |
| `x86_64` | `x86_64-linux-android` | Emulator on Intel/AMD dev hosts. |
| `x86` | `i686-linux-android` | Legacy 32-bit emulator. Included for completeness. |

Default `--min-sdk 23` matches Tink's floor. Adjust via
`--min-sdk N` to raise it; lowering is not supported.

## Backup rules (the biggest gotcha)

Android Auto Backup will silently copy the
`EncryptedSharedPreferences` file to Google's servers and restore it
on a new device. The new device has a different Keystore master
key. Decryption silently fails the first time the user opens the
vault. Tests pass; production crashes.

Mitigation:

1. Ship a `backup_rules.xml` (we ship one at
   `kotlin/openpay/src/main/res/xml/backup_rules.xml`).
2. Reference it from `AndroidManifest.xml` via
   `android:fullBackupContent="@xml/backup_rules"`.
3. On Android 12+, also configure `android:dataExtractionRules`.

Documented prominently in `kotlin/README.md` and at the top of
`KeystoreVault.kt`.

## AndroidX deprecation note

`androidx.security:security-crypto` was deprecated at 1.1.0-alpha07
in April 2025. The API and underlying Tink crypto are still
maintained; only the wrapper library is in deprecation. Two
go-forward paths, both documented in the Kotlin README:

1. Pin to `1.1.0-alpha06` (what this module ships).
2. Switch to `dev.spght:encryptedprefs-ktx` — identical API,
   imports change from `androidx.security.crypto.*` to
   `dev.spght.encryptedprefs.*`.

The KeystoreVault uses the AndroidX class names; a one-line import
change is all that's needed to switch.

## Test count

| Module | Rust unit | Kotlin host | Kotlin device | Total |
|---|---|---|---|---|
| `error.rs` | 11 | | | 11 |
| `c_api.rs` | 13 | | | 13 |
| `jni_bridge.rs` | 10 | | | 10 |
| `tests.rs` (cross-surface) | 6 | | | 6 |
| Kotlin `CoreTest.kt` | | 16 | | 16 |
| Kotlin `KeystoreVaultInstrumentedTest.kt` | | | 13 | 13 |
| **Phase 9 total** | **40** | **16** | **13** | **69** |

Rust tests run under `cargo test -p op-ffi-jni`. The two Kotlin
suites run under Gradle (`./gradlew :openpay:test` for the host
JVM tests, `./gradlew :openpay:connectedAndroidTest` for the
device tests). Gradle invocations require a host-architecture `.so`
to be staged on `java.library.path`; the host tests document this
prerequisite in the README.

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
| **9 op-ffi-jni** | **40 Rust + 29 Kotlin = 69** | **~2,800** |
| **Total** | **~460** | **~19,600** |

## Bugs caught during this phase

1. **First draft of `KeystoreVault` implemented `Vault`.** Writing
   the `tokenize(card: CardData)` body forced a confrontation with
   the fact that `CardData` deliberately hides its PAN bytes. There
   is no way to read the PAN out of a CardData handle from Kotlin —
   nor should there be. Resolved by having `KeystoreVault` take a
   PAN string directly via `tokenizeFromString` and not implement
   `Vault`. Documented prominently in both the class doc and the
   README.

2. **First draft of `CardData.fromHandle` allocated a dummy then
   replaced.** The original code went `: this("4242...", 12, 2030)`
   to call the public constructor, then atomically swapped the
   handle. That works but allocates and immediately frees a
   throwaway Rust box on every detokenize. Resolved by making the
   primary constructor `private` and taking a `Long` directly;
   the public PAN-validating constructor is a secondary `: this(nativeNew(...))`.
   `fromHandle` is now an internal companion factory.

3. **`tokenize`-from-card on `RustVault` had to consume two
   handles.** First draft passed `card.requireHandle()` followed
   by `card.close()`. That worked but left a window where another
   thread could read the handle. Resolved by adding
   `CardData.consumeHandle()` which atomically reads-and-zeros.

4. **JNI class names must use `/` not `.`.** Caught by
   `FfiError::exception_class` test that asserts the path starts
   with `dev/openpay/`. The Oracle JNI design spec documents this
   but it's an easy mistake to make.

5. **`androidx.security:security-crypto` deprecation discovered
   live.** Initially planned to use it without comment. Verified
   via search the library was deprecated April 2025; documented
   both the AndroidX-stable path and the `dev.spght` fork path
   prominently in the README and KeystoreVault class doc.

6. **`jni 0.22` vs `0.21` choice.** Initially considered using
   `jni 0.22` since it's the latest. Verified via the jni-rs
   releases page that 0.22 was just released and has significant
   breaking changes (`EnvUnowned` + mandatory `&mut self` on most
   JNIEnv methods). Pinned to 0.21 which is the stable
   widely-deployed choice and is what the Medium guides from May
   2026 and the Android source-tree examples all use.

## What's next

Phase 10 — `op-wasm`: wasm-bindgen surface for the browser. Will
expose vault + scorer to JavaScript via `wasm-pack`, using the Web
Crypto API where it makes sense.

Phase 11 — `op-orchestrator` + `kiosk-linux` + e2e harness:
cross-rail decision logic (route a payment through card → A2A
fallback based on amount, country, scorer output), and an example
unattended-checkout kiosk demonstrating the full stack working
end-to-end on a Linux device.

Skipping the 5.1 / 6.1 / 7.1 sub-phases per the user-approved scope:
those are refinement on already-shipped surfaces and don't unblock
anything. They become well-scoped maintenance once the
architectural completion of Phase 11 lands.
