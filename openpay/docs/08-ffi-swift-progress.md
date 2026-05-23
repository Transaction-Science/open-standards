# Phase 8 — `op-ffi-swift` Complete (Swift / iOS / macOS bridge)

**Status**: Draft v0.8
**Date**: 2026-05-17

## What shipped

`op-ffi-swift`: the bridge that exposes the OpenPay Rust core to
Swift / iOS / macOS callers. Dual surface:

1. **`swift-bridge` 0.1.59** — idiomatic Swift API generated from a
   single `#[swift_bridge::bridge]` declaration. Default build path.
2. **Plain C ABI** — `extern "C"` functions with raw-pointer ownership.
   Fallback for consumers that don't want `swift-bridge-build` in
   their Xcode workflow; they hand-roll Swift wrappers via
   `@_silgen_name`.

## Why two surfaces

The compact summary distilled this from this turn's reasoning:
swift-bridge is purpose-built for Rust↔Swift and generates idiomatic
Swift directly, but adds a `swift-bridge-build` build dependency that
some teams will not accept (especially in monorepos with strict
build-tool policies). The plain C ABI is universally consumable —
Objective-C, Swift via `@_silgen_name`, C, even AppleScript via
Foundation's C bridge. Shipping both costs ~600 LOC and gives every
consumer a path.

## Why no card / A2A rails

Card and A2A rail crates are heavy on transport dependencies (ureq,
rustls, OAuth, mTLS). Most iOS apps don't submit payments directly —
they hand a tokenized credential to a PSP SDK (Stripe, Adyen, Braintree)
that handles the wire protocol. The Swift bridge therefore exposes
**vault + fraud + token handoff**, not the rail drivers. Apps that
need direct rail submission link the rail crates at the workspace
level and bridge them via a custom extension.

This is the standard architectural pattern in the iOS payment ecosystem:
the merchant app touches a tokenized credential only.

## Verified ground truth

| Claim | Source |
|---|---|
| `swift-bridge` 0.1.59 is latest stable | docs.rs/crate/swift-bridge May 2026 |
| Inspired by cxx-rs; emits idiomatic Swift directly | chinedufn/swift-bridge README |
| Supports `Option<OpaqueRust>`, `Result`, async, opaque types | swift-bridge release notes |
| Target triples: `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios` (all tier-2) | rustc book platform-support/apple-ios |
| Simulator distinction via `-sim` suffix encoded into LC_BUILD_VERSION | rustc book; Apple Developer Forums thread 678075 |
| `xcodebuild -create-xcframework` is the canonical packaging command | Apple Developer Forums thread 673387 |
| Lipo simulator arches into a fat lib before xcodebuild | Apple Developer Forums; rhonabwy.com 2023 XCFramework guide |
| iOS Keychain `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` is the correct class for background-accessible non-migrating tokens | Apple Developer Documentation |

## Crate layout

```
crates/op-ffi-swift/
├── Cargo.toml                          # crate-type=[staticlib, cdylib, rlib]
├── build.rs                            # invokes swift_bridge_build::parse_bridges
├── src/
│   ├── lib.rs                          # crate root, modules, attrs
│   ├── error.rs                        # FfiError #[repr(i32)] (11 tests)
│   ├── bridge.rs                       # swift-bridge module + impls (15 tests)
│   ├── c_api.rs                        # plain C ABI (13 tests)
│   └── tests.rs                        # cross-surface integration (5 tests)
├── scripts/
│   └── build-xcframework.sh            # full Apple toolchain workflow
└── swift/
    ├── Package.swift                   # SwiftPM manifest
    ├── README.md                       # consumer integration guide
    └── OpenPay/OpenPay.swift           # idiomatic Swift wrapper layer
```

## The bridge surface

```rust
#[swift_bridge::bridge]
mod ffi {
    // Shared enums
    enum FraudDecisionFfi { Approve, Review, Decline, Freeze }
    enum TokenFormatFfi { Random, Deterministic }
    enum TokenLifetimeFfi { Reusable, SingleUse }

    // Shared struct (transparent both sides)
    struct TokenizationPolicyFfi {
        format: TokenFormatFfi,
        lifetime: TokenLifetimeFfi,
        ttl_seconds: u64,
    }

    extern "Rust" {
        // CardData
        type RustCardData;
        #[swift_bridge(associated_to = "RustCardData")]
        fn new(pan: &str, exp_month: u8, exp_year: u16) -> Option<RustCardData>;
        fn first_six(&self) -> String;
        fn last_four(&self) -> String;
        fn exp_month(&self) -> u8;
        fn exp_year(&self) -> u16;

        // VaultRef
        type RustVaultRef;
        fn as_string(&self) -> String;

        // Vault
        type RustVault;
        #[swift_bridge(associated_to = "RustVault")]
        fn ephemeral(name: &str) -> RustVault;
        fn tokenize(&self, card: RustCardData, policy: TokenizationPolicyFfi) -> Option<RustVaultRef>;
        fn detokenize(&self, token: &RustVaultRef) -> Option<RustCardData>;
        fn exists(&self, token: &RustVaultRef) -> bool;
        fn delete(&self, token: &RustVaultRef) -> bool;

        // Scorer
        type RustHeuristicScorer;
        #[swift_bridge(associated_to = "RustHeuristicScorer")]
        fn default() -> RustHeuristicScorer;
        fn name(&self) -> String;

        // Per-thread last-error slots
        fn last_error_vault(v: &RustVault) -> i32;
        fn last_error_card() -> i32;
    }
}
```

Wrapper types `RustCardData`, `RustVaultRef`, `RustVault`,
`RustHeuristicScorer` wrap the underlying `op-vault` and `op-fraud`
types. Method bodies live in plain Rust outside the macro. Errors
populate thread-local slots that Swift reads via `last_error_*`.

## Oracle discipline preserved at the FFI boundary

The `FfiError` enum collapses `op_vault::Error::NotFound`,
`AuthFailed`, and `InvalidToken` into a single `VaultLookupFailed`
discriminant — same rule as the vault layer, surfaced consistently to
Swift. A Swift caller cannot probe to distinguish "this token exists
with the wrong key" from "this token never existed." Tested in
`error.rs::vault_not_found_and_auth_failed_collapse_to_lookup_failed`.

## Thread-local last-error pattern

Each surface (bridge, C ABI) has its own `thread_local!` Cell of
`FfiError`. A method that returns `Option<T>` / `bool` / null pointer
on failure writes the discriminant to the thread-local before
returning. Swift / C reads the slot via `last_error_*()` immediately
after a failing call.

Why thread-locals rather than out-parameters: out-parameters would
require Swift to allocate scratch space per call, which is awkward
in the swift-bridge surface where `Option<T>` is the natural Swift
idiom. The C ABI also benefits — Swift's `@_silgen_name` callers
don't need to declare `inout` parameters.

Caveat: an iOS app that mixes UI work (main thread) and a worker
thread for tokenization must be careful to read `last_error_*` on the
same thread that made the failing call. This is a documented
contract in the README.

## C ABI ownership protocol

Documented at the module level in `c_api.rs`. Summary:

- `op_*_new` / `op_*_create` returns a `*mut T` allocated with
  `Box::into_raw`. Caller frees via `op_*_free`.
- `op_*_free` on null is a no-op.
- `op_vault_tokenize` **consumes** the `OpCardData` pointer (the
  card box is dropped on the Rust side, even on failure, to honor
  the ownership contract Swift expects from `swift-bridge`'s API).
- Status-returning calls return `i32`: 0 = success, positive = data,
  -1 = error (read `op_last_error`).
- Pointer-returning calls return null on error.
- All strings are NUL-terminated UTF-8, freed via `op_string_free`.

## Build flow

`scripts/build-xcframework.sh` produces `target/OpenPay.xcframework`:

1. `cargo build --target {aarch64-apple-ios,aarch64-apple-ios-sim,x86_64-apple-ios,aarch64-apple-darwin,x86_64-apple-darwin}` — five `.a` outputs.
2. `lipo -create aarch64-apple-ios-sim.a x86_64-apple-ios.a -output sim-fat/.a` — combine simulator arches.
3. `lipo -create aarch64-apple-darwin.a x86_64-apple-darwin.a -output mac-fat/.a` — combine macOS arches.
4. `xcodebuild -create-xcframework -library aarch64-apple-ios.a -library sim-fat.a -library mac-fat.a -output OpenPay.xcframework`.
5. Copy swift-bridge generated `OpenPay.swift`, `openpay-swift-bridge.h`, and `module.modulemap` into `target/swift/` for SwiftPM consumption.

Verified workflow against the Apple Developer Forums threads on
static library packaging (the canonical pattern is lipo simulator
slices first, then `xcodebuild -create-xcframework` with one library
per platform variant).

## Test coverage

| Module | Tests | Notes |
|---|---|---|
| `error.rs` | 11 | i32 round-trip, oracle collapse, message-non-leakage, all variants |
| `bridge.rs` | 15 | CardData construction (valid/invalid), vault round-trip, NotFound→VaultLookupFailed, malformed token collapse, policy decoding (single-use, ttl=0, deterministic), delete idempotency, exists, scorer name, thread-local isolation, success-clears-error |
| `c_api.rs` | 13 | Card lifecycle, invalid PAN, null PAN, vault round-trip, detokenize unknown, exists+delete, null pointer status returns, scorer lifecycle, string_free null, free null, tokenize null-card consumption, policy decode (single-use, unknown format) |
| `tests.rs` | 5 | Cross-surface coexistence, independent thread-local error slots, ephemeral vault on both surfaces, scorer name consistency, 100-iteration string-free loop |
| **Phase 8 total** | **44** | |
| **Cumulative Phases 1–8** | **391** | |

## Design decisions

### 1. Dual surface (swift-bridge primary, C ABI fallback)

Chose swift-bridge as the default because the deliverable is Swift,
and idiomatic Swift output is what merchant teams actually want to
read. cxx-rs would require an extra C++→Swift hop. Plain C ABI is
parallel for consumers without swift-bridge-build, which costs ~250
extra LOC and gives universal compatibility.

### 2. Thread-local errors over Result types

`swift-bridge` does support `Result<T, OpaqueErr>` returns, but Swift
would see them as throwing functions where the error type is an
opaque Rust struct. We want Swift callers to see a Swift `enum
OpenPayError`, not an opaque handle. Thread-local i32 codes + a
Swift wrapper layer that converts to `OpenPayError` is the
straightforward path.

### 3. No bridged `Vault` trait

We did NOT use `extern "Swift"` to let Swift code implement the Rust
`Vault` trait. The pattern would enable a Swift `KeychainVault` that
plugs into the Rust orchestrator. We considered it and deferred:

- iOS Keychain access is async at scale (background thread, biometric
  prompts) and bridging async-Swift to sync-Rust is fragile.
- The accessibility class, biometric policy, and access control flags
  are deployment decisions, not library decisions. We would have to
  expose a configuration surface that captures all of them, which is
  iOS-specific noise in a cross-platform crate.

The path forward (Phase 7.1+): ship a `KeychainVault` Swift class
that implements its own `Vault`-like protocol entirely in Swift, with
the Rust orchestrator calling into it via an `extern "Swift"` block.
Defer until merchant feedback says it's the right factoring.

### 4. Pinned `swift-bridge = "0.1"`

The chinedufn/swift-bridge README explicitly notes that the MSSV
policy is tight ("the short support window is disrupting projects").
We pin to the 0.1 series and treat minor upgrades as project events
rather than dependabot-driven background updates.

### 5. `unsafe_code` at module level

`lib.rs` declares `#![deny(unsafe_code)]` at crate level. The `c_api`
module annotates with `#[allow(unsafe_code)]` because raw-pointer FFI
requires unsafe. The bridge module remains safe code — swift-bridge's
macro generates the FFI code separately, outside of our crate's
namespace. The unsafe footprint is small, audited, and concentrated.

Note: an earlier version used `#![forbid(unsafe_code)]` which cannot
be relaxed at module level. Switched to `deny` for that reason.

## Bugs caught during construction

1. **`forbid(unsafe_code)` blocked the C ABI module.** `forbid` is a
   hard limit that no inner `#[allow]` can override. Switched to
   `deny` + module-level `#[allow]`.
2. **Initial directory creation used a literal brace pattern.** Bash
   without `extglob` interpreted `{src,swift,scripts}` as a directory
   name. Corrected.
3. **op-core direct dep was unused.** Pulled transitively through
   op-vault and op-fraud. Removed.
4. **Bridge tests' `last_error_card` access was via `crate::bridge`.**
   The `pub fn last_error_card()` lives at module scope in `bridge.rs`,
   not inside the `mod ffi { ... }` macro block. The integration tests
   reference it correctly.
5. **`Scorer` trait import oscillated.** First version called
   `Scorer::name(&self.inner)` requiring the trait import; second
   removed the trait import in favor of method-syntax; but
   `HeuristicScorer` has no inherent `name` method, so the trait must
   be in scope. Final: trait imported, called via `self.inner.name()`.

## What's NOT in this phase (explicitly deferred)

- **KeychainVault Swift implementation.** Belongs in consumer code or
  a follow-up phase (Phase 7.1+).
- **Apple Pay PKPayment handling.** Apple Pay payment tokens are
  network tokens, not PANs. They go to a PSP that resolves the
  cryptogram. Out of scope for OpenPay.
- **Async APIs.** swift-bridge supports `async fn`, but the OpenPay
  vault surface is synchronous by design (encryption is fast; the
  bottleneck is the rail driver which lives outside the FFI). We can
  add async wrappers in a future phase if merchant apps need them.
- **Network token bridging.** Visa VTS / Mastercard MDES. These are
  rail-driver concerns, not FFI concerns.
- **Orchestrator bridging.** The orchestrator type itself isn't
  exposed yet (no separate `op-orchestrator` crate exists). Phase 11
  will add this.

## Next: Phase 9 — `op-ffi-jni` (Kotlin / Android)

Same architectural shape as Phase 8 but using the `jni` crate to bridge
to Kotlin/Java. The vault on Android maps to AndroidKeystore via
`EncryptedSharedPreferences` + `MasterKey` (AndroidX Security 1.1.x)
backed by TEE / StrongBox. Same `Vault` trait surface, different
platform adapter.
