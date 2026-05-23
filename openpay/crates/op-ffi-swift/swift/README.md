# OpenPay for Swift / iOS / macOS

Native Swift bindings to the OpenPay Rust core. Built on
[`swift-bridge`](https://github.com/chinedufn/swift-bridge) with a
parallel plain C ABI surface for consumers that don't want the
bridge-build dependency.

## What you get

- **`OpenPay.Vault`** — tokenize / detokenize card data via a Rust
  vault. Default ships an in-memory reference implementation backed
  by AES-256-GCM-SIV (RFC 8452, misuse-resistant). Production: bring
  your own backed by iOS Keychain or your KMS.
- **`OpenPay.CardData`** — validated card representation. PAN never
  becomes a Swift `String`; it lives inside the Rust core, zeroized
  on drop.
- **`OpenPay.HeuristicScorer`** — fraud scorer. Heuristic rules out
  of the box; pluggable for ONNX or Burn-based models.
- **`OpenPayError`** — typed Swift errors mirroring the FFI enum,
  with oracle-discipline collapsing of token lookup failures.

## Integration

### Option 1: Prebuilt XCFramework (recommended)

```bash
# One-time: build the artifacts.
bash crates/op-ffi-swift/scripts/build-xcframework.sh --release
```

This produces:

- `target/OpenPay.xcframework/` — prebuilt static libs for iOS device,
  iOS simulator (Apple Silicon + Intel), and macOS (arm64 + Intel).
- `target/swift/OpenPay.swift` + `openpay-swift-bridge.h` +
  `module.modulemap` — the swift-bridge generated glue.

Then in your app's `Package.swift`:

```swift
.package(path: "../openpay/crates/op-ffi-swift/swift")
```

### Option 2: C ABI only

If you don't want the swift-bridge build dependency, build with the
`c-only` feature:

```bash
cargo build -p op-ffi-swift --features c-only --target aarch64-apple-ios --release
```

Then hand-roll a Swift wrapper using `@_silgen_name` against the
`op_*` functions documented in `src/c_api.rs`.

## Required Rust toolchain

```bash
rustup target add aarch64-apple-ios       # iOS device
rustup target add aarch64-apple-ios-sim   # iOS simulator (Apple Silicon)
rustup target add x86_64-apple-ios        # iOS simulator (Intel)
rustup target add aarch64-apple-darwin    # macOS arm64
rustup target add x86_64-apple-darwin     # macOS Intel
```

These are the standard tier-2 targets per the rustc platform support
table for `*-apple-ios` and `*-apple-darwin`.

## Usage

```swift
import OpenPay

// Construct an ephemeral vault (replace with KeychainVault in prod).
let vault = OpenPay.Vault.ephemeral(name: "checkout")

// Validate a card.
do {
    let card = try OpenPay.CardData(
        pan: "4242424242424242",
        expMonth: 12,
        expYear: 2030
    )
    // Tokenize for card-on-file.
    let token = try vault.tokenize(card: card, policy: .cardOnFile())
    print("Saved token: \(token.asString)")
    // Save token.asString to your durable store (Core Data, etc.).
} catch let e as OpenPayError {
    print("Validation or tokenization failed: \(e.localizedDescription)")
}

// Later, at submit time:
let token = OpenPay.VaultRef(wrapping: /* recovered from storage */)
do {
    let card = try vault.detokenize(token: token)
    // Hand `card` to your PSP SDK (Stripe, Adyen, …). The PAN is
    // never visible to Swift code; the PSP SDK calls the Rust core
    // again via the same FFI to read it.
} catch OpenPayError.tokenExpired {
    // ask the user to re-enter
} catch OpenPayError.vaultLookupFailed {
    // token is gone — re-enroll
} catch {
    // other error
}
```

## What's NOT in this crate

- **KeychainVault.** Writing `SecItemAdd` correctly is deployment-
  specific (which accessibility class, biometric gating, background
  access). Implementing the `Vault` Rust trait from Swift is supported
  by `swift-bridge` via `extern "Swift"` blocks; the example
  implementation belongs in your app code. We can ship a reference
  implementation in a future phase if there's demand.
- **Apple Pay PKPayment unwrapping.** Apple Pay payment tokens are
  network tokens, not PANs. They go straight to a PSP that resolves
  the cryptogram; OpenPay's vault doesn't see them. Bridge in your
  PSP's PKPayment handler directly.
- **Card and A2A rails.** These are large transport-heavy Rust crates
  not appropriate for every iOS bundle. If your app submits payments
  directly (rare for mobile), link them at the workspace level and
  expose via a custom bridge.

## PCI scope on iOS

The Rust core ensures PAN never appears in Swift `String`s. Combined
with using the iOS Keychain as the `Vault` backend, an iOS app
integrating OpenPay can plausibly stay at SAQ-A scope provided no
other code path touches PAN. The PCI DSS Tokenization Guidelines § 3.3
distinguishability rule is satisfied because OpenPay tokens carry the
`tok_v7_` prefix and are not numeric-only.

## Verified ground truth

- `swift-bridge` 0.1.59 (chinedufn/swift-bridge) is the latest stable
  as of May 2026. Inspired by `cxx-rs`; emits idiomatic Swift
  directly.
- Target triples verified against the rustc platform support pages
  for `*-apple-ios` (`aarch64-apple-ios`, `aarch64-apple-ios-sim`,
  `x86_64-apple-ios`) — all tier-2.
- XCFramework workflow verified against Apple Developer Forums
  threads (lipo simulator arches → xcodebuild -create-xcframework with
  one library per platform variant).
- iOS Keychain accessibility constants verified against Apple's
  developer documentation: `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`
  is the appropriate class for tokens needing background access that
  shouldn't migrate via iCloud backup.
