# OpenPay Android (`dev.openpay`)

Kotlin/Java wrapper around the OpenPay Rust core, packaged as a
standard AAR via the AGP 9 library plugin.

## What you get

- **`CardData`** — Validated PAN + expiration. Luhn / length / date
  sanity checked on construction. PAN bytes live in Rust memory and
  are never readable from Kotlin.
- **`VaultRef`** — Opaque token reference. Safe to persist, log, and
  transmit.
- **`Vault`** — Tokenize / detokenize / exists / delete interface.
- **`RustVault`** — In-memory reference vault for tests.
- **`KeystoreVault`** — Production vault backed by
  EncryptedSharedPreferences + the Android Keystore.
- **`HeuristicScorer`** — Rule-based fraud scorer.
- **`OpenPayException`** — Sealed-class hierarchy with one subclass
  per failure mode.
- **`TokenizationPolicy`** — Configurable random / deterministic +
  reusable / single-use + optional TTL.

## Build the native library first

The Kotlin module ships pre-built `.so` files under
`src/main/jniLibs/<ABI>/`. To regenerate them after a Rust change:

```bash
cargo install cargo-ndk
rustup target add aarch64-linux-android armv7-linux-androideabi \
                  x86_64-linux-android i686-linux-android

# From the crate root (`crates/op-ffi-jni/`):
bash scripts/build-android.sh --release
```

The script invokes `cargo ndk` for all four standard ABIs and stages
the outputs into `kotlin/openpay/src/main/jniLibs/<ABI>/`.

## Consume the library

### Option 1 — Gradle subproject

In your Android app's `settings.gradle.kts`:

```kotlin
include(":openpay")
project(":openpay").projectDir = file("../openpay/crates/op-ffi-jni/kotlin/openpay")
```

In your app's `build.gradle.kts`:

```kotlin
dependencies {
    implementation(project(":openpay"))
}
```

### Option 2 — Local Maven publication

```bash
cd kotlin
./gradlew :openpay:publishToMavenLocal
```

Then in your app:

```kotlin
repositories {
    mavenLocal()
}

dependencies {
    implementation("dev.openpay:openpay:0.1.0")
}
```

## Usage example

```kotlin
import dev.openpay.*

class CheckoutViewModel(private val context: Context) {

    // KeystoreVault persists across app launches via the Android
    // Keystore. One instance per app is fine; it's thread-safe.
    private val vault = KeystoreVault(context.applicationContext)

    fun saveCard(pan: String, expMonth: Byte, expYear: Short): String {
        val token = vault.tokenizeFromString(
            pan, expMonth, expYear,
            TokenizationPolicy.cardOnFile(),
        )
        try {
            return token.asString
        } finally {
            token.close()
        }
    }

    fun chargeCard(tokenStr: String) {
        VaultRef.fromString(tokenStr).use { ref ->
            try {
                vault.detokenize(ref).use { card ->
                    // Hand `card.firstSix` / `card.lastFour` to your
                    // analytics layer (safe per PCI 4.0.1 §3.4.1).
                    // The full PAN lives only inside the Rust heap
                    // until `card.close()`.
                    submitToAcquirer(card)
                }
            } catch (e: OpenPayException.VaultLookupFailed) {
                // Unknown/malformed/expired key — collapsed to a
                // single exception type for oracle discipline.
                showError("Card not found")
            } catch (e: OpenPayException.TokenExpired) {
                showError("Token expired. Please re-enter your card.")
            }
        }
    }
}
```

## Backup discipline (critical)

`KeystoreVault` persists encrypted records to a SharedPreferences
file. The encryption key lives in the Android Keystore, which is
**bound to the device**. If Android Auto Backup restores the
encrypted file to a different device, decryption silently fails.

Drop this into your app's `res/xml/backup_rules.xml`:

```xml
<full-backup-content>
    <exclude domain="sharedpref" path="openpay-vault.xml" />
</full-backup-content>
```

And reference it from your manifest:

```xml
<application
    android:fullBackupContent="@xml/backup_rules"
    ...>
```

A working example is shipped at
`kotlin/openpay/src/main/res/xml/backup_rules.xml`.

On Android 12+ (API 31+), also configure
`android:dataExtractionRules` per the AOSP docs.

## AndroidX security-crypto deprecation note

`androidx.security:security-crypto` was deprecated at version
1.1.0-alpha07 in April 2025. The API surface is stable; only future
maintenance is uncertain. Two paths forward:

1. **Stay on AndroidX.** The 1.1.0-alpha06 release (what this module
   pins by default) is what most production apps use. The classes
   continue to work and Tink (the underlying crypto) is still
   actively maintained by Google.
2. **Switch to the community fork.** Replace the dependency:

   ```kotlin
   // Before:
   implementation("androidx.security:security-crypto:1.1.0-alpha06")
   // After:
   implementation("dev.spght:encryptedprefs-ktx:<latest>")
   ```

   And in `KeystoreVault.kt`, change the imports:

   ```kotlin
   // Before:
   import androidx.security.crypto.EncryptedSharedPreferences
   import androidx.security.crypto.MasterKey
   // After:
   import dev.spght.encryptedprefs.EncryptedSharedPreferences
   import dev.spght.encryptedprefs.MasterKey
   ```

   The classes have identical APIs.

## Min SDK 23 (Android 6.0)

Required by Tink 1.18+, which is pulled in transitively by
`androidx.security:security-crypto:1.1.x`. The fork also targets 23+.
Lower min-SDK is not supportable without abandoning the
hardware-backed Keystore.

## Why doesn't `KeystoreVault` implement `Vault`?

`Vault.tokenize(card: CardData)` requires the implementation to read
the PAN bytes out of the `CardData` handle. But `CardData`
deliberately hides PAN bytes from Kotlin — they live in Rust memory
behind a `pub(crate)` accessor. That's the PCI scope boundary.

`KeystoreVault` therefore offers `tokenizeFromString(pan, ...)`,
which takes the PAN as a Kotlin string and writes it directly into
EncryptedSharedPreferences. This is the same pattern PSP SDKs like
Stripe and Adyen use in their Android integrations: the encrypted
store IS the Keystore, no intermediate cipher.

`RustVault` does implement `Vault` because the PAN never leaves Rust
in that flow — it stays inside the in-memory vault from
`tokenize(card)` to `detokenize(ref)`.

## Testing

Two test suites:

- **`src/test/`** runs on the host JVM (`./gradlew :openpay:test`).
  Covers `CardData`, `VaultRef`, `RustVault`, `HeuristicScorer`,
  `TokenizationPolicy`. Requires the `.so` on
  `java.library.path` — build a host-architecture variant first with
  `cargo build -p op-ffi-jni --release`.
- **`src/androidTest/`** runs on a device/emulator
  (`./gradlew :openpay:connectedAndroidTest`). Covers `KeystoreVault`
  which needs an Android `Context` and the Keystore.
