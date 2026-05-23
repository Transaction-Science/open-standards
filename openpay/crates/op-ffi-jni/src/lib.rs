//! # `op-ffi-jni` — Kotlin / Java / Android JNI bridge
//!
//! Exposes the OpenPay core to JVM languages via the `jni` crate.
//! Same architectural shape as Phase 8's [`op-ffi-swift`](../op_ffi_swift)
//! — opaque pointer handles + thread-local last-error + oracle-
//! discipline error collapsing — but using JNI conventions instead
//! of swift-bridge.
//!
//! ## Two parallel surfaces
//!
//! 1. **JNI module** ([`jni_bridge`]) — `#[no_mangle] pub extern "system" fn`
//!    functions named `Java_dev_openpay_<class>_<method>` that the
//!    JVM dispatches to from `native` declarations in the matching
//!    Kotlin class. The default and most common path.
//! 2. **Plain C ABI** ([`c_api`]) — identical surface to the C ABI
//!    in `op-ffi-swift`. Useful for JNI consumers who want to wrap
//!    our C functions in their own native code (e.g. an existing
//!    NDK module).
//!
//! ## Output artifacts
//!
//! `cargo ndk -t arm64-v8a -t armeabi-v7a -t x86 -t x86_64 build --release`
//! produces a `libop_ffi_jni.so` per ABI under
//! `target/<TRIPLE>/release/`. The `scripts/build-android.sh` helper
//! copies them into the Android module's `jniLibs/<ABI>/` tree.
//!
//! ## What does NOT cross the FFI
//!
//! - **Raw PAN.** The Kotlin `CardData` class wraps a Rust handle.
//!   The PAN string is read once on entry, then lives inside Rust
//!   until zeroized on drop.
//! - **Sensitive errors.** Throws typed `OpenPayException` subclasses
//!   that collapse oracle-leaking distinctions (NotFound vs AuthFailed
//!   vs InvalidToken all become `VaultLookupFailedException`).
//! - **Native handles.** Kotlin holds opaque `Long` IDs that point
//!   to heap-allocated Rust boxes. The Kotlin `finalize()` /
//!   `close()` paths call `op_*_free` deterministically.

#![deny(unsafe_code)]
#![warn(missing_docs)]

// JNI requires `unsafe` blocks at every native-method body; same
// situation as the Phase 8 C ABI. Scope the allowance per-module.
// Every fn here is a `Java_*` entry point with one uniform safety
// contract (the JVM passes valid `JNIEnv`/args), documented in the
// module header — a per-fn `# Safety` block would just repeat it.
#[allow(unsafe_code, clippy::missing_safety_doc)]
pub mod jni_bridge;

#[allow(unsafe_code)]
pub mod c_api;

pub mod error;

// Tests exercise the raw JNI / C ABI directly (pointers, manual
// frees), so they need the same unsafe-code allowance as the bridges.
#[cfg(test)]
#[allow(unsafe_code)]
mod tests;

pub use error::FfiError;
