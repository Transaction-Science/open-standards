//! # `op-ffi-swift` — Swift / iOS / macOS bridge
//!
//! Exposes the OpenPay surface to Swift callers. Two parallel surfaces:
//!
//! 1. **`swift-bridge` module** ([`bridge`]) — idiomatic Swift API
//!    generated from a single bridge declaration. Swift apps that use
//!    Swift Package Manager or CocoaPods pull in the generated
//!    `OpenPay.swift` file and call `let vault = RustVault.ephemeral()`
//!    style code. Default build path.
//!
//! 2. **Plain C ABI** ([`c_api`]) — `extern "C"` functions named
//!    `op_orchestrator_*`, `op_vault_*`, etc. For consumers who don't
//!    want `swift-bridge-build` integration; they write hand-rolled
//!    Swift wrappers using `@_silgen_name`. Enabled by default; can be
//!    isolated via `--features c-only`.
//!
//! ## Output artifacts
//!
//! `cargo build --target aarch64-apple-ios --release` produces:
//!
//! - `libop_ffi_swift.a` — static library (link this into the app)
//! - `target/.../build/op-ffi-swift-*/out/OpenPay.swift` — Swift glue
//! - `target/.../build/op-ffi-swift-*/out/openpay-swift-bridge.h` — C header
//! - `target/.../build/op-ffi-swift-*/out/module.modulemap`
//!
//! The `scripts/build-xcframework.sh` helper lipos device + simulator
//! arches into a fat lib and bundles into a `.xcframework`.
//!
//! ## What does NOT cross the FFI
//!
//! - **Raw PAN.** [`CardData`](op_vault::CardData) is exposed across
//!   the bridge as an opaque Rust type. Swift code constructs it from
//!   PAN bytes, hands it to the vault, and immediately drops the
//!   handle — Rust zeroizes on drop. The raw bytes never appear in a
//!   Swift `String`.
//! - **Sensitive errors.** The bridge translates fine-grained vault
//!   errors into a coarse enum that doesn't leak oracle information.
//! - **Memory ownership.** All opaque types are heap-allocated in Rust
//!   and freed via `swift-bridge`'s ownership protocol. Swift cannot
//!   leak Rust memory by holding references past their lifetime.

#![deny(unsafe_code)]
#![warn(missing_docs)]
// swift-bridge generated code may not satisfy our pedantic lints.
// Scope the relaxation tightly to the bridge module. The pointer
// round-trips and the `default` associated fn name are mandated by
// swift-bridge's codegen contract, not stylistic choices.
#[allow(clippy::unnecessary_cast, clippy::should_implement_trait)]
pub mod bridge;
#[allow(unsafe_code)]
pub mod c_api;
pub mod error;
// The integration tests drive the raw C ABI (pointers, CStr, manual
// frees) to prove both surfaces interoperate, so they need the same
// unsafe-code allowance as `c_api`.
#[cfg(test)]
#[allow(unsafe_code)]
mod tests;

pub use error::FfiError;
