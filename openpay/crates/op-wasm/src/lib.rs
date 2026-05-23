//! # `op-wasm` — Browser / Node.js bridge
//!
//! Exposes the OpenPay Rust core to JavaScript via [`wasm-bindgen`].
//! Same architectural shape as Phases 8 (Swift) and 9 (JNI): opaque
//! handles + oracle-discipline error collapsing + a thin
//! per-platform wrapper.
//!
//! ## Surface
//!
//! `#[wasm_bindgen]`-annotated Rust structs become JavaScript classes
//! directly. The bindgen toolchain generates:
//!
//! - `pkg/op_wasm.js` — the JS shim that consumers `import`.
//! - `pkg/op_wasm_bg.wasm` — the compiled wasm module.
//! - `pkg/op_wasm.d.ts` — TypeScript types.
//!
//! Consumers see:
//!
//! ```text
//! import { CardData, RustVault, HeuristicScorer, OpenPayError } from 'openpay';
//!
//! const vault = new RustVault('checkout');
//! const card = new CardData('4242424242424242', 12, 2030);
//! const tokenStr = vault.tokenize(card);   // card is consumed
//! const recovered = vault.detokenize(tokenStr);
//! console.log(recovered.lastFour);          // "4242"
//! recovered.free();
//! vault.free();
//! ```
//!
//! Note the explicit `.free()` calls. wasm-bindgen does not have a
//! finalizer mechanism (the host JS engine doesn't expose one
//! portably), so JS callers MUST free explicitly. The
//! TypeScript-friendly API also exposes `[Symbol.dispose]` so that
//! `using` statements (ES2026) handle cleanup automatically.
//!
//! ## What does NOT cross the boundary
//!
//! - **Raw PAN.** The `CardData` JS class wraps a Rust handle; the
//!   PAN string is read on construction and lives inside Rust until
//!   `.free()` is called.
//! - **Sensitive errors.** Errors are surfaced as `OpenPayError`
//!   instances whose `.code` field matches Phase 8 (Swift) and
//!   Phase 9 (JNI) discriminants for cross-platform observability.
//!
//! ## Browser crypto
//!
//! This crate uses the same `aes-gcm-siv 0.11` cipher as Phases 7/8/9.
//! It runs entirely in wasm without touching the Web Crypto API.
//!
//! A separate `WebCryptoVault` analogous to the
//! `KeystoreVault` (Phase 9) — using `SubtleCrypto` for
//! hardware-accelerated AES-GCM and IndexedDB for persistence — is
//! out of scope here because `SubtleCrypto` is async-only, and
//! reshaping the [`Vault`] trait to be async would ripple through
//! the entire stack. Web Crypto integration belongs in a later
//! phase as `op-wasm-webcrypto` once the async vault story is
//! designed.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod card_data;
pub mod error;
pub mod heuristic_scorer;
pub mod policy;
pub mod vault;
pub mod vault_ref;

#[cfg(feature = "console-panic-hook")]
mod panic_hook;

pub use card_data::CardData;
pub use error::{FfiError, OpenPayError};
pub use heuristic_scorer::HeuristicScorer;
pub use policy::{TokenFormat, TokenLifetime, TokenizationPolicy};
pub use vault::RustVault;
pub use vault_ref::VaultRef;

/// Install a panic hook that forwards Rust panics to `console.error`.
/// Only available when the `console-panic-hook` feature is enabled.
///
/// Production deployments should ship their own panic handler that
/// reports to their telemetry layer; this hook is a development aid.
#[cfg(feature = "console-panic-hook")]
mod panic_hook_export {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(js_name = "setPanicHook")]
    pub fn set_panic_hook() {
        super::panic_hook::install();
    }
}

#[cfg(feature = "console-panic-hook")]
pub use panic_hook_export::set_panic_hook;
