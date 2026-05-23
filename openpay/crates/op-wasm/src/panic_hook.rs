//! Console panic hook (feature-gated).
//!
//! Forwards Rust panics into `console.error` so they show up in
//! browser devtools instead of as opaque `RuntimeError: unreachable`
//! at the wasm trap site.

#![cfg(feature = "console-panic-hook")]

use std::sync::Once;

static INSTALL: Once = Once::new();

pub(crate) fn install() {
    INSTALL.call_once(|| {
        console_error_panic_hook::set_once();
    });
}
