//! JS-exposed in-memory vault backed by `op_vault::InMemoryVault`.
//!
//! Lifetime: as long as the JS object lives (and is not `.free()`'d).
//! Restarting the page wipes the vault, like an in-memory Map. For
//! durable browser storage, a future `WebCryptoVault` will persist
//! to IndexedDB under a Web Crypto-backed key; until that's built,
//! consumers persist token strings to localStorage and rebuild the
//! vault state from a server round-trip on page load.
//!
//! Same architectural pattern as Phase 8 (Swift) and Phase 9 (JNI):
//! a thin wrapper around the Phase 7 vault that mediates ownership
//! and error mapping.

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::card_data::CardData;
use crate::error::{FfiError, jsify_ffi, jsify_vault_err};
use crate::policy::TokenizationPolicy;
use crate::vault_ref::VaultRef;

/// In-memory vault. Tokens live until the vault is `.free()`'d or
/// the host JS environment exits.
#[wasm_bindgen]
pub struct RustVault {
    inner: Arc<op_vault::InMemoryVault>,
}

#[wasm_bindgen]
impl RustVault {
    /// Construct an ephemeral vault with the given name (used for
    /// telemetry only — does not affect behavior).
    #[wasm_bindgen(constructor)]
    pub fn new(name: String) -> RustVault {
        RustVault {
            inner: Arc::new(op_vault::InMemoryVault::ephemeral(name)),
        }
    }

    /// Vault name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        use op_vault::Vault;
        self.inner.name().to_owned()
    }

    /// Tokenize a card. **Consumes** the `card` argument — the JS
    /// object is invalidated by wasm-bindgen because the Rust
    /// signature takes `card` by value.
    ///
    /// Returns a [`VaultRef`] handle. The caller is responsible for
    /// `.free()`-ing it when done.
    pub fn tokenize(
        &self,
        card: CardData,
        policy: Option<TokenizationPolicy>,
    ) -> Result<VaultRef, JsValue> {
        use op_vault::Vault;
        let p = policy.unwrap_or_default().to_inner();
        let inner_card = card.into_inner();
        let vref = self
            .inner
            .tokenize(inner_card, p)
            .map_err(jsify_vault_err)?;
        Ok(VaultRef::from_inner(vref))
    }

    /// Detokenize. Returns a fresh [`CardData`] on success. Errors:
    ///
    /// - `OpenPayError.kind === "VaultLookupFailed"` — unknown,
    ///   malformed, or auth-failed token (collapsed for oracle
    ///   discipline).
    /// - `"TokenExpired"` — past its TTL.
    /// - `"TokenAlreadyConsumed"` — single-use token, already used.
    pub fn detokenize(&self, token: &VaultRef) -> Result<CardData, JsValue> {
        use op_vault::Vault;
        let inner = self
            .inner
            .detokenize(&token.inner)
            .map_err(jsify_vault_err)?;
        Ok(CardData::from_inner(inner))
    }

    /// Probe existence.
    pub fn exists(&self, token: &VaultRef) -> Result<bool, JsValue> {
        use op_vault::Vault;
        self.inner.exists(&token.inner).map_err(jsify_vault_err)
    }

    /// Delete a token. Returns `true` if a mapping was removed.
    /// Idempotent — returns `false` for already-absent tokens.
    pub fn delete(&self, token: &VaultRef) -> Result<bool, JsValue> {
        use op_vault::Vault;
        self.inner.delete(&token.inner).map_err(jsify_vault_err)
    }
}

/// Convenience: tokenize a PAN string in one call without
/// constructing an intermediate [`CardData`]. Equivalent to
/// `new CardData(pan, expMonth, expYear)` then `vault.tokenize(card)`,
/// but skips the JS round-trip.
///
/// The PAN string is read once and immediately handed to Rust; the
/// JS-side copy persists in the engine heap until garbage collection.
#[wasm_bindgen(js_name = "tokenizeFromString")]
pub fn tokenize_from_string(
    vault: &RustVault,
    pan: String,
    exp_month: u8,
    exp_year: u16,
    policy: Option<TokenizationPolicy>,
) -> Result<VaultRef, JsValue> {
    use op_vault::Vault;
    let card = op_vault::CardData::new(pan, exp_month, exp_year)
        .map_err(|_| jsify_ffi(FfiError::InvalidInput, "invalid card data"))?;
    let p = policy.unwrap_or_default().to_inner();
    let vref = vault.inner.tokenize(card, p).map_err(jsify_vault_err)?;
    Ok(VaultRef::from_inner(vref))
}

#[cfg(test)]
mod tests {
    // The wasm-bindgen-generated `#[wasm_bindgen]` macro produces
    // glue that calls extern functions which only exist in the wasm
    // runtime. We can't exercise the `RustVault` *class* in plain
    // `cargo test`, but we can exercise the inner logic by using
    // the underlying op_vault types directly. The
    // `tests/wasm_bindgen.rs` integration suite covers the JS
    // boundary in a real wasm host.

    use super::*;

    const VALID_VISA: &str = "4242424242424242";

    #[test]
    fn inner_vault_round_trip() {
        // This bypasses the wasm-bindgen wrapper but exercises the
        // exact code path that tokenize/detokenize would take.
        use op_vault::Vault;

        let vault = op_vault::InMemoryVault::ephemeral("test");
        let card = op_vault::CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
        let policy = TokenizationPolicy::card_on_file().to_inner();

        let vref = vault.tokenize(card, policy).unwrap();
        assert!(vref.as_str().starts_with("tok_v7_"));

        let recovered = vault.detokenize(&vref).unwrap();
        assert_eq!(recovered.last_four(), "4242");
    }

    #[test]
    fn unknown_token_collapses_to_lookup_failed() {
        use op_vault::Vault;

        let vault = op_vault::InMemoryVault::ephemeral("test");
        let fake = op_vault::VaultRef::new("tok_v7_nope");
        let err = vault.detokenize(&fake).unwrap_err();
        let ffi: FfiError = err.into();
        assert_eq!(ffi, FfiError::VaultLookupFailed);
    }

    #[test]
    fn malformed_token_also_collapses_to_lookup_failed() {
        // Oracle discipline preserved through the wasm boundary too.
        use op_vault::Vault;

        let vault = op_vault::InMemoryVault::ephemeral("test");
        let bad = op_vault::VaultRef::new("not-a-token");
        let err = vault.detokenize(&bad).unwrap_err();
        let ffi: FfiError = err.into();
        assert_eq!(ffi, FfiError::VaultLookupFailed);
    }

    #[test]
    fn policy_zero_ttl_maps_to_none() {
        // Critical: the JS sentinel for "no TTL" is `0`, not `null`,
        // because wasm-bindgen doesn't model Option<u64> directly.
        // Verify the boundary translation is correct.
        let p = TokenizationPolicy::new();
        assert_eq!(p.ttl_seconds(), 0);
        assert_eq!(p.to_inner().ttl_seconds, None);
    }
}
