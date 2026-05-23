//! JS-exposed [`CardData`] class.
//!
//! Same security stance as the Swift and Kotlin bridges: the
//! constructor reads the PAN once, hands it to the Rust core, and
//! the only accessors from JS are first-six / last-four / expiration.
//! The PAN bytes live in wasm linear memory inside an
//! `op_vault::CardData` until the wrapper is `.free()`'d.

use wasm_bindgen::prelude::*;

use crate::error::{FfiError, jsify_ffi};

/// Validated card data. Constructed from a PAN string; thereafter
/// only safe metadata is observable from JS.
///
/// The wasm-bindgen runtime tracks a pointer to a Rust-side heap
/// allocation. Calling `.free()` releases it; subsequent method
/// calls on the same JS object throw because the bindgen-generated
/// stub null-checks the pointer.
#[wasm_bindgen]
pub struct CardData {
    pub(crate) inner: op_vault::CardData,
}

#[wasm_bindgen]
impl CardData {
    /// Construct from a PAN, expiration month (1-12), and four-digit
    /// expiration year. Validates Luhn checksum + length + expiration
    /// sanity.
    ///
    /// Throws an `OpenPayError` with `.kind === "InvalidInput"` on
    /// validation failure.
    #[wasm_bindgen(constructor)]
    pub fn new(pan: String, exp_month: u8, exp_year: u16) -> Result<CardData, JsValue> {
        match op_vault::CardData::new(pan, exp_month, exp_year) {
            Ok(inner) => Ok(CardData { inner }),
            Err(_) => Err(jsify_ffi(FfiError::InvalidInput, "invalid card data")),
        }
    }

    /// First six digits (BIN). Safe to log per PCI DSS 4.0.1 §3.4.1.
    #[wasm_bindgen(getter, js_name = "firstSix")]
    pub fn first_six(&self) -> String {
        self.inner.first_six().to_owned()
    }

    /// Last four digits. Safe to log.
    #[wasm_bindgen(getter, js_name = "lastFour")]
    pub fn last_four(&self) -> String {
        self.inner.last_four().to_owned()
    }

    /// Expiration month (1-12).
    #[wasm_bindgen(getter, js_name = "expMonth")]
    pub fn exp_month(&self) -> u8 {
        self.inner.exp_month()
    }

    /// Expiration year (e.g. 2030).
    #[wasm_bindgen(getter, js_name = "expYear")]
    pub fn exp_year(&self) -> u16 {
        self.inner.exp_year()
    }
}

impl CardData {
    /// Internal: wrap an already-validated `op_vault::CardData`
    /// returned by `vault.detokenize`. No re-validation; used in the
    /// Rust → JS direction.
    pub(crate) fn from_inner(inner: op_vault::CardData) -> Self {
        Self { inner }
    }

    /// Internal: consume self and return the inner CardData, used by
    /// `vault.tokenize`. The wasm-bindgen runtime invalidates the JS
    /// pointer at the call site (because the method takes `self` by
    /// value).
    pub(crate) fn into_inner(self) -> op_vault::CardData {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_VISA: &str = "4242424242424242";

    #[test]
    fn valid_pan_constructs() {
        let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
        assert_eq!(card.first_six(), "424242");
        assert_eq!(card.last_four(), "4242");
        assert_eq!(card.exp_month(), 12);
        assert_eq!(card.exp_year(), 2030);
    }

    // The error arm of `CardData::new` builds a `JsValue`, which panics
    // on a non-wasm host ("cannot convert to JsValue outside of the Wasm
    // target"). The validation itself lives in `op_vault::CardData`, so
    // we assert rejection there on the host and let `wasm-pack test`
    // cover the JsValue mapping on the wasm target.

    #[test]
    fn invalid_luhn_rejected() {
        assert!(op_vault::CardData::new("1111111111111111".to_owned(), 12, 2030).is_err());
    }

    #[test]
    fn invalid_exp_month_rejected() {
        assert!(op_vault::CardData::new(VALID_VISA.to_owned(), 13, 2030).is_err());
        assert!(op_vault::CardData::new(VALID_VISA.to_owned(), 0, 2030).is_err());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn invalid_input_maps_to_js_error() {
        assert!(CardData::new("1111111111111111".to_owned(), 12, 2030).is_err());
        assert!(CardData::new(VALID_VISA.to_owned(), 13, 2030).is_err());
    }

    #[test]
    fn into_inner_round_trip() {
        let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
        let inner = card.into_inner();
        assert_eq!(inner.last_four(), "4242");
    }

    #[test]
    fn from_inner_no_revalidation() {
        // from_inner accepts whatever it's given because the caller
        // (vault.detokenize) has the only safe path producing it.
        let inner = op_vault::CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
        let card = CardData::from_inner(inner);
        assert_eq!(card.last_four(), "4242");
    }
}
