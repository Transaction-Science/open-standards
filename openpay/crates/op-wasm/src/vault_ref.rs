//! JS-exposed [`VaultRef`].
//!
//! Tokens are short opaque strings. Unlike the Swift and JNI
//! bridges where we kept a wrapper class around them for ownership
//! tracking, the JS path treats tokens as plain strings — that's
//! the idiomatic shape for web APIs (durably storeable in
//! localStorage, transmittable as JSON, etc.).
//!
//! The [`VaultRef`] class exists for one purpose: to give consumers
//! a typed handle they can pass into vault methods without
//! confusing token strings with PANs. JS's lack of type-level
//! distinction means strings get easily mixed up; a dedicated class
//! is cheap insurance.

use wasm_bindgen::prelude::*;

/// Opaque vault token reference. Wraps a token string with a typed
/// JS class so consumers can't accidentally pass a PAN where a token
/// is expected.
#[wasm_bindgen]
pub struct VaultRef {
    pub(crate) inner: op_vault::VaultRef,
}

#[wasm_bindgen]
impl VaultRef {
    /// Wrap a token string. Use when recovering a token from durable
    /// storage (localStorage, IndexedDB, server round-trip).
    ///
    /// No validation: malformed tokens are detected by the vault on
    /// detokenize and surface as `OpenPayError` with
    /// `.kind === "VaultLookupFailed"`.
    #[wasm_bindgen(js_name = "fromString")]
    pub fn from_string(token: String) -> VaultRef {
        VaultRef {
            inner: op_vault::VaultRef::new(token),
        }
    }

    /// The token string. Safe to persist to durable storage.
    #[wasm_bindgen(getter, js_name = "asString")]
    pub fn as_string(&self) -> String {
        self.inner.as_str().to_owned()
    }

    /// `toString()` so JS template literals and `console.log` see
    /// something useful. Same as `.asString`.
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        self.as_string()
    }
}

impl VaultRef {
    /// Internal: wrap an `op_vault::VaultRef` returned by
    /// `vault.tokenize`.
    pub(crate) fn from_inner(inner: op_vault::VaultRef) -> Self {
        Self { inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_via_from_string() {
        let original = "tok_v7_abc123";
        let v = VaultRef::from_string(original.to_owned());
        assert_eq!(v.as_string(), original);
        assert_eq!(v.to_string_js(), original);
    }

    #[test]
    fn from_inner_preserves_payload() {
        let inner = op_vault::VaultRef::new("tok_v7_xyz");
        let v = VaultRef::from_inner(inner);
        assert_eq!(v.as_string(), "tok_v7_xyz");
    }

    #[test]
    fn from_string_does_not_validate() {
        // Per design, malformed tokens are caught on detokenize, not
        // on construction. Verify we accept literal garbage.
        let v = VaultRef::from_string("definitely-not-a-token".to_owned());
        assert_eq!(v.as_string(), "definitely-not-a-token");
    }
}
