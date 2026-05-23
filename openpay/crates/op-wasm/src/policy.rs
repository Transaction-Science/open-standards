//! JS-exposed tokenization policy.
//!
//! wasm-bindgen has limited support for compound types across the
//! boundary (no `Option<u64>`, no nested structs by-value). The
//! idiomatic shape is a class with a constructor + getters/setters.
//! That's what we use here.
//!
//! Two convenience static factories — `singleUse(ttl)` and
//! `cardOnFile()` — cover the two most common configurations and
//! mirror the same names used on iOS and Android.

use wasm_bindgen::prelude::*;

/// Tokenization format.
///
/// wasm-bindgen exports Rust enums as TypeScript-friendly union
/// constants on the JS side: `TokenFormat.Random` and
/// `TokenFormat.Deterministic`.
#[wasm_bindgen]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenFormat {
    /// Fresh random token each call. PCI-aligned default.
    Random = 0,
    /// Same PAN → same token. Creates a query oracle.
    Deterministic = 1,
}

/// Token lifetime.
#[wasm_bindgen]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenLifetime {
    /// Token survives until explicitly deleted or expired.
    Reusable = 0,
    /// Consumed on first successful detokenize.
    SingleUse = 1,
}

/// Tokenization policy.
///
/// The fields are not directly settable as JS struct fields because
/// wasm-bindgen serializes them through getters/setters; equivalent
/// shape but works across the boundary.
#[wasm_bindgen]
#[derive(Copy, Clone, Debug)]
pub struct TokenizationPolicy {
    format: TokenFormat,
    lifetime: TokenLifetime,
    /// TTL in seconds. Zero means "no TTL".
    ///
    /// We use `0` as the sentinel rather than null because
    /// wasm-bindgen doesn't model `Option<u64>` directly without
    /// boxing. The op-vault crate's TokenizationPolicy uses
    /// `Option<u64>`; we translate at the boundary.
    ttl_seconds: u64,
}

#[wasm_bindgen]
impl TokenizationPolicy {
    /// Default: random format, reusable lifetime, no TTL.
    #[wasm_bindgen(constructor)]
    pub fn new() -> TokenizationPolicy {
        TokenizationPolicy {
            format: TokenFormat::Random,
            lifetime: TokenLifetime::Reusable,
            ttl_seconds: 0,
        }
    }

    /// Short-lived single-use token. Appropriate for 3DS
    /// authentication and other one-shot flows. Default TTL 120s.
    #[wasm_bindgen(js_name = "singleUse")]
    pub fn single_use(ttl_seconds: Option<u64>) -> TokenizationPolicy {
        TokenizationPolicy {
            format: TokenFormat::Random,
            lifetime: TokenLifetime::SingleUse,
            ttl_seconds: ttl_seconds.unwrap_or(120),
        }
    }

    /// Long-lived random reusable token for card-on-file.
    #[wasm_bindgen(js_name = "cardOnFile")]
    pub fn card_on_file() -> TokenizationPolicy {
        TokenizationPolicy {
            format: TokenFormat::Random,
            lifetime: TokenLifetime::Reusable,
            ttl_seconds: 0,
        }
    }

    /// Format getter.
    #[wasm_bindgen(getter)]
    pub fn format(&self) -> TokenFormat {
        self.format
    }

    /// Format setter.
    #[wasm_bindgen(setter)]
    pub fn set_format(&mut self, v: TokenFormat) {
        self.format = v;
    }

    /// Lifetime getter.
    #[wasm_bindgen(getter)]
    pub fn lifetime(&self) -> TokenLifetime {
        self.lifetime
    }

    /// Lifetime setter.
    #[wasm_bindgen(setter)]
    pub fn set_lifetime(&mut self, v: TokenLifetime) {
        self.lifetime = v;
    }

    /// TTL in seconds. `0` means "no TTL".
    #[wasm_bindgen(getter, js_name = "ttlSeconds")]
    pub fn ttl_seconds(&self) -> u64 {
        self.ttl_seconds
    }

    /// TTL setter. Pass `0` for "no TTL".
    #[wasm_bindgen(setter, js_name = "ttlSeconds")]
    pub fn set_ttl_seconds(&mut self, v: u64) {
        self.ttl_seconds = v;
    }
}

impl Default for TokenizationPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizationPolicy {
    /// Internal: lower to the op_vault representation.
    pub(crate) fn to_inner(self) -> op_vault::TokenizationPolicy {
        op_vault::TokenizationPolicy {
            format: match self.format {
                TokenFormat::Random => op_vault::TokenFormat::Random,
                TokenFormat::Deterministic => op_vault::TokenFormat::Deterministic,
            },
            lifetime: match self.lifetime {
                TokenLifetime::Reusable => op_vault::TokenLifetime::Reusable,
                TokenLifetime::SingleUse => op_vault::TokenLifetime::SingleUse,
            },
            ttl_seconds: if self.ttl_seconds == 0 {
                None
            } else {
                Some(self.ttl_seconds)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_random_reusable_no_ttl() {
        let p = TokenizationPolicy::new();
        assert_eq!(p.format(), TokenFormat::Random);
        assert_eq!(p.lifetime(), TokenLifetime::Reusable);
        assert_eq!(p.ttl_seconds(), 0);
    }

    #[test]
    fn single_use_default_ttl_is_120() {
        let p = TokenizationPolicy::single_use(None);
        assert_eq!(p.lifetime(), TokenLifetime::SingleUse);
        assert_eq!(p.ttl_seconds(), 120);
    }

    #[test]
    fn single_use_with_explicit_ttl() {
        let p = TokenizationPolicy::single_use(Some(60));
        assert_eq!(p.ttl_seconds(), 60);
    }

    #[test]
    fn card_on_file_is_reusable_no_ttl() {
        let p = TokenizationPolicy::card_on_file();
        assert_eq!(p.lifetime(), TokenLifetime::Reusable);
        assert_eq!(p.ttl_seconds(), 0);
    }

    #[test]
    fn setters_round_trip() {
        let mut p = TokenizationPolicy::new();
        p.set_format(TokenFormat::Deterministic);
        p.set_lifetime(TokenLifetime::SingleUse);
        p.set_ttl_seconds(45);
        assert_eq!(p.format(), TokenFormat::Deterministic);
        assert_eq!(p.lifetime(), TokenLifetime::SingleUse);
        assert_eq!(p.ttl_seconds(), 45);
    }

    #[test]
    fn to_inner_zero_ttl_maps_to_none() {
        let p = TokenizationPolicy::new();
        let inner = p.to_inner();
        assert_eq!(inner.ttl_seconds, None);
    }

    #[test]
    fn to_inner_nonzero_ttl_maps_to_some() {
        let mut p = TokenizationPolicy::new();
        p.set_ttl_seconds(120);
        let inner = p.to_inner();
        assert_eq!(inner.ttl_seconds, Some(120));
    }

    #[test]
    fn to_inner_format_mapping() {
        let mut p = TokenizationPolicy::new();
        p.set_format(TokenFormat::Random);
        assert_eq!(p.to_inner().format, op_vault::TokenFormat::Random);
        p.set_format(TokenFormat::Deterministic);
        assert_eq!(p.to_inner().format, op_vault::TokenFormat::Deterministic);
    }

    #[test]
    fn to_inner_lifetime_mapping() {
        let mut p = TokenizationPolicy::new();
        p.set_lifetime(TokenLifetime::Reusable);
        assert_eq!(p.to_inner().lifetime, op_vault::TokenLifetime::Reusable);
        p.set_lifetime(TokenLifetime::SingleUse);
        assert_eq!(p.to_inner().lifetime, op_vault::TokenLifetime::SingleUse);
    }
}
