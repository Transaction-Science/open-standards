//! [`TokenizationPolicy`] — operator-tunable rules for how the vault
//! mints tokens.
//!
//! Three orthogonal axes:
//!
//! 1. **Format** — random vs deterministic. Deterministic tokens let
//!    the same PAN map to the same token, which is useful for
//!    deduplication and analytics but creates a side channel that
//!    PCI DSS treats with suspicion. Random is the default and the
//!    only mode PCI scoping treats as "no value for PAN recovery."
//!
//! 2. **Lifetime** — how long the mapping is valid. Single-use tokens
//!    are consumed on first detokenization; reusable tokens survive
//!    until explicitly deleted or expired.
//!
//! 3. **Expiration** — optional wall-clock TTL. Single-use tokens may
//!    still benefit from a short TTL to bound the attack window.

use serde::{Deserialize, Serialize};

/// Format determines whether identical PANs produce identical tokens.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum TokenFormat {
    /// Default. Each call to [`Vault::tokenize`] returns a fresh random
    /// token, even for the same PAN. Recommended; matches PCI DSS
    /// guidance that tokens have no value for PAN recovery.
    ///
    /// [`Vault::tokenize`]: crate::Vault::tokenize
    Random,
    /// Same PAN → same token (within the same vault instance, with the
    /// same key). Useful for deduplication and analytics joins. Treat
    /// as a higher-risk mode: an attacker who can query the vault for
    /// known PANs can confirm whether they exist in the dataset.
    Deterministic,
}

/// Single-use vs reusable.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum TokenLifetime {
    /// Default. Token survives until explicitly deleted or expired.
    Reusable,
    /// Consumed on first successful [`Vault::detokenize`]. Subsequent
    /// resolution attempts return [`Error::AlreadyConsumed`].
    ///
    /// [`Vault::detokenize`]: crate::Vault::detokenize
    /// [`Error::AlreadyConsumed`]: crate::Error::AlreadyConsumed
    SingleUse,
}

/// Operator-tunable tokenization rules.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TokenizationPolicy {
    /// See [`TokenFormat`]. Default: `Random`.
    pub format: TokenFormat,
    /// See [`TokenLifetime`]. Default: `Reusable`.
    pub lifetime: TokenLifetime,
    /// Optional wall-clock TTL in seconds. `None` = never expires.
    ///
    /// Combine with `SingleUse` for high-value one-shot tokens
    /// (e.g. 3DS authentication completes within seconds; setting
    /// `ttl_seconds: Some(120)` gives a 2-minute window).
    pub ttl_seconds: Option<u64>,
}

impl Default for TokenizationPolicy {
    fn default() -> Self {
        Self {
            format: TokenFormat::Random,
            lifetime: TokenLifetime::Reusable,
            ttl_seconds: None,
        }
    }
}

impl TokenizationPolicy {
    /// A short-lived single-use token. Appropriate for 3DS authentication
    /// and other ephemeral flows where the token should never appear in
    /// logs as reusable.
    #[must_use]
    pub fn single_use(ttl_seconds: u64) -> Self {
        Self {
            format: TokenFormat::Random,
            lifetime: TokenLifetime::SingleUse,
            ttl_seconds: Some(ttl_seconds),
        }
    }

    /// A long-lived card-on-file token. Used for recurring billing and
    /// repeat customers. Random format only — deterministic on-file
    /// tokens defeat the scope-reduction benefit.
    #[must_use]
    pub fn card_on_file() -> Self {
        Self {
            format: TokenFormat::Random,
            lifetime: TokenLifetime::Reusable,
            ttl_seconds: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_random_reusable_no_ttl() {
        let p = TokenizationPolicy::default();
        assert_eq!(p.format, TokenFormat::Random);
        assert_eq!(p.lifetime, TokenLifetime::Reusable);
        assert_eq!(p.ttl_seconds, None);
    }

    #[test]
    fn single_use_helper_sets_three_fields() {
        let p = TokenizationPolicy::single_use(120);
        assert_eq!(p.format, TokenFormat::Random);
        assert_eq!(p.lifetime, TokenLifetime::SingleUse);
        assert_eq!(p.ttl_seconds, Some(120));
    }

    #[test]
    fn card_on_file_is_long_lived_random_reusable() {
        let p = TokenizationPolicy::card_on_file();
        assert_eq!(p.format, TokenFormat::Random);
        assert_eq!(p.lifetime, TokenLifetime::Reusable);
        assert!(
            p.ttl_seconds.is_none(),
            "card-on-file should not auto-expire"
        );
    }

    #[test]
    fn policy_round_trips_through_json() {
        let p = TokenizationPolicy::single_use(60);
        let s = serde_json::to_string(&p).unwrap();
        let back: TokenizationPolicy = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn token_format_round_trips() {
        for f in [TokenFormat::Random, TokenFormat::Deterministic] {
            let s = serde_json::to_string(&f).unwrap();
            let back: TokenFormat = serde_json::from_str(&s).unwrap();
            assert_eq!(f, back);
        }
    }
}
