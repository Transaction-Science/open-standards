//! Mastercard Digital Enablement Service (MDES) adapter.
//!
//! ## Production surface
//!
//! Real MDES calls go to `https://api.mastercard.com/mdes/...` with
//! OAuth 1.0a signed requests (Mastercard's `oauth1-signer`
//! convention). The full integration is gated behind the `live`
//! cargo feature on this crate; the default build uses the
//! deterministic stubs in this module.
//!
//! Stub determinism follows the same pattern as the VTS stub — every
//! method returns a value derived from a stable hash of its inputs.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use op_core::{CardNetwork, Money, NetworkToken, NetworkTokenLifecycleEvent, VaultRef};
use time::{Duration, OffsetDateTime};

use crate::Result;
use crate::network_token::cryptogram::Cryptogram;
use crate::network_token::provider::{
    LifecycleEvent, LifecycleStream, NetworkTokenProvider, ProviderConfig,
};

/// Mastercard MDES provider.
pub struct MdesProvider {
    config: ProviderConfig,
}

impl MdesProvider {
    /// Construct with the given config. Panics if `config.network` is
    /// not [`CardNetwork::Mastercard`].
    #[must_use]
    pub fn new(config: ProviderConfig) -> Self {
        assert!(
            matches!(config.network, CardNetwork::Mastercard),
            "MdesProvider requires CardNetwork::Mastercard, got {:?}",
            config.network
        );
        Self { config }
    }

    /// Borrow the config.
    #[must_use]
    pub const fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn stable_hash(parts: &[&str]) -> u64 {
        let mut h = DefaultHasher::new();
        for p in parts {
            p.hash(&mut h);
        }
        h.finish()
    }
}

impl NetworkTokenProvider for MdesProvider {
    fn name(&self) -> &'static str {
        "mdes"
    }

    fn network(&self) -> CardNetwork {
        CardNetwork::Mastercard
    }

    fn provision(&self, card_ref: &VaultRef) -> Result<NetworkToken> {
        // STUB: deterministic token. Live impl calls
        // POST /mdes/digitization/static/1/0/tokenize.
        let h = Self::stable_hash(&[card_ref.as_str(), &self.config.token_requestor_id]);
        let token_ref = format!("mdes_stub_{h:016x}");
        Ok(NetworkToken::new(
            token_ref,
            "5454",
            CardNetwork::Mastercard,
            true,
        ))
    }

    fn lifecycle(&self, token_ref: &str) -> Result<LifecycleStream> {
        let h = Self::stable_hash(&[token_ref]);
        let base =
            OffsetDateTime::from_unix_timestamp(1_700_000_000 + i64::from(h as u32 % 1000)).unwrap();
        let token = token_ref.to_owned();
        let events = vec![
            LifecycleEvent {
                token_ref: token.clone(),
                event: NetworkTokenLifecycleEvent::Provisioned,
                at: base,
                reason: None,
            },
            LifecycleEvent {
                token_ref: token,
                event: NetworkTokenLifecycleEvent::Suspended,
                at: base + Duration::seconds(120),
                reason: Some("stub:issuer_review".into()),
            },
        ];
        Ok(Box::new(events.into_iter()))
    }

    fn fetch_cryptogram(&self, token_ref: &str, amount: Money) -> Result<Cryptogram> {
        let amt = amount.minor_units.to_string();
        let h = Self::stable_hash(&[token_ref, &amt]);
        let avv = format!("MDES_AAV_{h:016x}");
        let expires_at = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap()
            + Duration::minutes(5);
        // Mastercard uses ECI "02" for fully authenticated.
        Ok(Cryptogram::new(avv, "02".into(), expires_at))
    }
}

/// Real HTTP integration, gated behind the `live` feature.
#[cfg(feature = "live")]
pub mod live {
    //! Real MDES HTTP calls (OAuth1, `api.mastercard.com`).
    //!
    //! Not implemented in this milestone.

    /// Marker: the live implementation is not yet shipped.
    pub const STATUS: &str = "MDES live HTTP integration: not yet implemented";
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    fn provider() -> MdesProvider {
        MdesProvider::new(ProviderConfig::new(CardNetwork::Mastercard, "tr_test_mc"))
    }

    #[test]
    fn stub_provision_is_deterministic() {
        let p = provider();
        let r = VaultRef::new("tok_vault_mc_1");
        let a = p.provision(&r).unwrap();
        let b = p.provision(&r).unwrap();
        assert_eq!(a.token_ref, b.token_ref);
        assert_eq!(a.network, CardNetwork::Mastercard);
    }

    #[test]
    fn stub_lifecycle_emits_provisioned_then_suspended() {
        let p = provider();
        let events: Vec<_> = p.lifecycle("tok_net_mc_1").unwrap().collect();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0].event,
            NetworkTokenLifecycleEvent::Provisioned
        ));
        assert!(matches!(
            events[1].event,
            NetworkTokenLifecycleEvent::Suspended
        ));
    }

    #[test]
    fn stub_cryptogram_eci_is_mc_authenticated() {
        let p = provider();
        let c = p
            .fetch_cryptogram("tok_net_mc_1", Money::from_minor(1, Currency::USD))
            .unwrap();
        assert_eq!(c.eci, "02");
    }
}
