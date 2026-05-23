//! Visa Token Service (VTS) adapter.
//!
//! ## Production surface
//!
//! Real VTS calls go to `https://api.visa.com/vts/...` over mTLS with
//! a Visa-issued client certificate. The full integration is gated
//! behind the `live` cargo feature on this crate (see [`live`]); the
//! default build uses the deterministic stubs in this module so the
//! workspace compiles and the conformance harness is reproducible
//! without network access or Visa sandbox credentials.
//!
//! ## Stub determinism
//!
//! Every stub method returns a value derived from a stable hash of
//! its inputs. Given the same `card_ref` (or `token_ref + amount`),
//! the stub returns the same `NetworkToken` / `Cryptogram`. This is
//! load-bearing: the conformance harness asserts reproducibility, and
//! a non-deterministic stub would produce a flaky harness.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use op_core::{CardNetwork, Money, NetworkToken, NetworkTokenLifecycleEvent, VaultRef};
use time::{Duration, OffsetDateTime};

use crate::Result;
use crate::network_token::cryptogram::Cryptogram;
use crate::network_token::provider::{
    LifecycleEvent, LifecycleStream, NetworkTokenProvider, ProviderConfig,
};

/// Visa Token Service provider.
///
/// Constructed once at startup with a [`ProviderConfig`]. The stub
/// implementation holds the config only; the live implementation also
/// owns an HTTP client and the mTLS bundle.
pub struct VtsProvider {
    config: ProviderConfig,
}

impl VtsProvider {
    /// Construct with the given config. Panics if `config.network` is
    /// not [`CardNetwork::Visa`] — wiring a Mastercard requestor to
    /// the VTS adapter is a configuration error caught at startup.
    #[must_use]
    pub fn new(config: ProviderConfig) -> Self {
        assert!(
            matches!(config.network, CardNetwork::Visa),
            "VtsProvider requires CardNetwork::Visa, got {:?}",
            config.network
        );
        Self { config }
    }

    /// Borrow the config (useful for adapters that wrap this one).
    #[must_use]
    pub const fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Hash an input deterministically. Used by every stub method so
    /// the conformance harness sees reproducible values.
    fn stable_hash(parts: &[&str]) -> u64 {
        let mut h = DefaultHasher::new();
        for p in parts {
            p.hash(&mut h);
        }
        h.finish()
    }
}

impl NetworkTokenProvider for VtsProvider {
    fn name(&self) -> &'static str {
        "vts"
    }

    fn network(&self) -> CardNetwork {
        CardNetwork::Visa
    }

    fn provision(&self, card_ref: &VaultRef) -> Result<NetworkToken> {
        // STUB: deterministic token derived from the vault ref. The
        // live implementation calls POST /vts/provisioning/tokens with
        // mTLS and returns the network-issued tokenReferenceId.
        let h = Self::stable_hash(&[card_ref.as_str(), &self.config.token_requestor_id]);
        let token_ref = format!("vts_stub_{h:016x}");
        Ok(NetworkToken::new(
            token_ref,
            "4242", // stub: real impl reads from the network response
            CardNetwork::Visa,
            true,
        ))
    }

    fn lifecycle(&self, token_ref: &str) -> Result<LifecycleStream> {
        // STUB: emit a deterministic canned sequence so the
        // conformance harness can drain and assert. Real VTS surfaces
        // these via webhook callbacks.
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
                event: NetworkTokenLifecycleEvent::Updated,
                at: base + Duration::seconds(60),
                reason: Some("stub:art_refresh".into()),
            },
        ];
        Ok(Box::new(events.into_iter()))
    }

    fn fetch_cryptogram(&self, token_ref: &str, amount: Money) -> Result<Cryptogram> {
        // STUB: deterministic cryptogram derived from (token_ref,
        // amount). Live impl calls VTS's cryptogramRetrieval surface.
        let amt = amount.minor_units.to_string();
        let h = Self::stable_hash(&[token_ref, &amt]);
        let avv = format!("VTS_TAVV_{h:016x}");
        // Use the unix epoch for stub determinism rather than
        // `OffsetDateTime::now_utc()`, so test runs are reproducible.
        let expires_at = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap()
            + Duration::minutes(5);
        Ok(Cryptogram::new(avv, "05".into(), expires_at))
    }
}

/// Real HTTP integration. Compiled in only when the `live` cargo
/// feature is enabled.
#[cfg(feature = "live")]
pub mod live {
    //! Real VTS HTTP calls (mTLS, `api.visa.com`).
    //!
    //! Not implemented in this milestone — the trait + stubs land
    //! first so downstream code can integrate against the type
    //! signatures. The live module exists as a feature gate so
    //! operators can opt into the live surface without conditional
    //! re-exports in `mod.rs`.

    /// Marker: the live implementation is not yet shipped.
    pub const STATUS: &str = "VTS live HTTP integration: not yet implemented";
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    fn provider() -> VtsProvider {
        VtsProvider::new(ProviderConfig::new(CardNetwork::Visa, "tr_test"))
    }

    #[test]
    fn stub_provision_is_deterministic() {
        let p = provider();
        let r = VaultRef::new("tok_vault_1");
        let a = p.provision(&r).unwrap();
        let b = p.provision(&r).unwrap();
        assert_eq!(a.token_ref, b.token_ref);
        assert_eq!(a.network, CardNetwork::Visa);
    }

    #[test]
    fn stub_lifecycle_emits_canned_sequence() {
        let p = provider();
        let stream: Vec<_> = p.lifecycle("tok_net_1").unwrap().collect();
        assert_eq!(stream.len(), 2);
        assert!(matches!(
            stream[0].event,
            NetworkTokenLifecycleEvent::Provisioned
        ));
        assert!(matches!(
            stream[1].event,
            NetworkTokenLifecycleEvent::Updated
        ));
    }

    #[test]
    fn stub_cryptogram_is_deterministic_for_same_inputs() {
        let p = provider();
        let a = p
            .fetch_cryptogram("tok_net_1", Money::from_minor(1234, Currency::USD))
            .unwrap();
        let b = p
            .fetch_cryptogram("tok_net_1", Money::from_minor(1234, Currency::USD))
            .unwrap();
        assert_eq!(a.avv, b.avv);
        assert_eq!(a.eci, "05");
    }

    #[test]
    fn stub_cryptogram_differs_across_inputs() {
        let p = provider();
        let a = p
            .fetch_cryptogram("tok_net_1", Money::from_minor(1234, Currency::USD))
            .unwrap();
        let b = p
            .fetch_cryptogram("tok_net_1", Money::from_minor(9999, Currency::USD))
            .unwrap();
        assert_ne!(a.avv, b.avv);
    }
}
