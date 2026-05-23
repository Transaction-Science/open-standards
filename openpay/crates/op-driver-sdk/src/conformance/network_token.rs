#![allow(
    clippy::items_after_test_module,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

//! Conformance harness for [`op_rails_card::NetworkTokenProvider`].
//!
//! Drives a candidate provider through a battery of contract checks
//! that mirror the network-token trait's behavioral guarantees:
//!
//! 1. `name()` is non-empty.
//! 2. `provision(card_ref)` returns a `NetworkToken` whose `network`
//!    matches the provider's declared `network()`.
//! 3. Two distinct `card_ref`s produce distinct `token_ref`s.
//! 4. `lifecycle(token_ref)` yields a serialisable event sequence.
//! 5. `fetch_cryptogram(token_ref, amount)` is deterministic for the
//!    same inputs (stub mode) and produces an ECI value.
//!
//! Real-network determinism is not required of live adapters; the
//! determinism check is documented as applicable only when the
//! adapter is configured in its stub / sandbox profile. The harness
//! does not introspect provider mode — it simply asserts the
//! invariant the docs promise.

use op_core::{CardNetwork, Currency, Money, VaultRef};
use op_rails_card::NetworkTokenProvider;
use serde::{Deserialize, Serialize};

/// What a network-token conformance run failed on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkTokenConformanceFailure {
    /// `name()` returned an empty string.
    EmptyName,
    /// `provision` returned a token whose `network` does not match
    /// the provider's declared `network()`.
    NetworkMismatch {
        /// What `network()` returned.
        provider: CardNetwork,
        /// What `provision().network` returned.
        token: CardNetwork,
    },
    /// `provision` returned a `NetworkToken` with an empty
    /// `token_ref` — downstream calls have nothing to key on.
    EmptyTokenRef,
    /// Two distinct vault references produced the same token
    /// reference. Real provisioning calls must be 1:1; collapsing
    /// distinct PANs onto the same token violates the network's
    /// uniqueness guarantee.
    DuplicateTokenRefAcrossVaults,
    /// `lifecycle` returned no events for a freshly provisioned
    /// token. The contract requires at minimum a
    /// `NetworkTokenLifecycleEvent::Provisioned` to be observable.
    EmptyLifecycleStream,
    /// A lifecycle event failed to round-trip through JSON
    /// serialisation. The orchestrator persists these on the webhook
    /// fanout path; non-serialisable events would crash that pipeline.
    LifecycleSerialisationFailed {
        /// Why serde failed.
        reason: String,
    },
    /// `fetch_cryptogram` is not deterministic for the same
    /// `(token_ref, amount)` pair when the provider claims it should
    /// be (stub / test mode).
    CryptogramNonDeterministic,
    /// `fetch_cryptogram` returned an empty `eci` string. The ECI is
    /// forwarded verbatim to the acquirer and must be present.
    CryptogramMissingEci,
    /// `fetch_cryptogram` returned an empty `avv` string.
    CryptogramMissingAvv,
}

impl core::fmt::Display for NetworkTokenConformanceFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyName => write!(f, "name() returned an empty string"),
            Self::NetworkMismatch { provider, token } => write!(
                f,
                "provision returned a token on {token:?} but provider declares {provider:?}"
            ),
            Self::EmptyTokenRef => write!(f, "provision returned an empty token_ref"),
            Self::DuplicateTokenRefAcrossVaults => {
                write!(f, "two distinct vault refs produced the same token_ref")
            }
            Self::EmptyLifecycleStream => {
                write!(f, "lifecycle() returned no events")
            }
            Self::LifecycleSerialisationFailed { reason } => {
                write!(f, "lifecycle event failed to serialise: {reason}")
            }
            Self::CryptogramNonDeterministic => write!(
                f,
                "fetch_cryptogram returned different values for the same inputs"
            ),
            Self::CryptogramMissingEci => write!(f, "cryptogram is missing its eci"),
            Self::CryptogramMissingAvv => write!(f, "cryptogram is missing its avv"),
        }
    }
}

/// Aggregated report from a conformance run.
#[derive(Debug, Clone)]
pub struct NetworkTokenConformanceReport {
    /// Which provider was tested (`provider.name()`).
    pub provider_name: String,
    /// The network the provider declared (`provider.network()`).
    pub network: CardNetwork,
    /// How many individual checks ran.
    pub checks_run: usize,
    /// Every check that produced a failure. Empty = green.
    pub failures: Vec<NetworkTokenConformanceFailure>,
}

impl NetworkTokenConformanceReport {
    /// True iff every check passed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Drive a network-token provider through the conformance battery.
///
/// Returns a [`NetworkTokenConformanceReport`]; an empty `failures`
/// vec means the provider satisfied every checked contract clause.
pub fn run_network_token<P: NetworkTokenProvider + ?Sized>(
    provider: &P,
) -> NetworkTokenConformanceReport {
    let mut failures = Vec::new();
    let mut checks_run = 0;

    // 1. name() non-empty.
    if provider.name().trim().is_empty() {
        failures.push(NetworkTokenConformanceFailure::EmptyName);
    }
    checks_run += 1;

    // 2. provision returns a token on the declared network with a
    //    non-empty token_ref.
    let vault_a = VaultRef::new("conf_net_vault_a");
    let provisioned_a = provider.provision(&vault_a);
    if let Ok(ref token) = provisioned_a {
        if token.token_ref.is_empty() {
            failures.push(NetworkTokenConformanceFailure::EmptyTokenRef);
        }
        if token.network != provider.network() {
            failures.push(NetworkTokenConformanceFailure::NetworkMismatch {
                provider: provider.network(),
                token: token.network,
            });
        }
    }
    checks_run += 1;

    // 3. Distinct vault refs -> distinct token refs.
    let vault_b = VaultRef::new("conf_net_vault_b");
    let provisioned_b = provider.provision(&vault_b);
    if let (Ok(a), Ok(b)) = (&provisioned_a, &provisioned_b)
        && !a.token_ref.is_empty()
        && a.token_ref == b.token_ref
    {
        failures.push(NetworkTokenConformanceFailure::DuplicateTokenRefAcrossVaults);
    }
    checks_run += 1;

    // 4. lifecycle() returns a drainable, serialisable sequence.
    // Live providers may legitimately return Err for an unknown
    // token in test mode; we only fail on the stub-mode "empty
    // stream" or serialisation failures.
    if let Ok(ref token) = provisioned_a
        && let Ok(stream) = provider.lifecycle(&token.token_ref)
    {
        let events: Vec<_> = stream.collect();
        if events.is_empty() {
            failures.push(NetworkTokenConformanceFailure::EmptyLifecycleStream);
        }
        for ev in &events {
            if let Err(e) = serde_json::to_string(ev) {
                failures.push(NetworkTokenConformanceFailure::LifecycleSerialisationFailed {
                    reason: e.to_string(),
                });
                break;
            }
        }
    }
    checks_run += 1;

    // 5. fetch_cryptogram determinism for the same inputs.
    if let Ok(ref token) = provisioned_a {
        let amount = Money::from_minor(1234, Currency::USD);
        let c1 = provider.fetch_cryptogram(&token.token_ref, amount);
        let c2 = provider.fetch_cryptogram(&token.token_ref, amount);
        if let (Ok(c1), Ok(c2)) = (&c1, &c2) {
            if c1.avv.is_empty() {
                failures.push(NetworkTokenConformanceFailure::CryptogramMissingAvv);
            }
            if c1.eci.is_empty() {
                failures.push(NetworkTokenConformanceFailure::CryptogramMissingEci);
            }
            if c1.avv != c2.avv || c1.eci != c2.eci {
                failures.push(NetworkTokenConformanceFailure::CryptogramNonDeterministic);
            }
        }
    }
    checks_run += 1;

    NetworkTokenConformanceReport {
        provider_name: provider.name().to_owned(),
        network: provider.network(),
        checks_run,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_rails_card::network_token::{MdesProvider, ProviderConfig, VtsProvider};

    #[test]
    fn vts_stub_passes_conformance() {
        let p = VtsProvider::new(ProviderConfig::new(CardNetwork::Visa, "tr_test_conf"));
        let report = run_network_token(&p);
        assert!(
            report.is_clean(),
            "VTS stub failed conformance: {:?}",
            report.failures
        );
        assert_eq!(report.provider_name, "vts");
        assert_eq!(report.network, CardNetwork::Visa);
    }

    #[test]
    fn mdes_stub_passes_conformance() {
        let p = MdesProvider::new(ProviderConfig::new(
            CardNetwork::Mastercard,
            "tr_test_conf_mc",
        ));
        let report = run_network_token(&p);
        assert!(
            report.is_clean(),
            "MDES stub failed conformance: {:?}",
            report.failures
        );
        assert_eq!(report.provider_name, "mdes");
        assert_eq!(report.network, CardNetwork::Mastercard);
    }

    /// Confirm the harness catches a provider that lies about its
    /// network.
    #[test]
    fn detects_network_mismatch() {
        use op_core::{NetworkToken, NetworkTokenLifecycleEvent};
        use op_rails_card::network_token::{Cryptogram, LifecycleEvent, LifecycleStream};

        struct LyingProvider;
        impl NetworkTokenProvider for LyingProvider {
            fn name(&self) -> &'static str {
                "liar"
            }
            fn network(&self) -> CardNetwork {
                CardNetwork::Visa
            }
            fn provision(&self, _r: &VaultRef) -> op_rails_card::Result<NetworkToken> {
                // BUG: claims Visa via network() but returns Mastercard.
                Ok(NetworkToken::new(
                    "tok_lie",
                    "4242",
                    CardNetwork::Mastercard,
                    true,
                ))
            }
            fn lifecycle(&self, token_ref: &str) -> op_rails_card::Result<LifecycleStream> {
                let ev = LifecycleEvent {
                    token_ref: token_ref.to_owned(),
                    event: NetworkTokenLifecycleEvent::Provisioned,
                    at: time::OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
                    reason: None,
                };
                Ok(Box::new(std::iter::once(ev)))
            }
            fn fetch_cryptogram(
                &self,
                _token_ref: &str,
                _amount: Money,
            ) -> op_rails_card::Result<Cryptogram> {
                Ok(Cryptogram::new(
                    "AVV".into(),
                    "05".into(),
                    time::OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
                ))
            }
        }

        let report = run_network_token(&LyingProvider);
        assert!(
            report
                .failures
                .iter()
                .any(|f| matches!(f, NetworkTokenConformanceFailure::NetworkMismatch { .. })),
            "expected NetworkMismatch failure, got {:?}",
            report.failures
        );
    }

    /// Confirm the harness catches a provider that returns the same
    /// token_ref for distinct vault refs.
    #[test]
    fn detects_duplicate_token_ref() {
        use op_core::{NetworkToken, NetworkTokenLifecycleEvent};
        use op_rails_card::network_token::{Cryptogram, LifecycleEvent, LifecycleStream};

        struct DuplicateProvider;
        impl NetworkTokenProvider for DuplicateProvider {
            fn name(&self) -> &'static str {
                "dup"
            }
            fn network(&self) -> CardNetwork {
                CardNetwork::Visa
            }
            fn provision(&self, _r: &VaultRef) -> op_rails_card::Result<NetworkToken> {
                Ok(NetworkToken::new(
                    "tok_always_same",
                    "4242",
                    CardNetwork::Visa,
                    true,
                ))
            }
            fn lifecycle(&self, token_ref: &str) -> op_rails_card::Result<LifecycleStream> {
                let ev = LifecycleEvent {
                    token_ref: token_ref.to_owned(),
                    event: NetworkTokenLifecycleEvent::Provisioned,
                    at: time::OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
                    reason: None,
                };
                Ok(Box::new(std::iter::once(ev)))
            }
            fn fetch_cryptogram(
                &self,
                _t: &str,
                _a: Money,
            ) -> op_rails_card::Result<Cryptogram> {
                Ok(Cryptogram::new(
                    "AVV".into(),
                    "05".into(),
                    time::OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap(),
                ))
            }
        }

        let report = run_network_token(&DuplicateProvider);
        assert!(
            report
                .failures
                .contains(&NetworkTokenConformanceFailure::DuplicateTokenRefAcrossVaults)
        );
    }
}

/// Compile-fail proof: the function below would not compile if you
/// tried to pass a `Vaulted<Card>` where `Tokenized<Card>` is
/// required. We keep this as a doctest so the type guarantee is
/// observable from rustdoc and CI.
///
/// ```compile_fail
/// use op_core::{Card, CardNetwork, Vaulted, VaultRef};
/// use op_rails_card::network_token::{route_to_token_only_rail, TokenOnlyRail};
///
/// struct DirectVtsRail;
/// impl TokenOnlyRail for DirectVtsRail {
///     fn name(&self) -> &'static str { "vts-direct" }
/// }
///
/// let v: Vaulted<Card> = Vaulted::new(
///     Card::new(CardNetwork::Visa, "4242".into(), 12, 2030),
///     VaultRef::new("tok_psp_vault"),
/// );
/// // ERROR: expected `Tokenized<Card>`, found `Vaulted<Card>`.
/// route_to_token_only_rail(&DirectVtsRail, v);
/// ```
///
/// The positive case (with a [`op_core::Tokenized<Card>`]) compiles
/// and runs:
///
/// ```
/// use op_core::{Card, CardNetwork, NetworkToken, Tokenized};
/// use op_rails_card::network_token::{route_to_token_only_rail, TokenOnlyRail};
///
/// struct DirectVtsRail;
/// impl TokenOnlyRail for DirectVtsRail {
///     fn name(&self) -> &'static str { "vts-direct" }
/// }
///
/// let t: Tokenized<Card> = Tokenized::new(
///     Card::new(CardNetwork::Visa, "4242".into(), 12, 2030),
///     NetworkToken::new("tok_net_x", "4242", CardNetwork::Visa, true),
/// );
/// let _ = route_to_token_only_rail(&DirectVtsRail, t);
/// ```
pub const COMPILE_FAIL_FIXTURE_DOC: &str =
    "see the doctest on this constant for the compile_fail / pass pair";
