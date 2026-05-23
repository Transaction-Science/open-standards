//! Network-token typestate.
//!
//! A *network token* is a PAN surrogate issued by a card network's
//! tokenization service — Visa Token Service (VTS), Mastercard Digital
//! Enablement Service (MDES), American Express Token Service, or
//! Discover Network Tokenization. Unlike a PSP vault token (see
//! [`crate::method::VaultRef`]), a network token is recognized by the
//! issuer and the network's rails directly:
//!
//! 1. **Liability shift.** Network-tokenized transactions carry a
//!    cryptogram that authenticates the device/credential to the
//!    issuer. On approval, chargeback liability for fraud shifts to
//!    the issuer (same posture as 3DS-authenticated transactions).
//!
//! 2. **Cross-merchant portability within consented network rails.**
//!    Once a credential is provisioned to a token-requester scope,
//!    the same token reference may be used across merchants that the
//!    network has authorized for that consent surface — the wallet
//!    case (Apple Pay, Google Pay, Click to Pay). This is distinct
//!    from a vault token, which is scoped to a single PSP / merchant
//!    account.
//!
//! 3. **PAN never exposed downstream.** The acquirer, processor,
//!    and merchant systems see only the token reference + cryptogram.
//!    PAN material lives behind the network's tokenization service
//!    and is never visible to `OpenPay` code paths. This is the
//!    PCI DSS scope-reduction primitive that makes
//!    `Tokenized<Card>` *strictly safer* than `Vaulted<Card>`.
//!
//! ## Typestate distinction
//!
//! [`crate::method::VaultRef`] models a card that has been *vaulted*:
//! the PAN is encrypted at rest in a PCI-certified vault and
//! addressable by an opaque PSP-issued token. [`Tokenized<C>`] models
//! a card that has been *network-tokenized*: the credential has been
//! re-issued as a network surrogate with its own life cycle (provision
//! → update → suspend/resume → delete) tracked by the network.
//!
//! Routing logic that requires the liability-shift / no-PAN-downstream
//! guarantees can constrain itself to `Tokenized<Card>` at the type
//! level; passing a `Vaulted<Card>` will fail to compile. See the
//! `TokenOnlyRail` example in
//! `op-rails-card::network_token` and the `trybuild` fixture in
//! `op-driver-sdk::conformance::network_token`.

use serde::{Deserialize, Serialize};

/// A card network that operates a network-token service.
///
/// Each network runs its own tokenization service with a distinct
/// cryptogram format, lifecycle webhook contract, and token-requester
/// onboarding flow. The [`crate::network_token::NetworkTokenProvider`]
/// trait (in `op-rails-card`) abstracts over these.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardNetwork {
    /// Visa — tokens issued by Visa Token Service (VTS). Cryptogram
    /// format is the TAVV (Token Authentication Verification Value),
    /// 28 bytes base64-encoded.
    Visa,
    /// Mastercard — tokens issued by Mastercard Digital Enablement
    /// Service (MDES). Cryptogram is the UCAF (Universal Cardholder
    /// Authentication Field).
    Mastercard,
    /// American Express — tokens issued by Amex Token Service.
    Amex,
    /// Discover — tokens issued by Discover Network Tokenization.
    Discover,
}

impl CardNetwork {
    /// Canonical short identifier (`"visa"`, `"mc"`, `"amex"`,
    /// `"discover"`) used in wire formats and provider routing.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Visa => "visa",
            Self::Mastercard => "mc",
            Self::Amex => "amex",
            Self::Discover => "discover",
        }
    }
}

/// A provisioned network token.
///
/// `token_ref` is opaque to `OpenPay`: it's the handle the network's
/// tokenization service issued (the `tokenRequestorId`-scoped
/// `tokenReferenceId`, or equivalent). It is **not** a PAN, **not** a
/// PSP vault id, and **not** a cryptogram — those are separate
/// concepts in the tokenization model.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NetworkToken {
    /// Network-issued token reference. Opaque string; format varies
    /// by network. Treat as PII and never log in full.
    pub token_ref: String,
    /// Last four digits of the underlying card, safe to log per
    /// PCI DSS 4.0.1 §3.4.1.
    pub last4: String,
    /// Which network issued the token.
    pub network: CardNetwork,
    /// True if the network supports issuing per-transaction
    /// cryptograms for this token. Wallet-provisioned tokens always
    /// support cryptograms; some card-on-file token-requester
    /// scopes may not.
    pub cryptogram_supported: bool,
}

impl NetworkToken {
    /// Construct. Caller is responsible for ensuring `token_ref`
    /// came from the network's tokenization service and not from
    /// user input.
    #[must_use]
    pub fn new(
        token_ref: impl Into<String>,
        last4: impl Into<String>,
        network: CardNetwork,
        cryptogram_supported: bool,
    ) -> Self {
        Self {
            token_ref: token_ref.into(),
            last4: last4.into(),
            network,
            cryptogram_supported,
        }
    }
}

/// A lifecycle event emitted by the network's tokenization service
/// for a provisioned token.
///
/// These map directly onto the events the networks emit on their
/// webhook surfaces:
///
/// - VTS: `PROVISIONED`, `UPDATED`, `SUSPENDED`, `RESUMED`, `DELETED`.
/// - MDES: `TOKEN_ACTIVATED`, `TOKEN_UPDATED`, `TOKEN_SUSPENDED`,
///   `TOKEN_RESUMED`, `TOKEN_DEACTIVATED`.
///
/// Drivers translate the network-specific names into this normalized
/// enum so downstream consumers (the ledger, webhook fanout) see a
/// uniform shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkTokenLifecycleEvent {
    /// Token was issued. Carries the credential into the active set.
    Provisioned,
    /// Token metadata changed (new expiry, new last4 after issuer
    /// reissue, art file refresh). The `token_ref` is unchanged.
    Updated,
    /// Token is temporarily disabled. Authorize attempts will
    /// hard-decline until a `Resumed` event arrives.
    Suspended,
    /// Token was re-enabled after suspension.
    Resumed,
    /// Token was permanently deactivated. Must be re-provisioned for
    /// future use; the `token_ref` will not come back.
    Deleted,
}

/// Typestate marker for a card that has been provisioned with a
/// network token.
///
/// `Tokenized<C>` wraps an inner card descriptor `C` together with
/// the provisioned [`NetworkToken`]. It is **distinct** from
/// `Vaulted<C>` (PAN-in-vault, no liability shift, single-PSP scope).
/// Routing code that requires the network-token guarantees should
/// take `Tokenized<C>` by value or by reference; passing a
/// `Vaulted<C>` will fail to compile.
///
/// The wrapper carries no behavior of its own — it's a phantom-state
/// marker that promotes a runtime distinction (which kind of token
/// is this?) into a compile-time one (the type-checker rejects code
/// paths that mix them up).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokenized<C> {
    /// The underlying card descriptor (BIN range, expiry hint, etc.).
    /// Opaque to `op-core`; rail drivers project it onto their
    /// PSP-specific shapes.
    pub card: C,
    /// The network-issued token.
    pub token: NetworkToken,
}

impl<C> Tokenized<C> {
    /// Wrap a card descriptor with its provisioned network token.
    pub const fn new(card: C, token: NetworkToken) -> Self {
        Self { card, token }
    }

    /// Borrow the network token.
    pub const fn token(&self) -> &NetworkToken {
        &self.token
    }

    /// Borrow the inner card descriptor.
    pub const fn card(&self) -> &C {
        &self.card
    }
}

/// Typestate marker for a card whose PAN has been stored in a PSP /
/// PCI-certified vault and is addressable by an opaque
/// [`crate::method::VaultRef`].
///
/// This is the *non*-network-tokenized branch of the card-credential
/// taxonomy: the PAN exists at rest in a vault, downstream calls may
/// see the vault token, and chargeback liability follows ordinary
/// card-present / card-not-present rules without the network's
/// cryptogram-based liability shift.
///
/// `Vaulted<C>` is exposed alongside [`Tokenized<C>`] so that downstream
/// rail code can branch on the credential's provenance in the type
/// system rather than at runtime.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vaulted<C> {
    /// The underlying card descriptor.
    pub card: C,
    /// The PSP-issued vault reference.
    pub vault_ref: crate::method::VaultRef,
}

impl<C> Vaulted<C> {
    /// Wrap a card descriptor with its PSP vault reference.
    pub const fn new(card: C, vault_ref: crate::method::VaultRef) -> Self {
        Self { card, vault_ref }
    }

    /// Borrow the vault reference.
    pub const fn vault_ref(&self) -> &crate::method::VaultRef {
        &self.vault_ref
    }

    /// Borrow the inner card descriptor.
    pub const fn card(&self) -> &C {
        &self.card
    }
}

/// A card credential descriptor.
///
/// Carries the BIN-range-derived network hint and the expiry. Holds no
/// PAN material. Wrap in [`Vaulted`] or [`Tokenized`] to obtain a usable
/// credential.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Card {
    /// Network the underlying card was issued on (derived from BIN).
    pub network: CardNetwork,
    /// Last four digits of the underlying PAN. Safe to log.
    pub last4: String,
    /// Two-digit expiry month, 1-12.
    pub exp_month: u8,
    /// Four-digit expiry year (e.g. 2030).
    pub exp_year: u16,
}

impl Card {
    /// Construct a card descriptor.
    #[must_use]
    pub const fn new(network: CardNetwork, last4: String, exp_month: u8, exp_year: u16) -> Self {
        Self {
            network,
            last4,
            exp_month,
            exp_year,
        }
    }
}

/// Sealed marker trait identifying the two card-credential
/// typestates: [`Vaulted`] and [`Tokenized`]. Implemented exclusively
/// for those two types in this crate; downstream code cannot add new
/// branches.
pub trait PaymentMethodKind: sealed::Sealed {
    /// Human-readable name (`"vaulted"` or `"tokenized"`) used in
    /// diagnostics and routing tables.
    const KIND: &'static str;
}

mod sealed {
    pub trait Sealed {}
    impl<C> Sealed for super::Vaulted<C> {}
    impl<C> Sealed for super::Tokenized<C> {}
}

impl<C> PaymentMethodKind for Vaulted<C> {
    const KIND: &'static str = "vaulted";
}

impl<C> PaymentMethodKind for Tokenized<C> {
    const KIND: &'static str = "tokenized";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::VaultRef;

    fn sample_card() -> Card {
        Card::new(CardNetwork::Visa, "4242".into(), 12, 2030)
    }

    #[test]
    fn card_network_id_stable() {
        assert_eq!(CardNetwork::Visa.id(), "visa");
        assert_eq!(CardNetwork::Mastercard.id(), "mc");
        assert_eq!(CardNetwork::Amex.id(), "amex");
        assert_eq!(CardNetwork::Discover.id(), "discover");
    }

    #[test]
    fn tokenized_and_vaulted_are_distinct_kinds() {
        assert_ne!(
            <Vaulted<Card> as PaymentMethodKind>::KIND,
            <Tokenized<Card> as PaymentMethodKind>::KIND
        );
    }

    #[test]
    fn tokenized_card_accessors() {
        let token = NetworkToken::new("tok_net_test", "4242", CardNetwork::Visa, true);
        let card = sample_card();
        let t = Tokenized::new(card, token);
        assert_eq!(t.token().token_ref, "tok_net_test");
        assert_eq!(t.card().last4, "4242");
    }

    #[test]
    fn vaulted_card_accessors() {
        let v = Vaulted::new(sample_card(), VaultRef::new("tok_psp_vault"));
        assert_eq!(v.vault_ref().as_str(), "tok_psp_vault");
        assert_eq!(v.card().last4, "4242");
    }

    #[test]
    fn lifecycle_events_serialize() {
        let json = serde_json::to_string(&NetworkTokenLifecycleEvent::Provisioned).unwrap();
        assert!(json.contains("Provisioned"));
    }
}
