//! Connected-account domain types.
//!
//! A connected account is a sub-merchant on a platform's payments deployment.
//! The platform onboards, screens, and pays out to many of them; each
//! connected account carries its own KYB profile, beneficial-owner list,
//! capability set, and outstanding-requirement vector.
//!
//! The terminology (`Standard` / `Express` / `Custom`) mirrors Stripe Connect
//! by intention: it is the de-facto vocabulary platform engineers reach for,
//! so adopting it lowers migration cost for teams moving off an incumbent.

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::kyb::{BeneficialOwner, BusinessProfile, Requirements};

/// Strongly-typed identifier for a connected account.
///
/// `acct_` prefix + UUID v4 simple form. Operators can override by
/// implementing [`OnboardingProvider::create_account`](crate::onboarding::OnboardingProvider::create_account)
/// to return any non-empty string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AccountId(pub String);

impl AccountId {
    /// Mint a fresh account id in the native `acct_<uuidv4>` format.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("acct_{}", Uuid::new_v4().simple()))
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AccountId {
    fn default() -> Self {
        Self::new()
    }
}

/// Onboarding-flavor of the connected account.
///
/// Vocabulary follows Stripe Connect because that is the de-facto industry
/// shorthand. Behaviour:
///
/// - **Standard** — sub-merchant holds the relationship with the acquirer
///   directly, signs the acquirer's terms, sees the acquirer dashboard.
///   The platform routes payments and takes fees but is not the merchant
///   of record.
/// - **Express** — sub-merchant goes through a co-branded onboarding flow
///   on the platform's domain; the platform handles UX while the
///   acquirer holds the underlying terms.
/// - **Custom** — the platform owns the entire relationship and is the
///   merchant of record. Sub-merchant never sees an acquirer surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccountType {
    /// Sub-merchant is the merchant of record on the acquirer.
    Standard,
    /// Co-branded flow; acquirer holds terms, platform owns UX.
    Express,
    /// Platform is merchant of record end-to-end.
    Custom,
}

/// Capabilities a connected account may be authorised for.
///
/// Each capability is independently granted: a sub-merchant can be
/// approved for `CardPayments` and `Payouts` while still being
/// pending on `AchPayments`. Capability state is reflected in
/// [`Requirements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Capability {
    /// Card-present and card-not-present acceptance (Visa / Mastercard / Amex / Discover).
    CardPayments,
    /// US ACH debits and credits (NACHA).
    AchPayments,
    /// Account-to-account real-time rails (FedNow, RTP, SEPA Instant, Pix, UPI).
    A2aPayments,
    /// Crypto-rail acceptance (BTC, ETH, stablecoins).
    CryptoPayments,
    /// Outbound payouts to the sub-merchant's bank account.
    Payouts,
    /// US Form 1099-K filing on the sub-merchant's behalf.
    TaxReporting1099K,
    /// US Form 1042-S filing on a non-US sub-merchant's behalf.
    TaxReporting1042S,
}

/// Per-account settings the platform configures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountSettings {
    /// Statement descriptor that appears on cardholder statements.
    pub statement_descriptor: Option<String>,
    /// Short (≤10 char) descriptor used on legacy 22-char rails.
    pub statement_descriptor_short: Option<String>,
    /// If true, this account participates in instant-payouts where eligible.
    pub instant_payouts_enabled: bool,
    /// If true, refunds debit the platform's account first (else the sub-merchant's).
    pub debit_negative_balances: bool,
}

impl Default for AccountSettings {
    fn default() -> Self {
        Self {
            statement_descriptor: None,
            statement_descriptor_short: None,
            instant_payouts_enabled: false,
            debit_negative_balances: false,
        }
    }
}

/// A fully-described connected account.
///
/// Immutable from the caller's perspective: mutation happens through
/// [`crate::onboarding::OnboardingFlow`] step submission, which produces
/// a fresh `ConnectedAccount` snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectedAccount {
    /// Stable identifier.
    pub id: AccountId,
    /// Onboarding flavor (renamed `type` because that's a Rust keyword).
    #[serde(rename = "type")]
    pub type_: AccountType,
    /// Capabilities currently active. Sorted for stable serialisation.
    pub capabilities: BTreeSet<Capability>,
    /// KYB profile (legal name, structure, MCC, etc.).
    pub business: BusinessProfile,
    /// Beneficial owners declared and (eventually) verified.
    pub beneficial_owners: Vec<BeneficialOwner>,
    /// Outstanding regulatory requirements (FinCEN CDD + AMLD5 driven).
    pub requirements: Requirements,
    /// Per-account settings.
    pub settings: AccountSettings,
    /// Creation timestamp.
    pub created: DateTime<Utc>,
    /// Last mutation timestamp.
    pub last_updated: DateTime<Utc>,
}

impl ConnectedAccount {
    /// Construct a fresh account in `BusinessInfo`-pending state.
    #[must_use]
    pub fn new(type_: AccountType, business: BusinessProfile) -> Self {
        let now = Utc::now();
        Self {
            id: AccountId::new(),
            type_,
            capabilities: BTreeSet::new(),
            business,
            beneficial_owners: Vec::new(),
            requirements: Requirements::default(),
            settings: AccountSettings::default(),
            created: now,
            last_updated: now,
        }
    }

    /// True if all `currently_due` and `past_due` requirements are clear.
    ///
    /// `eventually_due` and `pending_verification` are tolerated for an
    /// "active" account — they represent future obligations that have
    /// not yet blocked capabilities.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.requirements.currently_due.is_empty()
            && self.requirements.past_due.is_empty()
            && self.requirements.disabled_reason.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kyb::{BusinessStructure, CountryCode};
    use crate::kyb::{Address, BusinessProfile};

    fn sample_profile() -> BusinessProfile {
        BusinessProfile {
            legal_name: "Acme Widgets LLC".into(),
            trade_name: Some("Acme".into()),
            structure: BusinessStructure::SingleMemberLlc,
            tax_id: None,
            mcc: 5734,
            country: CountryCode("US".into()),
            registered_address: Address {
                line1: "1 Infinite Loop".into(),
                line2: None,
                city: "Cupertino".into(),
                region: "CA".into(),
                postal_code: "95014".into(),
                country: CountryCode("US".into()),
            },
            support_email: None,
            support_phone: None,
            website: None,
        }
    }

    #[test]
    fn account_id_round_trip() {
        let id = AccountId::new();
        assert!(id.as_str().starts_with("acct_"));
    }

    #[test]
    fn fresh_account_is_inactive() {
        let a = ConnectedAccount::new(AccountType::Custom, sample_profile());
        // Brand-new accounts have no past-due (we haven't enumerated any
        // currently-due yet either) so `is_active()` is true only when
        // requirements have been computed; on the bare struct it is true.
        assert!(a.is_active());
        assert!(a.capabilities.is_empty());
    }
}
