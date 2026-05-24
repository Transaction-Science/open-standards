//! Onboarding state machine.
//!
//! Sub-merchant onboarding is a stepwise process: collect business info,
//! collect a primary representative, collect beneficial owners, attach a
//! bank account, accept terms of service, verify the external account,
//! and finally verify the identity of the representative + owners.
//!
//! Each step has a corresponding `StepPayload` variant carrying the
//! data the step needs. The [`OnboardingProvider`] trait isolates the
//! actual side-effects so operators can wire one of:
//!
//! - **Native** — submit straight into the OpenPay account registry
//!   (this crate's [`NativeProvider`]); deterministic, in-process, the
//!   reference implementation for green-field deployments.
//! - **`StripeConnectAdapter`** — plug-in shape sketched in this module;
//!   operators migrating off Connect can implement the trait against
//!   Stripe's `/v1/accounts` API.
//! - **`AdyenMarketPayAdapter`** — same plug-in shape against Adyen's
//!   `Marketplace API`.
//!
//! Adapter implementations live downstream — this crate only ships the
//! trait surface and the native implementation. Documenting the
//! adapters here keeps the migration path explicit without forcing
//! the dependency on every operator.

use std::collections::BTreeMap;
use std::sync::Mutex;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::account::{AccountId, AccountSettings, AccountType, ConnectedAccount};
use crate::error::{Error, Result};
use crate::kyb::{
    BeneficialOwner, BusinessProfile, Requirements, validate_cdd,
};
use crate::tos::TosAcceptance;

/// Onboarding steps. Ordered loosely (some can be reordered or
/// concurrent), but [`OnboardingFlow::current_step`] always points at
/// the next-required-not-satisfied step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnboardingStep {
    /// Business profile + structure + tax ID.
    BusinessInfo,
    /// Primary representative (the natural person submitting the application).
    RepresentativeInfo,
    /// Beneficial owners per FinCEN CDD / AMLD5.
    BeneficialOwners,
    /// Sub-merchant's bank account for payouts.
    BankAccount,
    /// Terms-of-service acceptance.
    TosAcceptance,
    /// Verify the bank account (micro-deposit or Plaid-style flow).
    ExternalAccountVerification,
    /// Verify the representative's and owners' identities (document upload).
    IdentityVerification,
}

impl OnboardingStep {
    /// All steps in the canonical order.
    #[must_use]
    pub fn ordered() -> [Self; 7] {
        [
            Self::BusinessInfo,
            Self::RepresentativeInfo,
            Self::BeneficialOwners,
            Self::BankAccount,
            Self::TosAcceptance,
            Self::ExternalAccountVerification,
            Self::IdentityVerification,
        ]
    }
}

/// External account (bank account) for sub-merchant payouts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalAccount {
    /// ISO 3166-1 alpha-2 country.
    pub country: String,
    /// ISO 4217 currency code.
    pub currency: String,
    /// Routing number (US: ABA; EU: BIC) or local equivalent.
    pub routing: String,
    /// Account number (US: DDA; EU: IBAN).
    pub account_number: String,
    /// Account holder's name as it appears on the bank's records.
    pub account_holder_name: String,
}

/// Payload submitted with a single onboarding step.
///
/// Per-variant: the step-to-payload mapping is checked at runtime;
/// passing the wrong payload for a step returns [`Error::OnboardingStepInvalid`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepPayload {
    /// Body for [`OnboardingStep::BusinessInfo`].
    BusinessInfo(BusinessProfile),
    /// Body for [`OnboardingStep::RepresentativeInfo`].
    RepresentativeInfo(BeneficialOwner),
    /// Body for [`OnboardingStep::BeneficialOwners`]. Replaces the full roster.
    BeneficialOwners(Vec<BeneficialOwner>),
    /// Body for [`OnboardingStep::BankAccount`].
    BankAccount(ExternalAccount),
    /// Body for [`OnboardingStep::TosAcceptance`].
    TosAcceptance(TosAcceptance),
    /// Body for [`OnboardingStep::ExternalAccountVerification`].
    /// Carries the matched micro-deposit amounts (in minor units).
    ExternalAccountVerification {
        /// Two micro-deposit amounts the customer claims to have seen.
        amounts_minor: [i64; 2],
    },
    /// Body for [`OnboardingStep::IdentityVerification`]. Carries
    /// vault references to uploaded ID documents keyed by person's
    /// `legal_name`.
    IdentityVerification {
        /// Person's legal_name → vault reference to uploaded ID document.
        documents: BTreeMap<String, String>,
    },
}

impl StepPayload {
    /// Which step this payload satisfies.
    #[must_use]
    pub const fn step(&self) -> OnboardingStep {
        match self {
            Self::BusinessInfo(_) => OnboardingStep::BusinessInfo,
            Self::RepresentativeInfo(_) => OnboardingStep::RepresentativeInfo,
            Self::BeneficialOwners(_) => OnboardingStep::BeneficialOwners,
            Self::BankAccount(_) => OnboardingStep::BankAccount,
            Self::TosAcceptance(_) => OnboardingStep::TosAcceptance,
            Self::ExternalAccountVerification { .. } => {
                OnboardingStep::ExternalAccountVerification
            }
            Self::IdentityVerification { .. } => OnboardingStep::IdentityVerification,
        }
    }
}

/// Outcome of one [`OnboardingProvider::submit_step`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepResult {
    /// The step that was just submitted.
    pub step: OnboardingStep,
    /// Recomputed requirements after the step.
    pub requirements: Requirements,
    /// True if the flow can now advance past this step.
    pub accepted: bool,
    /// Operator-facing message (e.g. "micro-deposit amounts did not match").
    pub message: Option<String>,
}

/// Overall onboarding status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnboardingStatus {
    /// Account just created; no steps submitted.
    Created,
    /// At least one step submitted, but more remain.
    InProgress,
    /// All steps submitted; awaiting upstream verification of one or more.
    PendingVerification,
    /// All steps complete and verified; account is active.
    Active,
    /// Account is blocked; see `Requirements::disabled_reason`.
    Disabled,
}

/// The full onboarding flow for one connected account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnboardingFlow {
    /// Account being onboarded.
    pub acct_id: AccountId,
    /// Steps required (computed at flow start from [`AccountType`] +
    /// [`BusinessStructure`](crate::kyb::BusinessStructure)).
    pub steps: Vec<OnboardingStep>,
    /// Index into `steps` of the next step to submit.
    pub current_step: usize,
    /// Computed status.
    pub status: OnboardingStatus,
}

impl OnboardingFlow {
    /// Initial flow for a freshly created account.
    #[must_use]
    pub fn new(acct_id: AccountId) -> Self {
        Self {
            acct_id,
            steps: OnboardingStep::ordered().to_vec(),
            current_step: 0,
            status: OnboardingStatus::Created,
        }
    }
}

/// Side-effecting onboarding interface.
///
/// One implementation per backend. The reference implementation is
/// [`NativeProvider`] (in-process, tokio-friendly, no external calls).
///
/// Adapter shapes for incumbent migrations:
///
/// ```ignore
/// struct StripeConnectAdapter { client: stripe::Client, platform_acct: String }
/// impl OnboardingProvider for StripeConnectAdapter {
///     async fn create_account(&self, profile: &BusinessProfile) -> Result<AccountId> {
///         let acct = self.client.post("/v1/accounts", &serde_json::json!({
///             "type": "custom", "country": profile.country.0,
///             "business_type": match profile.structure { /* … */ },
///             // map BusinessProfile → Stripe params
///         })).await.map_err(|e| Error::Provider(e.to_string()))?;
///         Ok(AccountId(acct["id"].as_str().unwrap_or_default().to_string()))
///     }
///     /* submit_step / status mapping per Stripe Connect docs … */
/// }
///
/// struct AdyenMarketPayAdapter { client: AdyenClient }
/// impl OnboardingProvider for AdyenMarketPayAdapter { /* /v1/accountHolders */ }
/// ```
#[async_trait::async_trait]
pub trait OnboardingProvider: Send + Sync {
    /// Create a fresh connected account.
    ///
    /// # Errors
    /// Bubble any upstream provider failure as [`Error::Provider`].
    async fn create_account(&self, profile: &BusinessProfile) -> Result<AccountId>;

    /// Submit one onboarding step.
    ///
    /// # Errors
    /// - [`Error::AccountNotFound`] if `acct` is unknown.
    /// - [`Error::OnboardingStepInvalid`] if `step` does not match
    ///   `payload.step()` or the step is out of sequence.
    /// - [`Error::ScreeningBlocked`] if screening fired on a step.
    /// - [`Error::Provider`] for upstream provider errors.
    async fn submit_step(
        &self,
        acct: &AccountId,
        step: OnboardingStep,
        payload: StepPayload,
    ) -> Result<StepResult>;

    /// Fetch the latest onboarding status.
    ///
    /// # Errors
    /// [`Error::AccountNotFound`] if `acct` is unknown.
    async fn status(&self, acct: &AccountId) -> Result<OnboardingStatus>;
}

// Async trait helper. We pull `async_trait` in as a tiny dev-style
// dependency through tokio's transitive surface; if it isn't available
// we can collapse to a hand-rolled BoxFuture pattern.
#[doc(hidden)]
pub use async_trait as _async_trait_reexport;

/// Native, in-process [`OnboardingProvider`].
///
/// State lives in a `Mutex<BTreeMap>`; suitable for tests, kiosk
/// deployments, and the reference server. Production deployments
/// implement [`OnboardingProvider`] against their own account store.
pub struct NativeProvider {
    accounts: Mutex<BTreeMap<AccountId, ConnectedAccount>>,
    flows: Mutex<BTreeMap<AccountId, OnboardingFlow>>,
    bank_accounts: Mutex<BTreeMap<AccountId, ExternalAccount>>,
    tos: Mutex<BTreeMap<AccountId, TosAcceptance>>,
    /// Micro-deposit amounts the test rig "sent" to the bank account.
    /// In production this would be supplied by the rail driver.
    expected_micro_deposits: Mutex<BTreeMap<AccountId, [i64; 2]>>,
    account_type: AccountType,
}

impl NativeProvider {
    /// Build a fresh provider.
    #[must_use]
    pub fn new(account_type: AccountType) -> Self {
        Self {
            accounts: Mutex::new(BTreeMap::new()),
            flows: Mutex::new(BTreeMap::new()),
            bank_accounts: Mutex::new(BTreeMap::new()),
            tos: Mutex::new(BTreeMap::new()),
            expected_micro_deposits: Mutex::new(BTreeMap::new()),
            account_type,
        }
    }

    /// Read-only snapshot of an account.
    ///
    /// # Errors
    /// [`Error::AccountNotFound`].
    pub fn snapshot(&self, acct: &AccountId) -> Result<ConnectedAccount> {
        self.accounts
            .lock()
            .expect("accounts mutex poisoned")
            .get(acct)
            .cloned()
            .ok_or_else(|| Error::AccountNotFound(acct.0.clone()))
    }

    /// Read-only snapshot of the flow.
    ///
    /// # Errors
    /// [`Error::AccountNotFound`].
    pub fn flow(&self, acct: &AccountId) -> Result<OnboardingFlow> {
        self.flows
            .lock()
            .expect("flows mutex poisoned")
            .get(acct)
            .cloned()
            .ok_or_else(|| Error::AccountNotFound(acct.0.clone()))
    }

    /// Inject the micro-deposit amounts the test rig "sent" to the bank.
    pub fn arm_micro_deposits(&self, acct: &AccountId, amounts: [i64; 2]) {
        self.expected_micro_deposits
            .lock()
            .expect("micro-deposits mutex poisoned")
            .insert(acct.clone(), amounts);
    }

    fn requirements_for(structure: crate::kyb::BusinessStructure) -> Requirements {
        let mut req = Requirements::default();
        // Common to every account.
        req.currently_due.push("business.legal_name".into());
        req.currently_due.push("business.tax_id".into());
        req.currently_due.push("representative.identity".into());
        req.currently_due.push("external_account".into());
        req.currently_due.push("tos_acceptance".into());

        if structure.requires_beneficial_owners() {
            req.currently_due.push("beneficial_owners".into());
        }

        req.eventually_due
            .push("identity.document.verification".into());

        req
    }
}

#[async_trait::async_trait]
impl OnboardingProvider for NativeProvider {
    async fn create_account(&self, profile: &BusinessProfile) -> Result<AccountId> {
        let mut account = ConnectedAccount::new(self.account_type, profile.clone());
        account.requirements = Self::requirements_for(profile.structure);
        let id = account.id.clone();

        info!(acct = %id.0, "creating connected account");

        self.accounts
            .lock()
            .expect("accounts mutex poisoned")
            .insert(id.clone(), account);
        self.flows
            .lock()
            .expect("flows mutex poisoned")
            .insert(id.clone(), OnboardingFlow::new(id.clone()));

        Ok(id)
    }

    async fn submit_step(
        &self,
        acct: &AccountId,
        step: OnboardingStep,
        payload: StepPayload,
    ) -> Result<StepResult> {
        // Step/payload congruence check.
        if payload.step() != step {
            return Err(Error::OnboardingStepInvalid {
                reason: format!(
                    "payload variant ({:?}) does not match step ({step:?})",
                    payload.step()
                ),
            });
        }

        // Mutate the canonical account snapshot.
        let mut accounts = self.accounts.lock().expect("accounts mutex poisoned");
        let account = accounts
            .get_mut(acct)
            .ok_or_else(|| Error::AccountNotFound(acct.0.clone()))?;

        let mut accepted = true;
        let mut message: Option<String> = None;

        match payload {
            StepPayload::BusinessInfo(profile) => {
                account.business = profile;
                account.requirements.currently_due.retain(|r| {
                    r != "business.legal_name" && r != "business.tax_id"
                });
            }
            StepPayload::RepresentativeInfo(owner) => {
                // The representative is the first natural person on the
                // file and is automatically the control-prong individual
                // unless overridden by the BeneficialOwners step.
                let mut rep = owner;
                rep.control = true;
                // If the representative also crosses the ownership prong,
                // count them in the roster; otherwise just record control.
                let existing_rep = account
                    .beneficial_owners
                    .iter_mut()
                    .find(|o| o.person.legal_name == rep.person.legal_name);
                if let Some(slot) = existing_rep {
                    *slot = rep;
                } else {
                    account.beneficial_owners.push(rep);
                }
                account
                    .requirements
                    .currently_due
                    .retain(|r| r != "representative.identity");
            }
            StepPayload::BeneficialOwners(owners) => {
                // Validate against FinCEN CDD only if the structure
                // requires it.
                if account.business.structure.requires_beneficial_owners() {
                    validate_cdd(&owners)?;
                }
                // Merge with the representative if any.
                let mut merged: Vec<BeneficialOwner> = owners;
                // Preserve PEP flags / control flags already set on the rep.
                for existing in &account.beneficial_owners {
                    if !merged
                        .iter()
                        .any(|o| o.person.legal_name == existing.person.legal_name)
                    {
                        merged.push(existing.clone());
                    }
                }
                account.beneficial_owners = merged;
                account
                    .requirements
                    .currently_due
                    .retain(|r| r != "beneficial_owners");
            }
            StepPayload::BankAccount(ext) => {
                self.bank_accounts
                    .lock()
                    .expect("bank mutex poisoned")
                    .insert(acct.clone(), ext);
                account
                    .requirements
                    .currently_due
                    .retain(|r| r != "external_account");
                // External-account verification is now currently-due
                // (it transitions from eventually-due once the account
                // is on file).
                if !account
                    .requirements
                    .currently_due
                    .iter()
                    .any(|r| r == "external_account.verification")
                {
                    account
                        .requirements
                        .currently_due
                        .push("external_account.verification".into());
                }
            }
            StepPayload::TosAcceptance(tos) => {
                self.tos
                    .lock()
                    .expect("tos mutex poisoned")
                    .insert(acct.clone(), tos);
                account
                    .requirements
                    .currently_due
                    .retain(|r| r != "tos_acceptance");
            }
            StepPayload::ExternalAccountVerification { amounts_minor } => {
                let expected = self
                    .expected_micro_deposits
                    .lock()
                    .expect("micro-deposit mutex poisoned")
                    .get(acct)
                    .copied();
                match expected {
                    Some(arr) if arr == amounts_minor => {
                        account
                            .requirements
                            .currently_due
                            .retain(|r| r != "external_account.verification");
                    }
                    Some(_) => {
                        accepted = false;
                        message = Some(
                            "micro-deposit amounts did not match the values sent".into(),
                        );
                    }
                    None => {
                        accepted = false;
                        message = Some(
                            "no pending micro-deposits for this account; submit BankAccount first"
                                .into(),
                        );
                    }
                }
            }
            StepPayload::IdentityVerification { documents } => {
                // Mark every named person as documented; in production
                // this would post the document to an ID-verification
                // provider (Persona, Onfido, Stripe Identity, etc.).
                debug!(acct = %acct.0, docs = documents.len(), "identity verification submitted");
                account
                    .requirements
                    .eventually_due
                    .retain(|r| r != "identity.document.verification");
            }
        }

        account.last_updated = Utc::now();

        // Advance the flow cursor.
        let mut flows = self.flows.lock().expect("flow mutex poisoned");
        if let Some(flow) = flows.get_mut(acct) {
            if accepted {
                if let Some(pos) = flow.steps.iter().position(|s| *s == step) {
                    if pos >= flow.current_step {
                        flow.current_step = pos + 1;
                    }
                }
            }
            flow.status = if !account.requirements.past_due.is_empty()
                || account.requirements.disabled_reason.is_some()
            {
                OnboardingStatus::Disabled
            } else if account.requirements.currently_due.is_empty()
                && account.requirements.pending_verification.is_empty()
            {
                OnboardingStatus::Active
            } else if !account.requirements.pending_verification.is_empty() {
                OnboardingStatus::PendingVerification
            } else {
                OnboardingStatus::InProgress
            };
        }

        Ok(StepResult {
            step,
            requirements: account.requirements.clone(),
            accepted,
            message,
        })
    }

    async fn status(&self, acct: &AccountId) -> Result<OnboardingStatus> {
        let flows = self.flows.lock().expect("flow mutex poisoned");
        flows
            .get(acct)
            .map(|f| f.status)
            .ok_or_else(|| Error::AccountNotFound(acct.0.clone()))
    }
}

// We avoid a hard `async_trait` workspace dep by alias-importing the
// crate name; this attribute is provided by the upstream crate.
pub use async_trait::async_trait;

/// Internal helper: build default settings.
#[must_use]
pub fn default_account_settings() -> AccountSettings {
    AccountSettings::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kyb::{Address, BusinessStructure, CountryCode, Person, TaxId};
    use chrono::NaiveDate;

    fn sample_profile() -> BusinessProfile {
        BusinessProfile {
            legal_name: "Acme Widgets LLC".into(),
            trade_name: None,
            structure: BusinessStructure::SingleMemberLlc,
            tax_id: Some(TaxId::Ein("12-3456789".into())),
            mcc: 5734,
            country: CountryCode("US".into()),
            registered_address: Address {
                line1: "1 Main St".into(),
                line2: None,
                city: "Austin".into(),
                region: "TX".into(),
                postal_code: "78701".into(),
                country: CountryCode("US".into()),
            },
            support_email: Some("support@acme.example".into()),
            support_phone: None,
            website: None,
        }
    }

    #[allow(dead_code)]
    fn sample_owner(name: &str, ownership: f32, control: bool) -> BeneficialOwner {
        BeneficialOwner {
            person: Person {
                legal_name: name.into(),
                dob: NaiveDate::from_ymd_opt(1985, 5, 15).expect("date"),
                address: sample_profile().registered_address,
                ssn_last_4: Some("4321".into()),
                ssn_or_itin_full: None,
                government_id: None,
            },
            ownership_pct: ownership,
            control,
            is_pep: false,
        }
    }

    #[tokio::test]
    async fn step_payload_mismatch_errors() {
        let p = NativeProvider::new(AccountType::Custom);
        let acct = p.create_account(&sample_profile()).await.expect("create");
        let err = p
            .submit_step(
                &acct,
                OnboardingStep::BusinessInfo,
                StepPayload::TosAcceptance(crate::tos::TosAcceptance {
                    ip: "0.0.0.0".into(),
                    user_agent: "x".into(),
                    accepted_at: Utc::now(),
                    version_hash: "h".into(),
                }),
            )
            .await
            .expect_err("mismatch");
        assert!(matches!(err, Error::OnboardingStepInvalid { .. }));
    }
}
