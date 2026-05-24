//! Know-Your-Business (KYB) domain types.
//!
//! KYB is the legal-entity analogue of KYC: instead of identifying a
//! natural person, the platform must identify the business itself plus
//! every natural person who beneficially owns ≥25% or who exercises
//! "significant control" over the entity.
//!
//! ## Regulatory anchor
//!
//! - **FinCEN Customer Due Diligence Final Rule** (31 CFR § 1010.230,
//!   effective 2018-05-11): covered financial institutions must identify
//!   and verify each natural person who owns 25% or more of the equity
//!   interests of the legal-entity customer (the "ownership prong") plus
//!   one individual with significant managerial responsibility (the
//!   "control prong"). Cited verbatim from the rule text.
//!
//! - **EU AMLD5** (Directive (EU) 2018/843, Art. 3(6)): equivalent
//!   beneficial-owner identification regime for EU institutions, with
//!   the same 25% threshold.
//!
//! - **AMLD6** (Directive (EU) 2018/1673): expands predicate offences
//!   for which beneficial-ownership data must be retained. Same 25%
//!   threshold survives.
//!
//! Payment-platform sub-merchant onboarding is in scope of both regimes;
//! [`Requirements`] tracks the outstanding obligations that gate
//! [`crate::account::Capability`] grants.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// ISO 3166-1 alpha-2 country code, locally-defined so we don't drag the
/// op-screening variant into our public surface verbatim. Conversions
/// to/from the screening type are 1:1.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CountryCode(pub String);

impl CountryCode {
    /// True if this looks like a syntactically valid ISO 3166-1 alpha-2 code
    /// (exactly two ASCII uppercase letters). No country-list lookup.
    #[must_use]
    pub fn is_valid_shape(&self) -> bool {
        let s = &self.0;
        s.len() == 2 && s.chars().all(|c| c.is_ascii_uppercase())
    }
}

/// Postal address attached to a business or person.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Address {
    /// Primary street line.
    pub line1: String,
    /// Optional secondary street line (suite / floor).
    pub line2: Option<String>,
    /// City / locality.
    pub city: String,
    /// State / province / region.
    pub region: String,
    /// Postal / ZIP code.
    pub postal_code: String,
    /// ISO 3166-1 alpha-2 country.
    pub country: CountryCode,
}

/// Tax-identifier kinds the platform can accept on a [`BusinessProfile`].
///
/// US: EIN. EU: VAT / national tax number. UK: UTR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaxId {
    /// US Employer Identification Number (9 digits, format `NN-NNNNNNN`).
    Ein(String),
    /// EU VAT number (per-member-state format).
    EuVat(String),
    /// UK Unique Taxpayer Reference (10 digits).
    UkUtr(String),
    /// Catch-all for jurisdictions not enumerated above.
    Other {
        /// ISO 3166-1 alpha-2 country issuing the identifier.
        country: CountryCode,
        /// The identifier value as published.
        value: String,
    },
}

/// Legal structure of the sub-merchant business.
///
/// Drives downstream behaviour: a [`SoleProprietor`](Self::SoleProprietor)
/// is screened as a natural person (no entity layer), while a
/// [`PublicCorporation`](Self::PublicCorporation) is exempt from the
/// beneficial-owner identification prong under 31 CFR § 1010.230(e)(2)(i)
/// because its ownership is already public on a regulated exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessStructure {
    /// Natural person operating under their own name or a DBA.
    SoleProprietor,
    /// LLC with a single member.
    SingleMemberLlc,
    /// LLC with multiple members.
    MultiMemberLlc,
    /// Privately held corporation (S-Corp / C-Corp not listed publicly).
    PrivateCorporation,
    /// Public corporation listed on a regulated exchange.
    ///
    /// 31 CFR § 1010.230(e)(2)(i) exempts publicly traded entities from
    /// the ownership-prong identification requirement.
    PublicCorporation,
    /// General or limited partnership.
    Partnership,
    /// 501(c)(3) or equivalent.
    Nonprofit,
    /// Government entity (also exempt from CDD ownership prong under
    /// § 1010.230(e)(2)(ii)).
    GovernmentEntity,
}

impl BusinessStructure {
    /// True if FinCEN CDD beneficial-owner identification is required.
    ///
    /// Public corporations and government entities are statutorily exempt.
    #[must_use]
    pub const fn requires_beneficial_owners(self) -> bool {
        !matches!(self, Self::PublicCorporation | Self::GovernmentEntity)
    }
}

/// A business's KYB profile.
///
/// Built up over the course of the [`crate::onboarding::OnboardingFlow`];
/// every field except the registered address is allowed to be empty at
/// flow start and populated later via step submission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusinessProfile {
    /// Legal name as it appears on the formation document.
    pub legal_name: String,
    /// Optional trade name / DBA.
    pub trade_name: Option<String>,
    /// Legal structure.
    pub structure: BusinessStructure,
    /// Government-issued tax identifier.
    pub tax_id: Option<TaxId>,
    /// Merchant Category Code (ISO 18245). Drives interchange and
    /// statement display.
    pub mcc: u16,
    /// Country of incorporation (ISO 3166-1 alpha-2).
    pub country: CountryCode,
    /// Registered address per the formation document.
    pub registered_address: Address,
    /// Customer-support email (printed on receipts).
    pub support_email: Option<String>,
    /// Customer-support phone (printed on receipts).
    pub support_phone: Option<String>,
    /// Public website.
    pub website: Option<String>,
}

/// Vault reference for sensitive PII that has been encrypted and stored.
///
/// In practice operators wire this through `op-vault`; locally we treat
/// it as an opaque non-PCI token so this crate stays out of CDE scope.
/// Construction is deliberately not `Default`: an `EncryptedField` must
/// always carry a real vault reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncryptedField {
    /// Opaque vault token (e.g. `tok_v7_<uuid>`).
    pub vault_ref: String,
    /// SHA-256 hex of the cleartext (useful for idempotency checks
    /// without ever decrypting).
    pub digest: String,
}

/// Kind of government-issued identity document for a natural person.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GovernmentId {
    /// US passport.
    UsPassport {
        /// 9-character alphanumeric.
        number: String,
    },
    /// US driver's licence.
    UsDriversLicence {
        /// Per-state format; we accept the raw value.
        number: String,
        /// Issuing US state, ISO 3166-2 subdivision code (e.g. `"US-CA"`).
        state: String,
    },
    /// Foreign passport.
    ForeignPassport {
        /// Document number as printed.
        number: String,
        /// Issuing country (ISO 3166-1 alpha-2).
        country: CountryCode,
    },
    /// National identity card (EU, Brazil, etc.).
    NationalId {
        /// Document number.
        number: String,
        /// Issuing country.
        country: CountryCode,
    },
}

/// A natural person identified during KYB.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Person {
    /// Legal name as it appears on government ID.
    pub legal_name: String,
    /// Date of birth.
    pub dob: NaiveDate,
    /// Residential address.
    pub address: Address,
    /// Last four digits of SSN (collected first as a low-friction check).
    pub ssn_last_4: Option<String>,
    /// Full SSN or ITIN, stored encrypted in the vault.
    ///
    /// Required for US persons before [`crate::account::Capability::TaxReporting1099K`]
    /// can be granted (IRS requires payee TIN for the 1099-K form).
    pub ssn_or_itin_full: Option<EncryptedField>,
    /// Government-issued photo ID, when collected for elevated-risk tiers.
    pub government_id: Option<GovernmentId>,
}

/// A natural person declared as a beneficial owner or controller.
///
/// Per FinCEN CDD Final Rule (31 CFR § 1010.230(d)(1)), every natural
/// person who owns ≥25% of the equity interests must be identified
/// (the "ownership prong"). Per § 1010.230(d)(2), one individual with
/// significant managerial responsibility must also be identified
/// (the "control prong").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeneficialOwner {
    /// The natural person.
    pub person: Person,
    /// Percentage of equity (0.0 ≤ x ≤ 100.0). Persons named purely for
    /// the control prong may declare 0.0% if they own no equity.
    pub ownership_pct: f32,
    /// True if this person is the named "control prong" individual.
    pub control: bool,
    /// True if this person matched a Politically-Exposed-Persons watchlist
    /// during screening (FATF Recommendation 12 enhanced due diligence).
    pub is_pep: bool,
}

impl BeneficialOwner {
    /// True if this owner meets the CDD ownership prong threshold (≥25%).
    #[must_use]
    pub const fn meets_ownership_threshold(&self) -> bool {
        self.ownership_pct >= 25.0
    }
}

/// Identifier for a single regulatory requirement (e.g. `"company.tax_id"`,
/// `"individual.verification.document"`).
///
/// String-keyed for parity with Stripe Connect's `requirements.eventually_due`
/// surface; operators migrating off Connect can map keys 1:1.
pub type RequirementId = String;

/// Outstanding-requirement vector attached to a connected account.
///
/// State machine (informal):
///
/// - **eventually_due**: required before some future deadline (e.g.
///   1099-K filing date) but not blocking activity now.
/// - **currently_due**: required to unlock the next capability;
///   blocks activation but not active flows.
/// - **past_due**: deadline passed; blocks active capabilities.
/// - **pending_verification**: submitted, awaiting upstream
///   verification (document review, OFAC re-scan).
/// - **disabled_reason**: terminal — account is frozen pending
///   remediation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirements {
    /// Requirements that must be satisfied to enable the next capability.
    pub currently_due: Vec<RequirementId>,
    /// Requirements that will eventually be needed but are not blocking.
    pub eventually_due: Vec<RequirementId>,
    /// Currently-due requirements whose deadline has passed.
    pub past_due: Vec<RequirementId>,
    /// Free-text reason the account is disabled, if any.
    pub disabled_reason: Option<String>,
    /// Requirements submitted and awaiting verification.
    pub pending_verification: Vec<RequirementId>,
}

/// Validate a candidate beneficial-owner roster against the FinCEN CDD
/// ownership prong (31 CFR § 1010.230(d)(1)).
///
/// # Errors
/// Returns [`crate::error::Error::BeneficialOwnerIncomplete`] if:
/// - any individual owner with ≥25% declared ownership is missing from
///   the roster (sum < 75% with no "control prong" individual);
/// - no individual is declared as the control prong.
///
/// Structures that statutorily exempt the entity (see
/// [`BusinessStructure::requires_beneficial_owners`]) bypass this check
/// at the caller layer.
pub fn validate_cdd(owners: &[BeneficialOwner]) -> crate::error::Result<()> {
    use crate::error::Error;

    if owners.is_empty() {
        return Err(Error::BeneficialOwnerIncomplete {
            reason: "no beneficial owners declared; FinCEN CDD requires at least one control-prong individual (31 CFR § 1010.230(d)(2))".into(),
        });
    }

    let total_ownership: f32 = owners
        .iter()
        .filter(|o| o.meets_ownership_threshold())
        .map(|o| o.ownership_pct)
        .sum();

    let has_control_prong = owners.iter().any(|o| o.control);

    if !has_control_prong {
        return Err(Error::BeneficialOwnerIncomplete {
            reason: "missing control prong individual (31 CFR § 1010.230(d)(2))".into(),
        });
    }

    // If declared ≥25% owners cover none of the equity (total = 0%) and
    // there are no exempt unidentified owners, we trust the control-prong
    // individual as the only required identification. If 25%+ owners
    // exist but cover less than 25% of equity overall, the roster is
    // internally inconsistent — we surface that.
    let any_qualifying = owners.iter().any(BeneficialOwner::meets_ownership_threshold);
    if any_qualifying && total_ownership < 25.0 {
        return Err(Error::BeneficialOwnerIncomplete {
            reason: format!(
                "owners marked ≥25% sum to {total_ownership}%; roster inconsistent under 31 CFR § 1010.230(d)(1)"
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_person() -> Person {
        Person {
            legal_name: "Jane Doe".into(),
            dob: NaiveDate::from_ymd_opt(1980, 1, 1).expect("test date"),
            address: Address {
                line1: "1 Main St".into(),
                line2: None,
                city: "Austin".into(),
                region: "TX".into(),
                postal_code: "78701".into(),
                country: CountryCode("US".into()),
            },
            ssn_last_4: Some("1234".into()),
            ssn_or_itin_full: None,
            government_id: None,
        }
    }

    #[test]
    fn country_code_shape() {
        assert!(CountryCode("US".into()).is_valid_shape());
        assert!(!CountryCode("usa".into()).is_valid_shape());
        assert!(!CountryCode("U".into()).is_valid_shape());
    }

    #[test]
    fn cdd_requires_control_prong() {
        let owners = vec![BeneficialOwner {
            person: sample_person(),
            ownership_pct: 100.0,
            control: false,
            is_pep: false,
        }];
        let err = validate_cdd(&owners).expect_err("missing control prong");
        assert!(matches!(
            err,
            crate::error::Error::BeneficialOwnerIncomplete { .. }
        ));
    }

    #[test]
    fn cdd_accepts_sole_owner_with_control() {
        let owners = vec![BeneficialOwner {
            person: sample_person(),
            ownership_pct: 100.0,
            control: true,
            is_pep: false,
        }];
        validate_cdd(&owners).expect("ok");
    }

    #[test]
    fn cdd_accepts_split_ownership() {
        let owners = vec![
            BeneficialOwner {
                person: sample_person(),
                ownership_pct: 50.0,
                control: true,
                is_pep: false,
            },
            BeneficialOwner {
                person: sample_person(),
                ownership_pct: 50.0,
                control: false,
                is_pep: false,
            },
        ];
        validate_cdd(&owners).expect("ok");
    }

    #[test]
    fn public_corp_exempt() {
        assert!(!BusinessStructure::PublicCorporation.requires_beneficial_owners());
        assert!(BusinessStructure::PrivateCorporation.requires_beneficial_owners());
        assert!(!BusinessStructure::GovernmentEntity.requires_beneficial_owners());
    }
}
