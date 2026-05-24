//! KYB-time sanctions + PEP + adverse-media screening.
//!
//! At each [`OnboardingStep::BeneficialOwners`](crate::onboarding::OnboardingStep::BeneficialOwners)
//! submission the platform must:
//!
//! 1. Screen every natural person (representative + owners) against the
//!    sanctions lists in [`op-screening`].
//! 2. Screen the business legal name + trade name against the same lists.
//! 3. Screen every person against a **PEP** watchlist (Politically
//!    Exposed Persons). PEP lists are distinct from sanctions lists —
//!    being on a PEP list is not illegal but mandates enhanced due
//!    diligence per **FATF Recommendation 12** and EU AMLD5 Art. 20.
//!    Commercial PEP datasets are published by Refinitiv World-Check,
//!    Dow Jones Risk Center, and LexisNexis WorldCompliance; this
//!    crate ships the index-shape and a small fixture list, leaving
//!    operators to subscribe to the upstream feed of their choice.
//! 4. Optionally screen against an **adverse-media** index (negative
//!    news mentions); same shape as the PEP index.
//!
//! The combined [`ScreeningResult`] gates the [`crate::onboarding::StepResult`].

use std::collections::BTreeSet;

use op_screening::{
    Address as ScreeningAddress, MatchScore, SanctionsIndex, ScreenDecision, ScreenRequest,
    Screener,
};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::kyb::{Address, BeneficialOwner, BusinessProfile};

/// Top-level screening decision for a connected account.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    /// No hits anywhere; safe to proceed.
    Clear,
    /// One or more sanctions hits; block the account.
    Block,
    /// PEP or adverse-media hits, or sanctions ambiguity; route to manual review.
    Review,
}

/// Combined screening outcome for one owner-roster submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningResult {
    /// Sanctions matches across all subjects (persons + business).
    pub sanctions_hits: Vec<MatchScore>,
    /// PEP matches across all persons.
    pub pep_hits: Vec<MatchScore>,
    /// Adverse-media matches across all persons + business.
    pub adverse_media_hits: Vec<MatchScore>,
    /// Top-line decision.
    pub decision: Decision,
}

/// Convert our local [`Address`] into the screening-crate [`ScreeningAddress`].
fn to_screening_address(addr: &Address) -> ScreeningAddress {
    ScreeningAddress {
        street: Some(if let Some(l2) = addr.line2.as_deref() {
            format!("{}\n{l2}", addr.line1)
        } else {
            addr.line1.clone()
        }),
        city: Some(addr.city.clone()),
        region: Some(addr.region.clone()),
        postal_code: Some(addr.postal_code.clone()),
        country: Some(op_screening::CountryCode(addr.country.0.clone())),
    }
}

/// Coordinator that runs all three screens for a given roster + business.
///
/// `pep_screener` and `adverse_media_screener` are independent
/// [`Screener`] instances loaded with their own list fixtures. They're
/// optional so operators who only have a sanctions feed can still use
/// this crate (PEP / AM are then no-ops returning empty hit-vectors).
pub struct ConnectScreener {
    sanctions: Screener,
    pep: Option<Screener>,
    adverse_media: Option<Screener>,
}

impl ConnectScreener {
    /// Build a fresh screener.
    #[must_use]
    pub const fn new(
        sanctions: Screener,
        pep: Option<Screener>,
        adverse_media: Option<Screener>,
    ) -> Self {
        Self {
            sanctions,
            pep,
            adverse_media,
        }
    }

    /// Screen a roster + business in one shot.
    ///
    /// # Errors
    /// Any underlying `op-screening` failure bubbles up via [`crate::Error`].
    pub async fn screen(
        &self,
        business: &BusinessProfile,
        owners: &[BeneficialOwner],
    ) -> Result<ScreeningResult> {
        let mut sanctions_hits: Vec<MatchScore> = Vec::new();
        let mut pep_hits: Vec<MatchScore> = Vec::new();
        let mut adverse_media_hits: Vec<MatchScore> = Vec::new();
        let mut any_review = false;
        let mut any_block = false;

        // ---- Business ----
        let business_req = ScreenRequest {
            name: business.legal_name.clone(),
            dob: None,
            address: Some(to_screening_address(&business.registered_address)),
            additional_ids: vec![],
        };
        let s = self.sanctions.screen(&business_req).await?;
        match s.decision {
            ScreenDecision::Clear => {}
            ScreenDecision::Hit => any_block = true,
            ScreenDecision::AmbiguousNeedsReview => any_review = true,
        }
        sanctions_hits.extend(s.hits);

        if let Some(trade) = business.trade_name.as_deref() {
            let trade_req = ScreenRequest {
                name: trade.to_string(),
                dob: None,
                address: None,
                additional_ids: vec![],
            };
            let s = self.sanctions.screen(&trade_req).await?;
            match s.decision {
                ScreenDecision::Clear => {}
                ScreenDecision::Hit => any_block = true,
                ScreenDecision::AmbiguousNeedsReview => any_review = true,
            }
            sanctions_hits.extend(s.hits);
        }

        if let Some(am) = self.adverse_media.as_ref() {
            let am_res = am.screen(&business_req).await?;
            if matches!(am_res.decision, ScreenDecision::Hit | ScreenDecision::AmbiguousNeedsReview)
            {
                any_review = true;
            }
            adverse_media_hits.extend(am_res.hits);
        }

        // ---- Persons ----
        // Dedupe by name to avoid double-screening the rep when also in roster.
        let mut seen_names: BTreeSet<String> = BTreeSet::new();
        for owner in owners {
            let nm = owner.person.legal_name.clone();
            if !seen_names.insert(nm.clone()) {
                continue;
            }
            let req = ScreenRequest {
                name: nm,
                dob: Some(owner.person.dob),
                address: Some(to_screening_address(&owner.person.address)),
                additional_ids: vec![],
            };
            let s = self.sanctions.screen(&req).await?;
            match s.decision {
                ScreenDecision::Clear => {}
                ScreenDecision::Hit => any_block = true,
                ScreenDecision::AmbiguousNeedsReview => any_review = true,
            }
            sanctions_hits.extend(s.hits);

            if let Some(pep) = self.pep.as_ref() {
                let p = pep.screen(&req).await?;
                if matches!(
                    p.decision,
                    ScreenDecision::Hit | ScreenDecision::AmbiguousNeedsReview
                ) {
                    // PEP is enhanced-due-diligence, not block-on-sight.
                    any_review = true;
                }
                pep_hits.extend(p.hits);
            }
            if let Some(am) = self.adverse_media.as_ref() {
                let a = am.screen(&req).await?;
                if matches!(
                    a.decision,
                    ScreenDecision::Hit | ScreenDecision::AmbiguousNeedsReview
                ) {
                    any_review = true;
                }
                adverse_media_hits.extend(a.hits);
            }
        }

        let decision = if any_block {
            Decision::Block
        } else if any_review {
            Decision::Review
        } else {
            Decision::Clear
        };

        Ok(ScreeningResult {
            sanctions_hits,
            pep_hits,
            adverse_media_hits,
            decision,
        })
    }
}

/// Annotate a roster of owners with PEP flags by re-running the PEP
/// screen and toggling [`BeneficialOwner::is_pep`].
///
/// # Errors
/// Bubbles screening failures.
pub async fn annotate_pep_flags(
    pep: &Screener,
    owners: &mut [BeneficialOwner],
) -> Result<()> {
    for owner in owners.iter_mut() {
        let req = ScreenRequest {
            name: owner.person.legal_name.clone(),
            dob: Some(owner.person.dob),
            address: Some(to_screening_address(&owner.person.address)),
            additional_ids: vec![],
        };
        let res = pep.screen(&req).await?;
        owner.is_pep = !res.hits.is_empty();
    }
    Ok(())
}

/// Helper: build an in-memory PEP index from a fixture list of names.
#[must_use]
pub fn build_pep_index_from_names(names: &[&str]) -> SanctionsIndex {
    use chrono::Utc;
    use op_screening::{EntityType, SanctionedEntity, SanctionsList};

    let entities = names
        .iter()
        .enumerate()
        .map(|(i, n)| SanctionedEntity {
            id: format!("pep-{i}"),
            name: (*n).to_string(),
            name_aliases: vec![],
            entity_type: EntityType::Individual,
            dob: None,
            place_of_birth: None,
            addresses: vec![],
            nationalities: vec![],
            identifications: vec![],
            // PEP-list "programs" are typically office titles; we tag as
            // "PEP" so callers can distinguish from sanctions programs.
            programs: vec!["PEP".to_string()],
            last_updated: Utc::now(),
            // PEP lists aren't issued by treasury authorities, so we
            // reuse a sanctions-list discriminant slot for index purposes
            // only; the operator-facing surface should not surface this.
            source_list: SanctionsList::OfacConsolidated,
        })
        .collect();
    SanctionsIndex::build(entities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use op_screening::{AuditLog, ScreenerConfig};

    use crate::kyb::{Address, BusinessStructure, CountryCode, Person, TaxId};

    fn empty_index() -> SanctionsIndex {
        SanctionsIndex::build(vec![])
    }

    fn sample_business() -> BusinessProfile {
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
            support_email: None,
            support_phone: None,
            website: None,
        }
    }

    fn person(name: &str) -> Person {
        Person {
            legal_name: name.into(),
            dob: NaiveDate::from_ymd_opt(1980, 1, 1).expect("date"),
            address: sample_business().registered_address,
            ssn_last_4: None,
            ssn_or_itin_full: None,
            government_id: None,
        }
    }

    #[tokio::test]
    async fn pep_flagging_fires_on_fixture_list() {
        let pep_idx = build_pep_index_from_names(&["Hugo Chavez", "Vladimir Putin"]);
        let pep_screener = Screener::new(pep_idx, ScreenerConfig::default(), AuditLog::new());

        let mut owners = vec![BeneficialOwner {
            person: person("Hugo Chavez"),
            ownership_pct: 100.0,
            control: true,
            is_pep: false,
        }];

        annotate_pep_flags(&pep_screener, &mut owners)
            .await
            .expect("ok");
        assert!(owners[0].is_pep, "PEP flag should have fired");
    }

    #[tokio::test]
    async fn clear_when_lists_empty() {
        let sanctions = Screener::new(empty_index(), ScreenerConfig::default(), AuditLog::new());
        let cs = ConnectScreener::new(sanctions, None, None);
        let res = cs
            .screen(&sample_business(), &[BeneficialOwner {
                person: person("Jane Doe"),
                ownership_pct: 100.0,
                control: true,
                is_pep: false,
            }])
            .await
            .expect("ok");
        assert_eq!(res.decision, Decision::Clear);
    }
}
