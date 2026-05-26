//! The Reasoner's Epistemic Contract (spec §0.6, §3.8).
//!
//! Every claim labels its epistemic mode: `from_priors` for structural
//! reasoning drawn from the reasoner's training data, or
//! `from_retrieval` for present-tense factual content sourced at query
//! time. The verification layer rejects `from_priors` claims that
//! should have been retrieved (invariant I13).

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::knowledge_axes::KnowledgeAxes;

/// Two modes only. Mixing without per-claim labeling is forbidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicMode {
    /// The claim comes from the reasoner's training data. Confidence
    /// is about what was true at training cutoff. Permitted for
    /// structural reasoning, pattern recognition, generative
    /// composition, exploration. Forbidden for any claim where
    /// `KnowledgeAxes::can_live_in_weights()` returns false.
    FromPriors,
    /// The claim comes from a retrieval source. Required for any
    /// present-tense factual claim, any time-varying property, any
    /// decision-guiding recommendation.
    FromRetrieval,
}

/// Sub-class of a `from_priors` claim. Required when
/// `EpistemicMode::FromPriors`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorClaimClass {
    Structural,
    Exploratory,
    Pattern,
    Definitional,
}

/// Per-claim epistemic mode declaration. Required on every claim in
/// every Answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimAttribution {
    pub schema_version: String,
    pub epistemic_mode: EpistemicMode,

    /// Required when `epistemic_mode == FromPriors` — the reasoner's
    /// training cutoff date.
    #[serde(default)]
    pub prior_reference_cutoff: Option<NaiveDate>,
    /// Required when `epistemic_mode == FromPriors`.
    #[serde(default)]
    pub prior_claim_class: Option<PriorClaimClass>,

    /// Required when `epistemic_mode == FromRetrieval`.
    #[serde(default)]
    pub retrieval_timestamp: Option<DateTime<Utc>>,
    /// Required when `epistemic_mode == FromRetrieval`.
    #[serde(default)]
    pub retrieval_source_uri: Option<String>,
    /// Required when `epistemic_mode == FromRetrieval`.
    #[serde(default)]
    pub retrieval_axes: Option<KnowledgeAxes>,
}

impl ClaimAttribution {
    /// Internal consistency check: every required field for the
    /// declared mode is populated.
    pub fn is_valid(&self) -> Result<(), String> {
        match self.epistemic_mode {
            EpistemicMode::FromPriors => {
                if self.prior_reference_cutoff.is_none() {
                    return Err("from_priors claim missing prior_reference_cutoff".into());
                }
                if self.prior_claim_class.is_none() {
                    return Err("from_priors claim missing prior_claim_class".into());
                }
            }
            EpistemicMode::FromRetrieval => {
                if self.retrieval_timestamp.is_none() {
                    return Err("from_retrieval claim missing retrieval_timestamp".into());
                }
                if self.retrieval_axes.is_none() {
                    return Err("from_retrieval claim missing retrieval_axes".into());
                }
            }
        }
        Ok(())
    }

    /// Enforces Rules 1-3 of the Reasoner's Epistemic Contract: a
    /// `from_priors` claim is permitted only if its axes are
    /// weight-safe AND the claim's valid-time start is not more recent
    /// than the reasoner's training cutoff.
    pub fn is_compatible_with_axes(&self, axes: &KnowledgeAxes) -> Result<(), String> {
        match self.epistemic_mode {
            EpistemicMode::FromPriors => {
                if !axes.can_live_in_weights() {
                    return Err(
                        "from_priors not permitted: claim fails can_live_in_weights()".into(),
                    );
                }
                if let (Some(vts), Some(cutoff)) =
                    (axes.valid_time_start, self.prior_reference_cutoff)
                {
                    if vts.date_naive() > cutoff {
                        return Err(
                            "from_priors not permitted: valid_time more recent than cutoff"
                                .into(),
                        );
                    }
                }
            }
            EpistemicMode::FromRetrieval => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_axes::{GranularityClass, ScopeClass, TemporalStabilityClass};

    fn weight_safe_axes() -> KnowledgeAxes {
        KnowledgeAxes {
            schema_version: "5.0".into(),
            valid_time_start: None,
            valid_time_end: None,
            transaction_time: None,
            reference_time: Utc::now(),
            temporal_stability: TemporalStabilityClass::Invariant,
            granularity: GranularityClass::Coarse,
            granularity_notes: None,
            scope: ScopeClass::Universal,
            scope_domain: None,
            certainty: 1.0,
            certainty_basis: "test".into(),
            source_uri: None,
            source_authority_tier: 1,
            extraction_method: None,
            citation_chain: vec![],
            metadata: Default::default(),
        }
    }

    fn time_varying_axes() -> KnowledgeAxes {
        let mut a = weight_safe_axes();
        a.temporal_stability = TemporalStabilityClass::Fast;
        a.scope = ScopeClass::Particular;
        a
    }

    #[test]
    fn from_priors_requires_cutoff_and_class() {
        let a = ClaimAttribution {
            schema_version: "6.0".into(),
            epistemic_mode: EpistemicMode::FromPriors,
            prior_reference_cutoff: None,
            prior_claim_class: None,
            retrieval_timestamp: None,
            retrieval_source_uri: None,
            retrieval_axes: None,
        };
        assert!(a.is_valid().is_err());
    }

    #[test]
    fn from_retrieval_requires_timestamp_and_axes() {
        let a = ClaimAttribution {
            schema_version: "6.0".into(),
            epistemic_mode: EpistemicMode::FromRetrieval,
            prior_reference_cutoff: None,
            prior_claim_class: None,
            retrieval_timestamp: None,
            retrieval_source_uri: None,
            retrieval_axes: None,
        };
        assert!(a.is_valid().is_err());
    }

    #[test]
    fn from_priors_rejects_time_varying_claim() {
        let attr = ClaimAttribution {
            schema_version: "6.0".into(),
            epistemic_mode: EpistemicMode::FromPriors,
            prior_reference_cutoff: Some(NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()),
            prior_claim_class: Some(PriorClaimClass::Structural),
            retrieval_timestamp: None,
            retrieval_source_uri: None,
            retrieval_axes: None,
        };
        let err = attr.is_compatible_with_axes(&time_varying_axes()).unwrap_err();
        assert!(err.contains("can_live_in_weights"), "got: {err}");
    }

    #[test]
    fn from_priors_accepts_weight_safe_claim() {
        let attr = ClaimAttribution {
            schema_version: "6.0".into(),
            epistemic_mode: EpistemicMode::FromPriors,
            prior_reference_cutoff: Some(NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()),
            prior_claim_class: Some(PriorClaimClass::Structural),
            retrieval_timestamp: None,
            retrieval_source_uri: None,
            retrieval_axes: None,
        };
        attr.is_compatible_with_axes(&weight_safe_axes()).unwrap();
    }

    #[test]
    fn roundtrips_through_json() {
        let a = ClaimAttribution {
            schema_version: "6.0".into(),
            epistemic_mode: EpistemicMode::FromRetrieval,
            prior_reference_cutoff: None,
            prior_claim_class: None,
            retrieval_timestamp: Some(Utc::now()),
            retrieval_source_uri: Some("https://example".into()),
            retrieval_axes: Some(weight_safe_axes()),
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: ClaimAttribution = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }
}
