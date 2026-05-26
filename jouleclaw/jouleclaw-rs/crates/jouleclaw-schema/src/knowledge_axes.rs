//! The seven-axis shape of every knowledge claim (spec §0.5, §3.8).
//!
//! Knowledge representation literature decomposes any claim into seven
//! engineering-relevant axes: valid time, transaction time, reference
//! time, granularity, scope, certainty, provenance. Every retrieved
//! item, atomic claim, entailment result, and answer in this
//! architecture carries a [`KnowledgeAxes`] value. The planner uses
//! `can_live_in_weights()` to decide whether a query may answer from
//! priors or must retrieve.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::common::Metadata;

/// Rate at which the claim's truth value changes (axes 1-3 collapsed
/// into a class, plus the freshness-budget table in
/// [`KnowledgeAxes::freshness_budget_seconds`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TemporalStabilityClass {
    /// Mathematical / physical constants. Never stale.
    Invariant,
    /// Decades — scientific consensus.
    Slow,
    /// Years — laws, policies, taxonomies.
    Medium,
    /// Months — technology landscape, leadership.
    Fast,
    /// Days or hours — markets, news, weather.
    VeryFast,
}

/// Axis 5 — domain over which the claim applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopeClass {
    /// Holds for all instances in domain.
    Universal,
    /// Holds with stated exceptions.
    General,
    /// Holds for specific named instances.
    Particular,
    /// Holds in a bounded context only.
    Local,
}

/// Axis 4 — resolution at which the claim is expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GranularityClass {
    /// High-level, summary, approximate.
    Coarse,
    /// Standard reporting granularity.
    Medium,
    /// Specific, detailed, exact.
    Fine,
    /// Measurement-grade, all qualifiers.
    Precise,
}

/// The seven-axis description of a knowledge claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeAxes {
    pub schema_version: String,

    // Axes 1-3: Time
    pub valid_time_start: Option<DateTime<Utc>>,
    pub valid_time_end: Option<DateTime<Utc>>,
    pub transaction_time: Option<DateTime<Utc>>,
    pub reference_time: DateTime<Utc>,
    pub temporal_stability: TemporalStabilityClass,

    // Axis 4: Granularity
    pub granularity: GranularityClass,
    #[serde(default)]
    pub granularity_notes: Option<String>,

    // Axis 5: Scope
    pub scope: ScopeClass,
    #[serde(default)]
    pub scope_domain: Option<String>,

    // Axis 6: Certainty
    pub certainty: f64,
    pub certainty_basis: String,

    // Axis 7: Provenance
    #[serde(default)]
    pub source_uri: Option<String>,
    pub source_authority_tier: u8,
    #[serde(default)]
    pub extraction_method: Option<String>,
    #[serde(default)]
    pub citation_chain: Vec<String>,

    #[serde(default)]
    pub metadata: Metadata,
}

impl KnowledgeAxes {
    /// True iff the claim is axis-collapsed enough to safely live in
    /// model weights without falsification over time. Drives the
    /// Reasoner's Epistemic Contract (spec §0.6 Rule 2).
    pub fn can_live_in_weights(&self) -> bool {
        matches!(self.temporal_stability, TemporalStabilityClass::Invariant)
            && matches!(self.scope, ScopeClass::Universal)
            && self.certainty >= 0.99
            && self.source_authority_tier <= 2
    }

    pub fn must_be_retrieved(&self) -> bool {
        !self.can_live_in_weights()
    }

    /// Maximum staleness allowed by temporal stability class, in
    /// seconds. `None` means "never stale" (Invariant claims).
    pub fn freshness_budget_seconds(&self) -> Option<u64> {
        match self.temporal_stability {
            TemporalStabilityClass::Invariant => None,
            TemporalStabilityClass::Slow => Some(60 * 60 * 24 * 365 * 5),
            TemporalStabilityClass::Medium => Some(60 * 60 * 24 * 30),
            TemporalStabilityClass::Fast => Some(60 * 60 * 24 * 7),
            TemporalStabilityClass::VeryFast => Some(60 * 60),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes_template() -> KnowledgeAxes {
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
            certainty: 0.99,
            certainty_basis: "test".into(),
            source_uri: None,
            source_authority_tier: 1,
            extraction_method: None,
            citation_chain: vec![],
            metadata: Default::default(),
        }
    }

    #[test]
    fn invariant_universal_high_certainty_can_live_in_weights() {
        let a = axes_template();
        assert!(a.can_live_in_weights());
    }

    #[test]
    fn fast_changing_claim_must_be_retrieved() {
        let mut a = axes_template();
        a.temporal_stability = TemporalStabilityClass::Fast;
        assert!(a.must_be_retrieved());
        assert!(!a.can_live_in_weights());
    }

    #[test]
    fn particular_scope_blocks_weights() {
        let mut a = axes_template();
        a.scope = ScopeClass::Particular;
        assert!(a.must_be_retrieved());
    }

    #[test]
    fn low_certainty_blocks_weights() {
        let mut a = axes_template();
        a.certainty = 0.5;
        assert!(a.must_be_retrieved());
    }

    #[test]
    fn freshness_budget_table() {
        let mut a = axes_template();
        assert_eq!(a.freshness_budget_seconds(), None);
        a.temporal_stability = TemporalStabilityClass::Slow;
        assert_eq!(a.freshness_budget_seconds(), Some(60 * 60 * 24 * 365 * 5));
        a.temporal_stability = TemporalStabilityClass::VeryFast;
        assert_eq!(a.freshness_budget_seconds(), Some(60 * 60));
    }

    #[test]
    fn roundtrips_through_json() {
        let a = axes_template();
        let json = serde_json::to_string(&a).unwrap();
        let back: KnowledgeAxes = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }
}
