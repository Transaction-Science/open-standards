//! Atomic claims (spec §3.4 — deferred to v1, §6.3 atomization).
//!
//! Atomization decomposes a draft answer into atomic claims with
//! stakes determination (§6.3). Verification runs focused entailment
//! against the at-risk subset (§6.2). This module models the claim
//! object — text, axes, stakes — independent of the atomizer
//! implementation.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::Metadata;
use crate::knowledge_axes::KnowledgeAxes;

/// What the claim costs the user if it's wrong. Drives the "focused
/// entailment on at-risk claims" routing in §6.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaimStakes {
    /// Stylistic / interpretive content; wrong answer is recoverable.
    Low,
    /// Background facts that flavor the answer.
    Medium,
    /// Decision-guiding factual content; user may act on it.
    High,
    /// Safety- or compliance-critical.
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomicClaim {
    pub schema_version: String,
    pub claim_id: Uuid,
    /// The claim sentence in canonical form.
    pub text: String,
    /// Which segment of the draft this claim was extracted from.
    pub segment_id: String,
    pub stakes: ClaimStakes,
    /// The seven-axis shape (§3.8) that the verification layer uses to
    /// check freshness / scope / granularity consistency.
    pub knowledge_axes: KnowledgeAxes,
    /// Free-text rationale from the atomizer for how the claim was
    /// extracted; aids debugging without affecting verification.
    #[serde(default)]
    pub atomization_notes: Option<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_axes::{GranularityClass, ScopeClass, TemporalStabilityClass};
    use chrono::Utc;

    #[test]
    fn roundtrips_through_json() {
        let c = AtomicClaim {
            schema_version: "2.0".into(),
            claim_id: Uuid::new_v4(),
            text: "Paris is the capital of France.".into(),
            segment_id: "s0".into(),
            stakes: ClaimStakes::Medium,
            knowledge_axes: KnowledgeAxes {
                schema_version: "5.0".into(),
                valid_time_start: None,
                valid_time_end: None,
                transaction_time: None,
                reference_time: Utc::now(),
                temporal_stability: TemporalStabilityClass::Slow,
                granularity: GranularityClass::Coarse,
                granularity_notes: None,
                scope: ScopeClass::Particular,
                scope_domain: Some("France".into()),
                certainty: 1.0,
                certainty_basis: "wikidata".into(),
                source_uri: Some("wikidata:Q90".into()),
                source_authority_tier: 1,
                extraction_method: Some("structured_api".into()),
                citation_chain: vec![],
                metadata: Default::default(),
            },
            atomization_notes: None,
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: AtomicClaim = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
