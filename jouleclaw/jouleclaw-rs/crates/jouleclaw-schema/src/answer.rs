//! Answers and refusals (spec §3.4 deferred to v1, §7.1–§7.2).
//!
//! The Compose layer emits either an [`Answer`] (verified, with full
//! [`Provenance`]) or a [`Refusal`] (structured, never an exception
//! per invariant I10). Every answer segment is paired with a
//! [`ClaimAttribution`] so invariant I13 (epistemic mode declared)
//! can be enforced.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::atomic_claim::AtomicClaim;
use crate::common::Metadata;
use crate::entailment::EntailmentResult;
use crate::epistemic::ClaimAttribution;
use crate::knowledge_axes::KnowledgeAxes;
use crate::retrieved_item::RetrievedItem;

/// The shape of an emitted answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerStatus {
    /// All required invariants verified; answer is verified.
    Verified,
    /// Some non-critical violations remain; answer carries caveats.
    Degraded,
    /// Returned a current-best via anytime interruption; not final.
    Partial,
}

/// One segment of an answer — a span of text with its attribution,
/// axes, and the claim it derives from. The verified composer
/// assembles these into the final answer body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerSegment {
    pub segment_id: String,
    /// The actual text shown to the user.
    pub text: String,
    /// Epistemic mode declaration (I13).
    pub attribution: ClaimAttribution,
    /// The seven-axis shape of this segment's claim.
    pub knowledge_axes: KnowledgeAxes,
    /// The atomic claim id this segment was composed from. `None`
    /// for purely structural connective text.
    #[serde(default)]
    pub claim_id: Option<Uuid>,
    /// Source item ids cited by this segment. Used to build inline
    /// citations and for provenance-as-cache (§7.3).
    #[serde(default)]
    pub cited_item_ids: Vec<Uuid>,
    /// Free-text rationale (e.g. "labeled as inference because no
    /// premise directly entails"). Surfaces in `Degraded` answers.
    #[serde(default)]
    pub rationale: Option<String>,
}

/// Full provenance package. v3 §7.3 treats this as a
/// content-addressable cache key. Carrying retrieved items and
/// entailment results lets a later overlapping query reuse the work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub schema_version: String,
    /// The plan that produced this answer.
    pub plan_id: Uuid,
    /// Every retrieved item the composer used.
    pub items: Vec<RetrievedItem>,
    /// Every atomic claim the verifier judged.
    pub claims: Vec<AtomicClaim>,
    /// Every entailment result consulted (focused subset).
    pub entailments: Vec<EntailmentResult>,
    /// Cache key produced from the sub-queries (§7.3).
    pub cache_key: String,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    pub schema_version: String,
    pub answer_id: Uuid,
    pub plan_id: Uuid,
    pub status: AnswerStatus,
    pub segments: Vec<AnswerSegment>,
    pub provenance: Provenance,
    /// Invariant ids that were checked and passed before emission.
    pub invariants_verified: Vec<String>,
    /// Total joules spent across plan + execute + diagnose + compose.
    pub joules_spent_total: f64,
    /// Total wall-clock latency.
    pub latency_ms: u64,
    /// User-visible caveats for `Degraded` / `Partial` answers.
    #[serde(default)]
    pub caveats: Vec<String>,
    pub emitted_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Spec §6.6 / invariant I10: refusals are first-class structured
/// objects, not exceptions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Refusal {
    pub schema_version: String,
    pub refusal_id: Uuid,
    pub plan_id: Uuid,
    /// Stable identifier for downstream code (e.g.
    /// `"authority_tier_unsatisfiable"`, `"freshness_unavailable"`).
    pub reason_code: String,
    /// Human-readable explanation.
    pub reason_message: String,
    /// The violations that drove the refusal.
    pub blocking_violations: Vec<String>,
    pub emitted_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::EpistemicMode;
    use crate::knowledge_axes::{GranularityClass, ScopeClass, TemporalStabilityClass};

    fn axes() -> KnowledgeAxes {
        KnowledgeAxes {
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
        }
    }

    #[test]
    fn empty_answer_roundtrips() {
        let a = Answer {
            schema_version: "2.0".into(),
            answer_id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            status: AnswerStatus::Verified,
            segments: vec![AnswerSegment {
                segment_id: "s0".into(),
                text: "Paris.".into(),
                attribution: ClaimAttribution {
                    schema_version: "6.0".into(),
                    epistemic_mode: EpistemicMode::FromRetrieval,
                    prior_reference_cutoff: None,
                    prior_claim_class: None,
                    retrieval_timestamp: Some(Utc::now()),
                    retrieval_source_uri: Some("wikidata:Q90".into()),
                    retrieval_axes: Some(axes()),
                },
                knowledge_axes: axes(),
                claim_id: None,
                cited_item_ids: vec![],
                rationale: None,
            }],
            provenance: Provenance {
                schema_version: "2.0".into(),
                plan_id: Uuid::new_v4(),
                items: vec![],
                claims: vec![],
                entailments: vec![],
                cache_key: "ab".into(),
                metadata: Default::default(),
            },
            invariants_verified: vec!["I1".into(), "I13".into()],
            joules_spent_total: 5.0,
            latency_ms: 1200,
            caveats: vec![],
            emitted_at: Utc::now(),
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: Answer = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn refusal_roundtrips() {
        let r = Refusal {
            schema_version: "2.0".into(),
            refusal_id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            reason_code: "authority_tier_unsatisfiable".into(),
            reason_message: "no source meets tier 2 requirement".into(),
            blocking_violations: vec!["v_auth_3".into()],
            emitted_at: Utc::now(),
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: Refusal = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
