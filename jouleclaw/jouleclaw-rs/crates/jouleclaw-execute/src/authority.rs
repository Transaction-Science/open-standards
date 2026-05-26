//! Authority scoring (spec §5.4).
//!
//! v6 carries §5.4 forward from v1 unchanged but doesn't restate the
//! mechanism. Implementation here follows the standard
//! provenance-tier ladder documented on `KnowledgeAxes`: source
//! identity prefix + structured-KB classification picks the tier.
//! When a deployment ships its own scorer (e.g. journal-impact
//! weighting), implementations replace this with their own.

use chrono::Utc;
use uuid::Uuid;

use jouleclaw_schema::{AuthorityRecord, AuthorityTier, RetrievedItem, SourceType};

/// Trait wrapper so deployments can swap in their own scorer
/// without changing the orchestrator surface.
pub trait AuthorityScorer: Send + Sync {
    fn score(&self, item: &RetrievedItem) -> AuthorityRecord;
}

/// Built-in default scorer. Produces a tier from source identity +
/// `KnowledgeAxes.source_authority_tier`, with conservative defaults.
pub fn score_authority(item: &RetrievedItem) -> AuthorityRecord {
    let tier = pick_tier(item);
    let mut dims = std::collections::BTreeMap::new();
    if let SourceType::StructuredKb = item.source_type {
        dims.insert("structured_kb".into(), 1.0);
    }
    dims.insert(
        "axes_tier_consistency".into(),
        if item.knowledge_axes.source_authority_tier == tier.as_u8() {
            1.0
        } else {
            0.5
        },
    );
    AuthorityRecord {
        schema_version: "2.0".into(),
        record_id: Uuid::new_v4(),
        source_id: item.source_id.clone(),
        tier,
        dimensions: dims,
        assessed_at: Utc::now(),
        rationale: Some(rationale_for(item, tier)),
        metadata: Default::default(),
    }
}

fn pick_tier(item: &RetrievedItem) -> AuthorityTier {
    // 1. If the axes already declare a tier, trust it.
    if let Some(t) = AuthorityTier::from_u8(item.knowledge_axes.source_authority_tier) {
        return t;
    }
    // 2. Source-type hints.
    match item.source_type {
        SourceType::StructuredKb => AuthorityTier::Primary,
        SourceType::TextDocument => AuthorityTier::Secondary,
        SourceType::LiveFeed => AuthorityTier::Tertiary,
        SourceType::ToolOutput => AuthorityTier::Primary,
        SourceType::Image | SourceType::Audio | SourceType::Video => AuthorityTier::Tertiary,
    }
}

fn rationale_for(item: &RetrievedItem, tier: AuthorityTier) -> String {
    format!(
        "source_type={:?}, axes_tier={}, derived_tier={}",
        item.source_type,
        item.knowledge_axes.source_authority_tier,
        tier.as_u8()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_schema::{
        Attribution, Content, FreshnessClass, GranularityClass, KnowledgeAxes, Modality,
        RetrievalContext, RetrievalMethod, RetrievedItem, ScopeClass, ScoreType, SourceType,
        Temporal, TemporalStabilityClass,
    };

    fn make_item(stype: SourceType, axes_tier: u8) -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: "x".into(),
            source_url: None,
            source_type: stype,
            content: Content {
                modality: Modality::Text,
                text: Some("x".into()),
                media_ref: None,
                structured: None,
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "r".into(),
                matched_against: "x".into(),
                sub_id: "q0".into(),
                raw_score: 1.0,
                score_type: ScoreType::Exact,
                normalized_score: None,
                rank_in_store: 0,
                retrieval_method: RetrievalMethod::Sparql,
                hop_quality: None,
                hop_path: None,
                rap_step: "primary".into(),
                rap_attempts: 1,
            },
            temporal: Temporal {
                content_timestamp: None,
                retrieval_timestamp: Utc::now(),
                last_modified: None,
                freshness_class: FreshnessClass::Timeless,
            },
            attribution: Attribution::default(),
            knowledge_axes: KnowledgeAxes {
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
                source_authority_tier: axes_tier,
                extraction_method: None,
                citation_chain: vec![],
                metadata: Default::default(),
            },
            metadata: Default::default(),
        }
    }

    #[test]
    fn structured_kb_with_no_explicit_tier_defaults_to_primary() {
        let item = make_item(SourceType::StructuredKb, 99); // 99 won't parse as a tier
        let rec = score_authority(&item);
        assert_eq!(rec.tier, AuthorityTier::Primary);
    }

    #[test]
    fn explicit_axes_tier_wins_over_source_type() {
        let item = make_item(SourceType::StructuredKb, 3);
        let rec = score_authority(&item);
        assert_eq!(rec.tier, AuthorityTier::Tertiary);
    }

    #[test]
    fn community_tier_passes_through() {
        let item = make_item(SourceType::TextDocument, 4);
        let rec = score_authority(&item);
        assert_eq!(rec.tier, AuthorityTier::Community);
    }
}
