//! Draft composer (spec §7.1).
//!
//! Produces a draft answer as a sequence of `(segment_id, text)`
//! pairs in **verification order** — well-grounded content first,
//! interpretive content last. The diagnose pillar consumes the
//! pairs via [`jouleclaw_diagnose::atomize_sentences`]; the verified
//! composer (§7.2) then applies recovery actions per segment.
//!
//! Two shapes available:
//!
//! - [`TemplateComposer`] — templated, no LLM dependency. Emits one
//!   segment per retrieved item with the item's text quoted. Honest:
//!   the draft is no richer than what was retrieved, and the
//!   subsequent verification only entails its own quoted material.
//!   Sufficient for the minimum-viable end-to-end path.
//! - [`DraftComposer`] trait — pluggable surface for LLM-backed
//!   composers when the deployment can afford one. The diagnose
//!   pillar treats both the same way.

use jouleclaw_schema::{AuthorityRecord, QueryPlan, RetrievedItem};

#[derive(Debug)]
pub enum DraftError {
    EmptyRetrieval,
    Backend(String),
}

impl std::fmt::Display for DraftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRetrieval => write!(f, "no retrieved items to compose"),
            Self::Backend(s) => write!(f, "draft backend: {s}"),
        }
    }
}

impl std::error::Error for DraftError {}

/// One draft segment ready for atomization. `cited_item_ids` records
/// which retrieved items the segment was composed from; the verified
/// composer threads these into [`jouleclaw_schema::AnswerSegment`].
#[derive(Debug, Clone, PartialEq)]
pub struct DraftSegment {
    pub segment_id: String,
    pub text: String,
    pub cited_item_ids: Vec<uuid::Uuid>,
}

pub trait DraftComposer: Send + Sync {
    /// Produce a draft for the given plan + retrieved evidence.
    /// `authority` records may be empty; high-authority items
    /// should be ordered first per "verification order".
    fn draft(
        &self,
        plan: &QueryPlan,
        items: &[RetrievedItem],
        authority: &[AuthorityRecord],
    ) -> Result<Vec<DraftSegment>, DraftError>;
}

/// Templated composer that emits one segment per retrieved item,
/// ordered by authority tier (Primary first). The segment text is
/// the source's content verbatim — no inline source label.
/// Provenance lives in `cited_item_ids` (item UUIDs) and is
/// surfaced via [`jouleclaw_schema::AnswerSegment::attribution`] at
/// presentation time. Keeping the prefix out of the segment text
/// matters for verification: an entailer asked "does `X` entail
/// `Wikidata: X`?" tends to answer Neutral because the second
/// asserts a fact about a Wikidata page rather than the
/// proposition itself.
pub struct TemplateComposer;

impl TemplateComposer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TemplateComposer {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftComposer for TemplateComposer {
    fn draft(
        &self,
        _plan: &QueryPlan,
        items: &[RetrievedItem],
        authority: &[AuthorityRecord],
    ) -> Result<Vec<DraftSegment>, DraftError> {
        if items.is_empty() {
            return Err(DraftError::EmptyRetrieval);
        }
        let mut ordered: Vec<&RetrievedItem> = items.iter().collect();
        ordered.sort_by_key(|it| effective_tier(it, authority));

        let mut out = Vec::with_capacity(ordered.len());
        for (idx, item) in ordered.iter().enumerate() {
            let segment_id = format!("s{idx}");
            let label = item
                .attribution
                .publisher
                .clone()
                .unwrap_or_else(|| item.source_id.clone());
            let body = item
                .content
                .text
                .clone()
                .unwrap_or_else(|| format!("[non-text {} content]", label));
            // Ensure the segment ends with sentence-final
            // punctuation so the atomizer's sentence-splitter
            // recognizes it.
            let body_trimmed = body.trim().to_string();
            let text = if matches!(
                body_trimmed.chars().last(),
                Some('.') | Some('!') | Some('?')
            ) {
                body_trimmed
            } else {
                format!("{body_trimmed}.")
            };
            out.push(DraftSegment {
                segment_id,
                text,
                cited_item_ids: vec![item.item_id],
            });
        }
        Ok(out)
    }
}

/// Effective authority tier — prefers the explicit
/// `AuthorityRecord` when one matches the item.
fn effective_tier(item: &RetrievedItem, authority: &[AuthorityRecord]) -> u8 {
    authority
        .iter()
        .find(|r| r.source_id == item.source_id)
        .map(|r| r.tier.as_u8())
        .unwrap_or(item.knowledge_axes.source_authority_tier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use jouleclaw_schema::*;
    use uuid::Uuid;

    fn axes(tier: u8) -> KnowledgeAxes {
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
            scope_domain: None,
            certainty: 0.95,
            certainty_basis: "test".into(),
            source_uri: None,
            source_authority_tier: tier,
            extraction_method: None,
            citation_chain: vec![],
            metadata: Default::default(),
        }
    }

    fn item(source_id: &str, publisher: &str, text: &str, tier: u8) -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: source_id.into(),
            source_url: None,
            source_type: SourceType::StructuredKb,
            content: Content {
                modality: Modality::Text,
                text: Some(text.into()),
                media_ref: None,
                structured: None,
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "wikidata".into(),
                matched_against: "test".into(),
                sub_id: "q0".into(),
                raw_score: 1.0,
                score_type: ScoreType::Exact,
                normalized_score: Some(1.0),
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
            attribution: Attribution {
                publisher: Some(publisher.into()),
                license: Some("CC0".into()),
                canonical_url: None,
                ..Default::default()
            },
            knowledge_axes: axes(tier),
            metadata: Default::default(),
        }
    }

    fn minimal_plan() -> QueryPlan {
        QueryPlan {
            schema_version: "2.0".into(),
            plan_id: Uuid::new_v4(),
            original_query: OriginalQuery {
                text: Some("q".into()),
                image_ref: None,
                audio_ref: None,
                video_ref: None,
                language_detected: "en".into(),
                timestamp: Utc::now(),
            },
            intent: Intent::Lookup,
            modalities_in: vec![Modality::Text],
            modalities_out: vec![Modality::Text],
            decomposition: vec![],
            constraints: Default::default(),
            budget: Default::default(),
            invariants_satisfied: PlanInvariants {
                all_subqueries_have_at_least_one_store: true,
                dependency_graph_is_acyclic: true,
                total_estimated_latency_within_budget: true,
                estimated_cost_within_budget: true,
                required_modalities_covered: true,
            },
            metadata: Default::default(),
        }
    }

    #[test]
    fn template_composer_emits_one_segment_per_item() {
        let plan = minimal_plan();
        let items = vec![
            item("wikidata:Q90", "Wikidata", "Paris is the capital of France", 1),
            item("wikidata:Q183", "Wikidata", "Germany is in Europe", 1),
        ];
        let segs = TemplateComposer::new().draft(&plan, &items, &[]).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].segment_id, "s0");
        assert_eq!(segs[1].segment_id, "s1");
        assert!(segs[0].text.contains("Paris"));
    }

    #[test]
    fn template_composer_orders_by_authority_tier() {
        let plan = minimal_plan();
        let items = vec![
            item("blog:misinfo", "Some Blog", "Lyon is the capital", 4),
            item("wikidata:Q90", "Wikidata", "Paris is the capital", 1),
        ];
        let segs = TemplateComposer::new().draft(&plan, &items, &[]).unwrap();
        assert_eq!(segs.len(), 2);
        // Wikidata (tier 1) sorts before blog (tier 4).
        assert!(segs[0].text.contains("Paris"));
        assert!(segs[1].text.contains("Lyon"));
    }

    #[test]
    fn template_composer_terminates_sentence_with_period() {
        let plan = minimal_plan();
        let items = vec![item("wikidata:Q90", "Wikidata", "Paris is the capital", 1)];
        let segs = TemplateComposer::new().draft(&plan, &items, &[]).unwrap();
        assert!(segs[0].text.ends_with('.'));
    }

    #[test]
    fn template_composer_errors_on_empty_input() {
        let plan = minimal_plan();
        let err = TemplateComposer::new().draft(&plan, &[], &[]).unwrap_err();
        assert!(matches!(err, DraftError::EmptyRetrieval));
    }

    #[test]
    fn template_composer_threads_item_ids_into_citations() {
        let plan = minimal_plan();
        let item = item("wikidata:Q90", "Wikidata", "Paris is the capital.", 1);
        let item_id = item.item_id;
        let segs = TemplateComposer::new()
            .draft(&plan, &[item], &[])
            .unwrap();
        assert_eq!(segs[0].cited_item_ids, vec![item_id]);
    }

    #[test]
    fn authority_record_override_promotes_low_axes_tier_item() {
        // Item declares tier 4 in axes, but AuthorityRecord upgrades
        // it to tier 1 — the composer should sort it ahead.
        let plan = minimal_plan();
        let trusted = item("wikidata:Q90", "Wikidata", "Paris (trusted)", 4);
        let untrusted = item("blog:Q1", "Random Blog", "Lyon", 4);
        let upgrade = AuthorityRecord {
            schema_version: "2.0".into(),
            record_id: Uuid::new_v4(),
            source_id: "wikidata:Q90".into(),
            tier: AuthorityTier::Primary,
            dimensions: Default::default(),
            assessed_at: Utc::now(),
            rationale: None,
            metadata: Default::default(),
        };
        let segs = TemplateComposer::new()
            .draft(&plan, &[trusted, untrusted], &[upgrade])
            .unwrap();
        assert!(segs[0].text.contains("Paris (trusted)"));
    }
}
