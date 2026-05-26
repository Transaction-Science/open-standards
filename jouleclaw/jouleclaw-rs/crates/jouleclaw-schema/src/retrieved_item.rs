//! Items produced by the Execute pillar (spec §3.3).
//!
//! Every retriever returns [`RetrievedItem`]s. The `retrieval_context`
//! field carries the RAP step that produced it so the orchestrator
//! can audit the fallback path and the SystemCapabilities self-model
//! can observe which steps succeed in practice.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::Metadata;
use crate::knowledge_axes::KnowledgeAxes;
use crate::query_plan::Modality;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    StructuredKb,
    TextDocument,
    Image,
    Audio,
    Video,
    LiveFeed,
    ToolOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreType {
    Bm25,
    Cosine,
    Rrf,
    Exact,
    Combined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMethod {
    Sparql,
    Bm25,
    Dense,
    Hybrid,
    CrossModalHop,
    LiveSearch,
    ToolCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FreshnessClass {
    Live,
    Recent,
    Archival,
    Timeless,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcerptSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Content {
    pub modality: Modality,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub media_ref: Option<String>,
    #[serde(default)]
    pub structured: Option<serde_json::Value>,
    #[serde(default)]
    pub excerpt_span: Option<ExcerptSpan>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalContext {
    pub retriever_id: String,
    pub matched_against: String,
    pub sub_id: String,
    pub raw_score: f64,
    pub score_type: ScoreType,
    #[serde(default)]
    pub normalized_score: Option<f64>,
    pub rank_in_store: u32,
    pub retrieval_method: RetrievalMethod,
    #[serde(default)]
    pub hop_quality: Option<f64>,
    #[serde(default)]
    pub hop_path: Option<Vec<String>>,
    /// Which RAP step produced this item (e.g. `"primary"`,
    /// `"fallback_2"`).
    pub rap_step: String,
    pub rap_attempts: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Temporal {
    #[serde(default)]
    pub content_timestamp: Option<DateTime<Utc>>,
    pub retrieval_timestamp: DateTime<Utc>,
    #[serde(default)]
    pub last_modified: Option<DateTime<Utc>>,
    pub freshness_class: FreshnessClass,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Attribution {
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub publisher: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub canonical_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedItem {
    pub schema_version: String,
    pub item_id: Uuid,
    pub source_id: String,
    #[serde(default)]
    pub source_url: Option<String>,
    pub source_type: SourceType,
    pub content: Content,
    pub retrieval_context: RetrievalContext,
    pub temporal: Temporal,
    pub attribution: Attribution,
    /// v5 addition (spec §3.8): every retrieved item carries the
    /// seven-axis shape of the claim it represents.
    pub knowledge_axes: KnowledgeAxes,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_axes::{
        GranularityClass, KnowledgeAxes, ScopeClass, TemporalStabilityClass,
    };

    fn axes() -> KnowledgeAxes {
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

    fn item() -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: "wikidata:Q90".into(),
            source_url: Some("https://www.wikidata.org/wiki/Q90".into()),
            source_type: SourceType::StructuredKb,
            content: Content {
                modality: Modality::Text,
                text: Some("Paris".into()),
                media_ref: None,
                structured: None,
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "wikidata".into(),
                matched_against: "capital of France".into(),
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
                publisher: Some("Wikidata".into()),
                license: Some("CC0".into()),
                canonical_url: Some("https://www.wikidata.org/wiki/Q90".into()),
                ..Default::default()
            },
            knowledge_axes: axes(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn roundtrips_through_json() {
        let i = item();
        let json = serde_json::to_string(&i).unwrap();
        let back: RetrievedItem = serde_json::from_str(&json).unwrap();
        assert_eq!(i, back);
    }
}
