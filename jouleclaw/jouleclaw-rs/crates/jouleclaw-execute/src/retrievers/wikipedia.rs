//! Wikipedia REST API retriever — text-document tier 2 source.
//!
//! Counterpart to [`crate::retrievers::wikidata::WikidataRetriever`].
//! Where Wikidata returns short structured statements ("capital of
//! France = Paris"), Wikipedia returns prose extracts ("Paris is
//! the capital and most populous city of France, with an estimated
//! population of 2,102,650...") that the DeBERTa entailer can
//! cross-reference against the Wikidata claims to strengthen
//! verification, or substantiate answers Wikidata doesn't carry as
//! a property (population numbers, narrative descriptions, …).
//!
//! API: `GET https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}`
//! returns JSON with `title`, `extract` (plain-text summary), and
//! a permalink. No auth required.

use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

use async_trait::async_trait;

use jouleclaw_schema::{
    Attribution, Content, FreshnessClass, GranularityClass, KnowledgeAxes, Modality,
    RetrievalContext, RetrievalMethod, RetrievedItem, ScopeClass, ScoreType, SourceType, SubQuery,
    Temporal, TemporalStabilityClass,
};

use crate::retriever::{Retriever, RetrieverError};

const ENDPOINT_BASE: &str = "https://en.wikipedia.org/api/rest_v1/page/summary";
const USER_AGENT: &str = "joule-edge/0.1 (+https://github.com/jouleclaw-runtime/joule)";

pub struct WikipediaRetriever {
    id: String,
    client: reqwest::Client,
    endpoint: String,
    cache: Option<crate::retrievers::http_cache::HttpCache>,
}

impl WikipediaRetriever {
    pub fn new() -> Result<Self, RetrieverError> {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| RetrieverError::Backend(format!("reqwest build: {e}")))?;
        Ok(Self {
            id: "wikipedia".into(),
            client,
            endpoint: ENDPOINT_BASE.into(),
            cache: crate::retrievers::http_cache::default_http_cache(),
        })
    }

    pub fn with_endpoint(endpoint: impl Into<String>) -> Result<Self, RetrieverError> {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| RetrieverError::Backend(format!("reqwest build: {e}")))?;
        Ok(Self {
            id: "wikipedia".into(),
            client,
            endpoint: endpoint.into(),
            cache: crate::retrievers::http_cache::default_http_cache(),
        })
    }

    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self
    }

    async fn fetch_summary(&self, title: &str) -> Result<Value, RetrieverError> {
        let escaped = title.replace(' ', "_");
        let url = format!("{}/{}", self.endpoint, escaped);

        // HTTP cache lookup — wikipedia article summaries change
        // slowly enough that 24h cached responses are safe.
        let cache_key = self
            .cache
            .as_ref()
            .map(|c| c.key(&self.endpoint, &escaped));
        if let (Some(cache), Some(key)) = (&self.cache, cache_key.as_ref()) {
            if let Some(body) = cache.get(key) {
                return serde_json::from_str(&body).map_err(|e| {
                    RetrieverError::ParseFailed(format!("cache json: {e}"))
                });
            }
        }

        let resp = self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| RetrieverError::Backend(format!("wikipedia request: {e}")))?;

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(serde_json::json!({}));
        }
        if !status.is_success() {
            return Err(RetrieverError::Backend(format!(
                "wikipedia status {status}"
            )));
        }
        let body = resp
            .text()
            .await
            .map_err(|e| RetrieverError::Backend(format!("wikipedia body: {e}")))?;
        let value: Value = serde_json::from_str(&body)
            .map_err(|e| RetrieverError::ParseFailed(format!("wikipedia json: {e}")))?;

        if let (Some(cache), Some(key)) = (&self.cache, cache_key.as_ref()) {
            let _ = cache.put(key, &self.endpoint, &body);
        }
        Ok(value)
    }
}

#[async_trait]
impl Retriever for WikipediaRetriever {
    fn retriever_id(&self) -> &str {
        &self.id
    }

    async fn call(
        &self,
        method: &str,
        subquery: &SubQuery,
        _parameters: &serde_json::Map<String, Value>,
    ) -> Result<Vec<RetrievedItem>, RetrieverError> {
        match method {
            // Both names map to the same handler — `wikipedia_summary`
            // is the v2 name; `wikipedia_fallback` is the spec
            // §5.2-original name used by the Wikidata RAP cascade.
            "wikipedia_summary" | "wikipedia_fallback" => {
                self.summary(subquery).await
            }
            other => Err(RetrieverError::UnknownMethod(other.into())),
        }
    }
}

impl WikipediaRetriever {
    async fn summary(&self, subquery: &SubQuery) -> Result<Vec<RetrievedItem>, RetrieverError> {
        let title = candidate_title(&subquery.text);
        if title.is_empty() {
            return Ok(vec![]);
        }
        let body = self.fetch_summary(&title).await?;
        // Empty object body = the 404 path. No items.
        if body.as_object().map(|o| o.is_empty()).unwrap_or(true) {
            return Ok(vec![]);
        }
        Ok(parse_summary(&body, &subquery.sub_id))
    }
}

/// Strip relational prefixes ("capital of ", "population of ", …)
/// so a query phrased as a relation still points at the right
/// article. "population of Tokyo" → "Tokyo". Mirrors the prefix
/// list used by [`crate::retrievers::wikidata::WikidataRetriever`]
/// and [`jouleclaw_edge_cli`]'s refinement helper.
fn candidate_title(query: &str) -> String {
    let trimmed = query.trim().trim_end_matches(['?', '.', '!', ' ']);
    let lower = trimmed.to_lowercase();
    const PREFIXES: &[&str] = &[
        "capital of ",
        "the capital of ",
        "currency of ",
        "official language of ",
        "language of ",
        "head of state of ",
        "head of government of ",
        "continent of ",
        "country of ",
        "population of ",
        "area of ",
        "elevation of ",
        "borders of ",
    ];
    for p in PREFIXES {
        if let Some(rest) = lower.strip_prefix(p) {
            let byte_off = p.len();
            if byte_off <= trimmed.len() {
                return trimmed[byte_off..].trim().to_string();
            }
            return rest.trim().to_string();
        }
    }
    trimmed.to_string()
}

fn parse_summary(body: &Value, sub_id: &str) -> Vec<RetrievedItem> {
    let title = body
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let extract = body
        .get("extract")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if extract.is_empty() {
        return vec![];
    }
    let page_url = body
        .get("content_urls")
        .and_then(|v| v.get("desktop"))
        .and_then(|v| v.get("page"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            // Fall back to constructing the standard URL.
            Some(format!(
                "https://en.wikipedia.org/wiki/{}",
                title.replace(' ', "_")
            ))
        });
    let last_modified = body
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc));

    let text = if title.is_empty() {
        extract.clone()
    } else {
        format!("{title}: {extract}")
    };

    vec![RetrievedItem {
        schema_version: "2.0".into(),
        item_id: Uuid::new_v4(),
        source_id: format!("wikipedia:{}", title.replace(' ', "_")),
        source_url: page_url.clone(),
        source_type: SourceType::TextDocument,
        content: Content {
            modality: Modality::Text,
            text: Some(text),
            media_ref: None,
            structured: Some(Value::Object(body.as_object().cloned().unwrap_or_default())),
            excerpt_span: None,
        },
        retrieval_context: RetrievalContext {
            retriever_id: "wikipedia".into(),
            matched_against: title.clone(),
            sub_id: sub_id.to_string(),
            raw_score: 1.0,
            score_type: ScoreType::Combined,
            normalized_score: Some(0.9),
            rank_in_store: 0,
            retrieval_method: RetrievalMethod::LiveSearch,
            hop_quality: None,
            hop_path: None,
            rap_step: "primary".into(),
            rap_attempts: 1,
        },
        temporal: Temporal {
            content_timestamp: last_modified,
            retrieval_timestamp: Utc::now(),
            last_modified,
            freshness_class: FreshnessClass::Recent,
        },
        attribution: Attribution {
            publisher: Some("Wikipedia".into()),
            license: Some("CC BY-SA 4.0".into()),
            canonical_url: page_url,
            ..Default::default()
        },
        knowledge_axes: KnowledgeAxes {
            schema_version: "5.0".into(),
            valid_time_start: None,
            valid_time_end: None,
            transaction_time: last_modified,
            reference_time: Utc::now(),
            // Wikipedia content is community-edited prose. Most facts
            // are slow-to-medium stability; we default to Medium so
            // freshness violations fire when the query demands "current"
            // and the page is months stale.
            temporal_stability: TemporalStabilityClass::Medium,
            granularity: GranularityClass::Medium,
            granularity_notes: None,
            scope: ScopeClass::Particular,
            scope_domain: Some(title),
            // Wikipedia is tertiary-aggregator authority per the
            // schema's `AuthorityTier` ladder; trustworthy on
            // common topics, less so on niche or controversial ones.
            certainty: 0.85,
            certainty_basis: "wikipedia_consensus_prose".into(),
            source_uri: None,
            source_authority_tier: 3,
            extraction_method: Some("wikipedia_rest_summary".into()),
            citation_chain: vec![],
            metadata: Default::default(),
        },
        metadata: Default::default(),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_schema::Modality;

    #[test]
    fn candidate_title_strips_relational_prefix() {
        assert_eq!(candidate_title("capital of France"), "France");
        assert_eq!(candidate_title("population of Tokyo"), "Tokyo");
        assert_eq!(candidate_title("Paris"), "Paris");
    }

    #[test]
    fn candidate_title_handles_trailing_punctuation() {
        assert_eq!(candidate_title("Paris?"), "Paris");
        assert_eq!(candidate_title("Mont Blanc."), "Mont Blanc");
    }

    #[test]
    fn parse_summary_builds_item_with_attribution() {
        let body = serde_json::json!({
            "title": "Paris",
            "extract": "Paris is the capital and most populous city of France.",
            "content_urls": {
                "desktop": {
                    "page": "https://en.wikipedia.org/wiki/Paris"
                }
            },
            "timestamp": "2026-05-01T12:34:56Z"
        });
        let items = parse_summary(&body, "q1");
        assert_eq!(items.len(), 1);
        let it = &items[0];
        assert_eq!(it.source_id, "wikipedia:Paris");
        assert_eq!(
            it.source_url.as_deref(),
            Some("https://en.wikipedia.org/wiki/Paris")
        );
        let text = it.content.text.as_deref().unwrap();
        assert!(text.contains("Paris"));
        assert!(text.contains("capital"));
        // Wikipedia is tier 3 (tertiary aggregator).
        assert_eq!(it.knowledge_axes.source_authority_tier, 3);
        assert_eq!(it.knowledge_axes.certainty_basis, "wikipedia_consensus_prose");
    }

    #[test]
    fn parse_summary_skips_when_no_extract() {
        let body = serde_json::json!({
            "title": "Empty",
            "extract": ""
        });
        let items = parse_summary(&body, "q1");
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn unknown_method_errors() {
        let r = WikipediaRetriever::new().unwrap();
        let sub = jouleclaw_schema::SubQuery {
            sub_id: "q0".into(),
            text: "Paris".into(),
            depends_on: vec![],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["wikipedia".into()],
            priority: 1.0,
            rap_id: "x".into(),
        };
        let res = r
            .call("totally_unknown", &sub, &Default::default())
            .await;
        assert!(matches!(res, Err(RetrieverError::UnknownMethod(_))));
    }

    /// Live network smoke. Skipped by default; opt in with
    ///   cargo test -p jouleclaw-execute -- --ignored wikipedia_live
    #[tokio::test]
    #[ignore]
    async fn wikipedia_live_smoke() {
        let r = WikipediaRetriever::new().unwrap();
        let sub = jouleclaw_schema::SubQuery {
            sub_id: "q0".into(),
            text: "Paris".into(),
            depends_on: vec![],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["wikipedia".into()],
            priority: 1.0,
            rap_id: "wikipedia_summary_v1".into(),
        };
        let items = r
            .call("wikipedia_summary", &sub, &Default::default())
            .await
            .expect("wikipedia reachable");
        assert!(!items.is_empty(), "expected one summary item");
        assert!(items[0].source_id.starts_with("wikipedia:"));
        let text = items[0].content.text.as_deref().unwrap();
        eprintln!("  → {text}");
        assert!(text.to_lowercase().contains("paris"));
    }
}
