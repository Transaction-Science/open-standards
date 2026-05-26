//! Wikidata SPARQL retriever (spec §5.2 example RAP).
//!
//! Implements the methods named in the canonical Wikidata RAP:
//!
//! - `wikidata_primary` — entity-linked label match + predicate
//!   template SPARQL (covers the "lookup" query class the spec uses
//!   as its prototype: "what countries does Brazil border", "what
//!   is the capital of France").
//!
//! The remaining methods (`wikidata_alt_predicate`,
//! `wikidata_alt_entity`, `wikidata_property_path`,
//! `wikidata_text_search`, `wikipedia_fallback`) ship as stubs that
//! return `Err(UnknownMethod)` so the RAP executor falls through
//! cleanly. Filling them in is incremental work; the substrate is
//! complete.
//!
//! ## Timeouts
//!
//! Wikidata's public endpoint runs slow as of May 2026 — multi-second
//! response times are normal. Per-step timeouts in the RAP definition
//! (see spec §5.2 calibration note) account for this. The retriever
//! itself does not add its own timeout; the RAP executor controls it.

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

const WDQS_ENDPOINT: &str = "https://query.wikidata.org/sparql";
const USER_AGENT: &str = "joule-edge/0.1 (+https://github.com/jouleclaw-runtime/joule)";

pub struct WikidataRetriever {
    id: String,
    client: reqwest::Client,
    endpoint: String,
    cache: Option<crate::retrievers::http_cache::HttpCache>,
}

impl WikidataRetriever {
    /// Construct against the public WDQS endpoint.
    pub fn new() -> Result<Self, RetrieverError> {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| RetrieverError::Backend(format!("reqwest build: {e}")))?;
        Ok(Self {
            id: "wikidata".into(),
            client,
            endpoint: WDQS_ENDPOINT.into(),
            cache: crate::retrievers::http_cache::default_http_cache(),
        })
    }

    /// Construct against a custom endpoint (e.g. local Qlever mirror
    /// or test server).
    pub fn with_endpoint(endpoint: impl Into<String>) -> Result<Self, RetrieverError> {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| RetrieverError::Backend(format!("reqwest build: {e}")))?;
        Ok(Self {
            id: "wikidata".into(),
            client,
            endpoint: endpoint.into(),
            cache: crate::retrievers::http_cache::default_http_cache(),
        })
    }

    /// Disable the per-request HTTP cache for this retriever.
    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self
    }

    /// Use a custom cache instance instead of the default.
    pub fn with_cache(mut self, cache: crate::retrievers::http_cache::HttpCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Run a SPARQL query and parse the JSON response into the
    /// standard `head + results.bindings` shape. Wrapped with the
    /// HTTP cache when configured — repeat queries (same SPARQL
    /// string, same endpoint) return from disk in microseconds
    /// instead of paying the multi-second WDQS round-trip.
    async fn sparql(&self, query: &str) -> Result<Value, RetrieverError> {
        // Cache lookup.
        let cache_key = self
            .cache
            .as_ref()
            .map(|c| c.key(&self.endpoint, query));
        if let (Some(cache), Some(key)) = (&self.cache, cache_key.as_ref()) {
            if let Some(body) = cache.get(key) {
                let value: Value = serde_json::from_str(&body)
                    .map_err(|e| RetrieverError::ParseFailed(format!("cache json: {e}")))?;
                return Ok(value);
            }
        }

        // Cache miss → live request.
        let resp = self
            .client
            .get(&self.endpoint)
            .header("Accept", "application/sparql-results+json")
            .query(&[("query", query), ("format", "json")])
            .send()
            .await
            .map_err(|e| RetrieverError::Backend(format!("wdqs request: {e}")))?;
        if !resp.status().is_success() {
            return Err(RetrieverError::Backend(format!(
                "wdqs status {}",
                resp.status()
            )));
        }
        let body = resp
            .text()
            .await
            .map_err(|e| RetrieverError::Backend(format!("wdqs body: {e}")))?;
        let value: Value = serde_json::from_str(&body)
            .map_err(|e| RetrieverError::ParseFailed(format!("wdqs json: {e}")))?;

        // Cache store on success.
        if let (Some(cache), Some(key)) = (&self.cache, cache_key.as_ref()) {
            let _ = cache.put(key, &self.endpoint, &body);
        }
        Ok(value)
    }
}

#[async_trait]
impl Retriever for WikidataRetriever {
    fn retriever_id(&self) -> &str {
        &self.id
    }

    async fn call(
        &self,
        method: &str,
        subquery: &SubQuery,
        parameters: &serde_json::Map<String, Value>,
    ) -> Result<Vec<RetrievedItem>, RetrieverError> {
        match method {
            "wikidata_primary" => self.primary(subquery, parameters).await,
            "wikidata_property_path" => self.property_path(subquery, parameters).await,
            // Remaining RAP steps land incrementally.
            "wikidata_alt_predicate"
            | "wikidata_alt_entity"
            | "wikidata_text_search"
            | "wikipedia_fallback" => Err(RetrieverError::UnknownMethod(method.into())),
            other => Err(RetrieverError::UnknownMethod(other.into())),
        }
    }
}

impl WikidataRetriever {
    /// `wikidata_primary`: search for entities matching the
    /// sub-query's text, then return matching items.
    ///
    /// SPARQL strategy:
    ///   - `wikibase:label` service to find entities with English
    ///     labels matching the query string,
    ///   - rank by `wikibase:sitelinks` to prefer the more notable
    ///     match,
    ///   - LIMIT 5.
    ///
    /// The returned items carry `KnowledgeAxes` populated with what
    /// Wikidata can supply directly: scope=Particular,
    /// granularity=Medium, certainty=0.95 (community-edited),
    /// authority tier=1 (structured KB), provenance via the
    /// canonical Wikidata URL.
    async fn primary(
        &self,
        subquery: &SubQuery,
        _parameters: &serde_json::Map<String, Value>,
    ) -> Result<Vec<RetrievedItem>, RetrieverError> {
        let query_text = subquery.text.trim();
        if query_text.is_empty() {
            return Ok(vec![]);
        }
        let sparql = format!(
            r#"
            SELECT DISTINCT ?item ?itemLabel ?itemDescription ?sitelinks WHERE {{
              SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                bd:serviceParam wikibase:api "EntitySearch" .
                bd:serviceParam mwapi:search "{search}" .
                bd:serviceParam mwapi:language "en" .
                ?item wikibase:apiOutputItem mwapi:item .
              }}
              ?item wikibase:sitelinks ?sitelinks .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            ORDER BY DESC(?sitelinks)
            LIMIT 5
            "#,
            search = escape_sparql_string(query_text),
        );

        let body = self.sparql(&sparql).await?;
        let raw = parse_bindings(&body, &subquery.sub_id)?;
        // Filter out Wikimedia meta-entities (list / category /
        // disambiguation / template / project pages). When a
        // sub-query is phrased as a relation ("capital of France")
        // EntitySearch sometimes returns the meta-list-of-X article
        // instead of a real instance. Treating these as primary
        // hits poisons the diagnose pillar — they technically match
        // the literal query string but contradict what the user is
        // actually asking. Returning the filtered list (possibly
        // empty) lets the RAP's `OnEmpty` step (property_path)
        // fire and find the real answer.
        Ok(raw.into_iter().filter(|it| !is_meta_entity(it)).collect())
    }

    /// `wikidata_property_path`: pattern-match a relational
    /// sub-query against a small registry of `(prefix, property)`
    /// pairs, extract the subject entity, look it up via
    /// EntitySearch, then follow the relation. For example,
    /// "capital of France" matches `("capital of ", "P36")`,
    /// resolves "France" to Q142, and queries
    /// `wd:Q142 wdt:P36 ?capital` to produce Paris (Q90).
    ///
    /// Patterns are intentionally English-only and small — the
    /// full §4.1 LLM-based query understanding would replace this
    /// with proper intent extraction. This is the minimum that
    /// lets the RAP cascade close on relational lookups.
    async fn property_path(
        &self,
        subquery: &SubQuery,
        _parameters: &serde_json::Map<String, Value>,
    ) -> Result<Vec<RetrievedItem>, RetrieverError> {
        let text = subquery.text.trim().to_lowercase();
        let text = text.trim_start_matches("what is the ").trim_start_matches("what is ");
        let text = text.trim_end_matches(['?', '.', '!', ' ']);

        let Some((property, subject)) = match_property_path(text) else {
            // Nothing to do for this query.
            return Ok(vec![]);
        };

        let sparql = format!(
            r#"
            SELECT ?subject ?subjectLabel ?value ?valueLabel ?valueDescription ?sitelinks WHERE {{
              SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                bd:serviceParam wikibase:api "EntitySearch" .
                bd:serviceParam mwapi:search "{search}" .
                bd:serviceParam mwapi:language "en" .
                ?subject wikibase:apiOutputItem mwapi:item .
              }}
              ?subject wdt:{property} ?value .
              OPTIONAL {{ ?value wikibase:sitelinks ?sitelinks . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 5
            "#,
            search = escape_sparql_string(subject),
            property = property,
        );
        let body = self.sparql(&sparql).await?;
        parse_property_path_bindings(&body, &subquery.sub_id, property, subject)
    }
}

fn escape_sparql_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// True if a parsed binding looks like a Wikimedia infrastructure
/// page rather than a domain entity. These show up when
/// `EntitySearch` literal-matches a relational phrase
/// ("capital of France" → "list of capitals of France"); they're
/// not what the user is asking about.
fn is_meta_entity(item: &RetrievedItem) -> bool {
    let blob = item
        .content
        .text
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    const META_MARKERS: &[&str] = &[
        "wikimedia list article",
        "wikimedia category",
        "wikimedia disambiguation",
        "wikimedia template",
        "wikimedia project page",
        "wikinews article",
    ];
    META_MARKERS.iter().any(|m| blob.contains(m))
}

/// Pattern registry for relational sub-queries. Returns
/// `(wikidata_property_id, extracted_subject)` when the text
/// matches. Pure English text matching; the §4.1 LLM-based
/// understanding would replace this with proper intent extraction.
fn match_property_path(text: &str) -> Option<(&'static str, &str)> {
    // (prefix, property) where the entity is whatever follows the prefix.
    const PREFIX_PATTERNS: &[(&str, &str)] = &[
        ("capital of ", "P36"),
        ("the capital of ", "P36"),
        ("currency of ", "P38"),
        ("official language of ", "P37"),
        ("language of ", "P37"),
        ("head of state of ", "P35"),
        ("head of government of ", "P6"),
        ("continent of ", "P30"),
        ("inception of ", "P571"),
        ("country of ", "P17"),
    ];
    // (suffix, property) — "<entity> borders" etc.
    const SUFFIX_PATTERNS: &[(&str, &str)] = &[
        (" borders", "P47"),
        (" shares a border with", "P47"),
    ];
    for (prefix, property) in PREFIX_PATTERNS {
        if let Some(rest) = text.strip_prefix(prefix) {
            let subject = rest.trim();
            if !subject.is_empty() {
                return Some((property, subject));
            }
        }
    }
    for (suffix, property) in SUFFIX_PATTERNS {
        if let Some(head) = text.strip_suffix(suffix) {
            let subject = head.trim();
            if !subject.is_empty() {
                return Some((property, subject));
            }
        }
    }
    None
}

/// Parse `wdt:<property>` rows into `RetrievedItem`s. The value
/// entity (e.g. Q90 / Paris) becomes the item; the subject /
/// property are recorded in `structured` so downstream can
/// reconstruct the relation.
fn parse_property_path_bindings(
    body: &Value,
    sub_id: &str,
    property: &str,
    subject_query: &str,
) -> Result<Vec<RetrievedItem>, RetrieverError> {
    let bindings = body
        .get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
        .ok_or_else(|| RetrieverError::ParseFailed("missing results.bindings".into()))?;

    let mut items: Vec<RetrievedItem> = Vec::with_capacity(bindings.len());
    for (rank, binding) in bindings.iter().enumerate() {
        let value_uri = binding
            .get("value")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if value_uri.is_empty() {
            continue;
        }
        let value_label = binding
            .get("valueLabel")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let value_desc = binding
            .get("valueDescription")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let subject_label = binding
            .get("subjectLabel")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // The user-facing text restates the relation so a
        // downstream entailer sees a complete proposition.
        let text = if value_desc.is_empty() {
            format!(
                "The {property} of {subject_label} is {value_label}.",
                property = property_name(property),
            )
        } else {
            format!(
                "The {property} of {subject_label} is {value_label} ({value_desc}).",
                property = property_name(property),
            )
        };
        let sitelinks: u64 = binding
            .get("sitelinks")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        items.push(RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: short_id_from_uri(&value_uri),
            source_url: Some(value_uri.clone()),
            source_type: SourceType::StructuredKb,
            content: Content {
                modality: Modality::Text,
                text: Some(text),
                media_ref: None,
                structured: Some(Value::Object(
                    binding.as_object().cloned().unwrap_or_default(),
                )),
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "wikidata".into(),
                matched_against: format!("{subject_query} {property}"),
                sub_id: sub_id.to_string(),
                raw_score: sitelinks as f64,
                score_type: ScoreType::Exact,
                normalized_score: Some(if sitelinks == 0 {
                    0.5
                } else {
                    (sitelinks as f64).ln() / 10.0
                }),
                rank_in_store: rank as u32,
                retrieval_method: RetrievalMethod::Sparql,
                hop_quality: Some(1.0),
                hop_path: Some(vec![subject_query.into(), property.into()]),
                rap_step: "property_path".into(),
                rap_attempts: 1,
            },
            temporal: Temporal {
                content_timestamp: None,
                retrieval_timestamp: Utc::now(),
                last_modified: None,
                freshness_class: FreshnessClass::Recent,
            },
            attribution: Attribution {
                publisher: Some("Wikidata".into()),
                license: Some("CC0".into()),
                canonical_url: Some(value_uri),
                ..Default::default()
            },
            knowledge_axes: KnowledgeAxes {
                schema_version: "5.0".into(),
                valid_time_start: None,
                valid_time_end: None,
                transaction_time: None,
                reference_time: Utc::now(),
                temporal_stability: TemporalStabilityClass::Slow,
                granularity: GranularityClass::Medium,
                granularity_notes: None,
                scope: ScopeClass::Particular,
                scope_domain: Some(subject_query.into()),
                certainty: 0.95,
                certainty_basis: "wikidata_property_path".into(),
                source_uri: Some(format!(
                    "https://www.wikidata.org/wiki/{}",
                    short_id_from_uri(&items.last().map(|i| i.source_id.clone()).unwrap_or_default())
                )),
                source_authority_tier: 1,
                extraction_method: Some("sparql_property_path".into()),
                citation_chain: vec![],
                metadata: Default::default(),
            },
            metadata: Default::default(),
        });
    }
    Ok(items)
}

/// Human-readable name for the small Wikidata property registry we
/// understand. Used to build the entailment-ready sentence text.
fn property_name(pid: &str) -> &'static str {
    match pid {
        "P36" => "capital",
        "P38" => "currency",
        "P37" => "official language",
        "P35" => "head of state",
        "P6" => "head of government",
        "P30" => "continent",
        "P571" => "inception date",
        "P17" => "country",
        "P47" => "neighboring countries",
        _ => "property",
    }
}

fn parse_bindings(body: &Value, sub_id: &str) -> Result<Vec<RetrievedItem>, RetrieverError> {
    let bindings = body
        .get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
        .ok_or_else(|| RetrieverError::ParseFailed("missing results.bindings".into()))?;

    let mut items: Vec<RetrievedItem> = Vec::with_capacity(bindings.len());
    for (rank, binding) in bindings.iter().enumerate() {
        let item_uri = binding
            .get("item")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if item_uri.is_empty() {
            continue;
        }
        let label = binding
            .get("itemLabel")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let description = binding
            .get("itemDescription")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let text = if description.is_empty() {
            label.clone()
        } else {
            format!("{label}: {description}")
        };
        let sitelinks: u64 = binding
            .get("sitelinks")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        items.push(RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: short_id_from_uri(&item_uri),
            source_url: Some(item_uri.clone()),
            source_type: SourceType::StructuredKb,
            content: Content {
                modality: Modality::Text,
                text: Some(text),
                media_ref: None,
                structured: Some(Value::Object(
                    binding.as_object().cloned().unwrap_or_default(),
                )),
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "wikidata".into(),
                matched_against: sub_id.to_string(),
                sub_id: sub_id.to_string(),
                raw_score: sitelinks as f64,
                score_type: ScoreType::Exact,
                normalized_score: Some(if sitelinks == 0 {
                    0.0
                } else {
                    (sitelinks as f64).ln() / 10.0
                }),
                rank_in_store: rank as u32,
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
                freshness_class: FreshnessClass::Recent,
            },
            attribution: Attribution {
                publisher: Some("Wikidata".into()),
                license: Some("CC0".into()),
                canonical_url: Some(item_uri),
                ..Default::default()
            },
            knowledge_axes: KnowledgeAxes {
                schema_version: "5.0".into(),
                valid_time_start: None,
                valid_time_end: None,
                transaction_time: None,
                reference_time: Utc::now(),
                temporal_stability: TemporalStabilityClass::Slow,
                granularity: GranularityClass::Medium,
                granularity_notes: None,
                scope: ScopeClass::Particular,
                scope_domain: None,
                certainty: 0.95,
                certainty_basis: "wikidata_structured_kb".into(),
                source_uri: Some(format!(
                    "https://www.wikidata.org/wiki/{}",
                    short_id_from_uri(
                        body.get("results")
                            .and_then(|r| r.get("bindings"))
                            .and_then(|b| b.get(rank))
                            .and_then(|b| b.get("item"))
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                    )
                )),
                source_authority_tier: 1,
                extraction_method: Some("sparql_entity_search".into()),
                citation_chain: vec![],
                metadata: Default::default(),
            },
            metadata: Default::default(),
        });
    }
    Ok(items)
}

fn short_id_from_uri(uri: &str) -> String {
    uri.rsplit('/').next().unwrap_or(uri).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_quotes_and_backslashes() {
        assert_eq!(escape_sparql_string(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    #[test]
    fn short_id_extracts_q_number() {
        assert_eq!(
            short_id_from_uri("http://www.wikidata.org/entity/Q90"),
            "Q90"
        );
        assert_eq!(short_id_from_uri("plain"), "plain");
    }

    #[test]
    fn parse_bindings_builds_items_with_axes() {
        let body = serde_json::json!({
            "head": { "vars": ["item", "itemLabel", "itemDescription", "sitelinks"] },
            "results": {
                "bindings": [
                    {
                        "item": { "type": "uri", "value": "http://www.wikidata.org/entity/Q90" },
                        "itemLabel": { "type": "literal", "value": "Paris" },
                        "itemDescription": { "type": "literal", "value": "capital of France" },
                        "sitelinks": { "type": "literal", "value": "412" }
                    }
                ]
            }
        });
        let items = parse_bindings(&body, "q0").unwrap();
        assert_eq!(items.len(), 1);
        let it = &items[0];
        assert_eq!(it.source_id, "Q90");
        assert!(it.content.text.as_deref().unwrap().contains("Paris"));
        assert_eq!(it.retrieval_context.rap_step, "primary");
        assert_eq!(it.knowledge_axes.source_authority_tier, 1);
    }

    #[test]
    fn parse_bindings_returns_empty_on_no_results() {
        let body = serde_json::json!({
            "head": { "vars": [] },
            "results": { "bindings": [] }
        });
        let items = parse_bindings(&body, "q0").unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn parse_bindings_errors_on_malformed_body() {
        let body = serde_json::json!({"unexpected": "shape"});
        assert!(parse_bindings(&body, "q0").is_err());
    }

    #[test]
    fn meta_entity_filter_catches_wikimedia_list_articles() {
        let body = serde_json::json!({
            "head": { "vars": ["item", "itemLabel", "itemDescription", "sitelinks"] },
            "results": {
                "bindings": [
                    {
                        "item": { "type": "uri", "value": "http://www.wikidata.org/entity/Q2743079" },
                        "itemLabel": { "type": "literal", "value": "list of capitals of France" },
                        "itemDescription": { "type": "literal", "value": "Wikimedia list article" },
                        "sitelinks": { "type": "literal", "value": "7" }
                    },
                    {
                        "item": { "type": "uri", "value": "http://www.wikidata.org/entity/Q90" },
                        "itemLabel": { "type": "literal", "value": "Paris" },
                        "itemDescription": { "type": "literal", "value": "capital of France" },
                        "sitelinks": { "type": "literal", "value": "412" }
                    }
                ]
            }
        });
        let raw = parse_bindings(&body, "q0").unwrap();
        // Pre-filter both are present.
        assert_eq!(raw.len(), 2);
        let filtered: Vec<_> = raw.into_iter().filter(|it| !is_meta_entity(it)).collect();
        // Post-filter the list-article is gone.
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].source_id, "Q90");
    }

    #[test]
    fn match_property_path_recognizes_capital_of_subject() {
        assert_eq!(match_property_path("capital of france"), Some(("P36", "france")));
        assert_eq!(
            match_property_path("the capital of germany"),
            Some(("P36", "germany"))
        );
    }

    #[test]
    fn match_property_path_recognizes_currency_and_language() {
        assert_eq!(
            match_property_path("currency of japan"),
            Some(("P38", "japan"))
        );
        assert_eq!(
            match_property_path("language of brazil"),
            Some(("P37", "brazil"))
        );
        assert_eq!(
            match_property_path("official language of belgium"),
            Some(("P37", "belgium"))
        );
    }

    #[test]
    fn match_property_path_ignores_unmatched() {
        assert!(match_property_path("paris").is_none());
        assert!(match_property_path("eiffel tower").is_none());
        assert!(match_property_path("").is_none());
    }

    #[test]
    fn property_name_covers_registry() {
        assert_eq!(property_name("P36"), "capital");
        assert_eq!(property_name("P38"), "currency");
        assert_eq!(property_name("P37"), "official language");
        // Unknown returns generic fallback.
        assert_eq!(property_name("P99999"), "property");
    }

    #[test]
    fn parse_property_path_binding_builds_relational_item() {
        let body = serde_json::json!({
            "head": { "vars": ["subject", "subjectLabel", "value", "valueLabel", "valueDescription", "sitelinks"] },
            "results": {
                "bindings": [
                    {
                        "subject": { "type": "uri", "value": "http://www.wikidata.org/entity/Q142" },
                        "subjectLabel": { "type": "literal", "value": "France" },
                        "value": { "type": "uri", "value": "http://www.wikidata.org/entity/Q90" },
                        "valueLabel": { "type": "literal", "value": "Paris" },
                        "valueDescription": { "type": "literal", "value": "capital and most populous city in France" },
                        "sitelinks": { "type": "literal", "value": "412" }
                    }
                ]
            }
        });
        let items = parse_property_path_bindings(&body, "q0", "P36", "France").unwrap();
        assert_eq!(items.len(), 1);
        let it = &items[0];
        assert_eq!(it.source_id, "Q90");
        let text = it.content.text.as_deref().unwrap();
        // Should be a complete proposition mentioning subject + property + value.
        assert!(text.contains("France"));
        assert!(text.contains("Paris"));
        assert!(text.contains("capital"));
        assert_eq!(it.retrieval_context.rap_step, "property_path");
        assert_eq!(
            it.retrieval_context.hop_path.as_ref().unwrap(),
            &vec!["France".to_string(), "P36".to_string()]
        );
    }

    /// Live-network variant of the property-path retrieval. Hits
    /// Wikidata for real and asserts Paris/Q90 comes back. Skipped
    /// by default — opt in with
    ///   cargo test -p jouleclaw-execute -- --ignored wikidata_property_path_live
    #[tokio::test]
    #[ignore]
    async fn wikidata_property_path_live_capital_of_france() {
        let r = WikidataRetriever::new().unwrap();
        let sub = jouleclaw_schema::SubQuery {
            sub_id: "q0".into(),
            text: "capital of France".into(),
            depends_on: vec![],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["wikidata".into()],
            priority: 1.0,
            rap_id: "wikidata_sparql_v1".into(),
        };
        let items = r
            .call("wikidata_property_path", &sub, &Default::default())
            .await
            .expect("WDQS reachable");
        assert!(!items.is_empty(), "expected at least one P36 result");
        assert!(
            items.iter().any(|i| i.source_id == "Q90"),
            "expected Paris (Q90) in {:?}",
            items.iter().map(|i| &i.source_id).collect::<Vec<_>>()
        );
        eprintln!(
            "  → got: {:?}",
            items.iter().map(|i| (
                i.source_id.clone(),
                i.content.text.clone().unwrap_or_default()
            )).collect::<Vec<_>>()
        );
    }

    /// Live-network smoke: with the meta-entity filter,
    /// `wikidata_primary` on "capital of France" should now return
    /// zero items (because the only literal match is the list-of
    /// meta-entity), letting the RAP's OnEmpty step fire.
    #[tokio::test]
    #[ignore]
    async fn wikidata_primary_live_filters_meta_for_capital_query() {
        let r = WikidataRetriever::new().unwrap();
        let sub = jouleclaw_schema::SubQuery {
            sub_id: "q0".into(),
            text: "capital of France".into(),
            depends_on: vec![],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["wikidata".into()],
            priority: 1.0,
            rap_id: "wikidata_sparql_v1".into(),
        };
        let items = r
            .call("wikidata_primary", &sub, &Default::default())
            .await
            .expect("WDQS reachable");
        // No real entities are named "capital of France" — only the
        // Wikimedia list article. After filtering, items must be empty.
        for it in &items {
            eprintln!("  unexpected: {:?} — {:?}", it.source_id, it.content.text);
        }
        assert!(
            items.is_empty(),
            "primary should be empty after meta-entity filter"
        );
    }

    /// Live-network test against the real Wikidata endpoint. Ignored
    /// by default; opt in with:
    ///
    ///   cargo test -p jouleclaw-execute -- --ignored wikidata_live
    #[tokio::test]
    #[ignore]
    async fn wikidata_live_smoke() {
        let r = WikidataRetriever::new().unwrap();
        let sub = jouleclaw_schema::SubQuery {
            sub_id: "q0".into(),
            text: "Paris".into(),
            depends_on: vec![],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["wikidata".into()],
            priority: 1.0,
            rap_id: "wikidata_sparql_v1".into(),
        };
        let items = r
            .call("wikidata_primary", &sub, &Default::default())
            .await
            .expect("WDQS reachable");
        assert!(!items.is_empty());
        assert!(items.iter().any(|i| i.source_id == "Q90"));
    }
}
