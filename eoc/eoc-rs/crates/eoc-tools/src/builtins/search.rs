//! `SearchTool` — pluggable web-search wrapper.
//!
//! Four reference backends are stubbed: SerpApi, Bing Web Search,
//! Brave Search, and DuckDuckGo. Each `query()` issues a single HTTP
//! request and normalises results into a flat `[{title, url, snippet}]`
//! list so the model sees the same shape regardless of provider.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::Tool;

/// A pluggable search-engine adapter.
#[async_trait]
pub trait SearchAdapter: Send + Sync {
    /// Issue a search query, return normalised results.
    async fn query(&self, q: &str, n: usize) -> ToolResult<Vec<SearchHit>>;
}

/// Normalised hit shape.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Page title.
    pub title: String,
    /// Page URL.
    pub url: String,
    /// Snippet text.
    pub snippet: String,
}

/// Concrete backend selector.
pub enum SearchBackend {
    /// SerpApi (Google search wrapper).
    SerpApi(SerpApiBackend),
    /// Bing Web Search API.
    Bing(BingBackend),
    /// Brave Search API.
    Brave(BraveBackend),
    /// DuckDuckGo HTML / Instant Answer.
    DuckDuckGo(DuckDuckGoBackend),
}

#[async_trait]
impl SearchAdapter for SearchBackend {
    async fn query(&self, q: &str, n: usize) -> ToolResult<Vec<SearchHit>> {
        match self {
            SearchBackend::SerpApi(b) => b.query(q, n).await,
            SearchBackend::Bing(b) => b.query(q, n).await,
            SearchBackend::Brave(b) => b.query(q, n).await,
            SearchBackend::DuckDuckGo(b) => b.query(q, n).await,
        }
    }
}

/// SerpApi backend.
pub struct SerpApiBackend {
    /// SerpApi key.
    pub api_key: String,
    /// Base URL (override for tests / mocks).
    pub endpoint: String,
    /// Shared client.
    pub client: reqwest::Client,
}
impl SerpApiBackend {
    /// Construct.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: "https://serpapi.com/search".into(),
            client: reqwest::Client::new(),
        }
    }
}
#[async_trait]
impl SearchAdapter for SerpApiBackend {
    async fn query(&self, q: &str, n: usize) -> ToolResult<Vec<SearchHit>> {
        let v: Value = self
            .client
            .get(&self.endpoint)
            .query(&[
                ("q", q),
                ("api_key", self.api_key.as_str()),
                ("num", &n.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;
        Ok(v.get("organic_results")
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .take(n)
                    .map(|h| SearchHit {
                        title: h
                            .get("title")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .into(),
                        url: h.get("link").and_then(|v| v.as_str()).unwrap_or("").into(),
                        snippet: h
                            .get("snippet")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .into(),
                    })
                    .collect()
            })
            .unwrap_or_default())
    }
}

/// Bing Web Search backend.
pub struct BingBackend {
    /// Subscription key.
    pub api_key: String,
    /// Endpoint URL.
    pub endpoint: String,
    /// Shared client.
    pub client: reqwest::Client,
}
impl BingBackend {
    /// Construct.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: "https://api.bing.microsoft.com/v7.0/search".into(),
            client: reqwest::Client::new(),
        }
    }
}
#[async_trait]
impl SearchAdapter for BingBackend {
    async fn query(&self, q: &str, n: usize) -> ToolResult<Vec<SearchHit>> {
        let v: Value = self
            .client
            .get(&self.endpoint)
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .query(&[("q", q), ("count", &n.to_string())])
            .send()
            .await?
            .json()
            .await?;
        Ok(v.get("webPages")
            .and_then(|w| w.get("value"))
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .take(n)
                    .map(|h| SearchHit {
                        title: h
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .into(),
                        url: h.get("url").and_then(|v| v.as_str()).unwrap_or("").into(),
                        snippet: h
                            .get("snippet")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .into(),
                    })
                    .collect()
            })
            .unwrap_or_default())
    }
}

/// Brave Search backend.
pub struct BraveBackend {
    /// Subscription token.
    pub api_key: String,
    /// Endpoint URL.
    pub endpoint: String,
    /// Shared client.
    pub client: reqwest::Client,
}
impl BraveBackend {
    /// Construct.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: "https://api.search.brave.com/res/v1/web/search".into(),
            client: reqwest::Client::new(),
        }
    }
}
#[async_trait]
impl SearchAdapter for BraveBackend {
    async fn query(&self, q: &str, n: usize) -> ToolResult<Vec<SearchHit>> {
        let v: Value = self
            .client
            .get(&self.endpoint)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .query(&[("q", q), ("count", &n.to_string())])
            .send()
            .await?
            .json()
            .await?;
        Ok(v.get("web")
            .and_then(|w| w.get("results"))
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .take(n)
                    .map(|h| SearchHit {
                        title: h
                            .get("title")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .into(),
                        url: h.get("url").and_then(|v| v.as_str()).unwrap_or("").into(),
                        snippet: h
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .into(),
                    })
                    .collect()
            })
            .unwrap_or_default())
    }
}

/// DuckDuckGo backend (uses the unauthenticated Instant Answer endpoint;
/// useful for tests + low-volume usage. For high-volume usage operators
/// should swap in one of the keyed backends).
pub struct DuckDuckGoBackend {
    /// Endpoint URL.
    pub endpoint: String,
    /// Shared client.
    pub client: reqwest::Client,
}
impl Default for DuckDuckGoBackend {
    fn default() -> Self {
        Self::new()
    }
}
impl DuckDuckGoBackend {
    /// Construct.
    pub fn new() -> Self {
        Self {
            endpoint: "https://api.duckduckgo.com/".into(),
            client: reqwest::Client::new(),
        }
    }
}
#[async_trait]
impl SearchAdapter for DuckDuckGoBackend {
    async fn query(&self, q: &str, n: usize) -> ToolResult<Vec<SearchHit>> {
        let v: Value = self
            .client
            .get(&self.endpoint)
            .query(&[("q", q), ("format", "json"), ("no_html", "1")])
            .send()
            .await?
            .json()
            .await?;
        let mut hits = Vec::new();
        if let Some(arr) = v.get("RelatedTopics").and_then(|x| x.as_array()) {
            for t in arr.iter().take(n) {
                hits.push(SearchHit {
                    title: t
                        .get("Text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    url: t
                        .get("FirstURL")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    snippet: t
                        .get("Text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                });
            }
        }
        Ok(hits)
    }
}

/// The `SearchTool` itself.
pub struct SearchTool {
    /// Concrete backend.
    pub backend: SearchBackend,
    schema: ToolSchema,
}

impl SearchTool {
    /// Construct.
    pub fn new(backend: SearchBackend) -> Self {
        Self {
            backend,
            schema: ToolSchema::new(
                "search",
                "Web search via the configured backend.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer", "minimum": 1, "maximum": 50}
                    },
                    "required": ["query"]
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }
    async fn invoke(&self, args: Value) -> ToolResult<Value> {
        let q = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "search".into(),
                reason: "`query` required".into(),
            })?;
        let n = args
            .get("num_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;
        let hits = self.backend.query(q, n).await?;
        let arr: Vec<Value> = hits
            .into_iter()
            .map(|h| json!({"title": h.title, "url": h.url, "snippet": h.snippet}))
            .collect();
        Ok(json!({"results": arr}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_tool_advertises_schema() {
        let tool = SearchTool::new(SearchBackend::DuckDuckGo(DuckDuckGoBackend::new()));
        assert_eq!(tool.schema().name, "search");
    }
}
