//! # jouleclaw-fresh
//!
//! The "fresh retrieval with provenance" stage of the JouleClaw
//! cascade. Sits between `L2:Embed` and `L3:Model` — when the local
//! index can't close the query, JouleClaw consults the live world
//! before it consults its frozen weights.
//!
//! ## Why this exists
//!
//! Frozen model weights are **less trustworthy than current world
//! state**. Spending joules to synthesise an answer the live web
//! could return verbatim — with a verifiable URL — is the kind of
//! waste this standard exists to eliminate.
//!
//! ## What's in this crate
//!
//! - [`SearchProvider`] — the adapter trait. Brave, Tavily, Exa,
//!   Serper, and any future provider implement it.
//! - [`FreshFetch`] — HTTP fetch + clean-text extraction. Returns a
//!   [`RetrievedClaim`] with a fetch timestamp and a BLAKE3 content
//!   hash.
//! - [`TrustTier`] — source-trust ranking. Bootstrap data comes from
//!   the Wikipedia perennial-sources list (machine-readable via
//!   Wikimedia Enterprise's parsed-references endpoint).
//! - [`provenance_envelope`] — wraps a `RetrievedClaim` in a
//!   `jouleclaw-prov::ClaimProvenance` ready to embed in a
//!   cascade receipt. The signature is added downstream by a Smart
//!   Byte envelope sealer.
//!
//! ## What's NOT in this crate
//!
//! - Live HTTP clients. The standard ships traits + the in-memory
//!   `MockTransport`; operators wire `reqwest` / `hyper` / their own
//!   shim against the [`Transport`] trait. Keeps `jouleclaw-fresh`
//!   reqwest-free for embedded / WASM / no-tokio deployments.
//! - Browser automation (JS-rendered pages). Use `chromiumoxide`
//!   downstream and present results back through [`Transport`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use jouleclaw_prov::ClaimProvenance;
use serde::{Deserialize, Serialize};

/// Errors any stage of the fresh-retrieval pipeline can produce.
#[derive(Debug, thiserror::Error)]
pub enum FreshError {
    /// The underlying transport returned an error.
    #[error("transport: {0}")]
    Transport(String),
    /// The search provider returned a malformed response.
    #[error("search response: {0}")]
    SearchResponse(String),
    /// The trust table doesn't recognise the source domain.
    #[error("unknown source domain: {0}")]
    UnknownDomain(String),
    /// I/O failure reading the trust table.
    #[error("trust table io: {0}")]
    TrustIo(#[from] std::io::Error),
    /// JSON parse failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// One search-API hit — the URL pointing at the candidate evidence
/// and the snippet the provider returned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Candidate URL.
    pub url: String,
    /// Title returned by the provider.
    pub title: String,
    /// Short snippet (usually 1–3 lines) — useful for pre-filtering
    /// before paying the fetch tax.
    pub snippet: String,
}

/// Pluggable search-API adapter. Implementors: Brave, Tavily, Exa, Serper.
#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Stable provider identifier — `"brave"`, `"tavily"`, `"exa"`, …
    fn name(&self) -> &'static str;
    /// Execute a query and return the top-k hits.
    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>, FreshError>;
}

/// Pluggable HTTP transport — keeps reqwest out of the crate.
#[async_trait]
pub trait Transport: Send + Sync {
    /// GET a URL, return the body bytes + a recorded Content-Type.
    async fn get(&self, url: &str) -> Result<TransportResponse, FreshError>;
}

/// A response from the transport.
#[derive(Debug, Clone)]
pub struct TransportResponse {
    /// Body bytes as received from the wire.
    pub body: Vec<u8>,
    /// The Content-Type header, when present.
    pub content_type: Option<String>,
}

/// One retrieved claim ready for the cascade to consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedClaim {
    /// Source URL.
    pub url: String,
    /// Extracted clean text (HTML/PDF → plain text).
    pub content: String,
    /// BLAKE3 of the raw body bytes as observed at fetch time.
    pub content_hash: String,
    /// Timestamp of the fetch.
    pub fetched_at: DateTime<Utc>,
    /// Trust tier applied at lookup. Higher = more trustworthy.
    pub trust_tier: u8,
}

/// Source-trust tier table. Bootstrap from Wikipedia's perennial
/// sources list; downstream operators can extend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrustTable {
    /// Map from canonical domain (no scheme, no leading `www.`) →
    /// trust tier 0–10. Higher = more trustworthy.
    pub tiers: std::collections::HashMap<String, u8>,
    /// Tier returned when a domain is not in the table.
    pub default_tier: u8,
}

impl TrustTable {
    /// Construct a table with a uniform default tier and no entries.
    pub fn with_default(default_tier: u8) -> Self {
        Self {
            tiers: std::collections::HashMap::new(),
            default_tier,
        }
    }

    /// Insert (or update) a domain → tier entry. Domain is normalised
    /// (lowercased, leading `www.` stripped).
    pub fn insert(&mut self, domain: impl AsRef<str>, tier: u8) {
        self.tiers.insert(normalize_domain(domain.as_ref()), tier);
    }

    /// Look up the trust tier for a URL. Falls back to `default_tier`
    /// when the domain isn't in the table.
    pub fn tier_for_url(&self, url: &str) -> u8 {
        let host = host_of(url).unwrap_or_default();
        let domain = normalize_domain(&host);
        *self.tiers.get(&domain).unwrap_or(&self.default_tier)
    }

    /// Load from JSON: `{ "default_tier": N, "tiers": { "domain": N, … } }`.
    pub fn from_json(json: &str) -> Result<Self, FreshError> {
        Ok(serde_json::from_str(json)?)
    }
}

/// Normalise a domain for table lookup. Lowercase + strip `www.`.
fn normalize_domain(d: &str) -> String {
    let lc = d.to_ascii_lowercase();
    lc.strip_prefix("www.").unwrap_or(&lc).to_string()
}

/// Extract the host portion of a URL. Returns `None` if the URL is
/// not well-formed enough to find a host between `://` and the next
/// `/`.
fn host_of(url: &str) -> Option<String> {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let host = after_scheme.split('/').next().unwrap_or("");
    let host = host.split('?').next().unwrap_or(host);
    let host = host.split('#').next().unwrap_or(host);
    if host.is_empty() { None } else { Some(host.to_string()) }
}

/// Orchestrator: search-API → top-k URLs → fetch → extract → wrap as
/// [`RetrievedClaim`] with trust tier applied.
///
/// `text_extractor` is left as a free function the caller supplies
/// (so this crate can stay HTML-parser-free). The standard recommends
/// `dom_smoothie` (Readability port) for HTML and `pdf-extract` for
/// PDF, but neither is enforced.
pub struct FreshFetch<'a> {
    /// Search provider.
    pub search: &'a dyn SearchProvider,
    /// Transport for the follow-on fetch.
    pub transport: &'a dyn Transport,
    /// Trust table for source ranking.
    pub trust: &'a TrustTable,
}

impl<'a> FreshFetch<'a> {
    /// Run the full search-then-fetch pipeline.
    ///
    /// `text_extractor` converts raw body bytes (with optional
    /// content-type hint) to clean text. Caller's choice of library.
    pub async fn run(
        &self,
        query: &str,
        top_k: usize,
        text_extractor: impl Fn(&[u8], Option<&str>) -> String,
    ) -> Result<Vec<RetrievedClaim>, FreshError> {
        let hits = self.search.search(query, top_k).await?;
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            let resp = self.transport.get(&hit.url).await?;
            let content = text_extractor(&resp.body, resp.content_type.as_deref());
            let content_hash = blake3::hash(&resp.body).to_hex().to_string();
            let trust_tier = self.trust.tier_for_url(&hit.url);
            out.push(RetrievedClaim {
                url: hit.url,
                content,
                content_hash,
                fetched_at: Utc::now(),
                trust_tier,
            });
        }
        Ok(out)
    }
}

/// Wrap a [`RetrievedClaim`] in a [`ClaimProvenance`] ready to embed
/// in a `jouleclaw-prov::Receipt`. The Smart Byte envelope signature
/// is added downstream by the sealer.
pub fn provenance_envelope(claim: &RetrievedClaim) -> ClaimProvenance {
    ClaimProvenance {
        source: claim.url.clone(),
        content_hash: claim.content_hash.clone(),
        fetched_at: claim.fetched_at.to_rfc3339(),
        trust_tier: claim.trust_tier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSearch;
    #[async_trait]
    impl SearchProvider for MockSearch {
        fn name(&self) -> &'static str { "mock" }
        async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>, FreshError> {
            Ok((0..top_k).map(|i| SearchHit {
                url: format!("https://example.org/{query}/{i}"),
                title: format!("hit {i}"),
                snippet: format!("about {query}"),
            }).collect())
        }
    }

    struct MockTransport;
    #[async_trait]
    impl Transport for MockTransport {
        async fn get(&self, url: &str) -> Result<TransportResponse, FreshError> {
            Ok(TransportResponse {
                body: format!("<html><body>content of {url}</body></html>").into_bytes(),
                content_type: Some("text/html".into()),
            })
        }
    }

    #[test]
    fn trust_table_falls_back_to_default() {
        let mut t = TrustTable::with_default(3);
        t.insert("en.wikipedia.org", 9);
        assert_eq!(t.tier_for_url("https://en.wikipedia.org/wiki/X"), 9);
        assert_eq!(t.tier_for_url("https://random.example.com/page"), 3);
    }

    #[test]
    fn trust_table_normalises_www_prefix() {
        let mut t = TrustTable::with_default(0);
        t.insert("example.com", 7);
        assert_eq!(t.tier_for_url("https://www.example.com/p"), 7);
        assert_eq!(t.tier_for_url("https://example.com/p"), 7);
    }

    #[test]
    fn host_extraction() {
        assert_eq!(host_of("https://en.wikipedia.org/wiki/X").as_deref(), Some("en.wikipedia.org"));
        assert_eq!(host_of("http://example.com").as_deref(), Some("example.com"));
        assert_eq!(host_of("example.com/p").as_deref(), Some("example.com"));
        assert_eq!(host_of("ftp://").as_deref(), None);
    }

    #[tokio::test]
    async fn fresh_fetch_end_to_end() {
        let search = MockSearch;
        let transport = MockTransport;
        let mut trust = TrustTable::with_default(2);
        trust.insert("example.org", 6);
        let f = FreshFetch { search: &search, transport: &transport, trust: &trust };
        let claims = f.run("xyz", 3, |body, _ct| {
            // Toy extractor — just utf-8 decode.
            String::from_utf8_lossy(body).into_owned()
        }).await.expect("ok");
        assert_eq!(claims.len(), 3);
        assert!(claims[0].url.starts_with("https://example.org/xyz/"));
        assert_eq!(claims[0].trust_tier, 6);
        assert_eq!(claims[0].content_hash.len(), 64);
    }

    #[test]
    fn provenance_envelope_carries_fields() {
        let claim = RetrievedClaim {
            url: "https://en.wikipedia.org/wiki/X".into(),
            content: "X is a thing.".into(),
            content_hash: "deadbeef".into(),
            fetched_at: Utc::now(),
            trust_tier: 9,
        };
        let env = provenance_envelope(&claim);
        assert_eq!(env.source, claim.url);
        assert_eq!(env.content_hash, claim.content_hash);
        assert_eq!(env.trust_tier, 9);
    }
}
