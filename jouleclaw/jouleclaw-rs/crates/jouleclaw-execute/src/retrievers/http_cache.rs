//! On-disk HTTP-response cache for retriever calls.
//!
//! Sits below the answer-level provenance cache (which is in
//! jouleclaw-compose) and above the HTTP client. Keyed on the request
//! URL + parameters; stores the raw JSON response. Bridges the
//! gap between "user runs the same query twice" (answer cache
//! hits) and "user runs different queries that hit the same
//! underlying entity" (no answer-cache hit, but the SPARQL/REST
//! call would be identical → HTTP cache hits).
//!
//! Useful for users who don't run a local Wikidata mirror — every
//! distinct SPARQL string is fetched once across all queries.
//! For users who DO operate a local mirror, the HTTP layer
//! already serves quickly; this cache is still useful for
//! Wikipedia REST calls and reduces load on the local mirror.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedResponse {
    pub body: String,
    pub cached_at: DateTime<Utc>,
    pub endpoint: String,
}

pub struct HttpCache {
    dir: PathBuf,
    max_age: Duration,
}

#[derive(Debug)]
pub enum HttpCacheError {
    Io(String),
    Serialize(String),
}

impl std::fmt::Display for HttpCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "http cache io: {s}"),
            Self::Serialize(s) => write!(f, "http cache serialize: {s}"),
        }
    }
}

impl std::error::Error for HttpCacheError {}

impl HttpCache {
    /// Open (or create) the cache directory.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, HttpCacheError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)
            .map_err(|e| HttpCacheError::Io(format!("mkdir {}: {e}", dir.display())))?;
        Ok(Self {
            dir,
            max_age: Duration::from_secs(24 * 60 * 60),
        })
    }

    pub fn with_max_age(mut self, age: Duration) -> Self {
        self.max_age = age;
        self
    }

    /// Stable key from (endpoint, request body). Both go into the
    /// hash so the same SPARQL query against different endpoints
    /// (e.g. public vs local mirror) doesn't collide.
    pub fn key(&self, endpoint: &str, request: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(endpoint.as_bytes());
        hasher.update(b"\0");
        hasher.update(request.as_bytes());
        let digest = hasher.finalize();
        let mut out = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write;
            let _ = write!(out, "{:02x}", b);
        }
        out
    }

    /// Returns Some(body) if a fresh cache entry exists.
    pub fn get(&self, key: &str) -> Option<String> {
        let path = self.path_for(key);
        let raw = fs::read_to_string(&path).ok()?;
        let env: CachedResponse = serde_json::from_str(&raw).ok()?;
        let age_ms = (Utc::now() - env.cached_at)
            .num_milliseconds()
            .max(0) as u128;
        if age_ms > self.max_age.as_millis() {
            return None;
        }
        Some(env.body)
    }

    /// Store the response body. Atomic via tmp+rename.
    pub fn put(&self, key: &str, endpoint: &str, body: &str) -> Result<(), HttpCacheError> {
        let env = CachedResponse {
            body: body.to_string(),
            cached_at: Utc::now(),
            endpoint: endpoint.to_string(),
        };
        let json = serde_json::to_string(&env)
            .map_err(|e| HttpCacheError::Serialize(e.to_string()))?;
        let path = self.path_for(key);
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, json.as_bytes())
            .map_err(|e| HttpCacheError::Io(format!("write {}: {e}", tmp.display())))?;
        fs::rename(&tmp, &path)
            .map_err(|e| HttpCacheError::Io(format!("rename {}: {e}", path.display())))?;
        Ok(())
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Convenience helper: per-process default cache shared across
/// retrievers in the same run. Lives under
/// `$HOME/.cache/joule-edge/http/`. Returns None on filesystem
/// errors so callers fall back to direct HTTP transparently.
pub fn default_http_cache() -> Option<HttpCache> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".cache").join("joule-edge").join("http");
    HttpCache::open(dir).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "joule-http-cache-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn key_is_endpoint_request_sensitive() {
        let c = HttpCache::open(tempdir()).unwrap();
        let k1 = c.key("https://a.example", "query=foo");
        let k2 = c.key("https://b.example", "query=foo");
        let k3 = c.key("https://a.example", "query=bar");
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        // Length is sha256 hex.
        assert_eq!(k1.len(), 64);
    }

    #[test]
    fn store_and_retrieve_roundtrip() {
        let c = HttpCache::open(tempdir()).unwrap();
        let key = c.key("ep", "req");
        c.put(&key, "ep", "{\"hello\":\"world\"}").unwrap();
        assert_eq!(c.get(&key), Some(r#"{"hello":"world"}"#.into()));
    }

    #[test]
    fn miss_is_none() {
        let c = HttpCache::open(tempdir()).unwrap();
        assert!(c.get("never-stored").is_none());
    }

    #[test]
    fn stale_entry_is_none() {
        let c = HttpCache::open(tempdir())
            .unwrap()
            .with_max_age(Duration::from_millis(1));
        c.put("k", "ep", "body").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(c.get("k").is_none());
    }
}
