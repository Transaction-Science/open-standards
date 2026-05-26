//! On-disk cache for verified answers (spec §7.3, simplified).
//!
//! Stores the entire [`AnswerOrRefusal`] keyed on a hash of the
//! plan's sub-query cache keys. Repeat queries skip every
//! expensive stage (Wikidata HTTP + DeBERTa entailment + …) and
//! return in ~10 ms.
//!
//! Granularity is coarser than the spec's per-sub-query design —
//! we cache the whole pipeline output, not the intermediate
//! (items, entailments) per sub-query — but the UX outcome is the
//! same on repeated identical queries. The narrower granularity
//! (cached items shared across related queries) is a later
//! optimization; this is the floor.
//!
//! Layout on disk:
//!
//!   <cache_dir>/
//!     <key>.json    one envelope per cached query
//!
//! Each envelope carries `cached_at`, the cache key the answer
//! was stored under, and the [`AnswerOrRefusal`] itself.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use jouleclaw_schema::SubQuery;

use crate::provenance::cache_key_for_subquery;
use crate::verified::AnswerOrRefusal;

/// Envelope written to disk per cached entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAnswer {
    pub schema_version: String,
    pub cache_key: String,
    pub cached_at: DateTime<Utc>,
    pub query_text: String,
    pub result: AnswerOrRefusal,
}

/// On-disk cache store. Stateless — every read/write hits the
/// filesystem. For workloads that need in-memory caching, wrap
/// with an LRU on top.
pub struct CacheStore {
    dir: PathBuf,
    max_age: Duration,
}

#[derive(Debug)]
pub enum CacheError {
    Io(String),
    Serialize(String),
    Deserialize(String),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "cache io: {s}"),
            Self::Serialize(s) => write!(f, "cache serialize: {s}"),
            Self::Deserialize(s) => write!(f, "cache deserialize: {s}"),
        }
    }
}

impl std::error::Error for CacheError {}

impl CacheStore {
    /// Open or create a cache rooted at `dir`. Default max-age of
    /// 24 hours; tune with [`Self::with_max_age`].
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, CacheError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)
            .map_err(|e| CacheError::Io(format!("mkdir {}: {e}", dir.display())))?;
        Ok(Self {
            dir,
            max_age: Duration::from_secs(24 * 60 * 60),
        })
    }

    pub fn with_max_age(mut self, age: Duration) -> Self {
        self.max_age = age;
        self
    }

    /// Look up by cache key. Returns `None` for misses or
    /// stale-by-age entries. Returns `Err` for corrupted entries
    /// so the caller can decide whether to delete or skip.
    pub fn get(&self, key: &str) -> Result<Option<CachedAnswer>, CacheError> {
        let path = self.path_for(key);
        if !path.exists() {
            return Ok(None);
        }
        let mut f = fs::File::open(&path)
            .map_err(|e| CacheError::Io(format!("open {}: {e}", path.display())))?;
        let mut s = String::new();
        f.read_to_string(&mut s)
            .map_err(|e| CacheError::Io(format!("read {}: {e}", path.display())))?;
        let env: CachedAnswer =
            serde_json::from_str(&s).map_err(|e| CacheError::Deserialize(e.to_string()))?;
        let age_ms = (Utc::now() - env.cached_at)
            .num_milliseconds()
            .max(0) as u128;
        if age_ms > self.max_age.as_millis() {
            // Stale — treat as miss. Don't delete; the caller may
            // want to inspect or refresh.
            return Ok(None);
        }
        Ok(Some(env))
    }

    /// Store a `CachedAnswer`. Idempotent — overwrites if the key
    /// already exists.
    pub fn put(&self, env: &CachedAnswer) -> Result<(), CacheError> {
        let path = self.path_for(&env.cache_key);
        let json = serde_json::to_string_pretty(env)
            .map_err(|e| CacheError::Serialize(e.to_string()))?;
        // Atomic-ish write: write to a tmp file then rename. Avoids
        // half-written files if the process is killed mid-write.
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, json.as_bytes())
            .map_err(|e| CacheError::Io(format!("write {}: {e}", tmp.display())))?;
        fs::rename(&tmp, &path)
            .map_err(|e| CacheError::Io(format!("rename {}: {e}", path.display())))?;
        Ok(())
    }

    pub fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Compute a stable cache key for an entire plan's sub-query set.
/// Hashes the SHA-256 of each [`SubQuery`] (via
/// [`cache_key_for_subquery`]) plus a schema-version tag and the
/// resolved retrieval endpoints, so a composer-code change OR an
/// endpoint switch invalidates prior cached answers without manual
/// cleanup.
///
/// `version_tag` should encode the pipeline's wire-format
/// commitment — bump it when changing the prompts, the atomizer,
/// or the entailer model.
///
/// `wikidata_endpoint` / `wikipedia_endpoint` are the *resolved*
/// endpoints the pipeline will actually call. They're included in
/// the key so a query against `https://query.wikidata.org/sparql`
/// and the same query against a local Oxigraph mirror don't alias
/// — the mirror may carry a different snapshot of Wikidata than
/// the public endpoint, and the answer cache cannot tell which the
/// cached envelope was produced against.
pub fn cache_key_for_plan(
    sub_queries: &[SubQuery],
    version_tag: &str,
    wikidata_endpoint: Option<&str>,
    wikipedia_endpoint: Option<&str>,
) -> String {
    let mut parts: Vec<String> =
        sub_queries.iter().map(cache_key_for_subquery).collect();
    parts.sort();
    let canonical = serde_json::json!({
        "version": version_tag,
        "sub_queries": parts,
        "wikidata_endpoint": wikidata_endpoint,
        "wikipedia_endpoint": wikipedia_endpoint,
    });
    let bytes = canonical.to_string();
    let mut hasher = Sha256::new();
    hasher.update(bytes.as_bytes());
    let digest = hasher.finalize();
    hex_encode(&digest)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{:02x}", b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_schema::Modality;

    fn sub(sub_id: &str, text: &str) -> SubQuery {
        SubQuery {
            sub_id: sub_id.into(),
            text: text.into(),
            depends_on: vec![],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["wikidata".into()],
            priority: 1.0,
            rap_id: "rap".into(),
        }
    }

    fn refusal_envelope(key: &str, query: &str) -> CachedAnswer {
        use jouleclaw_schema::Refusal;
        use uuid::Uuid;
        CachedAnswer {
            schema_version: "1.0".into(),
            cache_key: key.into(),
            cached_at: Utc::now(),
            query_text: query.into(),
            result: AnswerOrRefusal::Refusal(Refusal {
                schema_version: "2.0".into(),
                refusal_id: Uuid::new_v4(),
                plan_id: Uuid::new_v4(),
                reason_code: "test".into(),
                reason_message: "fixture".into(),
                blocking_violations: vec![],
                emitted_at: Utc::now(),
                metadata: Default::default(),
            }),
        }
    }

    #[test]
    fn key_is_deterministic_for_identical_subqueries() {
        let a = vec![sub("q0", "capital of France"), sub("q1", "currency of Japan")];
        let b = vec![sub("q1", "currency of Japan"), sub("q0", "capital of France")];
        assert_eq!(
            cache_key_for_plan(&a, "v1", None, None),
            cache_key_for_plan(&b, "v1", None, None),
            "sub-query order shouldn't change the key"
        );
    }

    #[test]
    fn version_tag_invalidates_key() {
        let a = vec![sub("q0", "capital of France")];
        assert_ne!(
            cache_key_for_plan(&a, "v1", None, None),
            cache_key_for_plan(&a, "v2", None, None)
        );
    }

    /// Same plan against different endpoints must NOT alias — a local
    /// Oxigraph mirror can carry a different Wikidata snapshot than
    /// `query.wikidata.org`, and the cache cannot tell which one a
    /// stored answer was produced against.
    #[test]
    fn endpoint_invalidates_key() {
        let a = vec![sub("q0", "capital of France")];
        let public = cache_key_for_plan(&a, "v1", None, None);
        let local = cache_key_for_plan(
            &a,
            "v1",
            Some("http://localhost:7878/query"),
            None,
        );
        assert_ne!(public, local, "public vs local Wikidata endpoints must hash differently");

        let wp_changed = cache_key_for_plan(
            &a,
            "v1",
            None,
            Some("http://localhost:8080/wikipedia/api/rest_v1"),
        );
        assert_ne!(public, wp_changed, "Wikipedia endpoint must hash into the key too");
    }

    #[test]
    fn store_and_get_roundtrip() {
        let tmp = tempdir();
        let store = CacheStore::open(&tmp).unwrap();
        let env = refusal_envelope("abcdef", "test query");
        store.put(&env).unwrap();
        let back = store.get("abcdef").unwrap().expect("hit");
        assert_eq!(back.query_text, "test query");
    }

    #[test]
    fn miss_returns_none() {
        let tmp = tempdir();
        let store = CacheStore::open(&tmp).unwrap();
        assert!(store.get("never-stored").unwrap().is_none());
    }

    #[test]
    fn stale_entry_returns_none() {
        let tmp = tempdir();
        let store = CacheStore::open(&tmp).unwrap().with_max_age(Duration::from_secs(0));
        let env = refusal_envelope("abcdef", "test");
        store.put(&env).unwrap();
        // Sleep a moment to ensure age > 0.
        std::thread::sleep(Duration::from_millis(50));
        assert!(store.get("abcdef").unwrap().is_none());
    }

    #[test]
    fn put_is_atomic_no_partial_file() {
        let tmp = tempdir();
        let store = CacheStore::open(&tmp).unwrap();
        let env = refusal_envelope("k1", "q");
        store.put(&env).unwrap();
        // Should never have left a .tmp file behind.
        let leftover = std::fs::read_dir(store.dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("tmp"));
        assert!(!leftover);
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "joule-cache-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
