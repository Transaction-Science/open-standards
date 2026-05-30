//! JouleClaw agent memory layer — the substrate version of what the field
//! is converging on around CoALA + MCP.
//!
//! Implements the four-type taxonomy from CoALA (Sumers et al. 2023,
//! itself adapted from Tulving 1972) on the JouleClaw substrate:
//!
//! - **Working** — in-flight request context (lives at L0 Cache).
//! - **Episodic** — past instances, sessions, observations. Time-indexed.
//! - **Semantic** — abstracted, generalised facts. The output of
//!   consolidation.
//! - **Procedural** — resolved sub-tasks compiled to deterministic
//!   procedures. The `jouleclaw-skill` layer.
//!
//! Every captured [`MemoryFact`] is **content-addressed** via blake3 over
//! a canonical byte form (deterministic across runs and machines), so the
//! same fact captured twice shares an id and can be deduplicated.
//! Facts carry a `source_trust` tier (peer of `jouleclaw-fresh`'s
//! `TrustTable`) and optional `valid_from / valid_to` temporal windows
//! (peer of Zep/Graphiti). The field shape is ready for signed-receipt
//! emission via `jouleclaw-prov` and for direct exposure as MCP tools
//! through `jouleclaw-mcp`'s `joule-mcp@1` CBOR profile.
//!
//! ## Scope (v1, honest)
//!
//! This crate ships the **mechanism**: typed facts, content-addressing,
//! the [`MemoryStore`] trait, a deterministic [`InMemoryStore`] reference
//! impl with BM25 recall. The taxonomy and the canonicalisation are the
//! load-bearing pieces; everything else slots in as a follow-up:
//!
//! - Vector embeddings (Matryoshka via `jouleclaw-mrl`) — pluggable
//!   `Recaller` trait, BM25 default. Not in v1.
//! - Temporal-axis recall (`valid_from / valid_to` filtering) — fields
//!   already carry the data; recall filtering is additive.
//! - Fact extraction + entity resolution as a deterministic L1-tier
//!   pipeline — additive.
//! - Cross-encoder reranking — `jouleclaw-rerank` already provides the
//!   trait; recall returns scored hits ready for it.
//! - Consolidation scheduler (episodic → semantic, episodic → procedural
//!   via `jouleclaw-skill`) — additive.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────
// Taxonomy
// ─────────────────────────────────────────────────────────────────────

/// CoALA memory taxonomy (Tulving 1972; Sumers et al. *Cognitive
/// Architectures for Language Agents*, 2023). The wire form is
/// snake-case so it round-trips cleanly through MCP / JSON receipts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// In-flight request context. Volatile by nature — captured here only
    /// when promoted into a longer-lived tier.
    Working,
    /// Past instances: interactions, sessions, observations. Time-indexed.
    Episodic,
    /// Abstracted, generalised facts. Output of consolidation.
    Semantic,
    /// Resolved sub-tasks compiled to deterministic procedures. The
    /// [`jouleclaw-skill`](https://docs.rs/jouleclaw-skill) layer.
    Procedural,
}

impl MemoryType {
    /// Stable wire string — matches the serde representation, useful for
    /// tags and grouping.
    pub fn wire_tag(self) -> &'static str {
        match self {
            MemoryType::Working => "working",
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Procedural => "procedural",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Fact
// ─────────────────────────────────────────────────────────────────────

/// Source trust tier. Conforming consumers derive this from
/// `jouleclaw-fresh::TrustTable` (or any equivalent provenance ladder).
/// `0` is the floor (anonymous user input); `10` is the ceiling
/// (cryptographically attested authoritative source).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrustTier(pub u8);

impl TrustTier {
    /// Default tier for unannotated captures — neither hearsay nor
    /// authoritative.
    pub const DEFAULT: TrustTier = TrustTier(5);
}

impl Default for TrustTier {
    fn default() -> Self {
        TrustTier::DEFAULT
    }
}

/// A captured memory fact, content-addressed and tier-tagged.
///
/// The `id` is a blake3 hash over the **canonical bytes** of
/// `(kind, text, valid_from, valid_to, metadata)`. `created_at` is
/// excluded from the address so the same fact captured twice at
/// different times shares an id (the store can dedupe).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFact {
    /// Content-address: 64-hex-char blake3.
    pub id: String,
    /// The captured text.
    pub text: String,
    /// Caller-supplied tags (people, topics, source). `BTreeMap` so the
    /// content-address is deterministic across runs and platforms.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// CoALA memory type.
    pub kind: MemoryType,
    /// Wall-clock capture time, unix seconds.
    pub created_at: u64,
    /// Source trust tier.
    #[serde(default)]
    pub source_trust: TrustTier,
    /// Validity window — when the fact was/will be true. v1 lets the
    /// caller set these; recall-side temporal filtering is a follow-up.
    #[serde(default)]
    pub valid_from: Option<u64>,
    #[serde(default)]
    pub valid_to: Option<u64>,
}

impl MemoryFact {
    /// Compute the canonical content-address for `(kind, text, valid_*,
    /// metadata)`. `created_at` is excluded so duplicate captures
    /// collapse onto one id.
    pub fn content_address(
        text: &str,
        metadata: &BTreeMap<String, String>,
        kind: MemoryType,
        valid_from: Option<u64>,
        valid_to: Option<u64>,
    ) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"jouleclaw-memory/v1\n");
        hasher.update(kind.wire_tag().as_bytes());
        hasher.update(b"\n");
        hasher.update(text.as_bytes());
        hasher.update(b"\n");
        // Optional fields use a stable sentinel for "none" so the
        // canonical form never collides with a real value.
        write_optional_u64(&mut hasher, "valid_from", valid_from);
        write_optional_u64(&mut hasher, "valid_to", valid_to);
        // BTreeMap iterates in key order — deterministic.
        for (k, v) in metadata {
            hasher.update(b"meta:");
            hasher.update(k.as_bytes());
            hasher.update(b"=");
            hasher.update(v.as_bytes());
            hasher.update(b"\n");
        }
        hasher.finalize().to_hex().to_string()
    }
}

fn write_optional_u64(h: &mut blake3::Hasher, label: &str, v: Option<u64>) {
    h.update(label.as_bytes());
    h.update(b":");
    match v {
        Some(n) => {
            h.update(n.to_string().as_bytes());
        }
        None => {
            h.update(b"none");
        }
    }
    h.update(b"\n");
}

// ─────────────────────────────────────────────────────────────────────
// Options + stats + hits
// ─────────────────────────────────────────────────────────────────────

/// Options for [`MemoryStore::capture`]. All fields default to the
/// uncharacterised floor; the caller annotates what they know.
#[derive(Debug, Clone, Default)]
pub struct CaptureOptions {
    /// CoALA type. Default [`MemoryType::Episodic`] — most captures are
    /// instance-specific by nature.
    pub kind: Option<MemoryType>,
    /// Caller-supplied tags. Use stable keys (`person`, `topic`,
    /// `source`, `tag`) so cross-tool recall works.
    pub metadata: BTreeMap<String, String>,
    /// Source trust tier. Default [`TrustTier::DEFAULT`] (5).
    pub source_trust: Option<TrustTier>,
    /// Validity window.
    pub valid_from: Option<u64>,
    pub valid_to: Option<u64>,
}

/// Options for [`MemoryStore::recall`].
#[derive(Debug, Clone, Default)]
pub struct RecallOptions {
    /// Maximum number of hits to return. Default 10.
    pub k: Option<usize>,
    /// Filter to specific CoALA types. Empty = any.
    pub kinds: Vec<MemoryType>,
    /// Minimum trust tier required.
    pub min_trust: Option<TrustTier>,
    /// Only return facts whose validity window covers `as_of` (unix
    /// seconds). A fact is considered valid when
    /// `valid_from <= as_of <= valid_to`, treating either bound's
    /// absence as &ldquo;open&rdquo; (no constraint on that side). Facts
    /// with no validity window at all (both `valid_*` `None`) are
    /// always returned — they encode timeless statements rather than
    /// temporal facts. The Zep/Graphiti pattern, applied without
    /// re-embedding history.
    pub as_of: Option<u64>,
}

/// A single recall hit — fact plus scoring.
#[derive(Debug, Clone)]
pub struct RecallHit {
    pub fact: MemoryFact,
    pub score: f32,
}

/// Aggregate counters for [`MemoryStore::stats`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total: usize,
    pub by_type: BTreeMap<String, usize>,
    pub bytes_text: u64,
}

// ─────────────────────────────────────────────────────────────────────
// Store trait
// ─────────────────────────────────────────────────────────────────────

/// The memory store interface.
///
/// Conforming implementations may persist anywhere (in-memory, pg_vector,
/// content-addressed object store) so long as `capture` is idempotent
/// per content-address and `get` is lookup-energy.
pub trait MemoryStore: Send {
    /// Capture a fact. Returns the stored fact (with its content-address).
    /// Idempotent: capturing the same `(kind, text, valid_*, metadata)`
    /// twice updates `created_at` but does not duplicate the entry.
    fn capture(
        &mut self,
        text: &str,
        opts: CaptureOptions,
        now_secs: u64,
    ) -> MemoryFact;
    /// Recall facts by semantic / lexical similarity to `query`.
    fn recall(&self, query: &str, opts: RecallOptions) -> Vec<RecallHit>;
    /// Return the `limit` most recently captured facts, newest first.
    fn recent(&self, limit: usize) -> Vec<MemoryFact>;
    /// Aggregate counters.
    fn stats(&self) -> MemoryStats;
    /// Lookup a fact by its content-address. Returns `None` if not found.
    fn get(&self, id: &str) -> Option<MemoryFact>;
    /// Number of facts stored.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─────────────────────────────────────────────────────────────────────
// In-memory reference store (BM25 recall)
// ─────────────────────────────────────────────────────────────────────

/// In-memory reference store with deterministic BM25 recall over text.
///
/// BM25 is the field's lexical baseline; vector embeddings (Matryoshka
/// via `jouleclaw-mrl`) slot in as a future `Recaller` trait without
/// changing this surface.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    facts: Vec<MemoryFact>,
    /// id → index into `facts`.
    by_id: HashMap<String, usize>,
    /// Per-document term-frequency map.
    doc_terms: Vec<HashMap<String, u32>>,
    /// Document length (in tokens).
    doc_lens: Vec<u32>,
    /// Inverted index: term → set of doc indices containing it.
    inverted: HashMap<String, Vec<usize>>,
    /// Cached average doc length for BM25.
    avg_dl: f32,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn tokenize(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let lower = text.to_lowercase();
        for tok in lower.split(|c: char| !c.is_alphanumeric()) {
            if tok.is_empty() {
                continue;
            }
            out.push(tok.to_string());
        }
        out
    }

    fn recompute_avg_dl(&mut self) {
        if self.doc_lens.is_empty() {
            self.avg_dl = 0.0;
            return;
        }
        let sum: u64 = self.doc_lens.iter().map(|&n| n as u64).sum();
        self.avg_dl = sum as f32 / self.doc_lens.len() as f32;
    }
}

/// BM25 hyper-parameters — the standard defaults.
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Is `fact` valid at the optional `as_of` timestamp?
///
/// - `as_of = None` → no temporal filter applied (returns true).
/// - Both `valid_from` and `valid_to` are `None` on the fact → the fact
///   is timeless, always valid.
/// - `valid_from` set, `valid_to` None → valid from that point onward.
/// - `valid_from` None, `valid_to` set → valid up to that point.
/// - Both set → valid within the closed window `[from, to]`.
fn fact_valid_at(fact: &MemoryFact, as_of: Option<u64>) -> bool {
    let Some(t) = as_of else {
        return true;
    };
    if fact.valid_from.is_none() && fact.valid_to.is_none() {
        return true;
    }
    if let Some(from) = fact.valid_from {
        if t < from {
            return false;
        }
    }
    if let Some(to) = fact.valid_to {
        if t > to {
            return false;
        }
    }
    true
}

impl MemoryStore for InMemoryStore {
    fn capture(
        &mut self,
        text: &str,
        opts: CaptureOptions,
        now_secs: u64,
    ) -> MemoryFact {
        let kind = opts.kind.unwrap_or(MemoryType::Episodic);
        let id = MemoryFact::content_address(
            text,
            &opts.metadata,
            kind,
            opts.valid_from,
            opts.valid_to,
        );
        // Idempotent: if we've seen this content-address, update
        // `created_at` and return the stored fact.
        if let Some(&idx) = self.by_id.get(&id) {
            self.facts[idx].created_at = now_secs;
            return self.facts[idx].clone();
        }
        let fact = MemoryFact {
            id: id.clone(),
            text: text.to_string(),
            metadata: opts.metadata,
            kind,
            created_at: now_secs,
            source_trust: opts.source_trust.unwrap_or_default(),
            valid_from: opts.valid_from,
            valid_to: opts.valid_to,
        };
        let idx = self.facts.len();
        // Build the doc representation.
        let tokens = Self::tokenize(&fact.text);
        let mut tf: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        for t in &tokens {
            *tf.entry(t.clone()).or_default() += 1;
        }
        for term in tf.keys() {
            self.inverted
                .entry(term.clone())
                .or_default()
                .push(idx);
        }
        self.doc_lens.push(tokens.len() as u32);
        self.doc_terms.push(tf);
        self.facts.push(fact.clone());
        self.by_id.insert(id, idx);
        self.recompute_avg_dl();
        fact
    }

    fn recall(&self, query: &str, opts: RecallOptions) -> Vec<RecallHit> {
        let k = opts.k.unwrap_or(10).max(1);
        if self.facts.is_empty() {
            return Vec::new();
        }
        let q_tokens = Self::tokenize(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        let n = self.facts.len() as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();
        for term in &q_tokens {
            let Some(postings) = self.inverted.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            // BM25 IDF with the +1 smoothing variant.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &doc_idx in postings {
                let tf = self.doc_terms[doc_idx]
                    .get(term)
                    .copied()
                    .unwrap_or(0) as f32;
                let dl = self.doc_lens[doc_idx] as f32;
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / self.avg_dl.max(1.0)));
                let s = idf * ((tf * (BM25_K1 + 1.0)) / denom.max(f32::MIN_POSITIVE));
                *scores.entry(doc_idx).or_insert(0.0) += s;
            }
        }
        let allow_kinds: Option<std::collections::HashSet<MemoryType>> =
            if opts.kinds.is_empty() {
                None
            } else {
                Some(opts.kinds.iter().copied().collect())
            };
        let min_trust = opts.min_trust.map(|t| t.0).unwrap_or(0);
        let as_of = opts.as_of;
        let mut hits: Vec<RecallHit> = scores
            .into_iter()
            .filter_map(|(doc_idx, score)| {
                let fact = &self.facts[doc_idx];
                if let Some(allow) = &allow_kinds {
                    if !allow.contains(&fact.kind) {
                        return None;
                    }
                }
                if fact.source_trust.0 < min_trust {
                    return None;
                }
                if !fact_valid_at(fact, as_of) {
                    return None;
                }
                Some(RecallHit {
                    fact: fact.clone(),
                    score,
                })
            })
            .collect();
        // Deterministic ordering: by score desc, then by id asc as a
        // tie-breaker so equal-score recalls don't flicker between runs.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.fact.id.cmp(&b.fact.id))
        });
        hits.truncate(k);
        hits
    }

    fn recent(&self, limit: usize) -> Vec<MemoryFact> {
        let mut idxs: Vec<usize> = (0..self.facts.len()).collect();
        // Newest first — `created_at` desc, then insertion order desc
        // as a tie-breaker.
        idxs.sort_by(|&a, &b| {
            self.facts[b]
                .created_at
                .cmp(&self.facts[a].created_at)
                .then_with(|| b.cmp(&a))
        });
        idxs.truncate(limit);
        idxs.into_iter().map(|i| self.facts[i].clone()).collect()
    }

    fn stats(&self) -> MemoryStats {
        let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
        let mut bytes_text: u64 = 0;
        for f in &self.facts {
            *by_type.entry(f.kind.wire_tag().to_string()).or_default() += 1;
            bytes_text += f.text.len() as u64;
        }
        MemoryStats {
            total: self.facts.len(),
            by_type,
            bytes_text,
        }
    }

    fn get(&self, id: &str) -> Option<MemoryFact> {
        self.by_id.get(id).map(|&i| self.facts[i].clone())
    }

    fn len(&self) -> usize {
        self.facts.len()
    }
}

/// Shared store handle, suitable for use behind an MCP server.
pub type SharedMemoryStore<S> = Arc<Mutex<S>>;

/// A fresh shared [`InMemoryStore`].
pub fn shared_in_memory() -> SharedMemoryStore<InMemoryStore> {
    Arc::new(Mutex::new(InMemoryStore::new()))
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn content_address_is_deterministic_and_excludes_created_at() {
        // Identical content → identical address, regardless of when.
        let a = MemoryFact::content_address("hello", &meta(&[]), MemoryType::Episodic, None, None);
        let b = MemoryFact::content_address("hello", &meta(&[]), MemoryType::Episodic, None, None);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64); // blake3 hex
    }

    #[test]
    fn content_address_separates_by_kind_and_temporal_window() {
        let base = MemoryFact::content_address("x", &meta(&[]), MemoryType::Episodic, None, None);
        let by_kind = MemoryFact::content_address("x", &meta(&[]), MemoryType::Semantic, None, None);
        let by_valid = MemoryFact::content_address("x", &meta(&[]), MemoryType::Episodic, Some(1), None);
        assert_ne!(base, by_kind);
        assert_ne!(base, by_valid);
    }

    #[test]
    fn capture_returns_fact_with_content_address_id() {
        let mut store = InMemoryStore::new();
        let fact = store.capture(
            "Sarah is considering leaving consulting",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                metadata: meta(&[("person", "Sarah"), ("topic", "career")]),
                ..Default::default()
            },
            1_700_000_000,
        );
        assert_eq!(fact.id.len(), 64);
        assert_eq!(fact.kind, MemoryType::Episodic);
        assert_eq!(fact.created_at, 1_700_000_000);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn capture_is_idempotent_on_content_address() {
        let mut store = InMemoryStore::new();
        let a = store.capture("the same thought", CaptureOptions::default(), 100);
        let b = store.capture("the same thought", CaptureOptions::default(), 200);
        assert_eq!(a.id, b.id);
        assert_eq!(store.len(), 1, "no duplicate entry");
        // created_at refreshes to the latest capture.
        assert_eq!(store.get(&a.id).unwrap().created_at, 200);
    }

    #[test]
    fn recall_finds_relevant_facts_by_lexical_overlap() {
        let mut store = InMemoryStore::new();
        store.capture(
            "Sarah is considering leaving consulting to start her own firm",
            CaptureOptions {
                metadata: meta(&[("person", "Sarah")]),
                ..Default::default()
            },
            1,
        );
        store.capture(
            "Bought milk and eggs at the store on Tuesday",
            CaptureOptions::default(),
            2,
        );
        store.capture(
            "Sarah said she has been unhappy since the reorg last month",
            CaptureOptions {
                metadata: meta(&[("person", "Sarah")]),
                ..Default::default()
            },
            3,
        );

        let hits = store.recall("Sarah career", RecallOptions::default());
        assert!(!hits.is_empty());
        // The two Sarah-tagged facts should outrank the unrelated one.
        assert!(hits[0].fact.text.contains("Sarah"));
        assert!(hits.iter().all(|h| !h.fact.text.starts_with("Bought milk")));
    }

    #[test]
    fn recall_respects_kind_filter() {
        let mut store = InMemoryStore::new();
        store.capture(
            "alpha",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                ..Default::default()
            },
            1,
        );
        store.capture(
            "alpha",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                metadata: meta(&[("note", "abstracted")]),
                ..Default::default()
            },
            2,
        );
        let only_semantic = store.recall(
            "alpha",
            RecallOptions {
                kinds: vec![MemoryType::Semantic],
                ..Default::default()
            },
        );
        assert_eq!(only_semantic.len(), 1);
        assert_eq!(only_semantic[0].fact.kind, MemoryType::Semantic);
    }

    #[test]
    fn recall_respects_min_trust() {
        let mut store = InMemoryStore::new();
        store.capture(
            "hearsay",
            CaptureOptions {
                source_trust: Some(TrustTier(2)),
                ..Default::default()
            },
            1,
        );
        store.capture(
            "primary evidence",
            CaptureOptions {
                source_trust: Some(TrustTier(9)),
                ..Default::default()
            },
            2,
        );
        let trusted = store.recall(
            "hearsay primary",
            RecallOptions {
                min_trust: Some(TrustTier(5)),
                ..Default::default()
            },
        );
        assert_eq!(trusted.len(), 1);
        assert_eq!(trusted[0].fact.text, "primary evidence");
    }

    #[test]
    fn recent_returns_newest_first() {
        let mut store = InMemoryStore::new();
        store.capture("oldest", CaptureOptions::default(), 10);
        store.capture("middle", CaptureOptions::default(), 20);
        store.capture("newest", CaptureOptions::default(), 30);
        let r = store.recent(2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].text, "newest");
        assert_eq!(r[1].text, "middle");
    }

    #[test]
    fn stats_counts_by_type_and_bytes() {
        let mut store = InMemoryStore::new();
        store.capture(
            "a",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                ..Default::default()
            },
            1,
        );
        store.capture(
            "bb",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                ..Default::default()
            },
            2,
        );
        store.capture(
            "ccc",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                ..Default::default()
            },
            3,
        );
        let s = store.stats();
        assert_eq!(s.total, 3);
        assert_eq!(s.by_type.get("episodic").copied(), Some(1));
        assert_eq!(s.by_type.get("semantic").copied(), Some(2));
        assert_eq!(s.bytes_text, 1 + 2 + 3);
    }

    #[test]
    fn get_by_id_returns_stored_fact() {
        let mut store = InMemoryStore::new();
        let fact = store.capture("findable", CaptureOptions::default(), 1);
        let again = store.get(&fact.id).unwrap();
        assert_eq!(again, fact);
        assert!(store.get("00".repeat(32).as_str()).is_none());
    }

    #[test]
    fn fact_roundtrips_through_json() {
        let mut store = InMemoryStore::new();
        let f = store.capture(
            "round-trip me",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                metadata: meta(&[("topic", "memory")]),
                source_trust: Some(TrustTier(7)),
                valid_from: Some(1_000),
                valid_to: Some(2_000),
            },
            100,
        );
        let s = serde_json::to_string(&f).unwrap();
        let back: MemoryFact = serde_json::from_str(&s).unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn metadata_changes_content_address() {
        let id_no_meta = MemoryFact::content_address(
            "Sarah is considering leaving",
            &meta(&[]),
            MemoryType::Episodic,
            None,
            None,
        );
        let id_with_meta = MemoryFact::content_address(
            "Sarah is considering leaving",
            &meta(&[("person", "Sarah")]),
            MemoryType::Episodic,
            None,
            None,
        );
        assert_ne!(id_no_meta, id_with_meta);
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let mut store = InMemoryStore::new();
        store.capture("something", CaptureOptions::default(), 1);
        let h = store.recall("", RecallOptions::default());
        assert!(h.is_empty());
    }

    #[test]
    fn recall_respects_temporal_window_as_of() {
        let mut store = InMemoryStore::new();
        // A fact known true during Q1 only.
        store.capture(
            "Sarah works at Acme",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                metadata: meta(&[("person", "Sarah")]),
                valid_from: Some(100),
                valid_to: Some(200),
                ..Default::default()
            },
            1,
        );
        // A timeless fact (no window).
        store.capture(
            "Sarah is a person",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                metadata: meta(&[("person", "Sarah")]),
                ..Default::default()
            },
            2,
        );
        // A fact valid from Q2 onward.
        store.capture(
            "Sarah works at Globex",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                metadata: meta(&[("person", "Sarah")]),
                valid_from: Some(201),
                ..Default::default()
            },
            3,
        );

        // Inside the Acme window: Acme fact + the timeless one; no Globex.
        let inside = store.recall(
            "Sarah",
            RecallOptions {
                as_of: Some(150),
                ..Default::default()
            },
        );
        let texts: Vec<_> = inside.iter().map(|h| h.fact.text.as_str()).collect();
        assert!(texts.contains(&"Sarah works at Acme"));
        assert!(texts.contains(&"Sarah is a person"));
        assert!(!texts.contains(&"Sarah works at Globex"));

        // After the Acme window: Globex + timeless; no Acme.
        let after = store.recall(
            "Sarah",
            RecallOptions {
                as_of: Some(250),
                ..Default::default()
            },
        );
        let texts: Vec<_> = after.iter().map(|h| h.fact.text.as_str()).collect();
        assert!(texts.contains(&"Sarah works at Globex"));
        assert!(texts.contains(&"Sarah is a person"));
        assert!(!texts.contains(&"Sarah works at Acme"));

        // No `as_of` → no temporal filter.
        let all = store.recall("Sarah", RecallOptions::default());
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn k_caps_recall_results() {
        let mut store = InMemoryStore::new();
        for i in 0..20 {
            store.capture(&format!("alpha entry {i}"), CaptureOptions::default(), i as u64);
        }
        let h = store.recall(
            "alpha",
            RecallOptions {
                k: Some(5),
                ..Default::default()
            },
        );
        assert_eq!(h.len(), 5);
    }
}
