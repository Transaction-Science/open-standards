//! Semantic memory: an entity-relation triple store.
//!
//! Triples follow the classical RDF shape `(subject, predicate,
//! object)` (Berners-Lee 2001) and are stored in three flat indices
//! so lookups by S, P, or O are linear in the number of matches.
//! No external graph dependency is pulled in — the goal is
//! deterministic, embeddable, WASM-clean.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::MemoryResult;
use crate::memory::{Memory, MemoryItem, MemoryKind, MemoryRef};

/// An entity-relation-entity assertion.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    /// Subject identifier.
    pub subject: String,
    /// Predicate / relation name.
    pub predicate: String,
    /// Object identifier or literal.
    pub object: String,
    /// Source-episode id, optional, for provenance back to the
    /// episodic log.
    pub source_episode: Option<String>,
    /// Insertion timestamp in ms (for `Memory::recent`).
    pub timestamp_ms: u64,
}

impl Triple {
    /// Construct a triple at a given timestamp.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            source_episode: None,
            timestamp_ms,
        }
    }

    /// Attach a source-episode hex id (builder).
    #[must_use]
    pub fn with_source(mut self, source_episode: impl Into<String>) -> Self {
        self.source_episode = Some(source_episode.into());
        self
    }

    /// Stable BLAKE3-16 hex id of this triple (deterministic).
    #[must_use]
    pub fn id(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.subject.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(self.predicate.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(self.object.as_bytes());
        let h = hasher.finalize();
        let mut s = String::with_capacity(32);
        for b in &h.as_bytes()[..16] {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Render as `MemoryItem` for injection.
    #[must_use]
    pub fn as_item(&self) -> MemoryItem {
        MemoryItem::new(
            MemoryRef {
                kind: MemoryKind::Semantic,
                id: self.id(),
            },
            format!("({} {} {})", self.subject, self.predicate, self.object),
            self.timestamp_ms,
        )
    }
}

/// In-memory knowledge graph: a set of [`Triple`] plus S/P/O indices.
#[derive(Clone, Debug, Default)]
pub struct SemanticGraph {
    triples: Vec<Triple>,
    by_subject: HashMap<String, Vec<usize>>,
    by_predicate: HashMap<String, Vec<usize>>,
    by_object: HashMap<String, Vec<usize>>,
}

impl SemanticGraph {
    /// Fresh empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assert a triple into the graph. Duplicates (same S/P/O) are
    /// idempotent — the new timestamp wins, but no new row is added.
    pub fn assert(&mut self, triple: Triple) -> String {
        let id = triple.id();
        if let Some(existing) = self.triples.iter_mut().find(|t| t.id() == id) {
            existing.timestamp_ms = triple.timestamp_ms;
            return id;
        }
        let idx = self.triples.len();
        self.by_subject
            .entry(triple.subject.clone())
            .or_default()
            .push(idx);
        self.by_predicate
            .entry(triple.predicate.clone())
            .or_default()
            .push(idx);
        self.by_object
            .entry(triple.object.clone())
            .or_default()
            .push(idx);
        self.triples.push(triple);
        id
    }

    /// All triples with subject `s`.
    #[must_use]
    pub fn subject(&self, s: &str) -> Vec<&Triple> {
        self.by_subject
            .get(s)
            .map(|v| v.iter().filter_map(|&i| self.triples.get(i)).collect())
            .unwrap_or_default()
    }

    /// All triples with predicate `p`.
    #[must_use]
    pub fn predicate(&self, p: &str) -> Vec<&Triple> {
        self.by_predicate
            .get(p)
            .map(|v| v.iter().filter_map(|&i| self.triples.get(i)).collect())
            .unwrap_or_default()
    }

    /// All triples with object `o`.
    #[must_use]
    pub fn object(&self, o: &str) -> Vec<&Triple> {
        self.by_object
            .get(o)
            .map(|v| v.iter().filter_map(|&i| self.triples.get(i)).collect())
            .unwrap_or_default()
    }

    /// All triples.
    #[must_use]
    pub fn all(&self) -> &[Triple] {
        &self.triples
    }

    /// True iff a triple with the same S/P/O already exists.
    #[must_use]
    pub fn contains(&self, triple: &Triple) -> bool {
        let id = triple.id();
        self.triples.iter().any(|t| t.id() == id)
    }
}

impl Memory for SemanticGraph {
    fn kind(&self) -> MemoryKind {
        MemoryKind::Semantic
    }

    fn len(&self) -> usize {
        self.triples.len()
    }

    fn recent(&self, n: usize) -> MemoryResult<Vec<MemoryItem>> {
        let mut sorted: Vec<&Triple> = self.triples.iter().collect();
        sorted.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
        Ok(sorted.into_iter().take(n).map(Triple::as_item).collect())
    }
}
