//! EOC stage 3 — graph / triple-store.
//!
//! This stage answers structured questions from a small in-memory triple
//! store using a lightweight pattern matcher over the query text. It is a
//! deliberately minimal stub — production deployments substitute a real
//! graph backend (oxigraph, neptune, neo4j) and the DCY query language
//! described in EOC-4. The trait surface and joule accounting stay the
//! same.

#![forbid(unsafe_code)]

use std::sync::Mutex;

use async_trait::async_trait;
use eoc_cache::Stage;
use eoc_core::{JouleCost, Query, Response, Stage as StageKind};

/// A subject-predicate-object triple.
#[derive(Debug, Clone)]
pub struct Triple {
    /// Subject (the entity).
    pub subject: String,
    /// Predicate (the relation).
    pub predicate: String,
    /// Object (the target).
    pub object: String,
}

impl Triple {
    /// Construct a triple.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            subject: s.into(),
            predicate: p.into(),
            object: o.into(),
        }
    }
}

/// In-memory triple-store graph stage.
pub struct GraphStage {
    triples: Mutex<Vec<Triple>>,
    estimated_microjoules: u64,
}

impl GraphStage {
    /// Construct an empty graph stage.
    pub fn new() -> Self {
        Self {
            triples: Mutex::new(Vec::new()),
            estimated_microjoules: 500,
        }
    }

    /// Insert a triple.
    pub fn insert(&self, triple: Triple) {
        self.triples
            .lock()
            .expect("graph lock poisoned")
            .push(triple);
    }

    /// Insert many triples.
    pub fn extend(&self, triples: impl IntoIterator<Item = Triple>) {
        let mut g = self.triples.lock().expect("graph lock poisoned");
        for t in triples {
            g.push(t);
        }
    }

    /// Number of triples held.
    pub fn len(&self) -> usize {
        self.triples.lock().expect("graph lock poisoned").len()
    }

    /// Is the store empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for GraphStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for GraphStage {
    async fn try_resolve(&self, q: &Query) -> Option<Response> {
        // Lightweight matcher: for every triple (s,p,o), if the prompt
        // mentions both `s` and `p` (case-insensitive), return `o`.
        let prompt = q.prompt.to_lowercase();
        let g = self.triples.lock().expect("graph lock poisoned");
        for t in g.iter() {
            if prompt.contains(&t.subject.to_lowercase())
                && prompt.contains(&t.predicate.to_lowercase())
            {
                return Some(Response::new(
                    q.id,
                    t.object.clone(),
                    StageKind::Graph,
                    JouleCost::estimated(self.estimated_microjoules),
                ));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn graph_matches_known_relation() {
        let g = GraphStage::new();
        g.insert(Triple::new("Paris", "capital of", "France"));
        let q = Query::new("Paris is the capital of which country?");
        let r = g.try_resolve(&q).await.expect("hit");
        assert_eq!(r.payload, "France");
        assert_eq!(r.stage, StageKind::Graph);
    }

    #[tokio::test]
    async fn graph_misses_unknown() {
        let g = GraphStage::new();
        g.insert(Triple::new("Paris", "capital of", "France"));
        let q = Query::new("what is the speed of light?");
        assert!(g.try_resolve(&q).await.is_none());
    }
}
