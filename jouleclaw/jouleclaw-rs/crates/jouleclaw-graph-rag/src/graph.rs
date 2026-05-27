//! Minimal knowledge-graph trait surface and an in-memory reference impl.
//!
//! The donor (`verity-cascade::l125_graph_rag`) built co-occurrence
//! relationships from snippet overlap. JouleClaw's L1.25 instead asks a
//! consumer-supplied [`KnowledgeGraph`] to resolve canonical entities and
//! their neighbourhoods. Sites that want the old co-occurrence behaviour
//! can construct their own [`InMemoryKnowledgeGraph`] from the snippet set
//! before dispatching the query — the trait is intentionally storage-
//! agnostic (Neo4j, sled, in-memory HashMap, etc.).

use std::collections::{HashMap, HashSet};

/// Canonical identifier for an entity inside a [`KnowledgeGraph`].
///
/// Stored as `String` so consumers can use URIs, ULIDs, integers-as-text,
/// or whatever their store keys on. The trait does not care.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct EntityId(pub String);

impl EntityId {
    /// Construct a fresh id from anything `Into<String>`.
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying canonical string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for EntityId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for EntityId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// A resolved entity returned by a [`KnowledgeGraph`].
///
/// The shape is intentionally small — just enough that downstream tiers
/// (L1.375 StructContrast, L1.5 SsmReader) can construct a structured
/// prompt without re-querying the graph. Sites that need richer payloads
/// can wrap an [`Entity`] with their own per-store metadata.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Entity {
    /// Canonical id (the store's primary key).
    pub id: EntityId,
    /// Display name — what the extractor's surface form resolved to.
    pub name: String,
    /// Free-form type tag (e.g. `"Person"`, `"Concept"`, `"Quantity"`).
    /// The trait places no schema constraints; consumers SHOULD agree on
    /// a vocabulary with the downstream tiers they fan out to.
    pub kind: String,
    /// Optional short description for prose summaries. Empty if unknown.
    #[serde(default)]
    pub description: String,
}

/// A directed edge between two entities in a [`KnowledgeGraph`].
///
/// Edges carry a `weight` in `[0.0, 1.0]` so consumers can rank
/// neighbourhoods; sites that don't have a meaningful weight should
/// emit `1.0`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Edge {
    /// Source entity.
    pub from: EntityId,
    /// Destination entity.
    pub to: EntityId,
    /// Edge label (free-form predicate; e.g. `"depends_on"`).
    pub label: String,
    /// Edge weight in `[0.0, 1.0]`. Higher = stronger association.
    pub weight: f32,
}

/// Consumer-pluggable knowledge graph.
///
/// Implementations resolve surface forms to canonical entities and expose
/// neighbourhood traversal. The trait is intentionally **synchronous** —
/// L1.25 runs in the µJ regime; if your store is remote, wrap it with an
/// in-process cache before plugging it in.
///
/// `Send + Sync` because tiers move across threads in the cascade
/// runtime.
pub trait KnowledgeGraph: Send + Sync {
    /// Resolve a surface form (case-insensitive) to a canonical entity.
    ///
    /// Returns `None` when the name is unknown.
    fn lookup_entity(&self, name: &str) -> Option<Entity>;

    /// Return outgoing edges from `entity_id` up to `depth` hops.
    ///
    /// `depth == 0` MUST return the empty vector; `depth == 1` MUST
    /// return only the direct neighbours. Implementations SHOULD
    /// deduplicate edges and cap the response at a reasonable bound to
    /// keep the L1.25 cost envelope honest.
    fn neighbors(&self, entity_id: &EntityId, depth: u8) -> Vec<Edge>;

    /// True iff the graph has no entities. Used by the tier to gate
    /// `estimate_cost` — an empty graph short-circuits to `None`.
    ///
    /// Default: `false` (assume populated). Override for cheaper checks.
    fn is_empty(&self) -> bool {
        false
    }
}

// ─── In-memory reference impl ─────────────────────────────────────

/// Reference implementation backed by `HashMap`. Suitable for tests,
/// fixtures, and small embedded deployments. Production sites with
/// real knowledge graphs should write their own `KnowledgeGraph` impl
/// over their store of choice.
#[derive(Debug, Default, Clone)]
pub struct InMemoryKnowledgeGraph {
    /// Canonical id → entity record.
    entities: HashMap<EntityId, Entity>,
    /// Surface-form (lowercased) → canonical id. Resolution index.
    name_index: HashMap<String, EntityId>,
    /// Outgoing edges, keyed by source id.
    edges: HashMap<EntityId, Vec<Edge>>,
}

impl InMemoryKnowledgeGraph {
    /// Construct an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an entity. Overwrites any existing entity with the same id
    /// and indexes the entity's display name (case-insensitive) so
    /// [`KnowledgeGraph::lookup_entity`] can find it.
    pub fn insert_entity(&mut self, entity: Entity) -> &mut Self {
        self.name_index
            .insert(entity.name.to_lowercase(), entity.id.clone());
        self.entities.insert(entity.id.clone(), entity);
        self
    }

    /// Insert a directed edge. Both endpoints SHOULD already exist; the
    /// method does not validate this so callers can build the graph in
    /// any order.
    pub fn insert_edge(&mut self, edge: Edge) -> &mut Self {
        self.edges.entry(edge.from.clone()).or_default().push(edge);
        self
    }

    /// Number of entities currently stored.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// True iff no entities are stored.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// All known entity ids — useful for tests and offline batches.
    pub fn entity_ids(&self) -> Vec<EntityId> {
        self.entities.keys().cloned().collect()
    }
}

impl KnowledgeGraph for InMemoryKnowledgeGraph {
    fn lookup_entity(&self, name: &str) -> Option<Entity> {
        let key = name.to_lowercase();
        let id = self.name_index.get(&key)?;
        self.entities.get(id).cloned()
    }

    fn neighbors(&self, entity_id: &EntityId, depth: u8) -> Vec<Edge> {
        if depth == 0 {
            return Vec::new();
        }
        // BFS, deduplicated by `(from,to,label)`.
        let mut seen: HashSet<(EntityId, EntityId, String)> = HashSet::new();
        let mut out: Vec<Edge> = Vec::new();
        let mut frontier: Vec<EntityId> = vec![entity_id.clone()];
        for _ in 0..depth {
            let mut next_frontier: Vec<EntityId> = Vec::new();
            for node in &frontier {
                if let Some(edges) = self.edges.get(node) {
                    for e in edges {
                        let key = (e.from.clone(), e.to.clone(), e.label.clone());
                        if seen.insert(key) {
                            next_frontier.push(e.to.clone());
                            out.push(e.clone());
                        }
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        out
    }

    fn is_empty(&self) -> bool {
        InMemoryKnowledgeGraph::is_empty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> InMemoryKnowledgeGraph {
        let mut g = InMemoryKnowledgeGraph::new();
        g.insert_entity(Entity {
            id: EntityId::new("urn:rust"),
            name: "Rust".into(),
            kind: "Language".into(),
            description: "systems language".into(),
        });
        g.insert_entity(Entity {
            id: EntityId::new("urn:cargo"),
            name: "Cargo".into(),
            kind: "Tool".into(),
            description: "package manager".into(),
        });
        g.insert_entity(Entity {
            id: EntityId::new("urn:crates_io"),
            name: "crates.io".into(),
            kind: "Service".into(),
            description: "package registry".into(),
        });
        g.insert_edge(Edge {
            from: EntityId::new("urn:rust"),
            to: EntityId::new("urn:cargo"),
            label: "ships_with".into(),
            weight: 1.0,
        });
        g.insert_edge(Edge {
            from: EntityId::new("urn:cargo"),
            to: EntityId::new("urn:crates_io"),
            label: "publishes_to".into(),
            weight: 0.9,
        });
        g
    }

    #[test]
    fn lookup_resolves_case_insensitively() {
        let g = fixture();
        assert!(g.lookup_entity("rust").is_some());
        assert!(g.lookup_entity("RUST").is_some());
        assert!(g.lookup_entity("Rust").is_some());
    }

    #[test]
    fn lookup_missing_is_none() {
        let g = fixture();
        assert!(g.lookup_entity("python").is_none());
    }

    #[test]
    fn neighbors_depth_zero_is_empty() {
        let g = fixture();
        let n = g.neighbors(&EntityId::new("urn:rust"), 0);
        assert!(n.is_empty());
    }

    #[test]
    fn neighbors_depth_one_returns_direct_only() {
        let g = fixture();
        let n = g.neighbors(&EntityId::new("urn:rust"), 1);
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].to.as_str(), "urn:cargo");
    }

    #[test]
    fn neighbors_depth_two_traverses_transitively() {
        let g = fixture();
        let n = g.neighbors(&EntityId::new("urn:rust"), 2);
        assert_eq!(n.len(), 2);
        let labels: Vec<&str> = n.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"ships_with"));
        assert!(labels.contains(&"publishes_to"));
    }

    #[test]
    fn empty_graph_reports_empty() {
        let g = InMemoryKnowledgeGraph::new();
        assert!(KnowledgeGraph::is_empty(&g));
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn populated_graph_reports_non_empty() {
        let g = fixture();
        assert!(!KnowledgeGraph::is_empty(&g));
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn entity_id_from_str_and_string() {
        let a: EntityId = "x".into();
        let b: EntityId = String::from("x").into();
        let c = EntityId::new("x");
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a.as_str(), "x");
    }
}
