//! [`FraudGraph`] — the in-memory adjacency-list backing every algorithm
//! in this crate.
//!
//! ## Why adjacency list (not CSR)
//!
//! Fraud graphs are streaming: new payments arrive continuously, so the
//! graph mutates constantly. Compressed Sparse Row would be cheaper to
//! traverse but expensive to mutate. The Louvain / PageRank routines we
//! ship here are linear in `|E|` regardless of layout, and adjacency
//! lists let us insert in O(1) amortised. If a future operator wants a
//! CSR snapshot for batch analysis, they can build one from
//! [`FraudGraph::neighbours`].
//!
//! ## Identifiers
//!
//! Vertices are addressed by [`VertexId`] (a wrapper around `u32`). The
//! [`crate::entity::EntityKey`] of every vertex is also retained, so the
//! caller can do lookup-by-key without dragging the hashing layer into
//! the algorithm code.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::edge::{Edge, EdgeKind};
use crate::entity::{Entity, EntityKey};
use crate::error::{Error, Result};

/// Opaque dense vertex index.
///
/// Stable across the lifetime of a single [`FraudGraph`]. NOT stable
/// across re-ingest; persist by [`EntityKey`] if you need durability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VertexId(pub u32);

/// Per-vertex bookkeeping the algorithms don't care about but the
/// heuristics do (synthetic-identity, velocity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexMeta {
    /// Original entity.
    pub entity: Entity,
    /// Wall-clock first observation, in Unix seconds.
    pub first_seen: i64,
    /// Wall-clock last observation, in Unix seconds.
    pub last_seen: i64,
    /// Total transactions touching this vertex (any [`EdgeKind`]).
    pub tx_count: u64,
}

impl VertexMeta {
    fn new(entity: Entity, ts: i64) -> Self {
        Self {
            entity,
            first_seen: ts,
            last_seen: ts,
            tx_count: 0,
        }
    }
}

/// In-memory fraud graph. Single-writer; the caller is expected to wrap
/// in a `Mutex` or actor if multi-threaded ingest is required.
///
/// Memory: `~24 bytes/vertex + ~28 bytes/edge`. A 1M-vertex / 5M-edge
/// graph fits in roughly 170 MB.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FraudGraph {
    /// Dense vertex storage indexed by `VertexId.0 as usize`.
    vertices: Vec<VertexMeta>,
    /// EntityKey → vertex index, for resolution.
    index: HashMap<EntityKey, VertexId>,
    /// Adjacency: per vertex, the edges that touch it. Each edge is
    /// stored twice (once on each endpoint) for O(degree) neighbour
    /// scans without an extra lookup.
    adj: Vec<Vec<Edge>>,
    /// Total unique edges (each undirected edge counted once).
    edge_count: u64,
}

impl FraudGraph {
    /// Empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of *undirected* edges.
    pub fn edge_count(&self) -> u64 {
        self.edge_count
    }

    /// Insert or look up the vertex for an [`Entity`]. Idempotent: a
    /// second call with the same `EntityKey` returns the original
    /// [`VertexId`] and bumps `last_seen` / `tx_count` if `bump` is true.
    pub fn upsert_entity(&mut self, entity: Entity, ts: i64, bump: bool) -> VertexId {
        if let Some(id) = self.index.get(&entity.key).copied() {
            if bump {
                let m = &mut self.vertices[id.0 as usize];
                m.last_seen = m.last_seen.max(ts);
                m.first_seen = m.first_seen.min(ts);
                m.tx_count = m.tx_count.saturating_add(1);
            }
            return id;
        }
        let id = VertexId(u32::try_from(self.vertices.len()).unwrap_or(u32::MAX));
        let mut meta = VertexMeta::new(entity.clone(), ts);
        if bump {
            meta.tx_count = 1;
        }
        self.vertices.push(meta);
        self.adj.push(Vec::new());
        self.index.insert(entity.key, id);
        id
    }

    /// Find a vertex by entity key without inserting.
    pub fn lookup(&self, key: &EntityKey) -> Option<VertexId> {
        self.index.get(key).copied()
    }

    /// Borrow vertex metadata.
    pub fn meta(&self, id: VertexId) -> Result<&VertexMeta> {
        self.vertices
            .get(id.0 as usize)
            .ok_or(Error::VertexNotFound(id))
    }

    /// Mutable borrow of vertex metadata, mostly for tests.
    pub fn meta_mut(&mut self, id: VertexId) -> Result<&mut VertexMeta> {
        self.vertices
            .get_mut(id.0 as usize)
            .ok_or(Error::VertexNotFound(id))
    }

    /// Insert (or merge) an undirected edge between `u` and `v`.
    ///
    /// If an edge of the same `kind` already exists, its weight is added
    /// to the new one (so repeat co-occurrences accumulate). Different
    /// kinds between the same pair are stored as separate edges.
    pub fn add_edge(
        &mut self,
        u: VertexId,
        v: VertexId,
        kind: EdgeKind,
        weight: f32,
    ) -> Result<()> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::InvalidEdgeWeight(weight));
        }
        if (u.0 as usize) >= self.vertices.len() {
            return Err(Error::VertexNotFound(u));
        }
        if (v.0 as usize) >= self.vertices.len() {
            return Err(Error::VertexNotFound(v));
        }
        if u == v {
            // Self-loops are meaningless in this domain.
            return Ok(());
        }

        let edge = Edge::new(u, v, kind, weight);

        // Look for an existing edge of the same kind between u and v.
        let existing_u = self.adj[u.0 as usize]
            .iter()
            .position(|e| e.other(u) == Some(v) && e.kind == kind);
        if let Some(idx_u) = existing_u {
            let new_weight = self.adj[u.0 as usize][idx_u].weight + weight;
            self.adj[u.0 as usize][idx_u].weight = new_weight;
            if let Some(idx_v) = self.adj[v.0 as usize]
                .iter()
                .position(|e| e.other(v) == Some(u) && e.kind == kind)
            {
                self.adj[v.0 as usize][idx_v].weight = new_weight;
            }
            return Ok(());
        }

        self.adj[u.0 as usize].push(edge);
        self.adj[v.0 as usize].push(edge);
        self.edge_count = self.edge_count.saturating_add(1);
        Ok(())
    }

    /// Iterate the edges touching `v`.
    pub fn neighbours(&self, v: VertexId) -> Result<&[Edge]> {
        self.adj
            .get(v.0 as usize)
            .map(Vec::as_slice)
            .ok_or(Error::VertexNotFound(v))
    }

    /// Iterate every vertex id.
    pub fn vertices(&self) -> impl Iterator<Item = VertexId> + '_ {
        (0..self.vertices.len()).map(|i| VertexId(i as u32))
    }

    /// Iterate every unique undirected edge once.
    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.adj
            .iter()
            .enumerate()
            .flat_map(|(i, list)| {
                let from = VertexId(i as u32);
                list.iter().copied().filter(move |e| e.a == from)
            })
    }

    /// Convenience: ingest a payment "co-transaction" — every pair of
    /// entities present on a payment gets a [`EdgeKind::CoTransaction`]
    /// edge added (or merged) with weight 1.
    pub fn ingest_co_transaction(
        &mut self,
        entities: &[Entity],
        ts_unix: i64,
    ) -> Result<Vec<VertexId>> {
        let ids: Vec<VertexId> = entities
            .iter()
            .map(|e| self.upsert_entity(e.clone(), ts_unix, true))
            .collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                self.add_edge(ids[i], ids[j], EdgeKind::CoTransaction, 1.0)?;
            }
        }
        Ok(ids)
    }
}

/// Helper: current wall-clock as Unix seconds. Exposed so test code can
/// stub time deterministically.
pub fn now_unix() -> i64 {
    OffsetDateTime::now_utc().unix_timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityKind;

    fn ent(k: EntityKind, raw: &str) -> Entity {
        Entity::new(k, raw)
    }

    #[test]
    fn upsert_is_idempotent() {
        let mut g = FraudGraph::new();
        let a = g.upsert_entity(ent(EntityKind::EmailHash, "x@y"), 0, true);
        let b = g.upsert_entity(ent(EntityKind::EmailHash, "X@Y"), 100, true);
        assert_eq!(a, b);
        assert_eq!(g.meta(a).expect("vertex exists").tx_count, 2);
    }

    #[test]
    fn add_edge_merges_same_kind() {
        let mut g = FraudGraph::new();
        let u = g.upsert_entity(ent(EntityKind::Account, "u"), 0, false);
        let v = g.upsert_entity(ent(EntityKind::Account, "v"), 0, false);
        g.add_edge(u, v, EdgeKind::CoTransaction, 1.0).expect("ok");
        g.add_edge(u, v, EdgeKind::CoTransaction, 2.5).expect("ok");
        assert_eq!(g.edge_count(), 1);
        let e = g
            .neighbours(u)
            .expect("ok")
            .iter()
            .find(|e| e.other(u) == Some(v))
            .copied()
            .expect("edge present");
        assert!((e.weight - 3.5).abs() < 1e-6);
    }

    #[test]
    fn different_kinds_coexist() {
        let mut g = FraudGraph::new();
        let u = g.upsert_entity(ent(EntityKind::Account, "u"), 0, false);
        let v = g.upsert_entity(ent(EntityKind::Account, "v"), 0, false);
        g.add_edge(u, v, EdgeKind::CoTransaction, 1.0).expect("ok");
        g.add_edge(u, v, EdgeKind::SharesInstrument, 1.0).expect("ok");
        assert_eq!(g.edge_count(), 2);
    }
}
