//! Cluster topology types: mesh, ring, tree.
//!
//! Inference systems care about topology for two reasons:
//!
//! 1. **Collective communication patterns.** Tensor-parallel + expert-
//!    parallel sharding wants all-to-all (mesh); pipeline-parallel
//!    wants a directed ring; tree topologies show up in parameter
//!    servers and broadcast-reduce.
//! 2. **Pretty-printed status.** Operators want to dump the live wiring
//!    and read it at a glance.
//!
//! This module is intentionally light-weight: a [`Topology`] is just an
//! adjacency-list view over node ids plus a kind tag, with helpers for
//! constructing the three canonical shapes and a `Display` impl for
//! human-readable dumps.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{DistributedError, Result};

/// What shape the topology is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologyKind {
    /// Fully-connected mesh — every node sees every other node.
    Mesh,
    /// Directed ring — pipeline-parallel stages, ordered.
    Ring,
    /// Rooted tree — broadcast / reduce.
    Tree,
}

impl TopologyKind {
    /// Short tag.
    pub fn tag(&self) -> &'static str {
        match self {
            TopologyKind::Mesh => "mesh",
            TopologyKind::Ring => "ring",
            TopologyKind::Tree => "tree",
        }
    }
}

/// A logical topology over a fixed set of node ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology {
    /// Shape.
    pub kind: TopologyKind,
    /// Ordered node ids.
    pub nodes: Vec<String>,
    /// Adjacency, keyed by node id.
    pub edges: BTreeMap<String, Vec<String>>,
}

impl Topology {
    /// Fully-connected mesh.
    pub fn mesh<I, S>(nodes: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let ids: Vec<String> = nodes.into_iter().map(Into::into).collect();
        if ids.len() < 2 {
            return Err(DistributedError::InvalidTopology(
                "mesh requires at least 2 nodes".into(),
            ));
        }
        let mut edges = BTreeMap::new();
        for id in &ids {
            let nbrs: Vec<String> = ids.iter().filter(|x| *x != id).cloned().collect();
            edges.insert(id.clone(), nbrs);
        }
        Ok(Self {
            kind: TopologyKind::Mesh,
            nodes: ids,
            edges,
        })
    }

    /// Directed ring (i -> i+1 -> ... -> 0).
    pub fn ring<I, S>(nodes: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let ids: Vec<String> = nodes.into_iter().map(Into::into).collect();
        if ids.len() < 2 {
            return Err(DistributedError::InvalidTopology(
                "ring requires at least 2 nodes".into(),
            ));
        }
        let mut edges = BTreeMap::new();
        for (i, id) in ids.iter().enumerate() {
            let next = &ids[(i + 1) % ids.len()];
            edges.insert(id.clone(), vec![next.clone()]);
        }
        Ok(Self {
            kind: TopologyKind::Ring,
            nodes: ids,
            edges,
        })
    }

    /// Rooted tree of fan-out `fan_out`. First node is the root.
    pub fn tree<I, S>(nodes: I, fan_out: usize) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let ids: Vec<String> = nodes.into_iter().map(Into::into).collect();
        if ids.is_empty() {
            return Err(DistributedError::InvalidTopology(
                "tree requires at least 1 node".into(),
            ));
        }
        if fan_out < 1 {
            return Err(DistributedError::InvalidTopology(
                "fan_out must be >= 1".into(),
            ));
        }
        let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for id in &ids {
            edges.insert(id.clone(), Vec::new());
        }
        for (i, id) in ids.iter().enumerate() {
            for k in 1..=fan_out {
                let child_idx = i * fan_out + k;
                if child_idx >= ids.len() {
                    break;
                }
                edges
                    .entry(id.clone())
                    .or_default()
                    .push(ids[child_idx].clone());
            }
        }
        Ok(Self {
            kind: TopologyKind::Tree,
            nodes: ids,
            edges,
        })
    }

    /// Neighbours of `id`.
    pub fn neighbours(&self, id: &str) -> &[String] {
        self.edges
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True if there are no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl std::fmt::Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "topology({}) nodes={}", self.kind.tag(), self.nodes.len())?;
        for id in &self.nodes {
            let nbrs = self.neighbours(id);
            if nbrs.is_empty() {
                writeln!(f, "  {id}")?;
            } else {
                writeln!(f, "  {id} -> [{}]", nbrs.join(", "))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_is_complete() {
        let t = Topology::mesh(["a", "b", "c"]).expect("ok");
        assert_eq!(t.neighbours("a").len(), 2);
        assert_eq!(t.neighbours("b").len(), 2);
        assert_eq!(t.neighbours("c").len(), 2);
    }

    #[test]
    fn ring_closes() {
        let t = Topology::ring(["a", "b", "c"]).expect("ok");
        assert_eq!(t.neighbours("a"), &["b".to_string()]);
        assert_eq!(t.neighbours("c"), &["a".to_string()]);
    }

    #[test]
    fn tree_fans_out() {
        let t = Topology::tree(["r", "c1", "c2", "g1", "g2"], 2).expect("ok");
        assert_eq!(t.neighbours("r").len(), 2);
        assert_eq!(t.neighbours("c1").len(), 2);
    }

    #[test]
    fn empty_mesh_errors() {
        assert!(Topology::mesh(Vec::<String>::new()).is_err());
    }
}
