//! Connected-components labelling via union-find.
//!
//! The first question a fraud analyst asks about a new payment is:
//! "what cluster is this entity part of, and how big is it?". A
//! component of 2-3 entities is usually a person + their household;
//! a component of 200 entities sharing the same card is a ring.
//!
//! ## Algorithm
//!
//! Weighted union by rank with path compression. `O(E α(V))`, where
//! `α` is the inverse Ackermann function (effectively constant). Stable
//! against streaming updates: re-running on a graph after edge inserts
//! gives the same component labels as a fresh build.

use crate::edge::EdgeKind;
use crate::graph::{FraudGraph, VertexId};

/// Opaque component id. The integer value is meaningful only relative to
/// one [`ConnectedComponents`] result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ComponentId(pub u32);

/// Per-vertex component assignment plus per-component statistics.
#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    /// `labels[i]` = component for `VertexId(i as u32)`.
    labels: Vec<ComponentId>,
    /// `sizes[c.0 as usize]` = number of vertices in component `c`.
    sizes: Vec<u32>,
}

impl ConnectedComponents {
    /// Compute components considering *all* edge kinds.
    pub fn from_graph(g: &FraudGraph) -> Self {
        Self::from_graph_filtered(g, |_| true)
    }

    /// Compute components considering only edges whose kind satisfies
    /// `keep`. Lets analysts ask "what's the shared-instrument cluster
    /// here?" by passing `|k| k == EdgeKind::SharesInstrument`.
    pub fn from_graph_filtered<F: Fn(EdgeKind) -> bool>(g: &FraudGraph, keep: F) -> Self {
        let n = g.vertex_count();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut rank: Vec<u8> = vec![0; n];

        for e in g.edges() {
            if !keep(e.kind) {
                continue;
            }
            union(&mut parent, &mut rank, e.a.0, e.b.0);
        }

        // Compress and relabel into contiguous component ids.
        let mut canonical: Vec<u32> = vec![u32::MAX; n];
        let mut next_id: u32 = 0;
        let mut labels: Vec<ComponentId> = Vec::with_capacity(n);
        let mut sizes: Vec<u32> = Vec::new();
        for v in 0..n as u32 {
            let r = find(&mut parent, v);
            let cid = if canonical[r as usize] == u32::MAX {
                canonical[r as usize] = next_id;
                sizes.push(0);
                next_id += 1;
                canonical[r as usize]
            } else {
                canonical[r as usize]
            };
            sizes[cid as usize] += 1;
            labels.push(ComponentId(cid));
        }

        Self { labels, sizes }
    }

    /// Component for a given vertex.
    pub fn component_of(&self, v: VertexId) -> Option<ComponentId> {
        self.labels.get(v.0 as usize).copied()
    }

    /// Number of components.
    pub fn component_count(&self) -> usize {
        self.sizes.len()
    }

    /// Size of a component, or `None` if id is invalid.
    pub fn component_size(&self, c: ComponentId) -> Option<u32> {
        self.sizes.get(c.0 as usize).copied()
    }

    /// All members of a given component (`O(|V|)`).
    pub fn members(&self, c: ComponentId) -> Vec<VertexId> {
        self.labels
            .iter()
            .enumerate()
            .filter_map(|(i, &cid)| {
                if cid == c {
                    Some(VertexId(i as u32))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Iterate `(ComponentId, size)` sorted descending by size — useful
    /// for the analyst dashboard's "largest cluster" panel.
    pub fn components_by_size(&self) -> Vec<(ComponentId, u32)> {
        let mut v: Vec<(ComponentId, u32)> = self
            .sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| (ComponentId(i as u32), s))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        let gp = parent[p as usize];
        parent[x as usize] = gp;
        x = gp;
    }
    x
}

fn union(parent: &mut [u32], rank: &mut [u8], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    let (lo, hi) = match rank[ra as usize].cmp(&rank[rb as usize]) {
        core::cmp::Ordering::Less => (ra, rb),
        core::cmp::Ordering::Greater => (rb, ra),
        core::cmp::Ordering::Equal => {
            rank[ra as usize] = rank[ra as usize].saturating_add(1);
            (rb, ra)
        }
    };
    parent[lo as usize] = hi;
}
