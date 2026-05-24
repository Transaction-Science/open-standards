//! Money-laundering **layering pattern** detectors.
//!
//! Three classical patterns, all reducible to structural queries on the
//! [`crate::FraudGraph`] with `Transfer` edges:
//!
//! 1. **Structuring** — one source splits a sum into many sub-threshold
//!    transfers (e.g. <$10k each, to dodge CTR reporting).
//! 2. **Smurfing** — many low-value sources all flow into one
//!    destination (the inverse of structuring, viewed from the sink).
//! 3. **Mule networks** — long, thin transfer chains where money hops
//!    through N intermediate accounts to obscure provenance.
//!
//! Output is a list of [`LayeringPattern`] hits. A separate, paranoid
//! detector — usually an AML team — decides whether to file a SAR.

use crate::edge::EdgeKind;
use crate::graph::{FraudGraph, VertexId};

/// One detected laundering pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum LayeringPattern {
    /// One source splits a payment across many small destinations.
    Structuring {
        /// Source vertex.
        source: VertexId,
        /// Destination vertices.
        destinations: Vec<VertexId>,
        /// Total weight pushed out.
        total_weight: f32,
    },
    /// One destination receives from many small sources.
    Smurfing {
        /// Destination vertex.
        destination: VertexId,
        /// Source vertices.
        sources: Vec<VertexId>,
        /// Total weight received.
        total_weight: f32,
    },
    /// Long thin transfer chain (mule network).
    MuleChain {
        /// Ordered path of vertex ids, including both endpoints.
        chain: Vec<VertexId>,
    },
}

/// Detector configuration.
#[derive(Debug, Clone, Copy)]
pub struct LaunderingDetector {
    /// Per-edge weight ceiling. Edges whose `weight <=` this count as
    /// "small" and contribute to structuring / smurfing tallies. The
    /// number is in the same units the caller used when inserting
    /// edges (cents, dollars, USD, whatever).
    pub small_transfer_max: f32,
    /// Trigger threshold for structuring / smurfing: at least this many
    /// small transfers from / to the same vertex.
    pub min_small_count: u32,
    /// Mule chain detector: walk depth.
    pub mule_chain_depth: u32,
}

impl Default for LaunderingDetector {
    fn default() -> Self {
        Self {
            small_transfer_max: 9_999.0, // sub-CTR ($10k) threshold
            min_small_count: 4,
            mule_chain_depth: 4,
        }
    }
}

impl LaunderingDetector {
    /// Scan `g` and return every layering pattern that fires.
    pub fn detect(&self, g: &FraudGraph) -> Vec<LayeringPattern> {
        let mut out: Vec<LayeringPattern> = Vec::new();

        for v in g.vertices() {
            let nbrs = g.neighbours(v).unwrap_or(&[]);
            let mut small_neighbours: Vec<(VertexId, f32)> = Vec::new();
            let mut total = 0.0_f32;
            for e in nbrs {
                if e.kind != EdgeKind::Transfer {
                    continue;
                }
                if e.weight > self.small_transfer_max {
                    continue;
                }
                if let Some(other) = e.other(v) {
                    small_neighbours.push((other, e.weight));
                    total += e.weight;
                }
            }
            if (small_neighbours.len() as u32) >= self.min_small_count {
                // We can't distinguish source from destination on an
                // undirected edge; emit BOTH possible interpretations.
                // Downstream consumers filter by direction using the
                // actual transfer record.
                let peers: Vec<VertexId> =
                    small_neighbours.iter().map(|(p, _)| *p).collect();
                out.push(LayeringPattern::Structuring {
                    source: v,
                    destinations: peers.clone(),
                    total_weight: total,
                });
                out.push(LayeringPattern::Smurfing {
                    destination: v,
                    sources: peers,
                    total_weight: total,
                });
            }
        }

        // Mule chains: DFS for paths of length >= depth where every
        // intermediate vertex has degree-2 along Transfer edges (i.e.,
        // is a pure pass-through).
        for start in g.vertices() {
            self.collect_mule_chains(g, start, &mut out);
        }

        out
    }

    fn collect_mule_chains(
        &self,
        g: &FraudGraph,
        start: VertexId,
        out: &mut Vec<LayeringPattern>,
    ) {
        // We want chains start -> a -> b -> ... -> end where each
        // intermediate vertex has exactly 2 Transfer-edge neighbours.
        let mut chain: Vec<VertexId> = vec![start];
        let nbrs = g.neighbours(start).unwrap_or(&[]);
        for e in nbrs {
            if e.kind != EdgeKind::Transfer {
                continue;
            }
            let next = match e.other(start) {
                Some(o) => o,
                None => continue,
            };
            chain.push(next);
            self.walk(g, &mut chain, out);
            chain.pop();
        }
    }

    fn walk(
        &self,
        g: &FraudGraph,
        chain: &mut Vec<VertexId>,
        out: &mut Vec<LayeringPattern>,
    ) {
        if chain.len() as u32 > self.mule_chain_depth + 1 {
            return;
        }
        let cur = *chain.last().expect("chain has at least the start");
        let prev = chain[chain.len().saturating_sub(2)];
        let nbrs = g.neighbours(cur).unwrap_or(&[]);
        let transfer_nbrs: Vec<VertexId> = nbrs
            .iter()
            .filter(|e| e.kind == EdgeKind::Transfer)
            .filter_map(|e| e.other(cur))
            .collect();

        // Only continue through pure pass-through vertices (degree 2 on
        // Transfer edges).
        if chain.len() >= 2 && transfer_nbrs.len() != 2 {
            // Possibly a chain endpoint. Emit if we hit the depth target.
            if (chain.len() as u32) >= self.mule_chain_depth {
                out.push(LayeringPattern::MuleChain {
                    chain: chain.clone(),
                });
            }
            return;
        }

        for &next in &transfer_nbrs {
            if next == prev {
                continue; // don't backtrack
            }
            if chain.contains(&next) {
                continue; // no cycles
            }
            chain.push(next);
            self.walk(g, chain, out);
            chain.pop();
        }
    }
}
