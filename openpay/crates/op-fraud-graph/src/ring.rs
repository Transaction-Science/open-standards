//! Shared-instrument **ring detection**.
//!
//! The fraud playbook this targets: a stolen card (or stolen account)
//! used at many merchants in quick succession. Detection is structural:
//! the same payment-instrument entity appears as a neighbour of N
//! distinct merchant / account entities within a configured time window
//! and edge-weight floor. If `N >= min_distinct_merchants`, we emit a
//! [`Ring`].
//!
//! This is the cheap, deterministic ring detector. A learned scorer can
//! consume [`Ring::score`] as a feature alongside per-transaction
//! signals.

use std::collections::HashSet;

use crate::edge::EdgeKind;
use crate::entity::EntityKind;
use crate::graph::{FraudGraph, VertexId};

/// Configuration for [`RingDetector`].
#[derive(Debug, Clone, Copy)]
pub struct RingDetector {
    /// Trigger threshold: a hub must touch at least this many distinct
    /// merchant / account entities before we call it a ring.
    pub min_distinct_merchants: u32,
    /// Edges below this weight are ignored (filters out one-off
    /// co-occurrences from the noise).
    pub min_edge_weight: f32,
    /// The kinds of edges that count as a "shared instrument" link.
    /// By default: `SharesInstrument` and `SharesDevice`.
    pub instrument_edges: &'static [EdgeKind],
    /// The kinds of vertices that count as a "merchant or account"
    /// terminal — i.e. the things being touched.
    pub merchant_kinds: &'static [EntityKind],
}

impl Default for RingDetector {
    fn default() -> Self {
        Self {
            min_distinct_merchants: 5,
            min_edge_weight: 1.0,
            instrument_edges: &[EdgeKind::SharesInstrument, EdgeKind::SharesDevice],
            merchant_kinds: &[EntityKind::Account],
        }
    }
}

/// One detected ring: a hub entity plus the merchants / accounts it
/// touches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ring {
    /// The shared payment instrument (or device).
    pub hub: VertexId,
    /// Members of the ring (sorted ascending for reproducibility).
    pub members: Vec<VertexId>,
}

impl Ring {
    /// Number of distinct merchants / accounts in the ring.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Crude risk score in `[0, 1]`: saturates at 50 distinct members.
    pub fn score(&self) -> f32 {
        let n = self.members.len().min(50) as f32;
        n / 50.0
    }
}

impl RingDetector {
    /// Scan `g` and emit every ring above the threshold.
    pub fn detect(&self, g: &FraudGraph) -> Vec<Ring> {
        let mut rings: Vec<Ring> = Vec::new();

        for v in g.vertices() {
            // Iterate every vertex; only the ones that *could* be a hub
            // (i.e., that have enough qualifying neighbours) become rings.
            let nbrs = match g.neighbours(v) {
                Ok(n) => n,
                Err(_) => continue,
            };
            let mut merchants: HashSet<VertexId> = HashSet::new();
            for e in nbrs {
                if !self.instrument_edges.contains(&e.kind) {
                    continue;
                }
                if e.weight < self.min_edge_weight {
                    continue;
                }
                let other = match e.other(v) {
                    Some(o) => o,
                    None => continue,
                };
                let other_meta = match g.meta(other) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if !self.merchant_kinds.contains(&other_meta.entity.key.kind) {
                    continue;
                }
                merchants.insert(other);
            }
            if (merchants.len() as u32) >= self.min_distinct_merchants {
                let mut members: Vec<VertexId> = merchants.into_iter().collect();
                members.sort();
                rings.push(Ring { hub: v, members });
            }
        }

        rings.sort_by(|a, b| b.size().cmp(&a.size()));
        rings
    }
}
