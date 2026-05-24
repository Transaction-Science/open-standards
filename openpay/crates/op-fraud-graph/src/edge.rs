//! Edges in the fraud graph.
//!
//! An edge is *always* undirected from the algorithms' perspective, but we
//! attach a [`EdgeKind`] tag so heuristics can filter ("walk only along
//! `SharesInstrument` edges to find rings") and so weights can encode
//! domain meaning ("co-transaction within 60 seconds = strong link").

use serde::{Deserialize, Serialize};

use crate::graph::VertexId;

/// The semantic kind of a connection between two entities.
///
/// Adding a kind is forward-compatible: heuristics either care about it
/// explicitly (and filter for it) or treat it the same as the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Two entities appeared together on the same payment attempt — the
    /// generic "co-occurred" link. Almost everything starts here.
    CoTransaction,
    /// Two entities (typically two accounts) share the same payment
    /// instrument (card, bank account). This is the primary ring signal.
    SharesInstrument,
    /// Two entities share a device fingerprint or originating IP.
    /// Often catches velocity / multi-account abuse.
    SharesDevice,
    /// The "billing → shipping" lien — the billing address was
    /// observed shipping to a different address.
    BillingToShipping,
    /// Money actually moved: source → destination. Asymmetric in
    /// reality, but we still store it as an undirected weighted edge
    /// for the algorithms. The weight encodes amount and the
    /// `LaunderingDetector` walks the chain in time order.
    Transfer,
    /// Synthetic-identity link: two entities show signs of belonging
    /// to the same fabricated identity (matching DOB, fragments).
    SyntheticLink,
}

/// A typed, weighted, undirected edge between two [`VertexId`]s.
///
/// Edges are stored on both endpoints' adjacency lists by the graph
/// layer; callers do not need to insert twice.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// Lower-id endpoint (canonicalised so equal undirected edges are
    /// `==`-equal even after a swap).
    pub a: VertexId,
    /// Higher-id endpoint.
    pub b: VertexId,
    /// Semantic kind.
    pub kind: EdgeKind,
    /// Positive finite weight; meaningful to PageRank / Louvain.
    pub weight: f32,
}

impl Edge {
    /// Construct an edge with a canonical endpoint ordering.
    pub fn new(u: VertexId, v: VertexId, kind: EdgeKind, weight: f32) -> Self {
        let (a, b) = if u.0 <= v.0 { (u, v) } else { (v, u) };
        Self { a, b, kind, weight }
    }

    /// Given one endpoint, return the other.
    pub fn other(&self, v: VertexId) -> Option<VertexId> {
        if v == self.a {
            Some(self.b)
        } else if v == self.b {
            Some(self.a)
        } else {
            None
        }
    }
}
