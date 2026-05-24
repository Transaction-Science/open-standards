//! Louvain — modularity-greedy community detection.
//!
//! Louvain is the de-facto choice for finding "natural" clusters in a
//! weighted undirected graph: it greedily moves each vertex into the
//! neighbouring community that yields the largest modularity gain, then
//! contracts and repeats. Fraud teams use the output to group accounts
//! that behave as a single unit even when they don't share a direct
//! edge.
//!
//! ## Streaming
//!
//! Re-running Louvain on a graph after a small ingest is cheap because
//! the algorithm starts from each vertex in its own community and
//! converges in a couple of passes on typical fraud graphs. For "true"
//! incremental community detection (delta-Louvain), operators can keep
//! the previous [`Louvain::communities`] vector and pass it as the
//! starting point via [`Louvain::run_with_seed`].

use std::collections::HashMap;

use crate::graph::{FraudGraph, VertexId};

/// Opaque community label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Community(pub u32);

/// Louvain runner.
#[derive(Debug, Clone, Copy)]
pub struct Louvain {
    /// Maximum number of full passes (one pass = one round of per-vertex
    /// moves; reaches a local modularity maximum within ~10 passes on
    /// typical graphs).
    pub max_passes: u32,
    /// Modularity delta below which we declare convergence.
    pub min_modularity_gain: f64,
}

impl Default for Louvain {
    fn default() -> Self {
        Self {
            max_passes: 16,
            min_modularity_gain: 1e-6,
        }
    }
}

/// Output of one Louvain run.
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// `community[i]` = community for `VertexId(i as u32)`.
    pub community: Vec<Community>,
    /// Final modularity. Higher is "better separated".
    pub modularity: f64,
    /// Passes actually performed.
    pub passes: u32,
}

impl LouvainResult {
    /// Iterate `(Community, members)`.
    pub fn members_by_community(&self) -> HashMap<Community, Vec<VertexId>> {
        let mut out: HashMap<Community, Vec<VertexId>> = HashMap::new();
        for (i, &c) in self.community.iter().enumerate() {
            out.entry(c).or_default().push(VertexId(i as u32));
        }
        out
    }

    /// Community for a vertex.
    pub fn community_of(&self, v: VertexId) -> Option<Community> {
        self.community.get(v.0 as usize).copied()
    }

    /// Number of distinct communities.
    pub fn community_count(&self) -> usize {
        self.members_by_community().len()
    }
}

impl Louvain {
    /// Run from the canonical seed (each vertex in its own community).
    pub fn run(&self, g: &FraudGraph) -> LouvainResult {
        let n = g.vertex_count();
        let seed: Vec<Community> = (0..n as u32).map(Community).collect();
        self.run_with_seed(g, seed)
    }

    /// Run starting from a caller-supplied community labelling. Length
    /// must equal `g.vertex_count()`; out-of-range labels are kept as-is.
    pub fn run_with_seed(&self, g: &FraudGraph, seed: Vec<Community>) -> LouvainResult {
        let n = g.vertex_count();
        let mut community = if seed.len() == n {
            seed
        } else {
            (0..n as u32).map(Community).collect()
        };

        // Pre-compute per-vertex weighted degree.
        let mut degree: Vec<f64> = vec![0.0; n];
        let mut total_weight = 0.0_f64;
        for v in 0..n as u32 {
            let nbrs = g.neighbours(VertexId(v)).unwrap_or(&[]);
            let mut d = 0.0_f64;
            for e in nbrs {
                d += f64::from(e.weight);
            }
            degree[v as usize] = d;
            total_weight += d;
        }
        // Each undirected edge counted twice via degree sum.
        let m2 = total_weight.max(1e-12);

        // Community → sum of internal weights and total degrees.
        let mut comm_total: HashMap<u32, f64> = HashMap::new();
        for v in 0..n as u32 {
            *comm_total.entry(community[v as usize].0).or_insert(0.0) += degree[v as usize];
        }

        let mut passes = 0u32;
        let mut last_mod = modularity(g, &community, &degree, m2);

        for pass in 0..self.max_passes {
            passes = pass + 1;
            let mut moved = false;

            for v_idx in 0..n {
                let v = VertexId(v_idx as u32);
                let nbrs = g.neighbours(v).unwrap_or(&[]);
                let cur = community[v_idx];
                let k_i = degree[v_idx];

                // Sum of weights from v to each neighbouring community.
                let mut k_i_in: HashMap<u32, f64> = HashMap::new();
                for e in nbrs {
                    let other = match e.other(v) {
                        Some(o) => o,
                        None => continue,
                    };
                    let c = community[other.0 as usize].0;
                    *k_i_in.entry(c).or_insert(0.0) += f64::from(e.weight);
                }

                // Remove v from current community.
                if let Some(t) = comm_total.get_mut(&cur.0) {
                    *t -= k_i;
                }

                // Pick the best neighbouring community (including stay).
                let mut best_c = cur.0;
                let mut best_gain = 0.0_f64;
                for (&c, &k_i_c) in &k_i_in {
                    let sigma_tot = *comm_total.get(&c).unwrap_or(&0.0);
                    // Modularity gain of joining `c`.
                    let gain = k_i_c - sigma_tot * k_i / m2;
                    if gain > best_gain {
                        best_gain = gain;
                        best_c = c;
                    }
                }

                // Re-insert into chosen community.
                *comm_total.entry(best_c).or_insert(0.0) += k_i;
                if best_c != cur.0 {
                    community[v_idx] = Community(best_c);
                    moved = true;
                }
            }

            let new_mod = modularity(g, &community, &degree, m2);
            let delta = new_mod - last_mod;
            last_mod = new_mod;
            if !moved || delta < self.min_modularity_gain {
                break;
            }
        }

        LouvainResult {
            community,
            modularity: last_mod,
            passes,
        }
    }
}

/// Newman-Girvan modularity for the current labelling.
fn modularity(
    g: &FraudGraph,
    community: &[Community],
    degree: &[f64],
    m2: f64,
) -> f64 {
    if m2 <= 0.0 {
        return 0.0;
    }
    let mut q = 0.0_f64;
    for e in g.edges() {
        let ca = community[e.a.0 as usize];
        let cb = community[e.b.0 as usize];
        if ca == cb {
            let a_ij = f64::from(e.weight) * 2.0; // undirected stored once
            let expected = degree[e.a.0 as usize] * degree[e.b.0 as usize] / m2;
            q += a_ij - expected;
        }
    }
    // Diagonal: vertex with self-loop weight (we don't store self-loops,
    // but for completeness):
    // (none contributes for our graph)
    q / m2
}
