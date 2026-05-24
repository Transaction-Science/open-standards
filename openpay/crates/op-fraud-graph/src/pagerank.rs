//! PageRank — centrality on the undirected weighted fraud graph.
//!
//! In a fraud context PageRank reads as: *which entity is the hub of
//! its ring?* The mule account through which many addresses flow has a
//! higher PageRank than each address individually. We use it to rank
//! the "next account to investigate" in a cluster.
//!
//! ## Algorithm
//!
//! Power iteration with damping. Edge weights bias the random-walk
//! transition probabilities. Converges when the L1 norm of the score
//! delta drops below `tol`, or `max_iter` is exhausted (we still
//! return the partial result with [`Error::NotConverged`]).

use crate::error::{Error, Result};
use crate::graph::{FraudGraph, VertexId};

/// Configuration and runner for PageRank.
#[derive(Debug, Clone, Copy)]
pub struct PageRank {
    /// Damping factor — probability of following a link vs. teleporting.
    /// 0.85 is the canonical value from the original Brin-Page paper.
    pub damping: f32,
    /// Convergence tolerance (L1 norm of the delta).
    pub tol: f32,
    /// Iteration ceiling so we always terminate.
    pub max_iter: u32,
}

impl Default for PageRank {
    fn default() -> Self {
        Self {
            damping: 0.85,
            tol: 1e-6,
            max_iter: 100,
        }
    }
}

/// Result of one PageRank run: per-vertex score in `[0, 1]`, summing to 1.
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// Score per vertex, indexed by `VertexId.0 as usize`.
    pub scores: Vec<f32>,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether the L1 delta dropped below `tol`.
    pub converged: bool,
}

impl PageRankResult {
    /// Score for a specific vertex.
    pub fn score(&self, v: VertexId) -> Option<f32> {
        self.scores.get(v.0 as usize).copied()
    }

    /// Top-`k` vertices by score.
    pub fn top(&self, k: usize) -> Vec<(VertexId, f32)> {
        let mut v: Vec<(VertexId, f32)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (VertexId(i as u32), s))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        v.truncate(k);
        v
    }
}

impl PageRank {
    /// Validate the config (returns [`Error::InvalidConfig`] if damping
    /// is out of `(0,1)` or `tol`/`max_iter` are degenerate).
    pub fn validate(&self) -> Result<()> {
        if !(self.damping > 0.0 && self.damping < 1.0) {
            return Err(Error::InvalidConfig("damping must be in (0, 1)"));
        }
        if !(self.tol > 0.0 && self.tol.is_finite()) {
            return Err(Error::InvalidConfig("tol must be a finite positive number"));
        }
        if self.max_iter == 0 {
            return Err(Error::InvalidConfig("max_iter must be > 0"));
        }
        Ok(())
    }

    /// Run PageRank against `g`.
    pub fn run(&self, g: &FraudGraph) -> Result<PageRankResult> {
        self.validate()?;
        let n = g.vertex_count();
        if n == 0 {
            return Ok(PageRankResult {
                scores: Vec::new(),
                iterations: 0,
                converged: true,
            });
        }

        let n_f = n as f32;
        let mut scores = vec![1.0_f32 / n_f; n];
        let mut next = vec![0.0_f32; n];

        // Precompute weighted out-degree (== in-degree on undirected graph).
        let mut weight_sum: Vec<f32> = vec![0.0; n];
        for v in g.vertices() {
            // `neighbours` only fails on an unknown vertex id; we
            // enumerated them ourselves so this can't fire — but encode
            // the safety as a defaulted empty slice anyway.
            let nbrs = g.neighbours(v).unwrap_or(&[]);
            let mut s = 0.0_f32;
            for e in nbrs {
                s += e.weight;
            }
            weight_sum[v.0 as usize] = s;
        }

        let teleport = (1.0 - self.damping) / n_f;
        let mut converged = false;
        let mut iters = 0u32;

        for it in 0..self.max_iter {
            iters = it + 1;
            for slot in &mut next {
                *slot = teleport;
            }

            // Push contributions from each vertex to its neighbours.
            for v in 0..n as u32 {
                let s = weight_sum[v as usize];
                let pr = scores[v as usize];
                if s <= 0.0 || pr <= 0.0 {
                    // Sink: distribute uniformly (handle dangling nodes).
                    let share = self.damping * pr / n_f;
                    for slot in &mut next {
                        *slot += share;
                    }
                    continue;
                }
                let nbrs = g.neighbours(VertexId(v)).unwrap_or(&[]);
                for e in nbrs {
                    let other = e.other(VertexId(v)).map_or(v, |o| o.0);
                    let contribution = self.damping * pr * (e.weight / s);
                    next[other as usize] += contribution;
                }
            }

            // L1 delta.
            let mut delta = 0.0_f32;
            for i in 0..n {
                delta += (next[i] - scores[i]).abs();
            }
            core::mem::swap(&mut scores, &mut next);
            if delta < self.tol {
                converged = true;
                break;
            }
        }

        Ok(PageRankResult {
            scores,
            iterations: iters,
            converged,
        })
    }
}
