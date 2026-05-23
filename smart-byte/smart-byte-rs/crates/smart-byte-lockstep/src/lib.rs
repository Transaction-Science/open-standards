//! Deterministic lockstep BFT cluster simulation (MVP).
//!
//! Every [`Node`] in a [`Cluster`] runs the *same* state machine: a
//! BLAKE3 rolling hash that absorbs the canonical-CBOR encoding of each
//! ordered transition. After each [`Frame`] every honest node will
//! independently arrive at the same `state_hash`. A frame commits when
//! a BFT supermajority (`> 2/3` of nodes) vote for the same hash.
//!
//! Faulty nodes are modeled as nodes that *flip a bit* in their state
//! before voting; the cluster still commits if at least `floor(2n/3)+1`
//! honest nodes agree.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use smart_byte_core::{Blake3Hash, Said, ownership::Transition};

/// Errors during a frame step.
#[derive(Debug, thiserror::Error)]
pub enum LockstepError {
    #[error("cluster is empty")]
    EmptyCluster,
    #[error("no supermajority on frame {frame}: best vote {best}/{total}")]
    NoSupermajority {
        frame: u64,
        best: usize,
        total: usize,
    },
}

/// A single ordered batch of state transitions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub frame_number: u64,
    pub ordered_transitions: Vec<Transition>,
}

/// A simulated node in the cluster.
#[derive(Clone, Debug)]
pub struct Node {
    /// Stable identity.
    pub id: Said,
    /// Rolling state hash. Each frame absorbs into this.
    pub state: Blake3Hash,
    /// If true, this node introduces a deterministic per-frame bit-flip
    /// before voting. Used to model Byzantine misbehavior.
    pub faulty: bool,
}

impl Node {
    /// Construct a fresh node with all-zero state.
    pub fn new(id: Said) -> Self {
        Self {
            id,
            state: [0u8; 32],
            faulty: false,
        }
    }

    /// Mark this node as faulty.
    pub fn with_faulty(mut self, faulty: bool) -> Self {
        self.faulty = faulty;
        self
    }

    /// Compute this node's vote for the given frame, advancing its
    /// internal state to the post-frame hash. Faulty nodes flip the
    /// low bit of the announced vote (but still update their honest
    /// state internally; in a real cluster the state divergence would
    /// be the observable signal).
    fn step(&mut self, frame: &Frame) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.state);
        hasher.update(&frame.frame_number.to_le_bytes());
        for t in &frame.ordered_transitions {
            let bytes = serde_cbor::to_vec(t).expect("Transition CBOR is infallible");
            hasher.update(&bytes);
        }
        let honest = *hasher.finalize().as_bytes();
        self.state = honest;
        if self.faulty {
            let mut bad = honest;
            bad[0] ^= 0x01;
            bad
        } else {
            honest
        }
    }
}

/// The committed result of one lockstep frame.
#[derive(Clone, Debug)]
pub struct FrameCommit {
    pub frame: u64,
    pub state_hash: Blake3Hash,
    pub committed_by: HashSet<Said>,
}

/// A federated cluster of [`Node`]s running deterministic lockstep.
pub struct Cluster {
    pub nodes: Vec<Node>,
    pub frame: u64,
}

impl Cluster {
    /// Build a cluster of `n` honest nodes.
    pub fn new_honest(n: usize) -> Self {
        let nodes = (0..n)
            .map(|i| {
                let id = Said::hash(format!("node-{i}").as_bytes());
                Node::new(id)
            })
            .collect();
        Self { nodes, frame: 0 }
    }

    /// Build a cluster of `n` nodes where `faulty_count` of them are
    /// Byzantine. The faulty nodes are the *last* `faulty_count`
    /// entries; honest nodes occupy indices `0..n-faulty_count`.
    pub fn new_with_faulty(n: usize, faulty_count: usize) -> Self {
        let mut c = Self::new_honest(n);
        for i in (n - faulty_count)..n {
            c.nodes[i].faulty = true;
        }
        c
    }

    /// Run one lockstep frame. Every node steps; votes are tallied;
    /// the modal hash wins iff it has a `> 2/3` supermajority.
    pub fn step(&mut self, transitions: Vec<Transition>) -> Result<FrameCommit, LockstepError> {
        if self.nodes.is_empty() {
            return Err(LockstepError::EmptyCluster);
        }
        let frame = Frame {
            frame_number: self.frame,
            ordered_transitions: transitions,
        };
        let total = self.nodes.len();
        let mut tally: HashMap<Blake3Hash, HashSet<Said>> = HashMap::new();
        for node in &mut self.nodes {
            let vote = node.step(&frame);
            tally.entry(vote).or_default().insert(node.id);
        }
        // Pick the hash with the largest backing set.
        let (best_hash, voters) = tally
            .into_iter()
            .max_by_key(|(_, voters)| voters.len())
            .expect("non-empty cluster has at least one vote");
        // Strict > 2/3 (BFT supermajority).
        let needed = (2 * total) / 3 + 1;
        if voters.len() < needed {
            return Err(LockstepError::NoSupermajority {
                frame: frame.frame_number,
                best: voters.len(),
                total,
            });
        }
        let commit = FrameCommit {
            frame: frame.frame_number,
            state_hash: best_hash,
            committed_by: voters,
        };
        self.frame += 1;
        Ok(commit)
    }
}

/// Helper: synthesize a deterministic sequence of transitions for a
/// frame index. Used by tests and the CLI demo so that the demo output
/// is reproducible.
pub fn synthetic_transitions(frame_index: u64, count: usize) -> Vec<Transition> {
    (0..count)
        .map(|i| {
            let from = Said::hash(format!("from-{frame_index}-{i}").as_bytes());
            let to = Said::hash(format!("to-{frame_index}-{i}").as_bytes());
            let prior = if i == 0 {
                None
            } else {
                Some(Said::hash(format!("prior-{frame_index}-{i}").as_bytes()).0)
            };
            Transition::unsigned(from, to, prior)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_node_honest_cluster_converges() {
        let mut cluster = Cluster::new_honest(4);
        let mut commits = Vec::new();
        for f in 0..10 {
            let commit = cluster.step(synthetic_transitions(f, 3)).unwrap();
            assert_eq!(commit.committed_by.len(), 4);
            commits.push(commit);
        }
        // All ten frames should have committed under unanimous votes.
        assert_eq!(commits.len(), 10);
    }

    #[test]
    fn one_faulty_node_still_commits() {
        let mut cluster = Cluster::new_with_faulty(4, 1);
        // 3-of-4 honest nodes is exactly the floor(2*4/3)+1 = 3 threshold.
        for f in 0..10 {
            let commit = cluster.step(synthetic_transitions(f, 3)).unwrap();
            assert_eq!(commit.committed_by.len(), 3);
        }
    }

    #[test]
    fn two_faulty_nodes_in_four_breaks_supermajority() {
        // 2 honest, 2 faulty: best is 2/4 which is below the >2/3 = 3/4 threshold.
        let mut cluster = Cluster::new_with_faulty(4, 2);
        let err = cluster.step(synthetic_transitions(0, 3)).unwrap_err();
        assert!(matches!(err, LockstepError::NoSupermajority { .. }));
    }

    #[test]
    fn determinism_across_clusters() {
        let mut a = Cluster::new_honest(4);
        let mut b = Cluster::new_honest(4);
        for f in 0..5 {
            let ca = a.step(synthetic_transitions(f, 2)).unwrap();
            let cb = b.step(synthetic_transitions(f, 2)).unwrap();
            assert_eq!(ca.state_hash, cb.state_hash);
        }
    }
}
