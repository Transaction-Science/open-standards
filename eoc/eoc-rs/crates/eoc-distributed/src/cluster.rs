//! Cluster membership, heartbeats, and failure detection.
//!
//! A [`Cluster`] is a collection of [`Node`]s. Each node wraps a worker
//! id, its last-seen heartbeat, and bookkeeping for the phi-accrual style
//! detector used in Akka / Cassandra. The detector is intentionally simple
//! (fixed timeout) but lives behind the [`HeartbeatDetector`] trait so a
//! more sophisticated implementation can swap in.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{DistributedError, Result};

/// Status of a node from the detector's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Heartbeat within budget.
    Alive,
    /// Heartbeat is late but not yet declared dead.
    Suspect,
    /// Heartbeat budget exceeded; the node is presumed dead.
    Dead,
}

/// A single heartbeat sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heartbeat {
    /// Local monotonic clock at which the heartbeat was received.
    pub at: Instant,
}

impl Heartbeat {
    /// Capture "now".
    pub fn now() -> Self {
        Self { at: Instant::now() }
    }
}

/// One member of the cluster.
#[derive(Debug, Clone)]
pub struct Node {
    /// Worker id (matches [`crate::worker::Worker::id`]).
    pub id: String,
    /// Last received heartbeat.
    pub last_heartbeat: Heartbeat,
    /// Cached status from the detector.
    pub status: NodeStatus,
}

impl Node {
    /// Construct a freshly-alive node.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            last_heartbeat: Heartbeat::now(),
            status: NodeStatus::Alive,
        }
    }
}

/// A failure-detector strategy. The default implementation is a fixed
/// suspect / dead budget, matching the behaviour of Ray's GCS-side
/// detector well enough for most deployments.
pub trait HeartbeatDetector {
    /// Classify `now - last_heartbeat`.
    fn classify(&self, gap: Duration) -> NodeStatus;
}

/// Two-level fixed-budget detector: suspect after `suspect_after`, dead
/// after `dead_after`.
#[derive(Debug, Clone, Copy)]
pub struct FixedBudgetDetector {
    /// After this much silence, the node is moved to [`NodeStatus::Suspect`].
    pub suspect_after: Duration,
    /// After this much silence, the node is moved to [`NodeStatus::Dead`].
    pub dead_after: Duration,
}

impl Default for FixedBudgetDetector {
    fn default() -> Self {
        Self {
            suspect_after: Duration::from_secs(2),
            dead_after: Duration::from_secs(10),
        }
    }
}

impl HeartbeatDetector for FixedBudgetDetector {
    fn classify(&self, gap: Duration) -> NodeStatus {
        if gap >= self.dead_after {
            NodeStatus::Dead
        } else if gap >= self.suspect_after {
            NodeStatus::Suspect
        } else {
            NodeStatus::Alive
        }
    }
}

/// A cluster of worker nodes.
#[derive(Debug)]
pub struct Cluster<D: HeartbeatDetector = FixedBudgetDetector> {
    nodes: HashMap<String, Node>,
    detector: D,
}

impl Cluster<FixedBudgetDetector> {
    /// Construct a cluster with the default fixed-budget detector.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            detector: FixedBudgetDetector::default(),
        }
    }
}

impl Default for Cluster<FixedBudgetDetector> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: HeartbeatDetector> Cluster<D> {
    /// Construct with an explicit detector.
    pub fn with_detector(detector: D) -> Self {
        Self {
            nodes: HashMap::new(),
            detector,
        }
    }

    /// Register a new node. Replaces any previous entry with the same id.
    pub fn register(&mut self, node: Node) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Record a heartbeat for `id`. Errors if the node is unknown.
    pub fn heartbeat(&mut self, id: &str) -> Result<()> {
        let n = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| DistributedError::UnknownWorker(id.to_string()))?;
        n.last_heartbeat = Heartbeat::now();
        n.status = NodeStatus::Alive;
        Ok(())
    }

    /// Re-classify every node against the detector. Returns the count
    /// transitioned to [`NodeStatus::Dead`] this tick.
    pub fn tick(&mut self) -> usize {
        let now = Instant::now();
        let mut newly_dead = 0;
        for node in self.nodes.values_mut() {
            let gap = now.duration_since(node.last_heartbeat.at);
            let next = self.detector.classify(gap);
            if next == NodeStatus::Dead && node.status != NodeStatus::Dead {
                newly_dead += 1;
            }
            node.status = next;
        }
        newly_dead
    }

    /// Iterate every node.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Iterate every node currently considered alive.
    pub fn alive(&self) -> impl Iterator<Item = &Node> {
        self.nodes
            .values()
            .filter(|n| n.status == NodeStatus::Alive)
    }

    /// Look up a node by id.
    pub fn get(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Remove a node from the cluster.
    pub fn drop_node(&mut self, id: &str) -> Result<()> {
        if self.nodes.remove(id).is_none() {
            return Err(DistributedError::UnknownWorker(id.to_string()));
        }
        Ok(())
    }

    /// Count of registered nodes regardless of status.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the cluster has any nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_classifies_gaps() {
        let d = FixedBudgetDetector {
            suspect_after: Duration::from_millis(100),
            dead_after: Duration::from_millis(500),
        };
        assert_eq!(d.classify(Duration::from_millis(0)), NodeStatus::Alive);
        assert_eq!(d.classify(Duration::from_millis(200)), NodeStatus::Suspect);
        assert_eq!(d.classify(Duration::from_millis(800)), NodeStatus::Dead);
    }

    #[test]
    fn register_then_heartbeat() {
        let mut c = Cluster::new();
        c.register(Node::new("w-0"));
        assert!(c.heartbeat("w-0").is_ok());
        assert!(c.heartbeat("nope").is_err());
    }

    #[test]
    fn drop_node_removes() {
        let mut c = Cluster::new();
        c.register(Node::new("w-0"));
        assert_eq!(c.len(), 1);
        c.drop_node("w-0").expect("ok");
        assert_eq!(c.len(), 0);
        assert!(c.drop_node("w-0").is_err());
    }
}
