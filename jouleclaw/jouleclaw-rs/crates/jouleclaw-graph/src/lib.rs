//! Persisted run-state for JouleClaw — a durable, joule-stamped DAG that
//! survives the process. The shape the field has converged on for
//! multi-step agentic work (LangGraph-shape): nodes are tiers / agents /
//! stages, edges are typed handoffs, the run records which nodes have
//! completed, at what joule cost, against what energy-counter honesty
//! tier.
//!
//! This crate is the **data model + state machine**, not a driver. The
//! consumer (`jouleclaw-stack::agent`, a custom orchestrator, an MCP
//! Tasks server) decides which ready node to dispatch next and records
//! the result. Decoupling means the same persisted run survives a
//! restart, a tool-gateway swap, a new model release.
//!
//! ## Honest scope (v1)
//!
//! - Linear, branch, and join. Two merge strategies on a join node:
//!   [`MergeStrategy::AllRequired`] (every predecessor must complete —
//!   the natural fan-in) and [`MergeStrategy::AnyOne`] (first-completion
//!   wins — race semantics).
//! - **Acyclic by validation.** Cycles would mean retrying inside the
//!   graph; that is a separate concern from "did this step finish."
//!   The brief excluded time-travel replay in v1; this is its
//!   structural enforcement.
//! - **No human-in-the-loop checkpoints.** A pending HITL approval is
//!   modelled by the consumer holding a node un-recorded; the graph
//!   doesn't enforce a "waiting for approval" state in v1.
//! - **Driver-less.** The consumer dispatches; this crate validates,
//!   tracks, and persists.

#![forbid(unsafe_code)]

use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────
// Identity + kinds
// ─────────────────────────────────────────────────────────────────────

/// Stable identifier for a node within one run. Caller-assigned (not
/// generated) so the graph's wire form is deterministic.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// What a node represents. The discriminator is wire-stable; callers
/// pick the shape that fits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum NodeKind {
    /// A cascade tier identified by wire tag (e.g. `"L0"`, `"L3"`, `"L4.5"`).
    Tier(String),
    /// A named agent in the host's registry.
    Agent(String),
    /// A pipeline stage (retriever, extractor, reranker, …).
    Stage(String),
    /// Caller-defined. The string is the discriminator.
    Custom(String),
}

/// One node in the run plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    /// Short human-readable label.
    pub label: String,
    /// How to handle merging when multiple incoming edges arrive.
    /// Linear (single predecessor) nodes use any strategy without effect.
    #[serde(default)]
    pub merge: MergeStrategy,
}

/// Join semantics for nodes with more than one incoming edge.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// All incoming predecessors must complete before this node is
    /// ready. The natural fan-in / barrier semantics. Default.
    #[default]
    AllRequired,
    /// Ready as soon as ANY predecessor completes. First-completion-wins
    /// race semantics; the remaining predecessors are not awaited.
    AnyOne,
}

/// A directed edge in the run plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    /// Optional label so the consumer can name a branch (`"happy"`,
    /// `"retry"`, `"compensate"`). Round-trips through wire form; not
    /// used by the graph's state machine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────
// RunGraph (the static plan)
// ─────────────────────────────────────────────────────────────────────

/// The static plan for a run: a directed acyclic graph with an entry
/// node and one or more terminals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunGraph {
    pub entry: NodeId,
    pub terminals: Vec<NodeId>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// Errors building or validating a graph.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GraphError {
    #[error("entry node {0} not present in nodes")]
    EntryMissing(NodeId),
    #[error("terminal node {0} not present in nodes")]
    TerminalMissing(NodeId),
    #[error("no terminal nodes declared")]
    NoTerminals,
    #[error("duplicate node id {0}")]
    DuplicateNode(NodeId),
    #[error("edge references unknown node {0}")]
    UnknownNode(NodeId),
    #[error("graph contains a cycle (at least one node is reachable from itself)")]
    Cycle,
    #[error("node {0} is unreachable from entry")]
    Unreachable(NodeId),
}

impl RunGraph {
    /// Validate structural invariants: every reference resolves, there
    /// is at least one terminal, no cycles, every node is reachable from
    /// the entry. Returns every error found in one pass.
    pub fn validate(&self) -> Result<(), Vec<GraphError>> {
        let mut errors = Vec::new();

        // Duplicate ids.
        let mut seen: HashSet<NodeId> = HashSet::with_capacity(self.nodes.len());
        for n in &self.nodes {
            if !seen.insert(n.id) {
                errors.push(GraphError::DuplicateNode(n.id));
            }
        }

        let ids: HashSet<NodeId> = self.nodes.iter().map(|n| n.id).collect();

        // Entry + terminals exist.
        if !ids.contains(&self.entry) {
            errors.push(GraphError::EntryMissing(self.entry));
        }
        if self.terminals.is_empty() {
            errors.push(GraphError::NoTerminals);
        }
        for t in &self.terminals {
            if !ids.contains(t) {
                errors.push(GraphError::TerminalMissing(*t));
            }
        }

        // Edges resolve.
        for e in &self.edges {
            if !ids.contains(&e.from) {
                errors.push(GraphError::UnknownNode(e.from));
            }
            if !ids.contains(&e.to) {
                errors.push(GraphError::UnknownNode(e.to));
            }
        }

        if errors.is_empty() {
            // Acyclic + reachable: depth-first colour walk.
            if has_cycle(self) {
                errors.push(GraphError::Cycle);
            }
            if errors.is_empty() {
                let reachable = reachable_from(self, self.entry);
                for n in &self.nodes {
                    if !reachable.contains(&n.id) {
                        errors.push(GraphError::Unreachable(n.id));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Direct successors of `n` in the plan.
    pub fn successors(&self, n: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter_map(|e| (e.from == n).then_some(e.to))
            .collect()
    }

    /// Direct predecessors of `n` in the plan.
    pub fn predecessors(&self, n: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter_map(|e| (e.to == n).then_some(e.from))
            .collect()
    }

    /// Lookup the [`Node`] record for `id`.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

fn has_cycle(g: &RunGraph) -> bool {
    // Iterative DFS with three colours: 0=white, 1=grey, 2=black.
    let mut color: HashMap<NodeId, u8> = HashMap::with_capacity(g.nodes.len());
    for n in &g.nodes {
        color.insert(n.id, 0);
    }
    for n in &g.nodes {
        if color.get(&n.id).copied() != Some(0) {
            continue;
        }
        // Stack of (node, iter index over its successors).
        let mut stack: Vec<(NodeId, usize)> = vec![(n.id, 0)];
        color.insert(n.id, 1);
        while let Some((cur, idx)) = stack.last_mut().copied() {
            let succs = g.successors(cur);
            if idx >= succs.len() {
                // Done with cur.
                color.insert(cur, 2);
                stack.pop();
                continue;
            }
            let next = succs[idx];
            // Advance the iterator on the stack frame.
            stack.last_mut().unwrap().1 = idx + 1;
            match color.get(&next).copied().unwrap_or(0) {
                1 => return true, // back-edge → cycle
                0 => {
                    color.insert(next, 1);
                    stack.push((next, 0));
                }
                _ => {} // 2: already done, skip
            }
        }
    }
    false
}

fn reachable_from(g: &RunGraph, start: NodeId) -> HashSet<NodeId> {
    let mut out = HashSet::new();
    let mut stack = vec![start];
    while let Some(n) = stack.pop() {
        if !out.insert(n) {
            continue;
        }
        for s in g.successors(n) {
            stack.push(s);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Checkpoints + Run
// ─────────────────────────────────────────────────────────────────────

/// One joule-stamped completion record for a node in a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub node_id: NodeId,
    pub joules_uj: u64,
    pub energy_provenance: Provenance,
    /// Hash of the node's output bytes — opaque to the graph; the
    /// consumer chooses the hash (blake3 hex is the JouleClaw default).
    pub output_hash: String,
    /// Wall-clock unix seconds when the checkpoint was recorded.
    pub recorded_at: u64,
}

impl Checkpoint {
    pub fn new(
        node_id: NodeId,
        joules_uj: u64,
        energy_provenance: Provenance,
        output_hash: impl Into<String>,
        recorded_at: u64,
    ) -> Self {
        Self {
            node_id,
            joules_uj,
            energy_provenance,
            output_hash: output_hash.into(),
            recorded_at,
        }
    }
}

/// Overall status of a run.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    #[default]
    InProgress,
    Completed,
    Failed,
}

/// One execution of a [`RunGraph`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub run_id: String,
    pub graph: RunGraph,
    pub checkpoints: Vec<Checkpoint>,
    pub status: RunStatus,
}

/// Errors when recording checkpoints against a run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunError {
    #[error("graph validation failed: {0:?}")]
    InvalidGraph(Vec<GraphError>),
    #[error("checkpoint for unknown node {0}")]
    UnknownNode(NodeId),
    #[error("node {0} is not ready (predecessors incomplete)")]
    NotReady(NodeId),
    #[error("node {0} already recorded")]
    AlreadyRecorded(NodeId),
    #[error("run is no longer in progress (status: {0:?})")]
    NotInProgress(RunStatus),
}

impl Run {
    /// Start a run against a validated graph. Validation happens once,
    /// up front; subsequent reads trust the structure.
    pub fn start(graph: RunGraph, run_id: impl Into<String>) -> Result<Self, RunError> {
        graph.validate().map_err(RunError::InvalidGraph)?;
        Ok(Self {
            run_id: run_id.into(),
            graph,
            checkpoints: Vec::new(),
            status: RunStatus::InProgress,
        })
    }

    /// Set of node ids that have a checkpoint.
    pub fn completed_nodes(&self) -> HashSet<NodeId> {
        self.checkpoints.iter().map(|c| c.node_id).collect()
    }

    /// Nodes that are now ready to dispatch — present in the graph, not
    /// yet completed, and whose merge strategy is satisfied. The entry
    /// node is ready until it is recorded.
    pub fn ready_nodes(&self) -> Vec<NodeId> {
        let done = self.completed_nodes();
        let mut out = Vec::new();
        for n in &self.graph.nodes {
            if done.contains(&n.id) {
                continue;
            }
            let preds = self.graph.predecessors(n.id);
            let ready = if preds.is_empty() {
                // No predecessors → only the entry is ready by
                // construction. Validation guarantees a single entry.
                n.id == self.graph.entry
            } else {
                match n.merge {
                    MergeStrategy::AllRequired => preds.iter().all(|p| done.contains(p)),
                    MergeStrategy::AnyOne => preds.iter().any(|p| done.contains(p)),
                }
            };
            if ready {
                out.push(n.id);
            }
        }
        out.sort();
        out
    }

    /// Record a completed node. Updates the running status when all
    /// terminals are done.
    pub fn record(&mut self, checkpoint: Checkpoint) -> Result<(), RunError> {
        if self.status != RunStatus::InProgress {
            return Err(RunError::NotInProgress(self.status));
        }
        if self.graph.node(checkpoint.node_id).is_none() {
            return Err(RunError::UnknownNode(checkpoint.node_id));
        }
        if self.completed_nodes().contains(&checkpoint.node_id) {
            return Err(RunError::AlreadyRecorded(checkpoint.node_id));
        }
        if !self.ready_nodes().contains(&checkpoint.node_id) {
            return Err(RunError::NotReady(checkpoint.node_id));
        }
        self.checkpoints.push(checkpoint);
        // Run completes when every terminal has a checkpoint.
        let done = self.completed_nodes();
        if self.graph.terminals.iter().all(|t| done.contains(t)) {
            self.status = RunStatus::Completed;
        }
        Ok(())
    }

    /// Mark the run as failed (consumer decides when; e.g. a verifier
    /// rejected a critical node's output).
    pub fn fail(&mut self) {
        self.status = RunStatus::Failed;
    }

    /// Sum of microjoules across all checkpoints.
    pub fn total_joules_uj(&self) -> u64 {
        self.checkpoints
            .iter()
            .fold(0u64, |a, c| a.saturating_add(c.joules_uj))
    }

    /// Worst (lowest-honesty) energy provenance seen across checkpoints.
    /// Matches `jouleclaw-prov`'s "worst counter wins" floor rule.
    pub fn worst_provenance(&self) -> Provenance {
        let mut worst: Option<Provenance> = None;
        for c in &self.checkpoints {
            worst = Some(match worst {
                None => c.energy_provenance,
                Some(prev) => worst_provenance(prev, c.energy_provenance),
            });
        }
        worst.unwrap_or(Provenance::Estimator)
    }

    /// Convenience: run reached every terminal.
    pub fn is_done(&self) -> bool {
        self.status == RunStatus::Completed
    }
}

fn worst_provenance(a: Provenance, b: Provenance) -> Provenance {
    // Lower honesty wins. HwShunt > ModelBased > Estimator.
    let rank = |p: Provenance| match p {
        Provenance::HwShunt => 2,
        Provenance::ModelBased => 1,
        Provenance::Estimator => 0,
    };
    if rank(a) <= rank(b) {
        a
    } else {
        b
    }
}

// ─────────────────────────────────────────────────────────────────────
// Persistence
// ─────────────────────────────────────────────────────────────────────

/// Persistence interface for runs. The graph's reason for being is
/// surviving the process — a run started before a restart resumes from
/// its checkpoints by loading through this trait.
pub trait RunStore: Send {
    fn save(&mut self, run: &Run);
    fn load(&self, run_id: &str) -> Option<Run>;
    fn list(&self) -> Vec<String>;
}

/// In-memory reference store, for tests and single-process consumers.
/// Disk-backed implementations (file-per-run, sled, sqlite) plug in
/// through the [`RunStore`] trait.
#[derive(Debug, Default)]
pub struct InMemoryRunStore {
    runs: BTreeMap<String, Run>,
}

impl InMemoryRunStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl RunStore for InMemoryRunStore {
    fn save(&mut self, run: &Run) {
        self.runs.insert(run.run_id.clone(), run.clone());
    }
    fn load(&self, run_id: &str) -> Option<Run> {
        self.runs.get(run_id).cloned()
    }
    fn list(&self) -> Vec<String> {
        self.runs.keys().cloned().collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: u32, label: &str) -> Node {
        Node {
            id: NodeId(id),
            kind: NodeKind::Stage(label.to_string()),
            label: label.to_string(),
            merge: MergeStrategy::AllRequired,
        }
    }

    fn n_merge(id: u32, label: &str, merge: MergeStrategy) -> Node {
        Node {
            id: NodeId(id),
            kind: NodeKind::Stage(label.to_string()),
            label: label.to_string(),
            merge,
        }
    }

    fn edge(from: u32, to: u32) -> Edge {
        Edge {
            from: NodeId(from),
            to: NodeId(to),
            label: None,
        }
    }

    fn ck(id: u32, joules: u64, prov: Provenance) -> Checkpoint {
        Checkpoint::new(NodeId(id), joules, prov, "blake3:x", 1_700_000_000)
    }

    // ─── Validation ───────────────────────────────────────────────

    #[test]
    fn linear_graph_validates() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(3)],
            nodes: vec![n(1, "A"), n(2, "B"), n(3, "C")],
            edges: vec![edge(1, 2), edge(2, 3)],
        };
        assert!(g.validate().is_ok());
    }

    #[test]
    fn cycle_is_rejected() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2), edge(2, 1)],
        };
        let errs = g.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, GraphError::Cycle)));
    }

    #[test]
    fn unreachable_node_is_rejected() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B"), n(99, "orphan")],
            edges: vec![edge(1, 2)],
        };
        let errs = g.validate().unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, GraphError::Unreachable(id) if *id == NodeId(99))));
    }

    #[test]
    fn missing_entry_is_rejected() {
        let g = RunGraph {
            entry: NodeId(99),
            terminals: vec![NodeId(1)],
            nodes: vec![n(1, "A")],
            edges: vec![],
        };
        let errs = g.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, GraphError::EntryMissing(_))));
    }

    #[test]
    fn empty_terminals_rejected() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![],
            nodes: vec![n(1, "A")],
            edges: vec![],
        };
        let errs = g.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, GraphError::NoTerminals)));
    }

    #[test]
    fn duplicate_node_id_is_rejected() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(1)],
            nodes: vec![n(1, "A"), n(1, "A-again")],
            edges: vec![],
        };
        let errs = g.validate().unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, GraphError::DuplicateNode(_))));
    }

    #[test]
    fn edge_to_unknown_node_is_rejected() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(1)],
            nodes: vec![n(1, "A")],
            edges: vec![edge(1, 42)],
        };
        let errs = g.validate().unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, GraphError::UnknownNode(id) if *id == NodeId(42))));
    }

    // ─── Run state machine ────────────────────────────────────────

    #[test]
    fn linear_run_walks_entry_to_terminal() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(3)],
            nodes: vec![n(1, "A"), n(2, "B"), n(3, "C")],
            edges: vec![edge(1, 2), edge(2, 3)],
        };
        let mut run = Run::start(g, "r-1").expect("start");
        assert_eq!(run.ready_nodes(), vec![NodeId(1)]);
        assert!(!run.is_done());

        run.record(ck(1, 100, Provenance::HwShunt)).unwrap();
        assert_eq!(run.ready_nodes(), vec![NodeId(2)]);

        run.record(ck(2, 50, Provenance::HwShunt)).unwrap();
        assert_eq!(run.ready_nodes(), vec![NodeId(3)]);

        run.record(ck(3, 25, Provenance::HwShunt)).unwrap();
        assert!(run.is_done());
        assert_eq!(run.total_joules_uj(), 175);
    }

    #[test]
    fn branch_and_join_all_required() {
        // A → B, A → C, B + C → D
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(4)],
            nodes: vec![
                n(1, "A"),
                n(2, "B"),
                n(3, "C"),
                n_merge(4, "D", MergeStrategy::AllRequired),
            ],
            edges: vec![edge(1, 2), edge(1, 3), edge(2, 4), edge(3, 4)],
        };
        let mut run = Run::start(g, "r-2").unwrap();
        run.record(ck(1, 10, Provenance::HwShunt)).unwrap();
        // After A: B and C are both ready.
        let r = run.ready_nodes();
        assert_eq!(r, vec![NodeId(2), NodeId(3)]);
        // D is NOT ready until both B and C complete.
        run.record(ck(2, 20, Provenance::HwShunt)).unwrap();
        assert_eq!(run.ready_nodes(), vec![NodeId(3)]); // only C left
        run.record(ck(3, 30, Provenance::HwShunt)).unwrap();
        assert_eq!(run.ready_nodes(), vec![NodeId(4)]);
        run.record(ck(4, 40, Provenance::HwShunt)).unwrap();
        assert!(run.is_done());
    }

    #[test]
    fn branch_and_join_any_one_first_wins() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(4)],
            nodes: vec![
                n(1, "A"),
                n(2, "B"),
                n(3, "C"),
                n_merge(4, "D", MergeStrategy::AnyOne),
            ],
            edges: vec![edge(1, 2), edge(1, 3), edge(2, 4), edge(3, 4)],
        };
        let mut run = Run::start(g, "r-3").unwrap();
        run.record(ck(1, 10, Provenance::HwShunt)).unwrap();
        // B completes first; D should be ready even before C.
        run.record(ck(2, 20, Provenance::HwShunt)).unwrap();
        assert!(run.ready_nodes().contains(&NodeId(4)));
        // Recording D completes the run; C is unreachable per the
        // first-completion-wins semantics but the graph is still
        // structurally satisfied.
        run.record(ck(4, 40, Provenance::HwShunt)).unwrap();
        assert!(run.is_done());
    }

    #[test]
    fn cannot_record_unknown_node() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(1)],
            nodes: vec![n(1, "A")],
            edges: vec![],
        };
        let mut run = Run::start(g, "r-4").unwrap();
        let err = run.record(ck(99, 1, Provenance::HwShunt)).unwrap_err();
        assert!(matches!(err, RunError::UnknownNode(_)));
    }

    #[test]
    fn cannot_record_a_node_twice() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "r-5").unwrap();
        run.record(ck(1, 1, Provenance::HwShunt)).unwrap();
        let err = run.record(ck(1, 1, Provenance::HwShunt)).unwrap_err();
        assert!(matches!(err, RunError::AlreadyRecorded(_)));
    }

    #[test]
    fn cannot_record_node_whose_predecessors_unfinished() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "r-6").unwrap();
        let err = run.record(ck(2, 1, Provenance::HwShunt)).unwrap_err();
        assert!(matches!(err, RunError::NotReady(_)));
    }

    #[test]
    fn worst_provenance_is_the_floor() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(3)],
            nodes: vec![n(1, "A"), n(2, "B"), n(3, "C")],
            edges: vec![edge(1, 2), edge(2, 3)],
        };
        let mut run = Run::start(g, "r-7").unwrap();
        run.record(ck(1, 1, Provenance::HwShunt)).unwrap();
        run.record(ck(2, 1, Provenance::Estimator)).unwrap(); // floor
        run.record(ck(3, 1, Provenance::ModelBased)).unwrap();
        assert_eq!(run.worst_provenance(), Provenance::Estimator);
    }

    #[test]
    fn fail_blocks_further_record() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "r-8").unwrap();
        run.record(ck(1, 1, Provenance::HwShunt)).unwrap();
        run.fail();
        let err = run.record(ck(2, 1, Provenance::HwShunt)).unwrap_err();
        assert!(matches!(err, RunError::NotInProgress(RunStatus::Failed)));
    }

    // ─── Persistence ──────────────────────────────────────────────

    #[test]
    fn in_memory_run_store_round_trips() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "persist-1").unwrap();
        run.record(ck(1, 100, Provenance::HwShunt)).unwrap();

        let mut store = InMemoryRunStore::new();
        store.save(&run);

        let loaded = store.load("persist-1").expect("load");
        assert_eq!(loaded, run);
        assert_eq!(loaded.checkpoints[0].joules_uj, 100);

        assert_eq!(store.list(), vec!["persist-1"]);
        assert!(store.load("unknown").is_none());
    }

    #[test]
    fn run_round_trips_through_json() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(3)],
            nodes: vec![n(1, "A"), n(2, "B"), n(3, "C")],
            edges: vec![
                Edge {
                    from: NodeId(1),
                    to: NodeId(2),
                    label: Some("happy".into()),
                },
                edge(2, 3),
            ],
        };
        let mut run = Run::start(g, "json-1").unwrap();
        run.record(ck(1, 10, Provenance::HwShunt)).unwrap();
        run.record(ck(2, 20, Provenance::ModelBased)).unwrap();
        let bytes = serde_json::to_vec(&run).unwrap();
        let back: Run = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, run);
    }

    #[test]
    fn empty_label_does_not_serialize() {
        let e = Edge {
            from: NodeId(1),
            to: NodeId(2),
            label: None,
        };
        let j = serde_json::to_string(&e).unwrap();
        assert!(!j.contains("\"label\""), "got: {j}");
    }

    #[test]
    fn node_id_renders_as_n_prefix() {
        assert_eq!(NodeId(7).to_string(), "n7");
    }
}
