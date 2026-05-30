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
// Journal — append-only event log with blake3 hash chain
// ─────────────────────────────────────────────────────────────────────

/// One event in a [`Journal`]. The variant set is exhaustive and
/// versioned by the `jc_journal` schema; new variants are additive
/// and must serialise to a JSON discriminant that older readers
/// can ignore safely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunEvent {
    /// Run started against a graph with the given content address.
    /// Always the first entry in a journal.
    Started { graph_hash: String },
    /// Node was recorded as complete. Joule numbers and provenance
    /// are part of the event hash, so tampering with energy data
    /// after the fact breaks the chain.
    NodeRecorded {
        node_id: NodeId,
        joules_uj: u64,
        energy_provenance: Provenance,
        output_hash: String,
        recorded_at: u64,
    },
    /// An explicit energy observation that does NOT complete a node.
    /// Useful when a per-token attributor, an external meter, or a
    /// retry attempt produces an energy reading the runtime should
    /// account.
    EnergyObserved {
        node_id: NodeId,
        observed_uj: u64,
        provenance: Provenance,
    },
    /// The run failed.
    Failed {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<NodeId>,
        reason: String,
    },
    /// The run completed — every terminal has a checkpoint.
    Completed,
}

/// One journal entry — `(seq, event, prev_hash, this_hash)`.
///
/// `this_hash = blake3(canonical_bytes(seq, prev_hash, event_json))`.
/// Append-only ordering is guaranteed by sequence numbers, integrity
/// by the chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Position in the journal, starting at 0.
    pub seq: u64,
    /// The event itself.
    pub event: RunEvent,
    /// Blake3 hash of the prior entry. All zeros for the first
    /// entry.
    pub prev_hash: [u8; 32],
    /// Blake3 hash of this entry — over `seq || prev_hash ||
    /// canonical_json(event)`.
    pub this_hash: [u8; 32],
}

/// Append-only event log with hash chaining. The Run's source of
/// truth for replay; the joule numbers in events form a tamper-
/// evident energy ledger.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Journal {
    /// Entries in insertion order. Sequence numbers are dense
    /// (`0..entries.len()`).
    pub entries: Vec<JournalEntry>,
}

/// Errors when verifying or replaying a journal.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplayError {
    /// The graph failed structural validation.
    #[error("graph validation: {0:?}")]
    InvalidGraph(Vec<GraphError>),
    /// The events list was empty or missed the Started entry.
    #[error("journal must start with RunEvent::Started")]
    MissingStartEvent,
    /// The Started event's graph_hash didn't match the supplied
    /// graph's content address — replay against the wrong graph.
    #[error("graph hash mismatch: expected {expected}, journal said {found}")]
    GraphHashMismatch { expected: String, found: String },
    /// The hash chain is broken at this entry.
    #[error("hash chain broken at seq {0}")]
    BrokenChain(u64),
    /// Sequence numbers are not dense starting from 0.
    #[error("non-dense sequence at index {0}: got seq {1}")]
    NonDenseSeq(usize, u64),
}

impl Journal {
    /// New empty journal — same as [`Default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event, computing `prev_hash` and `this_hash` from
    /// the current tail. Returns the appended entry's sequence
    /// number.
    pub fn append(&mut self, event: RunEvent) -> u64 {
        let seq = self.entries.len() as u64;
        let prev_hash = self
            .entries
            .last()
            .map(|e| e.this_hash)
            .unwrap_or([0u8; 32]);
        let this_hash = Self::hash_entry(seq, &prev_hash, &event);
        self.entries.push(JournalEntry {
            seq,
            event,
            prev_hash,
            this_hash,
        });
        seq
    }

    /// Borrow the entry list (read-only — append is the only mutation).
    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }

    /// Last entry's `this_hash`, or all-zeros if the journal is empty.
    pub fn last_hash(&self) -> [u8; 32] {
        self.entries
            .last()
            .map(|e| e.this_hash)
            .unwrap_or([0u8; 32])
    }

    /// Sum of all `joules_uj` across `NodeRecorded` + `EnergyObserved`
    /// events. The journal's energy account, independent of
    /// `Run::checkpoints` — useful when verifying a serialised run
    /// hasn't been tampered with by re-checking joules against the
    /// journal.
    pub fn total_joules_uj(&self) -> u64 {
        let mut total = 0u64;
        for e in &self.entries {
            match &e.event {
                RunEvent::NodeRecorded { joules_uj, .. } => {
                    total = total.saturating_add(*joules_uj)
                }
                RunEvent::EnergyObserved { observed_uj, .. } => {
                    total = total.saturating_add(*observed_uj)
                }
                _ => {}
            }
        }
        total
    }

    /// Walk the journal and verify every entry's hash matches the
    /// recomputed value over `(seq, prev_hash, event)` AND sequence
    /// numbers are dense starting at 0.
    pub fn verify_chain(&self) -> Result<(), ReplayError> {
        Self::verify_chain_entries(&self.entries)
    }

    fn verify_chain_entries(entries: &[JournalEntry]) -> Result<(), ReplayError> {
        let mut prev_hash = [0u8; 32];
        for (i, e) in entries.iter().enumerate() {
            if e.seq != i as u64 {
                return Err(ReplayError::NonDenseSeq(i, e.seq));
            }
            if e.prev_hash != prev_hash {
                return Err(ReplayError::BrokenChain(e.seq));
            }
            let expected = Self::hash_entry(e.seq, &e.prev_hash, &e.event);
            if e.this_hash != expected {
                return Err(ReplayError::BrokenChain(e.seq));
            }
            prev_hash = e.this_hash;
        }
        Ok(())
    }

    fn hash_entry(seq: u64, prev_hash: &[u8; 32], event: &RunEvent) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"jc_journal/v1");
        hasher.update(&seq.to_be_bytes());
        hasher.update(prev_hash);
        // Canonical JSON via serde_json::to_vec on the tagged enum.
        // `RunEvent` uses `#[serde(tag = "kind", rename_all = ...)]`
        // so the output is deterministic across runs.
        let bytes = serde_json::to_vec(event).expect("RunEvent always serialises");
        hasher.update(&bytes);
        *hasher.finalize().as_bytes()
    }
}

impl RunGraph {
    /// Content address of the graph — blake3 over the canonical JSON
    /// of `(entry, terminals, nodes, edges)`. Used as the
    /// `graph_hash` in [`RunEvent::Started`] so the journal binds to
    /// the graph it was produced against.
    pub fn content_address(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("RunGraph always serialises");
        let mut h = blake3::Hasher::new();
        h.update(b"jc_graph/v1");
        h.update(&bytes);
        h.finalize().to_hex().to_string()
    }
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
///
/// The `journal` field is an append-only event log with blake3 hash
/// chaining. Every state-changing call ([`Run::record`] / [`Run::fail`])
/// appends a [`JournalEntry`]. The journal is what makes the run
/// **replayable across process restarts** — Temporal/Restate/Inngest
/// converged on this shape, and the same primitive serves
/// energy-orthogonal trust: the hash chain binds the joule numbers
/// into a tamper-evident witness.
///
/// Backward compat: `journal` carries `#[serde(default)]` so Runs
/// persisted before this field existed deserialise unchanged with
/// an empty journal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub run_id: String,
    pub graph: RunGraph,
    pub checkpoints: Vec<Checkpoint>,
    pub status: RunStatus,
    /// Append-only event log. Each entry is hash-chained to the
    /// prior. See [`Journal`] and [`RunEvent`].
    #[serde(default)]
    pub journal: Journal,
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
    ///
    /// Appends a [`RunEvent::Started`] journal entry seeding the hash
    /// chain with the graph's content-address.
    pub fn start(graph: RunGraph, run_id: impl Into<String>) -> Result<Self, RunError> {
        graph.validate().map_err(RunError::InvalidGraph)?;
        let graph_hash = graph.content_address();
        let mut run = Self {
            run_id: run_id.into(),
            graph,
            checkpoints: Vec::new(),
            status: RunStatus::InProgress,
            journal: Journal::default(),
        };
        run.journal.append(RunEvent::Started { graph_hash });
        Ok(run)
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
    /// terminals are done. Appends a [`RunEvent::NodeRecorded`] to
    /// the journal — the joule numbers and provenance are part of
    /// the hash chain, so tampering with energy data after the fact
    /// breaks the chain.
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
        let event = RunEvent::NodeRecorded {
            node_id: checkpoint.node_id,
            joules_uj: checkpoint.joules_uj,
            energy_provenance: checkpoint.energy_provenance,
            output_hash: checkpoint.output_hash.clone(),
            recorded_at: checkpoint.recorded_at,
        };
        self.checkpoints.push(checkpoint);
        self.journal.append(event);
        // Run completes when every terminal has a checkpoint.
        let done = self.completed_nodes();
        if self.graph.terminals.iter().all(|t| done.contains(t)) {
            self.status = RunStatus::Completed;
            self.journal.append(RunEvent::Completed);
        }
        Ok(())
    }

    /// Mark the run as failed (consumer decides when; e.g. a verifier
    /// rejected a critical node's output). Appends a
    /// [`RunEvent::Failed`] to the journal.
    pub fn fail(&mut self) {
        self.fail_with(None, String::new());
    }

    /// Like [`Run::fail`] but records a node id + reason in the
    /// journal entry. Useful for "node X's gate denied" style
    /// terminations.
    pub fn fail_with(&mut self, node_id: Option<NodeId>, reason: impl Into<String>) {
        self.status = RunStatus::Failed;
        self.journal.append(RunEvent::Failed {
            node_id,
            reason: reason.into(),
        });
    }

    /// Record an explicit energy observation that is NOT tied to
    /// completing a node. Useful when an external meter reports
    /// joule spend mid-node, or when an attribution algorithm
    /// emits a per-token observation without changing the run's
    /// node status.
    pub fn observe_energy(
        &mut self,
        node_id: NodeId,
        observed_uj: u64,
        provenance: Provenance,
    ) {
        self.journal.append(RunEvent::EnergyObserved {
            node_id,
            observed_uj,
            provenance,
        });
    }

    /// Replay a journal into a fresh Run. Reconstructs `checkpoints`
    /// and `status` from the events; the returned `Run`'s `journal`
    /// is identical to the input. Side effects of the original run
    /// (tool calls, model invocations) are NOT re-executed; their
    /// outputs/receipts were captured in the receipted events at
    /// original run time.
    pub fn replay_from(
        graph: RunGraph,
        events: &[JournalEntry],
        run_id: impl Into<String>,
    ) -> Result<Self, ReplayError> {
        graph.validate().map_err(ReplayError::InvalidGraph)?;
        Journal::verify_chain_entries(events)?;
        // Identify Started.graph_hash matches.
        let expected_graph_hash = graph.content_address();
        if let Some(first) = events.first() {
            if let RunEvent::Started { graph_hash } = &first.event {
                if graph_hash != &expected_graph_hash {
                    return Err(ReplayError::GraphHashMismatch {
                        expected: expected_graph_hash,
                        found: graph_hash.clone(),
                    });
                }
            } else {
                return Err(ReplayError::MissingStartEvent);
            }
        } else {
            return Err(ReplayError::MissingStartEvent);
        }
        let mut run = Self {
            run_id: run_id.into(),
            graph,
            checkpoints: Vec::new(),
            status: RunStatus::InProgress,
            journal: Journal {
                entries: events.to_vec(),
            },
        };
        for entry in events {
            match &entry.event {
                RunEvent::Started { .. } => {}
                RunEvent::NodeRecorded {
                    node_id,
                    joules_uj,
                    energy_provenance,
                    output_hash,
                    recorded_at,
                } => {
                    run.checkpoints.push(Checkpoint {
                        node_id: *node_id,
                        joules_uj: *joules_uj,
                        energy_provenance: *energy_provenance,
                        output_hash: output_hash.clone(),
                        recorded_at: *recorded_at,
                    });
                }
                RunEvent::Completed => {
                    run.status = RunStatus::Completed;
                }
                RunEvent::Failed { .. } => {
                    run.status = RunStatus::Failed;
                }
                RunEvent::EnergyObserved { .. } => {}
            }
        }
        Ok(run)
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

/// File-backed [`RunStore`] — one JSON file per run under a base
/// directory. The reason for picking file-per-run over a single
/// log/db: a crashed `save` only corrupts one run's file; resuming
/// the rest of the population is unaffected.
///
/// Atomicity: each save writes to `{run_id}.json.tmp`, fsyncs the
/// file, then renames into place — the standard "rename-is-atomic-
/// on-POSIX" pattern. A crash mid-save leaves the previous
/// `{run_id}.json` intact and a stale `.tmp` to clean up on next
/// boot.
///
/// File name policy: `run_id` is used verbatim as the basename. The
/// caller is responsible for choosing a filesystem-safe `run_id`
/// (the store rejects `/`, `\`, and `..` to keep persistence
/// in-directory). UUIDs / monotonic-id schemes satisfy this.
#[derive(Debug, Clone)]
pub struct FileRunStore {
    dir: std::path::PathBuf,
}

impl FileRunStore {
    /// Open (or create) a file-backed store under `dir`. The directory
    /// is created if it does not exist; existing files are left as-is.
    pub fn open(
        dir: impl Into<std::path::PathBuf>,
    ) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Where the store reads and writes.
    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }

    fn is_safe_run_id(run_id: &str) -> bool {
        !run_id.is_empty()
            && !run_id.contains('/')
            && !run_id.contains('\\')
            && !run_id.contains("..")
            && !run_id.starts_with('.')
    }

    fn run_path(&self, run_id: &str) -> std::path::PathBuf {
        self.dir.join(format!("{run_id}.json"))
    }
}

impl RunStore for FileRunStore {
    /// Save a run. Silently no-ops on unsafe `run_id` (the in-memory
    /// store would accept it; the file store would otherwise escape
    /// the directory). Returns nothing — the trait is a fire-and-
    /// forget contract; on-disk errors are recoverable on next save
    /// because the previous file is untouched until rename.
    fn save(&mut self, run: &Run) {
        if !Self::is_safe_run_id(&run.run_id) {
            return;
        }
        let Ok(bytes) = serde_json::to_vec(run) else {
            return;
        };
        let final_path = self.run_path(&run.run_id);
        let tmp_path = final_path.with_extension("json.tmp");
        // Best-effort atomic write: write tmp, rename over final.
        // If write fails halfway, the previous {run_id}.json is intact.
        if std::fs::write(&tmp_path, &bytes).is_err() {
            return;
        }
        let _ = std::fs::rename(&tmp_path, &final_path);
    }

    fn load(&self, run_id: &str) -> Option<Run> {
        if !Self::is_safe_run_id(run_id) {
            return None;
        }
        let bytes = std::fs::read(self.run_path(run_id)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    fn list(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut ids = Vec::new();
        for e in entries.flatten() {
            let name = e.file_name();
            let s = name.to_string_lossy();
            if let Some(stripped) = s.strip_suffix(".json") {
                ids.push(stripped.to_string());
            }
        }
        ids.sort();
        ids
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

    // ─── FileRunStore ───────────────────────────────────────────────

    fn tmpdir(label: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let p = std::env::temp_dir().join(format!("jouleclaw-graph-test-{label}-{pid}-{n}"));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn linear_run(id: &str) -> Run {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, id).unwrap();
        run.record(ck(1, 100, Provenance::HwShunt)).unwrap();
        run
    }

    #[test]
    fn file_run_store_saves_and_loads() {
        let dir = tmpdir("save-load");
        let mut store = FileRunStore::open(&dir).unwrap();
        let run = linear_run("r-1");
        store.save(&run);
        assert!(dir.join("r-1.json").exists());
        let loaded = store.load("r-1").expect("load");
        assert_eq!(loaded, run);
    }

    #[test]
    fn file_run_store_lists_run_ids_sorted() {
        let dir = tmpdir("list");
        let mut store = FileRunStore::open(&dir).unwrap();
        store.save(&linear_run("b-2"));
        store.save(&linear_run("a-1"));
        store.save(&linear_run("c-3"));
        assert_eq!(store.list(), vec!["a-1", "b-2", "c-3"]);
    }

    #[test]
    fn file_run_store_missing_id_returns_none() {
        let dir = tmpdir("missing");
        let store = FileRunStore::open(&dir).unwrap();
        assert!(store.load("nope").is_none());
        assert!(store.list().is_empty());
    }

    #[test]
    fn file_run_store_rejects_unsafe_run_id() {
        let dir = tmpdir("unsafe");
        let mut store = FileRunStore::open(&dir).unwrap();
        let mut run = linear_run("ok");
        run.run_id = "../escape".into();
        store.save(&run);
        assert!(!dir.join("..escape.json").exists());
        assert!(store.load("../escape").is_none());

        run.run_id = "with/slash".into();
        store.save(&run);
        assert!(store.list().is_empty());
    }

    #[test]
    fn file_run_store_overwrites_previous_save() {
        let dir = tmpdir("overwrite");
        let mut store = FileRunStore::open(&dir).unwrap();
        let mut run = linear_run("r-1");
        store.save(&run);
        run.record(ck(2, 250, Provenance::ModelBased)).unwrap();
        store.save(&run);
        let loaded = store.load("r-1").unwrap();
        assert_eq!(loaded.checkpoints.len(), 2);
        assert!(loaded.is_done());
    }

    #[test]
    fn file_run_store_round_trips_through_disk_in_a_fresh_handle() {
        let dir = tmpdir("rehandle");
        let mut store = FileRunStore::open(&dir).unwrap();
        store.save(&linear_run("durable-1"));
        drop(store);
        let store2 = FileRunStore::open(&dir).unwrap();
        let loaded = store2.load("durable-1").unwrap();
        assert_eq!(loaded.run_id, "durable-1");
        assert_eq!(loaded.checkpoints.len(), 1);
    }

    // ─── Journal + replay ────────────────────────────────────────────

    #[test]
    fn journal_seeds_with_started_event_carrying_graph_hash() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let expected_hash = g.content_address();
        let run = Run::start(g, "j-1").unwrap();
        assert_eq!(run.journal.entries().len(), 1);
        match &run.journal.entries()[0].event {
            RunEvent::Started { graph_hash } => assert_eq!(graph_hash, &expected_hash),
            other => panic!("expected Started, got {other:?}"),
        }
        // Hash chain seeded.
        assert_eq!(run.journal.entries()[0].prev_hash, [0u8; 32]);
        assert_ne!(run.journal.entries()[0].this_hash, [0u8; 32]);
    }

    #[test]
    fn record_appends_node_recorded_event_carrying_joules() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-2").unwrap();
        run.record(ck(1, 100, Provenance::HwShunt)).unwrap();
        assert_eq!(run.journal.entries().len(), 2);
        match &run.journal.entries()[1].event {
            RunEvent::NodeRecorded {
                node_id,
                joules_uj,
                energy_provenance,
                ..
            } => {
                assert_eq!(*node_id, NodeId(1));
                assert_eq!(*joules_uj, 100);
                assert_eq!(*energy_provenance, Provenance::HwShunt);
            }
            other => panic!("expected NodeRecorded, got {other:?}"),
        }
    }

    #[test]
    fn run_completes_emits_completed_event() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-3").unwrap();
        run.record(ck(1, 10, Provenance::HwShunt)).unwrap();
        run.record(ck(2, 20, Provenance::HwShunt)).unwrap();
        let last = run.journal.entries().last().unwrap();
        assert!(matches!(last.event, RunEvent::Completed));
    }

    #[test]
    fn fail_with_records_node_and_reason() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-4").unwrap();
        run.fail_with(Some(NodeId(1)), "gate denied");
        let last = run.journal.entries().last().unwrap();
        match &last.event {
            RunEvent::Failed { node_id, reason } => {
                assert_eq!(*node_id, Some(NodeId(1)));
                assert_eq!(reason, "gate denied");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn observe_energy_appends_energy_event() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-5").unwrap();
        run.observe_energy(NodeId(1), 7777, Provenance::HwShunt);
        let last = run.journal.entries().last().unwrap();
        match &last.event {
            RunEvent::EnergyObserved {
                node_id,
                observed_uj,
                provenance,
            } => {
                assert_eq!(*node_id, NodeId(1));
                assert_eq!(*observed_uj, 7777);
                assert_eq!(*provenance, Provenance::HwShunt);
            }
            other => panic!("expected EnergyObserved, got {other:?}"),
        }
    }

    #[test]
    fn journal_total_joules_sums_recorded_and_observed() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-6").unwrap();
        run.record(ck(1, 100, Provenance::HwShunt)).unwrap();
        run.observe_energy(NodeId(1), 50, Provenance::HwShunt);
        // Started + Completed events carry no joules; only NodeRecorded
        // + EnergyObserved.
        assert_eq!(run.journal.total_joules_uj(), 150);
    }

    #[test]
    fn verify_chain_passes_on_unmodified_journal() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-7").unwrap();
        run.record(ck(1, 10, Provenance::HwShunt)).unwrap();
        run.record(ck(2, 20, Provenance::HwShunt)).unwrap();
        run.journal.verify_chain().unwrap();
    }

    #[test]
    fn verify_chain_detects_tampered_joules() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-8").unwrap();
        run.record(ck(1, 10, Provenance::HwShunt)).unwrap();
        // Tamper: rewrite the joule number in the journal AFTER hashing.
        if let RunEvent::NodeRecorded { joules_uj, .. } =
            &mut run.journal.entries[1].event
        {
            *joules_uj = 999_999;
        }
        let err = run.journal.verify_chain().unwrap_err();
        assert!(matches!(err, ReplayError::BrokenChain(_)));
    }

    #[test]
    fn replay_reconstructs_run_state_from_journal() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g.clone(), "j-9").unwrap();
        run.record(ck(1, 100, Provenance::HwShunt)).unwrap();
        run.record(ck(2, 200, Provenance::ModelBased)).unwrap();
        let replayed = Run::replay_from(g, run.journal.entries(), "j-9-replay").unwrap();
        assert_eq!(replayed.checkpoints.len(), 2);
        assert_eq!(replayed.checkpoints[0].joules_uj, 100);
        assert_eq!(replayed.checkpoints[1].joules_uj, 200);
        assert!(replayed.is_done());
        assert_eq!(replayed.total_joules_uj(), 300);
    }

    #[test]
    fn replay_rejects_mismatched_graph_hash() {
        let g1 = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let g2 = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(3)],
            nodes: vec![n(1, "A"), n(3, "C")],
            edges: vec![edge(1, 3)],
        };
        let run = Run::start(g1, "j-10").unwrap();
        let err = Run::replay_from(g2, run.journal.entries(), "j-10-r").unwrap_err();
        assert!(matches!(err, ReplayError::GraphHashMismatch { .. }));
    }

    #[test]
    fn replay_rejects_missing_start_event() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let err = Run::replay_from(g, &[], "j-11").unwrap_err();
        assert!(matches!(err, ReplayError::MissingStartEvent));
    }

    #[test]
    fn run_round_trips_through_json_with_journal() {
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let mut run = Run::start(g, "j-12").unwrap();
        run.record(ck(1, 10, Provenance::HwShunt)).unwrap();
        let bytes = serde_json::to_vec(&run).unwrap();
        let back: Run = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.journal.entries().len(), 2); // Started + NodeRecorded
        back.journal.verify_chain().unwrap();
    }

    #[test]
    fn old_json_without_journal_deserializes_with_empty_journal() {
        // Synthesize the pre-extension JSON shape (no `journal` field).
        let g = RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![n(1, "A"), n(2, "B")],
            edges: vec![edge(1, 2)],
        };
        let json = serde_json::json!({
            "run_id": "legacy",
            "graph": g,
            "checkpoints": [],
            "status": "in_progress"
        });
        let run: Run = serde_json::from_value(json).unwrap();
        assert_eq!(run.run_id, "legacy");
        assert_eq!(run.journal.entries().len(), 0);
    }
}
