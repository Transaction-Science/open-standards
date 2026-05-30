//! L7 — reflection (meta-cognitive control plane).
//!
//! Where L5 routing learns *online* (every accepted answer updates the
//! episode memory immediately), L7 learns *offline*: it accumulates
//! observations of how each tier performed and, on a deliberate
//! [`ReflectionEngine::reflect`] pass, distils them into [`Lesson`]s —
//! "promote this tier, it resolves cheaply and reliably" or "demote
//! that one, it burns joules and still gets refused."
//!
//! The "async background" framing from the donor is intentionally *not*
//! baked in here. L7 exposes a synchronous `reflect()`; the consumer
//! calls it from whatever scheduler they like (a timer, a low-priority
//! thread, end-of-shift batch). Keeping it synchronous means the crate
//! has no runtime dependency and the lessons are reproducible from a
//! fixed observation set.

#![forbid(unsafe_code)]

use jouleclaw_cascade::types::TierId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One recorded dispatch outcome.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Stable hash of the query (caller-supplied; L7 does not parse text).
    pub query_fingerprint: u64,
    /// The tier that handled this dispatch.
    pub tier_used: TierId,
    /// Joules that tier spent.
    pub joules_spent: f64,
    /// Confidence of the produced answer in `[0, 1]`.
    pub confidence: f32,
    /// Whether the answer was accepted (vs refused / later contradicted).
    pub success: bool,
    /// Unix-seconds when this happened (for recency, optional use).
    pub timestamp_secs: u64,
}

/// What L7 thinks should happen to a tier's standing in the cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Move this tier earlier — it resolves cheaply and reliably.
    Promote,
    /// Move this tier later or gate it — poor success / high cost.
    Demote,
    /// No change warranted.
    KeepAsIs,
}

/// A distilled recommendation about one tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    /// Human-readable summary (for traces / dashboards).
    pub pattern: String,
    /// The tier this lesson concerns. Serialized via its wire tag.
    #[serde(with = "tier_wire")]
    pub tier: TierId,
    /// What to do.
    pub recommended_action: Action,
    /// How many observations back this lesson.
    pub support_count: u32,
    /// Mean joules across those observations.
    pub mean_joules: f64,
    /// Success rate in `[0, 1]` across those observations.
    pub success_rate: f32,
}

/// Tunables for the reflection pass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReflectionConfig {
    /// Ring-buffer capacity for observations.
    pub capacity: usize,
    /// Minimum observations of a tier before any lesson is emitted.
    pub min_support: u32,
    /// Success rate at or above this → Promote (if also cheap-ish).
    pub promote_success: f32,
    /// Success rate at or below this → Demote.
    pub demote_success: f32,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            capacity: 4096,
            min_support: 5,
            promote_success: 0.8,
            demote_success: 0.4,
        }
    }
}

/// The offline learner.
pub struct ReflectionEngine {
    observations: Vec<Observation>,
    lessons: Vec<Lesson>,
    cfg: ReflectionConfig,
}

impl Default for ReflectionEngine {
    fn default() -> Self {
        Self::new(ReflectionConfig::default())
    }
}

impl ReflectionEngine {
    pub fn new(cfg: ReflectionConfig) -> Self {
        Self {
            observations: Vec::new(),
            lessons: Vec::new(),
            cfg,
        }
    }

    /// Number of observations currently retained.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// The lessons from the most recent [`reflect`](Self::reflect) call.
    pub fn lessons(&self) -> &[Lesson] {
        &self.lessons
    }

    /// Append an observation, evicting the oldest if at capacity.
    pub fn record(&mut self, obs: Observation) {
        if self.observations.len() >= self.cfg.capacity {
            self.observations.remove(0);
        }
        self.observations.push(obs);
    }

    /// Run the offline pass: group observations by tier, compute mean
    /// joules + success rate, emit one [`Lesson`] per tier with enough
    /// support. Deterministic given a fixed observation set (lessons
    /// are sorted by tier wire tag). Caches the result; see
    /// [`lessons`](Self::lessons).
    pub fn reflect(&mut self) -> &[Lesson] {
        // tier -> (count, sum_joules, success_count)
        let mut agg: HashMap<TierId, (u32, f64, u32)> = HashMap::new();
        for o in &self.observations {
            let e = agg.entry(o.tier_used).or_insert((0, 0.0, 0));
            e.0 += 1;
            e.1 += o.joules_spent;
            if o.success {
                e.2 += 1;
            }
        }

        let mut lessons: Vec<Lesson> = Vec::new();
        for (tier, (count, sum_joules, succ)) in agg {
            if count < self.cfg.min_support {
                continue;
            }
            let mean_joules = sum_joules / count as f64;
            let success_rate = succ as f32 / count as f32;
            let action = if success_rate >= self.cfg.promote_success {
                Action::Promote
            } else if success_rate <= self.cfg.demote_success {
                Action::Demote
            } else {
                Action::KeepAsIs
            };
            let pattern = format!(
                "tier {} — {} samples, {:.0}% success, mean {:.3e} J → {:?}",
                tier.wire_tag(),
                count,
                success_rate * 100.0,
                mean_joules,
                action
            );
            lessons.push(Lesson {
                pattern,
                tier,
                recommended_action: action,
                support_count: count,
                mean_joules,
                success_rate,
            });
        }
        lessons.sort_by(|a, b| a.tier.wire_tag().cmp(b.tier.wire_tag()));
        self.lessons = lessons;
        &self.lessons
    }
}

/// Serde helper: serialize `TierId` as its stable wire tag string.
mod tier_wire {
    use super::TierId;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &TierId, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(t.wire_tag())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TierId, D::Error> {
        // Lessons are advisory and round-tripped rarely; we map the wire
        // tag back to the coarse representative tier. Fractional tags map
        // to their nearest concrete variant.
        let s = String::deserialize(d)?;
        Ok(parse_wire_tag(&s))
    }

    fn parse_wire_tag(s: &str) -> TierId {
        use jouleclaw_cascade::types::{L1Primitive, L2ModelId, L3ModelId, L4ModelId};
        match s {
            "L0" => TierId::L0,
            "L0.1" => TierId::L0_1FactLut,
            "L0.25" => TierId::L0_25FormulaFirst,
            "L0.5" => TierId::L0_5ToolCompute,
            "L0.75" => TierId::L0_75SsmRouter,
            "L1" => TierId::L1(L1Primitive::Retrieve),
            "L1.25" => TierId::L1_25GraphRag,
            "L1.375" => TierId::L1_375StructContrast,
            "L1.5" => TierId::L1_5SsmReader,
            "L2" => TierId::L2(L2ModelId(0)),
            "L2.5" => TierId::L2_5NeuralRerank,
            "L3" => TierId::L3(L3ModelId(0)),
            "L4" => TierId::L4(L4ModelId(0)),
            "L4.5" => TierId::L4_5Proof,
            "L5" => TierId::L5Routing,
            "L6" => TierId::L6Agent,
            "L7" => TierId::L7Reflection,
            "L8" => TierId::L8Tuner,
            "L9" => TierId::L9Supervisor,
            "L10" => TierId::L10Governor,
            _ => TierId::L0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Sleep-time Reflector — consumes (journal + memory) → emits findings
// ─────────────────────────────────────────────────────────────────────
//
// The "doer vs reflector" split AutoDream / Letta sleep-time / agent-
// retro converged on: the Reflector runs OFFLINE between sessions,
// reads the run journal + receipts + memory, and emits structured
// findings (no LLM required for the heuristic reference) + prune /
// edit suggestions the consumer applies on its own schedule.
//
// Energy is the orthogonal trust anchor: doom loops and silent
// skips are most cleanly detected by *energy repeated for no new
// outcome*. Memory consolidation reasons about *energy paid per
// fact*. The Reflector therefore reads RunEvent::EnergyObserved /
// NodeRecorded directly.

/// Input to a [`Reflector`]: the journal + (optionally) a memory
/// window the reflector may consider for pruning suggestions.
#[derive(Debug, Clone)]
pub struct ReflectorInput<'a> {
    /// Run journal events the reflector inspects. Typically the
    /// full output of [`jouleclaw_graph::Journal::entries`].
    pub journal: &'a [jouleclaw_graph::JournalEntry],
    /// Optional opaque memory ids the reflector may consider for
    /// pruning. The reflector does NOT read fact content here — the
    /// consumer surfaces ids it tracks as "looked at this turn" /
    /// "loaded but unused" so the reflector can suggest prunes
    /// without re-loading.
    pub memory_ids: &'a [String],
}

/// One structured finding from a [`Reflector`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Finding {
    /// The same node was recorded multiple times in a window (only
    /// possible via the journal's `NodeRecorded` events; the live
    /// `Run::record` blocks duplicates, but a replayed-then-
    /// extended journal can show repetitions across replay
    /// boundaries). Doom-loop signal.
    DoomLoop {
        node_id: jouleclaw_graph::NodeId,
        repetitions: u64,
        wasted_joules_uj: u64,
    },
    /// Energy observed for a node exceeds its expected band by a
    /// large ratio. The cheapest physics-grounded anomaly signal —
    /// surface independent of the model's claims.
    EnergyAnomaly {
        node_id: jouleclaw_graph::NodeId,
        observed_uj: u64,
        expected_uj_upper: u64,
        ratio: f64,
    },
    /// A `Failed` event with no preceding `EnergyObserved` / no
    /// `NodeRecorded` since the run started — the run silently
    /// stalled rather than reporting progress. Surfaces "the agent
    /// said it failed without trying" cases.
    SilentSkip {
        last_seq: u64,
        reason: String,
    },
    /// Memory id list contains duplicates by string equality —
    /// pruning candidate. (The consumer can dedupe by
    /// content-address upstream; this is the trailing safety net.)
    DuplicatedMemory {
        ids: Vec<String>,
    },
}

/// Per-severity classification of a [`Finding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Informational; pruning suggested but not urgent.
    Info,
    /// Worth applying the suggested prune before the next session.
    Warn,
}

/// Output of a reflector pass.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReflectorOutput {
    /// Structured findings, in detection order.
    pub findings: Vec<Finding>,
    /// Memory ids the reflector suggests pruning. Consumer applies
    /// on its own schedule.
    pub proposed_memory_prunes: Vec<String>,
    /// Sum of joules the reflector classifies as wasted (e.g. doom-
    /// loop repeats, anomaly spend above expected band). The single
    /// number a budget owner cares about — energy-grounded waste.
    pub total_wasted_joules_uj: u64,
}

/// A reflector — typed contract for the sleep-time pass.
pub trait Reflector: Send + Sync {
    /// Run the reflection pass over the input. Pure; the consumer
    /// applies output suggestions.
    fn reflect(&self, input: &ReflectorInput<'_>) -> ReflectorOutput;
}

/// Reference heuristic reflector — pure pattern detection over the
/// journal + memory id list. No LLM, no model dependency. The
/// thresholds are tunable but the defaults are conservative.
#[derive(Debug, Clone)]
pub struct HeuristicReflector {
    /// Same-node `NodeRecorded` repeats ≥ this count are flagged as
    /// doom loops. Default 3.
    pub doom_loop_repetitions: u64,
    /// Per-node expected energy upper band, microjoules. Anomaly
    /// fires when observed > this × `anomaly_min_ratio`. Default
    /// 1_000_000 µJ = 1 J — well above L0/L1 work, well below L4
    /// wire calls.
    pub expected_uj_upper: u64,
    /// Anomaly ratio threshold. Default 2.0 (twice the expected
    /// upper band).
    pub anomaly_min_ratio: f64,
}

impl Default for HeuristicReflector {
    fn default() -> Self {
        Self {
            doom_loop_repetitions: 3,
            expected_uj_upper: 1_000_000,
            anomaly_min_ratio: 2.0,
        }
    }
}

impl HeuristicReflector {
    /// Builder: override the doom-loop minimum.
    pub fn with_doom_loop_repetitions(mut self, n: u64) -> Self {
        self.doom_loop_repetitions = n.max(2);
        self
    }
    /// Builder: override the expected per-node energy upper band.
    pub fn with_expected_uj_upper(mut self, uj: u64) -> Self {
        self.expected_uj_upper = uj.max(1);
        self
    }
    /// Builder: override the anomaly ratio.
    pub fn with_anomaly_min_ratio(mut self, r: f64) -> Self {
        self.anomaly_min_ratio = r.max(1.0);
        self
    }
}

impl Reflector for HeuristicReflector {
    fn reflect(&self, input: &ReflectorInput<'_>) -> ReflectorOutput {
        use jouleclaw_graph::RunEvent;
        use std::collections::BTreeMap;

        let mut findings = Vec::new();
        let mut total_wasted: u64 = 0;

        // Per-node recorded counts + total joules.
        let mut record_count: BTreeMap<jouleclaw_graph::NodeId, u64> = BTreeMap::new();
        let mut record_joules: BTreeMap<jouleclaw_graph::NodeId, u64> = BTreeMap::new();
        let mut last_progress_seq: Option<u64> = None;

        for entry in input.journal {
            match &entry.event {
                RunEvent::NodeRecorded {
                    node_id, joules_uj, ..
                } => {
                    *record_count.entry(*node_id).or_insert(0) += 1;
                    let j = record_joules.entry(*node_id).or_insert(0);
                    *j = j.saturating_add(*joules_uj);
                    last_progress_seq = Some(entry.seq);
                }
                RunEvent::EnergyObserved {
                    node_id,
                    observed_uj,
                    ..
                } => {
                    last_progress_seq = Some(entry.seq);
                    if *observed_uj
                        > (self.expected_uj_upper as f64 * self.anomaly_min_ratio) as u64
                    {
                        let ratio = *observed_uj as f64 / self.expected_uj_upper as f64;
                        let wasted =
                            observed_uj.saturating_sub(self.expected_uj_upper);
                        total_wasted = total_wasted.saturating_add(wasted);
                        findings.push(Finding::EnergyAnomaly {
                            node_id: *node_id,
                            observed_uj: *observed_uj,
                            expected_uj_upper: self.expected_uj_upper,
                            ratio,
                        });
                    }
                }
                RunEvent::Failed { reason, .. } => {
                    if last_progress_seq.is_none() {
                        findings.push(Finding::SilentSkip {
                            last_seq: entry.seq,
                            reason: reason.clone(),
                        });
                    }
                }
                _ => {}
            }
        }

        // Doom-loop pass over node counts.
        for (node_id, count) in record_count {
            if count >= self.doom_loop_repetitions {
                let joules = record_joules.get(&node_id).copied().unwrap_or(0);
                // Charge all-but-the-first run as wasted.
                let wasted = joules
                    .saturating_mul(count.saturating_sub(1))
                    .saturating_div(count);
                total_wasted = total_wasted.saturating_add(wasted);
                findings.push(Finding::DoomLoop {
                    node_id,
                    repetitions: count,
                    wasted_joules_uj: wasted,
                });
            }
        }

        // Duplicate memory ids → prune suggestions.
        let mut seen = std::collections::BTreeSet::new();
        let mut dup_ids = Vec::new();
        for id in input.memory_ids {
            if !seen.insert(id) {
                dup_ids.push(id.clone());
            }
        }
        if !dup_ids.is_empty() {
            findings.push(Finding::DuplicatedMemory {
                ids: dup_ids.clone(),
            });
        }

        ReflectorOutput {
            findings,
            proposed_memory_prunes: dup_ids,
            total_wasted_joules_uj: total_wasted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{L1Primitive, L3ModelId};

    fn obs(tier: TierId, joules: f64, success: bool) -> Observation {
        Observation {
            query_fingerprint: 1,
            tier_used: tier,
            joules_spent: joules,
            confidence: if success { 0.9 } else { 0.2 },
            success,
            timestamp_secs: 0,
        }
    }

    #[test]
    fn empty_yields_no_lessons() {
        let mut e = ReflectionEngine::default();
        assert!(e.reflect().is_empty());
    }

    #[test]
    fn promote_on_high_success() {
        let mut e = ReflectionEngine::default();
        for _ in 0..10 {
            e.record(obs(TierId::L0_1FactLut, 5e-6, true));
        }
        let lessons = e.reflect();
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].recommended_action, Action::Promote);
        assert_eq!(lessons[0].support_count, 10);
    }

    #[test]
    fn demote_on_low_success() {
        let mut e = ReflectionEngine::default();
        for _ in 0..10 {
            e.record(obs(TierId::L3(L3ModelId(0)), 2.0, false));
        }
        let lessons = e.reflect();
        assert_eq!(lessons[0].recommended_action, Action::Demote);
    }

    #[test]
    fn keep_as_is_in_middle_band() {
        let mut e = ReflectionEngine::default();
        for i in 0..10 {
            e.record(obs(TierId::L1(L1Primitive::Retrieve), 1e-3, i % 2 == 0));
        }
        let lessons = e.reflect();
        assert_eq!(lessons[0].recommended_action, Action::KeepAsIs);
        assert!((lessons[0].success_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn below_min_support_skipped() {
        let mut e = ReflectionEngine::default();
        for _ in 0..3 {
            e.record(obs(TierId::L0, 1e-6, true));
        }
        assert!(e.reflect().is_empty());
    }

    #[test]
    fn mean_joules_computed() {
        let mut e = ReflectionEngine::default();
        e.record(obs(TierId::L0, 2.0, true));
        e.record(obs(TierId::L0, 4.0, true));
        e.record(obs(TierId::L0, 6.0, true));
        e.record(obs(TierId::L0, 8.0, true));
        e.record(obs(TierId::L0, 10.0, true));
        let lessons = e.reflect();
        assert!((lessons[0].mean_joules - 6.0).abs() < 1e-9);
    }

    #[test]
    fn lessons_sorted_by_wire_tag() {
        let mut e = ReflectionEngine::default();
        for _ in 0..5 {
            e.record(obs(TierId::L3(L3ModelId(0)), 2.0, true));
            e.record(obs(TierId::L0, 1e-6, true));
        }
        let lessons = e.reflect();
        assert_eq!(lessons.len(), 2);
        assert_eq!(lessons[0].tier.wire_tag(), "L0");
        assert_eq!(lessons[1].tier.wire_tag(), "L3");
    }

    #[test]
    fn capacity_evicts_oldest() {
        let cfg = ReflectionConfig {
            capacity: 3,
            ..Default::default()
        };
        let mut e = ReflectionEngine::new(cfg);
        for _ in 0..5 {
            e.record(obs(TierId::L0, 1e-6, true));
        }
        assert_eq!(e.len(), 3);
    }

    #[test]
    fn lessons_accessor_matches_reflect() {
        let mut e = ReflectionEngine::default();
        for _ in 0..6 {
            e.record(obs(TierId::L0, 1e-6, true));
        }
        let n = e.reflect().len();
        assert_eq!(e.lessons().len(), n);
    }

    #[test]
    fn lesson_serializes_tier_as_wire_tag() {
        let mut e = ReflectionEngine::default();
        for _ in 0..6 {
            e.record(obs(TierId::L0_25FormulaFirst, 200e-6, true));
        }
        e.reflect();
        let json = serde_json::to_string(&e.lessons()[0]).unwrap();
        assert!(json.contains("\"L0.25\""), "json = {json}");
    }

    #[test]
    fn reflect_is_deterministic() {
        let mut e = ReflectionEngine::default();
        for _ in 0..7 {
            e.record(obs(TierId::L0, 1e-6, true));
        }
        let a: Vec<Action> = e.reflect().iter().map(|l| l.recommended_action).collect();
        let b: Vec<Action> = e.reflect().iter().map(|l| l.recommended_action).collect();
        assert_eq!(a, b);
    }

    // ─── HeuristicReflector ──────────────────────────────────────────

    use jouleclaw_graph::{Edge, MergeStrategy, Node, NodeId, NodeKind, RunGraph, Run};

    fn small_graph() -> RunGraph {
        RunGraph {
            entry: NodeId(1),
            terminals: vec![NodeId(2)],
            nodes: vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Stage("A".into()),
                    label: "A".into(),
                    merge: MergeStrategy::AllRequired,
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::Stage("B".into()),
                    label: "B".into(),
                    merge: MergeStrategy::AllRequired,
                },
            ],
            edges: vec![Edge {
                from: NodeId(1),
                to: NodeId(2),
                label: None,
            }],
        }
    }

    #[test]
    fn reflector_empty_journal_emits_nothing() {
        let r = HeuristicReflector::default();
        let out = r.reflect(&ReflectorInput {
            journal: &[],
            memory_ids: &[],
        });
        assert!(out.findings.is_empty());
        assert_eq!(out.total_wasted_joules_uj, 0);
    }

    #[test]
    fn reflector_flags_energy_anomaly_against_band() {
        let g = small_graph();
        let mut run = Run::start(g, "r-1").unwrap();
        // Default band 1_000_000 µJ × 2.0 ratio = anomaly above 2 J.
        run.observe_energy(NodeId(1), 5_000_000, jouleclaw_energy::Provenance::HwShunt);
        let r = HeuristicReflector::default();
        let out = r.reflect(&ReflectorInput {
            journal: run.journal.entries(),
            memory_ids: &[],
        });
        assert!(
            matches!(&out.findings[..], [Finding::EnergyAnomaly { .. }]),
            "got {:?}",
            out.findings
        );
        // Wasted = observed - upper = 5_000_000 - 1_000_000 = 4_000_000.
        assert_eq!(out.total_wasted_joules_uj, 4_000_000);
    }

    #[test]
    fn reflector_no_anomaly_within_band() {
        let g = small_graph();
        let mut run = Run::start(g, "r-2").unwrap();
        run.observe_energy(NodeId(1), 1_500_000, jouleclaw_energy::Provenance::HwShunt);
        // 1.5× the band — under the 2.0 ratio threshold.
        let r = HeuristicReflector::default();
        let out = r.reflect(&ReflectorInput {
            journal: run.journal.entries(),
            memory_ids: &[],
        });
        assert!(out.findings.is_empty());
    }

    #[test]
    fn reflector_flags_doom_loop_from_replay_extended_journal() {
        // The live Run rejects duplicate NodeRecorded, but a journal
        // assembled across replay boundaries can show repetitions.
        // We construct the JournalEntry list directly to simulate.
        use jouleclaw_graph::{Journal, RunEvent};
        let g = small_graph();
        let mut journal = Journal::default();
        journal.append(RunEvent::Started {
            graph_hash: g.content_address(),
        });
        for _ in 0..3 {
            journal.append(RunEvent::NodeRecorded {
                node_id: NodeId(1),
                joules_uj: 50,
                energy_provenance: jouleclaw_energy::Provenance::HwShunt,
                output_hash: "h".into(),
                recorded_at: 0,
            });
        }
        let r = HeuristicReflector::default();
        let out = r.reflect(&ReflectorInput {
            journal: journal.entries(),
            memory_ids: &[],
        });
        let dl_count = out
            .findings
            .iter()
            .filter(|f| matches!(f, Finding::DoomLoop { .. }))
            .count();
        assert_eq!(dl_count, 1);
        // 2-of-3 runs wasted = (150 * 2)/3 = 100.
        assert_eq!(out.total_wasted_joules_uj, 100);
    }

    #[test]
    fn reflector_flags_silent_skip_when_failed_without_progress() {
        let g = small_graph();
        let mut run = Run::start(g, "r-4").unwrap();
        run.fail_with(None, "model refused");
        let r = HeuristicReflector::default();
        let out = r.reflect(&ReflectorInput {
            journal: run.journal.entries(),
            memory_ids: &[],
        });
        assert!(out.findings.iter().any(|f| matches!(f, Finding::SilentSkip { .. })));
    }

    #[test]
    fn reflector_no_silent_skip_when_failure_follows_progress() {
        let g = small_graph();
        let mut run = Run::start(g, "r-5").unwrap();
        run.record(jouleclaw_graph::Checkpoint::new(
            NodeId(1),
            10,
            jouleclaw_energy::Provenance::HwShunt,
            "h",
            0,
        ))
        .unwrap();
        run.fail_with(Some(NodeId(2)), "node 2 verifier failed");
        let r = HeuristicReflector::default();
        let out = r.reflect(&ReflectorInput {
            journal: run.journal.entries(),
            memory_ids: &[],
        });
        assert!(out.findings.iter().all(|f| !matches!(f, Finding::SilentSkip { .. })));
    }

    #[test]
    fn reflector_suggests_pruning_duplicate_memory_ids() {
        let r = HeuristicReflector::default();
        let memory_ids = vec!["a".into(), "b".into(), "a".into(), "c".into(), "b".into()];
        let out = r.reflect(&ReflectorInput {
            journal: &[],
            memory_ids: &memory_ids,
        });
        assert!(out.findings.iter().any(|f| matches!(f, Finding::DuplicatedMemory { .. })));
        // Dup ids include "a" and "b" (each duplicated once).
        assert_eq!(out.proposed_memory_prunes.len(), 2);
    }

    #[test]
    fn reflector_output_round_trips_through_json() {
        let out = ReflectorOutput {
            findings: vec![Finding::DoomLoop {
                node_id: NodeId(7),
                repetitions: 3,
                wasted_joules_uj: 100,
            }],
            proposed_memory_prunes: vec!["m1".into()],
            total_wasted_joules_uj: 100,
        };
        let bytes = serde_json::to_vec(&out).unwrap();
        let back: ReflectorOutput = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, out);
    }
}
