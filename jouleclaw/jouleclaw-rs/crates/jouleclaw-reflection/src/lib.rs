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
}
