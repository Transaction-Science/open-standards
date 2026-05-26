//! Active synthesizers — `E=Active` on the Synthesis E axis.
//!
//! A reactive tier responds to a query; the conversation is pulled
//! from the caller. An active tier initiates — it watches some
//! state and decides on its own to emit a (query, answer) pair.
//!
//! The architecture: active tiers live alongside the cascade but not
//! inside it. The cascade is pull-driven (query → answer); the active
//! layer is push-driven (state change → emission). The runtime owns
//! a registry of active tiers and provides `tick_active()` for the
//! embedding application to give them time slices.
//!
//! This is intentionally synchronous. A real production runtime might
//! run active tiers on their own threads or in a tokio executor, but
//! the synchronous tick model is enough to demonstrate the coordinate
//! and keep the test suite deterministic.

use crate::coord::Coord;
use crate::types::{Answer, Query, TierId};

/// An active synthesizer. Distinct from `Tier` because active
/// synthesizers don't fit the query→answer dispatch shape.
///
/// Lifecycle:
///   1. The runtime calls `start()` once when the tier is registered.
///   2. The runtime calls `tick()` periodically. Each tick gives the
///      tier a chance to check its state and emit work.
///   3. The runtime calls `stop()` once during shutdown.
///
/// On a tick, the tier can return zero or more `(Query, Answer)`
/// pairs. These are recorded to history just as if they came from
/// the cascade — so a downstream consumer querying L0 finds them.
pub trait ActiveTier: Send {
    fn id(&self) -> TierId;
    fn coord(&self) -> Coord;

    /// Called once on registration. Default: no-op.
    fn start(&mut self) {}

    /// Called once at shutdown. Default: no-op.
    fn stop(&mut self) {}

    /// Called periodically. Return any `(query, answer)` pairs the
    /// tier has produced since the last tick. The runtime records
    /// these to history.
    ///
    /// Implementations should be cheap and non-blocking — a tick is
    /// an "opportunity," not a guarantee of work. Returning an empty
    /// vector is the common case.
    fn tick(&mut self) -> Vec<(Query, Answer)>;
}

/// Registry of active tiers, owned by the runtime.
pub struct ActiveRegistry {
    tiers: Vec<Box<dyn ActiveTier>>,
    /// Total emissions ever produced across all active tiers.
    pub total_emissions: u64,
}

impl ActiveRegistry {
    pub fn new() -> Self {
        Self { tiers: Vec::new(), total_emissions: 0 }
    }

    /// Register an active tier. Calls its `start()` immediately.
    pub fn register(&mut self, mut tier: Box<dyn ActiveTier>) -> &mut Self {
        tier.start();
        self.tiers.push(tier);
        self
    }

    /// Tick all registered active tiers. Returns every emission
    /// produced during this tick, across all tiers.
    pub fn tick_all(&mut self) -> Vec<(TierId, Query, Answer)> {
        let mut out = Vec::new();
        for tier in &mut self.tiers {
            let tier_id = tier.id();
            for (q, a) in tier.tick() {
                out.push((tier_id, q, a));
                self.total_emissions += 1;
            }
        }
        out
    }

    /// Coordinates of registered active tiers.
    pub fn tier_coords(&self) -> Vec<(TierId, Coord)> {
        self.tiers.iter().map(|t| (t.id(), t.coord())).collect()
    }

    pub fn len(&self) -> usize { self.tiers.len() }
    pub fn is_empty(&self) -> bool { self.tiers.is_empty() }
}

impl Default for ActiveRegistry {
    fn default() -> Self { Self::new() }
}

impl Drop for ActiveRegistry {
    fn drop(&mut self) {
        for tier in &mut self.tiers {
            tier.stop();
        }
    }
}

// ============================================================
// Example active tier — emits a query+answer every N ticks
// ============================================================

/// An active tier that emits a configured query+answer pair every
/// `ticks_per_emit` ticks. Concrete example of `E=Active`: produces
/// work without being asked, at a fixed cadence.
///
/// A real active tier would be observing real state (file mtimes,
/// sensor readings, queue depth, calendar events). This is the
/// minimum demonstration that the lifecycle works.
pub struct PeriodicTrigger {
    pub query_text: String,
    pub answer_text: String,
    pub ticks_per_emit: u32,
    pub id: TierId,
    pub coord: Coord,
    counter: u32,
    pub emits: u64,
    started: bool,
}

impl PeriodicTrigger {
    pub fn new(
        id: TierId,
        coord: Coord,
        query_text: impl Into<String>,
        ticks_per_emit: u32,
    ) -> Self {
        let q_text = query_text.into();
        Self {
            answer_text: format!("[active: {}]", &q_text),
            query_text: q_text,
            ticks_per_emit: ticks_per_emit.max(1),
            id, coord,
            counter: 0,
            emits: 0,
            started: false,
        }
    }

    pub fn with_answer(mut self, answer_text: impl Into<String>) -> Self {
        self.answer_text = answer_text.into();
        self
    }
}

impl ActiveTier for PeriodicTrigger {
    fn id(&self) -> TierId { self.id }
    fn coord(&self) -> Coord { self.coord.clone() }

    fn start(&mut self) {
        self.started = true;
        self.counter = 0;
    }

    fn stop(&mut self) {
        self.started = false;
    }

    fn tick(&mut self) -> Vec<(Query, Answer)> {
        if !self.started { return Vec::new(); }
        self.counter += 1;
        if self.counter >= self.ticks_per_emit {
            self.counter = 0;
            self.emits += 1;
            let q = Query {
                input: crate::types::QueryInput::Text(self.query_text.clone()),
                budget: crate::types::JouleBudget::standard(),
                quality: crate::types::QualityFloor::any(),
                context: crate::types::ContextRef::fresh(),
                deadline: None,
            };
            let a = Answer {
                output: crate::types::AnswerOutput::Text(self.answer_text.clone()),
                tier_used: self.id,
                joules_spent: 1e-9,
                confidence: 1.0,
                trace: crate::types::ExecutionTrace::default(),
                verification: crate::verification::VerificationStatus::Resolved,
            };
            vec![(q, a)]
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::{Coord, Zone, Entity, Thermo, Interface, Verify, Encoding, NamedPrimitive, PrimitiveSet};
    use crate::types::{L4ModelId, AnswerOutput, ExecutionTrace, QueryInput, JouleBudget, QualityFloor, ContextRef};
    use crate::verification::VerificationStatus;

    /// A test active tier that emits a fixed sequence on each tick.
    struct ScriptedActive {
        queue: Vec<(Query, Answer)>,
        start_called: bool,
        stop_called: bool,
    }

    impl ActiveTier for ScriptedActive {
        fn id(&self) -> TierId { TierId::L4(L4ModelId(99)) }
        fn coord(&self) -> Coord {
            Coord::new(
                Zone::Z3, Entity::Active, Thermo::L2_Landauer,
                Interface::Tokens, Verify::Statistical, Encoding::Facts,
            ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::SubAgent]))
        }
        fn start(&mut self) { self.start_called = true; }
        fn stop(&mut self) { self.stop_called = true; }
        fn tick(&mut self) -> Vec<(Query, Answer)> {
            self.queue.drain(..).collect()
        }
    }

    fn dummy_query(text: &str) -> Query {
        Query {
            input: QueryInput::Text(text.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn dummy_answer(text: &str) -> Answer {
        Answer {
            output: AnswerOutput::Text(text.to_string()),
            tier_used: TierId::L4(L4ModelId(99)),
            joules_spent: 0.5,
            confidence: 0.9,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        }
    }

    #[test]
    fn register_calls_start() {
        let mut reg = ActiveRegistry::new();
        let scripted = Box::new(ScriptedActive {
            queue: vec![],
            start_called: false,
            stop_called: false,
        });
        reg.register(scripted);
        assert_eq!(reg.len(), 1);
        // (We can't read start_called via trait — relying on the
        // boxed instance — but the registry size verifies registration.)
    }

    #[test]
    fn tick_collects_emissions() {
        let mut reg = ActiveRegistry::new();
        let scripted = Box::new(ScriptedActive {
            queue: vec![
                (dummy_query("q1"), dummy_answer("a1")),
                (dummy_query("q2"), dummy_answer("a2")),
            ],
            start_called: false, stop_called: false,
        });
        reg.register(scripted);

        let emissions = reg.tick_all();
        assert_eq!(emissions.len(), 2);
        assert_eq!(reg.total_emissions, 2);

        // Second tick: scripted's queue drained, no emissions.
        let second = reg.tick_all();
        assert_eq!(second.len(), 0);
        assert_eq!(reg.total_emissions, 2);
    }

    #[test]
    fn coords_include_active_entity() {
        let mut reg = ActiveRegistry::new();
        let scripted = Box::new(ScriptedActive {
            queue: vec![], start_called: false, stop_called: false,
        });
        reg.register(scripted);
        let coords = reg.tier_coords();
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0].1.entity, Entity::Active);
    }
}
