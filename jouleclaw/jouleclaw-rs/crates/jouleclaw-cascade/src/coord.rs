//! Synthesis coordinate — `c = ⟨Z, E, T, P, I, V, R⟩`.
//!
//! Every tier in Joule has a coordinate that locates it on the
//! Periodic Stack of Digital Information Synthesis. The coordinate is
//! orthogonal to (and richer than) the legacy `TierId` enum — `TierId`
//! tells you "which tier slot" (L0/L1/L2/L3/L4); the coordinate tells
//! you "what kind of synthesis this tier performs and at what cost
//! class."
//!
//! The seven axes:
//!
//!   Z — Zone:           domain gradient. Where does math meet words.
//!   E — Entity kind:    persistent / reactive / active / emergent.
//!   T — Thermodynamic:  L₀ / L₁ / L₂ / L₂max cost class.
//!   P — Primitive set:  which compute primitives the tier composes.
//!   I — Interface:      none / tokens / signals / bodies.
//!   V — Verifiability:  full / citation / delayed / statistical / none.
//!   R — Encoding:       facts / navigation / world_model / grammar / none.
//!
//! Five axes are finite-valued (Z, E, T, I, V, R). One (P) is a set —
//! a subset of the 258 primitives in the compute stack. Together they
//! discretize a space of `5 · 4 · 4 · 4 · 5 · 5 = 8,000` cells (plus
//! the P set on top).
//!
//! Joule uses the coordinate for three things:
//!
//!   1. Tier selection. The router can ask "which tiers can satisfy a
//!      query that needs Z₂ work with V=citation?"
//!   2. Composition validation. When two tiers are stacked (e.g. L3
//!      drafts, L4 verifies), their coordinates must be compatible.
//!   3. Cost accounting. The T axis classifies the cost band; the
//!      P set determines the impedance mismatch μ(p, H) on the
//!      tier's substrate.
//!
//! See `https://synthesis.openie.dev/axes` for the full specification.

// ============================================================
// Axis 1 — Zone: where math meets words
// ============================================================

/// Z axis. The domain gradient between deterministic math and
/// unbounded generation.
///
/// `Z1` — math = words. Bounded determinism. Outputs derivable from
///        inputs (arithmetic, parsing, proof checking). Verification
///        is a comparison.
/// `Z1_2` — transition. Mostly Z1 with some Z2 character.
/// `Z2` — math ≈ words. Bounded inference. Authoritative structure
///        constrains valid answers without fully determining them
///        (law, regulated medicine, contract interpretation).
/// `Z2_3` — transition. Mostly Z2 with some Z3 character.
/// `Z3` — math ≠ words. Unbounded generation. No authoritative
///        ground truth (creative work, aesthetic generation).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Zone {
    Z1,
    Z1_2,
    Z2,
    Z2_3,
    Z3,
}

impl Zone {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Z1 => "Z1",
            Self::Z1_2 => "Z1↔2",
            Self::Z2 => "Z2",
            Self::Z2_3 => "Z2↔3",
            Self::Z3 => "Z3",
        }
    }
    pub fn all() -> [Zone; 5] {
        [Self::Z1, Self::Z1_2, Self::Z2, Self::Z2_3, Self::Z3]
    }
}

// ============================================================
// Axis 2 — Entity kind: what kind of thing the system is
// ============================================================

/// E axis. What mode of being the synthesizer occupies.
///
/// `Persistent` — static substance (a cache, a lookup table).
/// `Reactive`   — responds to inputs; no internal initiation.
/// `Active`     — initiates its own actions; continuity over time.
/// `Emergent`   — no independent existence; emerges from composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Entity {
    Persistent,
    Reactive,
    Active,
    Emergent,
}

impl Entity {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Persistent => "persistent",
            Self::Reactive => "reactive",
            Self::Active => "active",
            Self::Emergent => "emergent",
        }
    }
    pub fn all() -> [Entity; 4] {
        [Self::Persistent, Self::Reactive, Self::Active, Self::Emergent]
    }
}

// ============================================================
// Axis 3 — Thermodynamic class: the cost floor
// ============================================================

/// T axis. The Landauer-floor cost class of the synthesizer's
/// operation.
///
/// `L0_Free`    — free lookup. Cache hit, stored answer. No erasure.
/// `L1_Measure` — deterministic measurement. State read with bounded
///                erasure (a regex match, arithmetic eval).
/// `L2_Landauer` — every erasure pays the Landauer cost. The normal
///                regime for neural inference.
/// `L2_Max`     — wide-impedance regime. The algorithm asks for far
///                more than the silicon natively offers (long-context
///                attention on dense substrate, naive transformer on
///                CPU at scale).
///
/// The `T` value of a tier is the floor of its cost class on its
/// substrate, NOT an absolute joule number. Absolute joules are
/// produced by `CostEstimate` (in `cost.rs`).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Thermo {
    L0_Free,
    L1_Measure,
    L2_Landauer,
    L2_Max,
}

impl Thermo {
    pub fn name(&self) -> &'static str {
        match self {
            Self::L0_Free => "L0",
            Self::L1_Measure => "L1",
            Self::L2_Landauer => "L2",
            Self::L2_Max => "L2max",
        }
    }
    pub fn all() -> [Thermo; 4] {
        [Self::L0_Free, Self::L1_Measure, Self::L2_Landauer, Self::L2_Max]
    }
}

// ============================================================
// Axis 4 — Composition primitives: P ⊆ Stack258
// ============================================================

/// P axis. The set of compute primitives the tier composes.
///
/// In the full Synthesis framework this is a subset of the 258
/// primitives in the compute stack. Joule's runtime currently uses
/// a small named subset; we represent the rest as a bitset and
/// expose only the ones the runtime actually needs to know about.
///
/// The point of the P axis isn't enumeration — it's separability.
/// Two tiers with disjoint P sets are doing different work, even if
/// they share Z, T, and V.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrimitiveSet {
    /// Named primitives the runtime knows about explicitly.
    pub named: Vec<NamedPrimitive>,
    /// Count of additional unnamed primitives from Stack258.
    /// Used for impedance-mismatch estimation; never enumerated.
    pub stack258_count: u16,
}

/// Named primitives that Joule's cascade explicitly composes.
/// These are the operations the runtime can dispatch by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamedPrimitive {
    // L0 — memory access
    Lookup,
    PrefixMatch,

    // L1 — deterministic
    Tokenize,
    Detokenize,
    Regex,
    Parse,
    Arithmetic,
    Template,
    Retrieve,

    // L2 — embedding & classification
    Embed,
    Classify,
    NearestNeighbor,

    // L3/L4 — neural
    AttentionFull,
    AttentionSliding,
    AttentionGrouped,
    AttentionLatent,
    Convolution,
    StateSpace,
    MlpForward,
    KvUpdate,
    Sample,

    // composition
    SpeculativeDraft,
    SpeculativeVerify,
    ToolCall,
    SubAgent,
    CitationCheck,
}

impl PrimitiveSet {
    pub fn empty() -> Self {
        Self { named: Vec::new(), stack258_count: 0 }
    }

    pub fn of(prims: &[NamedPrimitive]) -> Self {
        Self {
            named: prims.iter().copied().collect(),
            stack258_count: 0,
        }
    }

    pub fn with_stack258_count(mut self, n: u16) -> Self {
        self.stack258_count = n;
        self
    }

    pub fn contains(&self, p: NamedPrimitive) -> bool {
        self.named.contains(&p)
    }

    pub fn size(&self) -> usize {
        self.named.len() + self.stack258_count as usize
    }

    /// True if `self` and `other` share at least one named primitive.
    /// Used to detect when two tiers are doing similar work.
    pub fn overlaps(&self, other: &PrimitiveSet) -> bool {
        self.named.iter().any(|p| other.named.contains(p))
    }
}

// ============================================================
// Axis 5 — Interface: where the loop closes
// ============================================================

/// I axis. Through what channel outputs reach the world.
///
/// `None`    — internal only; no external output.
/// `Tokens`  — text or token stream out.
/// `Signals` — structured signals (function calls, RPC, control).
/// `Bodies`  — physical actuation (robotics, manufacturing).
///
/// The interface determines failure modes and feedback-loop latency.
/// Bodies are stakes-bearing — failure has physical consequence.
/// Tokens and signals are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interface {
    None,
    Tokens,
    Signals,
    Bodies,
}

impl Interface {
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tokens => "tokens",
            Self::Signals => "signals",
            Self::Bodies => "bodies",
        }
    }
    pub fn all() -> [Interface; 4] {
        [Self::None, Self::Tokens, Self::Signals, Self::Bodies]
    }
}

// ============================================================
// Axis 6 — Verifiability: how the output can be checked
// ============================================================

/// V axis. How the synthesizer's output can be verified.
///
/// `Full`        — checked at decision time against ground truth
///                 (a math result against an evaluator).
/// `Citation`    — checked against a retrieved source.
/// `Delayed`     — checked after consequences play out.
/// `Statistical` — checked in aggregate (benchmark accuracy).
/// `None`        — no checking mechanism exists.
///
/// Verifiability is structural, not optional. The cascade's quality
/// floor maps to this axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verify {
    Full,
    Citation,
    Delayed,
    Statistical,
    None,
}

impl Verify {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Citation => "citation",
            Self::Delayed => "delayed",
            Self::Statistical => "statistical",
            Self::None => "none",
        }
    }

    /// Higher = stricter verification. Used to compare quality floors.
    pub fn strictness(&self) -> u8 {
        match self {
            Self::Full => 4,
            Self::Citation => 3,
            Self::Delayed => 2,
            Self::Statistical => 1,
            Self::None => 0,
        }
    }

    pub fn all() -> [Verify; 5] {
        [Self::Full, Self::Citation, Self::Delayed, Self::Statistical, Self::None]
    }
}

// ============================================================
// Axis 7 — Encoding regime: what the weights/state hold
// ============================================================

/// R axis. What the synthesizer's internal state actually encodes.
///
/// `Facts`      — direct factual content stored in the substrate.
/// `Navigation` — how to find an external source (RAG, tool use,
///                pointers). The mathground inversion: encode
///                navigation, not facts.
/// `WorldModel` — a model of the dynamics of some domain.
/// `Grammar`    — the syntactic / structural rules of a language
///                or format.
/// `None`       — stateless tier (pure function over inputs).
///
/// The R axis is the key distinction between RAG (R=navigation) and
/// a generative model (R=facts or worldmodel). Encoding facts is
/// L₂-expensive and only statistically verifiable. Encoding
/// navigation is L₁ with citation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    Facts,
    Navigation,
    WorldModel,
    Grammar,
    None,
}

impl Encoding {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Facts => "facts",
            Self::Navigation => "navigation",
            Self::WorldModel => "world_model",
            Self::Grammar => "grammar",
            Self::None => "none",
        }
    }
    pub fn all() -> [Encoding; 5] {
        [Self::Facts, Self::Navigation, Self::WorldModel, Self::Grammar, Self::None]
    }
}

// ============================================================
// The coordinate tuple
// ============================================================

/// The Synthesis coordinate. `c = ⟨Z, E, T, P, I, V, R⟩`.
///
/// Every tier in Joule has one. Two tiers with the same coordinate
/// are doing the same kind of synthesis (though possibly on different
/// substrates). Two tiers with different coordinates are doing
/// different work.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Coord {
    pub zone: Zone,
    pub entity: Entity,
    pub thermo: Thermo,
    pub primitives: PrimitiveSet,
    pub interface: Interface,
    pub verify: Verify,
    pub encoding: Encoding,
}

impl Coord {
    /// Construct a coordinate with placeholder P set. Useful for
    /// quickly stamping a tier with its discrete-axis values; fill in
    /// `primitives` separately.
    pub fn new(
        zone: Zone, entity: Entity, thermo: Thermo,
        interface: Interface, verify: Verify, encoding: Encoding,
    ) -> Self {
        Self {
            zone, entity, thermo,
            primitives: PrimitiveSet::empty(),
            interface, verify, encoding,
        }
    }

    pub fn with_primitives(mut self, p: PrimitiveSet) -> Self {
        self.primitives = p;
        self
    }

    /// Compact discrete-axis ID. Used for cell-counting and indexing.
    /// 5 · 4 · 4 · 4 · 5 · 5 = 8,000 possible values.
    pub fn cell_id(&self) -> u16 {
        let z = self.zone as u16;
        let e = self.entity as u16;
        let t = self.thermo as u16;
        let i = self.interface as u16;
        let v = self.verify as u16;
        let r = self.encoding as u16;
        // mixed-radix: Z·E·T·I·V·R with sizes 5,4,4,4,5,5
        z * (4*4*4*5*5)
          + e * (4*4*5*5)
          + t * (4*5*5)
          + i * (5*5)
          + v * 5
          + r
    }

    /// Human-readable cell label (without P).
    pub fn cell_label(&self) -> String {
        format!("{}·{}·{}·{}·{}·{}",
            self.zone.name(), self.entity.name(), self.thermo.name(),
            self.interface.name(), self.verify.name(), self.encoding.name())
    }
}

// ============================================================
// Pre-built coordinates for Joule's existing tiers
// ============================================================

/// Coordinates for each tier in the current Joule cascade.
/// Re-coordinating Joule onto the Synthesis map.
pub mod prebuilt {
    use super::*;

    /// L0 cache — exact-match hot lookup.
    /// Z1 (deterministic), persistent, L0_Free, full verification
    /// (output equals stored answer), encodes facts (the cached
    /// answers themselves).
    pub fn l0_cache() -> Coord {
        Coord::new(
            Zone::Z1, Entity::Persistent, Thermo::L0_Free,
            Interface::None, Verify::Full, Encoding::Facts,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::Lookup]))
    }

    /// L1::Execute — arithmetic evaluation.
    /// Z1, reactive (responds to a query), L1_Measure, full
    /// verification (the math is checkable), no encoding (stateless).
    pub fn l1_execute() -> Coord {
        Coord::new(
            Zone::Z1, Entity::Reactive, Thermo::L1_Measure,
            Interface::Tokens, Verify::Full, Encoding::None,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::Arithmetic]))
    }

    /// L1::Regex — pattern extraction.
    /// Z1, reactive, L1_Measure, full verification (the match either
    /// occurs or doesn't), grammar encoding (the patterns).
    pub fn l1_regex() -> Coord {
        Coord::new(
            Zone::Z1, Entity::Reactive, Thermo::L1_Measure,
            Interface::Tokens, Verify::Full, Encoding::Grammar,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::Regex]))
    }

    /// L1::TemplateFill — FAQ-shaped response.
    /// Z1, reactive, L1_Measure, full verification (the template
    /// fires or doesn't), facts encoding (the template strings).
    pub fn l1_template() -> Coord {
        Coord::new(
            Zone::Z1, Entity::Reactive, Thermo::L1_Measure,
            Interface::Tokens, Verify::Full, Encoding::Facts,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::Template]))
    }

    /// L2 embedder — semantic similarity.
    /// Z1↔2 (math approximates words), reactive, L2_Landauer,
    /// statistical verification, world-model encoding (the embedding
    /// space).
    pub fn l2_embedder() -> Coord {
        Coord::new(
            Zone::Z1_2, Entity::Reactive, Thermo::L2_Landauer,
            Interface::Signals, Verify::Statistical, Encoding::WorldModel,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::Embed]))
    }

    /// L2 classifier — intent / category.
    /// Z2 (bounded inference within a label set), reactive,
    /// L2_Landauer, statistical, world-model.
    pub fn l2_classifier() -> Coord {
        Coord::new(
            Zone::Z2, Entity::Reactive, Thermo::L2_Landauer,
            Interface::Signals, Verify::Statistical, Encoding::WorldModel,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::Classify]))
    }

    /// L3 small generalist — bounded generation.
    /// Z2↔3, reactive, L2_Landauer, statistical, facts+world-model.
    pub fn l3_small_model() -> Coord {
        Coord::new(
            Zone::Z2_3, Entity::Reactive, Thermo::L2_Landauer,
            Interface::Tokens, Verify::Statistical, Encoding::Facts,
        ).with_primitives(PrimitiveSet::of(&[
            NamedPrimitive::AttentionGrouped,
            NamedPrimitive::MlpForward,
            NamedPrimitive::KvUpdate,
            NamedPrimitive::Sample,
        ]))
    }

    /// L4 frontier model — unbounded generation.
    /// Z3, reactive, L2_Max (wide mismatch), statistical, facts.
    pub fn l4_frontier_model() -> Coord {
        Coord::new(
            Zone::Z3, Entity::Reactive, Thermo::L2_Max,
            Interface::Tokens, Verify::Statistical, Encoding::Facts,
        ).with_primitives(PrimitiveSet::of(&[
            NamedPrimitive::AttentionFull,
            NamedPrimitive::MlpForward,
            NamedPrimitive::KvUpdate,
            NamedPrimitive::Sample,
        ]))
    }

    /// Wire-protocol federated tier — remote cache.
    /// Z1 (deterministic lookup), persistent (the remote cache),
    /// L0_Free (the heavy work was paid elsewhere), citation
    /// verifiability (the wire metadata is the citation), facts.
    pub fn rpc_tier() -> Coord {
        Coord::new(
            Zone::Z1, Entity::Persistent, Thermo::L0_Free,
            Interface::Signals, Verify::Citation, Encoding::Facts,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::Lookup]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_id_within_range() {
        for z in Zone::all() {
            for e in Entity::all() {
                for t in Thermo::all() {
                    for i in Interface::all() {
                        for v in Verify::all() {
                            for r in Encoding::all() {
                                let c = Coord::new(z, e, t, i, v, r);
                                assert!(c.cell_id() < 8000,
                                    "cell_id {} out of range", c.cell_id());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cell_id_is_unique_per_discrete_tuple() {
        let mut seen = std::collections::HashSet::new();
        for z in Zone::all() {
            for e in Entity::all() {
                for t in Thermo::all() {
                    for i in Interface::all() {
                        for v in Verify::all() {
                            for r in Encoding::all() {
                                let c = Coord::new(z, e, t, i, v, r);
                                assert!(seen.insert(c.cell_id()),
                                    "duplicate cell_id at {:?}", c.cell_label());
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), 8000);
    }

    #[test]
    fn prebuilt_coordinates_have_distinct_cells() {
        // The eight named tiers in current Joule should each occupy
        // a distinct cell on the discrete axes.
        let coords = [
            prebuilt::l0_cache(),
            prebuilt::l1_execute(),
            prebuilt::l1_regex(),
            prebuilt::l1_template(),
            prebuilt::l2_embedder(),
            prebuilt::l2_classifier(),
            prebuilt::l3_small_model(),
            prebuilt::l4_frontier_model(),
            prebuilt::rpc_tier(),
        ];
        let cells: std::collections::HashSet<_> =
            coords.iter().map(|c| c.cell_id()).collect();
        // Not every tier needs a unique cell, but currently they do —
        // and if two tiers collapse to the same cell, that's worth
        // flagging.
        assert_eq!(cells.len(), coords.len(),
            "two pre-built tiers occupy the same cell — collision is OK \
             but flag it: {:?}",
            coords.iter().map(|c| c.cell_label()).collect::<Vec<_>>());
    }

    #[test]
    fn primitive_set_overlap_detects_shared_work() {
        let l3 = prebuilt::l3_small_model();
        let l4 = prebuilt::l4_frontier_model();
        // Both use MlpForward, KvUpdate, Sample — they overlap.
        assert!(l3.primitives.overlaps(&l4.primitives),
            "L3 and L4 should share primitives (MlpForward etc.)");
    }

    #[test]
    fn primitive_set_overlap_detects_distinct_work() {
        let l0 = prebuilt::l0_cache();
        let l4 = prebuilt::l4_frontier_model();
        // L0 is a lookup, L4 is neural — no shared named primitive.
        assert!(!l0.primitives.overlaps(&l4.primitives),
            "L0 cache and L4 frontier should not share primitives");
    }

    #[test]
    fn verify_strictness_ordering() {
        assert!(Verify::Full.strictness() > Verify::Citation.strictness());
        assert!(Verify::Citation.strictness() > Verify::Delayed.strictness());
        assert!(Verify::Delayed.strictness() > Verify::Statistical.strictness());
        assert!(Verify::Statistical.strictness() > Verify::None.strictness());
    }

    #[test]
    fn z1_tiers_are_cheap_z3_tiers_are_expensive() {
        // Sanity: any tier in Z1 should be at L0 or L1, not L2max.
        // Any tier in Z3 should be at L2 or L2max, not L0.
        assert!(matches!(prebuilt::l0_cache().thermo,
            Thermo::L0_Free | Thermo::L1_Measure));
        assert!(matches!(prebuilt::l1_execute().thermo,
            Thermo::L0_Free | Thermo::L1_Measure));
        assert!(matches!(prebuilt::l4_frontier_model().thermo,
            Thermo::L2_Landauer | Thermo::L2_Max));
    }
}
