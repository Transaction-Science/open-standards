//! Coordinate-based routing. The router expresses "what kind of tier
//! do I want" as a constraint over the Synthesis seven-axis space,
//! then picks tiers whose `Coord` satisfies the constraint.
//!
//! This replaces enum-based routing (`Vec<TierId>`) with a richer
//! abstraction. Two examples:
//!
//! ```ignore
//! // "Anything Z1, deterministic, with full verifiability."
//! let pred = CoordPredicate::new()
//!     .zone_in(&[Zone::Z1])
//!     .thermo_at_most(Thermo::L1_Measure)
//!     .verify_at_least(Verify::Full);
//!
//! // "Tier that can answer Z2 inference with citation."
//! let pred = CoordPredicate::new()
//!     .zone_in(&[Zone::Z2, Zone::Z1_2])
//!     .verify_at_least(Verify::Citation)
//!     .thermo_at_most(Thermo::L2_Landauer);
//! ```
//!
//! The predicate is the *intent* — the router doesn't have to know
//! which concrete tiers exist. The cascade reports its `tier_coords`;
//! the router filters and sorts.

use crate::coord::*;

// ============================================================
// Predicate over the seven axes
// ============================================================

/// A constraint over Synthesis coordinates. Each axis is independently
/// constrainable; unconstrained axes are wildcards.
///
/// The thermodynamic and verifiability axes support ordering ("at
/// most T", "at least V") because they're naturally ranked. The other
/// axes are set-membership only.
#[derive(Debug, Clone, Default)]
pub struct CoordPredicate {
    pub zones: Option<Vec<Zone>>,
    pub entities: Option<Vec<Entity>>,
    /// Maximum acceptable thermodynamic class (cost ceiling).
    /// L0_Free < L1_Measure < L2_Landauer < L2_Max.
    pub thermo_at_most: Option<Thermo>,
    pub interfaces: Option<Vec<Interface>>,
    /// Minimum acceptable verifiability (quality floor).
    /// None < Statistical < Delayed < Citation < Full.
    pub verify_at_least: Option<Verify>,
    pub encodings: Option<Vec<Encoding>>,
    /// Required primitive — tier must include this in its P set.
    pub requires_primitive: Option<NamedPrimitive>,
    /// Forbidden primitive — tier must NOT include this.
    pub forbids_primitive: Option<NamedPrimitive>,
}

impl CoordPredicate {
    pub fn new() -> Self { Self::default() }

    pub fn zone_in(mut self, zones: &[Zone]) -> Self {
        self.zones = Some(zones.to_vec());
        self
    }

    pub fn entity_in(mut self, entities: &[Entity]) -> Self {
        self.entities = Some(entities.to_vec());
        self
    }

    pub fn thermo_at_most(mut self, t: Thermo) -> Self {
        self.thermo_at_most = Some(t);
        self
    }

    pub fn interface_in(mut self, ifaces: &[Interface]) -> Self {
        self.interfaces = Some(ifaces.to_vec());
        self
    }

    pub fn verify_at_least(mut self, v: Verify) -> Self {
        self.verify_at_least = Some(v);
        self
    }

    pub fn encoding_in(mut self, encs: &[Encoding]) -> Self {
        self.encodings = Some(encs.to_vec());
        self
    }

    pub fn requires(mut self, p: NamedPrimitive) -> Self {
        self.requires_primitive = Some(p);
        self
    }

    pub fn forbids(mut self, p: NamedPrimitive) -> Self {
        self.forbids_primitive = Some(p);
        self
    }

    /// Does the given coordinate satisfy this predicate?
    pub fn matches(&self, c: &Coord) -> bool {
        if let Some(zs) = &self.zones {
            if !zs.contains(&c.zone) { return false; }
        }
        if let Some(es) = &self.entities {
            if !es.contains(&c.entity) { return false; }
        }
        if let Some(max_t) = self.thermo_at_most {
            if thermo_rank(c.thermo) > thermo_rank(max_t) { return false; }
        }
        if let Some(ifs) = &self.interfaces {
            if !ifs.contains(&c.interface) { return false; }
        }
        if let Some(min_v) = self.verify_at_least {
            if c.verify.strictness() < min_v.strictness() { return false; }
        }
        if let Some(es) = &self.encodings {
            if !es.contains(&c.encoding) { return false; }
        }
        if let Some(p) = self.requires_primitive {
            if !c.primitives.contains(p) { return false; }
        }
        if let Some(p) = self.forbids_primitive {
            if c.primitives.contains(p) { return false; }
        }
        true
    }
}

/// Order Thermo from cheapest to most expensive.
fn thermo_rank(t: Thermo) -> u8 {
    match t {
        Thermo::L0_Free => 0,
        Thermo::L1_Measure => 1,
        Thermo::L2_Landauer => 2,
        Thermo::L2_Max => 3,
    }
}

// ============================================================
// Sort strategies
// ============================================================

/// How to order tiers that satisfy the predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortStrategy {
    /// Cheapest thermodynamic class first. Joule's default.
    CostFirst,
    /// Strictest verifiability first. For high-stakes queries.
    QualityFirst,
    /// Persistent before reactive — prefer caches. Useful when the
    /// query has been seen before and we want to maximize L0 hits.
    PersistenceFirst,
    /// Cheapest *effective* cost first. Multiplies declared thermo
    /// rank by the learned-μ correction from calibration. A tier
    /// with declared T=L1 but learned μ=5 (consistently 5× over its
    /// estimate) sorts behind a tier with T=L2 and μ=1.
    ///
    /// Requires a learned-μ function; without one, falls back to
    /// `CostFirst` behavior.
    CalibratedCost,
}

impl SortStrategy {
    /// Comparison key for a coordinate under this strategy. Lower
    /// values sort first.
    pub fn key(&self, c: &Coord) -> u32 {
        match self {
            Self::CostFirst | Self::CalibratedCost => thermo_rank(c.thermo) as u32,
            Self::QualityFirst => {
                // Higher strictness = sorts first (smaller key).
                (4 - c.verify.strictness()) as u32
            }
            Self::PersistenceFirst => match c.entity {
                Entity::Persistent => 0,
                Entity::Reactive => 1,
                Entity::Active => 2,
                Entity::Emergent => 3,
            },
        }
    }

    /// Calibration-aware sort key: floating-point so we can multiply
    /// the discrete thermo rank by the learned μ correction. Falls
    /// back to `key()` for non-calibrated strategies.
    pub fn calibrated_key(&self, c: &Coord, learned_mu: f64) -> f64 {
        match self {
            Self::CalibratedCost => {
                // Effective rank = base rank × learned μ. A tier with
                // declared T=L0 (rank 0) and any μ keeps rank 0;
                // tiers further up the thermo ladder pay the multi-
                // plier on their rank.
                // Use rank+1 so L0 isn't immune to μ blow-up.
                let base = (thermo_rank(c.thermo) + 1) as f64;
                base * learned_mu.max(0.01)
            }
            _ => self.key(c) as f64,
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconstrained_predicate_matches_all() {
        let pred = CoordPredicate::new();
        assert!(pred.matches(&prebuilt::l0_cache()));
        assert!(pred.matches(&prebuilt::l4_frontier_model()));
    }

    #[test]
    fn zone_constraint_filters() {
        let pred = CoordPredicate::new().zone_in(&[Zone::Z1]);
        assert!(pred.matches(&prebuilt::l0_cache()));
        assert!(pred.matches(&prebuilt::l1_execute()));
        assert!(!pred.matches(&prebuilt::l4_frontier_model()));
    }

    #[test]
    fn thermo_ceiling_filters() {
        // "I want cheap tiers only" — at most L1.
        let pred = CoordPredicate::new().thermo_at_most(Thermo::L1_Measure);
        assert!(pred.matches(&prebuilt::l0_cache()));
        assert!(pred.matches(&prebuilt::l1_execute()));
        assert!(!pred.matches(&prebuilt::l2_embedder()));
        assert!(!pred.matches(&prebuilt::l4_frontier_model()));
    }

    #[test]
    fn verify_floor_filters() {
        // "I need full verification" — only the deterministic tiers.
        let pred = CoordPredicate::new().verify_at_least(Verify::Full);
        assert!(pred.matches(&prebuilt::l0_cache()));
        assert!(pred.matches(&prebuilt::l1_execute()));
        assert!(!pred.matches(&prebuilt::l4_frontier_model()));
        // RPC has citation, which is < full.
        assert!(!pred.matches(&prebuilt::rpc_tier()));
    }

    #[test]
    fn requires_primitive_filters() {
        let pred = CoordPredicate::new().requires(NamedPrimitive::Arithmetic);
        assert!(pred.matches(&prebuilt::l1_execute()));
        assert!(!pred.matches(&prebuilt::l1_regex()));
        assert!(!pred.matches(&prebuilt::l4_frontier_model()));
    }

    #[test]
    fn forbids_primitive_filters() {
        // "Don't use the frontier."
        let pred = CoordPredicate::new().forbids(NamedPrimitive::AttentionFull);
        assert!(pred.matches(&prebuilt::l0_cache()));
        assert!(pred.matches(&prebuilt::l3_small_model()));
        assert!(!pred.matches(&prebuilt::l4_frontier_model()));
    }

    #[test]
    fn compound_predicate() {
        // Z1 inference with full verification, cheap.
        let pred = CoordPredicate::new()
            .zone_in(&[Zone::Z1])
            .thermo_at_most(Thermo::L1_Measure)
            .verify_at_least(Verify::Full);
        assert!(pred.matches(&prebuilt::l0_cache()));
        assert!(pred.matches(&prebuilt::l1_execute()));
        assert!(pred.matches(&prebuilt::l1_regex()));
        assert!(!pred.matches(&prebuilt::l2_embedder()));
        assert!(!pred.matches(&prebuilt::l4_frontier_model()));
    }

    #[test]
    fn cost_first_sorts_cheapest_first() {
        let s = SortStrategy::CostFirst;
        let l0_key = s.key(&prebuilt::l0_cache());
        let l4_key = s.key(&prebuilt::l4_frontier_model());
        assert!(l0_key < l4_key);
    }

    #[test]
    fn quality_first_sorts_strictest_first() {
        let s = SortStrategy::QualityFirst;
        let l1_key = s.key(&prebuilt::l1_execute());      // Full
        let l4_key = s.key(&prebuilt::l4_frontier_model()); // Statistical
        // L1 (Full) should sort before L4 (Statistical).
        assert!(l1_key < l4_key);
    }

    #[test]
    fn persistence_first_sorts_caches_first() {
        let s = SortStrategy::PersistenceFirst;
        let cache_key = s.key(&prebuilt::l0_cache());      // Persistent
        let model_key = s.key(&prebuilt::l4_frontier_model()); // Reactive
        assert!(cache_key < model_key);
    }
}
