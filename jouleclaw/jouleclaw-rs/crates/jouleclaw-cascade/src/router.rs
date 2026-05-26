//! The `Router` trait — the cascade's dispatch decision.
//!
//! Today the runtime walks tiers in registration order, asking each
//! tier whether it can handle the query (via `estimate_cost`). That
//! works for a handful of tiers; it doesn't scale. The router examines
//! the query and produces a *routing plan*: which tiers to try, in
//! what order, with optional joule estimates.
//!
//! For R4 the router is rule-based — deterministic pattern matching
//! over the query text. R6 augments it with ML-driven classification.
//! The trait surface is the same either way; only the implementation
//! changes.
//!
//! Determinism contract: identical `Query` + identical router state →
//! identical `RoutingPlan`. The router IS allowed stochastic
//! sub-components (R6's classifier), but those must themselves be
//! deterministic given fixed weights.

use crate::types::*;

pub trait Router: Send {
    /// Produce a routing plan for the query.
    fn route(&self, q: &Query) -> RoutingPlan;

    /// Cheap upper bound on the joules this router would spend on
    /// `q`. Used by the cascade to ensure routing overhead stays
    /// within the query's budget.
    fn estimate_overhead(&self, q: &Query) -> f64;

    /// Coordinate-aware routing. Given the cascade's tier coords,
    /// produce a plan. Default implementation falls back to
    /// `route(q)`; coordinate-aware routers override this.
    ///
    /// `tier_coords` is `[(TierId, Coord)]` — the runtime gives the
    /// router the list of registered tiers along with their reported
    /// Synthesis coordinates.
    fn route_with_coords(
        &self,
        q: &Query,
        _tier_coords: &[(TierId, crate::coord::Coord)],
    ) -> RoutingPlan {
        self.route(q)
    }
}

/// What the router tells the runtime to do.
#[derive(Debug, Clone)]
pub struct RoutingPlan {
    /// Tiers to try, in priority order. The runtime walks this list
    /// and stops at the first tier that produces an answer.
    ///
    /// If empty, the runtime falls back to walking all registered
    /// tiers in their registered order (the pre-R4 behavior).
    pub tier_order: Vec<TierId>,

    /// Joules spent producing this plan. Counted against the query
    /// budget.
    pub router_joules: f64,

    /// Human-readable explanation of WHY this ordering. Surfaces in
    /// the answer's trace.
    pub reasoning: String,
}

impl RoutingPlan {
    /// A fallback plan that says "walk everything in registration
    /// order." Used when the router has no opinion about the query.
    pub fn fallback(router_joules: f64, reasoning: impl Into<String>) -> Self {
        Self {
            tier_order: Vec::new(),
            router_joules,
            reasoning: reasoning.into(),
        }
    }

    pub fn is_fallback(&self) -> bool {
        self.tier_order.is_empty()
    }
}
