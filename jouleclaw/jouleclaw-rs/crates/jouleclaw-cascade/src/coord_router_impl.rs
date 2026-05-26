//! Coordinate-aware router. Filters cascade tiers by `CoordPredicate`
//! and orders the survivors by `SortStrategy`.
//!
//! Unlike the rule-based router which hard-codes intent → tier maps,
//! this router doesn't know which concrete tiers exist — it asks the
//! cascade and matches by coordinate.

use crate::coord::Coord;
use crate::coord_route::{CoordPredicate, SortStrategy};
use crate::router::{Router, RoutingPlan};
use crate::types::*;

/// A routing rule: "for queries matching this text test, use this
/// coordinate predicate and sort strategy."
#[derive(Clone)]
pub struct CoordRule {
    /// Test function over the query text. Returns true if this rule
    /// applies to the query.
    pub matches: fn(&str) -> bool,
    pub predicate: CoordPredicate,
    pub sort: SortStrategy,
    pub label: &'static str,
}

pub struct CoordRouter {
    rules: Vec<CoordRule>,
    /// Predicate to use when no rule matches. Defaults to
    /// "any tier", sorted by cost.
    fallback_predicate: CoordPredicate,
    fallback_sort: SortStrategy,
    /// Joule cost of running this router. Rules-based, so flat.
    overhead_joules: f64,
    /// Optional learned-μ source. When set and a rule uses
    /// `SortStrategy::CalibratedCost`, the router consults this
    /// function to weight tier costs by observed reality.
    learned_mu: Option<Box<dyn Fn(&Coord) -> f64 + Send + Sync>>,
}

impl CoordRouter {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            fallback_predicate: CoordPredicate::new(),
            fallback_sort: SortStrategy::CostFirst,
            overhead_joules: 5e-9,
            learned_mu: None,
        }
    }

    pub fn with_rule(mut self, rule: CoordRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_fallback(
        mut self, predicate: CoordPredicate, sort: SortStrategy,
    ) -> Self {
        self.fallback_predicate = predicate;
        self.fallback_sort = sort;
        self
    }

    /// Provide a learned-μ function. The router consults it when a
    /// rule uses `SortStrategy::CalibratedCost`, weighting each tier's
    /// effective cost rank by `learned_mu(&coord)`.
    ///
    /// Typical use: `with_learned_mu(move |c| calibration.learned_mu(c))`.
    pub fn with_learned_mu<F>(mut self, f: F) -> Self
    where F: Fn(&Coord) -> f64 + Send + Sync + 'static,
    {
        self.learned_mu = Some(Box::new(f));
        self
    }

    /// Build a CoordRouter pre-loaded with sensible defaults for a
    /// general-purpose cascade.
    pub fn defaults() -> Self {
        use crate::coord::{Zone, Thermo, Verify};

        Self::new()
            // Arithmetic-shaped query: route through all Z1 deterministic
            // tiers in cost order. Execute (cheapest) tries first; if
            // it refuses (e.g. the query had digits but wasn't math),
            // the cascade falls through to other Z1 tiers.
            .with_rule(CoordRule {
                matches: is_arithmetic,
                predicate: CoordPredicate::new()
                    .zone_in(&[Zone::Z1])
                    .thermo_at_most(Thermo::L1_Measure),
                sort: SortStrategy::CostFirst,
                label: "arithmetic",
            })
            // Extraction — Z1, full or citation verifiability, all
            // deterministic tiers.
            .with_rule(CoordRule {
                matches: is_extraction,
                predicate: CoordPredicate::new()
                    .zone_in(&[Zone::Z1])
                    .thermo_at_most(Thermo::L1_Measure),
                sort: SortStrategy::CostFirst,
                label: "extraction",
            })
            // Greeting / FAQ — Z1, full verification only.
            .with_rule(CoordRule {
                matches: is_greeting,
                predicate: CoordPredicate::new()
                    .zone_in(&[Zone::Z1])
                    .thermo_at_most(Thermo::L1_Measure)
                    .verify_at_least(Verify::Full),
                sort: SortStrategy::CostFirst,
                label: "greeting",
            })
            // Default — anything goes, cheapest first.
            .with_fallback(
                CoordPredicate::new(),
                SortStrategy::CostFirst,
            )
    }

    /// Build a richer CoordRouter for cascades that include
    /// navigation (Retrieve) and body tiers. Stacks on top of
    /// `defaults()` with three additional rules:
    ///
    ///   * Knowledge questions ("what is", "explain") → prefer
    ///     R=Navigation tiers (Retrieve with citation) before falling
    ///     through to R=Facts tiers (raw model generation).
    ///   * World-action queries ("write", "send", "deploy") → only
    ///     route to I=Bodies tiers. Outside of body tiers, refuse.
    ///   * High-trust requests ("verify", "confirm") → require
    ///     V_Full or V_Citation.
    pub fn agentic() -> Self {
        use crate::coord::{Zone, Thermo, Verify, Encoding, Interface};

        Self::new()
            // Reuse the defaults' rules.
            .with_rule(CoordRule {
                matches: is_arithmetic,
                predicate: CoordPredicate::new()
                    .zone_in(&[Zone::Z1])
                    .thermo_at_most(Thermo::L1_Measure),
                sort: SortStrategy::CostFirst,
                label: "arithmetic",
            })
            .with_rule(CoordRule {
                matches: is_extraction,
                predicate: CoordPredicate::new()
                    .zone_in(&[Zone::Z1])
                    .thermo_at_most(Thermo::L1_Measure),
                sort: SortStrategy::CostFirst,
                label: "extraction",
            })
            .with_rule(CoordRule {
                matches: is_greeting,
                predicate: CoordPredicate::new()
                    .zone_in(&[Zone::Z1])
                    .thermo_at_most(Thermo::L1_Measure)
                    .verify_at_least(Verify::Full),
                sort: SortStrategy::CostFirst,
                label: "greeting",
            })
            // Knowledge questions — prefer R=Navigation (Retrieve)
            // before R=Facts (raw model). Citation > Statistical.
            .with_rule(CoordRule {
                matches: is_knowledge_question,
                predicate: CoordPredicate::new()
                    .encoding_in(&[Encoding::Navigation, Encoding::Facts])
                    .verify_at_least(Verify::Statistical),
                sort: SortStrategy::QualityFirst,
                label: "knowledge_question",
            })
            // World-action — only tiers with I=Bodies should respond.
            // Without an explicit body tier, the cascade refuses
            // (better than hallucinating a fake action).
            .with_rule(CoordRule {
                matches: is_world_action,
                predicate: CoordPredicate::new()
                    .interface_in(&[Interface::Bodies]),
                sort: SortStrategy::CostFirst,
                label: "world_action",
            })
            // High-trust — require checkable verification.
            .with_rule(CoordRule {
                matches: is_high_trust,
                predicate: CoordPredicate::new()
                    .verify_at_least(Verify::Citation),
                sort: SortStrategy::QualityFirst,
                label: "high_trust",
            })
            .with_fallback(
                CoordPredicate::new(),
                SortStrategy::CostFirst,
            )
    }
}

impl Default for CoordRouter {
    fn default() -> Self { Self::new() }
}

impl Router for CoordRouter {
    fn route(&self, _q: &Query) -> RoutingPlan {
        // Without cascade coords, we can't do coordinate routing.
        // Fall back.
        RoutingPlan::fallback(
            self.overhead_joules,
            "coord_router needs cascade coords; using default order".to_string(),
        )
    }

    fn estimate_overhead(&self, _q: &Query) -> f64 {
        self.overhead_joules
    }

    fn route_with_coords(
        &self, q: &Query, tier_coords: &[(TierId, Coord)],
    ) -> RoutingPlan {
        let text = match &q.input {
            QueryInput::Text(s) => s.as_str(),
            _ => "",
        };

        // Find the first matching rule.
        let (predicate, sort, label) = self.rules.iter()
            .find(|r| (r.matches)(text))
            .map(|r| (&r.predicate, r.sort, r.label))
            .unwrap_or((&self.fallback_predicate, self.fallback_sort, "fallback"));

        // Filter and sort tier coords.
        let mut matching: Vec<&(TierId, Coord)> = tier_coords.iter()
            .filter(|(_, c)| predicate.matches(c))
            .collect();

        // Use calibrated sort when the strategy demands it and we
        // have a learned-μ function. Otherwise the integer key.
        if sort == SortStrategy::CalibratedCost && self.learned_mu.is_some() {
            let mu_fn = self.learned_mu.as_ref().unwrap();
            matching.sort_by(|(_, a), (_, b)| {
                let ka = sort.calibrated_key(a, mu_fn(a));
                let kb = sort.calibrated_key(b, mu_fn(b));
                ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            matching.sort_by_key(|(_, c)| sort.key(c));
        }

        let tier_order: Vec<TierId> = matching.iter().map(|(t, _)| *t).collect();

        if tier_order.is_empty() {
            return RoutingPlan::fallback(
                self.overhead_joules,
                format!("coord_router({}): no tier matched, fallback", label),
            );
        }

        RoutingPlan {
            tier_order,
            router_joules: self.overhead_joules,
            reasoning: format!("coord_router({}): {} tier(s) match",
                label, matching.len()),
        }
    }
}

// ============================================================
// Built-in text predicates for the default rule set
// ============================================================

fn is_arithmetic(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    let has_digit = lower.chars().any(|c| c.is_ascii_digit());
    // Plus/star are unambiguous arithmetic operators. Minus and slash
    // appear in dates ("2026-05-14") and paths ("a/b"), so we don't
    // accept them as standalone arithmetic signals. Word-form helps.
    let has_op = lower.contains('+') || lower.contains('*')
              || lower.contains(" plus ") || lower.contains(" minus ")
              || lower.contains(" times ") || lower.contains(" divided ")
              // Allow ` - ` and ` / ` with surrounding spaces (less likely
              // to appear in dates/paths).
              || lower.contains(" - ") || lower.contains(" / ");
    has_digit && has_op
}

fn is_extraction(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("extract ")
        || lower.starts_with("find all ") || lower.starts_with("find every ")
        || lower.starts_with("list all ") || lower.starts_with("get all ")
        || lower.contains("extract all") || lower.contains("find every")
}

fn is_greeting(s: &str) -> bool {
    let trimmed = s.trim().to_ascii_lowercase();
    matches!(trimmed.as_str(),
        "hi" | "hello" | "hey" | "help"
        | "what is your name?" | "what version?" | "who are you?"
        | "what can you do?"
    )
}

/// "what is X", "explain X", "tell me about X" — knowledge-seeking
/// queries that prefer retrieval with citation.
fn is_knowledge_question(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    let knowledge_starts = [
        "what is ", "what are ", "what was ",
        "who is ", "who was ",
        "explain ", "tell me about ", "describe ",
        "define ", "definition of ",
    ];
    knowledge_starts.iter().any(|p| lower.starts_with(p))
        || (lower.starts_with("what ") && lower.contains(" of "))
}

/// "write X to Y", "send X to Y", "deploy X", "execute X" — world-
/// modifying actions. The cascade routes these through I=Bodies
/// tiers only.
fn is_world_action(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    let action_starts = [
        "write ", "send ", "post ", "deploy ", "execute ",
        "delete ", "remove ", "update ", "create ", "make ",
        "actuate ", "move ", "rotate ",
        "schedule ", "cancel ",
    ];
    action_starts.iter().any(|p| lower.starts_with(p))
}

/// "verify X", "confirm X", "is it true that X" — queries that demand
/// checkable answers (V=Citation or V=Full).
fn is_high_trust(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("verify ")
        || lower.starts_with("confirm ")
        || lower.starts_with("is it true ")
        || lower.starts_with("prove ")
        || lower.contains(" cite source")
        || lower.contains(" with citation")
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::{prebuilt, NamedPrimitive};

    fn text(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn full_cascade_coords() -> Vec<(TierId, Coord)> {
        vec![
            (TierId::L0, prebuilt::l0_cache()),
            (TierId::L1(L1Primitive::Execute), prebuilt::l1_execute()),
            (TierId::L1(L1Primitive::Regex), prebuilt::l1_regex()),
            (TierId::L1(L1Primitive::TemplateFill), prebuilt::l1_template()),
            (TierId::L2(L2ModelId(0)), prebuilt::l2_embedder()),
            (TierId::L3(L3ModelId(0)), prebuilt::l3_small_model()),
            (TierId::L4(L4ModelId(0)), prebuilt::l4_frontier_model()),
        ]
    }

    #[test]
    fn arithmetic_rule_dispatches_to_l1_execute() {
        let r = CoordRouter::defaults();
        let plan = r.route_with_coords(&text("47 * 23"), &full_cascade_coords());
        // Plan should include L1::Execute and order it before L2/L3/L4.
        assert!(plan.tier_order.contains(&TierId::L1(L1Primitive::Execute)),
            "arithmetic plan must include L1::Execute");
        // First tier should be L0 (free) or L1::Execute (cheapest L1).
        let first = plan.tier_order[0];
        assert!(matches!(first, TierId::L0 | TierId::L1(_)),
            "first tier should be L0 or L1, got {:?}", first);
        // No L2/L3/L4 in the plan (predicate is Z1+L1_Measure).
        for t in &plan.tier_order {
            assert!(matches!(t, TierId::L0 | TierId::L1(_)),
                "arithmetic plan should not include L2+, got {:?}", t);
        }
    }

    #[test]
    fn extraction_rule_restricts_to_z1_cheap() {
        let r = CoordRouter::defaults();
        let plan = r.route_with_coords(
            &text("extract all emails from foo@bar.com"),
            &full_cascade_coords(),
        );
        // Plan should only contain Z1 tiers at L0/L1.
        // L0, L1::Execute, L1::Regex, L1::Template all match.
        for tid in &plan.tier_order {
            assert!(matches!(tid,
                TierId::L0 | TierId::L1(_)),
                "extraction plan should not include L2/L3/L4, got {:?}", tid);
        }
        assert!(!plan.tier_order.is_empty());
    }

    #[test]
    fn greeting_rule_requires_full_verify() {
        let r = CoordRouter::defaults();
        let plan = r.route_with_coords(&text("hi"), &full_cascade_coords());
        // Verify::Full → L0 cache, L1 tiers (all have Full). Not L2/L3/L4.
        for tid in &plan.tier_order {
            assert!(matches!(tid,
                TierId::L0 | TierId::L1(_)),
                "greeting plan should be full-verify only, got {:?}", tid);
        }
    }

    #[test]
    fn fallback_when_no_rule_matches() {
        let r = CoordRouter::defaults();
        let plan = r.route_with_coords(
            &text("write a sonnet about clouds"),
            &full_cascade_coords(),
        );
        // No rule fires → fallback predicate (any tier) → all tiers
        // sorted by cost.
        assert_eq!(plan.tier_order.len(), full_cascade_coords().len());
        // First should be cheapest = L0.
        assert_eq!(plan.tier_order[0], TierId::L0);
        // Last should be most expensive = L4.
        assert_eq!(*plan.tier_order.last().unwrap(),
            TierId::L4(L4ModelId(0)));
    }

    #[test]
    fn empty_match_returns_fallback_plan() {
        // Predicate so restrictive nothing matches.
        let r = CoordRouter::new()
            .with_rule(CoordRule {
                matches: |_| true,
                predicate: CoordPredicate::new()
                    .requires(NamedPrimitive::SpeculativeDraft), // none of our tiers
                sort: SortStrategy::CostFirst,
                label: "impossible",
            });
        let plan = r.route_with_coords(&text("anything"), &full_cascade_coords());
        assert!(plan.is_fallback(),
            "no matching tier should return fallback plan");
    }

    #[test]
    fn quality_first_sort_prefers_full_verify() {
        let r = CoordRouter::new()
            .with_fallback(
                CoordPredicate::new(),
                SortStrategy::QualityFirst,
            );
        let plan = r.route_with_coords(&text("anything"), &full_cascade_coords());
        // First tier should have V=Full (highest strictness).
        let first_id = plan.tier_order[0];
        let coords = full_cascade_coords();
        let first_coord = coords.iter().find(|(t, _)| *t == first_id).unwrap();
        use crate::coord::Verify;
        assert_eq!(first_coord.1.verify, Verify::Full);
    }
}
