//! L1 — the lawful-primitive tier.
//!
//! L1 is the *deterministic* tier of the cascade. A "lawful primitive"
//! is a piece of pure compute that resolves a query without invoking
//! any model — arithmetic, regex extraction, unit conversion, finite
//! state walks, GCD / LCM / prime factorization, date arithmetic,
//! base conversion, etc.
//!
//! ## What lives here
//!
//! The trait surface only. JouleClaw deliberately does NOT ship a
//! lexicon of lawful primitives — that would either be incomplete
//! (and surprise consumers) or expand the standard's surface into
//! library territory.
//!
//! Consumers register their own primitives at runtime through
//! [`LawfulRegistry`]. Pattern-lang's `pattern-core` + `cortex-ir`
//! synthesizer is the largest known consumer; it plugs in 688+
//! verified primitives at startup. Smaller consumers ship a handful
//! of arithmetic helpers.
//!
//! Runtimes that ship no lawful library at all just construct an
//! empty registry — the cascade walks past L1 straight to L2.
//!
//! ## Why the trait is intentionally tiny
//!
//! Every primitive declares three things and nothing more:
//!
//! - a stable [`LawfulPrimitive::id`] for receipt accounting
//! - a [`LawfulPrimitive::try_resolve`] that returns `Some(answer)` or
//!   `None` (try-next-primitive)
//! - a [`LawfulPrimitive::declared_cost_uj`] for the joule budget
//!
//! That's the whole contract. No metadata, no schema, no version
//! handshake. If a primitive needs richer dispatch (typed inputs,
//! typed outputs, parser combinators) the consumer's lexicon
//! provides it; the JouleClaw L1 surface stays minimal.

use std::sync::Arc;

/// One deterministic primitive that can resolve a query class at the
/// L1 nanojoule cost class.
///
/// Implementations MUST be pure (same input → same output, every
/// time) and MUST NOT invoke any stochastic compute.
pub trait LawfulPrimitive: Send + Sync {
    /// Stable identifier — appears verbatim in cascade receipts.
    /// Format convention: `lawful:<library>:<primitive>` — for
    /// example `"lawful:pattern-core:gcd"`, `"lawful:units:f-to-c"`.
    fn id(&self) -> &str;

    /// Attempt to resolve the query. Returns `Some(answer)` when the
    /// primitive recognises and handles the query; `None` otherwise.
    ///
    /// Implementations MUST run in O(constant) wall-clock time
    /// relative to the cost class. A primitive that takes 100 ms is
    /// not a lawful primitive — it belongs in L2 or L3.
    fn try_resolve(&self, query: &str) -> Option<String>;

    /// Microjoule cost the primitive declares it will consume.
    /// The cascade accounts this against the budget on a successful
    /// `try_resolve` regardless of whether a real energy counter
    /// reports otherwise — this is the *declared* cost; the receipt's
    /// `tools_touched` entry carries the measured one.
    fn declared_cost_uj(&self) -> u64 {
        // The L1 nanojoule cost class. Override for primitives that
        // do real work (e.g. a regex extraction with a 50KB input
        // string MUST declare a higher cost).
        100
    }
}

/// Registry of lawful primitives. Walked in insertion order; the
/// first `Some(answer)` wins.
#[derive(Default, Clone)]
pub struct LawfulRegistry {
    primitives: Vec<Arc<dyn LawfulPrimitive>>,
}

impl std::fmt::Debug for LawfulRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LawfulRegistry")
            .field("primitive_count", &self.primitives.len())
            .finish()
    }
}

impl LawfulRegistry {
    /// Empty registry — the cascade walks past L1 straight to L2.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a primitive. Insertion order is preserved; earlier-
    /// registered primitives are probed first.
    pub fn register(mut self, primitive: Arc<dyn LawfulPrimitive>) -> Self {
        self.primitives.push(primitive);
        self
    }

    /// Number of registered primitives.
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Probe the registered primitives in order; return the first
    /// `Some(answer)` along with the primitive's id and declared cost.
    /// Returns `None` if no primitive recognises the query.
    pub fn try_resolve(&self, query: &str) -> Option<LawfulHit> {
        for p in &self.primitives {
            if let Some(answer) = p.try_resolve(query) {
                return Some(LawfulHit {
                    primitive_id: p.id().to_string(),
                    answer,
                    declared_cost_uj: p.declared_cost_uj(),
                });
            }
        }
        None
    }
}

/// A successful L1 resolution. Returned by [`LawfulRegistry::try_resolve`].
#[derive(Debug, Clone)]
pub struct LawfulHit {
    /// Which primitive answered the query.
    pub primitive_id: String,
    /// The deterministic answer.
    pub answer: String,
    /// Declared microjoule cost for receipt accounting.
    pub declared_cost_uj: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy primitive that resolves "what is gcd(a, b)?" for tiny inputs.
    struct ToyGcd;
    impl LawfulPrimitive for ToyGcd {
        fn id(&self) -> &str { "lawful:test:gcd" }
        fn try_resolve(&self, query: &str) -> Option<String> {
            let q = query.trim();
            let q = q.strip_prefix("gcd(")?;
            let q = q.strip_suffix(")")?;
            let (a, b) = q.split_once(',')?;
            let a: u32 = a.trim().parse().ok()?;
            let b: u32 = b.trim().parse().ok()?;
            // textbook Euclid
            let (mut x, mut y) = (a, b);
            while y != 0 { let t = y; y = x % y; x = t; }
            Some(x.to_string())
        }
    }

    /// A primitive that handles a small fixed unit conversion.
    struct ToyFahrenheitToCelsius;
    impl LawfulPrimitive for ToyFahrenheitToCelsius {
        fn id(&self) -> &str { "lawful:test:f-to-c" }
        fn try_resolve(&self, query: &str) -> Option<String> {
            let q = query.trim().strip_prefix("f-to-c ")?;
            let f: f64 = q.trim().parse().ok()?;
            let c = (f - 32.0) * 5.0 / 9.0;
            Some(format!("{c:.2}"))
        }
        fn declared_cost_uj(&self) -> u64 { 50 }
    }

    #[test]
    fn empty_registry_resolves_nothing() {
        let r = LawfulRegistry::new();
        assert!(r.is_empty());
        assert!(r.try_resolve("gcd(12, 8)").is_none());
    }

    #[test]
    fn registry_probes_in_insertion_order() {
        let r = LawfulRegistry::new()
            .register(Arc::new(ToyGcd))
            .register(Arc::new(ToyFahrenheitToCelsius));
        assert_eq!(r.len(), 2);
        let hit = r.try_resolve("gcd(12, 8)").expect("hit");
        assert_eq!(hit.primitive_id, "lawful:test:gcd");
        assert_eq!(hit.answer, "4");
        assert_eq!(hit.declared_cost_uj, 100);
    }

    #[test]
    fn first_match_wins() {
        // Custom primitive that pretends it can answer GCD but returns
        // the wrong answer — registered FIRST. ToyGcd registered second.
        // Registry walks insertion order, so the wrong primitive wins.
        struct WrongGcd;
        impl LawfulPrimitive for WrongGcd {
            fn id(&self) -> &str { "lawful:test:wrong-gcd" }
            fn try_resolve(&self, q: &str) -> Option<String> {
                if q.starts_with("gcd(") { Some("999".into()) } else { None }
            }
        }
        let r = LawfulRegistry::new()
            .register(Arc::new(WrongGcd))
            .register(Arc::new(ToyGcd));
        assert_eq!(r.try_resolve("gcd(12, 8)").unwrap().answer, "999");
    }

    #[test]
    fn unknown_query_returns_none() {
        let r = LawfulRegistry::new().register(Arc::new(ToyGcd));
        assert!(r.try_resolve("write a haiku about owls").is_none());
    }

    #[test]
    fn custom_cost_is_carried_through() {
        let r = LawfulRegistry::new().register(Arc::new(ToyFahrenheitToCelsius));
        let hit = r.try_resolve("f-to-c 212").expect("hit");
        assert_eq!(hit.declared_cost_uj, 50);
        assert_eq!(hit.answer, "100.00");
    }
}
