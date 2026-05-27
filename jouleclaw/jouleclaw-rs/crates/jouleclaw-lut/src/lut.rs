//! The [`Lut`] struct — hash-keyed exact-match lookup table.
//!
//! Pre-baked answers register here. Lookups normalise + hash + probe a
//! single `HashMap`. No eviction. No background work.

use std::collections::HashMap;

use chrono::Utc;

use crate::normalize::normalize;
use crate::types::{LutEntry, LutHit, LutKey};

/// Default declared cost for entries registered via [`Lut::register`]
/// without an explicit cost: **1 nJ** expressed in microjoules.
///
/// Stored as 1 (i.e. 1 µJ) because the entry-side declared cost field
/// is `u64` µJ and we can't represent sub-µJ at the entry surface. The
/// [`Tier`](crate::tier) impl reports the true sub-nanojoule energy
/// model independently (it does not divide the declared cost — see
/// `tier.rs` for the math).
pub const DEFAULT_DECLARED_COST_UJ: u64 = 1;

/// Literal lookup table — the pre-cascade resolver.
///
/// Construction:
/// ```
/// use jouleclaw_lut::Lut;
/// let mut lut = Lut::new();
/// lut.register("gcd 12 8", "4", "lawful:gcd");
/// assert!(lut.try_lookup("  GCD 12 8  ").is_some());
/// ```
#[derive(Debug, Default, Clone)]
pub struct Lut {
    entries: HashMap<LutKey, LutEntry>,
}

impl Lut {
    /// Construct an empty LUT.
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Register a pre-baked answer for `input`. Subsequent calls with
    /// the same normalised input OVERWRITE the entry. Uses
    /// [`DEFAULT_DECLARED_COST_UJ`] as the declared cost.
    pub fn register(
        &mut self,
        input: &str,
        output: impl Into<Vec<u8>>,
        source_tag: impl Into<String>,
    ) {
        self.register_with_cost(input, output, DEFAULT_DECLARED_COST_UJ, source_tag);
    }

    /// Register a pre-baked answer with an explicit declared cost in
    /// microjoules. Overwrites any existing entry for the same
    /// normalised input.
    pub fn register_with_cost(
        &mut self,
        input: &str,
        output: impl Into<Vec<u8>>,
        cost_uj: u64,
        source_tag: impl Into<String>,
    ) {
        let normalized = normalize(input);
        let key = LutKey::from_normalized(&normalized);
        let entry = LutEntry {
            output: output.into(),
            declared_cost_uj: cost_uj,
            source_tag: source_tag.into(),
            registered_at: Utc::now(),
        };
        self.entries.insert(key, entry);
    }

    /// Look up a normalised match for `input`. Returns the stored
    /// payload or `None`.
    pub fn try_lookup(&self, input: &str) -> Option<LutHit> {
        let normalized = normalize(input);
        let key = LutKey::from_normalized(&normalized);
        self.entries.get(&key).map(|e| LutHit {
            output: e.output.clone(),
            declared_cost_uj: e.declared_cost_uj,
            source_tag: e.source_tag.clone(),
        })
    }

    /// Compute the [`LutKey`] for `input` (normalised + hashed) without
    /// inserting or looking up anything. Mostly useful for tests and
    /// observability tooling.
    pub fn key_for(input: &str) -> LutKey {
        let normalized = normalize(input);
        LutKey::from_normalized(&normalized)
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any entries are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate registered `(key, entry)` pairs. Iteration order is the
    /// `HashMap` order and therefore not stable.
    pub fn iter(&self) -> impl Iterator<Item = (&LutKey, &LutEntry)> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_lookup_round_trips() {
        let mut lut = Lut::new();
        lut.register("gcd 12 8", "4", "lawful:gcd");
        let hit = lut.try_lookup("gcd 12 8").expect("registered key must hit");
        assert_eq!(hit.output, b"4");
        assert_eq!(hit.source_tag, "lawful:gcd");
        assert_eq!(hit.declared_cost_uj, DEFAULT_DECLARED_COST_UJ);
    }

    #[test]
    fn lookup_of_unregistered_returns_none() {
        let lut = Lut::new();
        assert!(lut.try_lookup("gcd 12 8").is_none());

        let mut lut = Lut::new();
        lut.register("hello", "hi", "test");
        assert!(lut.try_lookup("goodbye").is_none());
    }

    #[test]
    fn duplicate_register_overwrites() {
        let mut lut = Lut::new();
        lut.register("k", "v1", "src1");
        assert_eq!(lut.len(), 1);
        lut.register("k", "v2", "src2");
        assert_eq!(lut.len(), 1);
        let hit = lut.try_lookup("k").expect("hit");
        assert_eq!(hit.output, b"v2");
        assert_eq!(hit.source_tag, "src2");
    }

    #[test]
    fn normalize_collisions_hit_same_entry() {
        let mut lut = Lut::new();
        lut.register("  GCD 12 8  ", "4", "lawful:gcd");
        let hit = lut
            .try_lookup("gcd 12 8")
            .expect("normalised forms must collide");
        assert_eq!(hit.output, b"4");
        // Whitespace variants too.
        let hit2 = lut
            .try_lookup("\tgcd\n12   8")
            .expect("whitespace variant must collide");
        assert_eq!(hit2.output, b"4");
    }

    #[test]
    fn register_with_explicit_cost() {
        let mut lut = Lut::new();
        lut.register_with_cost("k", "v", 42, "src");
        let hit = lut.try_lookup("k").expect("hit");
        assert_eq!(hit.declared_cost_uj, 42);
    }

    #[test]
    fn len_and_is_empty_track_inserts() {
        let mut lut = Lut::new();
        assert!(lut.is_empty());
        assert_eq!(lut.len(), 0);
        lut.register("a", "1", "src");
        lut.register("b", "2", "src");
        assert!(!lut.is_empty());
        assert_eq!(lut.len(), 2);
    }

    #[test]
    fn iter_yields_all_entries() {
        let mut lut = Lut::new();
        lut.register("a", "1", "src");
        lut.register("b", "2", "src");
        let collected: Vec<&LutEntry> = lut.iter().map(|(_, e)| e).collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn key_for_is_deterministic() {
        assert_eq!(Lut::key_for("gcd 12 8"), Lut::key_for("  GCD 12 8  "));
        assert_ne!(Lut::key_for("gcd 12 8"), Lut::key_for("gcd 12 9"));
    }
}
