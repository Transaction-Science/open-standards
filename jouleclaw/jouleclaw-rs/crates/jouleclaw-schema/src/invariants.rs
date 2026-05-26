//! Invariant ids (spec §3.9).
//!
//! Architecture-level invariants enumerated as stable id strings.
//! Every emitted [`crate::Answer`] / [`crate::Refusal`] carries an
//! `invariants_verified` list naming which ones were checked. The
//! orchestrator refuses to emit any output where required invariants
//! aren't in that list.

use serde::{Deserialize, Serialize};

/// The thirteen architectural invariants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Invariant {
    /// I1: every claim in the answer has provenance.
    I1EveryClaimHasProvenance,
    /// I2: every asserted claim has at least one entailing source.
    I2AssertedImpliesEntailed,
    /// I3: every inference is labeled and has explicit premises.
    I3InferenceIsLabeled,
    /// I4: authority tier of cited sources meets the query minimum.
    I4AuthorityTierRespected,
    /// I5: freshness requirements are met by cited sources.
    I5FreshnessRespected,
    /// I6: conflicts are surfaced, not silently resolved.
    I6ConflictsSurfaced,
    /// I7: latency does not exceed the hard ceiling.
    I7LatencyBounded,
    /// I8: cost does not exceed the ceiling.
    I8CostBounded,
    /// I9: re-routing iterations are bounded.
    I9RerouteBounded,
    /// I10: refusals are structured, not exceptions.
    I10RefusalIsStructured,
    /// I11: energy does not exceed the ceiling. (v3)
    I11EnergyBounded,
    /// I12: axis-consistency bounded — no answer claims more precision
    /// on any of the seven knowledge axes than its supporting
    /// evidence. (v5)
    I12AxisConsistencyBounded,
    /// I13: epistemic mode declared — every claim labels its
    /// epistemic mode (from_priors vs from_retrieval) and the
    /// attribution is compatible with the claim's KnowledgeAxes. (v6)
    I13EpistemicModeDeclared,
}

impl Invariant {
    /// Stable wire id. Used as the string form in
    /// `invariants_verified` lists.
    pub fn id(&self) -> &'static str {
        match self {
            Self::I1EveryClaimHasProvenance => "I1",
            Self::I2AssertedImpliesEntailed => "I2",
            Self::I3InferenceIsLabeled => "I3",
            Self::I4AuthorityTierRespected => "I4",
            Self::I5FreshnessRespected => "I5",
            Self::I6ConflictsSurfaced => "I6",
            Self::I7LatencyBounded => "I7",
            Self::I8CostBounded => "I8",
            Self::I9RerouteBounded => "I9",
            Self::I10RefusalIsStructured => "I10",
            Self::I11EnergyBounded => "I11",
            Self::I12AxisConsistencyBounded => "I12",
            Self::I13EpistemicModeDeclared => "I13",
        }
    }

    pub fn all() -> &'static [Invariant] {
        &[
            Self::I1EveryClaimHasProvenance,
            Self::I2AssertedImpliesEntailed,
            Self::I3InferenceIsLabeled,
            Self::I4AuthorityTierRespected,
            Self::I5FreshnessRespected,
            Self::I6ConflictsSurfaced,
            Self::I7LatencyBounded,
            Self::I8CostBounded,
            Self::I9RerouteBounded,
            Self::I10RefusalIsStructured,
            Self::I11EnergyBounded,
            Self::I12AxisConsistencyBounded,
            Self::I13EpistemicModeDeclared,
        ]
    }
}

/// Convenience newtype wrapping the list of verified invariant ids
/// carried by every [`crate::Answer`] / [`crate::Refusal`] /
/// [`crate::VerificationReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct InvariantsVerified(pub Vec<String>);

impl InvariantsVerified {
    pub fn contains(&self, i: Invariant) -> bool {
        self.0.iter().any(|s| s == i.id())
    }

    pub fn push(&mut self, i: Invariant) {
        let id = i.id().to_string();
        if !self.0.contains(&id) {
            self.0.push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_stable_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for inv in Invariant::all() {
            assert!(seen.insert(inv.id()), "duplicate id: {}", inv.id());
        }
        assert_eq!(Invariant::all().len(), 13);
    }

    #[test]
    fn verified_set_dedupes_on_push() {
        let mut v = InvariantsVerified::default();
        v.push(Invariant::I1EveryClaimHasProvenance);
        v.push(Invariant::I1EveryClaimHasProvenance);
        v.push(Invariant::I13EpistemicModeDeclared);
        assert_eq!(v.0.len(), 2);
        assert!(v.contains(Invariant::I1EveryClaimHasProvenance));
        assert!(v.contains(Invariant::I13EpistemicModeDeclared));
        assert!(!v.contains(Invariant::I2AssertedImpliesEntailed));
    }
}
