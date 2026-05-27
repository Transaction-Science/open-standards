//! The [`OutputVerifier`] trait and its [`VerifyResult`] verdict.
//!
//! Every verifier in JouleClaw — regex, schema, hash, hand-rolled —
//! implements this single trait. The trait is deliberately small:
//! one name, one verdict, one declared cost. That is everything the
//! receipt ledger needs.

/// A deterministic check that disposes of one open-ended output.
///
/// Implementations MUST be pure: the same `output` MUST always yield
/// the same [`VerifyResult`]. They MUST also be honest about their
/// cost — see the crate-level docs on the verifier-honesty
/// assumption.
///
/// Implementations are required to be `Send + Sync` so a chain can
/// be held in a long-lived runtime and called from multiple tasks.
pub trait OutputVerifier: Send + Sync {
    /// Stable identifier used in receipts. Convention:
    /// `verify:<tag>`, e.g. `verify:digits`, `verify:schema/openai`.
    fn name(&self) -> &str;

    /// Run the check. MUST be pure with respect to `output`.
    fn verify(&self, output: &[u8]) -> VerifyResult;

    /// Microjoules this verifier declares it costs to run once.
    /// Used to account the verifier as a `ToolTouch` in the cascade
    /// receipt. Implementations should round up.
    fn declared_cost_uj(&self) -> u64;
}

/// The verdict from a single [`OutputVerifier`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyResult {
    /// The output satisfies the check.
    Pass,
    /// The output fails the check. `reason` is a short human-readable
    /// string suitable for inclusion in the receipt's failure record
    /// and downstream retry-tier decisions.
    Fail {
        /// Short explanation, e.g. `"output not utf-8"` or
        /// `"missing required field 'usage'"`.
        reason: String,
    },
}

impl VerifyResult {
    /// `true` iff this is a [`VerifyResult::Pass`].
    pub fn is_pass(&self) -> bool {
        matches!(self, VerifyResult::Pass)
    }

    /// Convenience constructor for a failure verdict.
    pub fn fail(reason: impl Into<String>) -> Self {
        VerifyResult::Fail {
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_is_pass() {
        assert!(VerifyResult::Pass.is_pass());
    }

    #[test]
    fn fail_is_not_pass() {
        let f = VerifyResult::fail("nope");
        assert!(!f.is_pass());
        match f {
            VerifyResult::Fail { reason } => assert_eq!(reason, "nope"),
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }
}
