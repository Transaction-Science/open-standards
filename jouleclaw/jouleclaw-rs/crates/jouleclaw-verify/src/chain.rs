//! Ordered, fail-fast composition of [`OutputVerifier`]s.
//!
//! A [`VerifierChain`] applies each verifier in turn. The first
//! refusal wins — subsequent verifiers are not run, and the
//! `prior_touches` collected so far are returned alongside the
//! failure so the cascade receipt can still record the work that
//! *was* done.
//!
//! On a fully-passing chain every verifier emits a
//! [`jouleclaw_prov::ToolTouch`] row in order:
//! - `tool_id` = `verifier.name()`
//! - `joules_uj` = `verifier.declared_cost_uj()`
//! - `energy_provenance` = [`Provenance::Estimator`] (verifier
//!   costs are declared, not hardware-measured — receipts must
//!   reflect that honestly).

use crate::verifier::{OutputVerifier, VerifyResult};
use jouleclaw_energy::Provenance;
use jouleclaw_prov::ToolTouch;

/// Outcome of running an output through a [`VerifierChain`].
#[derive(Debug, Clone)]
pub enum ChainResult {
    /// Every verifier in the chain passed. `tool_touches` is in
    /// chain order, one per verifier.
    Passed {
        /// One `ToolTouch` per verifier that ran, in chain order.
        tool_touches: Vec<ToolTouch>,
    },
    /// A verifier refused the output. `prior_touches` holds the
    /// `ToolTouch` rows for verifiers that *did* pass before the
    /// refusal — they should still be included in the receipt.
    FailedAt {
        /// `name()` of the refusing verifier.
        name: String,
        /// Refusal reason carried up from the verifier.
        reason: String,
        /// `ToolTouch` rows for verifiers that passed before the
        /// failure, in chain order.
        prior_touches: Vec<ToolTouch>,
    },
}

impl ChainResult {
    /// `true` iff this is a [`ChainResult::Passed`].
    pub fn is_passed(&self) -> bool {
        matches!(self, ChainResult::Passed { .. })
    }
}

/// Holds an ordered list of verifiers and runs them fail-fast.
///
/// Use [`VerifierChain::new`] + [`VerifierChain::with`] in builder
/// style:
///
/// ```ignore
/// use jouleclaw_verify::{VerifierChain, RegexVerifier};
/// let chain = VerifierChain::new()
///     .with(Box::new(RegexVerifier::must_match("^\\{").unwrap()))
///     .with(Box::new(RegexVerifier::must_not_match("password=").unwrap()));
/// let result = chain.verify(b"{\"k\":1}");
/// assert!(result.is_passed());
/// ```
pub struct VerifierChain {
    verifiers: Vec<Box<dyn OutputVerifier>>,
}

impl VerifierChain {
    /// Empty chain. An empty chain trivially passes — useful as a
    /// no-op gate when the runtime has chosen not to install any
    /// constraints.
    pub fn new() -> Self {
        Self {
            verifiers: Vec::new(),
        }
    }

    /// Append a verifier to the chain. Builder-style.
    pub fn with(mut self, verifier: Box<dyn OutputVerifier>) -> Self {
        self.verifiers.push(verifier);
        self
    }

    /// Number of verifiers in the chain.
    pub fn len(&self) -> usize {
        self.verifiers.len()
    }

    /// `true` iff the chain has no verifiers.
    pub fn is_empty(&self) -> bool {
        self.verifiers.is_empty()
    }

    /// Run `output` through every verifier in order, stopping at
    /// the first refusal.
    pub fn verify(&self, output: &[u8]) -> ChainResult {
        let mut touches: Vec<ToolTouch> = Vec::with_capacity(self.verifiers.len());
        for v in &self.verifiers {
            let touch = ToolTouch {
                tool_id: v.name().to_string(),
                joules_uj: v.declared_cost_uj(),
                // Verifier cost is a *declared* number, not a
                // hardware-shunt reading. Receipt honesty requires
                // we floor to Estimator here.
                energy_provenance: Provenance::Estimator,
            };
            match v.verify(output) {
                VerifyResult::Pass => {
                    touches.push(touch);
                }
                VerifyResult::Fail { reason } => {
                    return ChainResult::FailedAt {
                        name: v.name().to_string(),
                        reason,
                        prior_touches: touches,
                    };
                }
            }
        }
        ChainResult::Passed {
            tool_touches: touches,
        }
    }
}

impl Default for VerifierChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_verifier::BlakeHashVerifier;
    use crate::regex_verifier::RegexVerifier;
    use crate::verifier::OutputVerifier;

    #[test]
    fn empty_chain_passes() {
        let chain = VerifierChain::new();
        assert!(chain.is_empty());
        let r = chain.verify(b"anything");
        match r {
            ChainResult::Passed { tool_touches } => assert!(tool_touches.is_empty()),
            ChainResult::FailedAt { .. } => panic!("expected Passed"),
        }
    }

    #[test]
    fn all_pass_accumulates_tool_touches_in_order() {
        let payload = b"12345";
        let hex = blake3::hash(payload).to_hex().to_string();
        let chain = VerifierChain::new()
            .with(Box::new(
                RegexVerifier::must_match("^[0-9]+$")
                    .expect("compile")
                    .named("verify:digits")
                    .with_cost_uj(50),
            ))
            .with(Box::new(
                BlakeHashVerifier::new(hex)
                    .expect("construct")
                    .named("verify:hash/expected")
                    .with_cost_uj(20),
            ));
        assert_eq!(chain.len(), 2);
        let r = chain.verify(payload);
        match r {
            ChainResult::Passed { tool_touches } => {
                assert_eq!(tool_touches.len(), 2);
                assert_eq!(tool_touches[0].tool_id, "verify:digits");
                assert_eq!(tool_touches[0].joules_uj, 50);
                assert_eq!(tool_touches[0].energy_provenance, Provenance::Estimator);
                assert_eq!(tool_touches[1].tool_id, "verify:hash/expected");
                assert_eq!(tool_touches[1].joules_uj, 20);
                assert_eq!(tool_touches[1].energy_provenance, Provenance::Estimator);
            }
            ChainResult::FailedAt { .. } => panic!("expected Passed"),
        }
    }

    #[test]
    fn first_fail_short_circuits_with_prior_touches() {
        // Chain: [pass, FAIL, would-also-fail]. Expect: stop at the
        // second verifier with prior_touches containing exactly the
        // first verifier's touch.
        let chain = VerifierChain::new()
            .with(Box::new(
                RegexVerifier::must_match(".*")
                    .expect("compile")
                    .named("verify:any")
                    .with_cost_uj(11),
            ))
            .with(Box::new(
                RegexVerifier::must_match("^DOES NOT MATCH$")
                    .expect("compile")
                    .named("verify:strict")
                    .with_cost_uj(22),
            ))
            .with(Box::new(
                // Would also fail — must NOT be touched.
                RegexVerifier::must_match("^never reached$")
                    .expect("compile")
                    .named("verify:never")
                    .with_cost_uj(33),
            ));
        let r = chain.verify(b"hello world");
        match r {
            ChainResult::FailedAt {
                name,
                reason,
                prior_touches,
            } => {
                assert_eq!(name, "verify:strict");
                assert!(reason.contains("did not match"));
                assert_eq!(prior_touches.len(), 1);
                assert_eq!(prior_touches[0].tool_id, "verify:any");
                assert_eq!(prior_touches[0].joules_uj, 11);
            }
            ChainResult::Passed { .. } => panic!("expected FailedAt"),
        }
    }

    #[test]
    fn fail_fast_preserves_order() {
        // Five verifiers — all pass except the fourth. The receipt
        // must show exactly three prior touches, in the original
        // order.
        struct Counting {
            name: String,
            pass: bool,
            cost: u64,
        }
        impl OutputVerifier for Counting {
            fn name(&self) -> &str {
                &self.name
            }
            fn verify(&self, _output: &[u8]) -> VerifyResult {
                if self.pass {
                    VerifyResult::Pass
                } else {
                    VerifyResult::fail("nope")
                }
            }
            fn declared_cost_uj(&self) -> u64 {
                self.cost
            }
        }

        let chain = VerifierChain::new()
            .with(Box::new(Counting {
                name: "verify:a".into(),
                pass: true,
                cost: 1,
            }))
            .with(Box::new(Counting {
                name: "verify:b".into(),
                pass: true,
                cost: 2,
            }))
            .with(Box::new(Counting {
                name: "verify:c".into(),
                pass: true,
                cost: 3,
            }))
            .with(Box::new(Counting {
                name: "verify:d".into(),
                pass: false,
                cost: 4,
            }))
            .with(Box::new(Counting {
                name: "verify:e".into(),
                pass: true,
                cost: 5,
            }));
        let r = chain.verify(b"x");
        match r {
            ChainResult::FailedAt {
                name,
                prior_touches,
                ..
            } => {
                assert_eq!(name, "verify:d");
                let ids: Vec<&str> = prior_touches
                    .iter()
                    .map(|t| t.tool_id.as_str())
                    .collect();
                assert_eq!(ids, vec!["verify:a", "verify:b", "verify:c"]);
                let costs: Vec<u64> = prior_touches.iter().map(|t| t.joules_uj).collect();
                assert_eq!(costs, vec![1, 2, 3]);
            }
            ChainResult::Passed { .. } => panic!("expected FailedAt"),
        }
    }
}
