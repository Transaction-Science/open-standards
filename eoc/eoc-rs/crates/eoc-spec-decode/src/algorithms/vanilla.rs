//! Classic speculative decoding (Leviathan, Kalman & Matias 2022).
//!
//! The original paper presents both a greedy and a probabilistic
//! acceptance rule. This module implements the *greedy* rule:
//!
//! > Accept the draft's proposed token at position `i` iff it equals
//! > `argmax_t p_target(t | prefix ++ proposed[..i])`.
//!
//! The probabilistic variant lives in
//! [`crate::algorithms::sps_with_temperature`].

use crate::algorithms::AcceptanceDecision;
use crate::draft::TokenId;
use crate::error::{SpecDecodeError, SpecDecodeResult};

/// Vanilla greedy speculative decoder.
#[derive(Debug, Clone, Copy)]
pub struct VanillaSpeculative {
    /// Number of tokens the orchestrator asks the draft to propose per
    /// round. Typical values: 4–8.
    pub k: usize,
}

impl VanillaSpeculative {
    /// Construct with a draft block size.
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1) }
    }

    /// Apply the greedy acceptance rule at one position.
    pub fn accept(
        &self,
        proposed: TokenId,
        _draft_logits: &[f32],
        target_logits: &[f32],
    ) -> SpecDecodeResult<AcceptanceDecision> {
        if target_logits.is_empty() {
            return Err(SpecDecodeError::EmptyVerification);
        }
        let argmax = argmax(target_logits)?;
        Ok(if argmax == proposed {
            AcceptanceDecision::Accept
        } else {
            AcceptanceDecision::Reject
        })
    }
}

fn argmax(logits: &[f32]) -> SpecDecodeResult<TokenId> {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            return Err(SpecDecodeError::Sampling("NaN in target logits".into()));
        }
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    Ok(best as TokenId)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_when_draft_matches_argmax() {
        let v = VanillaSpeculative::new(4);
        let target = vec![0.0, 10.0, 0.0, 0.0];
        assert_eq!(
            v.accept(1, &[], &target).expect("non-empty"),
            AcceptanceDecision::Accept
        );
    }

    #[test]
    fn rejects_when_draft_disagrees() {
        let v = VanillaSpeculative::new(4);
        let target = vec![0.0, 10.0, 0.0, 0.0];
        assert_eq!(
            v.accept(2, &[], &target).expect("non-empty"),
            AcceptanceDecision::Reject
        );
    }

    #[test]
    fn rejects_empty_target() {
        let v = VanillaSpeculative::new(4);
        assert!(v.accept(0, &[], &[]).is_err());
    }

    #[test]
    fn k_minimum_is_one() {
        assert_eq!(VanillaSpeculative::new(0).k, 1);
    }
}
