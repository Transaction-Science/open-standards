//! SelfCheckGPT-style hallucination guard.
//!
//! Manakul et al. 2023 ("SelfCheckGPT: Zero-Resource Black-Box
//! Hallucination Detection for Generative Large Language Models",
//! arXiv:2303.08896) sample several stochastic generations of the
//! same answer and compute pairwise consistency. Low consistency
//! signals hallucination because a confident, knowledge-grounded
//! answer is reproducible while a hallucinated one drifts.
//!
//! The guard exposed here takes `n` candidate answers and rejects
//! the primary if the mean pairwise consistency falls below the
//! configured threshold.

use serde::{Deserialize, Serialize};

use crate::error::{RagError, RagResult};

/// Outcome of a consistency check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyVerdict {
    /// Mean pairwise consistency in `[0, 1]`.
    pub mean_consistency: f32,
    /// Whether the verdict passed the configured threshold.
    pub accepted: bool,
    /// Number of samples scored.
    pub n_samples: usize,
}

/// Configuration + implementation of the SelfCheckGPT-style guard.
pub struct SelfCheckGuard {
    /// Minimum mean pairwise consistency to accept the answer.
    pub threshold: f32,
}

impl SelfCheckGuard {
    /// Construct with `threshold` (default 0.5).
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Score the primary answer against `samples`. Returns a verdict.
    pub fn verdict(&self, primary: &str, samples: &[String]) -> ConsistencyVerdict {
        if samples.is_empty() {
            return ConsistencyVerdict {
                mean_consistency: 1.0,
                accepted: true,
                n_samples: 0,
            };
        }
        let consistencies: Vec<f32> = samples
            .iter()
            .map(|s| jaccard_chars(primary, s))
            .collect();
        let mean = consistencies.iter().sum::<f32>() / consistencies.len() as f32;
        ConsistencyVerdict {
            mean_consistency: mean,
            accepted: mean >= self.threshold,
            n_samples: samples.len(),
        }
    }

    /// Apply the guard. Returns `Ok` if the answer passes, otherwise a
    /// [`RagError::GuardRejected`].
    pub fn enforce(&self, primary: &str, samples: &[String]) -> RagResult<ConsistencyVerdict> {
        let v = self.verdict(primary, samples);
        if v.accepted {
            Ok(v)
        } else {
            Err(RagError::GuardRejected(format!(
                "mean consistency {:.3} < threshold {:.3}",
                v.mean_consistency, self.threshold
            )))
        }
    }
}

fn jaccard_chars(a: &str, b: &str) -> f32 {
    let ta = tokens(a);
    let tb = tokens(b);
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.iter().filter(|t| tb.contains(*t)).count() as f32;
    let union = {
        let mut u: Vec<&String> = ta.iter().collect();
        for t in &tb {
            if !u.contains(&t) {
                u.push(t);
            }
        }
        u.len() as f32
    };
    if union == 0.0 { 0.0 } else { inter / union }
}

fn tokens(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            for low in ch.to_lowercase() {
                cur.push(low);
            }
        } else if !cur.is_empty() {
            if !out.contains(&cur) {
                out.push(cur.clone());
            }
            cur.clear();
        }
    }
    if !cur.is_empty() && !out.contains(&cur) {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_consistency_passes() {
        let g = SelfCheckGuard::new(0.5);
        let primary = "The capital of France is Paris.";
        let samples = vec![
            "Paris is the capital of France.".to_string(),
            "France's capital is Paris.".to_string(),
        ];
        let v = g.verdict(primary, &samples);
        assert!(v.accepted, "verdict={:?}", v);
    }

    #[test]
    fn low_consistency_rejected() {
        let g = SelfCheckGuard::new(0.5);
        let primary = "The capital of France is Paris.";
        let samples = vec![
            "Bananas grow on trees in tropical climates.".to_string(),
            "Quantum chromodynamics studies the strong force.".to_string(),
        ];
        assert!(g.enforce(primary, &samples).is_err());
    }

    #[test]
    fn empty_samples_passes() {
        let g = SelfCheckGuard::new(0.5);
        let v = g.verdict("anything", &[]);
        assert!(v.accepted);
        assert_eq!(v.n_samples, 0);
    }
}
