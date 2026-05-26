//! Entailment results (spec §3.4 — deferred to v1, §6.4 entailment).
//!
//! The Diagnose pillar runs entailment between an [`AtomicClaim`]
//! (the hypothesis) and one or more [`RetrievedItem`]s (the
//! premises). v6 §6.4 keeps the v1 mechanism (DeBERTa-v3 NLI) and
//! restricts invocation to at-risk claims (§6.2). v3 §6.7 adds
//! anytime stopping via running e-values; we model the e-value here
//! as an optional companion to the categorical label.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::Metadata;

/// Three-way categorical NLI label. Matches MNLI / ANLI / FEVER
/// label sets so a DeBERTa-v3-NLI head can populate it directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntailmentLabel {
    /// Premise(s) entail the hypothesis.
    Entails,
    /// Premise(s) neither entail nor contradict.
    Neutral,
    /// Premise(s) contradict the hypothesis.
    Contradicts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntailmentResult {
    pub schema_version: String,
    pub result_id: Uuid,
    /// The hypothesis being judged.
    pub claim_id: Uuid,
    /// One or more retrieved item ids serving as premises. A single
    /// item is typical; multiple is allowed for premises that compose
    /// (e.g. "Paris is in France" + "France is in EU").
    pub premise_item_ids: Vec<Uuid>,
    pub label: EntailmentLabel,
    /// Calibrated probability of the chosen label (full softmax over
    /// {Entails, Neutral, Contradicts} sums to 1.0).
    pub label_probabilities: EntailmentProbabilities,
    /// Running e-value for the null hypothesis "claim not entailed"
    /// (v3 §6.7 anytime stopping). When `None`, the verifier ran
    /// fixed-cost rather than evidence-accumulating.
    #[serde(default)]
    pub running_e_value: Option<f64>,
    /// Reasoner / NLI model identifier that produced the judgment.
    pub model_id: String,
    /// Joules attributable to this entailment call. Feeds §8.5
    /// per-query energy accounting.
    pub joules_spent: f64,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Full softmax over the three labels. Carrying all three keeps the
/// upgrade path open: when DeBERTa replaces a prompt-based proxy or
/// vice versa, downstream code that reads only `label` keeps working
/// while code that needs distributional information has it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EntailmentProbabilities {
    pub entails: f64,
    pub neutral: f64,
    pub contradicts: f64,
}

impl EntailmentProbabilities {
    pub fn argmax(&self) -> EntailmentLabel {
        if self.entails >= self.neutral && self.entails >= self.contradicts {
            EntailmentLabel::Entails
        } else if self.contradicts >= self.neutral {
            EntailmentLabel::Contradicts
        } else {
            EntailmentLabel::Neutral
        }
    }

    pub fn sum(&self) -> f64 {
        self.entails + self.neutral + self.contradicts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        let p = EntailmentProbabilities {
            entails: 0.7,
            neutral: 0.2,
            contradicts: 0.1,
        };
        assert_eq!(p.argmax(), EntailmentLabel::Entails);

        let p = EntailmentProbabilities {
            entails: 0.1,
            neutral: 0.2,
            contradicts: 0.7,
        };
        assert_eq!(p.argmax(), EntailmentLabel::Contradicts);
    }

    #[test]
    fn roundtrips_through_json() {
        let r = EntailmentResult {
            schema_version: "2.0".into(),
            result_id: Uuid::new_v4(),
            claim_id: Uuid::new_v4(),
            premise_item_ids: vec![Uuid::new_v4()],
            label: EntailmentLabel::Entails,
            label_probabilities: EntailmentProbabilities {
                entails: 0.92,
                neutral: 0.07,
                contradicts: 0.01,
            },
            running_e_value: Some(12.7),
            model_id: "DeBERTa-v3-large-mnli-fever-anli-ling-wanli".into(),
            joules_spent: 0.04,
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: EntailmentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
