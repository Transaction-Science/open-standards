//! NLI inference trait + result types.
//!
//! The diagnose pillar (spec §6.4) consumes [`NliInference`]. The
//! real implementation backed by a loaded DeBERTa model lands in
//! later phases; this file defines the contract so consumers can be
//! built and tested against fixtures in the meantime.

use jouleclaw_schema::{EntailmentLabel, EntailmentProbabilities};

use crate::config::{NliLabel, NliLabelLayout};

#[derive(Debug, Clone, PartialEq)]
pub struct NliPrediction {
    pub label: NliLabel,
    /// Calibrated probabilities. Sum to 1.0 (within float epsilon).
    pub probabilities: EntailmentProbabilities,
    /// Joules estimated/measured for this inference call. Set to 0.0
    /// if the implementation doesn't track energy.
    pub joules_spent: f64,
}

impl NliPrediction {
    pub fn label_for_schema(&self) -> EntailmentLabel {
        self.label.into()
    }
}

#[derive(Debug)]
pub enum NliInferenceError {
    /// Input exceeded the model's maximum sequence length.
    InputTooLong { actual: usize, max: usize },
    /// Encoding failed (tokenizer or normalizer error).
    Encoding(String),
    /// Forward pass failed (kernel error, dtype mismatch, etc.).
    Forward(String),
}

impl std::fmt::Display for NliInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputTooLong { actual, max } => write!(f, "input {actual} > max {max}"),
            Self::Encoding(s) => write!(f, "encoding: {s}"),
            Self::Forward(s) => write!(f, "forward: {s}"),
        }
    }
}

impl std::error::Error for NliInferenceError {}

/// One NLI inference call. Implementations dispatch (premise,
/// hypothesis) to a model and return label probabilities.
pub trait NliInference: Send + Sync {
    fn predict(&self, premise: &str, hypothesis: &str) -> Result<NliPrediction, NliInferenceError>;

    /// Stable model identifier surfaced in
    /// [`jouleclaw_schema::EntailmentResult::model_id`].
    fn model_id(&self) -> &str;
}

/// Determine the [`NliLabel`] (and its probability vector) from a
/// model's 3-logit output, given the model's label layout.
pub fn predict_from_logits(
    logits: &[f32; 3],
    layout: NliLabelLayout,
) -> NliPrediction {
    let probs = softmax3(logits);
    let argmax_idx = if probs[0] >= probs[1] && probs[0] >= probs[2] {
        0
    } else if probs[1] >= probs[2] {
        1
    } else {
        2
    };
    let label = layout.label_at(argmax_idx);

    // Map probabilities to the schema's canonical {Entails, Neutral,
    // Contradicts} order regardless of the model's internal layout.
    let (p_entails, p_neutral, p_contradicts) = match layout {
        NliLabelLayout::EntailmentNeutralContradiction => (probs[0], probs[1], probs[2]),
        NliLabelLayout::ContradictionNeutralEntailment => (probs[2], probs[1], probs[0]),
    };
    NliPrediction {
        label,
        probabilities: EntailmentProbabilities {
            entails: p_entails as f64,
            neutral: p_neutral as f64,
            contradicts: p_contradicts as f64,
        },
        joules_spent: 0.0,
    }
}

fn softmax3(logits: &[f32; 3]) -> [f32; 3] {
    let max = logits[0].max(logits[1]).max(logits[2]);
    let e0 = (logits[0] - max).exp();
    let e1 = (logits[1] - max).exp();
    let e2 = (logits[2] - max).exp();
    let sum = e0 + e1 + e2;
    [e0 / sum, e1 / sum, e2 / sum]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entailment_layout_argmax_picks_entailment() {
        let logits = [3.0, 0.5, -1.0];
        let p = predict_from_logits(&logits, NliLabelLayout::EntailmentNeutralContradiction);
        assert_eq!(p.label, NliLabel::Entailment);
        assert!(p.probabilities.entails > 0.9);
        assert!(p.probabilities.contradicts < 0.05);
    }

    #[test]
    fn contradiction_layout_remaps_to_schema_order() {
        // Model output: index 0 = contradiction, index 2 = entailment.
        // Strong contradiction signal.
        let logits = [3.0, 0.0, -2.0];
        let p = predict_from_logits(&logits, NliLabelLayout::ContradictionNeutralEntailment);
        assert_eq!(p.label, NliLabel::Contradiction);
        assert!(p.probabilities.contradicts > 0.9);
        assert!(p.probabilities.entails < 0.05);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax3(&[1.0, 2.0, 3.0]);
        assert!((p[0] + p[1] + p[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn probabilities_sum_to_one_in_prediction() {
        let logits = [1.0, 0.5, 2.5];
        let p = predict_from_logits(&logits, NliLabelLayout::EntailmentNeutralContradiction);
        let s = p.probabilities.entails + p.probabilities.neutral + p.probabilities.contradicts;
        assert!((s - 1.0).abs() < 1e-6, "got {s}");
    }
}
