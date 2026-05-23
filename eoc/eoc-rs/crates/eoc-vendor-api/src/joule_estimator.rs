//! Joule estimation from token counts.
//!
//! Commercial vendors do not expose per-call hardware energy counters,
//! so this estimator multiplies reported input/output token counts by
//! per-model energy coefficients curated in
//! [`data/model_energy_profiles.json`](../../data/model_energy_profiles.json).
//!
//! The resulting [`JouleCost`] is always tagged
//! [`JouleSource::Estimated`](eoc_core::JouleSource::Estimated).

use std::collections::BTreeMap;

use eoc_core::{JouleCost, JouleSource};
use serde::Deserialize;

/// Per-model energy profile.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEnergyProfile {
    /// Joules consumed per input (prompt) token.
    pub joules_per_input_token: f64,
    /// Joules consumed per output (completion) token.
    pub joules_per_output_token: f64,
    /// Source citation for the coefficients (HF Energy Score, vendor
    /// disclosure, or estimator-derivation note).
    pub source: String,
}

/// Trait for plugging in a custom joule estimator.
pub trait JouleEstimator: Send + Sync {
    /// Estimate the joule cost of an inference given input/output token
    /// counts and the model name (vendor-qualified, e.g. `claude-3-5-sonnet`).
    fn estimate(&self, input_tokens: u32, output_tokens: u32, model: &str) -> JouleCost;
}

#[derive(Debug, Deserialize)]
struct ProfileFile {
    profiles: BTreeMap<String, ModelEnergyProfile>,
}

/// Default estimator backed by the embedded coefficient table.
#[derive(Debug, Clone)]
pub struct DefaultEstimator {
    /// model-name → profile.
    pub table: BTreeMap<String, ModelEnergyProfile>,
    /// Fallback joules per input token when a model is unknown.
    pub fallback_input: f64,
    /// Fallback joules per output token when a model is unknown.
    pub fallback_output: f64,
}

const EMBEDDED_PROFILES: &str = include_str!("../data/model_energy_profiles.json");

impl DefaultEstimator {
    /// Construct an estimator from the embedded profile table. Falls
    /// back to mid-range coefficients (0.10 J input, 0.50 J output —
    /// roughly claude-3.5-sonnet) when a model is unknown.
    pub fn builtin() -> Self {
        // Parsing the embedded JSON cannot reasonably fail; if it does
        // the crate's own data file is malformed, which is a build-time
        // bug, not a runtime concern. We fall back to an empty table.
        let table = serde_json::from_str::<ProfileFile>(EMBEDDED_PROFILES)
            .map(|p| p.profiles)
            .unwrap_or_default();
        Self {
            table,
            fallback_input: 0.10,
            fallback_output: 0.50,
        }
    }

    /// Construct an estimator from a custom JSON payload (same schema as
    /// the embedded file).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let parsed: ProfileFile = serde_json::from_str(json)?;
        Ok(Self {
            table: parsed.profiles,
            fallback_input: 0.10,
            fallback_output: 0.50,
        })
    }

    /// Override the fallback coefficients (consumes `self`). Useful for
    /// e.g. the Groq backend, whose LPU has dramatically lower per-token
    /// energy than a generic GPU.
    pub fn with_fallback(mut self, input: f64, output: f64) -> Self {
        self.fallback_input = input;
        self.fallback_output = output;
        self
    }

    /// Insert or overwrite a model profile (consumes `self`).
    pub fn with_profile(mut self, model: impl Into<String>, profile: ModelEnergyProfile) -> Self {
        self.table.insert(model.into(), profile);
        self
    }
}

impl JouleEstimator for DefaultEstimator {
    fn estimate(&self, input_tokens: u32, output_tokens: u32, model: &str) -> JouleCost {
        let (jin, jout) = match self.table.get(model) {
            Some(p) => (p.joules_per_input_token, p.joules_per_output_token),
            None => (self.fallback_input, self.fallback_output),
        };
        let joules = (input_tokens as f64) * jin + (output_tokens as f64) * jout;
        // Clamp negatives (shouldn't happen) and convert to micro-joules.
        let microjoules = (joules * 1_000_000.0).max(0.0) as u64;
        JouleCost {
            microjoules,
            source: JouleSource::Estimated,
        }
    }
}

impl Default for DefaultEstimator {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_loads_known_models() {
        let est = DefaultEstimator::builtin();
        assert!(est.table.contains_key("claude-3-5-sonnet"));
        assert!(est.table.contains_key("gpt-4o"));
        assert!(est.table.contains_key("gemini-1.5-pro"));
        assert!(est.table.contains_key("llama-3.1-70b"));
        assert!(est.table.contains_key("mixtral-8x7b"));
    }

    #[test]
    fn estimate_matches_table_arithmetic() {
        let est = DefaultEstimator::builtin();
        // claude-3-5-sonnet: 0.10 J/in, 0.50 J/out.
        // 1000 in + 500 out = 100 + 250 = 350 J = 350_000_000 µJ.
        let cost = est.estimate(1000, 500, "claude-3-5-sonnet");
        assert_eq!(cost.microjoules, 350_000_000);
        assert_eq!(cost.source, JouleSource::Estimated);
    }

    #[test]
    fn unknown_model_uses_fallback() {
        let est = DefaultEstimator::builtin();
        let cost = est.estimate(100, 100, "no-such-model");
        // 100*0.10 + 100*0.50 = 60 J.
        assert_eq!(cost.microjoules, 60_000_000);
    }

    #[test]
    fn custom_fallback_takes_effect() {
        let est = DefaultEstimator::builtin().with_fallback(0.001, 0.005);
        let cost = est.estimate(1000, 1000, "no-such-model");
        // 1000*0.001 + 1000*0.005 = 6 J.
        assert_eq!(cost.microjoules, 6_000_000);
    }
}
