//! Joule cost estimation for embedding inference.
//!
//! The estimator multiplies an input token count by a per-model coefficient
//! drawn from [`data/embedding_energy_profiles.json`](../../data/embedding_energy_profiles.json).
//! Tokens are approximated from character counts using a constant chars-per-token ratio.
//! Real measurements come from [`eoc_meter`] when hardware counters are present;
//! this estimator returns [`eoc_core::JouleSource::Estimated`].

use serde::{Deserialize, Serialize};

use eoc_core::JouleCost;

/// Energy profile for a single embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEnergyProfile {
    /// Canonical model name (e.g. `"text-embedding-3-small"`).
    pub name: String,
    /// Vendor or runtime hosting the model (`"openai"`, `"local"`, ...).
    pub vendor: String,
    /// Output dimensionality.
    pub dimensions: usize,
    /// Micro-joules per input token.
    pub microjoules_per_token: u64,
}

/// Estimator: maps `(model, text_len_chars)` to a [`JouleCost`].
#[derive(Debug, Clone)]
pub struct JouleEstimator {
    profiles: Vec<EmbeddingEnergyProfile>,
    chars_per_token: f64,
}

const BUNDLED_PROFILES: &str = include_str!("../data/embedding_energy_profiles.json");

#[derive(Debug, Deserialize)]
struct ProfileFile {
    #[serde(rename = "_chars_per_token", default = "default_chars_per_token")]
    chars_per_token: f64,
    models: Vec<EmbeddingEnergyProfile>,
}

fn default_chars_per_token() -> f64 {
    4.0
}

impl Default for JouleEstimator {
    fn default() -> Self {
        Self::from_bundled().unwrap_or_else(|_| Self {
            profiles: Vec::new(),
            chars_per_token: 4.0,
        })
    }
}

impl JouleEstimator {
    /// Load the bundled profile database.
    pub fn from_bundled() -> Result<Self, serde_json::Error> {
        let pf: ProfileFile = serde_json::from_str(BUNDLED_PROFILES)?;
        Ok(Self {
            profiles: pf.models,
            chars_per_token: pf.chars_per_token,
        })
    }

    /// Load profiles from a JSON string with the same schema as the bundled
    /// file.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let pf: ProfileFile = serde_json::from_str(s)?;
        Ok(Self {
            profiles: pf.models,
            chars_per_token: pf.chars_per_token,
        })
    }

    /// Look up a profile by canonical model name.
    pub fn profile(&self, model: &str) -> Option<&EmbeddingEnergyProfile> {
        self.profiles.iter().find(|p| p.name == model)
    }

    /// Estimate energy in micro-joules for embedding `text_len_chars`
    /// characters with `model`.
    ///
    /// Falls back to a conservative 20 µJ/token for unknown models.
    pub fn estimate(&self, model: &str, text_len_chars: usize) -> JouleCost {
        let coef = self
            .profile(model)
            .map(|p| p.microjoules_per_token)
            .unwrap_or(20);
        let tokens = (text_len_chars as f64 / self.chars_per_token).ceil() as u64;
        JouleCost::estimated(tokens.saturating_mul(coef))
    }

    /// Characters-per-token ratio used by [`Self::estimate`].
    pub fn chars_per_token(&self) -> f64 {
        self.chars_per_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_profiles_load() {
        let est = JouleEstimator::from_bundled().expect("bundled profiles parse");
        assert!(est.profile("text-embedding-3-small").is_some());
        assert!(est.profile("voyage-3").is_some());
        assert!(est.profile("bge-small-en-v1.5").is_some());
    }

    #[test]
    fn estimate_known_model() {
        let est = JouleEstimator::from_bundled().expect("bundled");
        // 400 chars / 4 chars-per-token = 100 tokens; 18 µJ/token → 1800 µJ.
        let cost = est.estimate("text-embedding-3-small", 400);
        assert_eq!(cost.microjoules, 1800);
    }

    #[test]
    fn unknown_model_falls_back() {
        let est = JouleEstimator::from_bundled().expect("bundled");
        // 40 chars → 10 tokens → 200 µJ.
        let cost = est.estimate("not-a-real-model", 40);
        assert_eq!(cost.microjoules, 200);
    }
}
