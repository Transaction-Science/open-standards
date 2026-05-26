//! `NliEngine` — the production-facing wrapper that ties the
//! tokenizer, weights, and forward pass together behind the
//! [`crate::nli::NliInference`] trait.
//!
//! ```ignore
//! let engine = NliEngine::from_dir("models/deberta-v3-large-mnli")?;
//! let pred = engine.predict(
//!     "Paris is the capital of France.",
//!     "France's capital is Paris.",
//! )?;
//! assert!(matches!(pred.label, NliLabel::Entailment));
//! ```
//!
//! Energy accounting is left to the caller: the engine itself doesn't
//! measure joules, but `NliPrediction.joules_spent` is populated by
//! the per-token cost model whenever the diagnose pillar wires this
//! engine into the joule cascade.

use std::path::Path;

use crate::config::NliLabelLayout;
use crate::forward::forward;
use crate::forward_batch::forward_batch;
use crate::loader::{LoaderError, ModelInventory};
use crate::nli::{predict_from_logits, NliInference, NliInferenceError, NliPrediction};
use crate::tokenizer::{DebertaTokenizer, TokenizerError};
use crate::weights::Weights;

#[derive(Debug)]
pub enum EngineError {
    Load(LoaderError),
    Tokenizer(TokenizerError),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(e) => write!(f, "load: {e}"),
            Self::Tokenizer(e) => write!(f, "tokenizer: {e}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<LoaderError> for EngineError {
    fn from(e: LoaderError) -> Self {
        Self::Load(e)
    }
}

impl From<TokenizerError> for EngineError {
    fn from(e: TokenizerError) -> Self {
        Self::Tokenizer(e)
    }
}

/// End-to-end DeBERTa-v3 NLI engine.
pub struct NliEngine {
    tokenizer: DebertaTokenizer,
    weights: Weights,
    model_id: String,
}

impl NliEngine {
    /// Load tokenizer + full model weights from a HuggingFace-style
    /// directory. Resident memory ~1.7 GB in fp32.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, EngineError> {
        let dir = dir.as_ref();
        let inventory = ModelInventory::from_dir(dir)?;
        let weights = Weights::load_full(dir, &inventory)?;
        let tokenizer = DebertaTokenizer::from_dir(dir)?;
        let model_id = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("deberta-v3")
            .to_string();
        Ok(Self {
            tokenizer,
            weights,
            model_id,
        })
    }

    pub fn label_layout(&self) -> NliLabelLayout {
        self.weights.config.label_layout
    }

    /// Run N (premise, hypothesis) pairs through a single batched
    /// forward pass. Returns predictions positionally — `out[i]`
    /// corresponds to `pairs[i]`.
    ///
    /// Best when the pairs have similar token lengths. Heterogeneous
    /// batches pad to the longest sequence, which wastes work on
    /// shorter pairs; for those workloads
    /// [`crate::forward::forward`] called per pair across a thread
    /// scope (the path `jouleclaw_diagnose::entail_batch` takes) is
    /// typically faster.
    pub fn predict_batch(
        &self,
        pairs: &[(&str, &str)],
    ) -> Result<Vec<NliPrediction>, NliInferenceError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let max_seq = self.weights.config.max_position_embeddings;

        let mut input_ids_batch = Vec::with_capacity(pairs.len());
        let mut attention_mask_batch = Vec::with_capacity(pairs.len());
        for (premise, hypothesis) in pairs {
            let enc = self
                .tokenizer
                .encode_pair(premise, hypothesis)
                .map_err(|e| NliInferenceError::Encoding(e.to_string()))?;
            if enc.token_ids.len() > max_seq {
                return Err(NliInferenceError::InputTooLong {
                    actual: enc.token_ids.len(),
                    max: max_seq,
                });
            }
            input_ids_batch.push(enc.token_ids);
            attention_mask_batch.push(enc.attention_mask);
        }

        let results = forward_batch(&input_ids_batch, &attention_mask_batch, &self.weights)
            .map_err(|e| NliInferenceError::Forward(e.to_string()))?;

        let layout = self.weights.config.label_layout;
        Ok(results
            .into_iter()
            .map(|r| {
                let logits: [f32; 3] = [r.logits[0], r.logits[1], r.logits[2]];
                predict_from_logits(&logits, layout)
            })
            .collect())
    }
}

impl NliInference for NliEngine {
    fn predict(&self, premise: &str, hypothesis: &str) -> Result<NliPrediction, NliInferenceError> {
        let encoded = self
            .tokenizer
            .encode_pair(premise, hypothesis)
            .map_err(|e| NliInferenceError::Encoding(e.to_string()))?;
        let max_seq = self.weights.config.max_position_embeddings;
        if encoded.token_ids.len() > max_seq {
            return Err(NliInferenceError::InputTooLong {
                actual: encoded.token_ids.len(),
                max: max_seq,
            });
        }
        let result = forward(&encoded.token_ids, &encoded.attention_mask, &self.weights)
            .map_err(|e| NliInferenceError::Forward(e.to_string()))?;

        // `forward` returns logits in the model's internal label
        // order. predict_from_logits handles remapping to the
        // schema's canonical {Entails, Neutral, Contradicts}.
        let logits_arr: [f32; 3] = [result.logits[0], result.logits[1], result.logits[2]];
        Ok(predict_from_logits(&logits_arr, self.weights.config.label_layout))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NliLabel;
    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .expect("workspace root")
    }

    fn model_dir() -> Option<PathBuf> {
        let p = workspace_root().join("models/deberta-v3-large-mnli");
        if p.join("model.safetensors").exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn engine_predicts_entailment_for_canonical_pair() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let engine = NliEngine::from_dir(&dir).expect("engine");
        let pred = engine
            .predict(
                "Paris is the capital of France.",
                "France's capital is Paris.",
            )
            .expect("predict");
        eprintln!(
            "label={:?} probs=(entails={:.4}, neutral={:.4}, contradicts={:.4})",
            pred.label,
            pred.probabilities.entails,
            pred.probabilities.neutral,
            pred.probabilities.contradicts,
        );
        assert!(matches!(pred.label, NliLabel::Entailment));
        assert!(
            pred.probabilities.entails > 0.99,
            "expected entailment confidence > 0.99, got {}",
            pred.probabilities.entails
        );
    }

    #[test]
    fn engine_predicts_contradiction() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let engine = NliEngine::from_dir(&dir).expect("engine");
        let pred = engine
            .predict("The cat sat on the mat.", "There was no cat anywhere.")
            .expect("predict");
        eprintln!(
            "contradiction case: label={:?} probs=(entails={:.4}, neutral={:.4}, contradicts={:.4})",
            pred.label,
            pred.probabilities.entails,
            pred.probabilities.neutral,
            pred.probabilities.contradicts,
        );
        assert!(matches!(pred.label, NliLabel::Contradiction));
    }

    #[test]
    fn engine_model_id_reflects_model_dir_name() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let engine = NliEngine::from_dir(&dir).expect("engine");
        assert_eq!(engine.model_id(), "deberta-v3-large-mnli");
    }

    /// `predict_batch` on N identical pairs must agree with N single
    /// `predict` calls — both label and probabilities.
    #[test]
    fn predict_batch_matches_per_pair_predict_on_homogeneous_input() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let engine = NliEngine::from_dir(&dir).expect("engine");

        let pair = ("Paris is the capital of France.", "France's capital is Paris.");
        let single = engine.predict(pair.0, pair.1).unwrap();
        let batched = engine.predict_batch(&[pair, pair, pair]).unwrap();
        assert_eq!(batched.len(), 3);
        for b in &batched {
            assert!(matches!(b.label, l if l as i32 == single.label as i32));
            let de = (b.probabilities.entails - single.probabilities.entails).abs();
            let dn = (b.probabilities.neutral - single.probabilities.neutral).abs();
            let dc = (b.probabilities.contradicts - single.probabilities.contradicts).abs();
            assert!(de < 1e-4 && dn < 1e-4 && dc < 1e-4);
        }
    }

    /// Wall-clock comparison of three dispatch strategies on a
    /// homogeneous batch (4 identical pairs). Reports timings; does
    /// not enforce a threshold — homogeneous batching *should* beat
    /// parallel-per-pair but the data is the data.
    ///
    /// Run with: `cargo test --release -p jouleclaw-deberta
    /// bench_homogeneous_batch -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn bench_homogeneous_batch() {
        use std::time::Instant;
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let engine = NliEngine::from_dir(&dir).expect("engine");

        let pair = ("Paris is the capital of France.", "France's capital is Paris.");
        let n = 4;
        let pairs: Vec<_> = (0..n).map(|_| pair).collect();

        let t = Instant::now();
        for p in &pairs {
            engine.predict(p.0, p.1).unwrap();
        }
        let serial = t.elapsed();

        let t = Instant::now();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n);
            for p in &pairs {
                handles.push(s.spawn(|| engine.predict(p.0, p.1).unwrap()));
            }
            for h in handles {
                let _ = h.join();
            }
        });
        let parallel = t.elapsed();

        let t = Instant::now();
        engine.predict_batch(&pairs).unwrap();
        let batched = t.elapsed();

        eprintln!(
            "homogeneous N={n}: serial={:.3}s  parallel={:.3}s  batched={:.3}s  \
             parallel/serial={:.2}x  batched/parallel={:.2}x  batched/serial={:.2}x",
            serial.as_secs_f64(),
            parallel.as_secs_f64(),
            batched.as_secs_f64(),
            parallel.as_secs_f64() / serial.as_secs_f64(),
            batched.as_secs_f64() / parallel.as_secs_f64(),
            batched.as_secs_f64() / serial.as_secs_f64(),
        );
    }

    /// Heterogeneous workload — short Wikidata-style claims mixed
    /// with longer Wikipedia-summary-style premises. Mirrors the
    /// actual `jouleclaw_diagnose::entail_batch` workload.
    #[test]
    #[ignore]
    fn bench_heterogeneous_batch() {
        use std::time::Instant;
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let engine = NliEngine::from_dir(&dir).expect("engine");

        let claim = "Paris is the capital of France.";
        let long_premise = "Paris is the capital and most populous city of France, \
                            with an estimated population of 2,165,423 residents in \
                            2019 in an area of more than 105 km². Since the 17th \
                            century, Paris has been one of Europe's major centres \
                            of finance, diplomacy, commerce, fashion, science, and \
                            the arts.";
        let short_premise_1 = "wd:Q142 wdt:P36 wd:Q90";
        let short_premise_2 = "France's capital is Paris.";

        let pairs: Vec<(&str, &str)> = vec![
            (long_premise, claim),
            (short_premise_1, claim),
            (short_premise_2, claim),
            (long_premise, claim),
        ];

        let t = Instant::now();
        for p in &pairs {
            engine.predict(p.0, p.1).unwrap();
        }
        let serial = t.elapsed();

        let t = Instant::now();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(pairs.len());
            for p in &pairs {
                handles.push(s.spawn(|| engine.predict(p.0, p.1).unwrap()));
            }
            for h in handles {
                let _ = h.join();
            }
        });
        let parallel = t.elapsed();

        let t = Instant::now();
        engine.predict_batch(&pairs).unwrap();
        let batched = t.elapsed();

        eprintln!(
            "heterogeneous N={}: serial={:.3}s  parallel={:.3}s  batched={:.3}s  \
             parallel/serial={:.2}x  batched/parallel={:.2}x  batched/serial={:.2}x",
            pairs.len(),
            serial.as_secs_f64(),
            parallel.as_secs_f64(),
            batched.as_secs_f64(),
            parallel.as_secs_f64() / serial.as_secs_f64(),
            batched.as_secs_f64() / parallel.as_secs_f64(),
            batched.as_secs_f64() / serial.as_secs_f64(),
        );
    }
}
