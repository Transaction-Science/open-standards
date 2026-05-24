//! Local bge-reranker cross-encoder (ONNX runtime).
//!
//! Models:
//!
//! * `bge-reranker-base` (XLM-RoBERTa-base, 278M params, ~25 µJ/pair on
//!   the HF Energy Score CPU rig).
//! * `bge-reranker-large` (XLM-RoBERTa-large, 560M params, ~80 µJ/pair).
//! * `bge-reranker-v2-m3` (BGE-M3 backbone, multilingual, ~95 µJ/pair).
//!
//! Cross-encoder architecture: tokenise `[CLS] query [SEP] document [SEP]`,
//! run a single forward pass, take the logit at position 0 as the
//! relevance score (higher = more relevant). The score range is unbounded;
//! sort within the candidate set, don't compare across queries.
//!
//! Gated behind the `local` feature so the heavy `ort` + `tokenizers`
//! deps stay opt-in.

use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_core::JouleCost;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::error::{RerankError, RerankResult};
use crate::reranker::{Candidate, Reranker, ScoredCandidate};

/// Per-model energy coefficient (µJ/pair). Drawn from HF Energy Score
/// reranker numbers (CPU baseline) — replace with measured values when
/// the runtime [`eoc_meter`] sees real counters.
fn microjoules_per_pair(model: &str) -> u64 {
    match model {
        "bge-reranker-base" => 25,
        "bge-reranker-large" => 80,
        "bge-reranker-v2-m3" => 95,
        _ => 60,
    }
}

/// Local bge-reranker cross-encoder.
pub struct BgeReranker {
    model_name: String,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl BgeReranker {
    /// Construct a reranker from a directory containing `model.onnx` and
    /// `tokenizer.json`. Recognises the three bge model names; passing
    /// any other name falls back to a default joule coefficient.
    pub fn from_dir(model_name: impl Into<String>, dir: impl Into<PathBuf>) -> RerankResult<Self> {
        let model_name = model_name.into();
        let dir = dir.into();
        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| RerankError::Local(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RerankError::Local(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| RerankError::Local(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RerankError::Local(e.to_string()))?;

        Ok(Self {
            model_name,
            session: Mutex::new(session),
            tokenizer,
            max_length: 512,
        })
    }

    /// Override the maximum token length per `(query, document)` pair.
    pub fn with_max_length(mut self, n: usize) -> Self {
        self.max_length = n;
        self
    }

    /// Estimate energy for re-ranking `pairs` candidates.
    pub fn joule_estimate(&self, pairs: usize) -> JouleCost {
        JouleCost::estimated((pairs as u64).saturating_mul(microjoules_per_pair(&self.model_name)))
    }

    fn score_pair(&self, query: &str, document: &str) -> RerankResult<f32> {
        let pair = format!("{query} [SEP] {document}");
        let enc = self
            .tokenizer
            .encode(pair, true)
            .map_err(|e| RerankError::Local(e.to_string()))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
            mask.truncate(self.max_length);
        }

        let len = ids.len();
        let ids_tensor =
            Value::from_array(([1usize, len], ids)).map_err(|e| RerankError::Local(e.to_string()))?;
        let mask_tensor =
            Value::from_array(([1usize, len], mask)).map_err(|e| RerankError::Local(e.to_string()))?;

        let mut sess = self.session.lock().expect("session lock");
        let outputs = sess
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| RerankError::Local(e.to_string()))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| RerankError::Local(e.to_string()))?;
        data.first()
            .copied()
            .ok_or_else(|| RerankError::Local("empty output tensor".into()))
    }
}

#[async_trait]
impl Reranker for BgeReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[Candidate],
    ) -> RerankResult<Vec<ScoredCandidate>> {
        let mut out: Vec<ScoredCandidate> = Vec::with_capacity(candidates.len());
        for c in candidates {
            let score = self.score_pair(query, &c.text)?;
            out.push(ScoredCandidate {
                candidate: c.clone(),
                score,
                rank: 0,
            });
        }
        out.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, c) in out.iter_mut().enumerate() {
            c.rank = i + 1;
        }
        Ok(out)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn max_pairs(&self) -> usize {
        // Local CPU inference — keep the budget modest by default.
        256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_model() -> Option<PathBuf> {
        let p = std::env::var("EOC_BGE_RERANKER_DIR").ok()?;
        let dir = PathBuf::from(p);
        if dir.join("model.onnx").is_file() && dir.join("tokenizer.json").is_file() {
            Some(dir)
        } else {
            None
        }
    }

    #[tokio::test]
    async fn local_inference_smoke_when_model_present() {
        let Some(dir) = skip_if_no_model() else {
            eprintln!("skipping: EOC_BGE_RERANKER_DIR not set or files missing");
            return;
        };
        let r = BgeReranker::from_dir("bge-reranker-base", dir).expect("load");
        let scored = r
            .rerank(
                "what is energy-optimized compute?",
                &[
                    Candidate::new("a", "EOC measures joules per query."),
                    Candidate::new("b", "Cats are mammals."),
                ],
            )
            .await
            .expect("rerank");
        assert_eq!(scored.len(), 2);
        // The relevant doc should score higher than the irrelevant one.
        assert!(scored[0].candidate.id == "a");
    }

    #[test]
    fn joule_estimate_scales() {
        let est = microjoules_per_pair("bge-reranker-large");
        assert_eq!(est, 80);
    }
}
