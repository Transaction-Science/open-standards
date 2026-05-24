//! Local ms-marco MiniLM cross-encoders (ONNX runtime).
//!
//! Models:
//!
//! * `cross-encoder/ms-marco-MiniLM-L-6-v2` — 22M params, the standard
//!   "cheap cross-encoder" baseline. ~5 µJ/pair on the HF Energy Score
//!   CPU rig.
//! * `cross-encoder/ms-marco-MiniLM-L-12-v2` — 33M params, slightly
//!   higher quality, ~9 µJ/pair.
//!
//! These are the canonical sentence-transformers cross-encoders trained
//! on MS MARCO passage ranking. They emit a single logit per pair that
//! tracks relevance — sort within the candidate set, don't compare
//! across queries.

use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_core::JouleCost;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::error::{RerankError, RerankResult};
use crate::reranker::{Candidate, Reranker, ScoredCandidate};

fn microjoules_per_pair(model: &str) -> u64 {
    match model {
        "cross-encoder/ms-marco-MiniLM-L-6-v2" => 5,
        "cross-encoder/ms-marco-MiniLM-L-12-v2" => 9,
        _ => 10,
    }
}

/// Local ms-marco MiniLM cross-encoder.
pub struct MsMarcoReranker {
    model_name: String,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl MsMarcoReranker {
    /// Construct a reranker from a directory containing `model.onnx` and
    /// `tokenizer.json`.
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

    /// Override the max token length per pair.
    pub fn with_max_length(mut self, n: usize) -> Self {
        self.max_length = n;
        self
    }

    /// Estimate energy for re-ranking `pairs` candidates.
    pub fn joule_estimate(&self, pairs: usize) -> JouleCost {
        JouleCost::estimated((pairs as u64).saturating_mul(microjoules_per_pair(&self.model_name)))
    }

    fn score_pair(&self, query: &str, document: &str) -> RerankResult<f32> {
        let enc = self
            .tokenizer
            .encode((query.to_string(), document.to_string()), true)
            .map_err(|e| RerankError::Local(e.to_string()))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        let mut type_ids: Vec<i64> = enc.get_type_ids().iter().map(|&x| x as i64).collect();
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
            mask.truncate(self.max_length);
            type_ids.truncate(self.max_length);
        }
        let len = ids.len();
        let ids_t =
            Value::from_array(([1usize, len], ids)).map_err(|e| RerankError::Local(e.to_string()))?;
        let mask_t =
            Value::from_array(([1usize, len], mask)).map_err(|e| RerankError::Local(e.to_string()))?;
        let types_t = Value::from_array(([1usize, len], type_ids))
            .map_err(|e| RerankError::Local(e.to_string()))?;

        let mut sess = self.session.lock().expect("session lock");
        let outputs = sess
            .run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "token_type_ids" => types_t,
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
impl Reranker for MsMarcoReranker {
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
        512
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_model() -> Option<PathBuf> {
        let p = std::env::var("EOC_MS_MARCO_RERANKER_DIR").ok()?;
        let dir = PathBuf::from(p);
        if dir.join("model.onnx").is_file() && dir.join("tokenizer.json").is_file() {
            Some(dir)
        } else {
            None
        }
    }

    #[tokio::test]
    async fn local_smoke_when_model_present() {
        let Some(dir) = skip_if_no_model() else {
            eprintln!("skipping: EOC_MS_MARCO_RERANKER_DIR not set");
            return;
        };
        let r = MsMarcoReranker::from_dir("cross-encoder/ms-marco-MiniLM-L-6-v2", dir).expect("load");
        let _ = r
            .rerank(
                "where is paris",
                &[
                    Candidate::new("a", "Paris is the capital of France."),
                    Candidate::new("b", "Bananas grow in tropical climates."),
                ],
            )
            .await
            .expect("rerank");
    }

    #[test]
    fn estimate_known_models() {
        assert_eq!(microjoules_per_pair("cross-encoder/ms-marco-MiniLM-L-6-v2"), 5);
        assert_eq!(microjoules_per_pair("cross-encoder/ms-marco-MiniLM-L-12-v2"), 9);
    }
}
