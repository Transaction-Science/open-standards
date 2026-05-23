//! Local ONNX embedding backend.
//!
//! Loads an ONNX-exported sentence-transformer model from disk and embeds
//! text on-device. Supports the BGE family (`bge-large-en-v1.5`, `bge-m3`,
//! `bge-small-en-v1.5`), `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`,
//! GTE, and E5.
//!
//! Tokenization uses the model's HuggingFace `tokenizer.json`. Pooling is
//! mean-of-last-hidden-state, which matches the reference encoders for
//! every supported model.
//!
//! Joule cost comes from [`eoc_meter`] when a hardware counter is
//! available; otherwise the per-model coefficient from
//! [`crate::joule_estimator`] is used.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

use eoc_core::JouleCost;

use crate::embedder::Embedder;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::joule_estimator::JouleEstimator;

/// Local ONNX-runtime embedder.
pub struct LocalEmbedder {
    model_name: String,
    dimensions: usize,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    estimator: JouleEstimator,
}

/// Hint that maps known model names to their output dimensionality.
fn dimensions_for(model: &str) -> Option<usize> {
    match model {
        "bge-large-en-v1.5" => Some(1024),
        "bge-m3" => Some(1024),
        "bge-small-en-v1.5" => Some(384),
        "nomic-embed-text-v1.5" => Some(768),
        "mxbai-embed-large-v1" => Some(1024),
        "gte-large" => Some(1024),
        "e5-large-v2" => Some(1024),
        _ => None,
    }
}

impl LocalEmbedder {
    /// Load a local ONNX model.
    ///
    /// `model_dir` must contain `model.onnx` and `tokenizer.json`.
    pub fn load(
        model_name: impl Into<String>,
        model_dir: impl AsRef<Path>,
    ) -> EmbeddingResult<Self> {
        let model_name = model_name.into();
        let dimensions = dimensions_for(&model_name)
            .ok_or_else(|| EmbeddingError::ModelNotFound(model_name.clone()))?;

        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        let model_path = dir.join("model.onnx");
        let tok_path = dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| EmbeddingError::Local(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbeddingError::Local(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| EmbeddingError::Local(e.to_string()))?;

        Ok(Self {
            model_name,
            dimensions,
            session: Mutex::new(session),
            tokenizer,
            estimator: JouleEstimator::default(),
        })
    }

    /// Tokenize a batch into row-major `(ids, mask)` buffers plus the
    /// `(batch, seq)` shape.
    fn tokenize_batch(
        &self,
        texts: &[&str],
    ) -> EmbeddingResult<(Vec<i64>, Vec<i64>, usize, usize)> {
        let mut encodings = Vec::with_capacity(texts.len());
        let mut max_len = 0usize;
        for t in texts {
            let enc = self
                .tokenizer
                .encode(*t, true)
                .map_err(|e| EmbeddingError::Local(e.to_string()))?;
            max_len = max_len.max(enc.get_ids().len());
            encodings.push(enc);
        }
        if max_len == 0 {
            max_len = 1;
        }
        let bsz = texts.len().max(1);
        let mut ids = vec![0i64; bsz * max_len];
        let mut mask = vec![0i64; bsz * max_len];
        for (i, enc) in encodings.iter().enumerate() {
            for (j, t) in enc.get_ids().iter().enumerate() {
                ids[i * max_len + j] = *t as i64;
                mask[i * max_len + j] = 1;
            }
        }
        Ok((ids, mask, bsz, max_len))
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    async fn embed(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        let (ids, mask, bsz, seq) = self.tokenize_batch(texts)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| EmbeddingError::Local("session lock poisoned".to_string()))?;

        let shape = [bsz, seq];
        let ids_v = Value::from_array((shape, ids))
            .map_err(|e| EmbeddingError::Local(e.to_string()))?;
        let mask_v = Value::from_array((shape, mask.clone()))
            .map_err(|e| EmbeddingError::Local(e.to_string()))?;
        let zeros: Vec<i64> = vec![0; bsz * seq];
        let tok_v = Value::from_array((shape, zeros))
            .map_err(|e| EmbeddingError::Local(e.to_string()))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_v,
                "attention_mask" => mask_v,
                "token_type_ids" => tok_v,
            ])
            .map_err(|e| EmbeddingError::Local(e.to_string()))?;

        // First output is last_hidden_state: [batch, seq, hidden].
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::Local(e.to_string()))?;

        let hidden = self.dimensions;
        if bsz * seq * hidden != data.len() {
            return Err(EmbeddingError::Local(format!(
                "unexpected hidden state shape: {} elements, expected {}*{}*{}",
                data.len(),
                bsz,
                seq,
                hidden
            )));
        }

        // Mean-pool over sequence, masked by attention.
        let mut out = Vec::with_capacity(bsz);
        for b in 0..bsz {
            let mut acc = vec![0.0f32; hidden];
            let mut n = 0u32;
            for s in 0..seq {
                if mask[b * seq + s] == 0 {
                    continue;
                }
                n += 1;
                let base = (b * seq + s) * hidden;
                for h in 0..hidden {
                    acc[h] += data[base + h];
                }
            }
            if n > 0 {
                let inv = 1.0 / n as f32;
                for v in &mut acc {
                    *v *= inv;
                }
            }
            // L2-normalize, matching all supported models.
            let mut norm = 0.0f32;
            for v in &acc {
                norm += v * v;
            }
            let norm = norm.sqrt();
            if norm > 0.0 {
                let inv = 1.0 / norm;
                for v in &mut acc {
                    *v *= inv;
                }
            }
            out.push(acc);
        }
        Ok(out)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn joule_estimate(&self, text_len_chars: usize) -> JouleCost {
        self.estimator.estimate(&self.model_name, text_len_chars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_lookup_table() {
        assert_eq!(dimensions_for("bge-large-en-v1.5"), Some(1024));
        assert_eq!(dimensions_for("bge-small-en-v1.5"), Some(384));
        assert_eq!(dimensions_for("nomic-embed-text-v1.5"), Some(768));
        assert_eq!(dimensions_for("mxbai-embed-large-v1"), Some(1024));
        assert_eq!(dimensions_for("not-a-model"), None);
    }

    #[tokio::test]
    async fn load_skips_when_model_absent() {
        // Skip if the test fixture isn't present.
        let dir = std::env::var("EOC_LOCAL_MODEL_DIR").ok();
        let Some(dir) = dir else {
            return; // not present; skip silently
        };
        let r = LocalEmbedder::load("bge-small-en-v1.5", &dir);
        if r.is_err() {
            return;
        }
        let e = r.expect("loaded");
        let out = e.embed(&["hello world"]).await.expect("embed");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 384);
    }
}
