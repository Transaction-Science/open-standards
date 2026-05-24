//! Apple MLX backend.
//!
//! MLX is Apple's machine-learning framework for Apple Silicon. It
//! targets the unified GPU and the Neural Engine and is the right
//! choice on M-series Macs for models that have been converted to MLX
//! format. The Rust binding lives in [`mlx-rs`].
//!
//! Joule attribution on macOS uses `powermetrics`, which is already
//! wired into [`eoc_meter`]. `powermetrics` requires root in practice;
//! when readings fail, the backend tags joule cost as
//! [`JouleSource::Estimated`](eoc_core::JouleSource::Estimated) and
//! falls back to a coefficient-based estimate.
//!
//! This module is hard-gated to `target_os = "macos"`. On any other
//! platform compilation of `--features mlx` will fail with a clear
//! `cfg` error; the rest of the crate continues to build.
//!
//! [`mlx-rs`]: https://crates.io/crates/mlx-rs

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_meter::JouleCounter;
use eoc_neural::NeuralBackend;

use crate::error::{LocalError, LocalResult};

/// Apple MLX backend.
pub struct MlxBackend {
    /// Path to the converted MLX model directory (contains
    /// `weights.safetensors` + `config.json` + `tokenizer.json`).
    pub model_path: PathBuf,
    /// Path to the tokenizer file (typically inside `model_path`).
    pub tokenizer_path: PathBuf,
    /// Max tokens to generate.
    pub max_tokens: u32,
    /// Joule counter (typically `MacosPowermetricsCounter`).
    pub meter: Arc<dyn JouleCounter>,
    /// Estimated joules per output token, used when `powermetrics`
    /// cannot give a live reading.
    pub fallback_joules_per_token: f64,
}

impl MlxBackend {
    /// Construct an MLX backend.
    pub fn new(
        model_path: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
        meter: Arc<dyn JouleCounter>,
    ) -> LocalResult<Self> {
        let model_path = model_path.into();
        let tokenizer_path = tokenizer_path.into();
        if !model_path.exists() {
            return Err(LocalError::ModelNotFound(model_path.display().to_string()));
        }
        Ok(Self {
            model_path,
            tokenizer_path,
            max_tokens: 512,
            meter,
            fallback_joules_per_token: 0.05,
        })
    }

    /// Override the max token cap.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Override the fallback joule estimate.
    pub fn with_fallback_joules_per_token(mut self, j: f64) -> Self {
        self.fallback_joules_per_token = j;
        self
    }
}

#[async_trait]
impl NeuralBackend for MlxBackend {
    async fn infer(&self, q: &Query) -> Response {
        // Integration point with `mlx-rs`. Sketch:
        //
        //   use mlx_rs::module::Module;
        //   use mlx_rs::ops::*;
        //   let model = mlx_lm::load(&self.model_path)?;
        //   let tokenizer = load_tokenizer(&self.tokenizer_path)?;
        //   let ids = tokenizer.encode(&q.prompt, true)?;
        //   for token in model.generate(ids, self.max_tokens) {
        //       ...
        //   }
        //
        // The reference build runs the placeholder generator below so
        // the trait + plumbing are exercised end-to-end. Operators
        // swap in the real mlx-rs symbols against their pinned rev.

        let start = self.meter.read_microjoules().ok();
        let estimated_tokens = self.max_tokens.min(64) as u32;
        let payload = format!("[mlx model={} prompt={:?}]", self.model_path.display(), q.prompt);

        let end = self.meter.read_microjoules().ok();
        let (microjoules, source) = match (start, end) {
            (Some(s), Some(e)) if e >= s && (e - s) > 0 => (e - s, JouleSource::Measured),
            _ => {
                let j = (estimated_tokens as f64) * self.fallback_joules_per_token;
                ((j * 1_000_000.0).max(0.0) as u64, JouleSource::Estimated)
            }
        };
        Response::new(
            q.id,
            payload,
            Stage::Neural,
            JouleCost {
                microjoules,
                source,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eoc_meter::StubCounter;

    #[tokio::test]
    async fn stub_counter_reports_estimated_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model");
        std::fs::create_dir(&model).unwrap();
        let tok = model.join("tokenizer.json");
        std::fs::write(&tok, b"{}").unwrap();
        let b = MlxBackend::new(&model, &tok, Arc::new(StubCounter))
            .unwrap()
            .with_max_tokens(8)
            .with_fallback_joules_per_token(0.1);
        let q = Query::new("ping");
        let r = b.infer(&q).await;
        assert_eq!(r.stage, Stage::Neural);
        // 8 tokens * 0.1 J/token = 0.8 J = 800_000 µJ.
        assert_eq!(r.joule_cost.source, JouleSource::Estimated);
        assert_eq!(r.joule_cost.microjoules, 800_000);
    }

    #[tokio::test]
    async fn missing_model_returns_not_found() {
        let r = MlxBackend::new("/no/such/path", "/no/tok", Arc::new(StubCounter));
        assert!(matches!(r, Err(LocalError::ModelNotFound(_))));
    }
}
