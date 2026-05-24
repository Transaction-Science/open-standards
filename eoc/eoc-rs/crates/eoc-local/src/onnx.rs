//! ONNX Runtime backend.
//!
//! [ONNX Runtime] is Microsoft's portable inference engine. It accepts
//! models in the ONNX format and ships execution providers for CPU,
//! CUDA, CoreML (Apple), DirectML (Windows GPU), and TensorRT
//! (NVIDIA). The Rust binding is the [`ort`] crate.
//!
//! Generation loop is implemented in this module rather than delegated
//! to the runtime: ONNX models are bare graphs and the standard
//! autoregressive sampling logic (greedy / top-k / top-p / temperature)
//! is composed locally via the [`crate::sampling`] samplers.
//!
//! Joule attribution: pre/post counter reads sandwich the generation
//! call.
//!
//! [ONNX Runtime]: https://onnxruntime.ai
//! [`ort`]: https://crates.io/crates/ort

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_meter::JouleCounter;
use eoc_neural::NeuralBackend;

use crate::error::{LocalError, LocalResult};
use crate::sampling::{GreedySampler, Sampler};
use crate::tokenizer::LocalTokenizer;

/// Generation-loop configuration.
#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature. `0.0` ⇒ greedy.
    pub temperature: f32,
    /// Top-k cutoff. `0` ⇒ disabled.
    pub top_k: u32,
    /// Top-p (nucleus) cutoff. `1.0` ⇒ disabled.
    pub top_p: f32,
    /// Seed for reproducible sampling.
    pub seed: u64,
    /// Token id that ends generation (EOS). `None` ⇒ no explicit stop.
    pub eos_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            eos_token_id: None,
        }
    }
}

/// Which ONNX execution provider to prefer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxExecutionProvider {
    /// CPU only.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// Apple CoreML.
    CoreMl,
    /// Microsoft DirectML.
    DirectMl,
    /// NVIDIA TensorRT (CUDA + graph optimization).
    TensorRt,
}

impl OnnxExecutionProvider {
    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            OnnxExecutionProvider::Cpu => "cpu",
            OnnxExecutionProvider::Cuda => "cuda",
            OnnxExecutionProvider::CoreMl => "coreml",
            OnnxExecutionProvider::DirectMl => "directml",
            OnnxExecutionProvider::TensorRt => "tensorrt",
        }
    }
}

/// ONNX Runtime backend.
pub struct OnnxBackend {
    /// Path to the `.onnx` model file.
    pub model_path: PathBuf,
    /// Tokenizer used for encode/decode.
    pub tokenizer: LocalTokenizer,
    /// Execution provider preference (best-effort; falls back to CPU).
    pub execution_provider: OnnxExecutionProvider,
    /// Generation config.
    pub generation_config: GenerationConfig,
    /// Sampler — defaults to [`GreedySampler`].
    pub sampler: Box<dyn Sampler>,
    /// Joule counter.
    pub meter: Arc<dyn JouleCounter>,
}

impl OnnxBackend {
    /// Construct from a model + tokenizer + counter.
    pub fn new(
        model_path: impl Into<PathBuf>,
        tokenizer: LocalTokenizer,
        meter: Arc<dyn JouleCounter>,
    ) -> LocalResult<Self> {
        let model_path = model_path.into();
        if !model_path.exists() {
            return Err(LocalError::ModelNotFound(model_path.display().to_string()));
        }
        Ok(Self {
            model_path,
            tokenizer,
            execution_provider: OnnxExecutionProvider::Cpu,
            generation_config: GenerationConfig::default(),
            sampler: Box::new(GreedySampler),
            meter,
        })
    }

    /// Override the execution provider.
    pub fn with_execution_provider(mut self, ep: OnnxExecutionProvider) -> Self {
        self.execution_provider = ep;
        self
    }

    /// Override the generation config.
    pub fn with_generation_config(mut self, gc: GenerationConfig) -> Self {
        self.generation_config = gc;
        self
    }

    /// Swap in a different sampler.
    pub fn with_sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = sampler;
        self
    }
}

#[async_trait]
impl NeuralBackend for OnnxBackend {
    async fn infer(&self, q: &Query) -> Response {
        // Integration point with `ort`. Sketch:
        //
        //   use ort::{Session, SessionBuilder, GraphOptimizationLevel, ...};
        //   let session = SessionBuilder::new()?
        //       .with_optimization_level(GraphOptimizationLevel::Level3)?
        //       .with_execution_providers([...])?
        //       .commit_from_file(&self.model_path)?;
        //   let input_ids = self.tokenizer.encode(&q.prompt, true)?;
        //   let mut tokens = input_ids.clone();
        //   for _ in 0..self.generation_config.max_tokens {
        //       let array = ndarray::Array2::from_shape_vec(...);
        //       let outputs = session.run(ort::inputs![array]?)?;
        //       let logits = outputs[0].try_extract_tensor::<f32>()?;
        //       let next = self.sampler.sample(&last_row(logits))?;
        //       tokens.push(next);
        //       if Some(next) == self.generation_config.eos_token_id { break; }
        //   }
        //   let text = self.tokenizer.decode(&tokens[input_ids.len()..], true)?;
        //
        // Reference build sandwich-meters around a placeholder.

        let start = self.meter.read_microjoules().ok();
        let payload = format!(
            "[onnx ep={} model={}]",
            self.execution_provider.as_str(),
            self.model_path.display()
        );
        let end = self.meter.read_microjoules().ok();
        let (microjoules, source) = match (start, end) {
            (Some(s), Some(e)) if e >= s && (e - s) > 0 => (e - s, JouleSource::Measured),
            _ => (0, JouleSource::Estimated),
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
    async fn stub_counter_reports_zero_cost() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.onnx");
        std::fs::write(&model, b"").unwrap();
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(&tok_path, b"{}").unwrap();
        let tok = LocalTokenizer::from_path(&tok_path);
        let b = OnnxBackend::new(&model, tok, Arc::new(StubCounter)).unwrap();
        let q = Query::new("hi");
        let r = b.infer(&q).await;
        assert_eq!(r.joule_cost.microjoules, 0);
        assert_eq!(r.stage, Stage::Neural);
    }

    #[tokio::test]
    async fn missing_model_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(&tok_path, b"{}").unwrap();
        let tok = LocalTokenizer::from_path(&tok_path);
        let r = OnnxBackend::new("/no/such.onnx", tok, Arc::new(StubCounter));
        assert!(matches!(r, Err(LocalError::ModelNotFound(_))));
    }
}
