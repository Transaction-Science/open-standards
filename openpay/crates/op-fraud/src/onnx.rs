//! ONNX-backed scorer.
//!
//! Loads an operator-trained model via `ort` 2.x and runs inference
//! against the [`FeatureVector`].
//!
//! ## Model contract
//!
//! The operator trains a model that:
//! - Takes a single input named `features` of shape `[1, FEATURES]` (=`[1, 32]`)
//!   dtype `float32`.
//! - Produces a single output named `score` of shape `[1, 1]` dtype `float32`,
//!   sigmoid-activated so the value is in `[0.0, 1.0]`.
//!
//! Most fraud-detection models (gradient-boosted trees converted via
//! `onnxmltools`, simple feed-forward nets from PyTorch with
//! `torch.onnx.export`) satisfy this contract out of the box.
//!
//! ## Runtime
//!
//! We use the `load-dynamic` feature so the operator places the ONNX
//! Runtime shared library at a path of their choosing and calls
//! [`init_runtime`] before constructing any scorers. This matches the
//! way operators handle other rail libraries (FedLine MQ client,
//! CloudHSM client): centrally configured, not bundled per crate.

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

use crate::error::{Error, Result};
use crate::features::{FEATURES, FeatureVector};
use crate::scorer::Scorer;

/// Initialize the ONNX Runtime by loading the dynamic library from
/// the given path. Must be called exactly once before constructing
/// any [`OnnxScorer`].
///
/// # Errors
/// `Error::Backend` if the library can't be loaded.
pub fn init_runtime(library_path: impl AsRef<Path>) -> Result<()> {
    ort::init_from(library_path.as_ref().to_string_lossy().to_string())
        .map_err(|e| Error::Backend(format!("ort init: {e}")))?
        .commit();
    Ok(())
}

/// ONNX-backed scorer.
///
/// The `Session` is wrapped in a `Mutex` because `ort::Session::run`
/// takes `&mut self`. For high-throughput deployments operators can
/// shard scoring across multiple `OnnxScorer` instances.
pub struct OnnxScorer {
    name: String,
    session: Mutex<Session>,
}

impl OnnxScorer {
    /// Construct from an ONNX model file on disk.
    ///
    /// # Errors
    /// `Error::ModelLoad` if the file can't be opened or parsed.
    /// `Error::Backend` if the ONNX Runtime can't build the session.
    pub fn from_file(name: impl Into<String>, model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| Error::Backend(format!("session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Backend(format!("opt level: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| Error::Backend(format!("intra threads: {e}")))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| Error::ModelLoad(format!("commit_from_file: {e}")))?;

        Ok(Self {
            name: name.into(),
            session: Mutex::new(session),
        })
    }

    /// Construct from a model byte buffer (useful for embedded models).
    ///
    /// # Errors
    /// As [`Self::from_file`].
    pub fn from_bytes(name: impl Into<String>, model_bytes: &[u8]) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| Error::Backend(format!("session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Backend(format!("opt level: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| Error::Backend(format!("intra threads: {e}")))?
            .commit_from_memory(model_bytes)
            .map_err(|e| Error::ModelLoad(format!("commit_from_memory: {e}")))?;

        Ok(Self {
            name: name.into(),
            session: Mutex::new(session),
        })
    }
}

impl Scorer for OnnxScorer {
    fn name(&self) -> &str {
        &self.name
    }

    fn score(&self, features: &FeatureVector) -> Result<f32> {
        // Convert the fixed-length array into a 1x32 ndarray.
        let input = Array2::<f32>::from_shape_vec((1, FEATURES), features.to_vec())
            .map_err(|e| Error::ModelOutput(format!("shape: {e}")))?;

        let tensor = TensorRef::from_array_view(&input)
            .map_err(|e| Error::Backend(format!("tensor view: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Backend(format!("lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| Error::Backend(format!("session.run: {e}")))?;

        // We expect a single output, indexed by position [0]. ort 2.x
        // returns DynValues; try_extract_array gives us an ndarray view.
        // Type annotation disambiguates the generic.
        let view: ndarray::ArrayViewD<f32> = outputs[0]
            .try_extract_array()
            .map_err(|e| Error::ModelOutput(format!("try_extract_array: {e}")))?;

        let score = *view
            .iter()
            .next()
            .ok_or_else(|| Error::ModelOutput("empty output tensor".into()))?;

        if !score.is_finite() {
            return Err(Error::ModelOutput(format!("non-finite score: {score}")));
        }
        if !(0.0..=1.0).contains(&score) {
            return Err(Error::ScoreOutOfRange(score));
        }

        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // We can't run real ONNX inference in unit tests without a real
    // libonnxruntime.so on the test runner. These tests exercise the
    // pre-flight validation logic that runs without touching the
    // session.

    #[test]
    fn from_file_returns_error_on_missing_path() {
        // Skip if no ONNX runtime is configured — the error would be
        // about init, not file lookup.
        let result = OnnxScorer::from_file("nonexistent", "/tmp/openpay-fraud-nonexistent.onnx");
        assert!(result.is_err());
    }

    #[test]
    fn from_bytes_returns_error_on_empty() {
        let result = OnnxScorer::from_bytes("empty", b"");
        assert!(result.is_err());
    }

    #[test]
    fn from_bytes_returns_error_on_garbage() {
        let result = OnnxScorer::from_bytes("garbage", b"not an onnx model");
        assert!(result.is_err());
    }
}
