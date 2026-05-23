//! Burn-backed scorer.
//!
//! Pure-Rust ML inference. No external runtime, no shared library. The
//! tradeoff vs the ONNX path: operators must train their model in Burn
//! (or convert via Burn's ONNX import) rather than directly importing a
//! PyTorch / TensorFlow model.
//!
//! ## Architecture
//!
//! Fixed feed-forward MLP:
//!
//! ```text
//! input [1, 32]
//!   → Linear(32 → 32)
//!   → ReLU
//!   → Linear(32 → 16)
//!   → ReLU
//!   → Linear(16 → 1)
//!   → Sigmoid
//! → output [1, 1] in [0.0, 1.0]
//! ```
//!
//! Operators training a model write Burn training loops using this same
//! `FraudMlp` module, then serialize the trained record with Burn's
//! `BinFileRecorder` and deploy the `.mpk` file alongside the binary.
//!
//! ## Why fixed architecture
//!
//! Two reasons:
//!
//! 1. **Type-level certainty.** A Burn model's parameters are typed at
//!    construction. Loading a record requires the exact module type the
//!    record was saved with. We fix the architecture so operators don't
//!    have to ship matching `FraudMlp` Rust source alongside the model.
//!
//! 2. **Fraud doesn't need depth.** Gradient-boosted trees with ~100
//!    features are the industry baseline. A 2-hidden-layer MLP with 32
//!    features matches their performance and trains in seconds on a CPU.
//!    Anything bigger has diminishing returns and inflates inference
//!    latency — bad on instant-rail timelines.
//!
//! Operators who need a different architecture (transformer, graph net,
//! ensemble) use the ONNX path instead.

use std::path::Path;
use std::sync::Mutex;

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu, Sigmoid};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_ndarray::{NdArray, NdArrayDevice};

use crate::error::{Error, Result};
use crate::features::{FEATURES, FeatureVector};
use crate::scorer::Scorer;

/// Hidden-layer width. Same scale as the input.
const HIDDEN1: usize = 32;
/// Second hidden width. Narrower for compression.
const HIDDEN2: usize = 16;

/// The fixed fraud-scoring MLP.
///
/// Public so operators training models import the exact same type the
/// inference side will load with. The module is generic over `B: Backend`
/// so training can run on WGPU / Candle and inference on NdArray.
#[derive(Module, Debug)]
pub struct FraudMlp<B: Backend> {
    /// First dense layer: input → hidden1.
    pub linear1: Linear<B>,
    /// Second dense layer: hidden1 → hidden2.
    pub linear2: Linear<B>,
    /// Output layer: hidden2 → 1.
    pub linear3: Linear<B>,
    /// ReLU activation between hidden layers.
    pub activation: Relu,
    /// Sigmoid output activation, mapping logits to `[0.0, 1.0]`.
    pub sigmoid: Sigmoid,
}

impl<B: Backend> FraudMlp<B> {
    /// Construct a freshly-initialized model. Weights are randomly
    /// initialized per Burn's defaults; the caller is expected to load
    /// a trained record before scoring.
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(FEATURES, HIDDEN1).init(device),
            linear2: LinearConfig::new(HIDDEN1, HIDDEN2).init(device),
            linear3: LinearConfig::new(HIDDEN2, 1).init(device),
            activation: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }

    /// Forward pass. Input shape `[batch, 32]`, output `[batch, 1]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        self.sigmoid.forward(x)
    }
}

/// CPU backend alias for inference. The NdArray backend is the only one
/// that compiles for no_std + WASM + iOS/Android consistently.
pub type InferenceBackend = NdArray<f32>;

/// Burn-backed scorer.
///
/// Wraps a trained [`FraudMlp`] and exposes the [`Scorer`] interface.
/// The session is wrapped in a `Mutex` to make the type `Sync`; Burn's
/// modules are themselves `Send + Sync` but we use a Mutex for the same
/// reason as the ONNX scorer — so we can present a uniform interior-
/// mutability story to the orchestrator.
pub struct BurnScorer {
    name: String,
    model: Mutex<FraudMlp<InferenceBackend>>,
    device: NdArrayDevice,
}

impl BurnScorer {
    /// Construct with a freshly-initialized model (random weights).
    /// For tests only — production must load a trained record.
    #[must_use]
    pub fn new_untrained(name: impl Into<String>) -> Self {
        let device = NdArrayDevice::default();
        let model = FraudMlp::<InferenceBackend>::new(&device);
        Self {
            name: name.into(),
            model: Mutex::new(model),
            device,
        }
    }

    /// Load a trained model from a Burn record file on disk.
    ///
    /// The file must have been produced by the same `FraudMlp`
    /// architecture using Burn's `BinFileRecorder<FullPrecisionSettings>`.
    ///
    /// # Errors
    /// `Error::ModelLoad` if the file can't be opened or doesn't match
    /// the expected module shape.
    pub fn from_file(name: impl Into<String>, model_path: impl AsRef<Path>) -> Result<Self> {
        let device = NdArrayDevice::default();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(model_path.as_ref().to_path_buf(), &device)
            .map_err(|e| Error::ModelLoad(format!("burn load: {e}")))?;
        let model = FraudMlp::<InferenceBackend>::new(&device).load_record(record);
        Ok(Self {
            name: name.into(),
            model: Mutex::new(model),
            device,
        })
    }
}

impl Scorer for BurnScorer {
    fn name(&self) -> &str {
        &self.name
    }

    fn score(&self, features: &FeatureVector) -> Result<f32> {
        let model = self
            .model
            .lock()
            .map_err(|e| Error::Backend(format!("lock poisoned: {e}")))?;

        // Build a [1, FEATURES] tensor. `TensorData::from` accepts a slice
        // unambiguously; `from_floats` accepts arrays but slice support
        // is API-dependent across Burn versions, so we go through
        // TensorData to be conservative.
        let data = burn::tensor::TensorData::new(features.to_vec(), [FEATURES]);
        let input =
            Tensor::<InferenceBackend, 1>::from_data(data, &self.device).reshape([1, FEATURES]);

        let output = model.forward(input);

        // Pull out the single scalar score.
        let data = output.into_data();
        let slice = data
            .as_slice::<f32>()
            .map_err(|e| Error::ModelOutput(format!("output not f32: {e:?}")))?;
        let score = *slice
            .first()
            .ok_or_else(|| Error::ModelOutput("empty output".into()))?;

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

    #[test]
    fn untrained_model_returns_bounded_score() {
        let s = BurnScorer::new_untrained("burn-untrained-test");
        let f = [0.5_f32; FEATURES];
        let score = s.score(&f).unwrap();
        assert!((0.0..=1.0).contains(&score), "score {score} out of range");
    }

    #[test]
    fn untrained_model_is_deterministic_per_instance() {
        // Same model + same input must give the same score. Untrained
        // models have random weights but those weights are fixed once
        // the model is constructed.
        let s = BurnScorer::new_untrained("burn-determinism");
        let f = [0.123_f32; FEATURES];
        let a = s.score(&f).unwrap();
        let b = s.score(&f).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn from_file_returns_error_on_missing_path() {
        let result = BurnScorer::from_file("missing", "/tmp/openpay-fraud-nonexistent.mpk");
        assert!(result.is_err());
    }

    #[test]
    fn burn_scorer_is_object_safe() {
        let s: Box<dyn Scorer> = Box::new(BurnScorer::new_untrained("dyn-test"));
        let f = [0.0_f32; FEATURES];
        let _ = s.score(&f);
    }

    #[test]
    fn name_is_stable() {
        let s = BurnScorer::new_untrained("my-model-v2");
        assert_eq!(s.name(), "my-model-v2");
    }

    #[test]
    fn fraud_mlp_forward_shape_is_correct() {
        let device = NdArrayDevice::default();
        let model = FraudMlp::<InferenceBackend>::new(&device);
        let input = Tensor::<InferenceBackend, 2>::zeros([1, FEATURES], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 1]);
    }

    #[test]
    fn forward_output_is_in_unit_interval() {
        // Sigmoid activation guarantees [0, 1] regardless of input.
        let device = NdArrayDevice::default();
        let model = FraudMlp::<InferenceBackend>::new(&device);
        let extreme_input = Tensor::<InferenceBackend, 2>::ones([1, FEATURES], &device) * 100.0;
        let output = model.forward(extreme_input);
        let data = output.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        let score = slice[0];
        assert!(
            (0.0..=1.0).contains(&score),
            "sigmoid output {score} must be in [0, 1]"
        );
    }
}
