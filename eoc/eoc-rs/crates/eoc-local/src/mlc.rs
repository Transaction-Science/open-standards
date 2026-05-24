//! MLC-LLM backend (TVM-compiled cross-platform models).
//!
//! [MLC-LLM](https://llm.mlc.ai) compiles a model graph through Apache
//! TVM to a target GPU runtime. It supports Vulkan, Metal, OpenCL, and
//! CUDA out of one source. The price is an extra compilation step
//! per-target.
//!
//! **Compilation is the operator's responsibility.** This crate
//! provides the *runtime loader* only. The expected on-disk layout is
//! the standard `mlc-chat-config.json` + compiled library + weights.
//!
//! Joule attribution piggy-backs on the `mlc_chat` runtime's own
//! reported timing where available; otherwise we sandwich the inference
//! call with [`JouleCounter`] reads exactly like the other backends.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_meter::JouleCounter;
use eoc_neural::NeuralBackend;

use crate::error::{LocalError, LocalResult};

/// MLC-LLM backend.
pub struct MlcBackend {
    /// Path to the directory containing `mlc-chat-config.json`,
    /// the compiled TVM library, and weights.
    pub model_dir: PathBuf,
    /// Path to the compiled model library (`.so` / `.dylib` /
    /// `.dll` / `.wasm`).
    pub library_path: Option<PathBuf>,
    /// Target backend identifier (`metal`, `cuda`, `vulkan`, `opencl`).
    pub device: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Joule counter for attribution.
    pub meter: Arc<dyn JouleCounter>,
}

impl MlcBackend {
    /// Construct from a model directory.
    pub fn new(model_dir: impl Into<PathBuf>, meter: Arc<dyn JouleCounter>) -> LocalResult<Self> {
        let model_dir = model_dir.into();
        if !model_dir.exists() {
            return Err(LocalError::ModelNotFound(model_dir.display().to_string()));
        }
        Ok(Self {
            model_dir,
            library_path: None,
            device: default_device().to_string(),
            max_tokens: 512,
            meter,
        })
    }

    /// Override the compiled library path. Defaults to discovering the
    /// shared library inside `model_dir`.
    pub fn with_library(mut self, path: impl Into<PathBuf>) -> Self {
        self.library_path = Some(path.into());
        self
    }

    /// Override the device target.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Override max tokens.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }
}

fn default_device() -> &'static str {
    if cfg!(target_os = "macos") {
        "metal"
    } else if cfg!(any(target_os = "linux", target_os = "windows")) {
        // Vulkan is the common cross-vendor GPU path on Linux and
        // Windows. CUDA / DirectML are available when the operator
        // overrides via `with_device()`.
        "vulkan"
    } else {
        "cpu"
    }
}

#[async_trait]
impl NeuralBackend for MlcBackend {
    async fn infer(&self, q: &Query) -> Response {
        // Integration point with `tvm-rt` / `mlc-llm` runtime. Sketch:
        //
        //   use tvm_rt::Module as TvmModule;
        //   let lib = TvmModule::load(&library_path)?;
        //   let mlc = mlc_llm::ChatModule::new(&self.model_dir, lib, &self.device)?;
        //   let response = mlc.generate(&q.prompt, GenerationOptions { max_tokens, ... })?;
        //
        // Reference build emits a placeholder and uses the meter
        // sandwich for attribution.

        let start = self.meter.read_microjoules().ok();
        let payload = format!(
            "[mlc device={} model_dir={}]",
            self.device,
            self.model_dir.display()
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
    async fn defaults_pick_a_device() {
        let dir = tempfile::tempdir().unwrap();
        let b = MlcBackend::new(dir.path(), Arc::new(StubCounter)).unwrap();
        assert!(!b.device.is_empty());
    }

    #[tokio::test]
    async fn missing_dir_yields_not_found() {
        let r = MlcBackend::new("/no/such/dir", Arc::new(StubCounter));
        assert!(matches!(r, Err(LocalError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn stub_counter_reports_zero() {
        let dir = tempfile::tempdir().unwrap();
        let b = MlcBackend::new(dir.path(), Arc::new(StubCounter)).unwrap();
        let q = Query::new("hello");
        let r = b.infer(&q).await;
        assert_eq!(r.joule_cost.microjoules, 0);
        assert_eq!(r.stage, Stage::Neural);
    }
}
