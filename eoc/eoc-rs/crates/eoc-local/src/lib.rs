//! EOC local-inference backends.
//!
//! This crate ingests the four dominant on-host inference stacks behind
//! the same [`eoc_neural::NeuralBackend`] trait that
//! [`eoc-vendor-api`](https://docs.rs/eoc-vendor-api) uses for commercial
//! endpoints. Where vendor APIs *estimate* joule cost from token counts,
//! local backends *measure* it — the host hardware is right there, and
//! [`eoc_meter::JouleCounter`] readings taken on either side of the
//! inference give an honest reading.
//!
//! ## Backends (all feature-gated)
//!
//! | Feature     | Stack       | Hardware                                  | Format |
//! |-------------|-------------|-------------------------------------------|--------|
//! | `llamacpp`  | llama.cpp   | CPU + Metal / CUDA / ROCm / Vulkan offload | GGUF  |
//! | `mlx`       | Apple MLX   | Apple Silicon GPU + Neural Engine          | safetensors / mlx |
//! | `mlc`       | MLC-LLM     | Vulkan / Metal / OpenCL / CUDA via TVM     | MLC-compiled |
//! | `onnx`      | ONNX Runtime| CPU / CUDA / CoreML / DirectML / TensorRT | ONNX   |
//!
//! The `gguf` feature exposes a read-only GGUF parser that is useful on
//! its own — for introspection, tooling, registry population — without
//! pulling in the llama.cpp runtime.
//!
//! ## Joule attribution
//!
//! Each backend takes a [`JouleCounter`](eoc_meter::JouleCounter), reads
//! it before and after inference, and reports the delta as
//! [`JouleSource::Measured`](eoc_core::JouleSource::Measured). When the
//! attached counter is [`StubCounter`](eoc_meter::StubCounter), cost is
//! zero — useful for tests, deterministic playback, and CI.
//!
//! ## WASM
//!
//! [`gguf`] and [`tokenizer`] compile to `wasm32-unknown-unknown`. The
//! inference backends do not — they all bind to native libraries that
//! assume an OS thread model.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod budget;
pub mod error;
pub mod model_registry;
pub mod sampling;

#[cfg(any(feature = "tokenizer-only", feature = "onnx"))]
pub mod tokenizer;

#[cfg(any(feature = "gguf", feature = "llamacpp"))]
pub mod gguf;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub mod mlx;

#[cfg(feature = "mlc")]
pub mod mlc;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use budget::{Budget, BudgetDecision, BudgetPolicy};
pub use error::{LocalError, LocalResult};
pub use model_registry::{ModelEntry, ModelRegistry, Quantization};
pub use sampling::{
    ComposeSampler, GreedySampler, MirostatSampler, Sampler, TemperatureSampler, TopKSampler,
    TopPSampler,
};

#[cfg(any(feature = "gguf", feature = "llamacpp"))]
pub use gguf::{GgufFile, GgufHeader, GgufMetadataValue, GgufTensorInfo};

#[cfg(feature = "llamacpp")]
pub use llamacpp::LlamaCppBackend;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use mlx::MlxBackend;

#[cfg(feature = "mlc")]
pub use mlc::MlcBackend;

#[cfg(feature = "onnx")]
pub use onnx::{GenerationConfig, OnnxBackend};

#[cfg(any(feature = "tokenizer-only", feature = "onnx"))]
pub use tokenizer::LocalTokenizer;
