//! Model format parsers.
//!
//! Supports loading models from various formats:
//! - SafeTensors (.safetensors) - Recommended, zero-copy friendly
//! - GGUF (.gguf) - Quantized models from llama.cpp ecosystem
//! - Raw binary - For custom formats

mod safetensors;
pub mod gguf;

pub use safetensors::{SafeTensorsFile, SafeTensorInfo};
pub use gguf::{GgufFile, GgufTensorInfo, GgufMetadata, GgmlType};
