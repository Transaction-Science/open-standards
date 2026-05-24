//! Native-sink neural decoders for WAI envelopes.
//!
//! Mirrors the browser-side runtime in `wai-web/demo/`: each
//! `wai.neural.<name>` capability has a wire format and a sink-installed
//! ONNX decoder. The browser loads the decoder via onnxruntime-web; this
//! module loads it via `ort` (the Rust binding to the same upstream ONNX
//! Runtime). Same models, same inputs, same outputs.
//!
//! The entry point is `decode_envelope`, which dispatches by capability
//! string from a parsed WAI manifest. Each capability has its own
//! per-medium output type (f32 audio samples + sample rate, RGB image,
//! frame sequence).
//!
//! Models are NOT bundled. The caller passes a `ModelRegistry` mapping
//! capability strings to ONNX file paths — exactly the same architecture
//! the browser uses with its `<meta name="wai-model-…">` tags.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::container::Wai;

mod runtime;
mod audio;
mod image;
mod video;
mod rans;

pub use audio::DecodedAudio;
pub use image::DecodedImage;
pub use video::DecodedVideo;

/// Maps `wai.neural.<name>` capability strings to ONNX decoder paths.
///
/// Build it once at startup; the runtime caches sessions internally.
/// The browser equivalent is the `NEURAL_MODELS` map in
/// `wai-web/demo/index.html`.
#[derive(Default, Debug, Clone)]
pub struct ModelRegistry {
    pub paths: HashMap<String, PathBuf>,
}

impl ModelRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(mut self, capability: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.paths.insert(capability.into(), path.into());
        self
    }

    pub fn get(&self, capability: &str) -> Option<&PathBuf> {
        self.paths.get(capability)
    }
}

/// Output of a neural-capability decode. The variant is determined by
/// the capability's media class — audio caps produce `Audio`, image caps
/// produce `Image`, video caps produce `Video`.
#[derive(Debug, Clone)]
pub enum Decoded {
    Audio(DecodedAudio),
    Image(DecodedImage),
    Video(DecodedVideo),
}

#[derive(Debug)]
pub enum DecodeError {
    UnknownCapability(String),
    ModelNotRegistered(String),
    Ort(String),
    InvalidPayload(String),
    Zstd(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownCapability(c) => write!(f, "unknown capability: {c}"),
            Self::ModelNotRegistered(c) => write!(f, "no ONNX model registered for {c}"),
            Self::Ort(e) => write!(f, "onnxruntime error: {e}"),
            Self::InvalidPayload(e) => write!(f, "invalid payload: {e}"),
            Self::Zstd(e) => write!(f, "zstd error: {e}"),
        }
    }
}
impl std::error::Error for DecodeError {}

/// Decode a parsed WAI1 envelope using the neural runtime.
pub fn decode_envelope(env: &Wai, registry: &ModelRegistry) -> Result<Decoded, DecodeError> {
    let cap = &env.manifest.model_requirement.capability;
    let path = registry.get(cap)
        .ok_or_else(|| DecodeError::ModelNotRegistered(cap.clone()))?;
    match cap.as_str() {
        "wai.neural.encodec32"    => audio::decode(&env.payload, path, 32_000).map(Decoded::Audio),
        "wai.neural.dac"          => audio::decode(&env.payload, path, 44_100).map(Decoded::Audio),
        "wai.neural.mimi"         => audio::decode(&env.payload, path, 24_000).map(Decoded::Audio),
        "wai.neural.wavtokenizer" => audio::decode(&env.payload, path, 24_000).map(Decoded::Audio),
        "wai.neural.bmshj2018"        => image::decode(&env.payload, path).map(Decoded::Image),
        "wai.neural.video_bmshj2018"  => video::decode(&env.payload, path).map(Decoded::Video),
        other => Err(DecodeError::UnknownCapability(other.to_string())),
    }
}
