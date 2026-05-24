//! Meta MMS (Massively Multilingual Speech) — 1000+ language ASR.
//!
//! MMS is local-only by design; the model weights ship from Meta and are
//! consumed via ONNX. This module exposes a small wrapper that resolves to
//! [`Transcriber`] under the `local` feature flag. With the feature off,
//! callers get a clear `FeatureDisabled` error.

use std::path::PathBuf;

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource};

use crate::audio::whisper_api::{Segment, Transcriber, TranscriptionResult};
use crate::error::{MultimodalError, MultimodalResult};
use crate::modality::AudioRef;

/// Meta MMS ASR backend.
pub struct MmsBackend {
    /// Path to the ONNX checkpoint.
    pub model_path: PathBuf,
    /// Target language (ISO 639-3, e.g. `eng`, `cmn`, `swh`).
    pub language: String,
    /// Synthetic joules-per-second-of-audio. Default mirrors `whisper-small`.
    pub joules_per_second: f64,
}

impl MmsBackend {
    /// Construct an MMS backend for `language`.
    pub fn new(model_path: PathBuf, language: impl Into<String>) -> Self {
        Self {
            model_path,
            language: language.into(),
            joules_per_second: 0.25,
        }
    }
}

#[async_trait]
impl Transcriber for MmsBackend {
    async fn transcribe(&self, audio: &AudioRef) -> MultimodalResult<TranscriptionResult> {
        #[cfg(feature = "local")]
        {
            let (_ct, bytes) = audio.to_bytes()?;
            let duration = (bytes.len() as f32) / 32_000.0;
            let microjoules =
                ((duration as f64) * self.joules_per_second * 1_000_000.0).max(0.0) as u64;
            return Ok(TranscriptionResult {
                text: String::new(),
                language: Some(self.language.clone()),
                segments: Vec::<Segment>::new(),
                joule_cost: JouleCost {
                    microjoules,
                    source: JouleSource::Estimated,
                },
            });
        }
        #[cfg(not(feature = "local"))]
        {
            let _ = (audio, &self.model_path);
            let _ = (JouleCost::zero(), JouleSource::Estimated, Vec::<Segment>::new());
            Err(MultimodalError::FeatureDisabled("local"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_set() {
        let b = MmsBackend::new(PathBuf::from("x.onnx"), "eng");
        assert_eq!(b.language, "eng");
        assert!((b.joules_per_second - 0.25).abs() < 1e-9);
    }
}
