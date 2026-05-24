//! Local Whisper backend (feature `local`).
//!
//! Loads ONNX-exported Whisper checkpoints — `whisper-tiny.en`,
//! `whisper-base.en`, `whisper-small.en`, `whisper-medium`, and
//! `whisper-large-v3-turbo`. The actual `ort::Session::run` call is left
//! to the deployment-time integrator; this module ships the trait surface,
//! checkpoint registry, and joule estimator so the call site is stable.

use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource};

use crate::audio::whisper_api::{Segment, Transcriber, TranscriptionResult};
use crate::error::MultimodalResult;
use crate::modality::AudioRef;

/// Local Whisper backend.
pub struct WhisperLocalBackend {
    /// Path to the ONNX checkpoint on disk.
    pub model_path: PathBuf,
    /// Canonical model name.
    pub model_name: String,
    /// Joules per second of audio. Smaller models are cheaper.
    pub joules_per_second: f64,
    /// Lazily-initialised ONNX session, mutex-guarded.
    session: Mutex<Option<ort::session::Session>>,
}

impl WhisperLocalBackend {
    /// Construct from a `(model_name, model_path)` pair. `model_name`
    /// must be one of the known Whisper checkpoint identifiers; unknown
    /// names take the small-model joule coefficient as a default.
    pub fn new(model_name: impl Into<String>, model_path: PathBuf) -> Self {
        let model_name = model_name.into();
        let joules_per_second = match model_name.as_str() {
            "whisper-tiny.en" => 0.05,
            "whisper-base.en" => 0.10,
            "whisper-small.en" => 0.25,
            "whisper-medium" => 0.60,
            "whisper-large-v3-turbo" => 1.20,
            _ => 0.25,
        };
        Self {
            model_path,
            model_name,
            joules_per_second,
            session: Mutex::new(None),
        }
    }
}

#[async_trait]
impl Transcriber for WhisperLocalBackend {
    async fn transcribe(&self, audio: &AudioRef) -> MultimodalResult<TranscriptionResult> {
        let (_ct, bytes) = audio.to_bytes()?;
        // Estimate duration as bytes / 32_000 — a reasonable proxy for a
        // 16-bit, 16 kHz PCM stream and good enough for joule attribution
        // when the real decoder is not wired up.
        let duration = (bytes.len() as f32) / 32_000.0;
        let microjoules =
            ((duration as f64) * self.joules_per_second * 1_000_000.0).max(0.0) as u64;
        // Place-holder until the integrator wires session.run().
        let _ = &self.session;
        let _ = &self.model_path;
        Ok(TranscriptionResult {
            text: String::new(),
            language: None,
            segments: Vec::<Segment>::new(),
            joule_cost: JouleCost {
                microjoules,
                source: JouleSource::Estimated,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_by_size() {
        let tiny = WhisperLocalBackend::new("whisper-tiny.en", PathBuf::from("x.onnx"));
        let large = WhisperLocalBackend::new("whisper-large-v3-turbo", PathBuf::from("x.onnx"));
        assert!(tiny.joules_per_second < large.joules_per_second);
    }
}
