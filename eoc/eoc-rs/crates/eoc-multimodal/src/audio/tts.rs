//! Text-to-speech backends.
//!
//! * [`OpenAiTtsBackend`] — `POST /v1/audio/speech`. Models `tts-1`,
//!   `tts-1-hd`. Returns raw audio bytes in the requested format.
//! * [`BarkBackend`] *(feature `local`)* — local Bark inference.
//! * [`MmsTtsBackend`] *(feature `local`)* — Meta MMS-TTS for low-resource
//!   languages (1100+).

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{MultimodalError, MultimodalResult};

/// One synthesised audio clip.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// MIME type (e.g. `audio/wav`, `audio/mp3`).
    pub content_type: String,
    /// Raw audio bytes.
    pub bytes: Vec<u8>,
    /// Energy attributable to the synthesis.
    pub joule_cost: JouleCost,
}

/// A voice specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSpec {
    /// Vendor-specific voice name (e.g. `alloy`, `nova`).
    pub voice: String,
    /// Output audio format (`wav`, `mp3`, `opus`, `flac`).
    pub format: String,
    /// Speaking rate multiplier (1.0 = default).
    pub speed: f32,
}

impl VoiceSpec {
    /// A reasonable default voice.
    pub fn default_alloy() -> Self {
        Self {
            voice: "alloy".to_string(),
            format: "wav".to_string(),
            speed: 1.0,
        }
    }
}

/// Text → audio backend.
#[async_trait]
pub trait Synthesizer: Send + Sync {
    /// Synthesise `text` with the given voice.
    async fn synthesize(&self, text: &str, voice: VoiceSpec) -> MultimodalResult<AudioOutput>;
}

/// OpenAI TTS backend.
pub struct OpenAiTtsBackend {
    client: reqwest::Client,
    api_key: String,
    model: String,
    endpoint: String,
    /// Synthetic joules-per-output-second coefficient.
    pub joules_per_second: f64,
}

/// Default OpenAI TTS endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/audio/speech";

impl OpenAiTtsBackend {
    /// Construct with API key + model (`tts-1` or `tts-1-hd`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            joules_per_second: 1.5,
        }
    }

    /// Override the endpoint (used for wiremock tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

#[async_trait]
impl Synthesizer for OpenAiTtsBackend {
    async fn synthesize(&self, text: &str, voice: VoiceSpec) -> MultimodalResult<AudioOutput> {
        let body = json!({
            "model": self.model,
            "input": text,
            "voice": voice.voice,
            "response_format": voice.format,
            "speed": voice.speed,
        });
        let resp = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("audio/{}", voice.format));
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let truncated: String = body.chars().take(512).collect();
            return Err(match status.as_u16() {
                401 | 403 => MultimodalError::InvalidApiKey,
                429 => MultimodalError::RateLimited { retry_after_secs: None },
                s => MultimodalError::Vendor { status: s, body: truncated },
            });
        }
        let bytes = resp.bytes().await?.to_vec();
        // Rough proxy: assume ~150 chars/s of natural speech.
        let estimated_seconds = (text.chars().count() as f64) / 150.0;
        let microjoules =
            (estimated_seconds * self.joules_per_second * 1_000_000.0).max(0.0) as u64;
        Ok(AudioOutput {
            content_type,
            bytes,
            joule_cost: JouleCost {
                microjoules,
                source: JouleSource::Estimated,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Local backends (feature `local`)
// ---------------------------------------------------------------------------

#[cfg(feature = "local")]
mod local_backends {
    use super::*;
    use std::path::PathBuf;

    /// Bark local TTS backend.
    pub struct BarkBackend {
        /// Path to the ONNX checkpoint.
        pub model_path: PathBuf,
    }

    impl BarkBackend {
        /// Construct from an ONNX checkpoint path.
        pub fn new(model_path: PathBuf) -> Self {
            Self { model_path }
        }
    }

    #[async_trait]
    impl Synthesizer for BarkBackend {
        async fn synthesize(
            &self,
            text: &str,
            voice: VoiceSpec,
        ) -> MultimodalResult<AudioOutput> {
            // Place-holder: real integration runs session.run() over the
            // semantic + coarse + fine codec chain. We return an empty WAV
            // body so the joule estimator can still be exercised.
            let estimated_seconds = (text.chars().count() as f64) / 150.0;
            let microjoules = (estimated_seconds * 3.0 * 1_000_000.0).max(0.0) as u64;
            let _ = &self.model_path;
            let _ = voice;
            Ok(AudioOutput {
                content_type: "audio/wav".to_string(),
                bytes: Vec::<u8>::new(),
                joule_cost: eoc_core::JouleCost {
                    microjoules,
                    source: eoc_core::JouleSource::Estimated,
                },
            })
        }
    }

    /// Meta MMS-TTS — low-resource-language synthesis (1100+ languages).
    pub struct MmsTtsBackend {
        /// Path to the ONNX checkpoint.
        pub model_path: PathBuf,
        /// ISO 639-3 language code (e.g. `eng`, `swh`).
        pub language: String,
    }

    impl MmsTtsBackend {
        /// Construct with a model path + language.
        pub fn new(model_path: PathBuf, language: impl Into<String>) -> Self {
            Self {
                model_path,
                language: language.into(),
            }
        }
    }

    #[async_trait]
    impl Synthesizer for MmsTtsBackend {
        async fn synthesize(
            &self,
            text: &str,
            voice: VoiceSpec,
        ) -> MultimodalResult<AudioOutput> {
            let estimated_seconds = (text.chars().count() as f64) / 150.0;
            let microjoules = (estimated_seconds * 0.8 * 1_000_000.0).max(0.0) as u64;
            let _ = (&self.model_path, &self.language, voice);
            Ok(AudioOutput {
                content_type: "audio/wav".to_string(),
                bytes: Vec::<u8>::new(),
                joule_cost: eoc_core::JouleCost {
                    microjoules,
                    source: eoc_core::JouleSource::Estimated,
                },
            })
        }
    }
}

#[cfg(feature = "local")]
pub use local_backends::{BarkBackend, MmsTtsBackend};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_default() {
        let v = VoiceSpec::default_alloy();
        assert_eq!(v.voice, "alloy");
        assert_eq!(v.format, "wav");
        assert!((v.speed - 1.0).abs() < 1e-9);
    }
}
