//! Vendor audio-transcription backends.
//!
//! Two paths are supported through the same [`Transcriber`] trait:
//!
//! * **OpenAI Whisper** — `POST /v1/audio/transcriptions` with a multipart
//!   form body. Default model is `whisper-1`.
//! * **Gemini audio** — Gemini accepts audio inline via `inlineData`. The
//!   transcription endpoint is the regular `generateContent` call with a
//!   `"Transcribe this audio"` prompt. This is implemented via
//!   [`crate::vision::GoogleVisionBackend`].
//!
//! The [`TranscriptionResult`] type carries the recognised text, the
//! detected language, segment timestamps, and joule cost.

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource};
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

use crate::error::{MultimodalError, MultimodalResult};
use crate::modality::AudioRef;

/// One Whisper segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Recognised text within the segment.
    pub text: String,
}

/// A transcription result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// The full transcript.
    pub text: String,
    /// Detected language (BCP-47 / ISO 639-1), when the vendor reports one.
    pub language: Option<String>,
    /// Optional segment timestamps.
    pub segments: Vec<Segment>,
    /// Energy attributable to this transcription.
    pub joule_cost: JouleCost,
}

/// Audio → text backend.
#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Transcribe `audio` and return the resulting text plus metadata.
    async fn transcribe(&self, audio: &AudioRef) -> MultimodalResult<TranscriptionResult>;
}

/// OpenAI Whisper API backend (`/v1/audio/transcriptions`).
pub struct WhisperApiBackend {
    client: reqwest::Client,
    api_key: String,
    model: String,
    endpoint: String,
    /// Synthetic joules-per-second-of-audio coefficient.
    pub joules_per_second: f64,
    /// Optional language hint (forwarded as the `language` form field).
    pub language_hint: Option<String>,
}

/// Default OpenAI audio transcription endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/audio/transcriptions";

impl WhisperApiBackend {
    /// Construct with API key + model (e.g. `whisper-1`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            joules_per_second: 5.0, // ~5 J/s on a generic ASR GPU.
            language_hint: None,
        }
    }

    /// Override the endpoint (used for wiremock tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the joules-per-second coefficient used for cost estimation.
    pub fn with_joules_per_second(mut self, j: f64) -> Self {
        self.joules_per_second = j;
        self
    }

    /// Set a language hint (BCP-47 code).
    pub fn with_language_hint(mut self, lang: impl Into<String>) -> Self {
        self.language_hint = Some(lang.into());
        self
    }
}

#[derive(Deserialize)]
struct WhisperResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f32>,
    #[serde(default)]
    segments: Vec<WhisperSegment>,
}

#[derive(Deserialize)]
struct WhisperSegment {
    #[serde(default)]
    start: f32,
    #[serde(default)]
    end: f32,
    #[serde(default)]
    text: String,
}

#[async_trait]
impl Transcriber for WhisperApiBackend {
    async fn transcribe(&self, audio: &AudioRef) -> MultimodalResult<TranscriptionResult> {
        let (content_type, bytes) = audio.to_bytes()?;
        let part = Part::bytes(bytes)
            .file_name(file_name_for(&content_type))
            .mime_str(&content_type)
            .map_err(|e| MultimodalError::Decode(e.to_string()))?;
        let mut form = Form::new()
            .text("model", self.model.clone())
            .text("response_format", "verbose_json")
            .part("file", part);
        if let Some(lang) = &self.language_hint {
            form = form.text("language", lang.clone());
        }
        let resp = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let truncated: String = body.chars().take(512).collect();
            return Err(match status.as_u16() {
                401 | 403 => MultimodalError::InvalidApiKey,
                429 => MultimodalError::RateLimited { retry_after_secs: None },
                404 => MultimodalError::ModelNotFound(truncated),
                s => MultimodalError::Vendor { status: s, body: truncated },
            });
        }
        let parsed: WhisperResponse = resp.json().await?;
        let segments = parsed
            .segments
            .into_iter()
            .map(|s| Segment {
                start: s.start,
                end: s.end,
                text: s.text,
            })
            .collect::<Vec<_>>();
        let duration = parsed
            .duration
            .or_else(|| segments.last().map(|s| s.end))
            .unwrap_or(0.0);
        let microjoules =
            ((duration as f64) * self.joules_per_second * 1_000_000.0).max(0.0) as u64;
        Ok(TranscriptionResult {
            text: parsed.text,
            language: parsed.language,
            segments,
            joule_cost: JouleCost {
                microjoules,
                source: JouleSource::Estimated,
            },
        })
    }
}

fn file_name_for(content_type: &str) -> String {
    match content_type {
        "audio/wav" => "audio.wav".to_string(),
        "audio/mp3" | "audio/mpeg" => "audio.mp3".to_string(),
        "audio/ogg" => "audio.ogg".to_string(),
        "audio/webm" => "audio.webm".to_string(),
        "audio/flac" => "audio.flac".to_string(),
        _ => "audio.bin".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_picks_extension() {
        assert_eq!(file_name_for("audio/wav"), "audio.wav");
        assert_eq!(file_name_for("audio/mp3"), "audio.mp3");
        assert_eq!(file_name_for("application/octet-stream"), "audio.bin");
    }

    #[test]
    fn builder_setters() {
        let b = WhisperApiBackend::new("k", "whisper-1")
            .with_joules_per_second(2.5)
            .with_language_hint("en");
        assert_eq!(b.model, "whisper-1");
        assert!((b.joules_per_second - 2.5).abs() < 1e-9);
        assert_eq!(b.language_hint.as_deref(), Some("en"));
    }
}
