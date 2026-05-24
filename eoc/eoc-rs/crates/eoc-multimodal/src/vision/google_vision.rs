//! Google Gemini vision-language backend.
//!
//! Endpoint: `POST /v1beta/models/{model}:generateContent` with auth
//! supplied via the `?key=` query parameter. Vision is enabled by passing
//! `inlineData` parts whose `mimeType` + `data` fields carry the base64
//! image. Audio + video are uniformly supported through the same shape,
//! which is why Gemini is the "unified-modality" backend the router
//! prefers when both image and audio are present.
//!
//! Models known to accept image content:
//!
//! * `gemini-2.0-pro`
//! * `gemini-2.0-flash`
//! * `gemini-1.5-pro`

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use eoc_vendor_api::joule_estimator::{DefaultEstimator, JouleEstimator};
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, field};

use crate::error::{MultimodalError, MultimodalResult};
use crate::modality::{AudioRef, ImageRef, MultimodalQuery, QueryPart, VideoRef};

/// Default Google base URL.
pub const DEFAULT_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Vision/audio-capable Gemini backend.
pub struct GoogleVisionBackend {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base: String,
    estimator: DefaultEstimator,
}

impl GoogleVisionBackend {
    /// Construct with API key + model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base: DEFAULT_BASE.to_string(),
            estimator: DefaultEstimator::builtin(),
        }
    }

    /// Override the base URL — used for wiremock tests where the path is
    /// just `/`.
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    fn endpoint(&self) -> String {
        // For test endpoints, just append `?key=` to whatever was given.
        if self.base.starts_with("http://127.0.0.1") || self.base.starts_with("http://localhost") {
            return format!("{}?key={}", self.base, self.api_key);
        }
        format!(
            "{}/{}:generateContent?key={}",
            self.base, self.model, self.api_key
        )
    }

    /// Build the Gemini request body. Public for snapshot testing.
    pub fn build_request_body(&self, q: &MultimodalQuery) -> MultimodalResult<serde_json::Value> {
        let mut parts = Vec::<serde_json::Value>::new();
        for part in &q.parts {
            match part {
                QueryPart::Text(t) => parts.push(json!({"text": t})),
                QueryPart::Image(img) => parts.push(image_part(img)?),
                QueryPart::Audio(audio) => parts.push(audio_part(audio)?),
                QueryPart::Video(video) => parts.push(video_part(video)?),
            }
        }
        Ok(json!({
            "contents": [{"role": "user", "parts": parts}]
        }))
    }

    /// Run multi-modal inference.
    pub async fn infer_multimodal(&self, q: &MultimodalQuery) -> MultimodalResult<Response> {
        debug!(
            target: "google_vision.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching gemini multimodal inference"
        );

        let body = self.build_request_body(q)?;
        let resp = self
            .client
            .post(self.endpoint())
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(classify(status, body));
        }

        let parsed: GeminiResponse = resp.json().await?;
        let text = parsed
            .candidates
            .into_iter()
            .flat_map(|c| c.content.parts)
            .map(|p| p.text)
            .collect::<Vec<_>>()
            .join("");
        let usage = parsed.usage_metadata.unwrap_or_default();
        let cost = self.estimator.estimate(
            usage.prompt_token_count,
            usage.candidates_token_count,
            &self.model,
        );
        Ok(Response::new(q.id, text, Stage::Neural, cost))
    }
}

fn image_part(img: &ImageRef) -> MultimodalResult<serde_json::Value> {
    match img {
        ImageRef::Url(u) => Ok(json!({
            "fileData": {"mimeType": "image/png", "fileUri": u}
        })),
        _ => {
            let (ct, b64) = img.to_base64()?;
            Ok(json!({
                "inlineData": {"mimeType": ct, "data": b64}
            }))
        }
    }
}

fn audio_part(audio: &AudioRef) -> MultimodalResult<serde_json::Value> {
    match audio {
        AudioRef::Url(u) => Ok(json!({
            "fileData": {"mimeType": "audio/wav", "fileUri": u}
        })),
        AudioRef::Bytes { content_type, .. } => {
            let bytes_ref = audio.to_bytes()?;
            let b64 = base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &bytes_ref.1,
            );
            Ok(json!({
                "inlineData": {"mimeType": content_type, "data": b64}
            }))
        }
        AudioRef::Base64(s) => Ok(json!({
            "inlineData": {"mimeType": "audio/wav", "data": s}
        })),
        AudioRef::File(_) => {
            let (ct, bytes) = audio.to_bytes()?;
            let b64 =
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
            Ok(json!({"inlineData": {"mimeType": ct, "data": b64}}))
        }
    }
}

fn video_part(video: &VideoRef) -> MultimodalResult<serde_json::Value> {
    match video {
        VideoRef::Url(u) => Ok(json!({
            "fileData": {"mimeType": "video/mp4", "fileUri": u}
        })),
        _ => {
            let (ct, bytes) = video.to_bytes()?;
            let b64 =
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
            Ok(json!({"inlineData": {"mimeType": ct, "data": b64}}))
        }
    }
}

fn classify(status: reqwest::StatusCode, body: String) -> MultimodalError {
    let truncated: String = body.chars().take(512).collect();
    match status.as_u16() {
        401 | 403 => MultimodalError::InvalidApiKey,
        429 => MultimodalError::RateLimited { retry_after_secs: None },
        404 => MultimodalError::ModelNotFound(truncated),
        s => MultimodalError::Vendor { status: s, body: truncated },
    }
}

#[derive(Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Content,
}

#[derive(Deserialize)]
struct Content {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Deserialize)]
struct Part {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize, Default, Clone, Copy)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
}

#[async_trait]
impl NeuralBackend for GoogleVisionBackend {
    async fn infer(&self, q: &Query) -> Response {
        let mm = MultimodalQuery::text(&q.prompt);
        match self.infer_multimodal(&mm).await {
            Ok(r) => Response::new(q.id, r.payload, Stage::Neural, r.joule_cost),
            Err(e) => Response::new(
                q.id,
                format!("[google-vision-error: {e}]"),
                Stage::Neural,
                JouleCost { microjoules: 0, source: JouleSource::Estimated },
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_uses_inline_data_for_bytes() {
        let b = GoogleVisionBackend::new("k", "gemini-1.5-pro");
        let q = MultimodalQuery::new(vec![
            QueryPart::Text("describe".to_string()),
            QueryPart::Image(ImageRef::Bytes {
                content_type: "image/png".to_string(),
                bytes: vec![10, 20, 30],
            }),
        ]);
        let body = b.build_request_body(&q).expect("ok");
        let parts = &body["contents"][0]["parts"];
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert!(
            !parts[1]["inlineData"]["data"]
                .as_str()
                .expect("data str")
                .is_empty()
        );
    }

    #[test]
    fn body_handles_audio_part() {
        let b = GoogleVisionBackend::new("k", "gemini-2.0-flash");
        let q = MultimodalQuery::new(vec![QueryPart::Audio(AudioRef::Bytes {
            content_type: "audio/wav".to_string(),
            bytes: vec![0xFF, 0xFE],
        })]);
        let body = b.build_request_body(&q).expect("ok");
        let part = &body["contents"][0]["parts"][0];
        assert_eq!(part["inlineData"]["mimeType"], "audio/wav");
    }
}
