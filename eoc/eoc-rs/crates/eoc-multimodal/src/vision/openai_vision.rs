//! OpenAI vision-language backend.
//!
//! Endpoint: `POST /v1/chat/completions`. Vision is enabled by passing a
//! `messages[0].content` array that mixes `{"type": "text"}` and
//! `{"type": "image_url"}` parts. Models known to accept image content:
//!
//! * `gpt-4o`, `gpt-4o-mini`
//! * `gpt-4.1`
//! * `o3`
//!
//! Joule cost is delegated to [`eoc_vendor_api::DefaultEstimator`] so the
//! same per-model coefficients used for text inference apply here.

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use eoc_vendor_api::joule_estimator::{DefaultEstimator, JouleEstimator};
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, field};

use crate::error::{MultimodalError, MultimodalResult};
use crate::modality::{ImageRef, MultimodalQuery, QueryPart};

/// Default OpenAI chat-completions endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

/// Vision-capable OpenAI chat backend.
pub struct OpenAiVisionBackend {
    client: reqwest::Client,
    api_key: String,
    model: String,
    endpoint: String,
    estimator: DefaultEstimator,
    max_tokens: u32,
}

impl OpenAiVisionBackend {
    /// Construct with an API key and target vision model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            estimator: DefaultEstimator::builtin(),
            max_tokens: 1024,
        }
    }

    /// Override the endpoint (used for wiremock tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override `max_tokens`.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Build the request body for snapshot testing. The body is one user
    /// message whose `content` is an array of typed parts; OpenAI accepts
    /// either `image_url` (URL) or a `data:` URI base64 form.
    pub fn build_request_body(&self, q: &MultimodalQuery) -> MultimodalResult<serde_json::Value> {
        let mut content = Vec::<serde_json::Value>::new();
        for part in &q.parts {
            match part {
                QueryPart::Text(t) => {
                    content.push(json!({"type": "text", "text": t}));
                }
                QueryPart::Image(img) => {
                    let url = encode_image_url(img)?;
                    content.push(json!({
                        "type": "image_url",
                        "image_url": {"url": url}
                    }));
                }
                QueryPart::Audio(_) | QueryPart::Video(_) => {
                    return Err(MultimodalError::Unsupported(part.modality()));
                }
            }
        }

        Ok(json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}]
        }))
    }

    /// Send the query and return a [`Response`].
    pub async fn infer_multimodal(&self, q: &MultimodalQuery) -> MultimodalResult<Response> {
        debug!(
            target: "openai_vision.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching openai vision inference"
        );

        let body = self.build_request_body(q)?;
        let resp = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(classify(status, body));
        }
        let parsed: ChatCompletionsResponse = resp.json().await?;
        let text = parsed
            .choices
            .into_iter()
            .map(|c| c.message.content)
            .collect::<Vec<_>>()
            .join("");
        let usage = parsed.usage.unwrap_or_default();
        let cost =
            self.estimator
                .estimate(usage.prompt_tokens, usage.completion_tokens, &self.model);
        Ok(Response::new(q.id, text, Stage::Neural, cost))
    }
}

fn encode_image_url(img: &ImageRef) -> MultimodalResult<String> {
    match img {
        ImageRef::Url(u) => Ok(u.clone()),
        _ => {
            let (ct, b64) = img.to_base64()?;
            Ok(format!("data:{ct};base64,{b64}"))
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
struct ChatCompletionsResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize, Default, Clone, Copy)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Adapter that lets [`OpenAiVisionBackend`] participate in the text-only
/// [`NeuralBackend`] trait — text parts go through, image parts must be
/// attached via metadata for this path; for full multi-modal use call
/// [`OpenAiVisionBackend::infer_multimodal`] directly.
#[async_trait]
impl NeuralBackend for OpenAiVisionBackend {
    async fn infer(&self, q: &Query) -> Response {
        let mm = MultimodalQuery::text(&q.prompt);
        match self.infer_multimodal(&mm).await {
            Ok(r) => Response::new(q.id, r.payload, Stage::Neural, r.joule_cost),
            Err(e) => Response::new(
                q.id,
                format!("[openai-vision-error: {e}]"),
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
    fn body_carries_text_and_image_parts() {
        let b = OpenAiVisionBackend::new("sk", "gpt-4o");
        let q = MultimodalQuery::new(vec![
            QueryPart::Text("what is in this image?".to_string()),
            QueryPart::Image(ImageRef::Url("https://example.com/x.png".to_string())),
        ]);
        let body = b.build_request_body(&q).expect("ok");
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["messages"][0]["role"], "user");
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what is in this image?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn body_data_uri_for_bytes() {
        let b = OpenAiVisionBackend::new("sk", "gpt-4o-mini");
        let q = MultimodalQuery::new(vec![QueryPart::Image(ImageRef::Bytes {
            content_type: "image/png".to_string(),
            bytes: vec![1, 2, 3],
        })]);
        let body = b.build_request_body(&q).expect("ok");
        let url = body["messages"][0]["content"][0]["image_url"]["url"]
            .as_str()
            .expect("url string");
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn audio_part_rejected() {
        let b = OpenAiVisionBackend::new("sk", "gpt-4o");
        let q = MultimodalQuery::new(vec![QueryPart::Audio(crate::AudioRef::Url(
            "u".to_string(),
        ))]);
        let err = b.build_request_body(&q).expect_err("rejected");
        assert!(matches!(err, MultimodalError::Unsupported(_)));
    }
}
