//! Anthropic vision-language backend.
//!
//! Endpoint: `POST /v1/messages`. Vision is enabled by passing `content`
//! blocks of type `image` whose `source` is either
//! `{"type": "base64", "media_type": "image/png", "data": "..."}` or
//! `{"type": "url", "url": "..."}`. Models known to accept image content:
//!
//! * `claude-3-5-sonnet`
//! * `claude-3-opus`
//! * `claude-3-haiku`

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use eoc_vendor_api::joule_estimator::{DefaultEstimator, JouleEstimator};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, field};

use crate::error::{MultimodalError, MultimodalResult};
use crate::modality::{ImageRef, MultimodalQuery, QueryPart};

/// Default Anthropic messages endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
/// Default Anthropic API version header.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Vision-capable Anthropic messages backend.
pub struct AnthropicVisionBackend {
    client: reqwest::Client,
    api_key: String,
    model: String,
    endpoint: String,
    max_tokens: u32,
    estimator: DefaultEstimator,
}

impl AnthropicVisionBackend {
    /// Construct with an API key and target vision model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            max_tokens: 1024,
            estimator: DefaultEstimator::builtin(),
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

    /// Build the request body for snapshot testing.
    pub fn build_request_body(&self, q: &MultimodalQuery) -> MultimodalResult<serde_json::Value> {
        let mut content = Vec::<serde_json::Value>::new();
        for part in &q.parts {
            match part {
                QueryPart::Text(t) => {
                    content.push(json!({"type": "text", "text": t}));
                }
                QueryPart::Image(img) => content.push(image_block(img)?),
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

    /// Run multi-modal inference.
    pub async fn infer_multimodal(&self, q: &MultimodalQuery) -> MultimodalResult<Response> {
        debug!(
            target: "anthropic_vision.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching anthropic vision inference"
        );

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().expect("static"));
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        let key = HeaderValue::from_str(&self.api_key)
            .map_err(|e| MultimodalError::Decode(e.to_string()))?;
        headers.insert(HeaderName::from_static("x-api-key"), key);

        let body = self.build_request_body(q)?;
        let resp = self
            .client
            .post(&self.endpoint)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(classify(status, body));
        }

        let parsed: AnthropicResponse = resp.json().await?;
        let text = parsed
            .content
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");
        let usage = parsed.usage.unwrap_or_default();
        let cost = self
            .estimator
            .estimate(usage.input_tokens, usage.output_tokens, &self.model);
        Ok(Response::new(q.id, text, Stage::Neural, cost))
    }
}

fn image_block(img: &ImageRef) -> MultimodalResult<serde_json::Value> {
    match img {
        ImageRef::Url(u) => Ok(json!({
            "type": "image",
            "source": {"type": "url", "url": u}
        })),
        _ => {
            let (ct, b64) = img.to_base64()?;
            Ok(json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": ct,
                    "data": b64,
                }
            }))
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
struct AnthropicResponse {
    content: Vec<TextBlock>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct TextBlock {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize, Default, Clone, Copy)]
struct Usage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

#[async_trait]
impl NeuralBackend for AnthropicVisionBackend {
    async fn infer(&self, q: &Query) -> Response {
        let mm = MultimodalQuery::text(&q.prompt);
        match self.infer_multimodal(&mm).await {
            Ok(r) => Response::new(q.id, r.payload, Stage::Neural, r.joule_cost),
            Err(e) => Response::new(
                q.id,
                format!("[anthropic-vision-error: {e}]"),
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
    fn body_uses_image_block_for_bytes() {
        let b = AnthropicVisionBackend::new("k", "claude-3-5-sonnet");
        let q = MultimodalQuery::new(vec![
            QueryPart::Text("describe".to_string()),
            QueryPart::Image(ImageRef::Bytes {
                content_type: "image/jpeg".to_string(),
                bytes: vec![1, 2, 3],
            }),
        ]);
        let body = b.build_request_body(&q).expect("ok");
        let content = &body["messages"][0]["content"];
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/jpeg");
        assert!(!content[1]["source"]["data"].as_str().expect("str").is_empty());
    }

    #[test]
    fn body_uses_url_source_for_url_ref() {
        let b = AnthropicVisionBackend::new("k", "claude-3-haiku");
        let q = MultimodalQuery::new(vec![QueryPart::Image(ImageRef::Url(
            "https://example.com/x.png".to_string(),
        ))]);
        let body = b.build_request_body(&q).expect("ok");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["source"]["type"], "url");
        assert_eq!(block["source"]["url"], "https://example.com/x.png");
    }
}
