//! Fireworks.ai chat-completions backend (OpenAI-compatible schema).

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use tracing::field;

use crate::auth::Auth;
use crate::config::VendorConfig;
use crate::error::VendorResult;
use crate::openai_compat;

/// Default Fireworks endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.fireworks.ai/inference/v1/chat/completions";

/// Fireworks chat backend.
pub struct FireworksBackend {
    client: reqwest::Client,
    auth: Auth,
    model: String,
    stream: bool,
    config: VendorConfig,
}

impl FireworksBackend {
    /// Construct with API key + model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth: Auth::Bearer(api_key.into()),
            model: model.into(),
            stream: true,
            config: VendorConfig::new(),
        }
    }

    /// Override the [`VendorConfig`].
    pub fn with_config(mut self, config: VendorConfig) -> Self {
        self.config = config;
        self
    }

    /// Disable SSE streaming.
    pub fn without_stream(mut self) -> Self {
        self.stream = false;
        self
    }

    fn endpoint(&self) -> &str {
        self.config
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
    }

    async fn try_infer(&self, q: &Query) -> VendorResult<Response> {
        tracing::debug!(
            target: "fireworks.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching fireworks inference"
        );
        openai_compat::execute(
            &self.client,
            self.endpoint(),
            &self.auth,
            &self.model,
            q,
            self.stream,
            &self.config,
        )
        .await
    }
}

#[async_trait]
impl NeuralBackend for FireworksBackend {
    async fn infer(&self, q: &Query) -> Response {
        match self.try_infer(q).await {
            Ok(r) => r,
            Err(e) => Response::new(
                q.id,
                format!("[fireworks-error: {e}]"),
                Stage::Neural,
                JouleCost { microjoules: 0, source: JouleSource::Estimated },
            ),
        }
    }
}
