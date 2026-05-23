//! Groq chat-completions backend (OpenAI-compatible schema).
//!
//! Groq's LPU has dramatically lower per-token energy than a generic GPU
//! deployment; the backend exposes [`GroqBackend::with_lpu_coefficient`]
//! to override the default joule estimator with Groq-specific numbers.

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use tracing::field;

use crate::auth::Auth;
use crate::config::VendorConfig;
use crate::error::VendorResult;
use crate::joule_estimator::DefaultEstimator;
use crate::openai_compat;

/// Default Groq endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.groq.com/openai/v1/chat/completions";

/// Default LPU coefficients — order-of-magnitude lower than GPU.
/// Source: Groq published throughput numbers (~750 tokens/sec/W class).
pub const DEFAULT_LPU_INPUT_J: f64 = 0.001;
/// Default LPU joules per output token.
pub const DEFAULT_LPU_OUTPUT_J: f64 = 0.005;

/// Groq chat backend.
pub struct GroqBackend {
    client: reqwest::Client,
    auth: Auth,
    model: String,
    stream: bool,
    config: VendorConfig,
}

impl GroqBackend {
    /// Construct with API key + target model (e.g. `llama-3.1-70b`).
    /// Installs LPU-tuned joule fallback coefficients by default.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let est =
            DefaultEstimator::builtin().with_fallback(DEFAULT_LPU_INPUT_J, DEFAULT_LPU_OUTPUT_J);
        let config = VendorConfig::new().with_joule_estimator(Box::new(est));
        Self {
            client: reqwest::Client::new(),
            auth: Auth::Bearer(api_key.into()),
            model: model.into(),
            stream: true,
            config,
        }
    }

    /// Override the LPU joule coefficients (consumes `self`).
    pub fn with_lpu_coefficient(mut self, input_j: f64, output_j: f64) -> Self {
        let est = DefaultEstimator::builtin().with_fallback(input_j, output_j);
        self.config = self.config.with_joule_estimator(Box::new(est));
        self
    }

    /// Override the full [`VendorConfig`].
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
            target: "groq.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching groq inference"
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
impl NeuralBackend for GroqBackend {
    async fn infer(&self, q: &Query) -> Response {
        match self.try_infer(q).await {
            Ok(r) => r,
            Err(e) => Response::new(
                q.id,
                format!("[groq-error: {e}]"),
                Stage::Neural,
                JouleCost { microjoules: 0, source: JouleSource::Estimated },
            ),
        }
    }
}
