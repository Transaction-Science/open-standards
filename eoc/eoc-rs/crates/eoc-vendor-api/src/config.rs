//! Per-backend configuration: endpoint override, timeout, retry policy,
//! pluggable joule estimator.

use std::time::Duration;

use crate::joule_estimator::{DefaultEstimator, JouleEstimator};

/// Retry behaviour for transient vendor failures.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum retry attempts after the initial request.
    pub max_retries: u32,
    /// Base back-off applied between attempts.
    pub initial_backoff: Duration,
    /// Multiplier applied to the back-off on each subsequent attempt.
    pub backoff_factor: f64,
}

impl RetryPolicy {
    /// A reasonable default: 3 attempts, exponential back-off starting at
    /// 250 ms.
    pub fn default_policy() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(250),
            backoff_factor: 2.0,
        }
    }

    /// Disable retries entirely.
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            initial_backoff: Duration::from_millis(0),
            backoff_factor: 1.0,
        }
    }

    /// Compute the back-off for retry attempt `attempt` (zero-indexed).
    pub fn backoff_for(&self, attempt: u32) -> Duration {
        let ms = self.initial_backoff.as_millis() as f64 * self.backoff_factor.powi(attempt as i32);
        Duration::from_millis(ms as u64)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::default_policy()
    }
}

/// Shared configuration for any vendor backend.
pub struct VendorConfig {
    /// Override the default endpoint — used by tests pointing at a
    /// `wiremock` server, or by self-hosted proxies.
    pub endpoint: Option<String>,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Retry policy.
    pub retry_policy: RetryPolicy,
    /// Pluggable joule estimator. Defaults to [`DefaultEstimator`].
    pub joule_estimator: Box<dyn JouleEstimator>,
}

impl VendorConfig {
    /// Default configuration — 60-second timeout, default retry policy,
    /// default joule estimator.
    pub fn new() -> Self {
        Self {
            endpoint: None,
            timeout: Duration::from_secs(60),
            retry_policy: RetryPolicy::default_policy(),
            joule_estimator: Box::new(DefaultEstimator::builtin()),
        }
    }

    /// Override the endpoint (consumes `self`).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Override the timeout (consumes `self`).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the retry policy (consumes `self`).
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Override the joule estimator (consumes `self`).
    pub fn with_joule_estimator(mut self, est: Box<dyn JouleEstimator>) -> Self {
        self.joule_estimator = est;
        self
    }
}

impl Default for VendorConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for VendorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VendorConfig")
            .field("endpoint", &self.endpoint)
            .field("timeout", &self.timeout)
            .field("retry_policy", &self.retry_policy)
            .field("joule_estimator", &"<dyn JouleEstimator>")
            .finish()
    }
}
