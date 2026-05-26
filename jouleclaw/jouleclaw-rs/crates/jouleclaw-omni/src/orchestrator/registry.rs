//! Worker registry — tracks all registered inference workers.

use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::types::{
    HealthStatus, LoadedModelInfo, PipelineType, WorkerHeartbeat, WorkerRegistration, WorkerState,
};

/// Thread-safe registry of inference workers.
#[derive(Debug, Clone)]
pub struct WorkerRegistry {
    workers: Arc<DashMap<String, WorkerState>>,
}

impl WorkerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
        }
    }

    /// Register a new worker (or update an existing one).
    pub fn register(&self, reg: WorkerRegistration) -> String {
        let worker_id = reg.worker_id.clone();
        self.workers
            .insert(worker_id.clone(), WorkerState::from_registration(reg));
        tracing::info!(worker_id = %worker_id, "Worker registered");
        worker_id
    }

    /// Remove a worker from the registry.
    pub fn deregister(&self, worker_id: &str) {
        if self.workers.remove(worker_id).is_some() {
            tracing::info!(worker_id = %worker_id, "Worker deregistered");
        }
    }

    /// Update worker state from a heartbeat.
    pub fn heartbeat(&self, hb: WorkerHeartbeat) {
        if let Some(mut state) = self.workers.get_mut(&hb.worker_id) {
            state.last_heartbeat = Instant::now();
            if hb.queue_depth > 0 {
                state.last_active = Instant::now();
            }
            state.queue_depth = hb.queue_depth;
            state.registration.loaded_models = hb.loaded_models;
            state.health_status = HealthStatus::Healthy;
            state.consecutive_failures = 0;
        }
    }

    /// Record a successful health check for a worker.
    pub fn record_health_success(&self, worker_id: &str, rtt_ms: f64) {
        if let Some(mut state) = self.workers.get_mut(worker_id) {
            state.last_heartbeat = Instant::now();
            state.health_status = HealthStatus::Healthy;
            state.consecutive_failures = 0;
            // Exponential moving average for RTT
            state.avg_rtt_ms = state.avg_rtt_ms * 0.7 + rtt_ms * 0.3;
        }
    }

    /// Record a failed health check for a worker.
    pub fn record_health_failure(&self, worker_id: &str, max_failures: u32) {
        if let Some(mut state) = self.workers.get_mut(worker_id) {
            state.consecutive_failures += 1;
            if state.consecutive_failures >= max_failures {
                state.health_status = HealthStatus::Unreachable;
                tracing::warn!(
                    worker_id = %worker_id,
                    failures = state.consecutive_failures,
                    "Worker marked unreachable"
                );
            } else {
                state.health_status = HealthStatus::Degraded;
            }
        }
    }

    /// Find the best worker for a given pipeline type.
    ///
    /// Selection criteria:
    /// 1. Worker must be Healthy
    /// 2. Worker must have the required capability
    /// 3. Worker must not be loading a model
    /// 4. Prefer lowest queue depth
    /// 5. Tiebreak by lowest RTT
    pub fn find_best_worker(&self, pipeline_type: PipelineType) -> Option<(String, String)> {
        let capability = pipeline_type.capability_str();

        // For system/models routes, any healthy worker will do
        let needs_capability = !matches!(pipeline_type, PipelineType::System | PipelineType::Models);

        let mut best: Option<(String, String, u32, f64)> = None;

        for entry in self.workers.iter() {
            let state = entry.value();

            // Must be healthy
            if state.health_status != HealthStatus::Healthy {
                continue;
            }

            // Must not be loading
            if state.is_loading {
                continue;
            }

            // Must have required capability
            if needs_capability && !state.registration.capabilities.contains(&capability.to_string()) {
                continue;
            }

            let score = (state.queue_depth, state.avg_rtt_ms);
            let is_better = match &best {
                None => true,
                Some((_, _, q, rtt)) => {
                    score.0 < *q || (score.0 == *q && score.1 < *rtt)
                }
            };

            if is_better {
                best = Some((
                    state.registration.worker_id.clone(),
                    state.registration.endpoint.clone(),
                    state.queue_depth,
                    state.avg_rtt_ms,
                ));
            }
        }

        best.map(|(id, endpoint, _, _)| (id, endpoint))
    }

    /// Get all workers with a specific capability (for model management).
    pub fn workers_with_capability(&self, capability: &str) -> Vec<(String, String)> {
        self.workers
            .iter()
            .filter(|entry| {
                let state = entry.value();
                state.health_status == HealthStatus::Healthy
                    && state.registration.capabilities.contains(&capability.to_string())
            })
            .map(|entry| {
                let state = entry.value();
                (
                    state.registration.worker_id.clone(),
                    state.registration.endpoint.clone(),
                )
            })
            .collect()
    }

    /// Get all registered worker IDs and their endpoints.
    pub fn all_workers(&self) -> Vec<(String, String, HealthStatus)> {
        self.workers
            .iter()
            .map(|entry| {
                let state = entry.value();
                (
                    state.registration.worker_id.clone(),
                    state.registration.endpoint.clone(),
                    state.health_status,
                )
            })
            .collect()
    }

    /// Get all loaded models across all healthy workers.
    pub fn all_loaded_models(&self) -> Vec<(String, Vec<LoadedModelInfo>)> {
        self.workers
            .iter()
            .filter(|e| e.value().health_status == HealthStatus::Healthy)
            .map(|e| {
                let state = e.value();
                (
                    state.registration.worker_id.clone(),
                    state.registration.loaded_models.clone(),
                )
            })
            .collect()
    }

    /// Get the total number of registered workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Get the number of healthy workers.
    pub fn healthy_count(&self) -> usize {
        self.workers
            .iter()
            .filter(|e| e.value().health_status == HealthStatus::Healthy)
            .count()
    }

    /// Mark a worker as currently loading a model.
    pub fn set_loading(&self, worker_id: &str, loading: bool) {
        if let Some(mut state) = self.workers.get_mut(worker_id) {
            state.is_loading = loading;
        }
    }

    /// Find cloud workers that have been idle for at least the given duration.
    ///
    /// Returns `(worker_id, instance_id)` pairs. Never returns local workers.
    pub fn idle_cloud_workers(&self, min_idle: Duration) -> Vec<(String, String)> {
        let now = Instant::now();
        self.workers
            .iter()
            .filter_map(|entry| {
                let state = entry.value();
                let instance_id = state.registration.instance_id.as_ref()?;
                if state.registration.provider == "local" {
                    return None;
                }
                if state.health_status != HealthStatus::Healthy {
                    return None;
                }
                if state.queue_depth > 0 || state.is_loading {
                    return None;
                }
                if now.duration_since(state.last_active) < min_idle {
                    return None;
                }
                Some((state.registration.worker_id.clone(), instance_id.clone()))
            })
            .collect()
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
