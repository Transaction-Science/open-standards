//! Background health checker — polls workers periodically.

use std::sync::Arc;
use std::time::Duration;

use super::registry::WorkerRegistry;
use super::types::OrchestratorConfig;

/// Spawn a background task that periodically health-checks all workers.
pub fn spawn_health_checker(
    registry: Arc<WorkerRegistry>,
    config: Arc<OrchestratorConfig>,
    client: reqwest::Client,
) -> tokio::task::JoinHandle<()> {
    let interval = Duration::from_secs(config.health_check_interval_secs);

    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            ticker.tick().await;

            let workers = registry.all_workers();
            if workers.is_empty() {
                continue;
            }

            // Health-check all workers concurrently
            let mut handles = Vec::with_capacity(workers.len());
            for (worker_id, endpoint, _status) in workers {
                let client = client.clone();
                let registry = Arc::clone(&registry);
                let max_failures = config.max_consecutive_failures;

                handles.push(tokio::spawn(async move {
                    let url = format!("{endpoint}/health");
                    let start = std::time::Instant::now();

                    match client
                        .get(&url)
                        .timeout(Duration::from_secs(5))
                        .send()
                        .await
                    {
                        Ok(resp) if resp.status().is_success() => {
                            let rtt_ms = start.elapsed().as_secs_f64() * 1000.0;
                            registry.record_health_success(&worker_id, rtt_ms);
                        }
                        Ok(resp) => {
                            tracing::warn!(
                                worker_id = %worker_id,
                                status = %resp.status(),
                                "Worker health check returned non-200"
                            );
                            registry.record_health_failure(&worker_id, max_failures);
                        }
                        Err(e) => {
                            tracing::warn!(
                                worker_id = %worker_id,
                                error = %e,
                                "Worker health check failed"
                            );
                            registry.record_health_failure(&worker_id, max_failures);
                        }
                    }
                }));
            }

            // Wait for all health checks to complete
            for handle in handles {
                let _ = handle.await;
            }
        }
    })
}
