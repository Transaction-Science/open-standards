//! Worker registration client — registers this server with an orchestrator.
//!
//! When the `ORCHESTRATOR_URL` environment variable is set, this module
//! spawns a background task that:
//! 1. Registers with the orchestrator on startup
//! 2. Sends periodic heartbeats with updated metrics

use std::time::Duration;

use super::AppState;

/// Spawn the worker registration + heartbeat background task.
///
/// Returns `None` if `ORCHESTRATOR_URL` is not set (standalone mode).
pub fn spawn_registration(state: AppState) -> Option<tokio::task::JoinHandle<()>> {
    let orchestrator_url = match std::env::var("ORCHESTRATOR_URL") {
        Ok(url) if !url.is_empty() => url.trim_end_matches('/').to_string(),
        _ => {
            tracing::info!("No ORCHESTRATOR_URL set — running in standalone mode");
            return None;
        }
    };

    tracing::info!("Orchestrator registration enabled: {}", orchestrator_url);

    let handle = tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client for worker registration");

        // Initial registration (retry until successful)
        loop {
            match register(&client, &orchestrator_url, &state).await {
                Ok(()) => {
                    tracing::info!("Registered with orchestrator at {}", orchestrator_url);
                    break;
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to register with orchestrator: {} — retrying in 5s",
                        e
                    );
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }

        // Periodic heartbeat
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            if let Err(e) = heartbeat(&client, &orchestrator_url, &state).await {
                tracing::warn!("Heartbeat to orchestrator failed: {}", e);
            }
        }
    });

    Some(handle)
}

/// Register this worker with the orchestrator.
async fn register(
    client: &reqwest::Client,
    orchestrator_url: &str,
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (capabilities, loaded_models) = collect_worker_info(state).await;

    let endpoint = std::env::var("WORKER_ENDPOINT").unwrap_or_else(|_| {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(8080);
        format!("http://localhost:{port}")
    });

    let provider = std::env::var("WORKER_PROVIDER").unwrap_or_else(|_| "local".into());

    let payload = serde_json::json!({
        "worker_id": state.worker_id,
        "endpoint": endpoint,
        "chip": get_chip_name(),
        "memory_bytes": get_system_memory(),
        "gpu_cores": get_gpu_cores(),
        "provider": provider,
        "loaded_models": loaded_models,
        "capabilities": capabilities,
        "instance_id": std::env::var("INSTANCE_ID").ok(),
    });

    let url = format!("{orchestrator_url}/internal/v1/register");
    let resp = client.post(&url).json(&payload).send().await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Registration failed: {} - {}", status, body).into());
    }

    Ok(())
}

/// Send a heartbeat to the orchestrator.
async fn heartbeat(
    client: &reqwest::Client,
    orchestrator_url: &str,
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (_, loaded_models) = collect_worker_info(state).await;
    let queue_depth = state.in_flight.load(std::sync::atomic::Ordering::Relaxed);
    let metrics = state.metrics.read().await;

    let payload = serde_json::json!({
        "worker_id": state.worker_id,
        "queue_depth": queue_depth,
        "loaded_models": loaded_models,
        "model_memory_bytes": metrics.memory_usage_bytes,
        "available_memory_bytes": get_system_memory().saturating_sub(metrics.memory_usage_bytes as u64),
    });

    let url = format!("{orchestrator_url}/internal/v1/heartbeat");
    client.post(&url).json(&payload).send().await?;

    Ok(())
}

/// Collect current worker capabilities and loaded model info.
async fn collect_worker_info(state: &AppState) -> (Vec<String>, Vec<serde_json::Value>) {
    let mut capabilities = vec!["system".to_string(), "models".to_string()];

    {
        let pipeline_guard = state.pipeline.read().await;
        if pipeline_guard.is_some() {
            capabilities.push("diffusion".into());
        }
    }
    {
        let text_handler = state.text_handler.read().await;
        if text_handler.is_model_loaded() {
            capabilities.push("llm".into());
        }
    }

    let models = state.models.read().await;
    let loaded_models: Vec<serde_json::Value> = models
        .iter()
        .map(|(id, info)| {
            serde_json::json!({
                "model_id": id,
                "pipeline_type": info.model_type,
                "memory_bytes": info.memory_bytes,
            })
        })
        .collect();

    (capabilities, loaded_models)
}

/// Get Apple Silicon chip name.
fn get_chip_name() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Apple Silicon".into())
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unknown".into()
    }
}

/// Get total system memory in bytes.
fn get_system_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Get GPU core count (Apple Silicon).
fn get_gpu_cores() -> u32 {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.gpu_count"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(10) // M4 default
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}
