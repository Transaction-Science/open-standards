//! Orchestrator Axum server — internal API + public API proxy.

use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::middleware as axum_mw;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use super::health::spawn_health_checker;
use super::middleware::{
    auth_middleware, cluster_auth_middleware, global_metrics, metrics_middleware,
    rate_limit_middleware,
};
use super::registry::WorkerRegistry;
use super::router::{proxy_request, OrchestratorState};
use super::types::{HealthStatus, OrchestratorConfig, PipelineType, WorkerHeartbeat, WorkerRegistration};

/// Response for cluster status.
#[derive(Serialize)]
struct ClusterStatus {
    status: String,
    total_workers: usize,
    healthy_workers: usize,
    workers: Vec<WorkerInfo>,
}

/// Per-worker summary in cluster status.
#[derive(Serialize)]
struct WorkerInfo {
    worker_id: String,
    endpoint: String,
    health: String,
}

/// Response for cluster model listing.
#[derive(Serialize)]
struct ClusterModels {
    models: Vec<ClusterModelEntry>,
}

/// A model loaded on a specific worker.
#[derive(Serialize)]
struct ClusterModelEntry {
    worker_id: String,
    model_id: String,
    pipeline_type: String,
    memory_bytes: u64,
}

/// Start the orchestrator server.
pub async fn start_orchestrator(
    config: OrchestratorConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    let port = config.port;
    let registry = Arc::new(WorkerRegistry::new());
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(10)
        .build()?;
    let config = Arc::new(config);

    // Pre-register seed workers (from WORKER_URLS env var)
    for (i, url) in config.seed_worker_urls.iter().enumerate() {
        let seed_reg = WorkerRegistration {
            worker_id: format!("seed-worker-{i}"),
            endpoint: url.clone(),
            chip: "unknown".into(),
            memory_bytes: 0,
            gpu_cores: 0,
            provider: "unknown".into(),
            loaded_models: Vec::new(),
            capabilities: vec![
                "llm".into(),
                "diffusion".into(),
                "whisper".into(),
                "tts".into(),
                "audio".into(),
                "video".into(),
                "3d".into(),
                "models".into(),
                "system".into(),
            ],
            instance_id: None,
        };
        registry.register(seed_reg);
        tracing::info!(url = %url, "Pre-registered seed worker");
    }

    // Spawn background health checker
    let _health_handle = spawn_health_checker(
        Arc::clone(&registry),
        Arc::clone(&config),
        client.clone(),
    );

    // Spawn auto-scaler (only activates if cloud provider credentials are set)
    let _scaler_handle = super::scaler::spawn_auto_scaler(
        super::scaler::ScalerConfig::default(),
        Arc::clone(&registry),
    );

    let state = OrchestratorState {
        registry: Arc::clone(&registry),
        client,
        config: Arc::clone(&config),
    };

    // Build router with middleware stack
    let app = Router::new()
        // Internal API (worker → orchestrator) — cluster_secret auth
        .route("/internal/v1/register", post(handle_register))
        .route("/internal/v1/heartbeat", post(handle_heartbeat))
        // Cluster management API
        .route("/internal/v1/cluster", get(handle_cluster_status))
        .route("/internal/v1/cluster/models", get(handle_cluster_models))
        .route("/internal/v1/models/load", post(handle_model_load))
        // Orchestrator health + metrics
        .route("/orchestrator/health", get(handle_orchestrator_health))
        .route("/orchestrator/metrics", get(handle_metrics))
        // Proxy all other requests to workers (catch-all)
        .fallback(any(proxy_request))
        // Middleware stack (applied bottom-up: metrics → auth → cluster_auth → rate_limit)
        .layer(axum_mw::from_fn(rate_limit_middleware))
        .layer(axum_mw::from_fn(cluster_auth_middleware))
        .layer(axum_mw::from_fn(auth_middleware))
        .layer(axum_mw::from_fn(metrics_middleware))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Orchestrator listening on {addr}");
    tracing::info!("Internal API: /internal/v1/{{register,heartbeat,cluster}}");
    tracing::info!("Metrics: /orchestrator/metrics");
    tracing::info!("All other routes proxied to workers");

    if std::env::var("API_KEYS").is_ok() {
        tracing::info!("API key authentication enabled");
    }
    if std::env::var("CLUSTER_SECRET").is_ok() {
        tracing::info!("Cluster secret authentication enabled");
    }
    if std::env::var("RATE_LIMIT_RPS").is_ok() {
        tracing::info!("Rate limiting enabled");
    }

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Handle worker registration.
async fn handle_register(
    State(state): State<OrchestratorState>,
    Json(reg): Json<WorkerRegistration>,
) -> impl IntoResponse {
    let worker_id = state.registry.register(reg);
    Json(serde_json::json!({
        "status": "registered",
        "worker_id": worker_id
    }))
}

/// Handle worker heartbeat.
async fn handle_heartbeat(
    State(state): State<OrchestratorState>,
    Json(hb): Json<WorkerHeartbeat>,
) -> impl IntoResponse {
    state.registry.heartbeat(hb);
    StatusCode::OK
}

/// Handle cluster status query.
async fn handle_cluster_status(
    State(state): State<OrchestratorState>,
) -> impl IntoResponse {
    let workers = state.registry.all_workers();
    let healthy = state.registry.healthy_count();

    let worker_infos: Vec<WorkerInfo> = workers
        .into_iter()
        .map(|(id, endpoint, health)| WorkerInfo {
            worker_id: id,
            endpoint,
            health: format!("{health:?}"),
        })
        .collect();

    Json(ClusterStatus {
        status: if healthy > 0 {
            "ok".into()
        } else {
            "degraded".into()
        },
        total_workers: state.registry.worker_count(),
        healthy_workers: healthy,
        workers: worker_infos,
    })
}

/// Handle cluster-wide model listing.
async fn handle_cluster_models(
    State(state): State<OrchestratorState>,
) -> impl IntoResponse {
    let worker_models = state.registry.all_loaded_models();

    let mut entries = Vec::new();
    for (worker_id, models) in worker_models {
        for model in models {
            entries.push(ClusterModelEntry {
                worker_id: worker_id.clone(),
                model_id: model.model_id,
                pipeline_type: model.pipeline_type,
                memory_bytes: model.memory_bytes,
            });
        }
    }

    Json(ClusterModels { models: entries })
}

/// Orchestrator's own health endpoint.
async fn handle_orchestrator_health(
    State(state): State<OrchestratorState>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "orchestrator",
        "total_workers": state.registry.worker_count(),
        "healthy_workers": state.registry.healthy_count(),
    }))
}

/// Prometheus metrics endpoint.
async fn handle_metrics(State(state): State<OrchestratorState>) -> Response {
    let metrics = global_metrics();
    let body = metrics.to_prometheus(
        state.registry.worker_count(),
        state.registry.healthy_count(),
    );
    Response::builder()
        .header("content-type", "text/plain; version=0.0.4; charset=utf-8")
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Request to load a model on a worker via the orchestrator.
#[derive(Deserialize)]
struct OrchestratorModelLoadRequest {
    model_id: String,
    model_type: Option<String>,
    worker_id: Option<String>,
}

/// Handle orchestrator-driven model loading.
///
/// Picks the best available worker and forwards a model load + fetch request.
async fn handle_model_load(
    State(state): State<OrchestratorState>,
    Json(payload): Json<OrchestratorModelLoadRequest>,
) -> impl IntoResponse {
    let model_type = payload.model_type.unwrap_or_else(|| "llm".into());

    // Pick target worker
    let target = if let Some(ref target_id) = payload.worker_id {
        state
            .registry
            .all_workers()
            .into_iter()
            .find(|(id, _, health)| id == target_id && *health == HealthStatus::Healthy)
            .map(|(id, endpoint, _)| (id, endpoint))
    } else {
        state.registry.find_best_worker(PipelineType::Models)
    };

    let (worker_id, worker_endpoint) = match target {
        Some(t) => t,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "error",
                    "model_id": payload.model_id,
                    "error": "No healthy worker available"
                })),
            );
        }
    };

    tracing::info!(
        model_id = %payload.model_id,
        worker_id = %worker_id,
        "Loading model on worker"
    );

    // Mark worker as loading
    state.registry.set_loading(&worker_id, true);

    // Forward model load request to worker
    let url = format!("{worker_endpoint}/api/v1/models");
    let body = serde_json::json!({
        "model_id": payload.model_id,
        "model_type": model_type,
    });

    let result = state
        .client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(1800))
        .send()
        .await;

    // Clear loading flag
    state.registry.set_loading(&worker_id, false);

    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!(
                model_id = %payload.model_id,
                worker_id = %worker_id,
                "Model loaded successfully on worker"
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "loaded",
                    "worker_id": worker_id,
                    "model_id": payload.model_id,
                })),
            )
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(
                model_id = %payload.model_id,
                worker_id = %worker_id,
                "Worker returned error: {} - {}",
                status,
                body
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "status": "error",
                    "worker_id": worker_id,
                    "model_id": payload.model_id,
                    "error": format!("Worker returned {}: {}", status, body),
                })),
            )
        }
        Err(e) => {
            tracing::error!(
                model_id = %payload.model_id,
                worker_id = %worker_id,
                "Failed to reach worker: {}",
                e
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "status": "error",
                    "worker_id": worker_id,
                    "model_id": payload.model_id,
                    "error": format!("Failed to reach worker: {}", e),
                })),
            )
        }
    }
}
