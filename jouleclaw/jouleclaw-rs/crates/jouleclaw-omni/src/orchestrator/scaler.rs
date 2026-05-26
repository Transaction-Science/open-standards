//! Auto-scaler — monitors fleet metrics and scales workers up/down.
//!
//! Scaling backends are pluggable via the `ScalingBackend` enum.
//! Currently supports:
//! - AWS EC2 Mac (24-hour minimum allocation — good for baseline)
//! - Scaleway Apple Silicon (hourly billing — good for burst)

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::registry::WorkerRegistry;

// ============================================================================
// Scaling Backend Types
// ============================================================================

/// Configuration for launching a new instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// Instance type (e.g. "mac-m4.metal", "MAC-M4")
    pub instance_type: String,
    /// Region (e.g. "us-east-1", "fr-par-3")
    pub region: String,
    /// Tags/labels for the instance
    pub tags: Vec<(String, String)>,
}

/// Info about a running instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    /// Provider-specific instance ID
    pub instance_id: String,
    /// Public IP (if available)
    pub public_ip: Option<String>,
    /// Private IP
    pub private_ip: Option<String>,
    /// Instance state: "running", "pending", "stopping", etc.
    pub state: String,
    /// When the instance was launched
    pub launched_at: Option<String>,
    /// Provider name
    pub provider: String,
}

type BackendResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Enum-based dispatch for scaling backends (avoids dyn-incompatible async trait).
pub enum ScalingBackend {
    /// AWS EC2 Mac instances.
    Aws(AwsBackend),
    /// Scaleway Apple Silicon instances.
    Scaleway(ScalewayBackend),
}

impl ScalingBackend {
    /// Launch a new instance via the appropriate cloud provider.
    pub async fn launch_instance(&self, config: &InstanceConfig) -> BackendResult<String> {
        match self {
            Self::Aws(b) => b.launch_instance(config).await,
            Self::Scaleway(b) => b.launch_instance(config).await,
        }
    }

    /// Terminate a running instance.
    pub async fn terminate_instance(&self, instance_id: &str) -> BackendResult<()> {
        match self {
            Self::Aws(b) => b.terminate_instance(instance_id).await,
            Self::Scaleway(b) => b.terminate_instance(instance_id).await,
        }
    }

    /// List all managed instances from this provider.
    pub async fn list_instances(&self) -> BackendResult<Vec<InstanceInfo>> {
        match self {
            Self::Aws(b) => b.list_instances().await,
            Self::Scaleway(b) => b.list_instances().await,
        }
    }

    /// Get the provider name string.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::Aws(b) => b.provider_name(),
            Self::Scaleway(b) => b.provider_name(),
        }
    }
}

// ============================================================================
// AWS EC2 Mac Backend
// ============================================================================

/// AWS EC2 Mac scaling backend.
///
/// Uses the AWS CLI (via shell) for instance management.
/// Requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
/// `AWS_DEFAULT_REGION` environment variables.
pub struct AwsBackend {
    /// AWS region
    region: String,
    /// Dedicated host ID (required for Mac instances)
    dedicated_host_id: Option<String>,
    /// Security group ID
    security_group_id: Option<String>,
    /// Subnet ID
    subnet_id: Option<String>,
    /// AMI ID for macOS
    ami_id: Option<String>,
}

impl AwsBackend {
    /// Create from environment variables.
    pub fn from_env() -> Self {
        Self {
            region: std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
            dedicated_host_id: std::env::var("AWS_DEDICATED_HOST_ID").ok(),
            security_group_id: std::env::var("AWS_SECURITY_GROUP_ID").ok(),
            subnet_id: std::env::var("AWS_SUBNET_ID").ok(),
            ami_id: std::env::var("AWS_MAC_AMI_ID").ok(),
        }
    }

    async fn launch_instance(&self, config: &InstanceConfig) -> BackendResult<String> {
        let host_id = self.dedicated_host_id.as_deref().ok_or(
            "AWS_DEDICATED_HOST_ID required for Mac instances"
        )?;

        let mut args = vec![
            "ec2".to_string(),
            "run-instances".to_string(),
            "--instance-type".to_string(),
            config.instance_type.clone(),
            "--placement".to_string(),
            format!("HostId={host_id}"),
            "--region".to_string(),
            self.region.clone(),
            "--count".to_string(),
            "1".to_string(),
        ];

        if let Some(ref ami) = self.ami_id {
            args.extend(["--image-id".to_string(), ami.clone()]);
        }
        if let Some(ref sg) = self.security_group_id {
            args.extend(["--security-group-ids".to_string(), sg.clone()]);
        }
        if let Some(ref subnet) = self.subnet_id {
            args.extend(["--subnet-id".to_string(), subnet.clone()]);
        }

        let output = tokio::task::spawn_blocking(move || {
            std::process::Command::new("aws").args(&args).output()
        }).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("AWS launch failed: {stderr}").into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let json: serde_json::Value = serde_json::from_str(&stdout)?;
        let instance_id = json["Instances"][0]["InstanceId"]
            .as_str()
            .ok_or("No InstanceId in response")?
            .to_string();

        tracing::info!(instance_id = %instance_id, "AWS instance launched");
        Ok(instance_id)
    }

    async fn terminate_instance(&self, instance_id: &str) -> BackendResult<()> {
        let region = self.region.clone();
        let id = instance_id.to_string();
        let output = tokio::task::spawn_blocking(move || {
            std::process::Command::new("aws")
                .args([
                    "ec2", "terminate-instances",
                    "--instance-ids", &id,
                    "--region", &region,
                ])
                .output()
        }).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("AWS terminate failed: {stderr}").into());
        }

        tracing::info!(instance_id = %instance_id, "AWS instance terminated");
        Ok(())
    }

    async fn list_instances(&self) -> BackendResult<Vec<InstanceInfo>> {
        let region = self.region.clone();
        let output = tokio::task::spawn_blocking(move || {
            std::process::Command::new("aws")
                .args([
                    "ec2", "describe-instances",
                    "--region", &region,
                    "--filters",
                    "Name=instance-type,Values=mac-m4.metal,mac-m4pro.metal",
                    "Name=instance-state-name,Values=running,pending",
                    "--output", "json",
                ])
                .output()
        }).await??;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let json: serde_json::Value = serde_json::from_str(&stdout)?;

        let mut instances = Vec::new();
        if let Some(reservations) = json["Reservations"].as_array() {
            for res in reservations {
                if let Some(insts) = res["Instances"].as_array() {
                    for inst in insts {
                        instances.push(InstanceInfo {
                            instance_id: inst["InstanceId"].as_str().unwrap_or("").into(),
                            public_ip: inst["PublicIpAddress"].as_str().map(|s| s.into()),
                            private_ip: inst["PrivateIpAddress"].as_str().map(|s| s.into()),
                            state: inst["State"]["Name"].as_str().unwrap_or("unknown").into(),
                            launched_at: inst["LaunchTime"].as_str().map(|s| s.into()),
                            provider: "aws".into(),
                        });
                    }
                }
            }
        }

        Ok(instances)
    }

    fn provider_name(&self) -> &str {
        "aws"
    }
}

// ============================================================================
// Scaleway Backend
// ============================================================================

/// Scaleway Apple Silicon scaling backend.
///
/// Uses the Scaleway API via HTTPS.
/// Requires `SCW_SECRET_KEY` and optionally `SCW_DEFAULT_ZONE`.
pub struct ScalewayBackend {
    /// Scaleway secret key
    secret_key: String,
    /// Scaleway zone (default: fr-par-3)
    zone: String,
    /// Scaleway project ID
    project_id: Option<String>,
    /// HTTP client
    client: reqwest::Client,
}

impl ScalewayBackend {
    /// Create from environment variables.
    pub fn from_env() -> Option<Self> {
        let secret_key = std::env::var("SCW_SECRET_KEY").ok()?;
        Some(Self {
            secret_key,
            zone: std::env::var("SCW_DEFAULT_ZONE").unwrap_or_else(|_| "fr-par-3".into()),
            project_id: std::env::var("SCW_PROJECT_ID").ok(),
            client: reqwest::Client::new(),
        })
    }

    fn api_url(&self, path: &str) -> String {
        format!(
            "https://api.scaleway.com/apple-silicon/v1alpha1/zones/{}{path}",
            self.zone
        )
    }

    async fn launch_instance(&self, config: &InstanceConfig) -> BackendResult<String> {
        let body = serde_json::json!({
            "name": format!("create-worker-{}", chrono_timestamp()),
            "type": config.instance_type,
            "project_id": self.project_id,
        });

        let resp = self.client
            .post(self.api_url("/servers"))
            .header("X-Auth-Token", &self.secret_key)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Scaleway launch failed: {status} - {body}").into());
        }

        let json: serde_json::Value = resp.json().await?;
        let server_id = json["id"].as_str().ok_or("No id in response")?.to_string();

        tracing::info!(server_id = %server_id, "Scaleway instance created");
        Ok(server_id)
    }

    async fn terminate_instance(&self, instance_id: &str) -> BackendResult<()> {
        // First power off, then delete
        let _ = self.client
            .post(self.api_url(&format!("/servers/{instance_id}/action")))
            .header("X-Auth-Token", &self.secret_key)
            .json(&serde_json::json!({"action": "power_off"}))
            .send()
            .await;

        tokio::time::sleep(Duration::from_secs(5)).await;

        let resp = self.client
            .delete(self.api_url(&format!("/servers/{instance_id}")))
            .header("X-Auth-Token", &self.secret_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Scaleway delete failed: {status} - {body}").into());
        }

        tracing::info!(instance_id = %instance_id, "Scaleway instance terminated");
        Ok(())
    }

    async fn list_instances(&self) -> BackendResult<Vec<InstanceInfo>> {
        let resp = self.client
            .get(self.api_url("/servers"))
            .header("X-Auth-Token", &self.secret_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Ok(Vec::new());
        }

        let json: serde_json::Value = resp.json().await?;
        let mut instances = Vec::new();

        if let Some(servers) = json["servers"].as_array() {
            for srv in servers {
                instances.push(InstanceInfo {
                    instance_id: srv["id"].as_str().unwrap_or("").into(),
                    public_ip: srv["ip"].as_str().map(|s| s.into()),
                    private_ip: None,
                    state: srv["status"].as_str().unwrap_or("unknown").into(),
                    launched_at: srv["created_at"].as_str().map(|s| s.into()),
                    provider: "scaleway".into(),
                });
            }
        }

        Ok(instances)
    }

    fn provider_name(&self) -> &str {
        "scaleway"
    }
}

// ============================================================================
// Auto-Scaling Monitor
// ============================================================================

/// Scaling decision.
#[derive(Debug, Clone)]
pub enum ScalingAction {
    /// No action needed
    None,
    /// Scale up: launch N new instances.
    ScaleUp {
        /// Number of instances to launch.
        count: usize,
        /// Human-readable reason for scaling.
        reason: String,
    },
    /// Scale down: terminate specific instances.
    ScaleDown {
        /// (worker_id, instance_id) pairs to terminate.
        targets: Vec<(String, String)>,
        /// Human-readable reason for scaling.
        reason: String,
    },
}

/// Auto-scaling configuration.
#[derive(Debug, Clone)]
pub struct ScalerConfig {
    /// Average queue depth threshold to trigger scale-up
    pub scale_up_queue_threshold: u32,
    /// Duration the threshold must be exceeded before scaling up
    pub scale_up_duration: Duration,
    /// Idle duration before considering scale-down
    pub scale_down_idle_duration: Duration,
    /// Minimum number of workers (never scale below this)
    pub min_workers: usize,
    /// Maximum number of workers
    pub max_workers: usize,
    /// Cooldown between scaling actions
    pub cooldown: Duration,
}

impl Default for ScalerConfig {
    fn default() -> Self {
        Self {
            scale_up_queue_threshold: 2,
            scale_up_duration: Duration::from_secs(60),
            scale_down_idle_duration: Duration::from_secs(900), // 15 minutes
            min_workers: 1,
            max_workers: 5,
            cooldown: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Auto-scaling monitor that evaluates fleet metrics and recommends actions.
pub struct AutoScaler {
    config: ScalerConfig,
    registry: Arc<WorkerRegistry>,
    /// When the last scaling action was taken
    last_action: Instant,
    /// Backends (tried in order)
    backends: Vec<ScalingBackend>,
}

impl AutoScaler {
    /// Create a new auto-scaler.
    pub fn new(
        config: ScalerConfig,
        registry: Arc<WorkerRegistry>,
        backends: Vec<ScalingBackend>,
    ) -> Self {
        Self {
            config,
            registry,
            last_action: Instant::now() - Duration::from_secs(600), // Allow immediate first action
            backends,
        }
    }

    /// Evaluate current fleet state and return a scaling recommendation.
    pub fn evaluate(&mut self) -> ScalingAction {
        let total = self.registry.worker_count();
        let healthy = self.registry.healthy_count();

        // Don't act during cooldown
        if self.last_action.elapsed() < self.config.cooldown {
            return ScalingAction::None;
        }

        // All workers unhealthy — emergency, but don't scale up (fix existing)
        if healthy == 0 && total > 0 {
            tracing::warn!("All {} workers unhealthy, not scaling", total);
            return ScalingAction::None;
        }

        // Below minimum — scale up
        if total < self.config.min_workers {
            let needed = self.config.min_workers - total;
            return ScalingAction::ScaleUp {
                count: needed,
                reason: format!(
                    "Below minimum: {total}/{} workers",
                    self.config.min_workers
                ),
            };
        }

        // Scale down evaluation (above minimum with idle workers)
        if total > self.config.min_workers {
            let idle = self.registry.idle_cloud_workers(self.config.scale_down_idle_duration);

            if !idle.is_empty() {
                let max_removable = total - self.config.min_workers;
                let targets: Vec<(String, String)> = idle
                    .into_iter()
                    .take(max_removable)
                    .collect();

                if !targets.is_empty() {
                    return ScalingAction::ScaleDown {
                        reason: format!(
                            "Idle workers (>{} min): {} instances",
                            self.config.scale_down_idle_duration.as_secs() / 60,
                            targets.len()
                        ),
                        targets,
                    };
                }
            }
        }

        ScalingAction::None
    }

    /// Execute a scaling action using available backends.
    pub async fn execute(&mut self, action: ScalingAction) {
        match action {
            ScalingAction::None => {}
            ScalingAction::ScaleUp { count, reason } => {
                tracing::info!(count = count, reason = %reason, "Scaling up");
                self.last_action = Instant::now();

                for _ in 0..count {
                    let config = InstanceConfig {
                        instance_type: "MAC-M4".into(), // Scaleway naming
                        region: "fr-par-3".into(),
                        tags: vec![("service".into(), "create-worker".into())],
                    };

                    // Try backends in order (prefer cheaper ones first)
                    for backend in &self.backends {
                        match backend.launch_instance(&config).await {
                            Ok(id) => {
                                tracing::info!(
                                    provider = backend.provider_name(),
                                    instance_id = %id,
                                    "Instance launched"
                                );
                                break;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    provider = backend.provider_name(),
                                    error = %e,
                                    "Failed to launch, trying next backend"
                                );
                            }
                        }
                    }
                }
            }
            ScalingAction::ScaleDown { targets, reason } => {
                tracing::info!(
                    count = targets.len(),
                    reason = %reason,
                    "Scaling down"
                );
                self.last_action = Instant::now();

                for (worker_id, instance_id) in &targets {
                    for backend in &self.backends {
                        if backend.terminate_instance(instance_id).await.is_ok() {
                            self.registry.deregister(worker_id);
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Spawn the auto-scaling monitor as a background task.
pub fn spawn_auto_scaler(
    config: ScalerConfig,
    registry: Arc<WorkerRegistry>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Build backends from env
        let mut backends: Vec<ScalingBackend> = Vec::new();

        // Prefer Scaleway (cheaper, hourly billing)
        if let Some(scw) = ScalewayBackend::from_env() {
            tracing::info!("Scaleway scaling backend enabled");
            backends.push(ScalingBackend::Scaleway(scw));
        }

        // AWS as fallback
        if std::env::var("AWS_ACCESS_KEY_ID").is_ok() {
            tracing::info!("AWS scaling backend enabled");
            backends.push(ScalingBackend::Aws(AwsBackend::from_env()));
        }

        if backends.is_empty() {
            tracing::info!("No scaling backends configured — auto-scaler disabled");
            return;
        }

        let mut scaler = AutoScaler::new(config, registry, backends);
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            let action = scaler.evaluate();
            scaler.execute(action).await;
        }
    })
}

/// Simple timestamp for naming (avoids chrono dependency).
fn chrono_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}
