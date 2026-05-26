//! Shared types for orchestrator ↔ worker communication.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Information about a loaded model on a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    /// Model identifier (e.g. "tinyllama-1.1b-chat")
    pub model_id: String,
    /// Pipeline type: "llm", "diffusion", "whisper", "tts", "audio", "video"
    pub pipeline_type: String,
    /// Approximate memory usage in bytes
    pub memory_bytes: u64,
}

/// Worker registration payload (sent by worker → orchestrator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistration {
    /// Unique worker ID (hostname-uuid)
    pub worker_id: String,
    /// Reachable endpoint URL (e.g. "http://10.0.1.5:8080")
    pub endpoint: String,
    /// Hardware chip name (e.g. "Apple M4")
    pub chip: String,
    /// Total system memory in bytes
    pub memory_bytes: u64,
    /// GPU core count
    pub gpu_cores: u32,
    /// Cloud provider: "aws", "scaleway", "macstadium", "local"
    pub provider: String,
    /// Models currently loaded
    pub loaded_models: Vec<LoadedModelInfo>,
    /// Pipeline types this worker supports
    pub capabilities: Vec<String>,
    /// Cloud instance ID (AWS EC2 instance-id, Scaleway server-id).
    /// Populated by cloud-launched workers; None for local workers.
    #[serde(default)]
    pub instance_id: Option<String>,
}

/// Worker heartbeat payload (sent every 10s by worker → orchestrator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    /// Worker ID
    pub worker_id: String,
    /// Current in-flight request count
    pub queue_depth: u32,
    /// Models currently loaded (may change between heartbeats)
    pub loaded_models: Vec<LoadedModelInfo>,
    /// Total memory used by loaded models
    pub model_memory_bytes: u64,
    /// Available memory for additional models
    pub available_memory_bytes: u64,
}

/// Health status of a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Worker is healthy and accepting requests
    Healthy,
    /// Worker is responding but degraded (high queue depth, etc.)
    Degraded,
    /// Worker has not responded to health checks
    Unreachable,
}

/// Internal state tracked per worker by the orchestrator.
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// Registration data
    pub registration: WorkerRegistration,
    /// Last successful heartbeat or health check
    pub last_heartbeat: Instant,
    /// Current health status
    pub health_status: HealthStatus,
    /// Current queue depth (updated via heartbeats)
    pub queue_depth: u32,
    /// Consecutive health check failures
    pub consecutive_failures: u32,
    /// Rolling average round-trip time to this worker (ms)
    pub avg_rtt_ms: f64,
    /// Whether this worker is currently loading a model
    pub is_loading: bool,
    /// When this worker last had a non-zero queue depth (for idle detection)
    pub last_active: Instant,
}

impl WorkerState {
    /// Create a new `WorkerState` from a registration.
    pub fn from_registration(reg: WorkerRegistration) -> Self {
        Self {
            registration: reg,
            last_heartbeat: Instant::now(),
            health_status: HealthStatus::Healthy,
            queue_depth: 0,
            consecutive_failures: 0,
            avg_rtt_ms: 0.0,
            is_loading: false,
            last_active: Instant::now(),
        }
    }
}

/// Pipeline type extracted from API routes for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineType {
    /// Text generation (LLM)
    Llm,
    /// Image generation (diffusion)
    Diffusion,
    /// Speech-to-text (Whisper)
    Whisper,
    /// Text-to-speech (Kokoro)
    Tts,
    /// Audio generation (Bark)
    Audio,
    /// Video generation (Wan, HunyuanVideo)
    Video,
    /// 3D generation (TripoSR, Trellis)
    ThreeD,
    /// Model management (library, load, etc.)
    Models,
    /// Health/metrics/status
    System,
}

impl PipelineType {
    /// Extract pipeline type from a request path.
    pub fn from_path(path: &str) -> Self {
        if path.starts_with("/api/v1/text") {
            Self::Llm
        } else if path.starts_with("/api/v1/images") || path.starts_with("/generate") {
            Self::Diffusion
        } else if path.starts_with("/api/v1/whisper") || path.starts_with("/api/v1/transcribe") {
            Self::Whisper
        } else if path.starts_with("/api/v1/tts") {
            Self::Tts
        } else if path.starts_with("/api/v1/audio") {
            Self::Audio
        } else if path.starts_with("/api/v1/video") {
            Self::Video
        } else if path.starts_with("/api/v1/3d") {
            Self::ThreeD
        } else if path.starts_with("/api/v1/models") {
            Self::Models
        } else {
            Self::System
        }
    }

    /// Get the capability string workers use for this pipeline type.
    pub fn capability_str(&self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Diffusion => "diffusion",
            Self::Whisper => "whisper",
            Self::Tts => "tts",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::ThreeD => "3d",
            Self::Models => "models",
            Self::System => "system",
        }
    }
}

/// Orchestrator configuration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Port to listen on
    pub port: u16,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Number of consecutive failures before marking worker unreachable
    pub max_consecutive_failures: u32,
    /// Shared secret for inter-service auth
    pub cluster_secret: Option<String>,
    /// Pre-configured worker URLs (for initial bootstrap before registration)
    pub seed_worker_urls: Vec<String>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            port: 9000,
            health_check_interval_secs: 5,
            max_consecutive_failures: 3,
            cluster_secret: None,
            seed_worker_urls: Vec::new(),
        }
    }
}
