//! Worker abstractions for the distributed inference pool.
//!
//! A [`Worker`] is anything that can advertise a [`Capability`] set, report
//! a [`Load`] snapshot, and (optionally) serve work. The trait is async so
//! a worker can wrap a local model, a remote HTTP backend, or a vLLM /
//! Triton endpoint indifferently. The router, scheduler, batch admission
//! and KV-cache locality modules all operate purely on [`Worker`] +
//! [`Load`] snapshots, so adding a new accelerator class is additive.

use serde::{Deserialize, Serialize};

/// Accelerator class advertised by a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Accelerator {
    /// CPU-only worker (llama.cpp, ggml, vendor BLAS).
    Cpu,
    /// Nvidia / AMD GPU (CUDA, ROCm).
    Gpu,
    /// Google TPU.
    Tpu,
    /// Apple Neural Engine / Qualcomm Hexagon / etc.
    Npu,
    /// Fabric of mixed accelerators (heterogeneous node).
    Mixed,
}

impl Accelerator {
    /// Stable short tag, useful in logs.
    pub fn tag(&self) -> &'static str {
        match self {
            Accelerator::Cpu => "cpu",
            Accelerator::Gpu => "gpu",
            Accelerator::Tpu => "tpu",
            Accelerator::Npu => "npu",
            Accelerator::Mixed => "mixed",
        }
    }
}

/// Static capabilities advertised by a worker at registration time.
///
/// The fields here are deliberately a superset of vLLM's `ModelConfig`,
/// Triton's model repository metadata, and Ray Serve's deployment
/// signature, so any one of those can be projected into a `Capability`
/// without losing routing-relevant information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Capability {
    /// Models served by this worker (one worker may host several).
    pub models: Vec<String>,
    /// Accelerator class.
    pub accelerator: Accelerator,
    /// Max concurrent requests the worker is willing to accept.
    pub max_concurrency: u32,
    /// Whether the worker supports continuous batching (vLLM / TGI).
    pub continuous_batching: bool,
    /// Whether the worker exposes a paged-attention KV cache.
    pub paged_kv: bool,
    /// Carbon zone the worker physically lives in. Matches the catalog
    /// used by `eoc-carbon`.
    pub zone: String,
}

impl Capability {
    /// True if this worker can serve `model`.
    pub fn serves(&self, model: &str) -> bool {
        self.models.iter().any(|m| m == model)
    }
}

/// Live load snapshot reported by a worker.
///
/// All fields are absolute, never deltas. `joules_per_token` is the
/// rolling estimate published by the worker's own meter — the scheduler
/// uses it directly for joule-weighted placement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Load {
    /// In-flight requests right now.
    pub in_flight: u32,
    /// Tokens queued ahead of any new request.
    pub queued_tokens: u32,
    /// EMA of per-request latency in milliseconds.
    pub p50_latency_ms: u32,
    /// EMA of per-request latency in milliseconds (tail).
    pub p99_latency_ms: u32,
    /// Energy efficiency reported by the worker (micro-joules per
    /// generated token). Lower is greener.
    pub micro_joules_per_token: u32,
    /// gCO2e/kWh observed in the worker's zone right now.
    pub g_co2e_per_kwh: f32,
}

impl Load {
    /// An idle, perfectly green load. Useful as a default in tests.
    pub fn idle() -> Self {
        Self {
            in_flight: 0,
            queued_tokens: 0,
            p50_latency_ms: 0,
            p99_latency_ms: 0,
            micro_joules_per_token: 0,
            g_co2e_per_kwh: 0.0,
        }
    }

    /// "Busyness" used by the least-busy router. Lower = better.
    pub fn busyness(&self) -> f64 {
        // Weighted: each in-flight request costs more than queued tokens
        // alone because requests already past admission also hold KV
        // entries. Coefficients match LiteLLM's default least-busy.
        (self.in_flight as f64) * 4.0 + (self.queued_tokens as f64) * 0.1
    }

    /// Joule cost projection for a hypothetical request of `tokens`
    /// generated tokens. Used by the joule-weighted router.
    pub fn projected_micro_joules(&self, tokens: u32) -> u64 {
        (self.micro_joules_per_token as u64) * (tokens as u64)
    }

    /// Grams of CO2 equivalent for `tokens` generated tokens at the
    /// worker's current zone intensity. Uses the worker's reported
    /// micro-joules-per-token and zone intensity directly.
    pub fn projected_g_co2e(&self, tokens: u32) -> f64 {
        let micro_j = self.projected_micro_joules(tokens) as f64;
        let joules = micro_j / 1_000_000.0;
        let kwh = joules / 3_600_000.0;
        kwh * (self.g_co2e_per_kwh as f64)
    }
}

/// A worker is a thing that serves inference requests.
///
/// Implementations wrap local models, remote HTTP backends (vLLM, Triton,
/// TGI), or speculative-decoding gateways. The trait is intentionally
/// minimal: routing, batching, KV-cache locality and failure detection
/// all happen *above* the worker.
#[async_trait::async_trait]
pub trait Worker: Send + Sync {
    /// Stable id for this worker. Must be unique inside a [`Cluster`].
    fn id(&self) -> &str;

    /// Static capabilities.
    fn capability(&self) -> &Capability;

    /// Current load snapshot.
    fn load(&self) -> Load;
}

/// In-memory worker used by tests and as a reference implementation. A
/// realistic backend would proxy to an out-of-process server, but
/// every routing decision in this crate is driven by snapshots, so we
/// can exercise everything in unit tests with no I/O.
#[derive(Debug, Clone)]
pub struct InMemoryWorker {
    id: String,
    cap: Capability,
    load: Load,
}

impl InMemoryWorker {
    /// Construct.
    pub fn new(id: impl Into<String>, cap: Capability, load: Load) -> Self {
        Self {
            id: id.into(),
            cap,
            load,
        }
    }

    /// Mutate the reported load — used by tests to simulate pressure.
    pub fn set_load(&mut self, load: Load) {
        self.load = load;
    }
}

#[async_trait::async_trait]
impl Worker for InMemoryWorker {
    fn id(&self) -> &str {
        &self.id
    }

    fn capability(&self) -> &Capability {
        &self.cap
    }

    fn load(&self) -> Load {
        self.load
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(model: &str, zone: &str) -> Capability {
        Capability {
            models: vec![model.to_string()],
            accelerator: Accelerator::Gpu,
            max_concurrency: 16,
            continuous_batching: true,
            paged_kv: true,
            zone: zone.to_string(),
        }
    }

    #[test]
    fn busyness_is_monotone() {
        let a = Load {
            in_flight: 1,
            queued_tokens: 0,
            ..Load::idle()
        };
        let b = Load {
            in_flight: 4,
            queued_tokens: 0,
            ..Load::idle()
        };
        assert!(a.busyness() < b.busyness());
    }

    #[test]
    fn joule_projection_is_linear() {
        let l = Load {
            micro_joules_per_token: 100,
            ..Load::idle()
        };
        assert_eq!(l.projected_micro_joules(10), 1_000);
        assert_eq!(l.projected_micro_joules(100), 10_000);
    }

    #[test]
    fn serves_filters_models() {
        let c = cap("llama-70b", "EU-FR");
        assert!(c.serves("llama-70b"));
        assert!(!c.serves("mistral-7b"));
    }
}
