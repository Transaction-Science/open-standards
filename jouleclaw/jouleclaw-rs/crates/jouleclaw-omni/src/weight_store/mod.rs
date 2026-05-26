//! Mode-capped tiered weight store for LLM inference.
//!
//! # Design Principle
//!
//! **Use the minimum memory required by the current operation, not the maximum available.**
//!
//! A 256 GB machine running a 120B model should NOT hold 57 GB resident.
//! It should hold ~3-4 GB — only what the current layer + active experts + embeddings demand.
//!
//! # Architecture
//!
//! ```text
//! Hot:   ~3-4 GB  (current layer weights + active experts + embed + LM head)
//! Warm:  mmap     (demand-paged, zero resident cost until touched)
//! Cold:  disk     (4TB SSD: /Volumes/macos_4TB_external/neuronexus_ai/model_library/)
//! ```
//!
//! The hot tier budget is set by the inference **mode**, not the model size.
//! A 671B MoE model and an 8B dense model use the same ~3-4 GB of hot RAM.

mod tier;
/// Tensor index and GGUF parser.
pub mod index;
mod metrics;
// JouleDB bridge dropped during the efficient-genai → jouleclaw-omni port:
// it path-depended on `joulesperbit/crates/joule-db-{weights,core}`, which
// is JoulesPerBit IP that does not port into the public JouleClaw open
// standard. Operators who want a JouleDB weight backend can re-add it
// downstream by implementing the `Tier` trait against their joule-db
// instance.

pub use tier::{Tier, TierConfig, TierManager};
pub use index::{TensorMeta, ModelIndex};
pub use metrics::{StoreMetrics, AccessStats};

use crate::core::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;

/// Mode-capped weight store.
///
/// Provides per-tensor access to model weights with automatic tiering.
/// The hot tier is capped by the inference mode, not by available RAM.
pub struct WeightStore {
    /// Model indices: model_id → tensor metadata
    models: RwLock<HashMap<String, ModelIndex>>,
    /// Tier manager: controls hot/warm/cold promotion/demotion
    tier_manager: TierManager,
    /// Access metrics
    metrics: Arc<StoreMetrics>,
    /// Base path for cold storage
    cold_path: PathBuf,
}

impl WeightStore {
    /// Create a new weight store with the given tier configuration.
    ///
    /// `cold_path` is the base directory for model files (e.g., the 4TB SSD).
    pub fn new(config: TierConfig, cold_path: impl Into<PathBuf>) -> Self {
        let metrics = Arc::new(StoreMetrics::new());
        Self {
            models: RwLock::new(HashMap::new()),
            tier_manager: TierManager::new(config, metrics.clone()),
            metrics,
            cold_path: cold_path.into(),
        }
    }

    /// Create a weight store with default config for LLM inference.
    ///
    /// Hot budget: 4 GB, warm budget: 4 GB, aggressive demotion (1s idle).
    pub fn for_llm_inference(cold_path: impl Into<PathBuf>) -> Self {
        Self::new(TierConfig::llm_default(), cold_path)
    }

    /// Index a GGUF model file without loading any weight data.
    ///
    /// Scans the GGUF header to extract tensor metadata (name, shape, dtype,
    /// byte offset, size). No tensor data is read — everything stays cold on disk.
    pub fn index_gguf(&self, model_id: &str, path: impl AsRef<Path>) -> Result<IndexStats> {
        let path = path.as_ref();
        let index = index::index_gguf_file(path)?;
        let stats = self.register_index(model_id, index);
        Ok(stats)
    }

    /// Index sharded GGUF files (multiple .gguf files for one model).
    pub fn index_gguf_sharded(&self, model_id: &str, dir: impl AsRef<Path>) -> Result<IndexStats> {
        let dir = dir.as_ref();
        let index = index::index_gguf_sharded(dir)?;
        let stats = self.register_index(model_id, index);
        Ok(stats)
    }

    fn register_index(&self, model_id: &str, index: ModelIndex) -> IndexStats {
        // Register all tensors as cold in the tier manager
        for meta in index.tensors.values() {
            self.tier_manager.register_cold(model_id, meta);
            self.metrics.increment_tensors();
        }
        let stats = IndexStats {
            tensor_count: index.tensors.len(),
            total_bytes: index.total_bytes,
            layer_count: index.layer_count(),
            expert_count: index.expert_count(),
            moe_layer_count: index.moe_layer_count(),
            ssm_layer_count: index.ssm_layer_count(),
        };
        self.models.write().insert(model_id.to_string(), index);
        stats
    }

    /// Get a tensor's raw bytes, respecting tier management.
    ///
    /// Returns a reference to the tensor data. The data may be:
    /// - In the hot tier (resident in RAM) — immediate return
    /// - In the warm tier (mmap'd) — page fault on first access
    /// - In the cold tier (disk) — triggers mmap promotion to warm
    ///
    /// The hot tier is capped. If promoting this tensor would exceed the budget,
    /// the least-recently-used hot tensor is demoted first.
    pub fn get_tensor(&self, model_id: &str, tensor_name: &str) -> Result<TensorRef> {
        let models = self.models.read();
        let model = models.get(model_id).ok_or_else(|| {
            Error::ModelLoad {
                model: model_id.to_string(),
                message: "model not indexed in weight store".to_string(),
                #[cfg(feature = "std")]
                source: None,
            }
        })?;
        let meta = model.get_tensor(tensor_name).ok_or_else(|| {
            Error::ModelLoad {
                model: model_id.to_string(),
                message: format!("tensor '{}' not found", tensor_name),
                #[cfg(feature = "std")]
                source: None,
            }
        })?;

        self.metrics.record_access(model_id, tensor_name);
        self.tier_manager.ensure_accessible(model_id, meta)
    }

    /// Prefetch all tensors for a given layer to the warm tier (mmap).
    ///
    /// This is a hint — the OS will page in the data when the GPU accesses it.
    /// Call this 1-2 layers ahead of the current execution point.
    pub fn prefetch_layer(&self, model_id: &str, layer_idx: usize) -> Result<()> {
        let models = self.models.read();
        let model = models.get(model_id).ok_or_else(|| {
            Error::ModelLoad {
                model: model_id.to_string(),
                message: "model not indexed in weight store".to_string(),
                #[cfg(feature = "std")]
                source: None,
            }
        })?;

        for meta in model.tensors_for_layer(layer_idx) {
            self.tier_manager.prefetch_warm(model_id, meta)?;
        }
        Ok(())
    }

    /// Prefetch specific MoE experts for a layer.
    pub fn prefetch_experts(
        &self,
        model_id: &str,
        layer_idx: usize,
        expert_ids: &[usize],
    ) -> Result<()> {
        let models = self.models.read();
        let model = models.get(model_id).ok_or_else(|| {
            Error::ModelLoad {
                model: model_id.to_string(),
                message: "model not indexed in weight store".to_string(),
                #[cfg(feature = "std")]
                source: None,
            }
        })?;

        for &eid in expert_ids {
            for meta in model.tensors_for_expert(layer_idx, eid) {
                self.tier_manager.prefetch_warm(model_id, meta)?;
            }
        }
        Ok(())
    }

    /// Evict all tensors for a completed layer from the hot tier.
    ///
    /// Call this after a layer is done executing to free hot budget for the next layer.
    pub fn evict_layer(&self, model_id: &str, layer_idx: usize) -> Result<()> {
        let models = self.models.read();
        let model = models.get(model_id).ok_or_else(|| {
            Error::ModelLoad {
                model: model_id.to_string(),
                message: "model not indexed in weight store".to_string(),
                #[cfg(feature = "std")]
                source: None,
            }
        })?;

        for meta in model.tensors_for_layer(layer_idx) {
            self.tier_manager.demote_to_cold(model_id, meta)?;
        }
        Ok(())
    }

    /// Pin tensors that should always be hot (embeddings, LM head, shared experts).
    pub fn pin_hot(&self, model_id: &str, tensor_names: &[&str]) -> Result<()> {
        let models = self.models.read();
        let model = models.get(model_id).ok_or_else(|| {
            Error::ModelLoad {
                model: model_id.to_string(),
                message: "model not indexed in weight store".to_string(),
                #[cfg(feature = "std")]
                source: None,
            }
        })?;

        for name in tensor_names {
            if let Some(meta) = model.get_tensor(name) {
                self.tier_manager.pin_hot(model_id, meta)?;
            }
        }
        Ok(())
    }

    /// Get a read lock on the model index map.
    /// Used by LazyLoader to iterate over tensor metadata.
    pub fn models_ref(&self) -> parking_lot::RwLockReadGuard<'_, HashMap<String, ModelIndex>> {
        self.models.read()
    }

    /// Get current memory usage statistics.
    pub fn stats(&self) -> StoreSnapshot {
        StoreSnapshot {
            hot_bytes: self.tier_manager.hot_bytes(),
            hot_budget: self.tier_manager.config().hot_budget_bytes,
            warm_bytes: self.tier_manager.warm_bytes(),
            cold_bytes: self.tier_manager.cold_bytes(),
            total_tensors: self.metrics.total_tensors(),
            total_accesses: self.metrics.total_accesses(),
            hot_hits: self.metrics.hot_hits(),
            warm_hits: self.metrics.warm_hits(),
            cold_hits: self.metrics.cold_hits(),
        }
    }
}

/// Reference to tensor data managed by the weight store.
pub struct TensorRef {
    /// Pointer to the tensor data (may be mmap'd or resident).
    pub data: *const u8,
    /// Length in bytes.
    pub len: usize,
    /// The mmap backing this data (keeps it alive).
    pub _mmap: Option<Arc<memmap2::Mmap>>,
    /// Metadata about this tensor.
    pub meta: TensorMeta,
}

// Safety: TensorRef is read-only and the backing mmap/allocation is Arc-protected.
unsafe impl Send for TensorRef {}
unsafe impl Sync for TensorRef {}

impl TensorRef {
    /// Get a slice of the tensor data.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

/// Statistics from indexing a model.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of tensors indexed.
    pub tensor_count: usize,
    /// Total bytes of all tensors.
    pub total_bytes: u64,
    /// Number of transformer layers detected.
    pub layer_count: usize,
    /// Number of MoE experts detected (0 for dense models).
    pub expert_count: usize,
    /// Number of layers that contain MoE expert tensors.
    pub moe_layer_count: usize,
    /// Number of layers that contain SSM (Mamba) tensors.
    pub ssm_layer_count: usize,
}

/// Snapshot of weight store memory usage.
#[derive(Debug, Clone)]
pub struct StoreSnapshot {
    /// Bytes currently in hot tier (resident RAM).
    pub hot_bytes: u64,
    /// Hot tier budget cap.
    pub hot_budget: u64,
    /// Bytes currently mmap'd (warm tier — may or may not be resident).
    pub warm_bytes: u64,
    /// Bytes on disk only (cold tier).
    pub cold_bytes: u64,
    /// Total tensors indexed.
    pub total_tensors: u64,
    /// Total tensor accesses.
    pub total_accesses: u64,
    /// Accesses served from hot tier.
    pub hot_hits: u64,
    /// Accesses served from warm tier.
    pub warm_hits: u64,
    /// Accesses served from cold tier.
    pub cold_hits: u64,
}

impl std::fmt::Display for StoreSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WeightStore: hot={:.1}MB/{:.1}MB warm={:.1}MB cold={:.1}MB \
             tensors={} accesses={} (hot:{} warm:{} cold:{})",
            self.hot_bytes as f64 / 1e6,
            self.hot_budget as f64 / 1e6,
            self.warm_bytes as f64 / 1e6,
            self.cold_bytes as f64 / 1e6,
            self.total_tensors,
            self.total_accesses,
            self.hot_hits,
            self.warm_hits,
            self.cold_hits,
        )
    }
}
