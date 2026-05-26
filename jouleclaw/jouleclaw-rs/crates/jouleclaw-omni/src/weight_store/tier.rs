//! Tier management: hot/warm/cold promotion and demotion with mode-capped budgets.

use super::{TensorMeta, TensorRef, StoreMetrics};
use crate::core::{Error, Result};
use parking_lot::RwLock;
use std::collections::{HashMap, BTreeMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Which tier a tensor currently lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Resident in RAM. Immediate access. Counts against hot budget.
    Hot,
    /// Memory-mapped. Zero resident cost until page-faulted. OS manages residency.
    Warm,
    /// On disk only. Must be mmap'd (promoted to warm) before access.
    Cold,
}

/// Configuration for the tiered weight store.
///
/// The hot budget is the MODE CAP — maximum RAM used regardless of model size.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Maximum bytes in hot tier. This is the mode cap.
    /// Default: 4 GB (enough for 1 layer + embeddings + active experts).
    pub hot_budget_bytes: u64,
    /// Maximum bytes in warm tier (mmap'd).
    /// Default: 4 GB (1-2 layers of prefetch window).
    pub warm_budget_bytes: u64,
    /// Seconds of idle before demoting from hot to cold.
    /// Default: 1 (aggressive — free memory immediately after layer completes).
    pub demotion_idle_secs: u64,
}

impl TierConfig {
    /// Default configuration for LLM inference.
    ///
    /// Hot: 4 GB, Warm: 4 GB, demotion after 1s idle.
    /// These values cap memory usage to ~8 GB total regardless of model size.
    pub fn llm_default() -> Self {
        Self {
            hot_budget_bytes: 4 * 1024 * 1024 * 1024,    // 4 GB
            warm_budget_bytes: 4 * 1024 * 1024 * 1024,    // 4 GB
            demotion_idle_secs: 1,
        }
    }

    /// Minimal config for small models (e.g., Qwen3-8B fits entirely in hot).
    pub fn small_model() -> Self {
        Self {
            hot_budget_bytes: 8 * 1024 * 1024 * 1024,    // 8 GB
            warm_budget_bytes: 2 * 1024 * 1024 * 1024,    // 2 GB
            demotion_idle_secs: 30,
        }
    }

    /// Tight config for maximum efficiency (e.g., 671B MoE on limited RAM).
    pub fn tight() -> Self {
        Self {
            hot_budget_bytes: 2 * 1024 * 1024 * 1024,    // 2 GB
            warm_budget_bytes: 2 * 1024 * 1024 * 1024,    // 2 GB
            demotion_idle_secs: 0,  // Demote immediately
        }
    }
}

/// Tracks the state of a tensor in the tier system.
struct TensorState {
    tier: Tier,
    last_accessed: Instant,
    access_count: u64,
    pinned: bool,
    /// For warm/hot: the mmap backing this tensor's file.
    mmap: Option<Arc<memmap2::Mmap>>,
    /// For hot: the resident allocation (if we explicitly read into RAM).
    resident_data: Option<Vec<u8>>,
}

/// Manages tensor tier transitions with mode-capped budgets.
pub struct TierManager {
    config: TierConfig,
    /// Per-tensor state: (model_id, tensor_name) → TensorState
    states: RwLock<HashMap<(String, String), TensorState>>,
    /// Current hot tier usage in bytes.
    hot_bytes: AtomicU64,
    /// Current warm tier usage in bytes.
    warm_bytes: AtomicU64,
    /// Total cold bytes (everything not hot or warm).
    cold_bytes: AtomicU64,
    /// LRU tracking for hot tier eviction: access_time → (model_id, tensor_name, size)
    hot_lru: RwLock<BTreeMap<Instant, (String, String, u64)>>,
    /// Metrics reference
    metrics: Arc<StoreMetrics>,
}

impl TierManager {
    pub fn new(config: TierConfig, metrics: Arc<StoreMetrics>) -> Self {
        Self {
            config,
            states: RwLock::new(HashMap::new()),
            hot_bytes: AtomicU64::new(0),
            warm_bytes: AtomicU64::new(0),
            cold_bytes: AtomicU64::new(0),
            hot_lru: RwLock::new(BTreeMap::new()),
            metrics,
        }
    }

    pub fn config(&self) -> &TierConfig {
        &self.config
    }

    pub fn hot_bytes(&self) -> u64 {
        self.hot_bytes.load(Ordering::Relaxed)
    }

    pub fn warm_bytes(&self) -> u64 {
        self.warm_bytes.load(Ordering::Relaxed)
    }

    pub fn cold_bytes(&self) -> u64 {
        self.cold_bytes.load(Ordering::Relaxed)
    }

    /// Ensure a tensor is accessible (at least warm/mmap'd).
    /// Returns a TensorRef pointing to the data.
    pub fn ensure_accessible(&self, model_id: &str, meta: &TensorMeta) -> Result<TensorRef> {
        let key = (model_id.to_string(), meta.name.clone());

        // Check if already accessible
        {
            let states = self.states.read();
            if let Some(state) = states.get(&key) {
                match state.tier {
                    Tier::Hot => {
                        self.metrics.record_hot_hit();
                        // Update LRU
                        drop(states);
                        self.touch_hot(model_id, &meta.name, meta.size_bytes);
                        return self.make_ref(model_id, meta);
                    }
                    Tier::Warm => {
                        self.metrics.record_warm_hit();
                        return self.make_ref(model_id, meta);
                    }
                    Tier::Cold => {
                        // Fall through to promote
                    }
                }
            }
        }

        // Cold → Warm: mmap the file region
        self.metrics.record_cold_hit();
        self.promote_to_warm(model_id, meta)?;
        self.make_ref(model_id, meta)
    }

    /// Promote a tensor from cold to warm (mmap the file).
    pub fn promote_to_warm(&self, model_id: &str, meta: &TensorMeta) -> Result<()> {
        let key = (model_id.to_string(), meta.name.clone());

        // Check warm budget — evict oldest warm if needed
        let current_warm = self.warm_bytes.load(Ordering::Relaxed);
        if current_warm + meta.size_bytes > self.config.warm_budget_bytes {
            self.evict_warm(meta.size_bytes)?;
        }

        // mmap the file containing this tensor
        let mmap = self.mmap_tensor_file(meta)?;
        let mmap = Arc::new(mmap);

        let mut states = self.states.write();
        states.insert(key, TensorState {
            tier: Tier::Warm,
            last_accessed: Instant::now(),
            access_count: 1,
            pinned: false,
            mmap: Some(mmap),
            resident_data: None,
        });
        self.warm_bytes.fetch_add(meta.size_bytes, Ordering::Relaxed);
        self.cold_bytes.fetch_sub(
            meta.size_bytes.min(self.cold_bytes.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );

        Ok(())
    }

    /// Prefetch a tensor to warm tier (non-blocking mmap + madvise WILLNEED).
    pub fn prefetch_warm(&self, model_id: &str, meta: &TensorMeta) -> Result<()> {
        let key = (model_id.to_string(), meta.name.clone());

        // Already accessible? Skip.
        {
            let states = self.states.read();
            if let Some(state) = states.get(&key) {
                if state.tier != Tier::Cold {
                    return Ok(());
                }
            }
        }

        // mmap + advise willneed (non-blocking read-ahead)
        self.promote_to_warm(model_id, meta)?;

        // Advise the OS to start reading pages
        let states = self.states.read();
        if let Some(state) = states.get(&key) {
            if let Some(ref mmap) = state.mmap {
                let offset = meta.file_offset as usize;
                let len = meta.size_bytes as usize;
                if offset + len <= mmap.len() {
                    unsafe {
                        libc::madvise(
                            mmap.as_ptr().add(offset) as *mut libc::c_void,
                            len,
                            libc::MADV_WILLNEED,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Pin a tensor in the hot tier (won't be evicted by LRU).
    pub fn pin_hot(&self, model_id: &str, meta: &TensorMeta) -> Result<()> {
        self.ensure_accessible(model_id, meta)?;

        let key = (model_id.to_string(), meta.name.clone());
        let mut states = self.states.write();
        if let Some(state) = states.get_mut(&key) {
            state.pinned = true;
        }
        Ok(())
    }

    /// Demote a tensor back to cold (drop mmap, advise DONTNEED).
    pub fn demote_to_cold(&self, model_id: &str, meta: &TensorMeta) -> Result<()> {
        let key = (model_id.to_string(), meta.name.clone());
        let mut states = self.states.write();

        if let Some(state) = states.get_mut(&key) {
            if state.pinned {
                return Ok(()); // Don't evict pinned tensors
            }

            // Advise OS to drop pages
            if let Some(ref mmap) = state.mmap {
                let offset = meta.file_offset as usize;
                let len = meta.size_bytes as usize;
                if offset + len <= mmap.len() {
                    unsafe {
                        libc::madvise(
                            mmap.as_ptr().add(offset) as *mut libc::c_void,
                            len,
                            libc::MADV_DONTNEED,
                        );
                    }
                }
            }

            match state.tier {
                Tier::Hot => {
                    self.hot_bytes.fetch_sub(meta.size_bytes, Ordering::Relaxed);
                    // Remove from LRU
                    let mut lru = self.hot_lru.write();
                    lru.retain(|_, (m, t, _)| !(m == model_id && t == &meta.name));
                }
                Tier::Warm => {
                    self.warm_bytes.fetch_sub(meta.size_bytes, Ordering::Relaxed);
                }
                Tier::Cold => return Ok(()),
            }

            state.tier = Tier::Cold;
            state.mmap = None;
            state.resident_data = None;
            self.cold_bytes.fetch_add(meta.size_bytes, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Register a tensor as cold (initial state after indexing).
    pub fn register_cold(&self, model_id: &str, meta: &TensorMeta) {
        let key = (model_id.to_string(), meta.name.clone());
        let mut states = self.states.write();
        states.entry(key).or_insert_with(|| TensorState {
            tier: Tier::Cold,
            last_accessed: Instant::now(),
            access_count: 0,
            pinned: false,
            mmap: None,
            resident_data: None,
        });
        self.cold_bytes.fetch_add(meta.size_bytes, Ordering::Relaxed);
    }

    // --- Internal helpers ---

    fn touch_hot(&self, model_id: &str, tensor_name: &str, size: u64) {
        let mut lru = self.hot_lru.write();
        // Remove old entry
        lru.retain(|_, (m, t, _)| !(m == model_id && t == tensor_name));
        // Insert with new timestamp
        lru.insert(Instant::now(), (model_id.to_string(), tensor_name.to_string(), size));
    }

    fn evict_warm(&self, needed: u64) -> Result<()> {
        // Simple: drop oldest warm entries until we have space
        // In production, this would be LRU-based
        let mut freed = 0u64;
        let mut states = self.states.write();
        let mut to_demote = Vec::new();

        for ((model_id, tensor_name), state) in states.iter() {
            if freed >= needed { break; }
            if state.tier == Tier::Warm && !state.pinned {
                to_demote.push((model_id.clone(), tensor_name.clone()));
                // We don't know exact size here without meta, but we can approximate
                freed += 1024 * 1024; // Conservative: assume 1MB per tensor
            }
        }

        for (m, t) in to_demote {
            if let Some(state) = states.get_mut(&(m.clone(), t.clone())) {
                state.tier = Tier::Cold;
                state.mmap = None;
            }
        }

        Ok(())
    }

    fn mmap_tensor_file(&self, meta: &TensorMeta) -> Result<memmap2::Mmap> {
        let file = std::fs::File::open(&meta.file_path).map_err(|e| {
            Error::Io { operation: "open".into(), message: format!("'{}': {}", meta.file_path.display(), e), #[cfg(feature = "std")] source: None }
        })?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::Io { operation: "mmap".into(), message: format!("'{}': {}", meta.file_path.display(), e), #[cfg(feature = "std")] source: None })?
        };
        Ok(mmap)
    }

    fn make_ref(&self, model_id: &str, meta: &TensorMeta) -> Result<TensorRef> {
        let key = (model_id.to_string(), meta.name.clone());
        let states = self.states.read();
        let state = states.get(&key).ok_or_else(|| {
            Error::Internal { message: format!("tensor state not found: {}", meta.name), #[cfg(feature = "std")] backtrace: None }
        })?;

        let mmap = state.mmap.as_ref().ok_or_else(|| {
            Error::Internal { message: format!("tensor '{}' has no backing mmap", meta.name), #[cfg(feature = "std")] backtrace: None }
        })?;

        let offset = meta.file_offset as usize;
        let len = meta.size_bytes as usize;

        if offset + len > mmap.len() {
            return Err(Error::InvalidArgument {
                name: "tensor_ref".into(),
                message: format!(
                    "tensor '{}' offset+size ({} + {}) exceeds file size ({})",
                    meta.name, offset, len, mmap.len()
                ),
            });
        }

        Ok(TensorRef {
            data: unsafe { mmap.as_ptr().add(offset) },
            len,
            _mmap: Some(mmap.clone()),
            meta: meta.clone(),
        })
    }
}
