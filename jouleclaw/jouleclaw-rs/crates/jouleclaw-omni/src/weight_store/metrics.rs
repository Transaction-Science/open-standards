//! Access metrics and statistics for the weight store.

use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks access patterns and tier hit rates.
pub struct StoreMetrics {
    total_tensors: AtomicU64,
    total_accesses: AtomicU64,
    hot_hits: AtomicU64,
    warm_hits: AtomicU64,
    cold_hits: AtomicU64,
}

impl StoreMetrics {
    pub fn new() -> Self {
        Self {
            total_tensors: AtomicU64::new(0),
            total_accesses: AtomicU64::new(0),
            hot_hits: AtomicU64::new(0),
            warm_hits: AtomicU64::new(0),
            cold_hits: AtomicU64::new(0),
        }
    }

    pub fn record_access(&self, _model_id: &str, _tensor_name: &str) {
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_hot_hit(&self) {
        self.hot_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_warm_hit(&self) {
        self.warm_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cold_hit(&self) {
        self.cold_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_tensors(&self) {
        self.total_tensors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn total_tensors(&self) -> u64 {
        self.total_tensors.load(Ordering::Relaxed)
    }

    pub fn total_accesses(&self) -> u64 {
        self.total_accesses.load(Ordering::Relaxed)
    }

    pub fn hot_hits(&self) -> u64 {
        self.hot_hits.load(Ordering::Relaxed)
    }

    pub fn warm_hits(&self) -> u64 {
        self.warm_hits.load(Ordering::Relaxed)
    }

    pub fn cold_hits(&self) -> u64 {
        self.cold_hits.load(Ordering::Relaxed)
    }

    /// Hot tier hit rate as a percentage.
    pub fn hot_hit_rate(&self) -> f64 {
        let total = self.total_accesses();
        if total == 0 { return 0.0; }
        self.hot_hits() as f64 / total as f64 * 100.0
    }
}

/// Per-tensor access statistics.
#[derive(Debug, Clone, Default)]
pub struct AccessStats {
    pub access_count: u64,
    pub last_accessed_ms: u64,
}
