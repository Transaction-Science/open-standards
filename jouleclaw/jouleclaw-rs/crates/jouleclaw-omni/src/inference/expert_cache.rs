//! Expert caching with frequency-based hot set management.
//!
//! Tracks per-expert access frequency across layers and maintains a "hot set"
//! of frequently-accessed experts. On Apple Silicon UMA, weights are memory-mapped
//! but routing and dispatch still have overhead. Keeping a hot set lets the next
//! upgrade (predictive pre-loading) skip routing for high-confidence experts.

use std::collections::HashSet;

/// Configuration for expert caching.
#[derive(Debug, Clone)]
pub struct ExpertCacheConfig {
    /// Enable expert caching (default: false).
    pub enable_expert_cache: bool,
    /// Number of experts to keep in the hot set per layer (default: 4).
    pub hot_experts_per_layer: usize,
    /// Exponential decay factor applied to frequencies each step (default: 0.95).
    pub frequency_decay: f32,
}

impl Default for ExpertCacheConfig {
    fn default() -> Self {
        Self {
            enable_expert_cache: false,
            hot_experts_per_layer: 4,
            frequency_decay: 0.95,
        }
    }
}

/// Per-layer expert frequency tracker and hot set.
struct LayerExpertState {
    /// Access frequency per expert (exponentially decayed).
    frequencies: Vec<f64>,
    /// Current hot set of expert IDs.
    hot_set: HashSet<usize>,
}

/// Expert cache tracking frequency and hot set across all layers.
pub struct ExpertCache {
    config: ExpertCacheConfig,
    /// Per-layer state: frequencies and hot sets.
    layers: Vec<LayerExpertState>,
    /// Total cache lookups (is_hot calls).
    total_lookups: u64,
    /// Total cache hits (is_hot returned true).
    total_hits: u64,
}

impl ExpertCache {
    /// Create a new expert cache.
    ///
    /// # Arguments
    /// * `config` - Cache configuration
    /// * `num_layers` - Number of transformer layers
    /// * `num_experts` - Number of experts per MoE layer
    pub fn new(config: ExpertCacheConfig, num_layers: usize, num_experts: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerExpertState {
                frequencies: vec![0.0; num_experts],
                hot_set: HashSet::new(),
            })
            .collect();

        Self {
            config,
            layers,
            total_lookups: 0,
            total_hits: 0,
        }
    }

    /// Record an expert access, incrementing its frequency counter.
    pub fn record_access(&mut self, layer: usize, expert_id: usize) {
        if let Some(state) = self.layers.get_mut(layer) {
            if let Some(freq) = state.frequencies.get_mut(expert_id) {
                *freq += 1.0;
            }
        }
    }

    /// Check if an expert is in the hot set for a given layer.
    /// Also tracks hit/miss statistics.
    pub fn is_hot(&mut self, layer: usize, expert_id: usize) -> bool {
        self.total_lookups += 1;
        if let Some(state) = self.layers.get(layer) {
            let hit = state.hot_set.contains(&expert_id);
            if hit {
                self.total_hits += 1;
            }
            hit
        } else {
            false
        }
    }

    /// Recompute the hot set for a layer from current frequencies.
    /// Selects the top-k experts by frequency.
    pub fn update_hot_set(&mut self, layer: usize) {
        let k = self.config.hot_experts_per_layer;
        if let Some(state) = self.layers.get_mut(layer) {
            // Build (expert_id, frequency) pairs and sort descending
            let mut ranked: Vec<(usize, f64)> = state
                .frequencies
                .iter()
                .enumerate()
                .map(|(id, &freq)| (id, freq))
                .collect();
            ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            state.hot_set.clear();
            for &(id, freq) in ranked.iter().take(k) {
                // Only include experts that have actually been accessed
                if freq > 0.0 {
                    state.hot_set.insert(id);
                }
            }
        }
    }

    /// Apply exponential decay to all frequencies across all layers.
    /// Call this periodically (e.g., once per generation step) to let
    /// recently-accessed experts rise above stale ones.
    pub fn decay_frequencies(&mut self) {
        let decay = self.config.frequency_decay as f64;
        for state in &mut self.layers {
            for freq in &mut state.frequencies {
                *freq *= decay;
            }
        }
    }

    /// Return the overall cache hit rate (fraction of is_hot calls that returned true).
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        self.total_hits as f64 / self.total_lookups as f64
    }

    /// Return total number of lookups.
    pub fn total_lookups(&self) -> u64 {
        self.total_lookups
    }

    /// Return total number of hits.
    pub fn total_hits(&self) -> u64 {
        self.total_hits
    }

    /// Get the hot set for a given layer (read-only).
    pub fn hot_set(&self, layer: usize) -> Option<&HashSet<usize>> {
        self.layers.get(layer).map(|s| &s.hot_set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_frequency_tracking() {
        let config = ExpertCacheConfig {
            enable_expert_cache: true,
            hot_experts_per_layer: 2,
            frequency_decay: 0.95,
        };
        let mut cache = ExpertCache::new(config, 2, 8);

        // Record accesses: experts 3 and 5 are popular on layer 0
        for _ in 0..10 {
            cache.record_access(0, 3);
            cache.record_access(0, 5);
        }
        for _ in 0..2 {
            cache.record_access(0, 1);
        }

        cache.update_hot_set(0);

        // Experts 3 and 5 should be hot
        assert!(cache.is_hot(0, 3));
        assert!(cache.is_hot(0, 5));
        // Expert 1 should not be (only top-2)
        assert!(!cache.is_hot(0, 1));
    }

    #[test]
    fn test_hit_rate() {
        let config = ExpertCacheConfig {
            enable_expert_cache: true,
            hot_experts_per_layer: 1,
            frequency_decay: 0.95,
        };
        let mut cache = ExpertCache::new(config, 1, 4);

        cache.record_access(0, 2);
        cache.update_hot_set(0);

        // Hit
        assert!(cache.is_hot(0, 2));
        // Miss
        assert!(!cache.is_hot(0, 0));

        assert!((cache.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_decay() {
        let config = ExpertCacheConfig {
            enable_expert_cache: true,
            hot_experts_per_layer: 1,
            frequency_decay: 0.5,
        };
        let mut cache = ExpertCache::new(config, 1, 2);

        cache.record_access(0, 0);
        cache.record_access(0, 0); // freq = 2.0

        cache.decay_frequencies(); // freq = 1.0
        cache.decay_frequencies(); // freq = 0.5

        // Now access expert 1 once (freq = 1.0 > 0.5)
        cache.record_access(0, 1);
        cache.update_hot_set(0);

        // Expert 1 should now be hotter than expert 0
        assert!(cache.is_hot(0, 1));
        assert!(!cache.is_hot(0, 0));
    }

    #[test]
    fn test_empty_hot_set_when_no_accesses() {
        let config = ExpertCacheConfig {
            enable_expert_cache: true,
            hot_experts_per_layer: 4,
            frequency_decay: 0.95,
        };
        let mut cache = ExpertCache::new(config, 1, 8);

        cache.update_hot_set(0);
        assert!(!cache.is_hot(0, 0));
        assert_eq!(cache.hit_rate(), 0.0);
    }
}
