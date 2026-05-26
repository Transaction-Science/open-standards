//! Resource efficiency monitoring.
//!
//! Provides real-time tracking of:
//! - Memory usage (allocated, cached, mapped)
//! - GPU utilization
//! - Thermal state (for Apple Silicon throttling)
//! - Power efficiency metrics

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Resource monitor for tracking system efficiency.
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Start time
    start_time: Instant,
    /// Memory stats
    memory: MemoryMonitor,
    /// Compute stats
    compute: ComputeMonitor,
    /// Power/thermal (Apple Silicon specific)
    #[cfg(target_os = "macos")]
    power: PowerMonitor,
}

impl ResourceMonitor {
    /// Create a new resource monitor.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            memory: MemoryMonitor::new(),
            compute: ComputeMonitor::new(),
            #[cfg(target_os = "macos")]
            power: PowerMonitor::new(),
        }
    }

    /// Get memory monitor.
    pub fn memory(&self) -> &MemoryMonitor {
        &self.memory
    }

    /// Get compute monitor.
    pub fn compute(&self) -> &ComputeMonitor {
        &self.compute
    }

    /// Get power monitor (macOS only).
    #[cfg(target_os = "macos")]
    pub fn power(&self) -> &PowerMonitor {
        &self.power
    }

    /// Get uptime.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get a snapshot of all stats.
    pub fn snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot {
            uptime: self.uptime(),
            memory: self.memory.stats(),
            compute: self.compute.stats(),
            #[cfg(target_os = "macos")]
            power: self.power.stats(),
        }
    }

    /// Print summary to tracing.
    pub fn log_summary(&self) {
        let snap = self.snapshot();

        tracing::info!(
            "Resource Summary: uptime={:?}, mem_used={}MB, mem_peak={}MB, ops={}",
            snap.uptime,
            snap.memory.used / (1024 * 1024),
            snap.memory.peak / (1024 * 1024),
            snap.compute.operations,
        );

        #[cfg(target_os = "macos")]
        tracing::info!(
            "Power: thermal={:?}, efficiency={:.2} ops/joule",
            snap.power.thermal_state,
            snap.compute.operations as f64 / snap.power.energy_joules.max(0.001),
        );
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage monitor.
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Currently allocated bytes
    allocated: AtomicUsize,
    /// Peak allocation
    peak: AtomicUsize,
    /// Bytes in cache
    cached: AtomicUsize,
    /// Memory-mapped bytes
    mmap: AtomicUsize,
    /// Allocation count
    alloc_count: AtomicUsize,
    /// Free count
    free_count: AtomicUsize,
}

impl MemoryMonitor {
    /// Create a new memory monitor.
    pub fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            cached: AtomicUsize::new(0),
            mmap: AtomicUsize::new(0),
            alloc_count: AtomicUsize::new(0),
            free_count: AtomicUsize::new(0),
        }
    }

    /// Record an allocation.
    pub fn record_alloc(&self, bytes: usize) {
        let new = self.allocated.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(new, Ordering::Relaxed);
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a free.
    pub fn record_free(&self, bytes: usize) {
        self.allocated.fetch_sub(bytes, Ordering::Relaxed);
        self.free_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache addition.
    pub fn record_cache(&self, bytes: usize) {
        self.cached.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record mmap.
    pub fn record_mmap(&self, bytes: usize) {
        self.mmap.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get current stats.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            used: self.allocated.load(Ordering::Relaxed),
            peak: self.peak.load(Ordering::Relaxed),
            cached: self.cached.load(Ordering::Relaxed),
            mmap: self.mmap.load(Ordering::Relaxed),
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            free_count: self.free_count.load(Ordering::Relaxed),
        }
    }

    /// Check if within budget.
    pub fn is_within_budget(&self, budget: usize) -> bool {
        self.allocated.load(Ordering::Relaxed) <= budget
    }

    /// Get efficiency ratio (how much of mmap is actually loaded).
    pub fn efficiency_ratio(&self) -> f64 {
        let mmap = self.mmap.load(Ordering::Relaxed);
        if mmap == 0 {
            1.0
        } else {
            self.allocated.load(Ordering::Relaxed) as f64 / mmap as f64
        }
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory statistics.
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Currently used bytes
    pub used: usize,
    /// Peak usage
    pub peak: usize,
    /// Cached bytes
    pub cached: usize,
    /// Memory-mapped bytes
    pub mmap: usize,
    /// Allocation count
    pub alloc_count: usize,
    /// Free count
    pub free_count: usize,
}

impl MemoryStats {
    /// Lazy loading efficiency (lower is better - less loaded).
    pub fn lazy_efficiency(&self) -> f64 {
        if self.mmap == 0 {
            1.0
        } else {
            self.used as f64 / self.mmap as f64
        }
    }
}

/// Compute operation monitor.
#[derive(Debug)]
pub struct ComputeMonitor {
    /// Total operations
    operations: AtomicU64,
    /// Total FLOPS
    flops: AtomicU64,
    /// Total compute time (nanoseconds)
    compute_time_ns: AtomicU64,
    /// Kernel dispatch count
    kernel_dispatches: AtomicU64,
}

impl ComputeMonitor {
    /// Create a new compute monitor.
    pub fn new() -> Self {
        Self {
            operations: AtomicU64::new(0),
            flops: AtomicU64::new(0),
            compute_time_ns: AtomicU64::new(0),
            kernel_dispatches: AtomicU64::new(0),
        }
    }

    /// Record an operation.
    pub fn record_op(&self, flops: u64, time_ns: u64) {
        self.operations.fetch_add(1, Ordering::Relaxed);
        self.flops.fetch_add(flops, Ordering::Relaxed);
        self.compute_time_ns.fetch_add(time_ns, Ordering::Relaxed);
    }

    /// Record kernel dispatch.
    pub fn record_dispatch(&self) {
        self.kernel_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    /// Get stats.
    pub fn stats(&self) -> ComputeStats {
        let ops = self.operations.load(Ordering::Relaxed);
        let flops = self.flops.load(Ordering::Relaxed);
        let time_ns = self.compute_time_ns.load(Ordering::Relaxed);
        let dispatches = self.kernel_dispatches.load(Ordering::Relaxed);

        ComputeStats {
            operations: ops,
            total_flops: flops,
            compute_time: Duration::from_nanos(time_ns),
            kernel_dispatches: dispatches,
            tflops: if time_ns > 0 {
                (flops as f64 / 1e12) / (time_ns as f64 / 1e9)
            } else {
                0.0
            },
        }
    }
}

impl Default for ComputeMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute statistics.
#[derive(Debug, Clone)]
pub struct ComputeStats {
    /// Total operations
    pub operations: u64,
    /// Total FLOPS
    pub total_flops: u64,
    /// Total compute time
    pub compute_time: Duration,
    /// Kernel dispatch count
    pub kernel_dispatches: u64,
    /// Achieved TFLOPS
    pub tflops: f64,
}

/// Power and thermal monitor (macOS/Apple Silicon).
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct PowerMonitor {
    /// Estimated energy consumed (joules)
    energy_joules: std::sync::atomic::AtomicU64,
    /// Sample count
    samples: AtomicU64,
    /// Last thermal state
    last_thermal: parking_lot::RwLock<ThermalState>,
}

#[cfg(target_os = "macos")]
impl PowerMonitor {
    /// Create a new power monitor.
    pub fn new() -> Self {
        Self {
            energy_joules: std::sync::atomic::AtomicU64::new(0),
            samples: AtomicU64::new(0),
            last_thermal: parking_lot::RwLock::new(ThermalState::Nominal),
        }
    }

    /// Sample current thermal state.
    pub fn sample_thermal(&self) -> ThermalState {
        let state = get_thermal_state();
        *self.last_thermal.write() = state;
        self.samples.fetch_add(1, Ordering::Relaxed);
        state
    }

    /// Record energy consumption estimate.
    pub fn record_energy(&self, joules: f64) {
        let millijoules = (joules * 1000.0) as u64;
        self.energy_joules.fetch_add(millijoules, Ordering::Relaxed);
    }

    /// Get stats.
    pub fn stats(&self) -> PowerStats {
        PowerStats {
            energy_joules: self.energy_joules.load(Ordering::Relaxed) as f64 / 1000.0,
            samples: self.samples.load(Ordering::Relaxed),
            thermal_state: *self.last_thermal.read(),
        }
    }

    /// Check if throttling.
    pub fn is_throttling(&self) -> bool {
        matches!(
            *self.last_thermal.read(),
            ThermalState::Critical | ThermalState::Serious
        )
    }
}

#[cfg(target_os = "macos")]
impl Default for PowerMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Thermal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    /// Normal operation
    Nominal,
    /// Slightly warm
    Fair,
    /// Throttling may occur
    Serious,
    /// Heavy throttling
    Critical,
}

#[cfg(target_os = "macos")]
fn get_thermal_state() -> ThermalState {
    // Use NSProcessInfo thermalState
    // For now, return nominal
    ThermalState::Nominal
}

/// Power statistics.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct PowerStats {
    /// Estimated energy consumption (joules)
    pub energy_joules: f64,
    /// Sample count
    pub samples: u64,
    /// Current thermal state
    pub thermal_state: ThermalState,
}

/// Complete resource snapshot.
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Uptime
    pub uptime: Duration,
    /// Memory stats
    pub memory: MemoryStats,
    /// Compute stats
    pub compute: ComputeStats,
    /// Power stats (macOS only)
    #[cfg(target_os = "macos")]
    pub power: PowerStats,
}

impl ResourceSnapshot {
    /// Calculate overall efficiency score.
    pub fn efficiency_score(&self) -> f64 {
        let memory_eff = 1.0 - self.memory.lazy_efficiency().min(1.0);
        let compute_eff = self.compute.tflops / 20.0; // Normalize to ~20 TFLOPS max

        (memory_eff + compute_eff) / 2.0
    }

    /// Format as string.
    pub fn summary(&self) -> String {
        format!(
            "Uptime: {:?}\n\
             Memory: {:.1}MB used, {:.1}MB peak, {:.1}% lazy efficiency\n\
             Compute: {} ops, {:.2} TFLOPS, {} dispatches",
            self.uptime,
            self.memory.used as f64 / (1024.0 * 1024.0),
            self.memory.peak as f64 / (1024.0 * 1024.0),
            self.memory.lazy_efficiency() * 100.0,
            self.compute.operations,
            self.compute.tflops,
            self.compute.kernel_dispatches,
        )
    }
}

/// Scoped timer for measuring operations.
pub struct ScopedTimer<'a> {
    monitor: &'a ComputeMonitor,
    start: Instant,
    flops: u64,
}

impl<'a> ScopedTimer<'a> {
    /// Start a new timer.
    pub fn new(monitor: &'a ComputeMonitor, estimated_flops: u64) -> Self {
        Self {
            monitor,
            start: Instant::now(),
            flops: estimated_flops,
        }
    }
}

impl Drop for ScopedTimer<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.monitor.record_op(self.flops, elapsed.as_nanos() as u64);
    }
}
