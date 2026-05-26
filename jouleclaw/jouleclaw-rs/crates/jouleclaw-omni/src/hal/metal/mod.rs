//! Metal backend for Apple Silicon.
//!
//! ## Unified Memory Architecture (UMA)
//!
//! Apple Silicon uses UMA where CPU and GPU share the same physical memory.
//! This eliminates the need for explicit data transfers:
//!
//! ```text
//! Traditional GPU (NVIDIA/AMD):
//! ┌─────────┐    PCIe     ┌─────────┐
//! │   CPU   │ ◄──────────►│   GPU   │
//! │  Memory │   Copy!     │  VRAM   │
//! └─────────┘             └─────────┘
//!
//! Apple Silicon UMA:
//! ┌─────────────────────────────────┐
//! │        Unified Memory           │
//! │   CPU ◄──────────────► GPU      │
//! │        Zero Copy!               │
//! └─────────────────────────────────┘
//! ```
//!
//! ## Key Optimizations
//!
//! 1. **Zero-Copy Buffers**: Use `MTLStorageModeShared` for CPU/GPU access
//! 2. **Lazy Loading**: Memory-map weights, GPU accesses trigger page faults
//! 3. **Resource Heaps**: Pre-allocate memory pools for fast allocation
//! 4. **Argument Buffers**: Reduce CPU overhead for kernel dispatch

#[cfg(feature = "metal")]
mod device;
#[cfg(feature = "metal")]
mod buffer;
#[cfg(feature = "metal")]
mod compute;
#[cfg(feature = "metal")]
pub mod shader;
#[cfg(feature = "metal")]
mod heap;
#[cfg(feature = "metal")]
pub mod lazy_load;
#[cfg(feature = "metal")]
mod ops;

#[cfg(feature = "metal")]
pub use device::MetalDevice;
#[cfg(feature = "metal")]
pub use buffer::{MetalBuffer, BufferPool, BorrowedMetalBuffer};
#[cfg(feature = "metal")]
pub use compute::{MetalCompute, ComputePipeline};
#[cfg(feature = "metal")]
pub use heap::MetalHeap;
#[cfg(feature = "metal")]
pub use lazy_load::{LazyTensor, LazyLoader, QuantizedTensor};
#[cfg(feature = "metal")]
pub use ops::MetalOps;

/// Apple Silicon chip generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleSiliconGen {
    /// M1 family (original, Pro, Max, Ultra)
    M1,
    /// M2 family
    M2,
    /// M3 family (with ray tracing)
    M3,
    /// M4 family (enhanced Neural Engine)
    M4,
    /// M5 family (latest)
    M5,
    /// Unknown/future
    Unknown,
}

impl AppleSiliconGen {
    /// Detect chip generation from device name.
    pub fn detect(name: &str) -> Self {
        if name.contains("M5") {
            Self::M5
        } else if name.contains("M4") {
            Self::M4
        } else if name.contains("M3") {
            Self::M3
        } else if name.contains("M2") {
            Self::M2
        } else if name.contains("M1") {
            Self::M1
        } else {
            Self::Unknown
        }
    }

    /// Memory bandwidth in GB/s (approximate).
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        match self {
            Self::M1 => 200.0,      // M1 Max: 400
            Self::M2 => 200.0,      // M2 Max: 400
            Self::M3 => 300.0,      // M3 Max: 400
            Self::M4 => 400.0,      // Estimated
            Self::M5 => 500.0,      // Estimated
            Self::Unknown => 100.0,
        }
    }

    /// Whether chip supports hardware ray tracing.
    pub fn supports_raytracing(&self) -> bool {
        matches!(self, Self::M3 | Self::M4 | Self::M5)
    }

    /// Neural Engine TOPS (approximate).
    pub fn neural_engine_tops(&self) -> f64 {
        match self {
            Self::M1 => 11.0,
            Self::M2 => 15.8,
            Self::M3 => 18.0,
            Self::M4 => 38.0,
            Self::M5 => 50.0,  // Estimated
            Self::Unknown => 10.0,
        }
    }

    /// GPU core count range (base model).
    pub fn gpu_cores_base(&self) -> usize {
        match self {
            Self::M1 => 8,
            Self::M2 => 10,
            Self::M3 => 10,
            Self::M4 => 10,
            Self::M5 => 12,  // Estimated
            Self::Unknown => 8,
        }
    }
}

/// Metal storage mode for buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    /// Shared between CPU and GPU (UMA optimal)
    Shared,
    /// Private to GPU only
    Private,
    /// Managed (system handles coherency)
    Managed,
    /// Memoryless (tile memory only)
    Memoryless,
}

impl Default for StorageMode {
    fn default() -> Self {
        // Shared is optimal for UMA
        Self::Shared
    }
}

/// Resource options for Metal buffers.
#[derive(Debug, Clone, Copy)]
pub struct ResourceOptions {
    /// Storage mode
    pub storage_mode: StorageMode,
    /// CPU cache mode
    pub cpu_cache_mode: CpuCacheMode,
    /// Hazard tracking mode
    pub hazard_tracking: HazardTracking,
}

impl Default for ResourceOptions {
    fn default() -> Self {
        Self {
            storage_mode: StorageMode::Shared,
            cpu_cache_mode: CpuCacheMode::WriteCombined,
            hazard_tracking: HazardTracking::Untracked,
        }
    }
}

impl ResourceOptions {
    /// Options optimized for read-only weights.
    pub fn weights() -> Self {
        Self {
            storage_mode: StorageMode::Shared,
            cpu_cache_mode: CpuCacheMode::WriteCombined,
            hazard_tracking: HazardTracking::Untracked,
        }
    }

    /// Options optimized for activations (frequent GPU writes).
    pub fn activations() -> Self {
        Self {
            storage_mode: StorageMode::Private,
            cpu_cache_mode: CpuCacheMode::DefaultCache,
            hazard_tracking: HazardTracking::Untracked,
        }
    }

    /// Options for staging buffers.
    pub fn staging() -> Self {
        Self {
            storage_mode: StorageMode::Shared,
            cpu_cache_mode: CpuCacheMode::DefaultCache,
            hazard_tracking: HazardTracking::Tracked,
        }
    }
}

/// CPU cache mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuCacheMode {
    /// Default caching
    DefaultCache,
    /// Write-combined (optimal for streaming writes)
    WriteCombined,
}

/// Hazard tracking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardTracking {
    /// Automatic hazard tracking
    Tracked,
    /// Manual synchronization (faster)
    Untracked,
}

/// Memory statistics for monitoring.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalMemoryStats {
    /// Current allocated bytes
    pub allocated: usize,
    /// Peak allocation
    pub peak: usize,
    /// Bytes in heaps
    pub heap_allocated: usize,
    /// Bytes memory-mapped (lazy loaded)
    pub mmap_bytes: usize,
    /// Active buffer count
    pub buffer_count: usize,
    /// Bytes waiting to be freed
    pub pending_free: usize,
}

impl MetalMemoryStats {
    /// Total memory in use.
    pub fn total_in_use(&self) -> usize {
        self.allocated + self.heap_allocated
    }

    /// Efficiency ratio (lower mmap ratio = more loaded into memory).
    pub fn lazy_load_ratio(&self) -> f64 {
        if self.mmap_bytes == 0 {
            0.0
        } else {
            self.allocated as f64 / self.mmap_bytes as f64
        }
    }
}
