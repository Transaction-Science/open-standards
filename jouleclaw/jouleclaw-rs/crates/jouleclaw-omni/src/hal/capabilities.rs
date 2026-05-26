//! Hardware capability detection and representation.

/// Memory information for a device.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total: usize,
    /// Currently available memory in bytes
    pub available: usize,
    /// Memory bandwidth in GB/s
    pub bandwidth_gbps: f64,
}

impl MemoryInfo {
    /// Get memory utilization as a percentage.
    pub fn utilization(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let used = self.total.saturating_sub(self.available);
        (used as f64 / self.total as f64) * 100.0
    }

    /// Check if there's enough memory for an allocation.
    pub fn can_allocate(&self, bytes: usize) -> bool {
        self.available >= bytes
    }
}

/// Matrix multiplication unit specification.
#[derive(Debug, Clone, Copy)]
pub struct MatrixUnit {
    /// M dimension of matrix unit
    pub m: u32,
    /// N dimension of matrix unit
    pub n: u32,
    /// K dimension of matrix unit
    pub k: u32,
}

impl MatrixUnit {
    /// Tensor core style (NVIDIA Ampere/Hopper)
    pub const fn tensor_core() -> Self {
        Self { m: 16, n: 8, k: 16 }
    }

    /// Apple Neural Engine style
    pub const fn apple_ane() -> Self {
        Self { m: 32, n: 32, k: 32 }
    }

    /// AMD Matrix Core style
    pub const fn amd_matrix() -> Self {
        Self { m: 16, n: 16, k: 16 }
    }

    /// Intel AMX style
    pub const fn intel_amx() -> Self {
        Self { m: 16, n: 64, k: 64 }
    }

    /// Compute optimal tile size for this matrix unit.
    pub fn optimal_tile_size(&self) -> (usize, usize, usize) {
        // Use multiples of the matrix unit dimensions
        let m = (self.m as usize) * 4;
        let n = (self.n as usize) * 4;
        let k = (self.k as usize) * 4;
        (m, n, k)
    }
}

/// Hardware capabilities for a device.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Number of compute units (SMs, CUs, cores)
    pub compute_units: usize,
    /// Maximum threads per compute unit
    pub max_threads_per_unit: usize,
    /// FP16 support
    pub supports_f16: bool,
    /// BF16 support
    pub supports_bf16: bool,
    /// FP8 support (E4M3/E5M2)
    pub supports_f8: bool,
    /// INT8 support
    pub supports_int8: bool,
    /// Matrix multiplication unit (tensor cores, etc.)
    pub matrix_unit: Option<MatrixUnit>,
    /// Maximum shared memory per block (bytes)
    pub max_shared_memory: usize,
    /// Warp/wavefront size
    pub warp_size: usize,
}

impl Capabilities {
    /// Check if device has high-throughput matrix operations.
    pub fn has_fast_matmul(&self) -> bool {
        self.matrix_unit.is_some()
    }

    /// Recommended block size for compute kernels.
    pub fn recommended_block_size(&self) -> usize {
        self.warp_size * 4 // 4 warps per block is often optimal
    }

    /// Estimate parallelism (total concurrent threads).
    pub fn max_parallelism(&self) -> usize {
        self.compute_units * self.max_threads_per_unit
    }

    /// Check if the device supports efficient low-precision compute.
    pub fn supports_low_precision(&self) -> bool {
        self.supports_f16 || self.supports_bf16 || self.supports_int8
    }

    /// Get the best supported floating-point type for this device.
    pub fn best_float_type(&self) -> crate::core::DType {
        if self.supports_f8 {
            crate::core::DType::F8E4M3
        } else if self.supports_bf16 {
            crate::core::DType::BF16
        } else if self.supports_f16 {
            crate::core::DType::F16
        } else {
            crate::core::DType::F32
        }
    }
}

/// Feature flags discovered at runtime.
#[derive(Debug, Clone, Default)]
pub struct RuntimeFeatures {
    /// CUDA compute capability (major.minor)
    pub cuda_compute_capability: Option<(u32, u32)>,
    /// Metal feature set
    pub metal_feature_set: Option<String>,
    /// ROCm architecture (e.g., "gfx1100")
    pub rocm_arch: Option<String>,
    /// CPU SIMD extensions
    pub cpu_features: CpuFeatures,
}

/// CPU SIMD feature detection.
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    /// SSE4.2 support
    pub sse42: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX-512 support
    pub avx512: bool,
    /// Intel AMX support
    pub amx: bool,
    /// ARM NEON support
    pub neon: bool,
    /// ARM SVE support
    pub sve: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse42: std::arch::is_x86_feature_detected!("sse4.2"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                avx512: std::arch::is_x86_feature_detected!("avx512f"),
                amx: std::arch::is_x86_feature_detected!("amx-tile"),
                neon: false,
                sve: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse42: false,
                avx2: false,
                avx512: false,
                amx: false,
                neon: true, // Always available on AArch64
                sve: std::arch::is_aarch64_feature_detected!("sve"),
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// Get the best SIMD width in bits.
    pub fn best_simd_width(&self) -> usize {
        if self.avx512 {
            512
        } else if self.avx2 {
            256
        } else if self.sse42 || self.neon {
            128
        } else {
            64
        }
    }
}
