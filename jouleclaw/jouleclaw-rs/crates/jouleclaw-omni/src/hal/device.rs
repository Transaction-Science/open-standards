//! Device types and traits for hardware abstraction.

use super::capabilities::{Capabilities, MemoryInfo};
use crate::core::{DType, Result};
use alloc::string::String;

/// Type of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU (with optional SIMD/AMX)
    Cpu,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Apple GPU via Metal
    Metal,
    /// AMD GPU via ROCm/HIP
    Rocm,
    /// Cross-platform via Vulkan
    Vulkan,
    /// RISC-V accelerator (e.g., Tenstorrent)
    RiscV,
}

impl core::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Metal => write!(f, "metal"),
            Self::Rocm => write!(f, "rocm"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::RiscV => write!(f, "riscv"),
        }
    }
}

/// Unique identifier for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId {
    /// Device type
    pub device_type: DeviceType,
    /// Device index within type
    pub index: u32,
}

impl DeviceId {
    /// Create a new device ID.
    pub const fn new(device_type: DeviceType, index: u32) -> Self {
        Self { device_type, index }
    }

    /// CPU device (index 0).
    pub const fn cpu() -> Self {
        Self::new(DeviceType::Cpu, 0)
    }

    /// First CUDA device.
    pub const fn cuda(index: u32) -> Self {
        Self::new(DeviceType::Cuda, index)
    }

    /// Metal device (typically only one).
    pub const fn metal() -> Self {
        Self::new(DeviceType::Metal, 0)
    }
}

impl core::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}", self.device_type, self.index)
    }
}

/// Information about a discovered device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub id: DeviceId,
    /// Human-readable device name
    pub name: String,
    /// Device type
    pub device_type: DeviceType,
    /// Memory information
    pub memory: MemoryInfo,
    /// Hardware capabilities
    pub capabilities: Capabilities,
}

impl DeviceInfo {
    /// Check if device supports a data type efficiently.
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        match dtype {
            DType::F32 | DType::I32 | DType::I64 | DType::I8 | DType::U8 | DType::U32 | DType::Bool => true,
            DType::F16 => self.capabilities.supports_f16,
            DType::BF16 => self.capabilities.supports_bf16,
            DType::F8E4M3 | DType::F8E5M2 => self.capabilities.supports_f8,
        }
    }

    /// Check if device has matrix multiplication units.
    pub fn has_matrix_unit(&self) -> bool {
        self.capabilities.matrix_unit.is_some()
    }

    /// Estimate TFLOPS for this device (rough approximation).
    pub fn estimated_tflops(&self, dtype: DType) -> f64 {
        let base = match self.device_type {
            DeviceType::Cpu => 0.5,
            DeviceType::Cuda => 30.0,
            DeviceType::Metal => 15.0,
            DeviceType::Rocm => 25.0,
            DeviceType::Vulkan => 10.0,
            DeviceType::RiscV => 5.0,
        };

        // Adjust for data type
        let multiplier = match dtype {
            DType::F8E4M3 | DType::F8E5M2 => 4.0,
            DType::F16 | DType::BF16 => 2.0,
            DType::F32 => 1.0,
            DType::I8 => 4.0,
            _ => 0.5,
        };

        base * multiplier * (self.capabilities.compute_units as f64)
    }
}

/// Trait for compute devices.
///
/// This is the main abstraction over different hardware backends.
/// Implementations handle device-specific memory management and
/// kernel execution.
pub trait Device: Send + Sync {
    /// Get device information.
    fn info(&self) -> &DeviceInfo;

    /// Allocate memory on the device.
    fn allocate(&self, size: usize) -> Result<super::DeviceBuffer>;

    /// Free device memory.
    fn free(&self, buffer: super::DeviceBuffer) -> Result<()>;

    /// Copy data from host to device.
    fn copy_to_device(&self, src: &[u8], dst: &mut super::DeviceBuffer) -> Result<()>;

    /// Copy data from device to host.
    fn copy_to_host(&self, src: &super::DeviceBuffer, dst: &mut [u8]) -> Result<()>;

    /// Copy data between device buffers.
    fn copy_device_to_device(
        &self,
        src: &super::DeviceBuffer,
        dst: &mut super::DeviceBuffer,
    ) -> Result<()>;

    /// Synchronize device (wait for all operations to complete).
    fn synchronize(&self) -> Result<()>;

    /// Execute a compiled kernel.
    fn execute_kernel(
        &self,
        kernel: &super::Kernel,
        args: &super::KernelArgs,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<()>;
}
