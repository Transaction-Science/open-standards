//! Hardware Abstraction Layer (HAL) for efficient-genai.
//!
//! Provides runtime hardware discovery and trait-based abstraction
//! over different compute backends (CUDA, Metal, ROCm, Vulkan, CPU).
//!
//! ## Apple Silicon (Metal + UMA)
//!
//! For Apple Silicon, we provide specialized support:
//! - **Unified Memory**: Zero-copy between CPU and GPU
//! - **Lazy Loading**: Memory-map model weights, load on demand
//! - **Metal Shaders**: Optimized kernels using simdgroup operations
//!
//! See [`metal`] module for details.

mod device;
mod capabilities;
mod memory;
mod kernel;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "metal")]
pub use self::metal::{
    AppleSiliconGen, MetalDevice, MetalBuffer, BufferPool,
    MetalCompute, MetalHeap, LazyLoader, LazyTensor,
    ResourceOptions, StorageMode, MetalMemoryStats,
};

pub use device::{Device, DeviceId, DeviceInfo, DeviceType};
pub use capabilities::{Capabilities, MatrixUnit, MemoryInfo};
pub use memory::{DeviceBuffer, DevicePtr, MemoryPool};
pub use kernel::{Kernel, KernelBuilder, KernelArgs};

use crate::core::{Error, Result};
use alloc::vec::Vec;

/// Discover all available compute devices at runtime.
///
/// This function probes all supported backends and returns
/// information about available hardware.
pub fn discover_devices() -> Result<Vec<DeviceInfo>> {
    let mut devices = Vec::new();

    // Always add CPU
    devices.push(discover_cpu());

    // Probe GPU backends based on platform
    #[cfg(feature = "cuda")]
    devices.extend(discover_cuda()?);

    #[cfg(feature = "metal")]
    devices.extend(discover_metal()?);

    #[cfg(feature = "rocm")]
    devices.extend(discover_rocm()?);

    #[cfg(feature = "vulkan")]
    devices.extend(discover_vulkan()?);

    Ok(devices)
}

fn discover_cpu() -> DeviceInfo {
    let num_cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    DeviceInfo {
        id: DeviceId::new(DeviceType::Cpu, 0),
        name: "CPU".into(),
        device_type: DeviceType::Cpu,
        memory: MemoryInfo {
            total: get_system_memory(),
            available: get_available_memory(),
            bandwidth_gbps: 50.0, // Typical DDR5
        },
        capabilities: Capabilities {
            compute_units: num_cores,
            max_threads_per_unit: 1,
            supports_f16: true,
            supports_bf16: cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64"),
            supports_f8: false,
            supports_int8: true,
            matrix_unit: None, // TODO: Detect AMX on Intel
            max_shared_memory: 0,
            warp_size: 1,
        },
    }
}

#[cfg(feature = "cuda")]
fn discover_cuda() -> Result<Vec<DeviceInfo>> {
    use cudarc::driver::CudaDevice;

    let mut devices = Vec::new();

    let count = cudarc::driver::result::device::get_count()
        .map_err(|e| Error::device_not_available("CUDA", format!("{e}")))?;

    for i in 0..count {
        let device = CudaDevice::new(i)
            .map_err(|e| Error::device_not_available("CUDA", format!("{e}")))?;

        let name = device.name()
            .map_err(|e| Error::device_not_available("CUDA", format!("{e}")))?;

        let (total_mem, free_mem) = device.memory_info()
            .map_err(|e| Error::device_not_available("CUDA", format!("{e}")))?;

        // Detect architecture capabilities
        let (major, minor) = device.compute_capability();
        let is_ampere_or_newer = major >= 8;
        let is_hopper_or_newer = major >= 9;

        devices.push(DeviceInfo {
            id: DeviceId::new(DeviceType::Cuda, i as u32),
            name,
            device_type: DeviceType::Cuda,
            memory: MemoryInfo {
                total: total_mem,
                available: free_mem,
                bandwidth_gbps: estimate_cuda_bandwidth(major, minor),
            },
            capabilities: Capabilities {
                compute_units: device.num_sms() as usize,
                max_threads_per_unit: 2048,
                supports_f16: true,
                supports_bf16: is_ampere_or_newer,
                supports_f8: is_hopper_or_newer,
                supports_int8: true,
                matrix_unit: Some(MatrixUnit {
                    m: if is_hopper_or_newer { 16 } else { 16 },
                    n: if is_hopper_or_newer { 16 } else { 8 },
                    k: if is_hopper_or_newer { 16 } else { 16 },
                }),
                max_shared_memory: device.shared_memory_per_block() as usize,
                warp_size: 32,
            },
        });
    }

    Ok(devices)
}

#[cfg(feature = "cuda")]
fn estimate_cuda_bandwidth(major: u32, minor: u32) -> f64 {
    match (major, minor) {
        (9, _) => 3350.0,  // H100 HBM3
        (8, 9) => 2040.0,  // RTX 4090
        (8, 6) => 768.0,   // RTX 3090
        (8, 0) => 2039.0,  // A100
        _ => 500.0,
    }
}

#[cfg(feature = "metal")]
fn discover_metal() -> Result<Vec<DeviceInfo>> {
    let device = ::metal::Device::system_default()
        .ok_or_else(|| Error::device_not_available("Metal", "no Metal device found"))?;

    let name = device.name().to_string();

    // Apple Silicon has unified memory
    let total_mem = get_system_memory();
    let available = get_available_memory();

    // Detect chip generation from name — bandwidth, matrix unit, and GPU core count
    let (bandwidth, matrix_unit, gpu_cores) = if name.contains("M4 Max") {
        (546.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 40)
    } else if name.contains("M4 Pro") {
        (273.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 20)
    } else if name.contains("M4") {
        (120.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 10)
    } else if name.contains("M3 Max") {
        (400.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 40)
    } else if name.contains("M3 Pro") {
        (200.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 18)
    } else if name.contains("M3") {
        (100.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 10)
    } else if name.contains("M2 Ultra") {
        (800.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 76)
    } else if name.contains("M2 Max") {
        (400.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 38)
    } else if name.contains("M2 Pro") {
        (200.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 19)
    } else if name.contains("M2") {
        (100.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 10)
    } else if name.contains("M1 Ultra") {
        (800.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 64)
    } else if name.contains("M1 Max") {
        (400.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 32)
    } else if name.contains("M1 Pro") {
        (200.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 16)
    } else if name.contains("M1") {
        (68.0, Some(MatrixUnit { m: 32, n: 32, k: 32 }), 8)
    } else {
        (100.0, None, 10)
    };

    Ok(vec![DeviceInfo {
        id: DeviceId::new(DeviceType::Metal, 0),
        name,
        device_type: DeviceType::Metal,
        memory: MemoryInfo {
            total: total_mem,
            available,
            bandwidth_gbps: bandwidth,
        },
        capabilities: Capabilities {
            compute_units: gpu_cores,
            max_threads_per_unit: 1024,
            supports_f16: true,
            supports_bf16: true,
            supports_f8: false,
            supports_int8: true,
            matrix_unit,
            max_shared_memory: 32768,
            warp_size: 32, // SIMD group size
        },
    }])
}

#[cfg(feature = "rocm")]
fn discover_rocm() -> Result<Vec<DeviceInfo>> {
    // TODO: Implement HIP device discovery
    Ok(Vec::new())
}

#[cfg(feature = "vulkan")]
fn discover_vulkan() -> Result<Vec<DeviceInfo>> {
    // TODO: Implement Vulkan device discovery
    Ok(Vec::new())
}

fn get_system_memory() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(8 * 1024 * 1024 * 1024)
    }

    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|s| s.parse::<usize>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(8 * 1024 * 1024 * 1024)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        8 * 1024 * 1024 * 1024 // 8 GB default
    }
}

fn get_available_memory() -> usize {
    // Simplified: return 80% of system memory
    (get_system_memory() * 8) / 10
}
