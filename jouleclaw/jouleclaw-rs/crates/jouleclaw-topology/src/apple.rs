//! Apple Silicon topology discovery.
//!
//! macOS-only implementation. On other platforms this module compiles to
//! no-op stubs that return an empty topology component, so dispatch code
//! elsewhere doesn't need its own cfg gates.

#![allow(dead_code)] // Many helpers are only called on macOS aarch64.

use jouleclaw_core::backend::{BackendId, Capabilities};
use jouleclaw_core::energy::{ComputeUnitId, EnergySourceId, MemoryTierId};
use jouleclaw_core::op::OpKind;
use std::collections::HashMap;

use crate::{
    BandwidthSpec, ComputeUnit, EnergyCounter, EnergySource, HostInfo,
    Interconnect, InterconnectKind, MemoryTier, NodeRef, Topology,
    Volatility, AccessPattern,
};

/// Discover Apple Silicon topology. On non-Apple-Silicon hosts returns
/// `None` so the generic discovery path can fall through.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn discover_apple_silicon() -> Option<Topology> {
    let host = HostInfo {
        os: "Darwin".into(),
        arch: "aarch64".into(),
        hostname: sysctl_string("kern.hostname").unwrap_or_default(),
        kernel_version: sysctl_string("kern.osrelease").unwrap_or_default(),
    };

    // CPU.
    let perf_cores = sysctl_u32("hw.perflevel0.physicalcpu").unwrap_or(0) as usize;
    let efficiency_cores = sysctl_u32("hw.perflevel1.physicalcpu").unwrap_or(0) as usize;
    let _total_cores = sysctl_u32("hw.physicalcpu").unwrap_or(0) as usize;

    let mut compute_units = Vec::new();
    let mut next_cu_id = 0u32;

    if perf_cores > 0 {
        compute_units.push(ComputeUnit {
            id: ComputeUnitId(next_cu_id),
            backend: BackendId::AppleCpuNeon,
            capabilities: standard_cpu_capabilities(),
            peak_flops: HashMap::new(),
            measured_flops: HashMap::new(),
            joules_per_flop: HashMap::new(),
            bandwidth_to: HashMap::new(),
            latency_to: HashMap::new(),
        });
        next_cu_id += 1;

        // AMX coprocessor: present on all M-series chips, programmable
        // indirectly via the Accelerate framework.
        compute_units.push(ComputeUnit {
            id: ComputeUnitId(next_cu_id),
            backend: BackendId::AppleAmx,
            capabilities: standard_amx_capabilities(),
            peak_flops: HashMap::new(),
            measured_flops: HashMap::new(),
            joules_per_flop: HashMap::new(),
            bandwidth_to: HashMap::new(),
            latency_to: HashMap::new(),
        });
        next_cu_id += 1;
    }
    let _ = efficiency_cores; // recorded for future scheduling decisions

    // GPU presence (Metal device).
    if metal_available() {
        compute_units.push(ComputeUnit {
            id: ComputeUnitId(next_cu_id),
            backend: BackendId::AppleGpuMetal,
            capabilities: standard_gpu_capabilities(),
            peak_flops: HashMap::new(),
            measured_flops: HashMap::new(),
            joules_per_flop: HashMap::new(),
            bandwidth_to: HashMap::new(),
            latency_to: HashMap::new(),
        });
        next_cu_id += 1;
    }

    // ANE presence.
    if ane_available() {
        compute_units.push(ComputeUnit {
            id: ComputeUnitId(next_cu_id),
            backend: BackendId::AppleAne,
            capabilities: standard_ane_capabilities(),
            peak_flops: HashMap::new(),
            measured_flops: HashMap::new(),
            joules_per_flop: HashMap::new(),
            bandwidth_to: HashMap::new(),
            latency_to: HashMap::new(),
        });
    }

    // Memory tiers. Apple Silicon's unified memory means the GPU and ANE
    // share DRAM with the CPU; tiers are L1d, L2, SLC, DRAM.
    let l1d_bytes = sysctl_u64("hw.perflevel0.l1dcachesize").unwrap_or(128 * 1024);
    let l2_bytes = sysctl_u64("hw.perflevel0.l2cachesize").unwrap_or(16 * 1024 * 1024);
    let dram_bytes = sysctl_u64("hw.memsize").unwrap_or(8 * 1024 * 1024 * 1024);

    let memory_tiers = vec![
        MemoryTier {
            id: MemoryTierId(0), name: "L1d".into(),
            bytes_capacity: l1d_bytes,
            bytes_per_second: 1_000_000_000_000, // ~1 TB/s typical
            joules_per_byte_read: 1e-12, joules_per_byte_write: 2e-12,
            volatility: Volatility::Volatile,
            access_pattern: AccessPattern::Random,
        },
        MemoryTier {
            id: MemoryTierId(1), name: "L2".into(),
            bytes_capacity: l2_bytes,
            bytes_per_second: 400_000_000_000,
            joules_per_byte_read: 5e-12, joules_per_byte_write: 8e-12,
            volatility: Volatility::Volatile,
            access_pattern: AccessPattern::Random,
        },
        MemoryTier {
            id: MemoryTierId(2), name: "SLC".into(),
            bytes_capacity: 16 * 1024 * 1024,
            bytes_per_second: 200_000_000_000,
            joules_per_byte_read: 2e-11, joules_per_byte_write: 3e-11,
            volatility: Volatility::Volatile,
            access_pattern: AccessPattern::Mixed,
        },
        MemoryTier {
            id: MemoryTierId(3), name: "DRAM".into(),
            bytes_capacity: dram_bytes,
            bytes_per_second: 400_000_000_000, // M-series unified mem is fast
            joules_per_byte_read: 1e-10, joules_per_byte_write: 1.5e-10,
            volatility: Volatility::Volatile,
            access_pattern: AccessPattern::Mixed,
        },
    ];

    // The defining feature of Apple Silicon: unified shared memory.
    let interconnects = vec![
        Interconnect {
            id: jouleclaw_core::energy::InterconnectId(0),
            kind: InterconnectKind::SharedMem,
            endpoints: (NodeRef::Memory(MemoryTierId(3)), NodeRef::Compute(ComputeUnitId(0))),
            bytes_per_second: 400_000_000_000,
            latency: std::time::Duration::from_nanos(50),
            joules_per_byte: 1e-11,
        },
    ];

    // Energy sources: package, GPU, ANE — read via IOReport in Phase 1.3.
    let energy_sources = vec![
        EnergySource {
            id: EnergySourceId(0), name: "package".into(),
            covers: vec![ComputeUnitId(0), ComputeUnitId(1)],
            counter: EnergyCounter::Placeholder { description: "IOReport package".into() },
        },
        EnergySource {
            id: EnergySourceId(1), name: "gpu".into(),
            covers: if metal_available() { vec![ComputeUnitId(2)] } else { vec![] },
            counter: EnergyCounter::Placeholder { description: "IOReport GPU".into() },
        },
        EnergySource {
            id: EnergySourceId(2), name: "ane".into(),
            covers: if ane_available() { vec![ComputeUnitId(3)] } else { vec![] },
            counter: EnergyCounter::Placeholder { description: "IOReport ANE".into() },
        },
    ];

    Some(Topology {
        host, compute_units, memory_tiers, interconnects, energy_sources,
    })
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn discover_apple_silicon() -> Option<Topology> {
    None
}

// ---- macOS-specific FFI helpers ----

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod sys {
    use std::ffi::{c_char, c_int, c_void, CString};

    // Rust 2024 edition requires extern blocks to be marked `unsafe`.
    // See https://doc.rust-lang.org/edition-guide/rust-2024/unsafe-extern.html
    unsafe extern "C" {
        pub fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    pub fn sysctl_u32(name: &str) -> Option<u32> {
        let c_name = CString::new(name).ok()?;
        let mut value: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                &mut value as *mut _ as *mut c_void,
                &mut len,
                std::ptr::null_mut(), 0)
        };
        if rc == 0 { Some(value) } else { None }
    }

    pub fn sysctl_u64(name: &str) -> Option<u64> {
        let c_name = CString::new(name).ok()?;
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe {
            sysctlbyname(
                c_name.as_ptr(),
                &mut value as *mut _ as *mut c_void,
                &mut len,
                std::ptr::null_mut(), 0)
        };
        if rc == 0 { Some(value) } else { None }
    }

    pub fn sysctl_string(name: &str) -> Option<String> {
        let c_name = CString::new(name).ok()?;
        // Probe size first.
        let mut len: usize = 0;
        let rc = unsafe {
            sysctlbyname(c_name.as_ptr(), std::ptr::null_mut(), &mut len,
                std::ptr::null_mut(), 0)
        };
        if rc != 0 || len == 0 { return None; }
        let mut buf = vec![0u8; len];
        let rc = unsafe {
            sysctlbyname(c_name.as_ptr(),
                buf.as_mut_ptr() as *mut c_void, &mut len,
                std::ptr::null_mut(), 0)
        };
        if rc != 0 { return None; }
        // Trim trailing NUL.
        if let Some(&0) = buf.last() { buf.pop(); }
        String::from_utf8(buf).ok()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sysctl_u32(name: &str) -> Option<u32> { sys::sysctl_u32(name) }
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sysctl_u64(name: &str) -> Option<u64> { sys::sysctl_u64(name) }
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sysctl_string(name: &str) -> Option<String> { sys::sysctl_string(name) }

/// Phase 1.2 stop: presence-only detection. Phase 1.3 will use the `metal`
/// crate to query device name, working set size, and command queue capability.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn metal_available() -> bool { true } // Metal is on every Apple Silicon Mac.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn metal_available() -> bool { false }

/// Phase 1.2 stop: presence-only. ANE is on every M-series chip.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ane_available() -> bool { true }

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn ane_available() -> bool { false }

// ---- Capability defaults ----

fn standard_cpu_capabilities() -> Capabilities {
    Capabilities {
        supported_ops: vec![
            OpKind::MatMul, OpKind::Softmax, OpKind::Norm,
            OpKind::Activation, OpKind::Add, OpKind::Mul,
            OpKind::Lookup, OpKind::Sample,
            OpKind::Tokenize, OpKind::Detokenize, OpKind::Regex,
            OpKind::Parse, OpKind::Retrieve, OpKind::TemplateFill,
            OpKind::CacheLookup,
        ],
        supports_deterministic_mode: true,
        supports_async: false,
        max_tensor_bytes: 64 * 1024 * 1024 * 1024,
    }
}

fn standard_amx_capabilities() -> Capabilities {
    Capabilities {
        supported_ops: vec![OpKind::MatMul],
        supports_deterministic_mode: true,
        supports_async: false,
        max_tensor_bytes: 64 * 1024 * 1024 * 1024,
    }
}

fn standard_gpu_capabilities() -> Capabilities {
    Capabilities {
        supported_ops: vec![
            OpKind::MatMul, OpKind::Softmax, OpKind::Norm,
            OpKind::Activation, OpKind::Add, OpKind::Mul,
            OpKind::Lookup,
        ],
        supports_deterministic_mode: true,
        supports_async: true,
        max_tensor_bytes: 32 * 1024 * 1024 * 1024,
    }
}

fn standard_ane_capabilities() -> Capabilities {
    Capabilities {
        supported_ops: vec![OpKind::MatMul, OpKind::Activation],
        supports_deterministic_mode: true,
        supports_async: true,
        max_tensor_bytes: 4 * 1024 * 1024 * 1024,
    }
}

#[allow(dead_code)]
fn unused_bandwidth_spec() -> BandwidthSpec {
    BandwidthSpec { bytes_per_second_read: 0, bytes_per_second_write: 0 }
}
