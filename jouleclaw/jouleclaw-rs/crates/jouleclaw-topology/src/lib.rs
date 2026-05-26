//! # jouleclaw-topology
//!
//! Hardware discovery and characterization. Phase 0 declares the data
//! structures; Phase 1.2 implements Apple Silicon discovery.
//!
//! See spec 03.

pub mod apple;

use jouleclaw_core::backend::{BackendId, Capabilities};
use jouleclaw_core::energy::{ComputeUnitId, EnergySourceId, InterconnectId, MemoryTierId};
use jouleclaw_core::op::OpKind;
use jouleclaw_core::tensor::Dtype;
use std::collections::HashMap;
use std::time::Duration;

/// Top-level topology structure. Built once at startup, queried by the
/// scheduler and memory planner.
#[derive(Debug, Clone)]
pub struct Topology {
    pub host: HostInfo,
    pub compute_units: Vec<ComputeUnit>,
    pub memory_tiers: Vec<MemoryTier>,
    pub interconnects: Vec<Interconnect>,
    pub energy_sources: Vec<EnergySource>,
}

#[derive(Debug, Clone)]
pub struct HostInfo {
    pub os: String,        // "Darwin", "Linux", ...
    pub arch: String,      // "aarch64", "x86_64", ...
    pub hostname: String,
    pub kernel_version: String,
}

#[derive(Debug, Clone)]
pub struct ComputeUnit {
    pub id: ComputeUnitId,
    pub backend: BackendId,
    pub capabilities: Capabilities,
    pub peak_flops: HashMap<Dtype, f64>,
    pub measured_flops: HashMap<Dtype, f64>,
    pub joules_per_flop: HashMap<Dtype, f64>,
    pub bandwidth_to: HashMap<MemoryTierId, BandwidthSpec>,
    pub latency_to: HashMap<MemoryTierId, Duration>,
}

#[derive(Debug, Clone, Copy)]
pub struct BandwidthSpec {
    pub bytes_per_second_read: u64,
    pub bytes_per_second_write: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryTier {
    pub id: MemoryTierId,
    pub name: String,
    pub bytes_capacity: u64,
    pub bytes_per_second: u64,
    pub joules_per_byte_read: f64,
    pub joules_per_byte_write: f64,
    pub volatility: Volatility,
    pub access_pattern: AccessPattern,
}

#[derive(Debug, Clone, Copy)]
pub enum Volatility { Volatile, Persistent }

#[derive(Debug, Clone, Copy)]
pub enum AccessPattern { Sequential, Random, Streaming, Mixed }

#[derive(Debug, Clone)]
pub struct Interconnect {
    pub id: InterconnectId,
    pub kind: InterconnectKind,
    pub endpoints: (NodeRef, NodeRef),
    pub bytes_per_second: u64,
    pub latency: Duration,
    pub joules_per_byte: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterconnectKind {
    /// Apple unified memory; effectively zero-cost between CPU/GPU/ANE.
    SharedMem,
    Pcie,
    NvLink,
    Cxl,
    Network,
}

#[derive(Debug, Clone)]
pub enum NodeRef {
    Compute(ComputeUnitId),
    Memory(MemoryTierId),
}

#[derive(Debug, Clone)]
pub struct EnergySource {
    pub id: EnergySourceId,
    pub name: String,
    pub covers: Vec<ComputeUnitId>,
    pub counter: EnergyCounter,
}

/// A live-readable energy counter. Phase 1 implementations:
/// - `IOReportEnergyCounter` (macOS / Apple Silicon)
/// - `RaplEnergyCounter` (Linux x86)
/// - `NvmlEnergyCounter` (NVIDIA)
#[derive(Debug, Clone)]
pub enum EnergyCounter {
    /// Phase 0 placeholder; Phase 1 replaces with real handles.
    Placeholder { description: String },
}

/// Reasons for which a kernel may be considered unsupported.
#[derive(Debug)]
pub enum DispatchUnsupported {
    NoBackendForOp(OpKind),
    DtypeUnsupported { op: OpKind, dtype: Dtype, backend: BackendId },
    DeterministicVariantUnavailable { op: OpKind, backend: BackendId },
}

/// Discover the running machine's topology.
///
/// Tries Apple Silicon detection first; falls through to a minimal
/// host-info-only topology on other platforms (Phase 2 will add x86/Linux
/// discovery).
pub fn discover() -> Topology {
    if let Some(t) = apple::discover_apple_silicon() {
        return t;
    }
    Topology {
        host: HostInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hostname: String::new(),
            kernel_version: String::new(),
        },
        compute_units: Vec::new(),
        memory_tiers: Vec::new(),
        interconnects: Vec::new(),
        energy_sources: Vec::new(),
    }
}
