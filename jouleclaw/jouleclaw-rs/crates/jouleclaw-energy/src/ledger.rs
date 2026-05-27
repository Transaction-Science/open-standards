//! Append-only energy ledger — the single source of truth for energy
//! consumption across a JouleClaw deployment.
//!
//! Every operational layer (TLS handshake, WASM execution, cascade
//! resolution, storage IO, etc.) deposits energy here via
//! [`EnergyLedger::record`]. Reads are O(1) thanks to a separately
//! maintained `total_uj` atomic.
//!
//! The ledger holds *integer microjoules only* — floating-point is
//! reserved for derived quantities (carbon, joules). This matches the
//! JouleClaw protocol's determinism guarantee.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ── Operational Layers ─────────────────────────────────────────────

/// Operational layer that consumed energy. Every microjoule deposited
/// in the ledger is attributed to exactly one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum OperationalLayer {
    /// TLS handshake (ECDHE key exchange, cert validation).
    TlsHandshake,
    /// JWP frame encode/decode (header + CBOR serialization).
    JwpFraming,
    /// Authentication (challenge-response, HMAC, signature verify).
    Authentication,
    /// WASM module execution (fuel-metered).
    WasmExecution,
    /// Container/VM lifecycle (start, stop, checkpoint).
    ContainerRuntime,
    /// Native binary process dispatch.
    ProcessDispatch,
    /// Cascade decision (oracle / cortex).
    CortexDecision,
    /// LLM inference (token generation).
    Inference,
    /// Block/object storage read.
    StorageRead,
    /// Block/object storage write.
    StorageWrite,
    /// KV store operations.
    KvOperation,
    /// Mesh gossip protocol (peer state sync).
    MeshGossip,
    /// Heartbeat / keepalive.
    MeshHeartbeat,
    /// Node discovery (mDNS, bootstrap).
    MeshDiscovery,
    /// Bulk data transfer between nodes.
    DataTransfer,
    /// Orchestrator evaluation cycle.
    SchedulerCycle,
    /// Migration: checkpoint + transfer + restore.
    Migration,
    /// Raft consensus round.
    ConsensusRound,
    /// CRDT merge / sync.
    CrdtSync,
    /// Hashing (SHA-256, BLAKE3).
    HashComputation,
    /// Digital signature creation or verification.
    SignatureOperation,
    /// Symmetric encryption/decryption (AES, ChaCha20).
    Encryption,
    /// Telemetry collection and export.
    Telemetry,
    /// Distributed tracing span processing.
    Tracing,
    /// JWP command decode + routing + response encode.
    CommandDispatch,
}

impl OperationalLayer {
    /// All variants, for iteration.
    pub const ALL: &[OperationalLayer] = &[
        Self::TlsHandshake,
        Self::JwpFraming,
        Self::Authentication,
        Self::WasmExecution,
        Self::ContainerRuntime,
        Self::ProcessDispatch,
        Self::CortexDecision,
        Self::Inference,
        Self::StorageRead,
        Self::StorageWrite,
        Self::KvOperation,
        Self::MeshGossip,
        Self::MeshHeartbeat,
        Self::MeshDiscovery,
        Self::DataTransfer,
        Self::SchedulerCycle,
        Self::Migration,
        Self::ConsensusRound,
        Self::CrdtSync,
        Self::HashComputation,
        Self::SignatureOperation,
        Self::Encryption,
        Self::Telemetry,
        Self::Tracing,
        Self::CommandDispatch,
    ];

    const fn index(self) -> usize {
        match self {
            Self::TlsHandshake => 0,
            Self::JwpFraming => 1,
            Self::Authentication => 2,
            Self::WasmExecution => 3,
            Self::ContainerRuntime => 4,
            Self::ProcessDispatch => 5,
            Self::CortexDecision => 6,
            Self::Inference => 7,
            Self::StorageRead => 8,
            Self::StorageWrite => 9,
            Self::KvOperation => 10,
            Self::MeshGossip => 11,
            Self::MeshHeartbeat => 12,
            Self::MeshDiscovery => 13,
            Self::DataTransfer => 14,
            Self::SchedulerCycle => 15,
            Self::Migration => 16,
            Self::ConsensusRound => 17,
            Self::CrdtSync => 18,
            Self::HashComputation => 19,
            Self::SignatureOperation => 20,
            Self::Encryption => 21,
            Self::Telemetry => 22,
            Self::Tracing => 23,
            Self::CommandDispatch => 24,
        }
    }

    /// Human-readable category grouping for dashboards.
    pub fn category(&self) -> &'static str {
        match self {
            Self::TlsHandshake | Self::JwpFraming | Self::Authentication => "protocol",
            Self::WasmExecution | Self::ContainerRuntime | Self::ProcessDispatch => "compute",
            Self::CortexDecision | Self::Inference => "ai",
            Self::StorageRead | Self::StorageWrite | Self::KvOperation => "storage",
            Self::MeshGossip | Self::MeshHeartbeat | Self::MeshDiscovery | Self::DataTransfer => {
                "network"
            }
            Self::SchedulerCycle | Self::Migration => "scheduling",
            Self::ConsensusRound | Self::CrdtSync => "consensus",
            Self::HashComputation | Self::SignatureOperation | Self::Encryption => "crypto",
            Self::Telemetry | Self::Tracing => "observability",
            Self::CommandDispatch => "dispatch",
        }
    }
}

/// Number of operational layers (compile-time constant).
const LAYER_COUNT: usize = 25;

// ── Energy Ledger ──────────────────────────────────────────────────

/// Thread-safe lock-free ledger of energy deposits per operational
/// layer. Hot path is two atomic adds.
pub struct EnergyLedger {
    layers: [AtomicU64; LAYER_COUNT],
    ops: [AtomicU64; LAYER_COUNT],
    total_uj: AtomicU64,
    total_ops: AtomicU64,
    carbon_gco2_per_kwh: AtomicU64, // f64 bits
}

impl EnergyLedger {
    /// Build a fresh ledger with all counters at zero.
    pub fn new() -> Self {
        Self {
            layers: std::array::from_fn(|_| AtomicU64::new(0)),
            ops: std::array::from_fn(|_| AtomicU64::new(0)),
            total_uj: AtomicU64::new(0),
            total_ops: AtomicU64::new(0),
            carbon_gco2_per_kwh: AtomicU64::new(0),
        }
    }

    /// Record energy (μJ) consumed by a layer. Hot path.
    #[inline]
    pub fn record(&self, layer: OperationalLayer, microjoules: u64) {
        let idx = layer.index();
        self.layers[idx].fetch_add(microjoules, Ordering::Relaxed);
        self.ops[idx].fetch_add(1, Ordering::Relaxed);
        self.total_uj.fetch_add(microjoules, Ordering::Relaxed);
        self.total_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an explicit batch (energy + operation count).
    #[inline]
    pub fn record_batch(&self, layer: OperationalLayer, microjoules: u64, op_count: u64) {
        let idx = layer.index();
        self.layers[idx].fetch_add(microjoules, Ordering::Relaxed);
        self.ops[idx].fetch_add(op_count, Ordering::Relaxed);
        self.total_uj.fetch_add(microjoules, Ordering::Relaxed);
        self.total_ops.fetch_add(op_count, Ordering::Relaxed);
    }

    /// Total energy in microjoules.
    #[inline]
    pub fn total_uj(&self) -> u64 {
        self.total_uj.load(Ordering::Relaxed)
    }

    /// Total energy in microwatt-hours. Useful for wire-protocol headers.
    #[inline]
    pub fn total_uwh(&self) -> u64 {
        self.total_uj.load(Ordering::Relaxed) / 3600
    }

    /// Total energy in joules (f64 — derived quantity).
    #[inline]
    pub fn total_joules(&self) -> f64 {
        self.total_uj.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Total operation count across all layers.
    #[inline]
    pub fn total_ops(&self) -> u64 {
        self.total_ops.load(Ordering::Relaxed)
    }

    /// Energy consumed by a single layer (μJ).
    #[inline]
    pub fn layer_uj(&self, layer: OperationalLayer) -> u64 {
        self.layers[layer.index()].load(Ordering::Relaxed)
    }

    /// Operation count for a single layer.
    #[inline]
    pub fn layer_ops(&self, layer: OperationalLayer) -> u64 {
        self.ops[layer.index()].load(Ordering::Relaxed)
    }

    /// Set the grid carbon intensity (gCO2 / kWh).
    pub fn set_carbon_intensity(&self, gco2_per_kwh: f64) {
        let bits = gco2_per_kwh.to_bits();
        self.carbon_gco2_per_kwh.store(bits, Ordering::Relaxed);
    }

    /// Current carbon intensity.
    pub fn carbon_intensity_gco2_per_kwh(&self) -> f64 {
        f64::from_bits(self.carbon_gco2_per_kwh.load(Ordering::Relaxed))
    }

    /// Total carbon emissions (gCO2e) using the configured intensity.
    pub fn total_carbon_gco2e(&self) -> f64 {
        let energy_uj = self.total_uj() as f64;
        let intensity = self.carbon_intensity_gco2_per_kwh();
        energy_uj * intensity / 3_600_000_000_000.0
    }

    /// Snapshot the ledger.
    pub fn snapshot(&self) -> LedgerSnapshot {
        let mut layers = Vec::with_capacity(LAYER_COUNT);
        for &layer in OperationalLayer::ALL {
            let idx = layer.index();
            let energy_uj = self.layers[idx].load(Ordering::Relaxed);
            let ops = self.ops[idx].load(Ordering::Relaxed);
            if energy_uj > 0 || ops > 0 {
                layers.push(LayerSnapshot {
                    layer,
                    category: layer.category().to_string(),
                    energy_uj,
                    ops,
                });
            }
        }
        let total_uj = self.total_uj();
        LedgerSnapshot {
            total_uj,
            total_uwh: total_uj / 3600,
            total_joules: total_uj as f64 / 1_000_000.0,
            total_ops: self.total_ops(),
            carbon_gco2_per_kwh: self.carbon_intensity_gco2_per_kwh(),
            total_carbon_gco2e: self.total_carbon_gco2e(),
            layers,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        }
    }

    /// Category-level aggregation of a snapshot.
    pub fn category_breakdown(&self) -> Vec<CategoryBreakdown> {
        let snapshot = self.snapshot();
        let total = snapshot.total_uj.max(1) as f64;

        let mut categories: std::collections::BTreeMap<String, (u64, u64)> =
            std::collections::BTreeMap::new();
        for ls in &snapshot.layers {
            let entry = categories.entry(ls.category.clone()).or_insert((0, 0));
            entry.0 += ls.energy_uj;
            entry.1 += ls.ops;
        }

        categories
            .into_iter()
            .map(|(cat, (energy_uj, ops))| CategoryBreakdown {
                category: cat,
                energy_uj,
                ops,
                pct: (energy_uj as f64 / total) * 100.0,
            })
            .collect()
    }

    /// Reset all counters. For tests / epoch boundaries.
    pub fn reset(&self) {
        for i in 0..LAYER_COUNT {
            self.layers[i].store(0, Ordering::Relaxed);
            self.ops[i].store(0, Ordering::Relaxed);
        }
        self.total_uj.store(0, Ordering::Relaxed);
        self.total_ops.store(0, Ordering::Relaxed);
    }
}

impl Default for EnergyLedger {
    fn default() -> Self {
        Self::new()
    }
}

// ── Snapshot Types ─────────────────────────────────────────────────

/// Point-in-time snapshot of the entire ledger.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LedgerSnapshot {
    /// Total energy in microjoules.
    pub total_uj: u64,
    /// Total energy in microwatt-hours (for wire headers).
    pub total_uwh: u64,
    /// Total energy in joules (for billing/display).
    pub total_joules: f64,
    /// Total operation count.
    pub total_ops: u64,
    /// Grid carbon intensity (gCO2 / kWh).
    pub carbon_gco2_per_kwh: f64,
    /// Total carbon emissions (gCO2e).
    pub total_carbon_gco2e: f64,
    /// Active layers (only those with any deposit).
    pub layers: Vec<LayerSnapshot>,
    /// Snapshot time (Unix ns).
    pub timestamp_ns: u64,
}

impl LedgerSnapshot {
    /// Cumulative μWh — convenience accessor for wire headers.
    pub fn cumulative_uwh(&self) -> u64 {
        self.total_uwh
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{:.3} J | {} µWh | {:.6} gCO₂e | {} ops across {} layers",
            self.total_joules,
            self.total_uwh,
            self.total_carbon_gco2e,
            self.total_ops,
            self.layers.len()
        )
    }
}

/// Snapshot of one operational layer.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LayerSnapshot {
    /// Layer identity.
    pub layer: OperationalLayer,
    /// Category grouping.
    pub category: String,
    /// Cumulative energy in μJ.
    pub energy_uj: u64,
    /// Operation count.
    pub ops: u64,
}

impl LayerSnapshot {
    /// Average μJ per operation.
    pub fn avg_uj_per_op(&self) -> f64 {
        if self.ops == 0 {
            0.0
        } else {
            self.energy_uj as f64 / self.ops as f64
        }
    }
}

/// Category-level aggregation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CategoryBreakdown {
    /// Category name.
    pub category: String,
    /// Total energy in μJ.
    pub energy_uj: u64,
    /// Total operations.
    pub ops: u64,
    /// Share of total energy (0–100).
    pub pct: f64,
}

// ── Default cost constants ────────────────────────────────────────
// Conservative upper-bound estimates used when hardware metering is
// unavailable. Calibrated against RAPL on Xeon and Apple Silicon
// `powermetrics` (inv-energy donor measurements).

/// Default per-operation energy cost estimates (μJ).
pub mod costs {
    /// TLS 1.3 ECDHE-P256 handshake (~300 μJ measured).
    pub const TLS_HANDSHAKE_UJ: u64 = 300;
    /// JWP frame encode/decode (~5 μJ).
    pub const JWP_FRAME_UJ: u64 = 5;
    /// HMAC-SHA256 challenge-response (~15 μJ).
    pub const AUTH_HMAC_UJ: u64 = 15;
    /// ECDSA P-256 signature verify (~50 μJ).
    pub const AUTH_ECDSA_VERIFY_UJ: u64 = 50;
    /// WASM fuel unit → μJ conversion factor.
    pub const WASM_FUEL_TO_UJ: f64 = 0.1;
    /// 4 KiB NVMe block read (~2 μJ).
    pub const BLOCK_READ_4K_UJ: u64 = 2;
    /// 4 KiB NVMe block write (~8 μJ).
    pub const BLOCK_WRITE_4K_UJ: u64 = 8;
    /// In-memory KV op (~1 μJ).
    pub const KV_OP_UJ: u64 = 1;
    /// Gossip pull round (~20 μJ).
    pub const GOSSIP_ROUND_UJ: u64 = 20;
    /// Heartbeat (~3 μJ).
    pub const HEARTBEAT_UJ: u64 = 3;
    /// Per-byte network transfer (~0.5 nJ).
    pub const TRANSFER_NJ_PER_BYTE: f64 = 0.5;
    /// One orchestrator cycle (~100 μJ).
    pub const SCHEDULER_CYCLE_UJ: u64 = 100;
    /// Migration checkpoint + transfer base (~10,000 μJ).
    pub const MIGRATION_BASE_UJ: u64 = 10_000;
    /// Single Raft round (~30 μJ).
    pub const RAFT_ROUND_UJ: u64 = 30;
    /// CRDT merge (~10 μJ).
    pub const CRDT_MERGE_UJ: u64 = 10;
    /// SHA-256 hash of 1 KiB (~3 μJ).
    pub const SHA256_1K_UJ: u64 = 3;
    /// BLAKE3 hash of 1 KiB (~1 μJ).
    pub const BLAKE3_1K_UJ: u64 = 1;
    /// AES-256-GCM encrypt 1 KiB (~2 μJ).
    pub const AES_ENCRYPT_1K_UJ: u64 = 2;
    /// Telemetry event emission (~5 μJ).
    pub const TELEMETRY_EVENT_UJ: u64 = 5;
    /// Tracing span processing (~3 μJ).
    pub const TRACING_SPAN_UJ: u64 = 3;
    /// JWP command dispatch (~10 μJ).
    pub const COMMAND_DISPATCH_UJ: u64 = 10;

    /// Bulk transfer cost.
    pub fn transfer_uj(bytes: u64) -> u64 {
        ((bytes as f64) * TRANSFER_NJ_PER_BYTE / 1000.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_read() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::TlsHandshake, 300);
        ledger.record(OperationalLayer::Authentication, 15);
        ledger.record(OperationalLayer::JwpFraming, 5);
        assert_eq!(ledger.total_uj(), 320);
        assert_eq!(ledger.total_ops(), 3);
        assert_eq!(ledger.layer_uj(OperationalLayer::TlsHandshake), 300);
    }

    #[test]
    fn uwh_conversion() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::WasmExecution, 7200);
        assert_eq!(ledger.total_uwh(), 2);
    }

    #[test]
    fn joules_conversion() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::Inference, 1_000_000);
        assert!((ledger.total_joules() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn carbon_accounting_kwh_scale() {
        let ledger = EnergyLedger::new();
        ledger.set_carbon_intensity(400.0);
        ledger.record(OperationalLayer::WasmExecution, 3_600_000_000_000);
        assert!((ledger.total_carbon_gco2e() - 400.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_only_active_layers() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::TlsHandshake, 100);
        ledger.record(OperationalLayer::MeshHeartbeat, 3);
        let snap = ledger.snapshot();
        assert_eq!(snap.layers.len(), 2);
        assert_eq!(snap.total_uj, 103);
    }

    #[test]
    fn category_breakdown_sums_correctly() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::TlsHandshake, 300);
        ledger.record(OperationalLayer::JwpFraming, 5);
        ledger.record(OperationalLayer::Authentication, 15);
        ledger.record(OperationalLayer::WasmExecution, 1000);
        let cats = ledger.category_breakdown();
        let protocol = cats.iter().find(|c| c.category == "protocol").expect("protocol");
        assert_eq!(protocol.energy_uj, 320);
        assert_eq!(protocol.ops, 3);
        let compute = cats.iter().find(|c| c.category == "compute").expect("compute");
        assert_eq!(compute.energy_uj, 1000);
    }

    #[test]
    fn batch_recording() {
        let ledger = EnergyLedger::new();
        ledger.record_batch(OperationalLayer::DataTransfer, 5000, 100);
        assert_eq!(ledger.layer_uj(OperationalLayer::DataTransfer), 5000);
        assert_eq!(ledger.layer_ops(OperationalLayer::DataTransfer), 100);
        assert_eq!(ledger.total_ops(), 100);
    }

    #[test]
    fn reset_clears_all() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::Inference, 999);
        ledger.reset();
        assert_eq!(ledger.total_uj(), 0);
        assert_eq!(ledger.layer_uj(OperationalLayer::Inference), 0);
    }

    #[test]
    fn all_layers_enumerated() {
        assert_eq!(OperationalLayer::ALL.len(), LAYER_COUNT);
        for (i, layer) in OperationalLayer::ALL.iter().enumerate() {
            assert_eq!(layer.index(), i);
        }
    }

    #[test]
    fn category_labels() {
        assert_eq!(OperationalLayer::TlsHandshake.category(), "protocol");
        assert_eq!(OperationalLayer::WasmExecution.category(), "compute");
        assert_eq!(OperationalLayer::CortexDecision.category(), "ai");
        assert_eq!(OperationalLayer::StorageRead.category(), "storage");
        assert_eq!(OperationalLayer::MeshGossip.category(), "network");
        assert_eq!(OperationalLayer::SchedulerCycle.category(), "scheduling");
        assert_eq!(OperationalLayer::ConsensusRound.category(), "consensus");
        assert_eq!(OperationalLayer::HashComputation.category(), "crypto");
        assert_eq!(OperationalLayer::Telemetry.category(), "observability");
        assert_eq!(OperationalLayer::CommandDispatch.category(), "dispatch");
    }

    #[test]
    fn avg_uj_per_op() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::CommandDispatch, 100);
        ledger.record(OperationalLayer::CommandDispatch, 200);
        let snap = ledger.snapshot();
        let cmd = snap
            .layers
            .iter()
            .find(|l| l.layer == OperationalLayer::CommandDispatch)
            .expect("dispatch layer");
        assert_eq!(cmd.ops, 2);
        assert!((cmd.avg_uj_per_op() - 150.0).abs() < 1e-10);
    }

    #[test]
    fn concurrent_recording() {
        use std::sync::Arc;
        let ledger = Arc::new(EnergyLedger::new());
        let mut handles = vec![];
        for _ in 0..10 {
            let l = ledger.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    l.record(OperationalLayer::JwpFraming, 5);
                }
            }));
        }
        for h in handles {
            h.join().expect("join");
        }
        assert_eq!(ledger.total_uj(), 50_000);
        assert_eq!(ledger.total_ops(), 10_000);
    }

    #[test]
    fn cost_constants_internally_consistent() {
        assert!(costs::TLS_HANDSHAKE_UJ > costs::HEARTBEAT_UJ);
        assert!(costs::AUTH_ECDSA_VERIFY_UJ > costs::AUTH_HMAC_UJ);
        assert!(costs::BLOCK_WRITE_4K_UJ > costs::BLOCK_READ_4K_UJ);
        assert!(costs::BLAKE3_1K_UJ < costs::SHA256_1K_UJ);
    }

    #[test]
    fn transfer_cost_calculation() {
        assert_eq!(costs::transfer_uj(1_000_000), 500);
    }
}
