//! # jouleclaw-energy
//!
//! Honest energy telemetry for JouleClaw.
//!
//! Every energy reading in JouleClaw carries a [`Provenance`] tag declaring
//! how the value was obtained:
//!
//! - [`Provenance::HwShunt`] — a real hardware shunt measurement (Intel/AMD
//!   x86 RAPL MSRs, NVIDIA discrete GPU cumulative-energy counter,
//!   Jetson INA3221 i2c shunts). Honest to the counter's resolution.
//! - [`Provenance::ModelBased`] — a vendor-provided model estimate derived
//!   from frequency, voltage, and utilization (Apple Silicon IOReport,
//!   NVML `power.draw` on older GPUs). Millijoule-quantized at best.
//! - [`Provenance::Estimator`] — a JouleClaw static cost model derived from
//!   architecture × precision × batch tables. Used only on platforms
//!   with no usable hardware counter (consumer AMD GPUs, ARM PMU).
//!
//! The thermodynamic circuit breaker is only as honest as the **worst**
//! counter in a request's span. The breaker MUST surface
//! [`EnergyCounter::resolution_uj`] and [`EnergyCounter::min_window_ns`] so
//! callers can refuse to claim microjoule accuracy on platforms that
//! cannot deliver it.
//!
//! See the JouleClaw spec section "Energy provenance contract" for the
//! normative behavior.

#![warn(missing_docs)]

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Platform backends + ledger live in submodules. Each module owns its
// own `#![forbid(unsafe_code)]` attribute, except `apple` which requires
// an `unsafe extern "C"` block for `libc::getloadavg`.
#[cfg(feature = "rapl")]
pub mod rapl;

#[cfg(feature = "nvml")]
pub mod nvml;

#[cfg(all(feature = "ioreport", target_os = "macos"))]
pub mod apple;

pub mod meter;
pub mod ledger;

/// Provenance of an energy reading. Load-bearing — every consumer of a
/// reading MUST inspect this before claiming accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Provenance {
    /// Real hardware shunt or coulomb counter. Honest to the counter's
    /// resolution. Examples: Intel/AMD x86 RAPL MSRs, NVIDIA discrete
    /// `nvmlDeviceGetTotalEnergyConsumption`, Jetson INA3221 i2c.
    HwShunt,
    /// Vendor-provided model estimate derived from frequency, voltage,
    /// and utilization. Millijoule-quantized at best; not a true
    /// measurement. Examples: Apple IOReport "Energy Model" channels,
    /// NVML `power.draw` interpolation, ROCm-SMI `power1_average`.
    ModelBased,
    /// JouleClaw static cost model from architecture × precision × batch
    /// tables. Used only when no hardware counter exists. Tolerance
    /// bands are wide by construction.
    Estimator,
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HwShunt => f.write_str("hw_shunt"),
            Self::ModelBased => f.write_str("model_based"),
            Self::Estimator => f.write_str("estimator"),
        }
    }
}

/// Hardware energy domain a counter reports on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EnergyDomain {
    /// Whole-SoC / whole-package rollup. The most-portable domain.
    SocTotal,
    /// CPU package (RAPL `package`, NVML rolled).
    CpuPkg,
    /// A single CPU core or PP0 domain.
    CpuCore,
    /// DRAM controller. Often absent on AMD x86.
    Dram,
    /// Discrete or integrated GPU.
    Gpu,
    /// Neural processing unit (Apple ANE, Intel NPU, AMD XDNA).
    Npu,
    /// Apple Neural Engine specifically. Subset of [`Self::Npu`] but
    /// separately addressable on Apple Silicon.
    Ane,
    /// External rail measured by an INA-class shunt (Jetson, custom hw).
    ExternalRail(u16),
}

/// A single energy reading with full provenance.
///
/// Energy is always microjoules (`u64`). Time is wall-clock nanoseconds
/// since `UNIX_EPOCH`. Both fields are integer; floating-point is forbidden
/// in the protocol for determinism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EnergyReading {
    /// Energy in microjoules. For [`Provenance::ModelBased`] or
    /// [`Provenance::Estimator`] sources the realistic floor is ~1 mJ
    /// (i.e. 1000 μJ) — see [`EnergyCounter::resolution_uj`].
    pub uj: u64,
    /// Wall-clock nanoseconds since UNIX_EPOCH at the moment the reading
    /// was taken.
    pub timestamp_ns: u64,
    /// Domain the reading covers.
    pub domain: EnergyDomain,
    /// Provenance of the reading.
    pub provenance: Provenance,
}

/// Errors a counter can report.
#[derive(Debug, thiserror::Error)]
pub enum EnergyError {
    /// The platform exposes no usable counter for this domain.
    #[error("no counter available for domain {0:?}")]
    NoCounter(EnergyDomain),
    /// The counter requires elevated privileges this process lacks.
    #[error("counter requires elevated privilege: {0}")]
    PrivilegeRequired(&'static str),
    /// Counter wrap or stale read.
    #[error("counter wrap or stale read")]
    CounterWrap,
    /// Underlying platform error.
    #[error("platform error: {0}")]
    Platform(String),
}

/// A hardware (or model-based, or estimated) energy counter for a single
/// [`EnergyDomain`].
///
/// Implementors MUST honestly declare:
///
/// - [`Self::resolution_uj`] — the smallest energy delta the counter can
///   meaningfully report. For RAPL on Intel this is 1 μJ (the MSR quantum).
///   For Apple IOReport this is ~1 mJ. For static estimators it is the
///   model's bin size.
/// - [`Self::min_window_ns`] — the smallest sampling window below which
///   readings are noise-dominated. RAPL: ~10 ms. NVML: ~50 ms. IOReport:
///   ~10 ms but estimator noise dominates anyway.
/// - [`Self::provenance`] — see [`Provenance`].
///
/// JouleClaw's circuit breaker enforces at the granularity of the
/// **worst** counter in a request's span. Lying here corrupts the entire
/// breaker semantic — implementations that overstate their honesty are
/// non-conformant.
pub trait EnergyCounter: Send + Sync {
    /// Which energy domain this counter reports on.
    fn domain(&self) -> EnergyDomain;

    /// Provenance of every reading this counter produces.
    fn provenance(&self) -> Provenance;

    /// Smallest meaningful energy delta the counter can report, in
    /// microjoules. Readings smaller than this are quantization noise.
    fn resolution_uj(&self) -> u64;

    /// Minimum sampling window in nanoseconds below which readings are
    /// noise-dominated. The breaker MUST NOT enforce sub-this windows.
    fn min_window_ns(&self) -> u64;

    /// Take a reading. May fail if the counter requires privilege the
    /// process lacks, if the underlying counter wrapped, or for platform
    /// errors.
    fn read(&self) -> Result<EnergyReading, EnergyError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock counter for property tests + integration scaffolding.
    /// Always returns the same configured reading.
    struct MockCounter {
        reading: EnergyReading,
        resolution_uj: u64,
        min_window_ns: u64,
    }

    impl EnergyCounter for MockCounter {
        fn domain(&self) -> EnergyDomain {
            self.reading.domain
        }
        fn provenance(&self) -> Provenance {
            self.reading.provenance
        }
        fn resolution_uj(&self) -> u64 {
            self.resolution_uj
        }
        fn min_window_ns(&self) -> u64 {
            self.min_window_ns
        }
        fn read(&self) -> Result<EnergyReading, EnergyError> {
            Ok(self.reading)
        }
    }

    #[test]
    fn provenance_display_is_snake_case() {
        assert_eq!(Provenance::HwShunt.to_string(), "hw_shunt");
        assert_eq!(Provenance::ModelBased.to_string(), "model_based");
        assert_eq!(Provenance::Estimator.to_string(), "estimator");
    }

    #[test]
    fn mock_counter_round_trip() {
        let mock = MockCounter {
            reading: EnergyReading {
                uj: 12_345,
                timestamp_ns: 1_700_000_000_000_000_000,
                domain: EnergyDomain::CpuPkg,
                provenance: Provenance::HwShunt,
            },
            resolution_uj: 1, // RAPL-class
            min_window_ns: 10_000_000, // 10 ms
        };
        let counter: &dyn EnergyCounter = &mock;
        let r = counter.read().expect("read");
        assert_eq!(r.uj, 12_345);
        assert_eq!(counter.provenance(), Provenance::HwShunt);
        assert_eq!(counter.resolution_uj(), 1);
    }

    #[test]
    fn ranking_provenance_by_honesty() {
        // The breaker MUST treat HwShunt as the most honest source.
        // This test is documentation that the ordering is intentional.
        let order = [Provenance::HwShunt, Provenance::ModelBased, Provenance::Estimator];
        assert_eq!(order[0], Provenance::HwShunt);
        assert_eq!(order[2], Provenance::Estimator);
    }
}
