//! Meter surface that sits *above* [`crate::EnergyCounter`].
//!
//! [`EnergyCounter`](crate::EnergyCounter) is the honest-by-construction
//! trait — every implementation declares [`Provenance`](crate::Provenance),
//! [`resolution_uj`](crate::EnergyCounter::resolution_uj), and
//! [`min_window_ns`](crate::EnergyCounter::min_window_ns). It is the
//! protocol-conformant surface.
//!
//! [`EnergyMeter`] is a thin convenience surface for *callers* (cascade,
//! breaker, dashboards) that want `name()`/`is_available()` plus optional
//! `battery_pct`/`thermal_state`. Every meter implements `EnergyCounter`
//! transitively — the meter just adds discovery + composition.
//!
//! The donor `inv-energy::EnergyMeter` used a `(Joules, Watts, ms)`
//! reading shape. JouleClaw's normative reading is `(uj, timestamp_ns,
//! domain, provenance)` (integer-only, deterministic). The legacy shape
//! is gone; backends report directly via `EnergyCounter::read()`.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{EnergyCounter, EnergyDomain, EnergyError, EnergyReading, Provenance};

/// Thermal state of a node. Optional metadata surfaced by meters that can
/// detect their host's thermal pressure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ThermalState {
    /// Normal operating temperature.
    #[default]
    Normal,
    /// Warm — approaching throttling threshold.
    Warm,
    /// Actively throttled to reduce temperature.
    Throttled,
    /// Critical — shedding workloads.
    Critical,
}

/// The energy source powering a node. Optional metadata exposed by some
/// platforms (battery vs wall power vs solar).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EnergySource {
    /// Plugged into wall power (grid).
    WallPower,
    /// Running on battery.
    Battery,
    /// Solar powered.
    Solar,
    /// Unknown energy source.
    #[default]
    Unknown,
}

/// Errors that the meter discovery / composition layer can report.
///
/// All real read errors come from `EnergyCounter::read()` as
/// [`EnergyError`].
#[derive(Debug, thiserror::Error)]
pub enum EnergyMeterError {
    /// No meter available for this platform.
    #[error("no energy meter available on this platform")]
    NoMeterAvailable,
    /// Underlying counter failed.
    #[error(transparent)]
    Counter(#[from] EnergyError),
}

/// Convenience trait callers use to discover, name, and probe meters.
/// Every meter is *also* an [`EnergyCounter`].
pub trait EnergyMeter: EnergyCounter {
    /// Short stable identifier (`"rapl"`, `"nvml"`, `"apple-silicon"`,
    /// `"estimation"`). For dashboards and logs.
    fn name(&self) -> &'static str;

    /// Whether the meter found usable hardware. A non-available meter
    /// must return an error from `read()`.
    fn is_available(&self) -> bool;

    /// Energy source powering the host, if known.
    fn energy_source(&self) -> EnergySource {
        EnergySource::Unknown
    }

    /// Thermal state of the host, if detectable.
    fn thermal_state(&self) -> ThermalState {
        ThermalState::Normal
    }

    /// Battery percentage (0–100), if applicable.
    fn battery_pct(&self) -> Option<u8> {
        None
    }
}

/// Composite meter that probes each candidate in order and routes
/// `read()` to the first available one.
///
/// `EnergyCounter` is *not* implemented on `CompositeMeter` directly
/// because its domain/provenance/resolution are dynamic. Use
/// [`Self::active`] to get the active backing meter.
pub struct CompositeMeter {
    meters: Vec<Box<dyn EnergyMeter>>,
    active_index: Option<usize>,
}

impl CompositeMeter {
    /// Build a composite, selecting the first available meter.
    pub fn new(meters: Vec<Box<dyn EnergyMeter>>) -> Self {
        let active_index = meters.iter().position(|m| m.is_available());
        Self {
            meters,
            active_index,
        }
    }

    /// Active backing meter, if any.
    pub fn active(&self) -> Option<&dyn EnergyMeter> {
        self.active_index.map(|i| self.meters[i].as_ref())
    }

    /// Active meter's short name (or `"none"`).
    pub fn active_name(&self) -> &'static str {
        self.active().map(|m| m.name()).unwrap_or("none")
    }

    /// `true` if at least one configured meter is available.
    pub fn is_available(&self) -> bool {
        self.active_index.is_some()
    }

    /// Take a reading via the active meter.
    pub fn read(&self) -> Result<EnergyReading, EnergyMeterError> {
        match self.active() {
            Some(m) => Ok(m.read()?),
            None => Err(EnergyMeterError::NoMeterAvailable),
        }
    }
}

/// Static-cost estimator. Always-available fallback. Honestly reports
/// [`Provenance::Estimator`].
///
/// The estimator assumes a single CPU package at a given TDP and
/// `idle_fraction` × TDP at idle, `1.0 × TDP` at saturation. It does
/// **not** read load on platforms outside macOS — on Linux/Windows/etc
/// it returns a constant mid-band estimate.
pub struct EstimationMeter {
    tdp_watts: f64,
    idle_fraction: f64,
    cumulative_uj: AtomicU64,
    last_timestamp_ms: AtomicU64,
}

impl EstimationMeter {
    /// Build with a TDP estimate (watts). `idle_fraction` defaults to
    /// `0.2` (20 % of TDP at idle).
    pub fn new(tdp_watts: f64) -> Self {
        Self {
            tdp_watts,
            idle_fraction: 0.2,
            cumulative_uj: AtomicU64::new(0),
            last_timestamp_ms: AtomicU64::new(0),
        }
    }

    fn estimate_watts(&self, cpu_load: f64) -> f64 {
        let idle = self.tdp_watts * self.idle_fraction;
        let active = self.tdp_watts * (1.0 - self.idle_fraction) * cpu_load;
        idle + active
    }

    fn cpu_load_estimate(&self) -> f64 {
        // Without a platform-specific syscall the estimator returns a
        // mid-band load. Apple-specific path lives in `apple.rs`.
        0.3
    }
}

impl EnergyCounter for EstimationMeter {
    fn domain(&self) -> EnergyDomain {
        EnergyDomain::SocTotal
    }

    fn provenance(&self) -> Provenance {
        Provenance::Estimator
    }

    fn resolution_uj(&self) -> u64 {
        // Static-cost-model bin size: 1 mJ floor.
        1_000
    }

    fn min_window_ns(&self) -> u64 {
        // Estimator noise dominates below ~100 ms.
        100_000_000
    }

    fn read(&self) -> Result<EnergyReading, EnergyError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| EnergyError::Platform(e.to_string()))?
            .as_millis() as u64;

        let load = self.cpu_load_estimate();
        let watts = self.estimate_watts(load);

        let last_ms = self.last_timestamp_ms.swap(now_ms, Ordering::Relaxed);
        if last_ms > 0 && now_ms > last_ms {
            let elapsed_secs = (now_ms - last_ms) as f64 / 1000.0;
            let delta_uj = (watts * elapsed_secs * 1_000_000.0) as u64;
            self.cumulative_uj.fetch_add(delta_uj, Ordering::Relaxed);
        }

        let total_uj = self.cumulative_uj.load(Ordering::Relaxed);
        Ok(EnergyReading {
            uj: total_uj,
            timestamp_ns: now_ms.saturating_mul(1_000_000),
            domain: EnergyDomain::SocTotal,
            provenance: Provenance::Estimator,
        })
    }
}

impl EnergyMeter for EstimationMeter {
    fn name(&self) -> &'static str {
        "estimation"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn energy_source(&self) -> EnergySource {
        EnergySource::Unknown
    }
}

/// Detect the best meter for this platform. Always returns *something*
/// — at minimum the [`EstimationMeter`] fallback.
pub fn detect_meter() -> CompositeMeter {
    let mut meters: Vec<Box<dyn EnergyMeter>> = Vec::new();

    #[cfg(all(feature = "ioreport", target_os = "macos"))]
    {
        meters.push(Box::new(crate::apple::AppleSiliconMeter::new()));
    }

    #[cfg(all(feature = "rapl", target_os = "linux"))]
    {
        meters.push(Box::new(crate::rapl::RaplCounter::new()));
    }

    #[cfg(feature = "nvml")]
    {
        meters.push(Box::new(crate::nvml::NvmlCounter::new()));
    }

    meters.push(Box::new(EstimationMeter::new(65.0)));
    CompositeMeter::new(meters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimation_meter_always_available() {
        let m = EstimationMeter::new(45.0);
        assert!(m.is_available());
        assert_eq!(m.name(), "estimation");
        assert_eq!(m.provenance(), Provenance::Estimator);
    }

    #[test]
    fn estimation_meter_reports_estimator_provenance() {
        let m = EstimationMeter::new(45.0);
        let r = m.read().expect("read");
        assert_eq!(r.provenance, Provenance::Estimator);
        assert_eq!(r.domain, EnergyDomain::SocTotal);
        // Resolution + window floors are honest about being coarse.
        assert!(m.resolution_uj() >= 1_000);
        assert!(m.min_window_ns() >= 100_000_000);
    }

    #[test]
    fn estimation_meter_accumulates_across_reads() {
        let m = EstimationMeter::new(45.0);
        let r1 = m.read().expect("r1");
        std::thread::sleep(std::time::Duration::from_millis(20));
        let r2 = m.read().expect("r2");
        assert!(r2.timestamp_ns >= r1.timestamp_ns);
        assert!(r2.uj >= r1.uj);
    }

    #[test]
    fn detect_returns_at_least_estimation() {
        let composite = detect_meter();
        assert!(composite.is_available());
        let r = composite.read().expect("composite read");
        // Provenance is honest no matter which backend is active.
        assert!(matches!(
            r.provenance,
            Provenance::HwShunt | Provenance::ModelBased | Provenance::Estimator
        ));
    }

    #[test]
    fn empty_composite_reports_no_meter() {
        let composite = CompositeMeter::new(vec![]);
        assert!(!composite.is_available());
        let err = composite.read().expect_err("empty must err");
        assert!(matches!(err, EnergyMeterError::NoMeterAvailable));
        assert_eq!(composite.active_name(), "none");
    }
}
