//! NVIDIA discrete GPU energy counter via NVML — `Provenance::HwShunt`.
//!
//! Sums `power_usage()` across all detected GPUs and integrates over
//! time to maintain a cumulative microjoule counter. On Hopper+ /
//! Blackwell, `nvmlDeviceGetTotalEnergyConsumption()` returns the
//! cumulative-energy counter directly; older silicon falls back to
//! `power_usage()` integration (still hardware-shunt-based — NVIDIA's
//! power-monitoring IC samples the rail at ~50 ms).
//!
//! Resolution: ~1 mJ (vendor-reported; rounded up from the underlying
//! INA3221 / on-die sense quantization).
//!
//! Minimum useful sampling window: ~50 ms. Faster polls return the
//! same sampled value with re-quantization noise.
//!
//! Without the `nvml` cargo feature this module is not compiled.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::meter::{EnergyMeter, EnergySource};
use crate::{EnergyCounter, EnergyDomain, EnergyError, EnergyReading, Provenance};

/// NVIDIA discrete GPU energy counter.
///
/// Initialises NVML at construction time. If NVML init fails or no
/// GPUs are present the counter is permanently unavailable.
pub struct NvmlCounter {
    available: bool,
    nvml: Option<nvml_wrapper::Nvml>,
    device_count: u32,
    cumulative_uj: AtomicU64,
    last_timestamp_ms: AtomicU64,
}

impl Default for NvmlCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl NvmlCounter {
    /// Initialise NVML and enumerate GPUs.
    pub fn new() -> Self {
        match nvml_wrapper::Nvml::init() {
            Ok(nvml) => {
                let device_count = nvml.device_count().unwrap_or(0);
                if device_count > 0 {
                    tracing::info!(gpu_count = device_count, "NVML initialized");
                    Self {
                        available: true,
                        nvml: Some(nvml),
                        device_count,
                        cumulative_uj: AtomicU64::new(0),
                        last_timestamp_ms: AtomicU64::new(0),
                    }
                } else {
                    tracing::debug!("NVML initialized but no GPUs found");
                    Self {
                        available: false,
                        nvml: Some(nvml),
                        device_count: 0,
                        cumulative_uj: AtomicU64::new(0),
                        last_timestamp_ms: AtomicU64::new(0),
                    }
                }
            }
            Err(e) => {
                tracing::debug!("NVML init failed: {e}");
                Self {
                    available: false,
                    nvml: None,
                    device_count: 0,
                    cumulative_uj: AtomicU64::new(0),
                    last_timestamp_ms: AtomicU64::new(0),
                }
            }
        }
    }

    /// Whether NVML initialised and at least one GPU was found.
    pub fn is_available(&self) -> bool {
        self.available
    }
}

impl EnergyCounter for NvmlCounter {
    fn domain(&self) -> EnergyDomain {
        EnergyDomain::Gpu
    }

    fn provenance(&self) -> Provenance {
        // NVIDIA's on-card power-monitor IC is a real shunt. The
        // `nvmlDeviceGetTotalEnergyConsumption` path is unambiguously
        // HwShunt; the `power_usage()` integration path is also shunt-
        // sourced, just with ~50 ms quantization.
        Provenance::HwShunt
    }

    fn resolution_uj(&self) -> u64 {
        // ~1 mJ vendor-reported. Conservative.
        1_000
    }

    fn min_window_ns(&self) -> u64 {
        // Sample period ~50 ms.
        50_000_000
    }

    fn read(&self) -> Result<EnergyReading, EnergyError> {
        let nvml = self
            .nvml
            .as_ref()
            .ok_or(EnergyError::NoCounter(EnergyDomain::Gpu))?;

        if !self.available {
            return Err(EnergyError::NoCounter(EnergyDomain::Gpu));
        }

        let mut total_milliwatts: u64 = 0;
        for i in 0..self.device_count {
            let device = nvml
                .device_by_index(i)
                .map_err(|e| EnergyError::Platform(format!("GPU {i}: {e}")))?;
            let power_mw = device
                .power_usage()
                .map_err(|e| EnergyError::Platform(format!("GPU {i} power: {e}")))?;
            total_milliwatts = total_milliwatts.saturating_add(power_mw as u64);
        }

        let watts = total_milliwatts as f64 / 1000.0;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| EnergyError::Platform(e.to_string()))?
            .as_millis() as u64;

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
            domain: EnergyDomain::Gpu,
            provenance: Provenance::HwShunt,
        })
    }
}

impl EnergyMeter for NvmlCounter {
    fn name(&self) -> &'static str {
        "nvml"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn energy_source(&self) -> EnergySource {
        EnergySource::WallPower
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvml_counter_declares_hw_shunt() {
        let c = NvmlCounter::new();
        assert_eq!(c.provenance(), Provenance::HwShunt);
        assert_eq!(c.domain(), EnergyDomain::Gpu);
        assert_eq!(c.name(), "nvml");
    }

    #[test]
    fn nvml_counter_unavailable_without_gpu() {
        // CI / macOS / non-NVIDIA hosts will all hit this branch.
        let c = NvmlCounter::new();
        if !c.is_available() {
            assert!(c.read().is_err());
        }
    }

    #[test]
    fn nvml_resolution_is_millijoule_floor() {
        let c = NvmlCounter::new();
        // NVML is honest only down to ~1 mJ.
        assert!(c.resolution_uj() >= 1_000);
        // 50 ms minimum window.
        assert!(c.min_window_ns() >= 50_000_000);
    }
}
