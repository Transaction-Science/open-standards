//! Intel / AMD x86 RAPL energy counter — `Provenance::HwShunt`.
//!
//! Reads from the Linux `powercap` sysfs interface
//! (`/sys/class/powercap/intel-rapl:0/energy_uj`). The kernel exposes
//! the package-domain energy counter directly in microjoules with the
//! MSR's native resolution (1 μJ on Intel, ~15 μJ on AMD Zen).
//!
//! Resolution: 1 μJ. Minimum useful sampling window: ~10 ms (the MSR
//! itself updates at ~1 kHz; below this readings are quantization
//! noise).
//!
//! On non-Linux targets the counter constructs successfully but is
//! permanently unavailable — `read()` returns
//! [`EnergyError::NoCounter`].

#![forbid(unsafe_code)]

use std::path::Path;
#[cfg(target_os = "linux")]
use std::time::{SystemTime, UNIX_EPOCH};

use crate::meter::{EnergyMeter, EnergySource};
use crate::{EnergyCounter, EnergyDomain, EnergyError, EnergyReading, Provenance};

/// Default RAPL package-0 energy file on Linux.
pub const RAPL_ENERGY_PATH: &str = "/sys/class/powercap/intel-rapl:0/energy_uj";

/// Intel / AMD x86 RAPL package-domain energy counter.
pub struct RaplCounter {
    available: bool,
    path: &'static str,
}

impl Default for RaplCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl RaplCounter {
    /// Build a counter against the default `intel-rapl:0` path.
    pub fn new() -> Self {
        let available = cfg!(target_os = "linux") && Path::new(RAPL_ENERGY_PATH).exists();
        Self {
            available,
            path: RAPL_ENERGY_PATH,
        }
    }

    /// Whether this counter found a readable powercap entry.
    pub fn is_available(&self) -> bool {
        self.available
    }
}

impl EnergyCounter for RaplCounter {
    fn domain(&self) -> EnergyDomain {
        EnergyDomain::CpuPkg
    }

    fn provenance(&self) -> Provenance {
        // MSR-backed hardware shunt. Honest to 1 μJ.
        Provenance::HwShunt
    }

    fn resolution_uj(&self) -> u64 {
        // RAPL MSR native quantum.
        1
    }

    fn min_window_ns(&self) -> u64 {
        // MSR updates ~1 kHz. Below 10 ms readings are quantization noise.
        10_000_000
    }

    fn read(&self) -> Result<EnergyReading, EnergyError> {
        if !self.available {
            return Err(EnergyError::NoCounter(EnergyDomain::CpuPkg));
        }

        #[cfg(target_os = "linux")]
        {
            let raw = std::fs::read_to_string(self.path)
                .map_err(|e| EnergyError::Platform(format!("reading {}: {e}", self.path)))?;
            let uj: u64 = raw
                .trim()
                .parse()
                .map_err(|e| EnergyError::Platform(format!("parsing {}: {e}", self.path)))?;

            let timestamp_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| EnergyError::Platform(e.to_string()))?
                .as_nanos() as u64;

            Ok(EnergyReading {
                uj,
                timestamp_ns,
                domain: EnergyDomain::CpuPkg,
                provenance: Provenance::HwShunt,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Branch is dead on non-Linux but keeps the closure happy.
            let _ = self.path;
            Err(EnergyError::Platform(
                "RAPL only available on Linux".to_string(),
            ))
        }
    }
}

impl EnergyMeter for RaplCounter {
    fn name(&self) -> &'static str {
        "rapl"
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
    fn rapl_counter_declares_hw_shunt() {
        let c = RaplCounter::new();
        assert_eq!(c.provenance(), Provenance::HwShunt);
        assert_eq!(c.domain(), EnergyDomain::CpuPkg);
        assert_eq!(c.resolution_uj(), 1);
        assert_eq!(c.min_window_ns(), 10_000_000);
    }

    #[test]
    fn rapl_counter_unavailable_off_linux() {
        let c = RaplCounter::new();
        if cfg!(not(target_os = "linux")) {
            assert!(!c.is_available());
            assert!(matches!(c.read(), Err(EnergyError::NoCounter(_))));
        }
        assert_eq!(c.name(), "rapl");
    }
}
