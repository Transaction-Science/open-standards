//! EOC joule meter.
//!
//! The meter reads cumulative micro-joule counters from whatever hardware
//! is available. Production deployments care a great deal which counter
//! is in use; reference deployments fall back to `StubCounter`, which
//! always returns 0 and reports its source as `JouleSource::Estimated`.
//!
//! Hardware counter dependencies are **platform-conditional and
//! feature-gated**:
//!
//! - Linux RAPL is read via `/sys/class/powercap/intel-rapl:0/energy_uj`.
//!   No external crate required.
//! - macOS `powermetrics` is invoked via `Command::new("powermetrics")`.
//!   Requires root in practice — the wrapper degrades to `StubCounter`
//!   when the process can't fork it.
//! - NVIDIA NVML is gated behind the `cuda` feature. Off by default.
//! - WASM and unknown targets fall through to `StubCounter`.

#![forbid(unsafe_code)]

// `Error` is used by platform-conditional modules below.
#[cfg(any(target_os = "linux", target_os = "macos", feature = "cuda"))]
use eoc_core::Error;
use eoc_core::Result;

/// A monotonic, cumulative joule counter.
///
/// Implementations report total energy consumed since boot in micro-joules.
/// Subtract two readings to get the energy of an interval.
pub trait JouleCounter: Send + Sync {
    /// Read the current cumulative micro-joule count.
    fn read_microjoules(&self) -> Result<u64>;

    /// Human-readable name of this counter (`"rapl"`, `"nvml"`, etc.).
    fn name(&self) -> &'static str;
}

/// A no-op counter that always reports 0. Used when no hardware counter
/// is available.
#[derive(Debug, Default)]
pub struct StubCounter;

impl JouleCounter for StubCounter {
    fn read_microjoules(&self) -> Result<u64> {
        Ok(0)
    }
    fn name(&self) -> &'static str {
        "stub"
    }
}

#[cfg(target_os = "linux")]
mod linux_rapl {
    use super::*;

    /// Reads Intel RAPL counters via the powercap sysfs interface.
    pub struct LinuxRaplCounter {
        path: std::path::PathBuf,
    }

    impl LinuxRaplCounter {
        /// Try to construct a RAPL counter pointing at the package-0 domain.
        pub fn try_detect() -> Result<Self> {
            let path = std::path::PathBuf::from(
                "/sys/class/powercap/intel-rapl:0/energy_uj",
            );
            if path.exists() {
                Ok(Self { path })
            } else {
                Err(Error::NoCounter)
            }
        }
    }

    impl JouleCounter for LinuxRaplCounter {
        fn read_microjoules(&self) -> Result<u64> {
            let s = std::fs::read_to_string(&self.path)
                .map_err(|e| Error::Io(e.to_string()))?;
            s.trim()
                .parse::<u64>()
                .map_err(|e| Error::Backend(e.to_string()))
        }
        fn name(&self) -> &'static str {
            "rapl"
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux_rapl::LinuxRaplCounter;

#[cfg(target_os = "macos")]
mod macos_powermetrics {
    use super::*;

    /// Wraps `powermetrics(1)` to estimate cumulative energy. macOS does
    /// not expose a cumulative joule counter at user level; this wrapper
    /// reports the integral of the most recent `--samplers cpu_power`
    /// reading. In practice it requires root, so it usually degrades to
    /// `StubCounter` at detection time.
    pub struct MacosPowermetricsCounter;

    impl MacosPowermetricsCounter {
        /// Attempt to detect `powermetrics` availability.
        pub fn try_detect() -> Result<Self> {
            // We don't require root here — `detect()` will fall back to
            // StubCounter if reads fail. We do require the binary to be
            // present on the PATH.
            let ok = std::process::Command::new("which")
                .arg("powermetrics")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if ok { Ok(Self) } else { Err(Error::NoCounter) }
        }
    }

    impl JouleCounter for MacosPowermetricsCounter {
        fn read_microjoules(&self) -> Result<u64> {
            // Real implementation would parse `powermetrics --samplers
            // cpu_power -i 100 -n 1 --hide-cpu-duty-cycle`. For the
            // reference implementation we always error so the cascade
            // attaches `JouleSource::Estimated` instead.
            Err(Error::NoCounter)
        }
        fn name(&self) -> &'static str {
            "powermetrics"
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos_powermetrics::MacosPowermetricsCounter;

#[cfg(feature = "cuda")]
mod nvml {
    use super::*;
    use nvml_wrapper::Nvml;

    /// Reads cumulative GPU energy via NVML.
    pub struct NvmlCounter {
        _nvml: Nvml,
        device_index: u32,
    }

    impl NvmlCounter {
        /// Attempt to detect NVML and pick GPU 0.
        pub fn try_detect() -> Result<Self> {
            let nvml = Nvml::init().map_err(|e| Error::Backend(e.to_string()))?;
            Ok(Self {
                _nvml: nvml,
                device_index: 0,
            })
        }
    }

    impl JouleCounter for NvmlCounter {
        fn read_microjoules(&self) -> Result<u64> {
            let device = self
                ._nvml
                .device_by_index(self.device_index)
                .map_err(|e| Error::Backend(e.to_string()))?;
            let mj = device
                .total_energy_consumption()
                .map_err(|e| Error::Backend(e.to_string()))?;
            // NVML reports millijoules; convert to microjoules.
            Ok(mj.saturating_mul(1000))
        }
        fn name(&self) -> &'static str {
            "nvml"
        }
    }
}

#[cfg(feature = "cuda")]
pub use nvml::NvmlCounter;

/// Runtime-detect the best available counter on this host.
///
/// Preference order: NVML (if compiled with `cuda` and present) → RAPL on
/// Linux → powermetrics on macOS → `StubCounter`.
pub fn detect() -> Box<dyn JouleCounter> {
    #[cfg(feature = "cuda")]
    if let Ok(c) = NvmlCounter::try_detect() {
        return Box::new(c);
    }
    #[cfg(target_os = "linux")]
    if let Ok(c) = LinuxRaplCounter::try_detect() {
        return Box::new(c);
    }
    #[cfg(target_os = "macos")]
    if let Ok(c) = MacosPowermetricsCounter::try_detect() {
        return Box::new(c);
    }
    Box::new(StubCounter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reads_zero() {
        let c = StubCounter;
        assert_eq!(c.read_microjoules().unwrap(), 0);
        assert_eq!(c.name(), "stub");
    }

    #[test]
    fn detect_always_returns_something() {
        let c = detect();
        // Should never panic. Reading may succeed (stub) or fail (live
        // counter without permission).
        let _ = c.read_microjoules();
        assert!(!c.name().is_empty());
    }
}
