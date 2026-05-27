//! Apple Silicon energy estimator — `Provenance::Estimator`.
//!
//! **HONEST DISCLOSURE.** The production-quality path is IOReport's
//! `"Energy Model"` channel group (`PMP`/`Energy Model` keys), which
//! returns vendor-modelled package energy at ~1 mJ / ~10 ms. JouleClaw
//! has not yet wired the IOReport FFI; until then this backend uses
//! the system 1-minute load average, normalises it against
//! `available_parallelism`, and interpolates between a fixed
//! `idle_watts = 8 W` and `peak_watts = 40 W` band.
//!
//! That is *not* a hardware measurement. The reading therefore reports
//! [`Provenance::Estimator`], a microjoule resolution of ~1 mJ
//! (matching the estimator's bin size), and a minimum window of
//! ~100 ms (load-average is updated by the kernel at ~5 s but quoted
//! at much higher frequency; the practical noise floor is ~100 ms).
//!
//! When the IOReport bridge ships, this module should be replaced or
//! re-tagged as `Provenance::ModelBased`. **DO NOT change the tag in
//! place** — the breaker depends on it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::meter::{EnergyMeter, EnergySource};
use crate::{EnergyCounter, EnergyDomain, EnergyError, EnergyReading, Provenance};

const IDLE_WATTS: f64 = 8.0;
const PEAK_WATTS: f64 = 40.0;

/// Apple Silicon SoC energy estimator (load-average × power band).
pub struct AppleSiliconMeter {
    available: bool,
    cumulative_uj: AtomicU64,
    last_timestamp_ms: AtomicU64,
}

impl Default for AppleSiliconMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl AppleSiliconMeter {
    /// Build a meter. Available only on macOS aarch64.
    pub fn new() -> Self {
        let available = cfg!(target_os = "macos") && cfg!(target_arch = "aarch64");
        Self {
            available,
            cumulative_uj: AtomicU64::new(0),
            last_timestamp_ms: AtomicU64::new(0),
        }
    }

    /// Whether this estimator's host matches macOS / Apple Silicon.
    pub fn is_available(&self) -> bool {
        self.available
    }

    fn read_load_avg() -> f64 {
        // The only `unsafe` block in jouleclaw-energy. `getloadavg` writes
        // up to 3 entries into the supplied array and returns the number
        // of samples written. It cannot fail in a way that violates
        // memory safety as long as the buffer pointer is valid for the
        // requested element count.
        let mut load: [f64; 3] = [0.0; 3];
        // SAFETY: `load` is a 3-element stack array; we pass its base
        // pointer and length 3. Both arguments are valid for the call.
        unsafe {
            libc::getloadavg(load.as_mut_ptr(), 3);
        }
        load[0]
    }

    fn estimate_watts(&self) -> f64 {
        let load_1 = Self::read_load_avg();
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as f64;
        let util = (load_1 / num_cpus).clamp(0.0, 1.0);
        IDLE_WATTS + (PEAK_WATTS - IDLE_WATTS) * util
    }
}

impl EnergyCounter for AppleSiliconMeter {
    fn domain(&self) -> EnergyDomain {
        EnergyDomain::SocTotal
    }

    fn provenance(&self) -> Provenance {
        // Until IOReport is wired this is unambiguously an Estimator.
        Provenance::Estimator
    }

    fn resolution_uj(&self) -> u64 {
        // Estimator bin size: 1 mJ.
        1_000
    }

    fn min_window_ns(&self) -> u64 {
        // Load-average noise floor: ~100 ms.
        100_000_000
    }

    fn read(&self) -> Result<EnergyReading, EnergyError> {
        if !self.available {
            return Err(EnergyError::NoCounter(EnergyDomain::SocTotal));
        }

        let watts = self.estimate_watts();
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
            domain: EnergyDomain::SocTotal,
            provenance: Provenance::Estimator,
        })
    }
}

impl EnergyMeter for AppleSiliconMeter {
    fn name(&self) -> &'static str {
        "apple-silicon"
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
    fn apple_meter_declares_estimator_provenance() {
        let m = AppleSiliconMeter::new();
        // The honesty tag is what the breaker reads. It MUST be
        // Estimator until IOReport ships.
        assert_eq!(m.provenance(), Provenance::Estimator);
        assert_eq!(m.domain(), EnergyDomain::SocTotal);
        assert_eq!(m.name(), "apple-silicon");
        assert!(m.resolution_uj() >= 1_000);
        assert!(m.min_window_ns() >= 100_000_000);
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn apple_meter_reads_a_value_on_apple_silicon() {
        let m = AppleSiliconMeter::new();
        assert!(m.is_available());
        let r = m.read().expect("apple read");
        assert_eq!(r.provenance, Provenance::Estimator);
        assert_eq!(r.domain, EnergyDomain::SocTotal);
    }
}
