//! `eoc-carbon` — carbon-intensity ingest, datacenter PUE, region-aware
//! scheduling, and joules → gCO2e accounting for the EOC stack.
//!
//! The crate is deterministic-first: every public API can be exercised
//! without network using the built-in mock backends. Live HTTP clients
//! for Electricity Maps, WattTime, and CO2 Signal are gated behind the
//! `http` Cargo feature so the crate compiles on `wasm32-unknown-unknown`
//! and air-gapped CI.
//!
//! ## Pipeline
//!
//! 1. **Measure** joules with [`eoc_meter`] (already in the workspace).
//! 2. **Look up** grid intensity for the region serving the request via
//!    [`intensity::CarbonIntensity`] (live provider or [`iea_baseline`]
//!    fallback).
//! 3. **Multiply** by datacenter [`pue::Pue`] overhead.
//! 4. **Account** with [`account::CarbonAccount`] to get per-request
//!    gCO2e.
//! 5. **Optionally** publish a GSF Software Carbon Intensity score via
//!    [`sci_bridge::SciScore`].
//! 6. **Optionally** shift the next batch of work to a greener region or
//!    a greener window with [`scheduler::RegionScheduler`] /
//!    [`scheduler::DemandShifter`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod account;
pub mod co2signal;
pub mod electricity_maps;
pub mod error;
pub mod iea_baseline;
pub mod intensity;
pub mod pue;
pub mod scheduler;
pub mod sci_bridge;
pub mod watttime;

pub use account::{CarbonAccount, CarbonAccounting};
pub use error::{CarbonError, Result};
pub use intensity::{CarbonIntensity, ProviderKind, Zone};
pub use pue::{Pue, PueTable};
pub use scheduler::{DemandShifter, RegionScheduler, ShiftDecision};
pub use sci_bridge::{SciInputs, SciScore};

/// gCO2e per kWh in a "typical" world-average grid (IEA 2024). Useful as
/// a last-resort fallback when even the IEA per-country table cannot be
/// resolved.
pub const WORLD_AVERAGE_G_CO2E_PER_KWH: f64 = 481.0;
