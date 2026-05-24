//! Power Usage Effectiveness (PUE) tables for AWS / GCP / Azure regions.
//!
//! PUE is total facility power divided by IT power: a multiplier ≥ 1 that
//! captures cooling, distribution, and lighting overhead. Hyperscalers
//! publish per-region (or trailing-twelve-month fleetwide) numbers; we
//! ship the published values rounded to 0.01.

use crate::error::{CarbonError, Result};

/// A datacenter PUE value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pue(pub f64);

impl Pue {
    /// Construct a PUE. Values below 1.0 are clamped to 1.0 (physically
    /// impossible to do better than 1.0 — facility power == IT power).
    pub fn new(v: f64) -> Self {
        Self(if v < 1.0 { 1.0 } else { v })
    }

    /// Raw multiplier.
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Default for Pue {
    /// Industry average ~1.58 (Uptime Institute 2024).
    fn default() -> Self {
        Self(1.58)
    }
}

/// One PUE table row.
#[derive(Debug, Clone, Copy)]
pub struct PueRow {
    /// Cloud provider name (lowercase).
    pub cloud: &'static str,
    /// Provider-native region tag (`us-east-1`, `europe-west4`,
    /// `eastus`, …).
    pub region: &'static str,
    /// PUE for that region.
    pub pue: f64,
}

/// Static per-region PUE table for AWS, GCP, and Azure. Values are
/// published by each cloud and rounded to 0.01.
pub const PUE_TABLE: &[PueRow] = &[
    // AWS — fleet trailing-twelve-month 1.15, regional rounded.
    PueRow { cloud: "aws", region: "us-east-1",      pue: 1.15 },
    PueRow { cloud: "aws", region: "us-east-2",      pue: 1.12 },
    PueRow { cloud: "aws", region: "us-west-2",      pue: 1.13 },
    PueRow { cloud: "aws", region: "eu-west-1",      pue: 1.11 },
    PueRow { cloud: "aws", region: "eu-north-1",     pue: 1.09 },
    PueRow { cloud: "aws", region: "ap-northeast-1", pue: 1.18 },
    PueRow { cloud: "aws", region: "ap-southeast-1", pue: 1.22 },

    // GCP — fleet TTM 1.10, regional rounded.
    PueRow { cloud: "gcp", region: "us-central1",   pue: 1.11 },
    PueRow { cloud: "gcp", region: "us-east4",      pue: 1.10 },
    PueRow { cloud: "gcp", region: "us-west1",      pue: 1.10 },
    PueRow { cloud: "gcp", region: "europe-west1",  pue: 1.08 },
    PueRow { cloud: "gcp", region: "europe-west4",  pue: 1.07 },
    PueRow { cloud: "gcp", region: "europe-north1", pue: 1.08 },
    PueRow { cloud: "gcp", region: "asia-east1",    pue: 1.16 },

    // Azure — fleet TTM 1.18, regional rounded.
    PueRow { cloud: "azure", region: "eastus",        pue: 1.18 },
    PueRow { cloud: "azure", region: "westus2",       pue: 1.15 },
    PueRow { cloud: "azure", region: "northeurope",   pue: 1.12 },
    PueRow { cloud: "azure", region: "westeurope",    pue: 1.12 },
    PueRow { cloud: "azure", region: "japaneast",     pue: 1.20 },
    PueRow { cloud: "azure", region: "australiaeast", pue: 1.22 },
];

/// Lookup table — case-insensitive on cloud and region.
pub struct PueTable;

impl PueTable {
    /// Look up a per-region PUE. Returns `UnknownZone` if not in the
    /// table (which the caller can swallow into [`Pue::default`]).
    pub fn lookup(cloud: &str, region: &str) -> Result<Pue> {
        PUE_TABLE
            .iter()
            .find(|r| r.cloud.eq_ignore_ascii_case(cloud) && r.region.eq_ignore_ascii_case(region))
            .map(|r| Pue::new(r.pue))
            .ok_or_else(|| CarbonError::UnknownZone(format!("{cloud}:{region}")))
    }

    /// Look up a PUE, falling back to the industry-average
    /// [`Pue::default`] (1.58) when the row is missing.
    pub fn lookup_or_default(cloud: &str, region: &str) -> Pue {
        Self::lookup(cloud, region).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pue_clamped_to_unity() {
        assert_eq!(Pue::new(0.5).value(), 1.0);
        assert_eq!(Pue::new(1.07).value(), 1.07);
    }

    #[test]
    fn gcp_europe_west4_is_low() {
        let p = PueTable::lookup("gcp", "europe-west4").expect("present");
        assert!(p.value() < 1.10);
    }

    #[test]
    fn unknown_region_falls_back_to_default() {
        let p = PueTable::lookup_or_default("aws", "made-up-region");
        assert_eq!(p, Pue::default());
    }

    #[test]
    fn case_insensitive_lookup() {
        let a = PueTable::lookup("AWS", "US-EAST-1").expect("ok");
        let b = PueTable::lookup("aws", "us-east-1").expect("ok");
        assert_eq!(a, b);
    }
}
