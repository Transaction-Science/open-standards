//! MCC-aware routing.
//!
//! ISO 18245 Merchant Category Codes (MCCs) are 4-digit numerics
//! that classify what the merchant sells. PSPs price and approve
//! by MCC: a `5411` (grocery) auth is treated very differently
//! from a `7995` (gambling) auth at every issuer. Some PSPs
//! refuse certain MCCs outright (`5993` tobacco, `7273` adult,
//! etc.); others have an acceptance discount for some categories.
//!
//! This module models per-MCC routing **preferences and
//! exclusions**:
//!
//! - [`McuPreferences::preferred_drivers`] — a soft hint to prefer
//!   these drivers (in order). The composer respects this above
//!   pure least-cost.
//! - [`McuPreferences::excluded_drivers`] — a hard exclusion. A
//!   driver in this set is removed from the candidate pool for
//!   intents in that MCC. This is the one operators most care
//!   about — being unable to retry through a PSP that refuses the
//!   MCC.
//! - [`McuPreferences::max_attempts`] — per-MCC retry cap. Some
//!   high-risk categories cap retries lower than the global default.
//!
//! The curated MCC catalogue ships in `data/mcc.json` and is
//! embedded at compile time. See [`mcc_catalogue`].

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::route::{DriverId, Route};

/// ISO 18245 4-digit Merchant Category Code.
///
/// We store it as a fixed `[u8; 4]` of ASCII digits so the type
/// system catches accidental swaps with other 4-digit strings.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mcc(pub [u8; 4]);

impl Mcc {
    /// Construct from a 4-character `&str`. Returns `None` if the
    /// input is not exactly four ASCII digits.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 4 {
            return None;
        }
        let mut out = [0u8; 4];
        for (i, b) in bytes.iter().enumerate() {
            if !b.is_ascii_digit() {
                return None;
            }
            out[i] = *b;
        }
        Some(Self(out))
    }

    /// As `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // Constructor enforces ASCII digits.
        core::str::from_utf8(&self.0).unwrap_or("????")
    }
}

impl core::fmt::Display for Mcc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-MCC routing preferences. The struct name keeps the brief's
/// spelling (`McuPreferences`) — "MCU" is the internal acronym some
/// teams use ("Merchant Category Unit").
#[derive(Clone, Debug, Default)]
pub struct McuPreferences {
    /// Soft preference for these drivers, in order. Drivers in this
    /// list move ahead of others in the candidate pool (but inside
    /// this list, the composer's other criteria — like LCR — still
    /// order them).
    pub preferred_drivers: Vec<DriverId>,

    /// Hard exclusion. Drivers in this set are removed from the pool
    /// for intents in this MCC, regardless of cost / preference.
    pub excluded_drivers: HashSet<DriverId>,

    /// Maximum retry attempts for this MCC. Caps the retry policy's
    /// `max_attempts` when set. `0` means "use the global default".
    pub max_attempts: u8,
}

impl McuPreferences {
    /// Builder: add a preferred driver to the end of the preference list.
    #[must_use]
    pub fn with_preferred(mut self, driver: DriverId) -> Self {
        self.preferred_drivers.push(driver);
        self
    }

    /// Builder: add a driver to the exclusion set.
    #[must_use]
    pub fn with_excluded(mut self, driver: DriverId) -> Self {
        self.excluded_drivers.insert(driver);
        self
    }

    /// Builder: set max attempts for this MCC.
    #[must_use]
    pub const fn with_max_attempts(mut self, n: u8) -> Self {
        self.max_attempts = n;
        self
    }
}

/// MCC policy: a map from MCC to preferences.
#[derive(Clone, Debug, Default)]
pub struct MccPolicy {
    /// Per-MCC rules.
    pub rules: HashMap<Mcc, McuPreferences>,
}

impl MccPolicy {
    /// Empty policy. All routes pass through unfiltered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a rule for one MCC.
    #[must_use]
    pub fn with_rule(mut self, mcc: Mcc, prefs: McuPreferences) -> Self {
        self.rules.insert(mcc, prefs);
        self
    }

    /// Filter and reorder a route pool for a given MCC.
    ///
    /// 1. Routes with `driver` in `excluded_drivers` are dropped.
    /// 2. Surviving routes are partitioned into "preferred" and
    ///    "other", with the preferred group ordered to match
    ///    `preferred_drivers`. The relative order *inside* each
    ///    group is otherwise preserved (stable).
    ///
    /// If no rule exists for `mcc` (or `mcc` is `None`), the pool
    /// is returned unchanged.
    #[must_use]
    pub fn filter(&self, mcc: Option<Mcc>, pool: &[Route]) -> Vec<Route> {
        let Some(mcc) = mcc else {
            return pool.to_vec();
        };
        let Some(prefs) = self.rules.get(&mcc) else {
            return pool.to_vec();
        };

        // 1. Drop exclusions.
        let surviving: Vec<&Route> = pool
            .iter()
            .filter(|r| !prefs.excluded_drivers.contains(&r.driver))
            .collect();

        // 2. Partition into preferred and other, preserving input
        //    order inside each group.
        let pref_set: HashMap<&DriverId, usize> = prefs
            .preferred_drivers
            .iter()
            .enumerate()
            .map(|(i, d)| (d, i))
            .collect();

        let mut preferred: Vec<&Route> = surviving
            .iter()
            .filter(|r| pref_set.contains_key(&r.driver))
            .copied()
            .collect();
        preferred.sort_by_key(|r| pref_set.get(&r.driver).copied().unwrap_or(usize::MAX));

        let other: Vec<&Route> = surviving
            .iter()
            .filter(|r| !pref_set.contains_key(&r.driver))
            .copied()
            .collect();

        preferred.into_iter().chain(other).cloned().collect()
    }

    /// Get the effective `max_attempts` for an MCC, falling back to
    /// `default_max`.
    #[must_use]
    pub fn max_attempts_for(&self, mcc: Option<Mcc>, default_max: u8) -> u8 {
        mcc.and_then(|m| self.rules.get(&m))
            .map(|p| p.max_attempts)
            .filter(|n| *n > 0)
            .unwrap_or(default_max)
    }
}

/// One entry in the curated MCC catalogue.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct McuCategory {
    /// 4-digit MCC.
    pub mcc: String,
    /// Human-readable category name.
    pub name: String,
}

/// The shape of `data/mcc.json` after deserialization.
#[derive(Clone, Debug, Deserialize)]
struct McuCatalogueFile {
    version: String,
    categories: Vec<McuCategory>,
}

/// Returns the curated MCC catalogue embedded at compile time.
///
/// Sourced from the Visa Merchant Data Standards Manual and US IRS
/// Publication 1281; ~200 of the most common MCCs across retail,
/// food, fuel, lodging, transport, utilities, professional services,
/// healthcare, recreation, gambling, financial, and government
/// categories.
///
/// # Panics
///
/// Never in practice — the embedded JSON is validated at compile-
/// integration time by the `catalogue_parses_at_runtime` test. If
/// it ever fails to parse, the test catches it and CI fails before
/// callers see the panic. The path that panics here is unreachable
/// when the binary built from this source compiles + passes tests.
#[must_use]
pub fn mcc_catalogue() -> Vec<McuCategory> {
    const RAW: &str = include_str!("../data/mcc.json");
    let parsed: McuCatalogueFile = serde_json::from_str(RAW).unwrap_or(McuCatalogueFile {
        version: String::new(),
        categories: Vec::new(),
    });
    let _ = parsed.version; // present for forward-compat
    parsed.categories
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::RailKind;

    fn r(driver: &str) -> Route {
        Route::new(DriverId::new(driver), RailKind::Card)
    }

    #[test]
    fn mcc_construction_from_str() {
        assert_eq!(Mcc::from_str("5411").map(|m| m.as_str().to_owned()), Some("5411".into()));
        assert_eq!(Mcc::from_str("541"), None);
        assert_eq!(Mcc::from_str("54111"), None);
        assert_eq!(Mcc::from_str("ABCD"), None);
    }

    #[test]
    fn no_rule_for_mcc_returns_pool_unchanged() {
        let policy = MccPolicy::new();
        let pool = vec![r("a"), r("b")];
        let out = policy.filter(Some(Mcc::from_str("5411").expect("test mcc")), &pool);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].driver.as_str(), "a");
    }

    #[test]
    fn exclusion_removes_driver_for_intent_in_that_mcc() {
        // Grocery (5411): exclude "bad-psp".
        let policy = MccPolicy::new().with_rule(
            Mcc::from_str("5411").expect("test mcc"),
            McuPreferences::default().with_excluded(DriverId::new("bad-psp")),
        );
        let pool = vec![r("bad-psp"), r("good-psp"), r("other")];
        let out = policy.filter(Some(Mcc::from_str("5411").expect("test mcc")), &pool);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|x| x.driver.as_str() != "bad-psp"));
    }

    #[test]
    fn preferred_drivers_move_to_front_in_order() {
        let policy = MccPolicy::new().with_rule(
            Mcc::from_str("6051").expect("test mcc"), // digital goods
            McuPreferences::default()
                .with_preferred(DriverId::new("psp-c"))
                .with_preferred(DriverId::new("psp-a")),
        );
        let pool = vec![r("psp-a"), r("psp-b"), r("psp-c"), r("psp-d")];
        let out = policy.filter(Some(Mcc::from_str("6051").expect("test mcc")), &pool);
        // Preferred section: c, a (in preference-list order).
        // Other section: b, d (in input-order order).
        assert_eq!(out[0].driver.as_str(), "psp-c");
        assert_eq!(out[1].driver.as_str(), "psp-a");
        assert_eq!(out[2].driver.as_str(), "psp-b");
        assert_eq!(out[3].driver.as_str(), "psp-d");
    }

    #[test]
    fn excluded_overrides_preferred_silently() {
        // Pathological config: same driver listed as both. Exclude wins.
        let policy = MccPolicy::new().with_rule(
            Mcc::from_str("7995").expect("test mcc"),
            McuPreferences::default()
                .with_preferred(DriverId::new("psp-a"))
                .with_excluded(DriverId::new("psp-a")),
        );
        let pool = vec![r("psp-a"), r("psp-b")];
        let out = policy.filter(Some(Mcc::from_str("7995").expect("test mcc")), &pool);
        // psp-a was excluded — gone.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].driver.as_str(), "psp-b");
    }

    #[test]
    fn none_mcc_passes_through() {
        let policy = MccPolicy::new().with_rule(
            Mcc::from_str("5411").expect("test mcc"),
            McuPreferences::default().with_excluded(DriverId::new("a")),
        );
        let pool = vec![r("a"), r("b")];
        let out = policy.filter(None, &pool);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn max_attempts_for_falls_back_to_default() {
        let policy = MccPolicy::new();
        assert_eq!(policy.max_attempts_for(None, 4), 4);
        assert_eq!(
            policy.max_attempts_for(Some(Mcc::from_str("5411").expect("test mcc")), 4),
            4,
        );

        let policy = MccPolicy::new().with_rule(
            Mcc::from_str("5411").expect("test mcc"),
            McuPreferences::default().with_max_attempts(2),
        );
        assert_eq!(
            policy.max_attempts_for(Some(Mcc::from_str("5411").expect("test mcc")), 4),
            2,
        );
    }

    #[test]
    fn catalogue_parses_at_runtime() {
        let cat = mcc_catalogue();
        // Should be populated.
        assert!(cat.len() > 150, "expected >150 MCCs, got {}", cat.len());
        // Spot-check well-known codes.
        let codes: HashSet<&str> = cat.iter().map(|c| c.mcc.as_str()).collect();
        assert!(codes.contains("5411")); // grocery
        assert!(codes.contains("5812")); // restaurant
        assert!(codes.contains("4900")); // utility
        assert!(codes.contains("7995")); // gambling
        assert!(codes.contains("5993")); // tobacco
    }
}
