//! Holdback (reserve) policy.
//!
//! A holdback is the slice of a batch's gross amount that the
//! operator withholds from payout to cover risk: open disputes,
//! likely chargebacks, refund pressure. Two layers:
//!
//! - `flat_rate_bps` — the operator's static reserve, expressed
//!   in basis points (1 bp = 0.01%). For a 50bp reserve on a
//!   $10,000 batch, holdback is $50.
//! - `dispute_adjustment_bps` — an additional reserve that scales
//!   with the dispute load on the underlying transactions
//!   (operator-supplied — we don't read the dispute store from
//!   here, callers compute it and pass it in).
//!
//! Holdback floors at the gross — we never "release" more than
//! the operator earned.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Configuration knob for holdback computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HoldbackPolicy {
    /// Static reserve in basis points. `0` means no reserve.
    pub flat_rate_bps: u16,
    /// Per-batch ceiling in basis points, including the dispute
    /// adjustment. `10_000` (= 100%) means no ceiling beyond the
    /// gross. Common operator values: `2_000` (20%).
    pub max_total_bps: u16,
}

impl Default for HoldbackPolicy {
    fn default() -> Self {
        Self {
            flat_rate_bps: 0,
            max_total_bps: 10_000,
        }
    }
}

impl HoldbackPolicy {
    /// A policy that withholds nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            flat_rate_bps: 0,
            max_total_bps: 10_000,
        }
    }

    /// Build a policy from a flat-rate basis-point reserve.
    #[must_use]
    pub const fn flat(flat_rate_bps: u16) -> Self {
        Self {
            flat_rate_bps,
            max_total_bps: 10_000,
        }
    }

    /// Builder: cap the total reserve at `max_total_bps` (flat +
    /// dispute adjustment).
    #[must_use]
    pub const fn with_ceiling(mut self, max_total_bps: u16) -> Self {
        self.max_total_bps = max_total_bps;
        self
    }

    /// Compute the holdback for a `gross` batch and an operator-
    /// supplied `dispute_adjustment_bps`.
    ///
    /// # Errors
    /// Propagates [`op_core::Error::Overflow`] if intermediate
    /// arithmetic overflows `i64` (only with absurd inputs).
    pub fn compute(&self, gross: Money, dispute_adjustment_bps: u16) -> Result<Holdback> {
        let combined_bps =
            u32::from(self.flat_rate_bps).saturating_add(u32::from(dispute_adjustment_bps));
        let effective_bps = combined_bps.min(u32::from(self.max_total_bps));
        // gross * effective_bps / 10_000 with checked arithmetic.
        // i64::MAX / 10_000 ≈ 9.2e14 minor units. Even a wildly
        // extreme batch ($1B in cents = 1e11 minor) * 65_535 fits.
        let minor = gross.minor_units;
        let scaled = i128::from(minor) * i128::from(effective_bps) / 10_000;
        let scaled_minor = i64::try_from(scaled).map_err(|_| op_core::Error::Overflow)?;
        let reserve = Money::from_minor(scaled_minor.abs(), gross.currency);
        // Clamp to gross.
        let reserve = if reserve.minor_units > gross.minor_units.abs() {
            Money::from_minor(gross.minor_units.abs(), gross.currency)
        } else {
            reserve
        };
        Ok(Holdback {
            gross,
            reserve,
            flat_rate_bps: self.flat_rate_bps,
            dispute_adjustment_bps,
        })
    }
}

/// The result of applying [`HoldbackPolicy`] to a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Holdback {
    /// The gross batch amount before reserve.
    pub gross: Money,
    /// The withheld portion.
    pub reserve: Money,
    /// The flat rate applied (basis points).
    pub flat_rate_bps: u16,
    /// The risk adjustment applied (basis points).
    pub dispute_adjustment_bps: u16,
}

impl Holdback {
    /// Net amount paid out (`gross − reserve`).
    ///
    /// # Errors
    /// Propagates [`op_core::Error`] on the subtraction.
    pub fn net(&self) -> Result<Money> {
        Ok(self.gross.checked_sub(self.reserve)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};

    #[test]
    fn no_reserve_passes_through() {
        let p = HoldbackPolicy::none();
        let h = p
            .compute(Money::from_minor(1_000_000, Currency::USD), 0)
            .unwrap();
        assert_eq!(h.reserve, Money::from_minor(0, Currency::USD));
        assert_eq!(h.net().unwrap(), h.gross);
    }

    #[test]
    fn flat_50_bps_on_10k() {
        // 50bp on $10,000.00 = $50.00 = 5_000 minor.
        let p = HoldbackPolicy::flat(50);
        let h = p
            .compute(Money::from_minor(1_000_000, Currency::USD), 0)
            .unwrap();
        assert_eq!(h.reserve, Money::from_minor(5_000, Currency::USD));
        assert_eq!(h.net().unwrap(), Money::from_minor(995_000, Currency::USD));
    }

    #[test]
    fn dispute_adjustment_adds_to_flat() {
        // 100bp flat + 200bp dispute = 300bp on $1,000 = $30.
        let p = HoldbackPolicy::flat(100);
        let h = p
            .compute(Money::from_minor(100_000, Currency::USD), 200)
            .unwrap();
        assert_eq!(h.reserve, Money::from_minor(3_000, Currency::USD));
    }

    #[test]
    fn ceiling_caps_combined() {
        // 100bp flat + 9000bp dispute would be 91% reserve, but
        // ceiling is 20%.
        let p = HoldbackPolicy::flat(100).with_ceiling(2_000);
        let h = p
            .compute(Money::from_minor(100_000, Currency::USD), 9_000)
            .unwrap();
        // 20% of $1,000 = $200 = 20_000 minor.
        assert_eq!(h.reserve, Money::from_minor(20_000, Currency::USD));
    }

    #[test]
    fn reserve_clamps_to_gross() {
        // Wildly large dispute_adjustment (uncapped) should never
        // exceed the gross.
        let p = HoldbackPolicy::flat(u16::MAX).with_ceiling(u16::MAX);
        let gross = Money::from_minor(100_000, Currency::USD);
        let h = p.compute(gross, u16::MAX).unwrap();
        assert_eq!(h.reserve, gross);
        assert_eq!(h.net().unwrap(), Money::from_minor(0, Currency::USD));
    }
}
