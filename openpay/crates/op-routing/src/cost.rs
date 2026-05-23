//! Cost models for least-cost routing.
//!
//! A landed-cost estimate has four components:
//!
//! 1. **Interchange** — paid by the acquirer to the issuer, set by
//!    the card network's interchange table. Country + MCC + card
//!    type dependent. Expressed in basis points of the principal.
//!
//! 2. **Scheme fees** — paid by the acquirer to the network (Visa,
//!    Mastercard, etc.) Per-transaction + per-volume. Also bps.
//!
//! 3. **PSP markup** — what the PSP (Stripe, Adyen, Worldpay,
//!    Hyperswitch) keeps on top. Where the operator has commercial
//!    leverage.
//!
//! 4. **Fixed per-transaction** — flat-fee component. Critical at
//!    small ticket sizes; trivial at large.
//!
//! Total estimated cost for principal `P` in minor units is:
//!
//! ```text
//! cost = floor(P * (interchange + scheme + psp) / 10_000) + fixed
//! ```
//!
//! All math is integer-exact on `i64` minor units. No `f64`.

use op_core::Money;
use std::collections::HashMap;

use crate::route::{DriverId, Route};

/// Basis points (1 bp = 0.01 %). Values are `u32` to comfortably hold
/// any reasonable scheme + interchange total (real-world maxima are
/// under 500 bp).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bps(pub u32);

impl Bps {
    /// Construct.
    #[must_use]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    /// Inner value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A cost model — the bp + fixed components used by each
/// [`CostEstimator`] flavor.
///
/// Different estimator implementations interpret the same struct
/// differently: `InterchangePlusEstimator` adds all four components;
/// `BlendedRateEstimator` collapses interchange / scheme / psp into
/// a single blended bp number; `TieredFixedEstimator` ignores bps
/// for amounts under a tier threshold and applies a flat fee.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Interchange portion in basis points.
    pub interchange: Bps,
    /// Scheme fees in basis points.
    pub scheme_fees: Bps,
    /// PSP markup in basis points.
    pub psp_markup: Bps,
    /// Fixed per-transaction component (must match the currency of
    /// any intent priced against this model).
    pub fixed_per_tx: Money,
}

impl CostModel {
    /// Construct.
    #[must_use]
    pub const fn new(
        interchange: Bps,
        scheme_fees: Bps,
        psp_markup: Bps,
        fixed_per_tx: Money,
    ) -> Self {
        Self {
            interchange,
            scheme_fees,
            psp_markup,
            fixed_per_tx,
        }
    }

    /// Total bps across all variable components.
    #[must_use]
    pub const fn total_bps(&self) -> u32 {
        self.interchange.0 + self.scheme_fees.0 + self.psp_markup.0
    }
}

/// Pluggable cost estimator.
///
/// Implementors compute the landed cost in minor units for a given
/// intent + route. The crate ships three reference flavors
/// ([`InterchangePlusEstimator`], [`BlendedRateEstimator`],
/// [`TieredFixedEstimator`]). Operators with calibrated cost models
/// implement their own.
pub trait CostEstimator: Send + Sync {
    /// Estimate the landed cost of charging `intent.amount` via `route`.
    ///
    /// The return is in the same currency as `intent.amount`.
    /// Saturating arithmetic — overflow returns `i64::MAX` minor
    /// units rather than panicking. Real transaction sizes never
    /// approach overflow.
    fn estimate(&self, intent: &PaymentIntentRef<'_>, route: &Route) -> Money;
}

/// A minimal payment-intent view that the estimator trait reads.
///
/// We borrow this rather than re-exporting `op-orchestrator`'s
/// `PaymentIntent` so `op-routing` does not pull the orchestrator
/// crate. Operators construct a `PaymentIntentRef` from whatever
/// intent type they hold.
#[derive(Clone, Copy, Debug)]
pub struct PaymentIntentRef<'a> {
    /// Principal to charge.
    pub amount: Money,
    /// ISO 18245 MCC (if known).
    pub mcc: Option<&'a str>,
    /// ISO 3166-1 alpha-2 customer country (if known).
    pub customer_country: Option<&'a str>,
    /// ISO 3166-1 alpha-2 merchant country (if known).
    pub merchant_country: Option<&'a str>,
}

impl<'a> PaymentIntentRef<'a> {
    /// Construct from a bare amount.
    #[must_use]
    pub const fn from_amount(amount: Money) -> Self {
        Self {
            amount,
            mcc: None,
            customer_country: None,
            merchant_country: None,
        }
    }

    /// Builder: set MCC.
    #[must_use]
    pub const fn with_mcc(mut self, mcc: &'a str) -> Self {
        self.mcc = Some(mcc);
        self
    }

    /// Builder: set customer country.
    #[must_use]
    pub const fn with_customer_country(mut self, c: &'a str) -> Self {
        self.customer_country = Some(c);
        self
    }

    /// Builder: set merchant country.
    #[must_use]
    pub const fn with_merchant_country(mut self, c: &'a str) -> Self {
        self.merchant_country = Some(c);
        self
    }
}

/// Saturating `principal_minor * bps / 10_000` in `i64`.
fn apply_bps(principal_minor: i64, bps: u32) -> i64 {
    // Use i128 for the intermediate to avoid mid-multiplication
    // overflow on large principals.
    let intermediate = i128::from(principal_minor).saturating_mul(i128::from(bps)) / 10_000_i128;
    i64::try_from(intermediate).unwrap_or(i64::MAX)
}

/// Interchange-plus: charges interchange + scheme + psp markup +
/// fixed, all separate. The honest, transparent commercial structure.
///
/// Per-route models keyed by `(driver, country?)`. If the intent's
/// country is set and a country-specific model exists, that wins;
/// otherwise the driver default is used. If neither is present, the
/// route is priced as `i64::MAX` minor units (effectively
/// disqualified).
#[derive(Clone, Debug, Default)]
pub struct InterchangePlusEstimator {
    /// Driver-default models.
    default_by_driver: HashMap<DriverId, CostModel>,
    /// Per-country overrides keyed by `(driver, country)`.
    overrides: HashMap<(DriverId, String), CostModel>,
}

impl InterchangePlusEstimator {
    /// Empty estimator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default model for a driver (used when no country
    /// override matches).
    #[must_use]
    pub fn with_default(mut self, driver: DriverId, model: CostModel) -> Self {
        self.default_by_driver.insert(driver, model);
        self
    }

    /// Set a country-specific model for a driver.
    #[must_use]
    pub fn with_override(
        mut self,
        driver: DriverId,
        country: impl Into<String>,
        model: CostModel,
    ) -> Self {
        self.overrides.insert((driver, country.into()), model);
        self
    }

    fn pick_model(&self, route: &Route) -> Option<&CostModel> {
        if let Some(country) = route.country.as_deref()
            && let Some(m) = self
                .overrides
                .get(&(route.driver.clone(), country.to_owned()))
        {
            return Some(m);
        }
        self.default_by_driver.get(&route.driver)
    }
}

impl CostEstimator for InterchangePlusEstimator {
    fn estimate(&self, intent: &PaymentIntentRef<'_>, route: &Route) -> Money {
        let currency = intent.amount.currency;
        let Some(model) = self.pick_model(route) else {
            return Money::from_minor(i64::MAX, currency);
        };
        let variable = apply_bps(intent.amount.minor_units, model.total_bps());
        let fixed = if model.fixed_per_tx.currency == currency {
            model.fixed_per_tx.minor_units
        } else {
            // Mismatched currency on the fixed component is an
            // operator configuration error; treat as zero rather
            // than panic so a bad model doesn't take down routing.
            0
        };
        let total = variable.saturating_add(fixed);
        Money::from_minor(total, currency)
    }
}

/// Blended-rate: collapses interchange + scheme + psp into a single
/// rate quoted in basis points. The simpler commercial structure
/// most acquirers offer mid-market merchants.
///
/// Model lookup uses `CostModel.psp_markup` as the blended rate;
/// `interchange` and `scheme_fees` are ignored. Fixed-per-tx still
/// applies.
#[derive(Clone, Debug, Default)]
pub struct BlendedRateEstimator {
    by_driver: HashMap<DriverId, CostModel>,
}

impl BlendedRateEstimator {
    /// Empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a blended model. Only `psp_markup` (read as blended
    /// rate) and `fixed_per_tx` are consulted; the other bps fields
    /// are ignored by this estimator.
    #[must_use]
    pub fn with(mut self, driver: DriverId, model: CostModel) -> Self {
        self.by_driver.insert(driver, model);
        self
    }
}

impl CostEstimator for BlendedRateEstimator {
    fn estimate(&self, intent: &PaymentIntentRef<'_>, route: &Route) -> Money {
        let currency = intent.amount.currency;
        let Some(model) = self.by_driver.get(&route.driver) else {
            return Money::from_minor(i64::MAX, currency);
        };
        let variable = apply_bps(intent.amount.minor_units, model.psp_markup.0);
        let fixed = if model.fixed_per_tx.currency == currency {
            model.fixed_per_tx.minor_units
        } else {
            0
        };
        Money::from_minor(variable.saturating_add(fixed), currency)
    }
}

/// Tiered-fixed: flat-fee tiers ladder-style. For each route, the
/// operator supplies an ordered list of tiers `(threshold_minor,
/// fee_minor)`; the estimator picks the smallest threshold strictly
/// greater than the principal. Used by interchange-fee programs and
/// some A2A rails (`FedNow` flat $0.045, RTP flat $0.045, etc.).
#[derive(Clone, Debug, Default)]
pub struct TieredFixedEstimator {
    tiers_by_driver: HashMap<DriverId, Vec<(i64, Money)>>,
}

impl TieredFixedEstimator {
    /// Empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register tiers for a driver. The crate sorts them ascending
    /// by threshold on insert so the caller can supply any order.
    #[must_use]
    pub fn with(mut self, driver: DriverId, mut tiers: Vec<(i64, Money)>) -> Self {
        tiers.sort_by_key(|(t, _)| *t);
        self.tiers_by_driver.insert(driver, tiers);
        self
    }
}

impl CostEstimator for TieredFixedEstimator {
    fn estimate(&self, intent: &PaymentIntentRef<'_>, route: &Route) -> Money {
        let currency = intent.amount.currency;
        let Some(tiers) = self.tiers_by_driver.get(&route.driver) else {
            return Money::from_minor(i64::MAX, currency);
        };
        // Pick the first tier whose threshold is >= principal. If
        // principal exceeds every threshold, charge the top tier.
        let principal = intent.amount.minor_units;
        let chosen = tiers
            .iter()
            .find(|(t, _)| *t >= principal)
            .or_else(|| tiers.last())
            .map(|(_, fee)| *fee);
        let Some(fee) = chosen else {
            return Money::from_minor(i64::MAX, currency);
        };
        if fee.currency != currency {
            return Money::from_minor(i64::MAX, currency);
        }
        fee
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, RailKind};

    fn intent_usd(minor: i64) -> PaymentIntentRef<'static> {
        PaymentIntentRef::from_amount(Money::from_minor(minor, Currency::USD))
    }

    fn route(name: &str) -> Route {
        Route::new(DriverId::new(name), RailKind::Card).with_country("US")
    }

    #[test]
    fn interchange_plus_sums_all_components() {
        let est = InterchangePlusEstimator::new().with_default(
            DriverId::new("stripe"),
            CostModel::new(
                Bps::new(150), // interchange 1.5%
                Bps::new(25),  // scheme 0.25%
                Bps::new(25),  // psp 0.25%
                Money::from_minor(30, Currency::USD),
            ),
        );
        // 10000 minor ($100) at 2% = 200 minor, + 30 fixed = 230.
        let cost = est.estimate(&intent_usd(10_000), &route("stripe"));
        assert_eq!(cost.minor_units, 230);
        assert_eq!(cost.currency, Currency::USD);
    }

    #[test]
    fn interchange_plus_unknown_driver_is_disqualified() {
        let est = InterchangePlusEstimator::new();
        let cost = est.estimate(&intent_usd(10_000), &route("ghost"));
        assert_eq!(cost.minor_units, i64::MAX);
    }

    #[test]
    fn interchange_plus_country_override_wins() {
        let est = InterchangePlusEstimator::new()
            .with_default(
                DriverId::new("adyen"),
                CostModel::new(
                    Bps::new(200),
                    Bps::new(0),
                    Bps::new(0),
                    Money::from_minor(0, Currency::USD),
                ),
            )
            .with_override(
                DriverId::new("adyen"),
                "US",
                CostModel::new(
                    Bps::new(100),
                    Bps::new(0),
                    Bps::new(0),
                    Money::from_minor(0, Currency::USD),
                ),
            );
        let cost = est.estimate(&intent_usd(10_000), &route("adyen"));
        // US override (1%) beats default (2%) → 100 minor.
        assert_eq!(cost.minor_units, 100);
    }

    #[test]
    fn blended_rate_uses_psp_markup_only() {
        let est = BlendedRateEstimator::new().with(
            DriverId::new("worldpay"),
            CostModel::new(
                Bps::new(9999), // interchange (ignored)
                Bps::new(9999), // scheme (ignored)
                Bps::new(290),  // 2.9% blended
                Money::from_minor(30, Currency::USD),
            ),
        );
        // 10000 * 290 / 10000 = 290, + 30 fixed = 320.
        let cost = est.estimate(&intent_usd(10_000), &route("worldpay"));
        assert_eq!(cost.minor_units, 320);
    }

    #[test]
    fn tiered_fixed_picks_first_matching_threshold() {
        let est = TieredFixedEstimator::new().with(
            DriverId::new("fednow"),
            vec![
                (10_000, Money::from_minor(5, Currency::USD)),
                (1_000_000, Money::from_minor(45, Currency::USD)),
            ],
        );
        // $50 → first tier (5 minor).
        assert_eq!(
            est.estimate(&intent_usd(5_000), &route("fednow"))
                .minor_units,
            5
        );
        // $9999 → first tier still (10_000 is >= 9999).
        assert_eq!(
            est.estimate(&intent_usd(9_999), &route("fednow"))
                .minor_units,
            5
        );
        // $200 → first tier.
        assert_eq!(
            est.estimate(&intent_usd(20_000), &route("fednow"))
                .minor_units,
            45
        );
        // $20_000 → top tier (last entry).
        assert_eq!(
            est.estimate(&intent_usd(99_999_999), &route("fednow"))
                .minor_units,
            45
        );
    }

    #[test]
    fn tiered_fixed_sorts_input_tiers() {
        let est = TieredFixedEstimator::new().with(
            DriverId::new("rtp"),
            vec![
                // Intentionally unsorted.
                (1_000_000, Money::from_minor(45, Currency::USD)),
                (10_000, Money::from_minor(5, Currency::USD)),
            ],
        );
        assert_eq!(
            est.estimate(&intent_usd(5_000), &route("rtp")).minor_units,
            5
        );
    }

    #[test]
    fn apply_bps_no_overflow_at_max() {
        let r = apply_bps(i64::MAX, 290);
        assert!(r < i64::MAX);
        assert!(r > 0);
    }

    #[test]
    fn cost_model_total_bps_sums() {
        let m = CostModel::new(
            Bps::new(150),
            Bps::new(25),
            Bps::new(25),
            Money::from_minor(0, Currency::USD),
        );
        assert_eq!(m.total_bps(), 200);
    }

    #[test]
    fn currency_mismatch_on_fixed_treated_as_zero() {
        let est = InterchangePlusEstimator::new().with_default(
            DriverId::new("eurpsp"),
            CostModel::new(
                Bps::new(100),
                Bps::new(0),
                Bps::new(0),
                Money::from_minor(50, Currency::EUR), // wrong currency
            ),
        );
        // Variable = 100 minor; fixed dropped to 0 due to currency
        // mismatch; result = 100.
        let cost = est.estimate(&intent_usd(10_000), &route("eurpsp"));
        assert_eq!(cost.minor_units, 100);
        assert_eq!(cost.currency, Currency::USD);
    }
}
