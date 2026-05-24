//! Principal-vs-agent indicators per ASC 606-10-55-37 through 55-40.
//!
//! When another party is involved in providing goods or services to a
//! customer, the entity must determine whether its promise is a
//! performance obligation to provide the specified goods or services
//! itself (principal) or to arrange for the other party to provide
//! them (agent).
//!
//! ASC 606-10-55-39 lists three indicators that an entity is the
//! principal:
//!
//! 1. The entity is **primarily responsible** for fulfilling the promise
//!    to provide the specified good or service.
//! 2. The entity has **inventory risk** before the specified good or
//!    service has been transferred to the customer (or after transfer of
//!    control, such as a right of return).
//! 3. The entity has **discretion in establishing the price** the
//!    customer pays.
//!
//! The standard explicitly rejected the older indicator framework's
//! "controls credit risk" factor — that is now considered a secondary
//! indicator at best.
//!
//! This module exposes the indicator vector and a default scoring
//! function. The classification is consultative, not deterministic;
//! firms should review with their auditor and pin the answer per
//! contract / per arrangement.

use serde::{Deserialize, Serialize};

use crate::contract::Presentation;

/// Vector of indicators per ASC 606-10-55-39. `None` means the
/// indicator was not assessed (treated as a soft "no" for scoring).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Indicators {
    /// The entity is primarily responsible for fulfilling the promise.
    pub primarily_responsible: Option<bool>,
    /// The entity bears inventory risk before transfer.
    pub inventory_risk: Option<bool>,
    /// The entity has discretion in establishing the price.
    pub price_discretion: Option<bool>,
}

impl Indicators {
    /// Builder: set `primarily_responsible`.
    #[must_use]
    pub const fn with_primarily_responsible(mut self, b: bool) -> Self {
        self.primarily_responsible = Some(b);
        self
    }

    /// Builder: set `inventory_risk`.
    #[must_use]
    pub const fn with_inventory_risk(mut self, b: bool) -> Self {
        self.inventory_risk = Some(b);
        self
    }

    /// Builder: set `price_discretion`.
    #[must_use]
    pub const fn with_price_discretion(mut self, b: bool) -> Self {
        self.price_discretion = Some(b);
        self
    }
}

/// Default classifier: principal if at least two of three indicators
/// resolve to `Some(true)`. Otherwise agent.
///
/// This is consciously a coarse rule — the standard intentionally does
/// not specify a hard threshold and expects judgement. Operators that
/// need a different policy should bypass this function and set
/// [`crate::contract::Presentation`] directly on each obligation.
#[must_use]
pub fn classify(indicators: &Indicators) -> Presentation {
    let yes_count = [
        indicators.primarily_responsible,
        indicators.inventory_risk,
        indicators.price_discretion,
    ]
    .iter()
    .filter(|x| **x == Some(true))
    .count();
    if yes_count >= 2 {
        Presentation::Gross
    } else {
        Presentation::Net
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marketplace_facilitator_is_agent() {
        let ind = Indicators::default()
            .with_primarily_responsible(false)
            .with_inventory_risk(false)
            .with_price_discretion(false);
        assert_eq!(classify(&ind), Presentation::Net);
    }

    #[test]
    fn full_stack_retailer_is_principal() {
        let ind = Indicators::default()
            .with_primarily_responsible(true)
            .with_inventory_risk(true)
            .with_price_discretion(true);
        assert_eq!(classify(&ind), Presentation::Gross);
    }

    #[test]
    fn two_of_three_is_principal() {
        let ind = Indicators::default()
            .with_primarily_responsible(true)
            .with_inventory_risk(false)
            .with_price_discretion(true);
        assert_eq!(classify(&ind), Presentation::Gross);
    }

    #[test]
    fn one_of_three_is_agent() {
        let ind = Indicators::default()
            .with_primarily_responsible(true)
            .with_inventory_risk(false)
            .with_price_discretion(false);
        assert_eq!(classify(&ind), Presentation::Net);
    }
}
