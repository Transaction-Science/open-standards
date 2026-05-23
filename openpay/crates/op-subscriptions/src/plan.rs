//! Plan: the price template a subscription is created against.

use op_core::Money;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};

/// Opaque plan id (`UUIDv7`, time-sortable).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlanId(pub Uuid);

impl PlanId {
    /// Mint a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
    /// Wrap an existing UUID.
    #[must_use]
    pub const fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }
    /// The wrapped UUID.
    #[must_use]
    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for PlanId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for PlanId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Billing interval unit. Combined with [`Plan::interval_count`]
/// the full period is `interval_count × interval` (e.g.
/// `interval_count=3, interval=Month` is a quarterly plan).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interval {
    /// Daily.
    Day,
    /// Weekly (7 calendar days).
    Week,
    /// Monthly, calendar-aware (Jan 31 → Feb 28/29 → Mar 31, etc.).
    Month,
    /// Yearly, calendar-aware.
    Year,
}

impl Interval {
    /// Short stable code (`"day"`, `"week"`, `"month"`, `"year"`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Week => "week",
            Self::Month => "month",
            Self::Year => "year",
        }
    }
}

/// Price template. Snapshotted into a [`Subscription`] at creation
/// — later plan edits don't re-price existing subscriptions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// Stable id.
    pub id: PlanId,
    /// Operator-side display name (`"Pro Monthly"`, ...).
    pub name: String,
    /// Charge amount per billing period.
    pub amount: Money,
    /// Interval unit.
    pub interval: Interval,
    /// `interval_count × interval` is the full period.
    pub interval_count: u32,
    /// Free-trial length, in days. `None` = no trial.
    pub trial_days: Option<u32>,
    /// Free-form metadata.
    pub metadata: Vec<(String, String)>,
}

impl Plan {
    /// Construct.
    ///
    /// # Errors
    /// [`Error::Invalid`] if `interval_count` is 0 or `amount` is
    /// negative.
    pub fn new(
        name: impl Into<String>,
        amount: Money,
        interval: Interval,
        interval_count: u32,
    ) -> Result<Self> {
        if interval_count == 0 {
            return Err(Error::Invalid("interval_count must be ≥ 1".into()));
        }
        if amount.minor_units < 0 {
            return Err(Error::Invalid(format!(
                "plan amount must be non-negative, got {}",
                amount.minor_units
            )));
        }
        Ok(Self {
            id: PlanId::new(),
            name: name.into(),
            amount,
            interval,
            interval_count,
            trial_days: None,
            metadata: Vec::new(),
        })
    }

    /// Builder: set trial length in days.
    #[must_use]
    pub const fn with_trial_days(mut self, days: u32) -> Self {
        self.trial_days = Some(days);
        self
    }

    /// Builder: append metadata.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};

    #[test]
    fn rejects_zero_interval_count() {
        assert!(
            Plan::new(
                "p",
                Money::from_minor(100, Currency::USD),
                Interval::Month,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_negative_amount() {
        assert!(
            Plan::new(
                "p",
                Money::from_minor(-1, Currency::USD),
                Interval::Month,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn builder_sets_trial() {
        let p = Plan::new(
            "p",
            Money::from_minor(100, Currency::USD),
            Interval::Month,
            1,
        )
        .unwrap()
        .with_trial_days(14);
        assert_eq!(p.trial_days, Some(14));
    }
}
