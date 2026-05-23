//! Subscription type + status state machine.

use op_core::PaymentMethod;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::plan::Plan;

/// Opaque subscription id (`UUIDv7`, time-sortable).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SubscriptionId(pub Uuid);

impl SubscriptionId {
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

impl Default for SubscriptionId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for SubscriptionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Lifecycle state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    /// Inside the free trial. Bills nothing this period.
    Trialing {
        /// When the trial ends (unix epoch seconds).
        trial_end_unix_secs: u64,
    },
    /// Trial passed (or no trial); charging normally each period.
    Active,
    /// A billing attempt failed; dunning policy decides retries.
    PastDue {
        /// When the first failure of this dunning run happened.
        failed_at_unix_secs: u64,
        /// How many retries have been attempted so far.
        retry_count: u32,
    },
    /// Operator paused billing (e.g. customer is on vacation).
    Paused {
        /// When the pause was applied.
        paused_at_unix_secs: u64,
    },
    /// Terminal. Cancellation can be immediate or scheduled to
    /// fire at the end of the current period; on transition this
    /// variant is set unconditionally — the period-end vs immediate
    /// distinction lives in `cancel_at_period_end` on the
    /// `Subscription` until the cancel actually takes effect.
    Canceled {
        /// When the cancellation took effect (unix epoch seconds).
        canceled_at_unix_secs: u64,
    },
}

impl Status {
    /// Short stable code (`"trialing"`, `"active"`, ...).
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Trialing { .. } => "trialing",
            Self::Active => "active",
            Self::PastDue { .. } => "past_due",
            Self::Paused { .. } => "paused",
            Self::Canceled { .. } => "canceled",
        }
    }

    /// True iff no further transitions are legal.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Canceled { .. })
    }
}

/// One subscription record.
///
/// `Subscription` does not implement `PartialEq` because its
/// `method: PaymentMethod` field carries opaque token data
/// (`Token` wraps bytes that should not participate in equality
/// semantics for security reasons). Compare on
/// [`Subscription::id`] when round-trip testing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Subscription {
    /// Stable id.
    pub id: SubscriptionId,
    /// Operator's idempotency key.
    pub external_id: Option<String>,
    /// Operator-side customer identifier.
    pub customer_ref: String,
    /// Plan snapshot at creation. Editing the plan in the catalog
    /// does NOT re-price this subscription.
    pub plan: Plan,
    /// How the customer is charged (vault token, A2A key, crypto
    /// address, etc.).
    pub method: PaymentMethod,
    /// Lifecycle state.
    pub status: Status,
    /// Start of the current billing period.
    pub current_period_start_unix_secs: u64,
    /// End of the current billing period.
    pub current_period_end_unix_secs: u64,
    /// `true` = the subscription will cancel when the current
    /// period ends. `false` = renews normally (or, if `status` is
    /// `Canceled`, was canceled immediately).
    pub cancel_at_period_end: bool,
    /// Free-form operator metadata.
    pub metadata: Vec<(String, String)>,
}

impl Subscription {
    /// Construct a fresh subscription. Picks the starting status
    /// based on the plan's trial: `Trialing` if `trial_days` is
    /// set, otherwise `Active`. The current period is computed by
    /// the [`crate::BillingScheduler`] via [`first_period`].
    ///
    /// # Errors
    /// Forwarded from [`crate::scheduler::BillingScheduler::first_period`].
    pub fn new(
        customer_ref: impl Into<String>,
        plan: Plan,
        method: PaymentMethod,
        now_unix_secs: u64,
    ) -> Result<Self> {
        let (start, end) = crate::scheduler::first_period(&plan, now_unix_secs);
        let status = match plan.trial_days {
            Some(d) if d > 0 => Status::Trialing {
                trial_end_unix_secs: now_unix_secs + (u64::from(d) * 86_400),
            },
            _ => Status::Active,
        };
        Ok(Self {
            id: SubscriptionId::new(),
            external_id: None,
            customer_ref: customer_ref.into(),
            plan,
            method,
            status,
            current_period_start_unix_secs: start,
            current_period_end_unix_secs: end,
            cancel_at_period_end: false,
            metadata: Vec::new(),
        })
    }

    /// Builder: attach an external id.
    #[must_use]
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Builder: append metadata.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }

    // ---- State transitions ----

    /// `Trialing → Active` when the trial end has passed (the
    /// scheduler calls this after the trial-end timestamp is in
    /// the past). Idempotent against already-Active state.
    ///
    /// # Errors
    /// `Error::InvalidTransition` from any non-Trialing/Active state.
    pub fn promote_from_trial(&mut self) -> Result<()> {
        match self.status {
            Status::Trialing { .. } => {
                self.status = Status::Active;
                Ok(())
            }
            Status::Active => Ok(()),
            ref other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Mark a billing attempt failed. Tips into `PastDue` or
    /// increments the retry counter if already there. Idempotent
    /// in the sense that two calls with monotonic timestamps
    /// produce predictable counter values.
    ///
    /// # Errors
    /// `Error::InvalidTransition` from terminal states.
    pub fn record_billing_failure(&mut self, at_unix_secs: u64) -> Result<()> {
        match &self.status {
            Status::Trialing { .. } | Status::Active => {
                self.status = Status::PastDue {
                    failed_at_unix_secs: at_unix_secs,
                    retry_count: 0,
                };
                Ok(())
            }
            Status::PastDue {
                failed_at_unix_secs,
                retry_count,
            } => {
                let first = *failed_at_unix_secs;
                let next = retry_count.saturating_add(1);
                self.status = Status::PastDue {
                    failed_at_unix_secs: first,
                    retry_count: next,
                };
                Ok(())
            }
            other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// `PastDue → Active` after a successful retry.
    ///
    /// # Errors
    /// `Error::InvalidTransition` if not `PastDue`.
    pub fn record_billing_recovered(&mut self) -> Result<()> {
        match self.status {
            Status::PastDue { .. } => {
                self.status = Status::Active;
                Ok(())
            }
            ref other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Pause. Legal from `Active` or `Trialing`. Paused
    /// subscriptions don't roll periods.
    ///
    /// # Errors
    /// `Error::InvalidTransition` from `PastDue` / terminal.
    pub fn pause(&mut self, at_unix_secs: u64) -> Result<()> {
        match self.status {
            Status::Active | Status::Trialing { .. } => {
                self.status = Status::Paused {
                    paused_at_unix_secs: at_unix_secs,
                };
                Ok(())
            }
            ref other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Resume. Legal from `Paused` only.
    ///
    /// # Errors
    /// `Error::InvalidTransition` from any non-Paused state.
    pub fn resume(&mut self) -> Result<()> {
        match self.status {
            Status::Paused { .. } => {
                self.status = Status::Active;
                Ok(())
            }
            ref other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Schedule cancellation at the end of the current period.
    /// `cancel_at_period_end` flips to `true`; the actual status
    /// transition to `Canceled` happens when the scheduler rolls
    /// the period and sees the flag.
    pub fn schedule_cancel_at_period_end(&mut self) {
        self.cancel_at_period_end = true;
    }

    /// Cancel immediately. Sets status to `Canceled`. Idempotent.
    pub fn cancel_now(&mut self, at_unix_secs: u64) {
        self.status = Status::Canceled {
            canceled_at_unix_secs: at_unix_secs,
        };
        self.cancel_at_period_end = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Interval, Plan};
    use op_core::{Currency, Money, VaultRef};

    fn plan_basic() -> Plan {
        Plan::new(
            "basic",
            Money::from_minor(900, Currency::USD),
            Interval::Month,
            1,
        )
        .unwrap()
    }

    fn method() -> PaymentMethod {
        PaymentMethod::Vault(VaultRef::new("tok_v7_test"))
    }

    #[test]
    fn no_trial_starts_active() {
        let s = Subscription::new("cust-1", plan_basic(), method(), 1_700_000_000).unwrap();
        assert_eq!(s.status.code(), "active");
    }

    #[test]
    fn trial_starts_trialing() {
        let p = plan_basic().with_trial_days(14);
        let s = Subscription::new("cust-1", p, method(), 1_700_000_000).unwrap();
        match s.status {
            Status::Trialing {
                trial_end_unix_secs,
            } => {
                assert_eq!(trial_end_unix_secs, 1_700_000_000 + 14 * 86_400);
            }
            other => panic!("expected trialing, got {other:?}"),
        }
    }

    #[test]
    fn promote_from_trial_active_idempotent() {
        let p = plan_basic().with_trial_days(7);
        let mut s = Subscription::new("c", p, method(), 1_700_000_000).unwrap();
        s.promote_from_trial().unwrap();
        assert_eq!(s.status, Status::Active);
        s.promote_from_trial().unwrap(); // idempotent
    }

    #[test]
    fn billing_failure_increments_retry() {
        let mut s = Subscription::new("c", plan_basic(), method(), 1_000).unwrap();
        s.record_billing_failure(2_000).unwrap();
        match s.status {
            Status::PastDue { retry_count, .. } => assert_eq!(retry_count, 0),
            _ => panic!("expected PastDue"),
        }
        s.record_billing_failure(3_000).unwrap();
        match s.status {
            Status::PastDue {
                retry_count,
                failed_at_unix_secs,
            } => {
                assert_eq!(retry_count, 1);
                assert_eq!(failed_at_unix_secs, 2_000); // first failure preserved
            }
            _ => panic!("expected PastDue"),
        }
    }

    #[test]
    fn recover_brings_active() {
        let mut s = Subscription::new("c", plan_basic(), method(), 1_000).unwrap();
        s.record_billing_failure(2_000).unwrap();
        s.record_billing_recovered().unwrap();
        assert_eq!(s.status, Status::Active);
    }

    #[test]
    fn pause_resume_cycle() {
        let mut s = Subscription::new("c", plan_basic(), method(), 1_000).unwrap();
        s.pause(2_000).unwrap();
        assert!(matches!(s.status, Status::Paused { .. }));
        s.resume().unwrap();
        assert_eq!(s.status, Status::Active);
    }

    #[test]
    fn cancel_now_is_terminal() {
        let mut s = Subscription::new("c", plan_basic(), method(), 1_000).unwrap();
        s.cancel_now(2_000);
        assert!(s.status.is_terminal());
        // Further transitions blocked.
        assert!(s.pause(3_000).is_err());
    }

    #[test]
    fn schedule_cancel_at_period_end_flips_flag() {
        let mut s = Subscription::new("c", plan_basic(), method(), 1_000).unwrap();
        assert!(!s.cancel_at_period_end);
        s.schedule_cancel_at_period_end();
        assert!(s.cancel_at_period_end);
        // Status unchanged until the scheduler rolls the period.
        assert_eq!(s.status, Status::Active);
    }
}
