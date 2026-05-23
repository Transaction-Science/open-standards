//! [`Refund`] domain type, its [`Status`] state machine, and
//! [`RefundId`].

use op_core::Money;
use op_ledger::TransactionId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::reason::RefundReason;

/// Opaque id for a refund.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RefundId(pub Uuid);

impl RefundId {
    /// Mint a fresh id (UUID v7 — time-sortable).
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Wrap an existing UUID (replay / migration).
    #[must_use]
    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    /// The wrapped UUID.
    #[must_use]
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for RefundId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for RefundId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Where a refund is in its lifecycle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    /// Operator created the refund record; not yet sent to the rail.
    Requested,
    /// Sent to the rail; awaiting response.
    Submitted {
        /// PSP-assigned id for the refund operation.
        psp_refund_id: String,
    },
    /// Rail acknowledged; awaiting settlement.
    Approved {
        /// PSP-assigned id (carried forward from `Submitted`).
        psp_refund_id: String,
    },
    /// Rail confirmed settlement; the operator can now post a
    /// reversing ledger transaction.
    Settled {
        /// PSP-assigned id.
        psp_refund_id: String,
        /// When the rail says it settled (unix epoch seconds).
        settled_at_unix_secs: u64,
    },
    /// Rail rejected the refund. Terminal.
    Declined {
        /// Normalized error code.
        code: String,
        /// Human-readable message.
        message: String,
    },
    /// Rail returned an error *after* approving (rare — usually a
    /// post-settlement reversal failure). Terminal.
    Failed {
        /// Normalized error code.
        code: String,
        /// Human-readable message.
        message: String,
    },
}

impl Status {
    /// True for `Settled`, `Declined`, `Failed`.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Settled { .. } | Self::Declined { .. } | Self::Failed { .. }
        )
    }

    /// Short string code, useful for filtering and reporting.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Requested => "requested",
            Self::Submitted { .. } => "submitted",
            Self::Approved { .. } => "approved",
            Self::Settled { .. } => "settled",
            Self::Declined { .. } => "declined",
            Self::Failed { .. } => "failed",
        }
    }
}

/// One refund record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refund {
    /// Stable id.
    pub id: RefundId,
    /// The ledger transaction being refunded.
    pub original_tx_id: TransactionId,
    /// Operator-supplied idempotency token. Conventionally re-used
    /// as the `external_id` on the reversing ledger transaction
    /// when the refund settles — see crate-level docs.
    pub external_id: Option<String>,
    /// Refund amount. `<=` the original tx's amount; the store
    /// enforces aggregate-amount-vs-original (production stores) or
    /// trusts the caller (in-memory reference).
    pub amount: Money,
    /// Why the refund was issued.
    pub reason: RefundReason,
    /// Current lifecycle position.
    pub status: Status,
    /// When the operator created the record (unix epoch seconds).
    pub requested_at_unix_secs: u64,
    /// Free-form metadata: order line ids, internal ticket
    /// references, anything operator-side that doesn't fit the
    /// structured fields.
    pub metadata: Vec<(String, String)>,
}

impl Refund {
    /// Construct a refund in the [`Status::Requested`] state.
    ///
    /// # Errors
    /// [`Error::Invalid`] if `amount` is negative.
    pub fn new(
        original_tx_id: TransactionId,
        amount: Money,
        reason: RefundReason,
        requested_at_unix_secs: u64,
    ) -> Result<Self> {
        if amount.minor_units < 0 {
            return Err(Error::Invalid(format!(
                "refund amount must be non-negative, got {}",
                amount.minor_units
            )));
        }
        Ok(Self {
            id: RefundId::new(),
            original_tx_id,
            external_id: None,
            amount,
            reason,
            status: Status::Requested,
            requested_at_unix_secs,
            metadata: Vec::new(),
        })
    }

    /// Builder: attach an external id for idempotency.
    #[must_use]
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Builder: attach a metadata key/value pair.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }

    // ---- State transitions ----

    /// Move from `Requested` to `Submitted`. The rail call has been
    /// sent and the PSP returned the refund id.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] from any state but `Requested`.
    pub fn submit(&mut self, psp_refund_id: impl Into<String>) -> Result<()> {
        match self.status {
            Status::Requested => {
                self.status = Status::Submitted {
                    psp_refund_id: psp_refund_id.into(),
                };
                Ok(())
            }
            ref other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Move from `Submitted` to `Approved`. The rail acknowledged
    /// the refund request but hasn't yet confirmed settlement.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if not in `Submitted`.
    pub fn approve(&mut self) -> Result<()> {
        match &self.status {
            Status::Submitted { psp_refund_id } => {
                let psp = psp_refund_id.clone();
                self.status = Status::Approved { psp_refund_id: psp };
                Ok(())
            }
            other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Move from `Approved` (or directly from `Submitted` for rails
    /// that don't have a separate approve step) to `Settled`.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] from any terminal state or
    /// `Requested`.
    pub fn settle(&mut self, settled_at_unix_secs: u64) -> Result<()> {
        match &self.status {
            Status::Submitted { psp_refund_id } | Status::Approved { psp_refund_id } => {
                let psp = psp_refund_id.clone();
                self.status = Status::Settled {
                    psp_refund_id: psp,
                    settled_at_unix_secs,
                };
                Ok(())
            }
            other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }

    /// Move from `Requested` / `Submitted` / `Approved` to
    /// `Declined`.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] from any terminal state.
    pub fn decline(&mut self, code: impl Into<String>, message: impl Into<String>) -> Result<()> {
        if self.status.is_terminal() {
            return Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: self.status.code().to_owned(),
            });
        }
        self.status = Status::Declined {
            code: code.into(),
            message: message.into(),
        };
        Ok(())
    }

    /// Move from `Approved` to `Failed` (post-approval failure —
    /// rare, usually a settlement reversal).
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] unless in `Approved`.
    pub fn fail_after_approval(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<()> {
        match &self.status {
            Status::Approved { .. } => {
                self.status = Status::Failed {
                    code: code.into(),
                    message: message.into(),
                };
                Ok(())
            }
            other => Err(Error::InvalidTransition {
                id: self.id.to_string(),
                state: other.code().to_owned(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};

    fn fresh() -> Refund {
        Refund::new(
            TransactionId::new(),
            Money::from_minor(500, Currency::USD),
            RefundReason::CustomerRequest,
            1_000,
        )
        .unwrap()
    }

    #[test]
    fn negative_amount_rejected_at_construction() {
        let err = Refund::new(
            TransactionId::new(),
            Money::from_minor(-1, Currency::USD),
            RefundReason::CustomerRequest,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
    }

    #[test]
    fn zero_amount_is_allowed_void_pattern() {
        let r = Refund::new(
            TransactionId::new(),
            Money::from_minor(0, Currency::USD),
            RefundReason::MerchantInitiated,
            0,
        )
        .unwrap();
        assert_eq!(r.amount.minor_units, 0);
    }

    #[test]
    fn happy_path_state_machine() {
        let mut r = fresh();
        assert_eq!(r.status.code(), "requested");
        r.submit("psp-1").unwrap();
        assert_eq!(r.status.code(), "submitted");
        r.approve().unwrap();
        assert_eq!(r.status.code(), "approved");
        r.settle(2_000).unwrap();
        assert_eq!(r.status.code(), "settled");
        assert!(r.status.is_terminal());
    }

    #[test]
    fn can_settle_directly_from_submitted() {
        // Some rails skip the approve step; the state machine
        // tolerates that.
        let mut r = fresh();
        r.submit("psp-1").unwrap();
        r.settle(2_000).unwrap();
        assert_eq!(r.status.code(), "settled");
    }

    #[test]
    fn decline_from_any_non_terminal_state() {
        for stage in 0..3 {
            let mut r = fresh();
            if stage >= 1 {
                r.submit("psp-1").unwrap();
            }
            if stage >= 2 {
                r.approve().unwrap();
            }
            r.decline("insufficient_time", "rail closed window")
                .unwrap();
            assert_eq!(r.status.code(), "declined");
        }
    }

    #[test]
    fn terminal_states_reject_further_transitions() {
        let mut r = fresh();
        r.submit("psp-1").unwrap();
        r.settle(2_000).unwrap();
        assert!(r.submit("psp-2").is_err());
        assert!(r.approve().is_err());
        assert!(r.settle(3_000).is_err());
        assert!(r.decline("x", "y").is_err());
    }

    #[test]
    fn approve_only_from_submitted() {
        let mut r = fresh();
        assert!(r.approve().is_err()); // Requested
    }

    #[test]
    fn fail_after_approval_only_from_approved() {
        let mut r = fresh();
        r.submit("p").unwrap();
        assert!(r.fail_after_approval("x", "y").is_err());
        r.approve().unwrap();
        r.fail_after_approval("settlement_reversed", "issuer pulled it back")
            .unwrap();
        assert_eq!(r.status.code(), "failed");
    }
}
