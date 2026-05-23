//! Settlement batches.
//!
//! A [`Batch`] groups posted ledger transactions that will move
//! together over a single payout rail. Lifecycle:
//!
//! ```text
//!   Open  ──close()──►  Closed  ──pay(reference)──►  Paying
//!                                                       │
//!                                          ┌────settled()
//!                                          │
//!                                          ▼
//!                                         Paid
//!                                          │
//!                                          └─fail(reason)──►  Failed
//! ```
//!
//! - `Open` accepts new posted-tx entries.
//! - `Closed` is finalized — entry list and holdback frozen.
//! - `Paying` is the in-flight state (NACHA submitted to ODFI,
//!   pacs.008 sent to the rail, etc.). External rail reference
//!   is recorded.
//! - `Paid` is terminal-success.
//! - `Failed` is terminal-failure (rail rejected the file).
//!
//! Reaching `Paid` or `Failed` is the operator's responsibility —
//! we don't poll the rail.

use op_core::{Currency, Money};
use op_ledger::TransactionId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::holdback::Holdback;
use crate::payout::PayoutRail;

/// Opaque batch id (`UUIDv7`, time-sortable).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BatchId(pub Uuid);

impl BatchId {
    /// Generate a fresh time-sortable id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
    /// Wrap an existing UUID.
    #[must_use]
    pub const fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }
    /// The underlying UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for BatchId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for BatchId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Batch lifecycle state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    /// Accepting new posted-tx entries.
    Open,
    /// Frozen — entry list and holdback set in stone.
    Closed {
        /// When the close happened (unix epoch seconds).
        closed_at_unix_secs: u64,
    },
    /// Payout submitted to the rail, awaiting confirmation.
    Paying {
        /// Operator's external rail reference (NACHA trace number,
        /// pacs.008 message id, wire reference, etc.).
        rail_reference: String,
        /// When the submission happened.
        submitted_at_unix_secs: u64,
    },
    /// Funds confirmed delivered to the beneficiary.
    Paid {
        /// Operator's external rail reference.
        rail_reference: String,
        /// When the settlement was confirmed.
        settled_at_unix_secs: u64,
    },
    /// Rail rejected the batch. Terminal — operators must open a
    /// new batch with the rolled-over transactions.
    Failed {
        /// Operator-supplied failure code (`R03`, `RJCT`, ...).
        code: String,
        /// Human-readable message.
        message: String,
        /// When the failure was recorded.
        failed_at_unix_secs: u64,
    },
}

impl Status {
    /// Short string code (`"open"` / `"closed"` / ...).
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Closed { .. } => "closed",
            Self::Paying { .. } => "paying",
            Self::Paid { .. } => "paid",
            Self::Failed { .. } => "failed",
        }
    }

    /// True if no further transitions are allowed.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Paid { .. } | Self::Failed { .. })
    }
}

/// One posted ledger transaction included in a batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchEntry {
    /// The ledger transaction id this entry settles.
    pub tx_id: TransactionId,
    /// The amount in this batch's currency.
    pub amount: Money,
    /// Caller-supplied free-form reference (order id, intent id).
    pub reference: Option<String>,
}

/// A group of posted transactions destined for one payout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Batch {
    /// Stable id.
    pub id: BatchId,
    /// All entries share this currency.
    pub currency: Currency,
    /// The rail that will move this batch's funds.
    pub rail: PayoutRail,
    /// Caller-supplied idempotency key. Two batches with the same
    /// `external_id` are treated as one logical batch by the store.
    pub external_id: Option<String>,
    /// Lifecycle state.
    pub status: Status,
    /// Included transactions, insertion order preserved.
    pub entries: Vec<BatchEntry>,
    /// Frozen at `close()` time.
    pub holdback: Option<Holdback>,
    /// When the batch opened.
    pub opened_at_unix_secs: u64,
    /// Free-form metadata (operator-side annotations).
    pub metadata: Vec<(String, String)>,
}

impl Batch {
    /// Open a fresh batch for `currency`/`rail`.
    #[must_use]
    pub fn open(currency: Currency, rail: PayoutRail, opened_at_unix_secs: u64) -> Self {
        Self {
            id: BatchId::new(),
            currency,
            rail,
            external_id: None,
            status: Status::Open,
            entries: Vec::new(),
            holdback: None,
            opened_at_unix_secs,
            metadata: Vec::new(),
        }
    }

    /// Builder: attach an external id for idempotency.
    #[must_use]
    pub fn with_external_id<S: Into<String>>(mut self, ext: S) -> Self {
        self.external_id = Some(ext.into());
        self
    }

    /// Append a posted-tx entry. Only legal in `Open` status.
    ///
    /// # Errors
    /// - [`Error::InvalidTransition`] if not `Open`.
    /// - [`Error::CurrencyMismatch`] if `amount.currency` differs.
    pub fn add_entry(
        &mut self,
        tx_id: TransactionId,
        amount: Money,
        reference: Option<String>,
    ) -> Result<()> {
        if !matches!(self.status, Status::Open) {
            return Err(Error::InvalidTransition {
                from: self.status.code().to_owned(),
                to: "open(add_entry)".to_owned(),
            });
        }
        if amount.currency != self.currency {
            return Err(Error::CurrencyMismatch {
                batch: self.currency.code().to_owned(),
                tx: amount.currency.code().to_owned(),
            });
        }
        self.entries.push(BatchEntry {
            tx_id,
            amount,
            reference,
        });
        Ok(())
    }

    /// Sum of every entry's amount.
    ///
    /// # Errors
    /// Propagates [`op_core::Error::Overflow`] on absurd sums.
    pub fn gross(&self) -> Result<Money> {
        let mut total = Money::from_minor(0, self.currency);
        for e in &self.entries {
            total = total.checked_add(e.amount)?;
        }
        Ok(total)
    }

    /// Transition `Open → Closed`, applying the holdback policy
    /// against the gross.
    ///
    /// # Errors
    /// - [`Error::InvalidTransition`] if not `Open`.
    /// - Bubbles up holdback arithmetic errors.
    pub fn close(&mut self, holdback: Holdback, closed_at_unix_secs: u64) -> Result<()> {
        if !matches!(self.status, Status::Open) {
            return Err(Error::InvalidTransition {
                from: self.status.code().to_owned(),
                to: "closed".to_owned(),
            });
        }
        self.holdback = Some(holdback);
        self.status = Status::Closed {
            closed_at_unix_secs,
        };
        Ok(())
    }

    /// Transition `Closed → Paying`, recording the rail's external
    /// reference.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if not `Closed`.
    pub fn pay<S: Into<String>>(
        &mut self,
        rail_reference: S,
        submitted_at_unix_secs: u64,
    ) -> Result<()> {
        if !matches!(self.status, Status::Closed { .. }) {
            return Err(Error::InvalidTransition {
                from: self.status.code().to_owned(),
                to: "paying".to_owned(),
            });
        }
        self.status = Status::Paying {
            rail_reference: rail_reference.into(),
            submitted_at_unix_secs,
        };
        Ok(())
    }

    /// Transition `Paying → Paid`.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if not `Paying`.
    pub fn settled(&mut self, settled_at_unix_secs: u64) -> Result<()> {
        let Status::Paying { rail_reference, .. } = &self.status else {
            return Err(Error::InvalidTransition {
                from: self.status.code().to_owned(),
                to: "paid".to_owned(),
            });
        };
        let rail_reference = rail_reference.clone();
        self.status = Status::Paid {
            rail_reference,
            settled_at_unix_secs,
        };
        Ok(())
    }

    /// Transition `Paying → Failed`.
    ///
    /// # Errors
    /// [`Error::InvalidTransition`] if not `Paying`.
    pub fn fail<C: Into<String>, M: Into<String>>(
        &mut self,
        code: C,
        message: M,
        failed_at_unix_secs: u64,
    ) -> Result<()> {
        if !matches!(self.status, Status::Paying { .. }) {
            return Err(Error::InvalidTransition {
                from: self.status.code().to_owned(),
                to: "failed".to_owned(),
            });
        }
        self.status = Status::Failed {
            code: code.into(),
            message: message.into(),
            failed_at_unix_secs,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holdback::HoldbackPolicy;
    use op_core::Currency;

    fn tx() -> TransactionId {
        TransactionId::new()
    }

    #[test]
    fn opens_in_open_status() {
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        assert_eq!(b.status.code(), "open");
        assert!(b.entries.is_empty());
    }

    #[test]
    fn rejects_currency_mismatch() {
        let mut b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        let err = b
            .add_entry(tx(), Money::from_minor(100, Currency::EUR), None)
            .unwrap_err();
        assert!(matches!(err, Error::CurrencyMismatch { .. }));
    }

    #[test]
    fn happy_path_through_lifecycle() {
        let mut b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        b.add_entry(
            tx(),
            Money::from_minor(7_500, Currency::USD),
            Some("o-1".into()),
        )
        .unwrap();
        b.add_entry(
            tx(),
            Money::from_minor(2_500, Currency::USD),
            Some("o-2".into()),
        )
        .unwrap();
        let gross = b.gross().unwrap();
        assert_eq!(gross, Money::from_minor(10_000, Currency::USD));
        let policy = HoldbackPolicy::flat(50);
        let hb = policy.compute(gross, 0).unwrap();
        b.close(hb, 2_000).unwrap();
        assert_eq!(b.status.code(), "closed");
        b.pay("trace-1", 3_000).unwrap();
        assert_eq!(b.status.code(), "paying");
        b.settled(4_000).unwrap();
        assert_eq!(b.status.code(), "paid");
        assert!(b.status.is_terminal());
    }

    #[test]
    fn cannot_add_after_close() {
        let mut b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        b.close(
            HoldbackPolicy::none()
                .compute(Money::from_minor(0, Currency::USD), 0)
                .unwrap(),
            2_000,
        )
        .unwrap();
        let err = b
            .add_entry(tx(), Money::from_minor(100, Currency::USD), None)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidTransition { .. }));
    }

    #[test]
    fn fail_from_paying() {
        let mut b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        b.add_entry(tx(), Money::from_minor(100, Currency::USD), None)
            .unwrap();
        b.close(
            HoldbackPolicy::none()
                .compute(b.gross().unwrap(), 0)
                .unwrap(),
            2_000,
        )
        .unwrap();
        b.pay("trace-1", 3_000).unwrap();
        b.fail("R01", "Insufficient funds", 4_000).unwrap();
        assert_eq!(b.status.code(), "failed");
    }

    #[test]
    fn cannot_fail_before_paying() {
        let mut b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        let err = b.fail("R01", "bad", 4_000).unwrap_err();
        assert!(matches!(err, Error::InvalidTransition { .. }));
    }
}
